"""Generation and native-tool parsing shared with the measured integration."""
from __future__ import annotations

from dataclasses import dataclass
import gc
import importlib
from pathlib import Path
import threading
import time

from .protocol import ChatRequest, RequestError, parse_output


@dataclass(frozen=True)
class Configuration:
    base_path: Path
    overlay_path: Path
    trust_base_code: bool = False
    context_length: int = 262144


@dataclass(frozen=True)
class CommittedSnapshot:
    text: str
    token_count: int
    terminated: bool


class CommittedTokens:
    """Accept finalized SDK canvases; drafts never become application output."""
    def __init__(self, tokenizer, budget, prefix='', *, pad_token_id, eos_token_ids):
        self.tokenizer, self.budget, self.prefix = tokenizer, budget, prefix
        self.pad_token_id = pad_token_id
        self.eos_token_ids = frozenset([eos_token_ids] if isinstance(eos_token_ids, int)
                                       else eos_token_ids or ())
        self.tokens, self.prompt_seen = [], False
        self.first_commit_seconds = None
        self.started = time.monotonic()
        self.lock = threading.Lock()

    def put(self, tensor):
        values = tensor.tolist()
        if values and isinstance(values[0], list):
            values = values[0]
        if not self.prompt_seen:
            self.prompt_seen = True
            return
        with self.lock:
            if self.first_commit_seconds is None:
                self.first_commit_seconds = time.monotonic() - self.started
            self.tokens.extend(values[:max(0, self.budget - len(self.tokens))])

    def end(self):
        pass

    def snapshot(self) -> CommittedSnapshot:
        with self.lock:
            tokens = list(self.tokens)
        terminated = False
        # The native sampler pads completed canvases using generation_config;
        # tokenizer padding may name a different token in ModelOpt exports.
        for position, token in enumerate(tokens):
            if token in self.eos_token_ids:
                terminated = True
                tokens = tokens[:position + 1]
                break
            if token == self.pad_token_id:
                tokens = tokens[:position]
                break
        return CommittedSnapshot(self.prefix + self.tokenizer.decode(tokens, skip_special_tokens=False),
                                 len(tokens), terminated)


class Engine:
    def __init__(self, configuration):
        self.configuration = configuration
        self.lock = threading.Lock()
        self.ready, self.failure = False, None
        self.model = self.tokenizer = self.torch = None
        self.active_cancel = None
        self.execution_metadata = {}

    def load(self):
        from .loader import load_model
        self.model, self.tokenizer, self.execution_metadata = load_model(
            self.configuration.base_path, self.configuration.overlay_path,
            trust_base_code=self.configuration.trust_base_code)
        self.torch = importlib.import_module('torch')
        self.ready = True

    def health(self):
        result = {'status': 'ok' if self.ready else ('failed' if self.failure else 'loading'), 'model': 'diffusion-gemma', 'context_length': self.configuration.context_length, 'streaming': 'buffered_finalized_canvases', 'busy': self.lock.locked()}
        if self.ready:
            cuda = self.torch.cuda
            result['memory'] = {'cuda_allocated_bytes': cuda.memory_allocated(),
                                'cuda_reserved_bytes': cuda.memory_reserved(),
                                'cuda_peak_allocated_bytes': cuda.max_memory_allocated()}
            if self.execution_metadata:
                result['execution'] = self.execution_metadata
        return result

    def generate(self, request, cancel, progress=None):
        from transformers import StoppingCriteria, StoppingCriteriaList
        torch = self.torch
        tokenized = self.tokenizer.apply_chat_template(request.messages, tools=request.tools or None, enable_thinking=request.thinking, add_generation_prompt=True, tokenize=True, return_tensors='pt', return_dict=True)
        prompt_tokens = tokenized['input_ids'].shape[1]
        canvas = self.model.config.canvas_length
        rounded = ((request.max_tokens + canvas - 1) // canvas) * canvas
        if prompt_tokens + rounded > self.configuration.context_length:
            raise RequestError('prompt and output canvas exceed configured context length', code='context_length_exceeded')
        # Tool continuations can open the thought channel in the prompt itself.
        # Its marker must be restored when parsing only newly generated tokens.
        suffix = self.tokenizer.decode(tokenized['input_ids'][0, -32:].tolist(), skip_special_tokens=False)
        prefix = '<|channel>thought\n' if suffix.endswith('<|channel>thought\n') else ''
        input_ids = tokenized['input_ids'].to('cuda')
        committed = CommittedTokens(self.tokenizer, request.max_tokens, prefix,
            pad_token_id=self.model.generation_config.pad_token_id,
            eos_token_ids=self.model.generation_config.eos_token_id)
        if progress is not None:
            progress['committed'] = committed
        self.active_cancel = cancel

        class Stop(StoppingCriteria):
            def __call__(_, input_ids, scores, **kwargs):
                stopped = cancel.is_set()
                if request.stop and not stopped:
                    text = self.tokenizer.decode(input_ids[0, prompt_tokens:].tolist(), skip_special_tokens=False)
                    stopped = any(value in text for value in request.stop)
                return torch.full((input_ids.shape[0],), stopped, dtype=torch.bool, device=input_ids.device)

        kwargs = {'max_new_tokens': request.max_tokens, 'streamer': committed, 'stopping_criteria': StoppingCriteriaList([Stop()])}
        if request.temperature is not None:
            # The native sampler requires a positive, strictly descending schedule.
            # Preserve its 2:1 range; zero is approximated with a near-greedy range.
            initial_temperature = max(float(request.temperature), 2e-5)
            kwargs.update(t_min=initial_temperature / 2, t_max=initial_temperature)
        started = time.monotonic()
        try:
            if cancel.is_set():
                raise InterruptedError('request cancelled')
            # The fused sampler reads logits._version, which inference_mode removes.
            with torch.no_grad():
                output = self.model.generate(input_ids=input_ids, **kwargs)
            torch.cuda.synchronize()
            snapshot = committed.snapshot()
            raw, completion_tokens = snapshot.text, snapshot.token_count
            cut = min((raw.find(stop) for stop in request.stop if stop in raw), default=len(raw))
            stopped = cut < len(raw)
            exhausted = completion_tokens >= request.max_tokens and not snapshot.terminated
            raw = raw[:cut]
            names = {tool['function']['name'] for tool in request.tools}
            terminal_reason = 'stop' if stopped or not exhausted else 'length'
            try:
                message = parse_output(raw, names)
            except ValueError:
                if not (stopped or exhausted):
                    raise
                message = parse_output(raw, names, partial=True)
                # A truncated batch cannot safely authorize even its completed calls.
                message.pop('tool_calls', None)
            reason = 'tool_calls' if message.get('tool_calls') else terminal_reason
            return {'message': message, 'finish_reason': reason, 'usage': {'prompt_tokens': prompt_tokens, 'completion_tokens': completion_tokens, 'total_tokens': prompt_tokens + completion_tokens}, 'timings': {'generation_seconds': time.monotonic() - started, 'first_finalized_canvas_seconds': committed.first_commit_seconds, 'delivery': 'buffered'}}
        finally:
            self.active_cancel = None
            # Generation output owns the KV cache, which must not survive a turn.
            if 'output' in locals():
                del output
            del input_ids, tokenized
            gc.collect()
            torch.cuda.empty_cache()

    def cancel_active(self):
        if self.active_cancel is not None:
            self.active_cancel.set()
