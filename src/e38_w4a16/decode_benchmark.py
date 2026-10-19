"""Single-stream decode timing after initial encoder prefill, without a streamer."""
from __future__ import annotations

import gc
from importlib.metadata import version
import json
from pathlib import Path
import platform
import statistics
import time

CASES = {
    'python_module': {'thinking': False, 'prompt':
        'Write a complete Python 3 module implementing an LRU cache with a doubly linked list, '
        'a TTL cache, unit tests, type annotations, docstrings, and an executable demonstration. '
        'Return code only. Include thorough tests and examples.'},
    'mathematical_proof': {'thinking': True, 'prompt':
        'Derive the formula for the sum of squares from 1 to n using telescoping cubes, '
        'verify it explicitly for n=1 through n=10, and then prove it by mathematical induction. '
        'Show all intermediate algebra and checks.'},
    'technical_essay': {'thinking': False, 'prompt':
        'Write a detailed technical explanation of CPU caches, cache coherence, locality, '
        'memory bandwidth, and how these affect matrix multiplication. Include practical '
        'examples and implementation advice. Write at least 1800 words in English.'},
}


def committed_tokens(values, pad_token_id, eos_token_ids):
    eos = {eos_token_ids} if isinstance(eos_token_ids, int) else set(eos_token_ids or ())
    for position, token in enumerate(values):
        if token in eos:
            return values[:position + 1], True
        if token == pad_token_id:
            return values[:position], False
    return values, False


def measure(model, tokenizer, case, *, max_tokens=1024, max_denoising_steps=48, seed=42):
    import torch
    if max_tokens <= 0 or max_tokens % model.config.canvas_length:
        raise ValueError('The output budget must be a positive multiple of the model canvas')
    torch.manual_seed(seed)
    encoded = tokenizer.apply_chat_template([{'role': 'user', 'content': case['prompt']}],
        enable_thinking=case['thinking'], add_generation_prompt=True, tokenize=True,
        return_tensors='pt', return_dict=True)
    ids = encoded['input_ids'].to('cuda')
    prompt_tokens = ids.shape[1]
    boundary = {}

    def after_initial_encoder(_module, _args, _output):
        # Later encoder calls commit generated canvases and belong to decode.
        torch.cuda.synchronize()
        boundary['decode_started'] = time.perf_counter()
        handle.remove()

    handle = model.model.encoder.register_forward_hook(after_initial_encoder)
    try:
        torch.cuda.synchronize()
        started = time.perf_counter()
        with torch.no_grad():
            output = model.generate(input_ids=ids, max_new_tokens=max_tokens,
                                    max_denoising_steps=max_denoising_steps)
        torch.cuda.synchronize()
        finished = time.perf_counter()
        if 'decode_started' not in boundary:
            raise RuntimeError('The initial encoder timing boundary was not observed')
        # Copying tokens to CPU and detokenizing are deliberately outside timing.
        values = output.sequences[0, prompt_tokens:].tolist()
        tokens, terminated = committed_tokens(values, model.generation_config.pad_token_id,
                                               model.generation_config.eos_token_id)
        decode_seconds = finished - boundary['decode_started']
        if not tokens or decode_seconds <= 0:
            raise RuntimeError('The generation produced no measurable committed output')
        result = {'prompt_tokens': prompt_tokens, 'completion_tokens': len(tokens),
            'output_budget': max_tokens, 'max_denoising_steps': max_denoising_steps,
            'seed': seed, 'thinking': case['thinking'], 'terminated': terminated,
            'prompt_setup_and_prefill_seconds': boundary['decode_started'] - started,
            'decode_seconds': decode_seconds, 'generation_seconds': finished - started,
            'decode_tokens_per_second': len(tokens) / decode_seconds,
            'generated_text': tokenizer.decode(tokens, skip_special_tokens=False)}
        value = getattr(output, 'tokens_per_forward', None)
        if value is not None:
            result['tokens_per_forward'] = float(value)
        return result
    finally:
        handle.remove()
        if 'output' in locals():
            del output
        del ids, encoded
        gc.collect()
        torch.cuda.empty_cache()


def run_sweep(model, tokenizer, output_path):
    """Bounded workload/config search; retain outputs and repeated winner samples."""
    import torch
    path = Path(output_path)
    report = {'schema_version': 1, 'complete': False, 'batch_size': 1,
        'boundary': 'After initial encoder prefill through final GPU completion; includes later canvas encoder commits',
        'excluded': ['tokenization', 'initial prompt prefill', 'streamer delivery', 'final CPU transfer and detokenization'],
        'warmup': 'One 256-token run per denoising-step setting; excluded from results',
        'quality_note': '16/8 step limits may reduce quality. Output inspection is separate from throughput.',
        'cases': CASES, 'samples': []}
    sampler = model.generation_config.sampler_config
    report['configuration'] = {'canvas_length': model.config.canvas_length,
        'native_max_denoising_steps': model.generation_config.max_denoising_steps,
        'entropy_bound': sampler.get('entropy_bound') if isinstance(sampler, dict) else sampler.entropy_bound,
        **{key: getattr(model.generation_config, key) for key in
           ('t_min', 't_max', 'stability_threshold', 'confidence_threshold')}}
    report['runtime'] = {'python': platform.python_version(), 'torch': str(torch.__version__),
        'transformers': version('transformers'), 'triton': version('triton'),
        'cuda': torch.version.cuda, 'gpu': torch.cuda.get_device_name(0)}

    def save():
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_name(path.name + '.tmp')
        temporary.write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n')
        temporary.replace(path)

    def run(case_name, steps, *, phase, seed=42):
        result = measure(model, tokenizer, CASES[case_name], max_denoising_steps=steps, seed=seed)
        result.update(case=case_name, phase=phase)
        report['samples'].append(result)
        save()
        return result

    try:
        for steps in (48, 16, 8):
            report['phase'] = 'warmup_' + str(steps)
            save()
            measure(model, tokenizer, CASES['python_module'], max_tokens=256, max_denoising_steps=steps)
            for name in CASES:
                report['phase'] = 'sweep_' + str(steps) + '_' + name
                save()
                run(name, steps, phase='sweep')
        candidates = list(report['samples'])
        standard = max((s for s in candidates if s['max_denoising_steps'] == 48),
                       key=lambda sample: sample['decode_tokens_per_second'])
        peak = max(candidates, key=lambda sample: sample['decode_tokens_per_second'])
        groups = {(s['case'], s['max_denoising_steps']) for s in (standard, peak)}
        for case_name, steps in sorted(groups):
            for seed in (43, 44):
                report['phase'] = 'repeat_' + str(steps) + '_' + case_name
                save()
                run(case_name, steps, phase='repeat', seed=seed)
        report['winner_groups'] = []
        for case_name, steps in sorted(groups):
            samples = [s for s in report['samples'] if s['case'] == case_name and s['max_denoising_steps'] == steps]
            rates = [s['decode_tokens_per_second'] for s in samples]
            report['winner_groups'].append({'case': case_name, 'max_denoising_steps': steps,
                'samples': len(samples), 'median_tokens_per_second': statistics.median(rates),
                'peak_tokens_per_second': max(rates),
                'aggregate_tokens_per_second': sum(s['completion_tokens'] for s in samples) / sum(s['decode_seconds'] for s in samples)})
        report['cuda_peak_allocated_bytes'] = torch.cuda.max_memory_allocated()
        report['complete'] = True
        report['phase'] = 'complete'
    except Exception as error:
        report['error'] = {'type': type(error).__name__, 'message': str(error)}
        raise
    finally:
        save()
    return report
