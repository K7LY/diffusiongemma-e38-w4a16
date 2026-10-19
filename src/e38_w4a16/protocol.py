from __future__ import annotations

from dataclasses import dataclass
import json
import math
import re
from uuid import uuid4
from typing import Literal


class RequestError(ValueError):
    def __init__(self, message: str, *, code: Literal['invalid_request', 'context_length_exceeded'] = 'invalid_request'):
        super().__init__(message)
        self.code = code


def strict_json(value: str):
    def invalid(_):
        raise RequestError('JSON must not contain non-finite numbers')
    def finite_float(text):
        result = float(text)
        if not math.isfinite(result):
            invalid(text)
        return result
    return json.loads(value, parse_constant=invalid, parse_float=finite_float)


def text_content(value) -> str:
    if value is None:
        return ''
    if isinstance(value, str):
        return value
    if not isinstance(value, list):
        raise RequestError('message content must be text')
    parts = []
    for item in value:
        if not isinstance(item, dict) or item.get('type') != 'text' or not isinstance(item.get('text'), str):
            raise RequestError('DiffusionGemma currently accepts text only; images, audio and video are unsupported')
        parts.append(item['text'])
    return '\n'.join(parts)


def function_name(value):
    if not isinstance(value, str) or not re.fullmatch(r'[A-Za-z_][A-Za-z0-9_.-]{0,127}', value):
        raise RequestError('invalid tool function name')
    return value


@dataclass(frozen=True)
class ChatRequest:
    messages: list[dict]
    tools: list[dict]
    max_tokens: int
    stream: bool
    thinking: bool
    stop: tuple[str, ...]
    temperature: float | None
    include_usage: bool

    @classmethod
    def parse(cls, body: object, context_length: int):
        if not isinstance(body, dict):
            raise RequestError('request must be a JSON object')
        if body.get('model', 'diffusion-gemma') not in {'diffusion-gemma', 'diffusiongemma-w4a16'}:
            raise RequestError('unknown model')
        if body.get('n', 1) != 1:
            raise RequestError('only n=1 is supported')
        response_format = body.get('response_format') or {}
        if not isinstance(response_format, dict) or response_format.get('type', 'text') != 'text':
            raise RequestError('structured response formats are unsupported')
        for unsupported in ('logprobs', 'logit_bias', 'top_logprobs', 'seed', 'frequency_penalty', 'presence_penalty'):
            if body.get(unsupported) not in (None, False, 0, {}):
                raise RequestError(unsupported + ' is unsupported by this diffusion adapter')
        if body.get('top_p', 1) != 1:
            raise RequestError('top_p is unsupported by the pinned diffusion sampler')
        messages = body.get('messages')
        if not isinstance(messages, list) or not messages:
            raise RequestError('messages must be a nonempty array')
        normalized, known_calls = [], {}
        for message in messages:
            if not isinstance(message, dict) or message.get('role') not in {'system', 'developer', 'user', 'assistant', 'tool'}:
                raise RequestError('invalid message role')
            role = message['role']
            item = {'role': role, 'content': text_content(message.get('content'))}
            for key in ('reasoning', 'reasoning_content'):
                if message.get(key) is not None:
                    if not isinstance(message[key], str):
                        raise RequestError(key + ' must be text')
                    item[key] = message[key]
            calls = message.get('tool_calls')
            if calls:
                if role != 'assistant' or not isinstance(calls, list):
                    raise RequestError('tool_calls must belong to an assistant')
                item['tool_calls'] = []
                for call in calls:
                    if not isinstance(call, dict) or call.get('type', 'function') != 'function' or not isinstance(call.get('function'), dict):
                        raise RequestError('invalid tool call')
                    function = call['function']
                    name = function_name(function.get('name'))
                    arguments = function.get('arguments', {})
                    if isinstance(arguments, str):
                        try:
                            arguments = strict_json(arguments)
                        except (ValueError, TypeError):
                            raise RequestError('tool call arguments must be a JSON object') from None
                    if not isinstance(arguments, dict):
                        raise RequestError('tool call arguments must be a JSON object')
                    identifier = call.get('id')
                    if not isinstance(identifier, str) or not identifier or identifier in known_calls:
                        raise RequestError('tool calls require unique IDs')
                    known_calls[identifier] = name
                    item['tool_calls'].append({'id': identifier, 'type': 'function', 'function': {'name': name, 'arguments': arguments}})
            if role == 'tool':
                identifier = message.get('tool_call_id')
                if identifier not in known_calls:
                    raise RequestError('tool response does not match an earlier tool call')
                item.update(tool_call_id=identifier, name=known_calls[identifier])
            normalized.append(item)
        tools = body.get('tools') or []
        if not isinstance(tools, list):
            raise RequestError('tools must be an array')
        names = set()
        for tool in tools:
            if not isinstance(tool, dict) or tool.get('type') != 'function' or not isinstance(tool.get('function'), dict):
                raise RequestError('only function tools are supported')
            function = tool['function']
            name = function_name(function.get('name'))
            if name in names or not isinstance(function.get('parameters', {}), dict):
                raise RequestError('invalid or duplicate tool definition')
            names.add(name)
        if body.get('tool_choice', 'auto') not in ('auto', 'none'):
            raise RequestError('only auto and none tool choice are supported')
        if body.get('tool_choice') == 'none':
            tools = []
        budget = body.get('max_completion_tokens', body.get('max_tokens', min(8192, context_length // 4)))
        if isinstance(budget, bool) or not isinstance(budget, int) or not 1 <= budget <= context_length:
            raise RequestError('max_tokens must be a positive integer within the configured context')
        stream = body.get('stream', False)
        if not isinstance(stream, bool):
            raise RequestError('stream must be boolean')
        template = body.get('chat_template_kwargs') or {}
        if not isinstance(template, dict) or set(template) - {'enable_thinking'}:
            raise RequestError('unsupported chat template options')
        thinking = template.get('enable_thinking', False)
        if not isinstance(thinking, bool):
            raise RequestError('enable_thinking must be boolean')
        stop = body.get('stop') or []
        if isinstance(stop, str):
            stop = [stop]
        if not isinstance(stop, list) or not all(isinstance(s, str) and s for s in stop):
            raise RequestError('stop must be text or an array of nonempty strings')
        temperature = body.get('temperature')
        if temperature is not None and (isinstance(temperature, bool) or not isinstance(temperature, (int, float)) or not math.isfinite(temperature) or not 0 <= temperature <= 2):
            raise RequestError('temperature must be between zero and two')
        options = body.get('stream_options') or {}
        if not isinstance(options, dict):
            raise RequestError('stream_options must be an object')
        return cls(normalized, tools, budget, stream, thinking, tuple(stop), temperature, bool(options.get('include_usage')))


class GemmaArguments:
    """Decode the checkpoint's delimited strings without rewriting their contents."""
    delimiter = '<|"|>'

    def __init__(self, text):
        self.text, self.position = text, 0

    def whitespace(self):
        while self.position < len(self.text) and self.text[self.position].isspace():
            self.position += 1

    def value(self):
        self.whitespace()
        start = self.position
        if self.text.startswith(self.delimiter, start):
            end = self.text.find(self.delimiter, start + len(self.delimiter))
            if end < 0:
                raise ValueError('unterminated Gemma string')
            self.position = end + len(self.delimiter)
            return self.text[start + len(self.delimiter):end]
        if start >= len(self.text):
            raise ValueError('missing argument value')
        char = self.text[start]
        if char == '{':
            result = {}
            self.position += 1
            self.whitespace()
            if self.text[self.position:self.position + 1] == '}':
                self.position += 1
                return result
            while True:
                self.whitespace()
                if self.text.startswith(self.delimiter, self.position) or self.text[self.position:self.position + 1] == '"':
                    key = self.value()
                else:
                    match = re.match(r'[A-Za-z_][A-Za-z0-9_.-]*', self.text[self.position:])
                    if not match:
                        raise ValueError('invalid argument key')
                    key = match.group()
                    self.position += len(key)
                self.whitespace()
                if self.text[self.position:self.position + 1] != ':' or key in result:
                    raise ValueError('missing colon or duplicate key')
                self.position += 1
                result[key] = self.value()
                self.whitespace()
                separator = self.text[self.position:self.position + 1]
                self.position += 1
                if separator == '}':
                    return result
                if separator != ',':
                    raise ValueError('invalid object separator')
        if char == '[':
            result = []
            self.position += 1
            self.whitespace()
            if self.text[self.position:self.position + 1] == ']':
                self.position += 1
                return result
            while True:
                result.append(self.value())
                self.whitespace()
                separator = self.text[self.position:self.position + 1]
                self.position += 1
                if separator == ']':
                    return result
                if separator != ',':
                    raise ValueError('invalid array separator')
        value, count = json.JSONDecoder(parse_constant=lambda _: (_ for _ in ()).throw(ValueError('non-finite argument'))).raw_decode(self.text[start:])
        self.position = start + count
        return value

    def parse(self):
        value = self.value()
        self.whitespace()
        if self.position != len(self.text) or not isinstance(value, dict):
            raise ValueError('tool arguments must be one object')
        return value


def parse_output(raw: str, tool_names: set[str], *, partial=False) -> dict:
    reasoning, content, calls = [], [], []
    # The model may open a tool-response boundary rather than close a normal turn.
    raw = raw.split('<|tool_response>', 1)[0]
    position = 0
    marker = re.compile(r'<\|channel>thought\n|<\|tool_call>|<turn\|>|<eos>|<pad>')
    while position < len(raw):
        match = marker.search(raw, position)
        if not match:
            content.append(raw[position:])
            break
        content.append(raw[position:match.start()])
        token = match.group()
        if token in {'<turn|>', '<eos>', '<pad>'}:
            break
        if token == '<|channel>thought\n':
            end = raw.find('<channel|>', match.end())
            if end < 0:
                reasoning.append(raw[match.end():])
                break
            reasoning.append(raw[match.end():end])
            position = end + len('<channel|>')
        else:
            end = raw.find('<tool_call|>', match.end())
            if end < 0:
                if partial:
                    break
                raise ValueError('model returned an incomplete tool call')
            call = re.fullmatch(r'\s*call:([A-Za-z_][A-Za-z0-9_.-]*)\s*(\{.*\})\s*', raw[match.end():end], re.S)
            if not call or call[1] not in tool_names:
                if partial:
                    break
                raise ValueError('model returned an invalid or unrequested tool')
            try:
                arguments = GemmaArguments(call[2]).parse()
            except ValueError:
                if partial:
                    break
                raise
            calls.append({'id': 'call_' + uuid4().hex, 'type': 'function', 'function': {'name': call[1], 'arguments': json.dumps(arguments, ensure_ascii=False, allow_nan=False)}})
            position = end + len('<tool_call|>')
    result = {'role': 'assistant', 'content': ''.join(content).strip() or None}
    if reasoning:
        result['reasoning_content'] = ''.join(reasoning).strip()
    if calls:
        result['tool_calls'] = calls
    return result
