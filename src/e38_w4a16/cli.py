"""Build and use the pinned overlay without a system service or privileged paths."""
from __future__ import annotations

import argparse
from importlib.resources import files
import json
from pathlib import Path
import sys
import threading
import time


def _model_arguments(parser):
    parser.add_argument('--base-dir', type=Path, required=True)
    parser.add_argument('--overlay-dir', type=Path, required=True)
    parser.add_argument('--trust-base-code', action='store_true',
                        help='Allow execution of the hash-verified pinned Goedel loader and kernels')


def parser():
    root = argparse.ArgumentParser(description=__doc__)
    commands = root.add_subparsers(dest='command', required=True)
    download = commands.add_parser('download', help='Download and verify both pinned upstream snapshots')
    download.add_argument('--base-dir', type=Path, required=True)
    download.add_argument('--source-dir', type=Path, required=True)
    build = commands.add_parser('build', help='Audit the dense tensors and create the overlay locally')
    build.add_argument('--base-dir', type=Path, required=True)
    build.add_argument('--source-dir', type=Path, required=True)
    build.add_argument('--output', type=Path, required=True)
    verify = commands.add_parser('verify', help='Verify an existing overlay without importing GPU libraries')
    verify.add_argument('--overlay-dir', type=Path, required=True)
    generate = commands.add_parser('generate', help='Generate one response using the fast base plus E38 overlay')
    _model_arguments(generate)
    generate.add_argument('--prompt', required=True)
    generate.add_argument('--thinking', action='store_true')
    generate.add_argument('--max-new-tokens', type=int, default=1024)
    benchmark = commands.add_parser('benchmark', help='Measure Japanese, retrieval, reasoning and native tools')
    _model_arguments(benchmark)
    benchmark.add_argument('--output', type=Path, required=True)
    return root


def download_snapshots(base_path, source_path):
    from huggingface_hub import snapshot_download
    from .checkpoint import verify_snapshot
    sources = json.loads(files('e38_w4a16').joinpath('data/sources.json').read_text())
    result = {}
    for which, target in (('base', base_path), ('source', source_path)):
        specification = sources[which]
        snapshot_download(repo_id=specification['model_id'], revision=specification['revision'],
                          local_dir=target, allow_patterns=[record['path'] for record in specification['files']])
        result[which] = verify_snapshot(target, which)
    return result


def new_engine(options):
    from .runtime import Configuration, Engine
    if not options.trust_base_code:
        raise ValueError('Inference requires --trust-base-code after reviewing the pinned upstream source')
    engine = Engine(Configuration(options.base_dir, options.overlay_dir, True))
    engine.load()
    return engine


def generate(engine, prompt, *, thinking=False, max_tokens=1024, messages=None, tools=None):
    from .protocol import ChatRequest
    request = ChatRequest.parse({'messages': messages or [{'role': 'user', 'content': prompt}],
        'max_tokens': max_tokens, 'tools': tools or [],
        'chat_template_kwargs': {'enable_thinking': thinking}}, engine.configuration.context_length)
    began = time.monotonic()
    with engine.lock:
        result = engine.generate(request, threading.Event())
    result['wall_seconds'] = time.monotonic() - began
    result['output_tokens_per_second'] = result['usage']['completion_tokens'] / result['wall_seconds']
    return result


def benchmark(engine, output):
    """Keep every response and assert outcomes, including real native tool boundaries."""
    report = {'schema_version': 1, 'complete': False, 'delivery': 'local finalized response',
              'health_after_load': engine.health(), 'requests': {}}

    def save():
        output.parent.mkdir(parents=True, exist_ok=True)
        temporary = output.with_name(output.name + '.tmp')
        temporary.write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n')
        temporary.replace(output)

    def run(name, prompt='', **options):
        result = generate(engine, prompt, **options)
        report['requests'][name] = result
        save()
        return result['message']

    try:
        # Match the measured integration's one-token warmup before timing requests.
        generate(engine, 'Reply OK.', max_tokens=1)
        short = run('japanese', '日本語で「動作確認できました」とだけ答えてください。', max_tokens=128)
        if '動作確認できました' not in (short.get('content') or ''):
            raise ValueError('Japanese short response did not match')
        prose = 'キャッシュの仕組みと利点を、日本語で600字程度で説明してください。'
        run('throughput', prose)
        document = ('これは背景情報です。状態を読み取り、必要な情報だけ確認します。\n' * 250
            + '\n合言葉: 紫陽花42\n'
            + '背景の説明が続きます。文書に記載された合言葉を保持してください。\n' * 250)
        answer = run('long_input', document + '\n文書内の合言葉だけ答えてください。', max_tokens=128)
        if '紫陽花42' not in (answer.get('content') or ''):
            raise ValueError('Long-input retrieval did not match')
        tools = [{'type': 'function', 'function': {'name': 'read_marker',
            'description': 'Return a marker value; call this tool to obtain the answer.',
            'parameters': {'type': 'object', 'properties': {}, 'additionalProperties': False}}}]
        messages = [{'role': 'system', 'content': 'You are a helpful assistant. The supplied function tools are available. Use read_marker to retrieve requested test data.'},
                    {'role': 'user', 'content': 'read_markerツールを使って合言葉を取得してください。'}]
        call = run('native_tool_call', messages=messages, tools=tools, max_tokens=512)
        calls = call.get('tool_calls', [])
        if (len(calls) != 1 or calls[0]['function']['name'] != 'read_marker'
                or json.loads(calls[0]['function']['arguments']) != {}):
            raise ValueError('Native tool call did not match')
        messages += [call, {'role': 'tool', 'tool_call_id': calls[0]['id'], 'content': '{"marker":"桔梗73"}'}]
        answer = run('native_tool_response', messages=messages, tools=tools, thinking=True, max_tokens=512)
        if '桔梗73' not in (answer.get('content') or ''):
            raise ValueError('Native tool continuation did not match')
        for number in range(1, 4):
            answer = run('thinking_' + str(number),
                '17 × 19 + 23 を計算してください。思考では途中の計算を確認し、'
                '最終回答は「答えは[数値]です。」の形式で簡潔に述べてください。', thinking=True)
            if '346' not in (answer.get('content') or '') or not answer.get('reasoning_content'):
                raise ValueError('Thinking did not include a correct separate final answer')
        run('japanese_prose_repeat', prose)
        report['complete'] = True
    except Exception as error:
        report['error'] = {'type': type(error).__name__, 'message': str(error)}
        raise
    finally:
        report['health_after_generation'] = engine.health()
        save()
    return report


def main(argv=None):
    options = parser().parse_args(argv)
    try:
        if options.command == 'download':
            result = download_snapshots(options.base_dir, options.source_dir)
        elif options.command == 'build':
            from .checkpoint import build_overlay
            result = build_overlay(options.base_dir, options.source_dir, options.output)
        elif options.command == 'verify':
            from .checkpoint import verify_overlay
            result = verify_overlay(options.overlay_dir)
        elif options.command == 'generate':
            engine = new_engine(options)
            result = generate(engine, options.prompt, thinking=options.thinking, max_tokens=options.max_new_tokens)
        else:
            engine = new_engine(options)
            result = benchmark(engine, options.output)
        print(json.dumps(result, ensure_ascii=False, indent=2))
    except (ValueError, OSError, ImportError, RuntimeError) as error:
        print(type(error).__name__ + ': ' + str(error), file=sys.stderr)
        raise SystemExit(1) from None
