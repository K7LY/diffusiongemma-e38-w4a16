"""Build a byte-exact E38 dense overlay without importing model code or Torch."""
from __future__ import annotations

import hashlib
from importlib import resources
import json
import math
import os
from pathlib import Path, PurePosixPath
import re
import shutil
import stat
import struct
import tempfile

FORMAT = 'dg-w4a16-e38-overlay-v1'
OVERLAY_FILE = 'e38-overlay.safetensors'
MANIFEST_FILE = 'manifest.json'
OVERLAY_BYTES = 372705672
OVERLAY_SHA256 = '7a913d9bdb580fe06c42efc0833d3eab9ec7d2d255144e9d7b102458e124fec7'
TENSOR_SHAPES = {
    f'model.decoder.layers.{layer}.{projection}.weight':
        (2816, 2112 if projection == 'mlp.down_proj' else (8192 if layer == 11 else 4096))
    for layer in range(7, 17) for projection in ('self_attn.o_proj', 'mlp.down_proj')
}
_CHUNK = 8 * 1024 * 1024
_WIDTHS = {'BOOL': 1, 'U8': 1, 'I8': 1, 'F8_E4M3': 1, 'F8_E5M2': 1,
           'F8_E8M0': 1, 'I16': 2, 'U16': 2, 'F16': 2, 'BF16': 2,
           'I32': 4, 'U32': 4, 'F32': 4, 'I64': 8, 'U64': 8, 'F64': 8}


def _object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f'Duplicate JSON key: {key}')
        result[key] = value
    return result


def _json(raw):
    return json.loads(raw, object_pairs_hook=_object)


def _data(name):
    return _json(resources.files('e38_w4a16').joinpath('data', name).read_text())


def _spec(which):
    if which not in ('base', 'source'):
        raise ValueError('Snapshot must be base or source')
    return _data('sources.json')[which]


def _audit():
    value = _data('dense-audit.json')
    records = value['changed']
    if (value['shared_dense_tensors'] != 987 or value['base_only_tensors']
            or value['source_only_tensors'] or len(records) != len(TENSOR_SHAPES)
            or {item['name'] for item in records} != set(TENSOR_SHAPES)):
        raise ValueError('Packaged dense audit has an unexpected tensor set')
    for item in records:
        if (item['dtype'] != 'BF16' or item['shape'] != list(TENSOR_SHAPES[item['name']])
                or not _sha(item['sha256']) or not _sha(item['base_sha256'])):
            raise ValueError('Packaged dense audit has invalid tensor metadata')
    return value


def _sha(value):
    return isinstance(value, str) and re.fullmatch(r'[0-9a-f]{64}', value) is not None


def _root(path):
    path = Path(path).expanduser().absolute()
    if not stat.S_ISDIR(path.lstat().st_mode):
        raise ValueError(f'Expected a real directory, without a symlink: {path}')
    return path


def _relative(name):
    if not isinstance(name, str) or not name:
        raise ValueError('Invalid snapshot file path')
    relative = PurePosixPath(name)
    if (relative.is_absolute() or relative.as_posix() != name
            or '..' in relative.parts or '\\' in name or name == '.'):
        raise ValueError(f'Invalid snapshot file path: {name}')
    return relative


def _file(root, name):
    relative = _relative(name)
    current = root
    for part in relative.parts[:-1]:
        current /= part
        if not stat.S_ISDIR(current.lstat().st_mode):
            raise ValueError(f'Snapshot contains a symlink or non-directory: {current}')
    target = root / relative
    if not stat.S_ISREG(target.lstat().st_mode):
        raise ValueError(f'Expected a regular file, without a symlink: {target}')
    return target


def _hash(path, begin=0, length=None):
    digest = hashlib.sha256()
    flags = os.O_RDONLY | getattr(os, 'O_NOFOLLOW', 0)
    with os.fdopen(os.open(path, flags), 'rb') as source:
        source.seek(begin)
        while length is None or length:
            block = source.read(_CHUNK if length is None else min(_CHUNK, length))
            if not block:
                if length:
                    raise ValueError(f'Truncated tensor data: {path}')
                break
            digest.update(block)
            if length is not None:
                length -= len(block)
    return digest.hexdigest()


def verify_snapshot(path, which: str) -> dict:
    """Rehash every pinned file. No downloaded Python code is executed.

    Download with ``local_dir`` rather than a symlink-based HF snapshot. Extra
    metadata (including HF's ``.cache`` directory) does not establish provenance.
    The loader separately rejects unlisted executable code and weight files.
    """
    root, spec = _root(path), _spec(which)
    seen = set()
    for record in spec['files']:
        name = record['path']
        if (name in seen or type(record['bytes']) is not int or record['bytes'] < 0
                or not _sha(record['sha256'])):
            raise ValueError('Invalid pinned snapshot record')
        target = _file(root, name)
        if target.stat().st_size != record['bytes'] or _hash(target) != record['sha256']:
            raise ValueError(f'Pinned {which} file differs: {name}')
        seen.add(name)
    if not seen:
        raise ValueError('Pinned snapshot is empty')
    return spec


def _header(path):
    with path.open('rb') as source:
        prefix = source.read(8)
        if len(prefix) != 8:
            raise ValueError('Truncated safetensors prefix')
        size, = struct.unpack('<Q', prefix)
        if not 0 < size <= 16 * 1024 * 1024:
            raise ValueError('Unexpected safetensors header size')
        raw = source.read(size)
        if len(raw) != size:
            raise ValueError('Truncated safetensors header')
    body = _json(raw)
    if not isinstance(body, dict):
        raise ValueError('Safetensors header must be an object')
    return 8 + size, body


def _span(path, data_start, value):
    if not isinstance(value, dict) or set(value) != {'dtype', 'shape', 'data_offsets'}:
        raise ValueError('Unexpected safetensors tensor record')
    shape, offsets, dtype = value['shape'], value['data_offsets'], value['dtype']
    if (not isinstance(shape, list) or any(type(n) is not int or n < 0 for n in shape)
            or not isinstance(offsets, list) or len(offsets) != 2
            or any(type(n) is not int for n in offsets) or dtype not in _WIDTHS
            or offsets[0] < 0 or offsets[1] < offsets[0]
            or offsets[1] - offsets[0] != math.prod(shape) * _WIDTHS[dtype]
            or data_start + offsets[1] > path.stat().st_size):
        raise ValueError('Invalid safetensors tensor shape or byte offsets')
    return data_start + offsets[0], offsets[1] - offsets[0]


def _dense(root, spec):
    result = {}
    for file in spec['files']:
        name = file['path']
        if not (name.startswith('model') and name.endswith('.safetensors')):
            continue
        path = _file(root, name)
        start, header = _header(path)
        for key, value in header.items():
            if key == '__metadata__' or '.experts.' in key:
                continue
            if key in result:
                raise ValueError(f'Duplicate dense tensor: {key}')
            begin, size = _span(path, start, value)
            result[key] = (path, begin, size, value)
    return result


def _identity(spec):
    return {key: spec[key] for key in ('model_id', 'revision')}


def _tensor_records():
    return [{key: item[key] for key in ('name', 'dtype', 'shape', 'sha256')}
            for item in sorted(_audit()['changed'], key=lambda item: item['name'])]


def _receipt():
    return {'format': FORMAT, 'quantization': 'W4A16',
            'base': _identity(_spec('base')), 'source': _identity(_spec('source')),
            'files': [{'path': OVERLAY_FILE, 'bytes': OVERLAY_BYTES, 'sha256': OVERLAY_SHA256}],
            'tensors': _tensor_records()}


def verify_overlay(overlay_path) -> dict:
    """Verify the standalone data artifact against the packaged, pinned audit."""
    root = _root(overlay_path)
    if {item.name for item in root.iterdir()} != {OVERLAY_FILE, MANIFEST_FILE}:
        raise ValueError('Overlay must contain exactly its manifest and data file')
    manifest_path = _file(root, MANIFEST_FILE)
    if manifest_path.stat().st_size > 128 * 1024:
        raise ValueError('Overlay manifest is unexpectedly large')
    receipt = _json(manifest_path.read_text())
    if receipt != _receipt():
        raise ValueError('Overlay manifest differs from the pinned provenance and dense audit')
    target = _file(root, OVERLAY_FILE)
    if target.stat().st_size != OVERLAY_BYTES or _hash(target) != OVERLAY_SHA256:
        raise ValueError('Overlay file checksum or size differs')
    start, header = _header(target)
    if set(header) != set(TENSOR_SHAPES):
        raise ValueError('Overlay contains unexpected tensors')
    offset = 0
    for item in receipt['tensors']:
        value = header[item['name']]
        begin, size = _span(target, start, value)
        if (value['dtype'] != item['dtype'] or value['shape'] != item['shape']
                or begin != start + offset or _hash(target, begin, size) != item['sha256']):
            raise ValueError(f'Overlay tensor differs: {item["name"]}')
        offset += size
    if start + offset != target.stat().st_size:
        raise ValueError('Overlay contains trailing bytes')
    return receipt


def _write_overlay(path, tensors):
    header, offset = {}, 0
    for item in _tensor_records():
        name = item['name']
        size = tensors[name][2]
        header[name] = {'dtype': item['dtype'], 'shape': item['shape'],
                        'data_offsets': [offset, offset + size]}
        offset += size
    raw = json.dumps(header, separators=(',', ':')).encode()
    raw += b' ' * (-len(raw) % 8)
    with path.open('xb') as target:
        target.write(struct.pack('<Q', len(raw)))
        target.write(raw)
        for name in header:
            source_path, begin, size, _ = tensors[name]
            with source_path.open('rb') as source:
                source.seek(begin)
                remaining = size
                while remaining:
                    block = source.read(min(remaining, _CHUNK))
                    if not block:
                        raise ValueError(f'Truncated source tensor: {name}')
                    target.write(block)
                    remaining -= len(block)
        target.flush()
        os.fsync(target.fileno())


def build_overlay(base_path, source_path, output_path) -> dict:
    """Audit all 987 dense tensors and atomically write the 20-tensor overlay.

    Existing, matching output is safe to reuse. Unexpected output is never
    overwritten; only a newly created temporary directory is removed on failure.
    """
    base, source = _root(base_path), _root(source_path)
    base_spec, source_spec = verify_snapshot(base, 'base'), verify_snapshot(source, 'source')
    original, modified = _dense(base, base_spec), _dense(source, source_spec)
    audit = _audit()
    if set(original) != set(modified) or len(original) != audit['shared_dense_tensors']:
        raise ValueError('Dense tensor names differ from the pinned 987-tensor audit')
    expected = {item['name']: item for item in audit['changed']}
    changed = set()
    for name in sorted(original):
        old_path, old_begin, old_size, old = original[name]
        new_path, new_begin, new_size, new = modified[name]
        if old['dtype'] != new['dtype'] or old['shape'] != new['shape'] or old_size != new_size:
            raise ValueError(f'Dense tensor metadata differs: {name}')
        old_sha, new_sha = _hash(old_path, old_begin, old_size), _hash(new_path, new_begin, new_size)
        if old_sha != new_sha:
            record = expected.get(name)
            if (record is None or old_sha != record['base_sha256'] or new_sha != record['sha256']
                    or new['dtype'] != record['dtype'] or new['shape'] != record['shape']):
                raise ValueError(f'Dense change differs from the pinned E38 audit: {name}')
            changed.add(name)
    if changed != set(expected):
        raise ValueError('Exactly the 20 audited E38 changes are required')
    output = Path(output_path).expanduser().absolute()
    if output.exists() or output.is_symlink():
        return verify_overlay(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=f'.{output.name}-', dir=output.parent))
    try:
        _write_overlay(temporary / OVERLAY_FILE, modified)
        receipt = _receipt()
        (temporary / MANIFEST_FILE).write_text(json.dumps(receipt, indent=2) + '\n')
        verify_overlay(temporary)
        # rename refuses to replace a populated output created by another builder.
        temporary.rename(output)
        return receipt
    finally:
        if temporary.exists():
            shutil.rmtree(temporary)
