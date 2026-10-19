import hashlib
import json
from pathlib import Path
import struct
import tempfile
import unittest
from unittest.mock import patch

from e38_w4a16 import checkpoint as cp


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def tensor_file(values):
    header, payload = {}, bytearray()
    for name in sorted(values):
        value = values[name]
        header[name] = {'dtype': 'BF16', 'shape': [1, 2],
                        'data_offsets': [len(payload), len(payload) + len(value)]}
        payload.extend(value)
    raw = json.dumps(header, separators=(',', ':')).encode()
    raw += b' ' * (-len(raw) % 8)
    return struct.pack('<Q', len(raw)) + raw + payload


class Fixture:
    def __init__(self, root):
        self.base, self.source, self.output = [root / name for name in ('base', 'source', 'overlay')]
        self.base.mkdir()
        self.source.mkdir()
        self.shapes = {name: (1, 2) for name in cp.TENSOR_SHAPES}
        self.original = {name: b'\x00\x00\x00\x00' for name in self.shapes}
        self.original.update({f'unchanged.{n}': b'\x01\x00\x02\x00' for n in range(967)})
        self.modified = dict(self.original)
        self.modified.update({name: b'\x03\x00\x04\x00' for name in self.shapes})
        self.audit = {'shared_dense_tensors': 987, 'base_only_tensors': [], 'source_only_tensors': [],
                      'changed': [{'name': name, 'dtype': 'BF16', 'shape': [1, 2],
                                   'sha256': sha(self.modified[name]), 'base_sha256': sha(self.original[name])}
                                  for name in self.shapes]}
        self.sources = {}
        self.write_source('base', self.original)
        self.write_source('source', self.modified)
        self.expected = tensor_file({name: self.modified[name] for name in self.shapes})

    def write_source(self, which, values):
        root = getattr(self, which)
        (root / 'model.safetensors').write_bytes(tensor_file(values))
        (root / 'config.json').write_bytes(b'{}')
        self.sources[which] = {'model_id': f'fixture/{which}', 'revision': 'a' * 40,
                               'files': [{'path': path.name, 'bytes': path.stat().st_size,
                                          'sha256': sha(path.read_bytes())}
                                         for path in sorted(root.iterdir())]}

    def data(self, name):
        return self.sources if name == 'sources.json' else self.audit

    def patches(self):
        return (patch.object(cp, '_data', side_effect=self.data),
                patch.object(cp, 'TENSOR_SHAPES', self.shapes),
                patch.object(cp, 'OVERLAY_BYTES', len(self.expected)),
                patch.object(cp, 'OVERLAY_SHA256', sha(self.expected)))


class CheckpointTests(unittest.TestCase):
    def fixture(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        value = Fixture(Path(temporary.name))
        for context in value.patches():
            context.start()
            self.addCleanup(context.stop)
        return value

    def test_production_policy_is_twenty_dense_bf16_tensors(self):
        self.assertEqual(len(cp.TENSOR_SHAPES), 20)
        self.assertEqual(cp.TENSOR_SHAPES['model.decoder.layers.11.self_attn.o_proj.weight'], (2816, 8192))
        self.assertEqual(sum(a * b * 2 for a, b in cp.TENSOR_SHAPES.values()), 372703232)
        self.assertEqual(len(cp._audit()['changed']), 20)

    def test_all_pinned_files_are_rehashed_and_same_size_corruption_is_rejected(self):
        value = self.fixture()
        self.assertEqual(cp.verify_snapshot(value.base, 'base')['model_id'], 'fixture/base')
        (value.base / '.cache').mkdir()
        (value.base / '.cache' / 'download.lock').write_bytes(b'not model data')
        cp.verify_snapshot(value.base, 'base')
        (value.base / 'config.json').write_bytes(b'[]')
        with self.assertRaisesRegex(ValueError, 'Pinned base file differs: config.json'):
            cp.verify_snapshot(value.base, 'base')

    def test_paths_symlinks_and_duplicate_records_are_rejected(self):
        value = self.fixture()
        config = value.base / 'config.json'
        config.unlink()
        config.symlink_to(value.source / 'config.json')
        with self.assertRaisesRegex(ValueError, 'without a symlink'):
            cp.verify_snapshot(value.base, 'base')
        config.unlink()
        config.write_bytes(b'{}')
        for name in ('../config.json', '/config.json', 'x/../config.json', 'x\\config.json'):
            value.sources['base']['files'][0]['path'] = name
            with self.subTest(name=name), self.assertRaises(ValueError):
                cp.verify_snapshot(value.base, 'base')
        value.sources['base']['files'] = [value.sources['source']['files'][0]] * 2
        with self.assertRaises(ValueError):
            cp.verify_snapshot(value.base, 'base')

    def test_build_is_byte_exact_and_existing_verified_output_is_reusable(self):
        value = self.fixture()
        receipt = cp.build_overlay(value.base, value.source, value.output)
        self.assertEqual((value.output / cp.OVERLAY_FILE).read_bytes(), value.expected)
        self.assertEqual({p.name for p in value.output.iterdir()}, {cp.MANIFEST_FILE, cp.OVERLAY_FILE})
        self.assertEqual(len(receipt['tensors']), 20)
        self.assertEqual(receipt, cp.build_overlay(value.base, value.source, value.output))
        self.assertEqual(receipt, cp.verify_overlay(value.output))
        self.assertNotIn(str(value.base), json.dumps(receipt))

    def test_every_dense_tensor_is_audited_even_after_new_snapshot_checksum(self):
        value = self.fixture()
        for defect in ('unexpected_dense_change', 'missing_change', 'wrong_e38_bytes', 'missing_tensor'):
            modified = dict(value.modified)
            if defect == 'unexpected_dense_change': modified['unchanged.966'] = b'abcd'
            if defect == 'missing_change': modified[next(iter(value.shapes))] = b'\0\0\0\0'
            if defect == 'wrong_e38_bytes': modified[next(iter(value.shapes))] = b'abcd'
            if defect == 'missing_tensor': del modified['unchanged.966']
            value.write_source('source', modified)
            with self.subTest(defect=defect), self.assertRaises(ValueError):
                cp.build_overlay(value.base, value.source, value.output)
            self.assertFalse(value.output.exists())

    def test_overlay_tampering_and_unrelated_output_are_preserved_and_rejected(self):
        value = self.fixture()
        cp.build_overlay(value.base, value.source, value.output)
        manifest = value.output / cp.MANIFEST_FILE
        original = manifest.read_text()
        body = json.loads(original)
        body['source']['revision'] = 'b' * 40
        manifest.write_text(json.dumps(body))
        with self.assertRaisesRegex(ValueError, 'pinned provenance'):
            cp.verify_overlay(value.output)
        manifest.write_text(original)
        (value.output / 'keep.txt').write_text('unrelated user data')
        with self.assertRaisesRegex(ValueError, 'exactly its manifest'):
            cp.build_overlay(value.base, value.source, value.output)
        self.assertEqual((value.output / 'keep.txt').read_text(), 'unrelated user data')
        (value.output / 'keep.txt').unlink()
        path = value.output / cp.OVERLAY_FILE
        raw = bytearray(path.read_bytes())
        raw[-1] ^= 1
        path.write_bytes(raw)
        with self.assertRaisesRegex(ValueError, 'checksum or size'):
            cp.verify_overlay(value.output)

    def test_interrupted_build_removes_only_its_own_temporary_directory(self):
        value = self.fixture()
        unrelated = value.output.parent / '.overlay-old'
        unrelated.mkdir()
        (unrelated / 'keep.txt').write_text('keep')
        with patch.object(cp, '_write_overlay', side_effect=RuntimeError('disk failure')):
            with self.assertRaisesRegex(RuntimeError, 'disk failure'):
                cp.build_overlay(value.base, value.source, value.output)
        self.assertEqual((unrelated / 'keep.txt').read_text(), 'keep')
        self.assertEqual(sorted(p.name for p in value.output.parent.iterdir()), ['.overlay-old', 'base', 'source'])

    def test_duplicate_json_keys_and_invalid_tensor_offsets_are_rejected(self):
        value = self.fixture()
        path = value.base / 'model.safetensors'
        raw = b'{"x":{},"x":{}}'
        path.write_bytes(struct.pack('<Q', len(raw)) + raw)
        with self.assertRaisesRegex(ValueError, 'Duplicate JSON key'):
            cp._header(path)
        for offsets in ([0, 2], [-1, 3], [0, True], [0, 100000]):
            with self.subTest(offsets=offsets), self.assertRaises(ValueError):
                cp._span(path, 8, {'dtype': 'BF16', 'shape': [1, 2], 'data_offsets': offsets})


if __name__ == '__main__':
    unittest.main()
