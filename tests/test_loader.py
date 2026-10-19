from contextlib import nullcontext
import hashlib
from pathlib import Path
from types import SimpleNamespace
import sys
import tempfile
import unittest
from unittest.mock import patch

from e38_w4a16 import checkpoint as cp
from e38_w4a16 import loader

SHAPES = {'model.decoder.layers.7.self_attn.o_proj.weight': (1, 2),
          'model.decoder.layers.7.mlp.down_proj.weight': (1, 2)}


class Tensor:
    def __init__(self, shape, payload=b'\1\2\3\4', dtype='BF16'):
        self.shape, self.payload, self.dtype = shape, payload, dtype
        self.device, self.is_meta = 'cpu', False

    def contiguous(self): return self
    def view(self, dtype): return self
    def numpy(self): return self
    def tobytes(self): return self.payload
    def to(self, **kwargs): return self


class Parameter(Tensor):
    def copy_(self, other): self.payload = other.payload


class Model:
    def __init__(self):
        self.parameters = {}
        for name, shape in SHAPES.items():
            value = Parameter(shape, b'base')
            self.parameters[name] = value
            self.parameters[name.replace('model.decoder.', 'model.encoder.language_model.')] = value
        self.expert = object()

    def get_parameter(self, name): return self.parameters[name]


class Handle:
    def __init__(self, tensors): self.tensors = tensors
    def __enter__(self): return self
    def __exit__(self, *args): pass
    def keys(self): return self.tensors.keys()
    def get_tensor(self, name): return self.tensors[name]


def receipt():
    return {'base': {'model_id': 'fixture/base'}, 'source': {'model_id': 'fixture/source'},
            'files': [{'sha256': 'a' * 64}],
            'tensors': [{'name': name, 'shape': list(shape), 'dtype': 'BF16',
                         'sha256': hashlib.sha256(b'\1\2\3\4').hexdigest()}
                        for name, shape in SHAPES.items()]}


class LoaderTests(unittest.TestCase):
    def apply(self, model, tensors):
        torch = SimpleNamespace(bfloat16='BF16', uint8='U8',
                                nn=SimpleNamespace(Parameter=Parameter), no_grad=nullcontext)
        with patch.object(cp, 'TENSOR_SHAPES', SHAPES), \
             patch.dict(sys.modules, {'torch': torch, 'safetensors': SimpleNamespace(safe_open=lambda *a, **k: Handle(tensors))}):
            return loader._apply_overlay(model, Path('/unused'), receipt())

    def test_copy_preserves_parameter_ties_and_expert_buffers(self):
        model = Model()
        identities, expert = dict(model.parameters), model.expert
        tensors = {name: Tensor(shape) for name, shape in SHAPES.items()}
        for _ in range(2):
            self.assertEqual(self.apply(model, tensors), 2)
            self.assertIs(model.expert, expert)
            for name, target in identities.items():
                self.assertIs(model.parameters[name], target)
                self.assertEqual(target.payload, b'\1\2\3\4')

    def test_all_replacements_validate_before_any_copy(self):
        last = list(SHAPES)[-1]
        for defect in ('hash', 'dtype', 'shape', 'extra_key', 'alias', 'meta'):
            model = Model()
            tensors = {name: Tensor(shape) for name, shape in SHAPES.items()}
            if defect == 'hash': tensors[last].payload = b'wrong'
            if defect == 'dtype': tensors[last].dtype = 'F16'
            if defect == 'shape': tensors[last].shape = (2, 1)
            if defect == 'extra_key': tensors['arbitrary.expert.weight'] = Tensor((1, 2))
            if defect == 'alias':
                name = last.replace('model.decoder.', 'model.encoder.language_model.')
                model.parameters[name] = Parameter((1, 2), b'base')
            if defect == 'meta': model.parameters[last].is_meta = True
            with self.subTest(defect=defect), self.assertRaises(ValueError):
                self.apply(model, tensors)
            self.assertTrue(all(item.payload == b'base' for item in model.parameters.values()))

    def test_custom_code_requires_explicit_trust_before_any_io_or_import(self):
        for trust in (False, None, 1, 'yes'):
            with patch.object(cp, 'verify_snapshot') as verify, patch.object(loader.importlib, 'import_module') as imports:
                with self.subTest(trust=trust), self.assertRaisesRegex(ValueError, 'trust_base_code=True'):
                    loader.load_model('/missing', '/missing', trust_base_code=trust)
                verify.assert_not_called()
                imports.assert_not_called()

    def test_modified_base_or_overlay_cannot_reach_custom_python(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            for boundary in ('verify_snapshot', 'verify_overlay'):
                with patch.object(cp, 'verify_snapshot', return_value={'files': []}) as base, \
                     patch.object(cp, 'verify_overlay', return_value=receipt()) as overlay, \
                     patch.object(loader.importlib, 'import_module') as imports:
                    (base if boundary == 'verify_snapshot' else overlay).side_effect = ValueError('bad bytes')
                    with self.assertRaisesRegex(ValueError, 'bad bytes'):
                        loader.load_model(root, root, trust_base_code=True)
                    imports.assert_not_called()

    def test_module_from_another_checkpoint_is_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            finder = loader._PinnedImports(root, {'files': [{'path': 'kernels/load_w4_checkpoint.py'}]})
            with patch.dict(sys.modules, {'load_w4_checkpoint': SimpleNamespace(__file__='/other/kernels/load_w4_checkpoint.py')}):
                with self.assertRaisesRegex(RuntimeError, 'another checkpoint'):
                    finder.validate_cache()

    def test_extra_code_weights_bytecode_and_symlinks_are_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            spec = {'files': []}
            (root / '.cache').mkdir()
            (root / '.cache' / 'metadata').write_text('HF metadata')
            loader._check_extra_code(root, spec)
            for name in ('unexpected.py', 'model-extra.safetensors', '__pycache__/old.pyc'):
                path = root / name
                path.parent.mkdir(exist_ok=True)
                path.write_bytes(b'bad')
                with self.subTest(name=name), self.assertRaisesRegex(ValueError, 'Unlisted checkpoint'):
                    loader._check_extra_code(root, spec)
                path.unlink()
            target = root / 'linked.py'
            target.symlink_to(root / '.cache' / 'metadata')
            with self.assertRaisesRegex(ValueError, 'file symlink'):
                loader._check_extra_code(root, spec)

    def test_pinned_loader_uses_cuda_turbo_and_restores_process_import_settings(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / 'kernels').mkdir()
            code = root / 'kernels' / 'load_w4_checkpoint.py'
            code.write_text("def load_fast(path, *, device, tier):\n    return (path, device, tier), 'tokenizer'\n")
            spec = {'files': [{'path': 'kernels/load_w4_checkpoint.py'}]}
            old_path, old_bytecode, old_finders = list(sys.path), sys.dont_write_bytecode, list(sys.meta_path)
            previous = {name: sys.modules.pop(name) for name in ('kernels', 'kernels.load_w4_checkpoint') if name in sys.modules}
            try:
                with patch.object(cp, 'verify_snapshot', return_value=spec), \
                     patch.object(cp, 'verify_overlay', return_value=receipt()), \
                     patch.object(loader, '_apply_overlay', return_value=20) as apply:
                    model, tokenizer, metadata = loader.load_model(root, root, trust_base_code=True)
                    self.assertEqual(model, (str(root), 'cuda', 'turbo'))
                    self.assertEqual(tokenizer, 'tokenizer')
                    self.assertEqual(metadata['modified_dense_tensors'], 20)
                    self.assertFalse(metadata['expert_requantization'])
                    apply.assert_called_once()
                self.assertEqual(sys.path, old_path)
                self.assertEqual(sys.meta_path, old_finders)
                self.assertEqual(sys.dont_write_bytecode, old_bytecode)
                self.assertFalse((root / 'kernels' / '__pycache__').exists())
            finally:
                for name in ('kernels', 'kernels.load_w4_checkpoint'):
                    sys.modules.pop(name, None)
                sys.modules.update(previous)


if __name__ == '__main__':
    unittest.main()
