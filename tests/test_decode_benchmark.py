from contextlib import nullcontext
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from e38_w4a16 import decode_benchmark as bench


class Encoder:
    def __init__(self): self.callback = None
    def register_forward_hook(self, callback):
        self.callback = callback
        return SimpleNamespace(remove=lambda: setattr(self, 'callback', None))
    def finish(self):
        if self.callback is not None: self.callback(self, (), None)


class DecodeTests(unittest.TestCase):
    def test_only_initial_prefill_is_excluded_and_padding_is_never_counted(self):
        clock, syncs, encoder = [0.0], [], Encoder()
        ids = SimpleNamespace(shape=(1, 2), to=lambda *a: ids)
        tokenizer = SimpleNamespace(apply_chat_template=lambda *a, **kw: {'input_ids': ids},
                                    decode=lambda values, **kw: repr(values))

        class Sequence:
            def __getitem__(self, key):
                self.key = key
                return SimpleNamespace(tolist=lambda: [42, 43, 106, 0, 0])

        sequence = Sequence()

        def generate(**options):
            self.assertNotIn('streamer', options)
            clock[0] = 0.4
            encoder.finish()
            clock[0] = 1.2
            encoder.finish()
            clock[0] = 2.0
            return SimpleNamespace(sequences=sequence, tokens_per_forward=3.0)

        model = SimpleNamespace(config=SimpleNamespace(canvas_length=256),
            generation_config=SimpleNamespace(pad_token_id=0, eos_token_id=[1, 50, 106]),
            model=SimpleNamespace(encoder=encoder), generate=generate)
        torch = SimpleNamespace(manual_seed=lambda seed: None, no_grad=nullcontext,
            cuda=SimpleNamespace(synchronize=lambda: syncs.append(clock[0]), empty_cache=lambda: None))
        with patch.dict('sys.modules', {'torch': torch}), patch.object(bench.time, 'perf_counter', side_effect=lambda: clock[0]):
            result = bench.measure(model, tokenizer, {'prompt': 'fixture', 'thinking': False})
        self.assertAlmostEqual(result['decode_seconds'], 1.6)
        self.assertAlmostEqual(result['prompt_setup_and_prefill_seconds'], 0.4)
        self.assertEqual(result['completion_tokens'], 3)
        self.assertTrue(result['terminated'])
        self.assertAlmostEqual(result['decode_tokens_per_second'], 3 / 1.6)
        self.assertEqual(syncs, [0, 0.4, 2.0])
        self.assertIsNone(encoder.callback)
        self.assertEqual(sequence.key[1].start, 2)

    def test_generation_terminators_and_actual_padding_have_distinct_boundaries(self):
        self.assertEqual(bench.committed_tokens([4, 0, 9], 0, [1, 106]), ([4], False))
        self.assertEqual(bench.committed_tokens([4, 106, 0, 9], 0, [1, 106]), ([4, 106], True))
