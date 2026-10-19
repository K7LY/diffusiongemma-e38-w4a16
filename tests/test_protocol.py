from types import SimpleNamespace
import unittest

from e38_w4a16.protocol import ChatRequest, RequestError, parse_output
from e38_w4a16.runtime import CommittedTokens


class ProtocolTests(unittest.TestCase):
    def test_native_tool_call_keeps_reasoning_and_structured_arguments_separate(self):
        raw = '<|channel>thought\n確認します。<channel|><|tool_call>call:read_marker{}<tool_call|><|tool_response>'
        value = parse_output(raw, {'read_marker'})
        self.assertEqual(value['reasoning_content'], '確認します。')
        self.assertIsNone(value['content'])
        self.assertEqual(value['tool_calls'][0]['function'], {'name': 'read_marker', 'arguments': '{}'})

    def test_undeclared_tools_and_unfinished_arguments_cannot_become_valid_calls(self):
        for text in ('<|tool_call>call:undeclared{}<tool_call|>', '<|tool_call>call:read_marker{'):
            with self.subTest(text=text), self.assertRaises(ValueError):
                parse_output(text, {'read_marker'})
        truncated = parse_output('before<|tool_call>call:read_marker{', {'read_marker'}, partial=True)
        self.assertNotIn('tool_calls', truncated)
        self.assertEqual(truncated['content'], 'before')

    def test_generation_eos_is_counted_once_and_canvas_padding_is_not_output(self):
        tokenizer = SimpleNamespace(pad_token_id=1, decode=lambda values, **kwargs: repr(values))
        for terminal in (1, 50, 106):
            with self.subTest(terminal=terminal):
                stream = CommittedTokens(tokenizer, 256, pad_token_id=0, eos_token_ids=[1, 50, 106])
                stream.put(SimpleNamespace(tolist=lambda: [[9, 10]]))
                stream.put(SimpleNamespace(tolist=lambda: [[42, terminal] + [0] * 254]))
                self.assertEqual(stream.snapshot().token_count, 2)
                self.assertTrue(stream.snapshot().terminated)

    def test_unmatched_tool_result_and_unsupported_modalities_are_rejected(self):
        for messages in ([{'role': 'tool', 'tool_call_id': 'missing', 'content': 'data'}],
                         [{'role': 'user', 'content': [{'type': 'image_url'}]}]):
            with self.subTest(messages=messages), self.assertRaises(RequestError):
                ChatRequest.parse({'messages': messages}, 262144)

    def test_output_budget_and_context_bounds_are_validated(self):
        for max_tokens in (0, True, 262145):
            with self.subTest(max_tokens=max_tokens), self.assertRaises(RequestError):
                ChatRequest.parse({'messages': [{'role': 'user', 'content': 'test'}], 'max_tokens': max_tokens}, 262144)
