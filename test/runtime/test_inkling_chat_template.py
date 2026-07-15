"""CPU-only golden tests for the packaged TMLv0 Inkling chat template."""

import json
import math
import os
import sys
import unittest
from pathlib import Path

import jinja2

# CI registration is parsed statically; this test itself is CPU-only.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from ci_system.ci_register import register_cuda_ci  # noqa: E402

register_cuda_ci(est_time=1, suite="runtime-1gpu")


TEMPLATE_PATH = (
    Path(__file__).resolve().parents[2]
    / "python"
    / "tokenspeed"
    / "runtime"
    / "chat_templates"
    / "inkling.jinja"
)


def _raise_exception(message):
    raise ValueError(message)


def _tojson(value, ensure_ascii=True, sort_keys=False, separators=None):
    if separators is not None:
        separators = tuple(separators)
    return json.dumps(
        value,
        ensure_ascii=ensure_ascii,
        sort_keys=sort_keys,
        separators=separators,
    )


def _render(messages, **kwargs):
    env = jinja2.Environment()
    env.globals["raise_exception"] = _raise_exception
    env.filters["tojson"] = _tojson
    template = env.from_string(TEMPLATE_PATH.read_text(encoding="utf-8"))
    return template.render(
        messages=messages,
        add_generation_prompt=kwargs.pop("add_generation_prompt", True),
        tools=kwargs.pop("tools", []),
        **kwargs,
    )


class TestInklingChatTemplate(unittest.TestCase):
    def test_completion_golden_has_no_synthetic_suffix_or_effort(self):
        expected = "<|message_user|><|content_text|>hi<|end_message|>"

        self.assertEqual(_render([{"role": "user", "content": "hi"}]), expected)
        self.assertEqual(
            _render(
                [{"role": "user", "content": "hi"}],
                add_generation_prompt=False,
            ),
            expected,
        )

    def test_tool_call_golden_has_model_author_name(self):
        rendered = _render(
            [
                {"role": "user", "content": "weather"},
                {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [
                        {
                            "id": "call_1",
                            "type": "function",
                            "function": {
                                "name": "get_weather",
                                "arguments": '{"city":"SF"}',
                            },
                        }
                    ],
                },
            ]
        )

        self.assertEqual(
            rendered,
            "<|message_user|><|content_text|>weather<|end_message|>"
            "<|message_model|>get_weather<|content_invoke_tool_json|>"
            '{"name":"get_weather","args":{"city":"SF"}}<|end_message|>'
            "<|content_model_end_sampling|>",
        )

    def test_standard_tool_result_does_not_infer_author_name(self):
        rendered = _render(
            [
                {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [
                        {
                            "id": "call_1",
                            "function": {"name": "lookup", "arguments": {}},
                        }
                    ],
                },
                {
                    "role": "tool",
                    "tool_call_id": "call_1",
                    "content": "ok",
                },
            ]
        )

        self.assertEqual(
            rendered,
            "<|message_model|>lookup<|content_invoke_tool_json|>"
            '{"name":"lookup","args":{}}<|end_message|>'
            "<|content_model_end_sampling|>"
            "<|message_tool|><|content_text|>ok<|end_message|>",
        )

    def test_turn1_tool_system_effort_prefix_order_golden(self):
        rendered = _render(
            [
                {"role": "system", "content": "sys"},
                {"role": "user", "content": "weather"},
            ],
            tools=[
                {
                    "type": "function",
                    "function": {
                        "name": "get_weather",
                        "description": "Get weather",
                        "parameters": {
                            "type": "object",
                            "properties": {"city": {"type": "string"}},
                        },
                    },
                }
            ],
            reasoning_effort=0.9,
        )

        self.assertEqual(
            rendered,
            "<|message_system|>tool_declare<|content_xml|>"
            '[{"description":"Get weather","name":"get_weather",'
            '"parameters":{"properties":{"city":{"type":"string"}},'
            '"type":"object"},"type":"function"}]<|end_message|>'
            "<|message_system|><|content_text|>sys<|end_message|>"
            "<|message_system|><|content_text|>Thinking effort level: 0.9"
            "<|end_message|>"
            "<|message_user|><|content_text|>weather<|end_message|>",
        )
        self.assertEqual(rendered.count("Thinking effort level:"), 1)

    def test_turn2_keeps_one_prefix_effort_and_closes_historical_assistant(self):
        rendered = _render(
            [
                {"role": "system", "content": "sys"},
                {"role": "user", "content": "q"},
                {"role": "assistant", "content": "a"},
                {"role": "user", "content": "q2"},
            ],
            tools=[
                {
                    "type": "function",
                    "function": {
                        "name": "lookup",
                        "description": "Look up",
                        "parameters": {},
                    },
                }
            ],
            reasoning_effort=0.9,
        )

        self.assertEqual(
            rendered,
            "<|message_system|>tool_declare<|content_xml|>"
            '[{"description":"Look up","name":"lookup","parameters":{},'
            '"type":"function"}]<|end_message|>'
            "<|message_system|><|content_text|>sys<|end_message|>"
            "<|message_system|><|content_text|>Thinking effort level: 0.9"
            "<|end_message|>"
            "<|message_user|><|content_text|>q<|end_message|>"
            "<|message_model|><|content_text|>a<|end_message|>"
            "<|content_model_end_sampling|>"
            "<|message_user|><|content_text|>q2<|end_message|>",
        )
        self.assertEqual(rendered.count("Thinking effort level:"), 1)
        self.assertEqual(rendered.count("<|content_model_end_sampling|>"), 1)

    def test_omitted_or_null_reasoning_effort_inserts_nothing(self):
        self.assertEqual(_render([]), "")
        self.assertEqual(_render([], reasoning_effort=None), "")

    def test_reasoning_effort_named_levels_and_boundaries(self):
        for reasoning_effort, expected_value in (
            ("none", "0"),
            ("minimal", "0"),
            ("low", "0.2"),
            ("medium", "0.7"),
            ("high", "0.9"),
            ("xhigh", "0.99"),
            ("max", "0.99"),
            (0, "0"),
            ("0.75", "0.75"),
            (0.99, "0.99"),
        ):
            with self.subTest(reasoning_effort=reasoning_effort):
                self.assertEqual(
                    _render([], reasoning_effort=reasoning_effort),
                    "<|message_system|><|content_text|>Thinking effort level: "
                    f"{expected_value}<|end_message|>",
                )

    def test_reasoning_effort_rejects_values_outside_closed_range(self):
        for reasoning_effort in (
            -0.01,
            0.991,
            1,
            "1",
            "bogus",
            math.nan,
            math.inf,
            -math.inf,
            True,
            {},
        ):
            with self.subTest(reasoning_effort=reasoning_effort):
                with self.assertRaisesRegex(ValueError, r"\[0, 0\.99\]"):
                    _render([], reasoning_effort=reasoning_effort)

    def test_reasoning_effort_uses_ties_to_even_rounding(self):
        for reasoning_effort, expected_value in (
            (0.005, "0"),
            (0.0051, "0"),
            (0.0149, "0.02"),
            (0.015, "0.02"),
            (0.025, "0.02"),
            (0.545, "0.55"),
            (0.575, "0.57"),
            (0.755, "0.76"),
            (0.985, "0.98"),
        ):
            with self.subTest(reasoning_effort=reasoning_effort):
                self.assertEqual(
                    _render([], reasoning_effort=reasoning_effort),
                    "<|message_system|><|content_text|>Thinking effort level: "
                    f"{expected_value}<|end_message|>",
                )

    def test_empty_scalar_content_emits_empty_text_frame(self):
        for role, suffix in (
            ("system", ""),
            ("user", ""),
            ("assistant", "<|content_model_end_sampling|>"),
        ):
            with self.subTest(role=role):
                role_marker = "model" if role == "assistant" else role
                self.assertEqual(
                    _render([{"role": role, "content": None}]),
                    f"<|message_{role_marker}|><|content_text|><|end_message|>"
                    + suffix,
                )

    def test_empty_assistant_text_is_omitted_when_turn_has_tool_call(self):
        expected = (
            "<|message_model|>lookup<|content_invoke_tool_json|>"
            '{"name":"lookup","args":{}}<|end_message|>'
            "<|content_model_end_sampling|>"
        )
        tool_calls = [{"function": {"name": "lookup", "arguments": {}}}]

        for content in (None, ""):
            with self.subTest(content=content):
                self.assertEqual(
                    _render(
                        [
                            {
                                "role": "assistant",
                                "content": content,
                                "tool_calls": tool_calls,
                            }
                        ]
                    ),
                    expected,
                )

    def test_renders_multipart_content_in_authored_order(self):
        rendered = _render(
            [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "describe"},
                        {"type": "image"},
                        {"type": "audio"},
                        {"type": "audio"},
                        {"type": "text", "text": ""},
                    ],
                }
            ]
        )

        self.assertEqual(
            rendered,
            "<|message_user|><|content_text|>describe<|end_message|>"
            "<|message_user|><|content_image|><|end_message|>"
            "<|message_user|><|content_audio_input|><|audio_end|><|end_message|>"
            "<|message_user|><|content_audio_input|><|audio_end|><|end_message|>"
            "<|message_user|><|content_text|><|end_message|>",
        )
        self.assertEqual(rendered.count("<|audio_end|>"), 2)

    def test_assistant_multiframe_turn_has_only_one_sampling_end(self):
        rendered = _render(
            [
                {
                    "role": "assistant",
                    "reasoning_content": "think",
                    "content": "answer",
                    "tool_calls": [
                        {
                            "function": {
                                "name": "lookup",
                                "arguments": {"q": "x"},
                            }
                        }
                    ],
                }
            ]
        )

        self.assertEqual(rendered.count("<|content_model_end_sampling|>"), 1)
        self.assertTrue(rendered.endswith("<|content_model_end_sampling|>"))

    def test_each_of_two_historical_assistant_turns_has_one_sampling_end(self):
        rendered = _render(
            [
                {"role": "user", "content": "q"},
                {"role": "assistant", "content": "a"},
                {"role": "user", "content": "q2"},
                {"role": "assistant", "content": "a2"},
                {"role": "user", "content": "q3"},
            ],
            reasoning_effort=0.7,
        )

        self.assertEqual(
            rendered,
            "<|message_system|><|content_text|>Thinking effort level: 0.7"
            "<|end_message|>"
            "<|message_user|><|content_text|>q<|end_message|>"
            "<|message_model|><|content_text|>a<|end_message|>"
            "<|content_model_end_sampling|>"
            "<|message_user|><|content_text|>q2<|end_message|>"
            "<|message_model|><|content_text|>a2<|end_message|>"
            "<|content_model_end_sampling|>"
            "<|message_user|><|content_text|>q3<|end_message|>",
        )
        self.assertEqual(rendered.count("Thinking effort level:"), 1)
        self.assertEqual(rendered.count("<|content_model_end_sampling|>"), 2)

    def test_rejects_external_and_unknown_content_part_types(self):
        for part_type in (
            "input_text",
            "image_url",
            "input_image",
            "audio_url",
            "input_audio",
            "video",
        ):
            with self.subTest(part_type=part_type):
                with self.assertRaisesRegex(
                    ValueError, "unsupported canonical Inkling content part type"
                ):
                    _render([{"role": "user", "content": [{"type": part_type}]}])

    def test_rejects_content_part_without_canonical_type(self):
        with self.assertRaisesRegex(ValueError, "require a canonical type"):
            _render([{"role": "user", "content": [{"text": "hello"}]}])

    def test_preserves_user_text_whitespace_exactly(self):
        user_text = "  leading spaces\n\tmiddle\ntrailing spaces  "

        rendered = _render(
            [{"role": "user", "content": [{"type": "text", "text": user_text}]}]
        )

        self.assertEqual(
            rendered,
            "<|message_user|><|content_text|>" + user_text + "<|end_message|>",
        )


if __name__ == "__main__":
    unittest.main()
