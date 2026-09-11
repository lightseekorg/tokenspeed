# Copyright (c) 2026 LightSeek Foundation
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

"""V4.1 text-only chat integration and checkpoint encoder parity.

Set DEEPSEEK_V41_REFERENCE_DIR to a trusted local V4.1 snapshot to run the
reference checks. These execute its standalone encoding/encoding.py, load only
its tokenizer (no model weights), and verify the 99092-entry Engram vocabulary.
The remaining tests run without a checkpoint or network access.
"""

import copy
import importlib.util
import os
from pathlib import Path
from unittest.mock import MagicMock, Mock, patch

import pytest
from tokenizers import Tokenizer, models, pre_tokenizers, processors
from transformers import BatchEncoding, PreTrainedTokenizerFast

from tokenspeed.runtime.utils import hf_transformers_utils as hf


@pytest.fixture
def base_tokenizer():
    backend = Tokenizer(
        models.WordLevel(
            vocab={"<bos>": 0, "<unk>": 1, "<pad>": 2, "hello": 3},
            unk_token="<unk>",
        )
    )
    backend.pre_tokenizer = pre_tokenizers.Whitespace()
    backend.post_processor = processors.TemplateProcessing(
        single="<bos> $A", special_tokens=[("<bos>", 0)]
    )
    return PreTrainedTokenizerFast(
        tokenizer_object=backend,
        bos_token="<bos>",
        unk_token="<unk>",
        pad_token="<pad>",
    )


@pytest.mark.parametrize("effort", [None, "low", "high", "max", 1, 37, 100])
def test_effort_is_forwarded_without_v4_filtering(base_tokenizer, effort):
    encoder = Mock(return_value="<bos> hello")
    tokenizer = hf._wrap_deepseek_tokenizer(base_tokenizer, encoder, is_v41=True)
    messages = [{"role": "user", "content": "hello"}]
    assert (
        tokenizer.apply_chat_template(
            messages,
            tools=None,
            tokenize=False,
            enable_thinking=True,
            drop_thinking=False,
            reasoning_effort=effort,
        )
        == "<bos> hello"
    )
    encoder.assert_called_once_with(
        messages, thinking_mode="thinking", drop_thinking=False, reasoning_effort=effort
    )


@pytest.mark.parametrize("return_dict", [False, True])
@pytest.mark.parametrize("return_tensors", [None, "pt"])
def test_tokenization_has_one_bos_and_preserves_backend(
    base_tokenizer, return_dict, return_tensors
):
    backend_state = base_tokenizer.backend_tokenizer.to_str()
    encoder = Mock(return_value="<bos> hello")
    tokenizer = hf._wrap_deepseek_tokenizer(base_tokenizer, encoder, is_v41=True)
    assert tokenizer.backend_tokenizer.to_str() == backend_state
    result = tokenizer.apply_chat_template(
        [{"role": "user", "content": "hello"}],
        tools=None,
        tokenize=True,
        return_dict=return_dict,
        return_tensors=return_tensors,
        return_attention_mask=True,
        truncation=True,
        max_length=4,
        padding="max_length",
        add_special_tokens=True,
    )
    if return_dict:
        assert isinstance(result, BatchEncoding)
        mask = result["attention_mask"]
        assert (mask.tolist() if return_tensors else mask) == (
            [[1, 1, 0, 0]] if return_tensors else [1, 1, 0, 0]
        )
        result = result["input_ids"]
    assert (result.tolist() if return_tensors else result) == (
        [[0, 3, 2, 2]] if return_tensors else [0, 3, 2, 2]
    )
    assert tokenizer is not base_tokenizer
    assert tokenizer.backend_tokenizer is base_tokenizer.backend_tokenizer
    assert len(tokenizer) == len(base_tokenizer) == 4
    assert tokenizer.get_vocab() == base_tokenizer.get_vocab()
    assert tokenizer.get_added_vocab() == base_tokenizer.get_added_vocab()


@pytest.mark.parametrize("field", ["content", "content_blocks"])
@pytest.mark.parametrize(
    "content",
    [
        [{"type": "image_url", "image_url": {"url": "unused.png"}}],
        [{"type": "image", "source": {"type": "base64", "data": "unused"}}],
        [{"type": "input_audio", "input_audio": {"data": "unused", "format": "wav"}}],
        [{"type": "video_url", "video_url": {"url": "unused.mp4"}}],
        [{"type": "tool_result", "content": [{"type": "image", "url": "unused.png"}]}],
        [{"type": "tool_result", "content": "<｜deepseek_image｜>"}],
        [{"type": "text", "text": "<｜deepseek_image｜>"}],
        "<｜deepseek_image｜>",
    ],
)
def test_media_is_rejected_before_encoding(base_tokenizer, field, content):
    encoder = Mock(return_value="must not encode media")
    tokenizer = hf._wrap_deepseek_tokenizer(base_tokenizer, encoder, is_v41=True)
    with pytest.raises(ValueError, match="text-only"):
        tokenizer.apply_chat_template(
            [{"role": "user", field: content}], tools=None, tokenize=False
        )
    encoder.assert_not_called()


def test_v41_requires_explicit_trust_before_loading():
    with (
        patch.object(hf, "snapshot_download") as download,
        patch.object(hf.AutoTokenizer, "from_pretrained") as load,
        patch.object(hf, "_load_deepseek_encode_messages") as encoder,
        pytest.raises(ValueError, match="--trust-remote-code"),
    ):
        hf.get_tokenizer(
            "deepseek-ai/DeepSeek-V4.1-Flash",
            tokenizer_mode="auto",
            trust_remote_code=False,
            tokenizer_revision=None,
            revision=None,
            architectures=["DeepseekV41ForCausalLM"],
        )
    download.assert_not_called()
    load.assert_not_called()
    encoder.assert_not_called()


def test_remote_v41_uses_pinned_snapshot_encoder(base_tokenizer, tmp_path):
    commit = "b" * 40
    snapshot = str(tmp_path / "snapshots" / commit)
    repo = "deepseek-ai/DeepSeek-V4.1-Flash"
    lock = MagicMock()
    encoder = Mock(return_value="<bos> hello")
    with (
        patch(
            "tokenspeed.runtime.model_loader.weight_utils.get_lock", return_value=lock
        ),
        patch.object(hf, "snapshot_download", return_value=snapshot) as download,
        patch.object(
            hf.AutoTokenizer, "from_pretrained", return_value=base_tokenizer
        ) as load,
        patch.object(
            hf, "_load_deepseek_encode_messages", return_value=encoder
        ) as load_encoder,
        patch.object(hf, "_load_deepseek_v4_encode_messages") as load_v4,
        patch.object(hf, "cached_file") as cached,
    ):
        tokenizer = hf.get_tokenizer(
            repo,
            tokenizer_mode="auto",
            trust_remote_code=True,
            tokenizer_revision=None,
            revision="moving-branch",
            architectures=["DeepseekV41ForCausalLM"],
        )
    download.assert_called_once_with(
        repo,
        revision="moving-branch",
        ignore_patterns=["*.pt", "*.safetensors", "*.bin"],
    )
    load.assert_called_once_with(
        repo,
        revision=commit,
        trust_remote_code=True,
        clean_up_tokenization_spaces=False,
    )
    load_encoder.assert_called_once_with(
        os.path.join(snapshot, "encoding", "encoding.py")
    )
    load_v4.assert_not_called()
    cached.assert_not_called()
    lock.__enter__.assert_called_once()
    lock.__exit__.assert_called_once()
    assert tokenizer.name_or_path == repo
    assert tokenizer.init_kwargs["name_or_path"] == repo
    assert (
        tokenizer.apply_chat_template(
            [{"role": "user", "content": "hello"}], tools=None, tokenize=False
        )
        == "<bos> hello"
    )


def test_missing_local_v41_encoder_does_not_fall_back(base_tokenizer, tmp_path):
    with (
        patch.object(hf.AutoTokenizer, "from_pretrained", return_value=base_tokenizer),
        patch.object(hf, "snapshot_download") as download,
        patch.object(hf, "cached_file") as cached,
        pytest.raises(RuntimeError, match="encoding/encoding.py"),
    ):
        hf.get_tokenizer(
            str(tmp_path),
            tokenizer_mode="auto",
            trust_remote_code=True,
            tokenizer_revision=None,
            revision=None,
            architectures=["DeepseekV41ForCausalLM"],
        )
    download.assert_not_called()
    cached.assert_not_called()


@pytest.fixture(scope="module")
def reference():
    directory = os.environ.get("DEEPSEEK_V41_REFERENCE_DIR")
    if directory is None:
        pytest.skip("set DEEPSEEK_V41_REFERENCE_DIR to a trusted V4.1 snapshot")
    spec = importlib.util.spec_from_file_location(
        "_reference_deepseek_v41_encoding", Path(directory) / "encoding" / "encoding.py"
    )
    assert spec is not None and spec.loader is not None
    encoder = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(encoder)
    # Keep encoder parity independent of model config initialization; exercise
    # AutoTokenizer through get_tokenizer separately below.
    raw = PreTrainedTokenizerFast.from_pretrained(
        directory,
        trust_remote_code=True,
        local_files_only=True,
        clean_up_tokenization_spaces=False,
    )
    tokenizer = hf._wrap_deepseek_tokenizer(
        copy.deepcopy(raw),
        hf._load_deepseek_encode_messages(
            os.path.join(directory, "encoding", "encoding.py")
        ),
        is_v41=True,
    )
    hf.reconcile_special_tokens(tokenizer)
    assert raw.chat_template is None
    return encoder, raw, tokenizer


def test_local_checkpoint_auto_selection(reference):
    _, raw, wrapped = reference
    tokenizer = hf.get_tokenizer(
        raw.name_or_path,
        tokenizer_mode="auto",
        trust_remote_code=True,
        tokenizer_revision=None,
        revision=None,
        architectures=["DeepseekV41ForCausalLM"],
        local_files_only=True,
    )
    messages = [{"role": "user", "content": "hello"}]
    options = dict(tools=None, tokenize=True, thinking=True, reasoning_effort="low")
    assert tokenizer.apply_chat_template(
        messages, **options
    ) == wrapped.apply_chat_template(messages, **options)
    assert tokenizer.get_vocab() == raw.get_vocab()
    assert len(tokenizer) == len(raw)


@pytest.mark.parametrize("thinking_key", ["thinking", "enable_thinking"])
@pytest.mark.parametrize(
    ("thinking", "effort", "budget"),
    [
        (False, None, None),
        (True, None, 75),
        (True, "low", 50),
        (True, "high", 75),
        (True, "max", 100),
        (True, 1, 1),
        (True, 37, 37),
        (True, 100, 100),
    ],
)
@pytest.mark.parametrize("mid_system", [False, True])
def test_chat_and_thinking_match_reference(
    reference, thinking_key, thinking, effort, budget, mid_system
):
    encoder, raw, tokenizer = reference
    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "What is 2+2?"},
    ]
    if mid_system:
        messages.extend(
            [
                {
                    "role": "assistant",
                    "content": "4",
                    "reasoning_content": "old reasoning",
                },
                {"role": "system", "content": "Now answer in French."},
            ]
        )
    original = copy.deepcopy(messages)
    expected = encoder.encode_messages(
        messages,
        thinking_mode="thinking" if thinking else "chat",
        context=None,
        drop_thinking=True,
        add_default_bos_token=True,
        reasoning_effort=effort,
        return_multi_modal_data=False,
    )
    options = {
        thinking_key: thinking,
        "reasoning_effort": effort,
        "drop_thinking": True,
    }
    prompt = tokenizer.apply_chat_template(
        messages, tools=None, tokenize=False, add_generation_prompt=True, **options
    )
    assert prompt == expected
    assert prompt.endswith(
        "<｜Assistant｜><think>" if thinking else "<｜Assistant｜></think>"
    )
    if budget is not None:
        assert prompt.count(f"Reasoning Effort: {budget} (range 1-100,") == 1
    else:
        assert "Reasoning Effort:" not in prompt
    if mid_system:
        assert "<｜System｜>Now answer in French.<｜Assistant｜>" in prompt
        assert "old reasoning" not in prompt
    result = tokenizer.apply_chat_template(
        messages,
        tools=None,
        tokenize=True,
        return_dict=True,
        return_attention_mask=True,
        **options,
    )
    assert isinstance(result, BatchEncoding)
    assert result == raw(expected, add_special_tokens=False, return_attention_mask=True)
    assert result["input_ids"].count(raw.bos_token_id) == 1
    assert result["input_ids"][0] == raw.bos_token_id
    assert messages == original


def test_text_blocks_and_spaced_dsml_match_reference(reference):
    encoder, _, tokenizer = reference
    tools = [
        {
            "type": "function",
            "function": {
                "name": "lookup",
                "description": "Look up a value",
                "parameters": {
                    "type": "object",
                    "properties": {"query": {"type": "string"}},
                },
            },
        }
    ]
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "Find "},
                {"type": "text", "text": "a value."},
            ],
        },
        {
            "role": "assistant",
            "content": None,
            "reasoning_content": "Need a lookup.",
            "tool_calls": [
                {
                    "id": "call_1",
                    "type": "function",
                    "function": {
                        "name": "lookup",
                        "arguments": '{"query":"value"}',
                    },
                }
            ],
        },
        {
            "role": "tool",
            "tool_call_id": "call_1",
            "content": [{"type": "text", "text": "42"}],
        },
        {"role": "system", "content": "Summarize the result."},
    ]
    original = copy.deepcopy((messages, tools))
    expected = encoder.encode_messages(
        [{"role": "system", "tools": tools}] + messages,
        thinking_mode="thinking",
        context=None,
        drop_thinking=True,
        add_default_bos_token=True,
        reasoning_effort="low",
        return_multi_modal_data=False,
    )
    prompt = tokenizer.apply_chat_template(
        messages,
        tools=tools,
        tokenize=False,
        thinking=True,
        reasoning_effort="low",
        drop_thinking=True,
    )
    assert prompt == expected
    assert '<｜DSML｜ calls>\n<｜DSML｜ invoke name="lookup">' in prompt
    assert (
        '<｜DSML｜ parameter name="query" string="true">value</｜DSML｜ parameter>'
        in prompt
    )
    assert "<｜DSML｜tool_calls>" not in prompt
    assert "<tool_result>42</tool_result>" in prompt
    assert "<think>Need a lookup.</think>" in prompt
    assert (messages, tools) == original


@pytest.mark.parametrize("effort", [0, 101, True, 1.5, "medium", "37"])
def test_invalid_effort_is_not_silently_replaced(reference, effort):
    _, _, tokenizer = reference
    with pytest.raises(AssertionError, match="Invalid reasoning effort"):
        tokenizer.apply_chat_template(
            [{"role": "user", "content": "hello"}],
            tools=None,
            tokenize=False,
            thinking=True,
            reasoning_effort=effort,
        )


def test_engram_token_map_is_unchanged(reference):
    from tokenspeed.runtime.models.deepseek_v41_engram import build_compressed_token_map

    _, raw, tokenizer = reference
    assert len(tokenizer) == len(raw)
    assert tokenizer.get_vocab() == raw.get_vocab()
    assert tokenizer.get_added_vocab() == raw.get_added_vocab()
    assert tokenizer.backend_tokenizer.to_str() == raw.backend_tokenizer.to_str()
    expected_map, expected_size = build_compressed_token_map(raw)
    token_map, size = build_compressed_token_map(tokenizer)
    assert size == expected_size == 99092
    assert token_map == expected_map
