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

import os
import sys
from types import SimpleNamespace

import pytest
from tokenizers import Tokenizer, models
from transformers import PretrainedConfig, PreTrainedTokenizerFast

from tokenspeed.runtime.utils.hf_transformers_utils import (
    get_hf_text_config,
    reconcile_special_tokens,
)

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from ci_system.ci_register import register_cuda_ci

register_cuda_ci(est_time=5, suite="runtime-1gpu")


@pytest.mark.parametrize("metadata", ["missing", "null", "fp8"])
def test_optional_quantization_metadata_on_wrapped_text_config(metadata):
    quant = {"quant_method": "fp8", "ignore": ["language_model.lm_head"]}
    fields = (
        {}
        if metadata == "missing"
        else {"quantization_config": quant if metadata == "fp8" else None}
    )
    config = PretrainedConfig(
        architectures=["Qwen2ForCausalLM"],
        text_config=PretrainedConfig(num_attention_heads=2),
        **fields,
    )
    text = get_hf_text_config(config)
    assert text.to_dict().get("quantization_config") == (
        {"quant_method": "fp8", "ignore": ["lm_head"]} if metadata == "fp8" else None
    )


@pytest.mark.parametrize("text_quant", [None, {"quant_method": "awq"}])
def test_outer_quantization_only_fills_missing_text_policy(text_quant):
    text = PretrainedConfig(num_attention_heads=2, quantization_config=text_quant)
    wrapper = PretrainedConfig(
        architectures=["Qwen2ForCausalLM"],
        text_config=text,
        quantization_config={"quant_method": "fp8"},
    )
    resolved = get_hf_text_config(wrapper)
    assert resolved.quantization_config == (
        {"quant_method": "fp8"} if text_quant is None else text_quant
    )


def test_reconcile_registers_missing_specials_without_replacing_tokens():
    backend = Tokenizer(models.WordLevel({"[UNK]": 0, "hello": 1}, unk_token="[UNK]"))
    tokenizer = PreTrainedTokenizerFast(tokenizer_object=backend, unk_token="[UNK]")
    tokenizer.add_special_tokens({"additional_special_tokens": ["<existing>"]})
    tokenizer.backend_tokenizer.add_special_tokens(["<missing>"])
    existing = tokenizer.convert_tokens_to_ids("<existing>")
    missing = tokenizer.convert_tokens_to_ids("<missing>")
    before = tokenizer.get_vocab()
    assert missing not in tokenizer.all_special_ids

    reconcile_special_tokens(tokenizer)
    reconcile_special_tokens(tokenizer)

    assert {existing, missing} <= set(tokenizer.all_special_ids)
    assert tokenizer.get_vocab() == before
    assert tokenizer.decode([existing, 1, missing], skip_special_tokens=True) == "hello"


@pytest.mark.parametrize(
    "added_vocab,stop_ids",
    [
        (
            {"<|eom_id|>": 128_009, "<|eom|>": 200_007, "<|eot|>": 200_008},
            {128_009, 200_008},
        ),
        ({"<|eom|>": 200_007}, None),
    ],
)
def test_reconcile_special_tokens_stops_on_eot_but_not_eom(
    added_vocab, stop_ids
) -> None:
    """<|eom|> ends a channel, not the turn, so only turn-final markers stop."""
    tokenizer = SimpleNamespace(
        added_tokens_decoder=None,
        additional_stop_token_ids=None,
        get_added_vocab=lambda: added_vocab,
    )
    reconcile_special_tokens(tokenizer)
    assert tokenizer.additional_stop_token_ids == stop_ids


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__]))
