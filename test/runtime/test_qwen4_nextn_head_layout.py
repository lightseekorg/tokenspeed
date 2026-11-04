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

import pytest
import torch
from torch import nn
from transformers import PretrainedConfig

import tokenspeed.runtime.layers.logits_processor as logits_module
import tokenspeed.runtime.models.qwen4_exp_nextn as nextn_module
from tokenspeed.runtime.distributed.mapping import Mapping
from tokenspeed.runtime.execution.factory import _check_shared_head_layout
from tokenspeed.runtime.execution.forward_batch_info import ForwardMode
from tokenspeed.runtime.layers.logits_processor import LogitsMetadata
from tokenspeed.runtime.models.base.causal_lm import BaseCausalLM
from tokenspeed.runtime.models.qwen4_exp_nextn import Qwen4ExpForCausalLMNextN


class _EmbeddingOnlyDraft(nn.Module):
    """CPU stand-in for transformer weights; real heads handle the logits."""

    def __init__(self, config, mapping, quant_config, prefix):
        super().__init__()
        self.qsa_indexers = []
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)


class _HeadOnlyTarget(BaseCausalLM):
    model_cls = _EmbeddingOnlyDraft


@pytest.mark.parametrize("rank", range(8))
@pytest.mark.parametrize("attn_tp", [1, 2, 4])
def test_attention_dp_mtp_shares_target_head_layout(monkeypatch, rank, attn_tp):
    mapping = Mapping(
        rank=rank, world_size=8, attn_tp_size=attn_tp, attn_dp_size=8 // attn_tp
    )
    config = PretrainedConfig(
        hc_count=4,
        hidden_size=4,
        vocab_size=8,
        rms_norm_eps=1e-6,
        num_hidden_layers=1,
        tie_word_embeddings=False,
    )
    # The GPU transformer and CUDA cache management are outside this test;
    # the head, logits processors and factory layout check are real classes.
    monkeypatch.setattr(nextn_module, "Qwen4ExpDraftModel", _EmbeddingOnlyDraft)
    monkeypatch.setattr(torch.cuda, "empty_cache", lambda: None)
    monkeypatch.setattr(torch.cuda, "synchronize", lambda: None)
    target = _HeadOnlyTarget(config, mapping)
    with torch.no_grad():
        target.lm_head.weight.copy_(
            torch.arange(target.lm_head.weight.numel()).reshape_as(
                target.lm_head.weight
            )
            / 100
            - 0.16
        )
    draft = Qwen4ExpForCausalLMNextN(config, mapping)
    _check_shared_head_layout(target, draft)
    draft.set_embed_and_head(target.model.embed_tokens.weight, target.lm_head.weight)

    hidden = (
        torch.arange(3 * config.hidden_size, dtype=torch.float32).reshape(3, -1) / 7
    )
    metadata = LogitsMetadata(forward_mode=ForwardMode.DECODE, query_shard=None)
    actual = draft.logits_processor._get_logits(
        hidden, draft.lm_head, metadata, require_full_vocab=False
    )
    expected = torch.nn.functional.linear(hidden, target.lm_head.weight)
    torch.testing.assert_close(actual, expected)


@pytest.mark.parametrize("rank", range(8))
def test_attention_dp_mtp_shares_vocab_sharded_head(monkeypatch, rank):
    mapping = Mapping(
        rank=rank, world_size=8, attn_tp_size=1, attn_dp_size=8, lm_head_tp_size=2
    )
    config = PretrainedConfig(
        hc_count=4,
        hidden_size=4,
        vocab_size=128,
        rms_norm_eps=1e-6,
        num_hidden_layers=1,
        tie_word_embeddings=False,
    )
    monkeypatch.setattr(nextn_module, "Qwen4ExpDraftModel", _EmbeddingOnlyDraft)
    monkeypatch.setattr(torch.cuda, "empty_cache", lambda: None)
    monkeypatch.setattr(torch.cuda, "synchronize", lambda: None)
    target = _HeadOnlyTarget(config, mapping)
    draft = Qwen4ExpForCausalLMNextN(config, mapping)
    _check_shared_head_layout(target, draft)
    full_weight = torch.arange(128 * 4, dtype=torch.float32).reshape(128, 4) / 100
    width = target.lm_head.weight.shape[0]
    start = mapping.lm_head.tp_rank * width
    with torch.no_grad():
        target.lm_head.weight.copy_(full_weight[start : start + width])
    draft.set_embed_and_head(target.model.embed_tokens.weight, target.lm_head.weight)
    rows = [1 + r % 3 for r in range(8)]
    hidden = [
        torch.arange(n * 4, dtype=torch.float32).reshape(n, 4) / 7 + r
        for r, n in enumerate(rows)
    ]

    # Only the distributed transport is simulated on CPU. The real target and
    # draft heads, layout validation, and logits GEMM handle distinct DP rows
    # and vocabulary shards; the peer contributes its independently computed shard.
    def gather(local, group, counts):
        return torch.cat(
            [
                (local if r == rank else hidden[r])[:count]
                for r, count in zip(group, counts)
            ]
        )

    def transpose(local_logits, group, *, input_split_sizes):
        offset = sum(input_split_sizes[: group.index(rank)])
        return torch.cat(
            [
                (
                    local_logits[offset : offset + rows[rank]]
                    if r == rank
                    else torch.nn.functional.linear(
                        hidden[rank], full_weight[i * width : (i + 1) * width]
                    )
                )
                for i, r in enumerate(group)
            ],
            dim=1,
        )

    monkeypatch.setattr(logits_module, "token_all_gather", gather)
    monkeypatch.setattr(logits_module, "all_to_all_transpose", transpose)
    metadata = LogitsMetadata(
        forward_mode=ForwardMode.DECODE,
        query_shard=None,
        global_num_tokens=rows,
        all_decode_or_idle=True,
    )
    actual = draft.logits_processor._get_logits(
        hidden[rank], draft.lm_head, metadata, require_full_vocab=False
    )
    torch.testing.assert_close(
        actual, torch.nn.functional.linear(hidden[rank], full_weight)
    )
