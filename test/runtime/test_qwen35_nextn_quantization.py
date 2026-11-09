from types import SimpleNamespace
from unittest import mock

import pytest
import torch
from torch import nn
from transformers import PretrainedConfig

import tokenspeed.runtime.layers.logits_processor as logits_module
import tokenspeed.runtime.models.qwen3_5_nextn as nextn_module
from tokenspeed.runtime.distributed.mapping import Mapping
from tokenspeed.runtime.execution.factory import _check_shared_head_layout
from tokenspeed.runtime.execution.forward_batch_info import ForwardMode
from tokenspeed.runtime.layers.dense.nvfp4 import Nvfp4W4A16LinearMethod
from tokenspeed.runtime.layers.logits_processor import (
    LogitsMetadata,
    LogitsProcessor,
    should_apply_lm_head_quant_method,
)
from tokenspeed.runtime.layers.quantization.modelopt_mixed import ModelOptMixedConfig
from tokenspeed.runtime.layers.quantization.nvfp4 import Nvfp4Config
from tokenspeed.runtime.models.base.causal_lm import BaseCausalLM
from tokenspeed.runtime.models.qwen3_5_nextn import (
    Qwen3_5ForConditionalGenerationNextN,
    _resolve_mtp_quant_config,
)


def test_unquantized_mtp_checkpoint_disables_draft_quantization():
    quant_config = Nvfp4Config(exclude_modules=["mtp.layers.0*"])

    assert _resolve_mtp_quant_config(quant_config) is None


def test_quantized_mtp_checkpoint_keeps_draft_quantization():
    quant_config = Nvfp4Config()

    assert _resolve_mtp_quant_config(quant_config) is quant_config


def test_attention_dp_draft_lm_head_receives_quant_config(monkeypatch):
    quant_config = ModelOptMixedConfig(quantized_layers={"lm_head": "W4A16_NVFP4"})
    lm_head = nn.Linear(4, 8, bias=False)
    replicated_linear = mock.Mock(return_value=lm_head)
    monkeypatch.setattr(nextn_module, "ReplicatedLinear", replicated_linear)
    monkeypatch.setattr(
        nextn_module,
        "Qwen3_5DraftForCausalLM",
        mock.Mock(return_value=nn.Module()),
    )
    monkeypatch.setattr(
        nextn_module, "GemmaRMSNorm", mock.Mock(return_value=nn.Identity())
    )
    monkeypatch.setattr(
        nextn_module, "LogitsProcessor", mock.Mock(return_value=nn.Module())
    )
    mapping = Mapping(rank=0, world_size=2, attn_tp_size=1, attn_dp_size=2)
    config = SimpleNamespace(
        hidden_size=4,
        vocab_size=8,
        rms_norm_eps=1e-6,
        num_hidden_layers=1,
        tie_word_embeddings=False,
    )

    draft = Qwen3_5ForConditionalGenerationNextN(
        config,
        mapping,
        quant_config=quant_config,
    )

    assert draft.lm_head is lm_head
    replicated_linear.assert_called_once_with(
        4,
        8,
        bias=False,
        quant_config=quant_config,
        prefix="lm_head",
    )


class _EmbeddingOnlyDraft(nn.Module):
    """CPU stand-in for the transformer; only its shared embedding is needed."""

    def __init__(self, config, mapping, quant_config, prefix):
        super().__init__()
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)


class _HeadOnlyTarget(BaseCausalLM):
    model_cls = _EmbeddingOnlyDraft


@pytest.mark.parametrize("rank", range(8))
@pytest.mark.parametrize("attn_tp", [1, 2, 4])
@pytest.mark.parametrize("tied", [False, True])
def test_attention_dp_mtp_shares_target_head_layout(monkeypatch, rank, attn_tp, tied):
    mapping = Mapping(
        rank=rank, world_size=8, attn_tp_size=attn_tp, attn_dp_size=8 // attn_tp
    )
    config = PretrainedConfig(
        hidden_size=4,
        vocab_size=8,
        rms_norm_eps=1e-6,
        num_hidden_layers=1,
        tie_word_embeddings=tied,
    )
    # The GPU transformer and CUDA cache management are outside this test;
    # the head, logits processors and factory layout check are real classes.
    monkeypatch.setattr(nextn_module, "Qwen3_5DraftForCausalLM", _EmbeddingOnlyDraft)
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
    draft = Qwen3_5ForConditionalGenerationNextN(config, mapping)
    _check_shared_head_layout(target, draft)
    draft.set_embed_and_head_module(target.model.embed_tokens.weight, target.lm_head)

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
        hidden_size=4,
        vocab_size=128,
        rms_norm_eps=1e-6,
        num_hidden_layers=1,
        tie_word_embeddings=False,
    )
    monkeypatch.setattr(nextn_module, "Qwen3_5DraftForCausalLM", _EmbeddingOnlyDraft)
    monkeypatch.setattr(torch.cuda, "empty_cache", lambda: None)
    monkeypatch.setattr(torch.cuda, "synchronize", lambda: None)
    target = _HeadOnlyTarget(config, mapping)
    draft = Qwen3_5ForConditionalGenerationNextN(config, mapping)
    _check_shared_head_layout(target, draft)
    full_weight = torch.arange(128 * 4, dtype=torch.float32).reshape(128, 4) / 100
    width = target.lm_head.weight.shape[0]
    start = mapping.lm_head.tp_rank * width
    with torch.no_grad():
        target.lm_head.weight.copy_(full_weight[start : start + width])
    draft.set_embed_and_head_module(target.model.embed_tokens.weight, target.lm_head)
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


class _DraftModel(nn.Module):
    def __init__(self, tied: bool):
        super().__init__()
        self.model = nn.Module()
        self.model.embed_tokens = nn.Embedding(4, 4)
        self.lm_head = self.model.embed_tokens if tied else nn.Linear(4, 4, bias=False)


def _prepared_nvfp4_head():
    lm_head = nn.Module()
    lm_head.register_parameter(
        "weight",
        nn.Parameter(torch.empty((4, 2), dtype=torch.uint8), requires_grad=False),
    )
    lm_head.register_parameter(
        "weight_scale",
        nn.Parameter(torch.ones((1,), dtype=torch.float32), requires_grad=False),
    )
    lm_head.alpha = torch.ones((1,), dtype=torch.float32)
    lm_head.input_size_per_partition = 4
    lm_head.output_size_per_partition = 4
    lm_head.quant_method = Nvfp4W4A16LinearMethod(SimpleNamespace(group_size=16))
    return lm_head


@pytest.mark.parametrize("tied", [False, True])
def test_mtp_shares_complete_quantized_lm_head(monkeypatch, tied):
    draft = _DraftModel(tied)
    target_embedding = nn.Embedding(4, 4)
    target_lm_head = _prepared_nvfp4_head()
    expected_logits = torch.ones((1, 4), dtype=torch.bfloat16)
    target_lm_head.quant_method.apply = mock.Mock(return_value=expected_logits)
    monkeypatch.setattr(torch.cuda, "empty_cache", mock.Mock())
    monkeypatch.setattr(torch.cuda, "synchronize", mock.Mock())

    Qwen3_5ForConditionalGenerationNextN.set_embed_and_head_module(
        draft, target_embedding.weight, target_lm_head
    )

    assert draft.model.embed_tokens.weight is target_embedding.weight
    assert draft.lm_head is target_lm_head
    assert draft._modules["lm_head"] is target_lm_head
    assert should_apply_lm_head_quant_method(draft.lm_head, draft.lm_head.quant_method)

    processor = LogitsProcessor(
        config=SimpleNamespace(model_type="qwen3_5", vocab_size=4),
        dp_lm_head_tp=False,
    )
    hidden_states = torch.ones((1, 4), dtype=torch.bfloat16)
    metadata = LogitsMetadata(forward_mode=ForwardMode.DECODE, query_shard=None)
    with mock.patch("torch.matmul", side_effect=AssertionError("dense fallback")):
        logits = processor._get_logits(
            hidden_states, draft.lm_head, metadata, require_full_vocab=False
        )

    torch.testing.assert_close(logits, expected_logits)
    target_lm_head.quant_method.apply.assert_called_once_with(
        target_lm_head, hidden_states, None
    )
