from types import SimpleNamespace
from unittest import mock

import pytest
import torch
from torch import nn

import tokenspeed.runtime.models.qwen3_5_nextn as nextn_module
from tokenspeed.runtime.execution.forward_batch_info import ForwardMode
from tokenspeed.runtime.layers.dense.nvfp4 import Nvfp4W4A16LinearMethod
from tokenspeed.runtime.layers.logits_processor import (
    LogitsMetadata,
    LogitsProcessor,
    should_apply_lm_head_quant_method,
)
from tokenspeed.runtime.layers.quantization.modelopt_mixed import ModelOptMixedConfig
from tokenspeed.runtime.layers.quantization.nvfp4 import Nvfp4Config
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
    mapping = SimpleNamespace(
        attn=SimpleNamespace(
            has_dp=True,
            tp_rank=0,
            tp_size=1,
            tp_group=None,
        )
    )
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
        config=SimpleNamespace(model_type="qwen3_5", vocab_size=4)
    )
    hidden_states = torch.ones((1, 4), dtype=torch.bfloat16)
    metadata = LogitsMetadata(forward_mode=ForwardMode.DECODE)
    with mock.patch("torch.matmul", side_effect=AssertionError("dense fallback")):
        logits = processor._get_logits(hidden_states, draft.lm_head, metadata)

    torch.testing.assert_close(logits, expected_logits)
    target_lm_head.quant_method.apply.assert_called_once_with(
        target_lm_head, hidden_states, None
    )
