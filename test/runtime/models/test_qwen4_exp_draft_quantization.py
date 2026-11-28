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

"""Draft-only quantization of the Qwen3.8 MTP draft: the FP8 LM-head copy and
the MXFP4 routed experts (--speculative-draft-lm-head-quantization,
--speculative-draft-moe-quantization)."""

import pytest
import torch
from tokenspeed_kernel.numerics.torch.quantization import fake_quantize_mxfp4_even
from transformers import PretrainedConfig

import tokenspeed.runtime.models.qwen4_exp_nextn as qwen4_exp_nextn
from tokenspeed.runtime.distributed.mapping import Mapping
from tokenspeed.runtime.execution.forward_batch_info import ForwardMode
from tokenspeed.runtime.layers.dense.fp8_weight_only import (
    Fp8WeightOnlyLinearMethod,
    Fp8WeightOnlyLMHead,
)
from tokenspeed.runtime.layers.dense.unquant import UnquantizedLinearMethod
from tokenspeed.runtime.layers.linear import ReplicatedLinear
from tokenspeed.runtime.layers.logits_processor import LogitsMetadata
from tokenspeed.runtime.layers.moe.expert import moe_expert_weight_dtype
from tokenspeed.runtime.layers.quantization.mxfp4 import (
    Mxfp4Config,
    dequantize_mxfp4_to_bf16,
    quantize_bf16_to_mxfp4,
)
from tokenspeed.runtime.layers.vocab_parallel_embedding import ParallelLMHead
from tokenspeed.runtime.utils.server_args import (
    ServerArgs,
    draft_quantization_refusal,
)

_requires_cuda = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="requires a GPU"
)


# ---------------------------------------------------------------------------
# MXFP4 weight quantization (Quark "even" scales, RNE codes)
# ---------------------------------------------------------------------------


def _codes(packed: torch.Tensor) -> torch.Tensor:
    """Unpacked E2M1 codes, the even element from the low nibble."""
    return torch.stack((packed & 0x0F, packed >> 4), dim=-1).flatten(-2)


def test_mxfp4_quantization_matches_the_quark_reference():
    torch.manual_seed(0)
    weight = torch.randn(64, 512) * torch.logspace(-6, 2, 64)[:, None]
    weight[0, :32] = 0  # an all-zero group
    weight = weight.bfloat16()
    packed, scales = quantize_bf16_to_mxfp4(weight)
    assert packed.dtype == scales.dtype == torch.uint8
    assert packed.shape == (64, 256) and scales.shape == (64, 16)
    torch.testing.assert_close(
        dequantize_mxfp4_to_bf16(packed, scales),
        fake_quantize_mxfp4_even(weight).bfloat16(),
        rtol=0,
        atol=0,
    )


def test_mxfp4_quantization_rounds_scales_and_ties_like_quark():
    group = torch.zeros(32)
    # amax 6 = 1.5 * 2**2: scale 2**0, so these values keep their size.
    group[:8] = torch.tensor([6.0, -6.0, 2.5, 3.5, 5.0, 0.25, 0.75, -0.1])
    # amax 7 = 1.75 * 2**2 rounds up to 2**3: scale 2**1, and 7 / 2 = 3.5
    # ties to the even code (4), so 7 quantizes to 8.
    second = torch.zeros(32)
    second[0], second[1] = 7.0, 1.0
    packed, scales = quantize_bf16_to_mxfp4(torch.stack((group, second)).bfloat16())
    assert scales[:, 0].tolist() == [127, 128]
    codes = _codes(packed)
    # 6 -> code 7, -6 -> sign | 7, 2.5 -> 2 (code 4, even), 3.5 -> 4 (code 6,
    # even), 5 -> 4 (code 6, even), 0.25 -> 0 (code 0, even), 0.75 -> 1
    # (code 2, even), -0.1 -> -0 (the sign bit kept).
    assert codes[0, :8].tolist() == [7, 15, 4, 6, 6, 0, 2, 8]
    assert codes[1, :2].tolist() == [6, 1]
    values = dequantize_mxfp4_to_bf16(packed, scales).float()
    assert values[1, 0].item() == 8.0 and values[1, 1].item() == 1.0


def test_mxfp4_quantization_reproduces_serialized_weights():
    # A Quark checkpoint's experts, dequantized, quantize back to the values
    # they came from.
    torch.manual_seed(1)
    packed = torch.randint(0, 256, (32, 64), dtype=torch.uint8)
    scales = torch.randint(110, 135, (32, 4), dtype=torch.uint8)
    weight = dequantize_mxfp4_to_bf16(packed, scales)
    again = dequantize_mxfp4_to_bf16(*quantize_bf16_to_mxfp4(weight))
    assert torch.equal(again, weight)


def test_mxfp4_quantization_rejects_unsupported_weights():
    with pytest.raises(ValueError, match="divisible by 32"):
        quantize_bf16_to_mxfp4(torch.zeros(4, 48, dtype=torch.bfloat16))
    with pytest.raises(TypeError, match="16-bit"):
        quantize_bf16_to_mxfp4(torch.zeros(4, 64))


# ---------------------------------------------------------------------------
# MXFP4 draft experts: the config and the checkpoint stream
# ---------------------------------------------------------------------------


def _quark_target_config() -> Mxfp4Config:
    """The target's Quark W4A4 config, its exclusions naming the MTP layer's
    linear modules only, as the Qwen3.8 checkpoint lists them."""
    return Mxfp4Config(
        ignored_layers=[
            "lm_head",
            "mtp.fc_hidden",
            "mtp.layers.0.self_attn.q_proj",
            "mtp.layers.0.self_attn.k_proj",
            "mtp.layers.0.self_attn.v_proj",
            "mtp.layers.0.self_attn.o_proj",
            "mtp.layers.0.mlp.shared_expert.gate_proj",
            "mtp.layers.0.mlp.shared_expert.up_proj",
            "mtp.layers.0.mlp.shared_expert.down_proj",
        ],
        is_checkpoint_mxfp4_serialized=True,
        use_dynamic_mxfp4_activations=True,
        quant_method="quark",
    )


def test_mxfp4_draft_config_quantizes_only_the_routed_experts():
    target = _quark_target_config()
    assert qwen4_exp_nextn._draft_quant_config(target, "none") is None

    draft = qwen4_exp_nextn._draft_quant_config(target, "mxfp4")
    assert moe_expert_weight_dtype(draft, "mtp.layers.0.mlp") == "mxfp4"
    assert draft.use_dynamic_mxfp4_activations
    # fc_embedding is BF16 in the checkpoint but missing from Quark's list.
    for prefix in (
        "mtp.fc_embedding",
        "mtp.fc_hidden",
        "mtp.layers.0.self_attn.qkv_proj",
        "mtp.layers.0.mlp.shared_expert.gate_up_proj",
        "mtp.layers.0.mlp.shared_expert.down_proj",
    ):
        linear = ReplicatedLinear(64, 64, bias=False, quant_config=draft, prefix=prefix)
        assert isinstance(linear.quant_method, UnquantizedLinearMethod), prefix


def test_mxfp4_draft_config_needs_a_quark_target_with_a_bf16_mtp_layer():
    with pytest.raises(ValueError, match="AMD Quark MXFP4"):
        qwen4_exp_nextn._draft_quant_config(None, "mxfp4")
    # An export whose MTP experts are already MXFP4.
    packed = Mxfp4Config(quant_method="mxfp4", is_checkpoint_mxfp4_serialized=True)
    with pytest.raises(ValueError, match="AMD Quark MXFP4"):
        qwen4_exp_nextn._draft_quant_config(packed, "mxfp4")


def test_mtp_expert_stream_serializes_bf16_experts_as_mxfp4():
    torch.manual_seed(2)
    expert = torch.randn(64, 128).bfloat16()
    other = torch.randn(64, 128).bfloat16()
    stream = [
        ("mtp.layers.0.mlp.experts.3.down_proj.weight", expert),
        ("mtp.layers.0.mlp.shared_expert.down_proj.weight", other),
        ("model.language_model.layers.3.mlp.experts.3.down_proj.weight", other),
    ]
    out = list(qwen4_exp_nextn.quantize_mtp_experts_to_mxfp4(stream, "cpu"))
    names = [name for name, _ in out]
    assert names == [
        "mtp.layers.0.mlp.experts.3.down_proj.weight",
        "mtp.layers.0.mlp.experts.3.down_proj.weight_scale",
        "mtp.layers.0.mlp.shared_expert.down_proj.weight",
        "model.language_model.layers.3.mlp.experts.3.down_proj.weight",
    ]
    torch.testing.assert_close(
        dequantize_mxfp4_to_bf16(out[0][1], out[1][1]),
        fake_quantize_mxfp4_even(expert).bfloat16(),
        rtol=0,
        atol=0,
    )
    assert out[2][1] is other and out[3][1] is other

    packed = [("mtp.layers.0.mlp.experts.0.gate_proj.weight", out[0][1])]
    with pytest.raises(ValueError, match="BF16 checkpoint"):
        list(qwen4_exp_nextn.quantize_mtp_experts_to_mxfp4(packed, "cpu"))


# ---------------------------------------------------------------------------
# FP8 draft LM head
# ---------------------------------------------------------------------------


class _EmbeddingOnlyDraft(torch.nn.Module):
    """Stand-in for the draft's GPU transformer; the heads and logits
    processor are real."""

    def __init__(self, config, mapping, quant_config, prefix):
        super().__init__()
        self.qsa_indexers = []
        self.embed_tokens = torch.nn.Embedding(config.vocab_size, config.hidden_size)


def _target_head(vocab: int, hidden: int, device: str) -> ParallelLMHead:
    head = ParallelLMHead(
        vocab,
        hidden,
        params_dtype=torch.bfloat16,
        tp_rank=0,
        tp_size=1,
        tp_group=(0,),
    ).to(device)
    with torch.no_grad():
        head.weight.copy_(torch.randn_like(head.weight) * 0.05)
    return head


def test_fp8_head_copy_keeps_the_source_and_its_geometry():
    torch.manual_seed(3)
    head = _target_head(320, 64, "cpu")
    source = head.weight.detach().clone()
    copy = Fp8WeightOnlyLMHead(head)
    assert copy.weight.dtype == torch.float8_e4m3fn
    assert copy.weight.shape == head.weight.shape
    assert isinstance(copy.quant_method, Fp8WeightOnlyLinearMethod)
    assert copy.num_embeddings_per_partition == head.num_embeddings_per_partition
    assert copy.shard_indices is head.shard_indices
    # Rows dequantize to the source within E4M3's half-ULP (2**-4 relative).
    dequantized = copy.weight.float() * copy.weight_scale[:, None]
    torch.testing.assert_close(dequantized, source.float(), rtol=2**-4, atol=1e-6)
    assert head.weight.dtype == torch.bfloat16 and torch.equal(head.weight, source)


def test_fp8_head_copy_refuses_a_head_it_cannot_copy():
    linear = ReplicatedLinear(64, 320, bias=False, params_dtype=torch.bfloat16)
    with pytest.raises(ValueError, match="vocabulary-parallel"):
        Fp8WeightOnlyLMHead(linear)
    head = _target_head(320, 64, "cpu").float()
    with pytest.raises(ValueError, match="16-bit"):
        Fp8WeightOnlyLMHead(head)


@_requires_cuda
def test_draft_projects_through_the_fp8_copy_and_the_target_head_is_unchanged(
    monkeypatch,
):
    monkeypatch.setattr(qwen4_exp_nextn, "Qwen4ExpDraftModel", _EmbeddingOnlyDraft)
    torch.manual_seed(4)
    vocab, hidden = 1024, 256
    config = PretrainedConfig(
        hc_count=4,
        hidden_size=hidden,
        vocab_size=vocab,
        rms_norm_eps=1e-6,
        num_hidden_layers=1,
        tie_word_embeddings=False,
        draft_moe_quantization="none",
    )
    draft = qwen4_exp_nextn.Qwen4ExpForCausalLMNextN(
        config, Mapping(rank=0, world_size=1)
    ).cuda()
    target = _target_head(vocab, hidden, "cuda")
    source = target.weight.detach().clone()
    embed = torch.nn.Embedding(vocab, hidden, dtype=torch.bfloat16).cuda().weight
    draft.set_embed_and_head_module(embed, target)
    draft.quantize_lm_head("fp8_weight_only")
    assert isinstance(draft.lm_head, Fp8WeightOnlyLMHead)

    hidden_states = torch.randn(5, hidden, device="cuda", dtype=torch.bfloat16)
    metadata = LogitsMetadata(forward_mode=ForwardMode.DECODE, query_shard=None)
    logits = draft.logits_processor._get_logits(
        hidden_states, draft.lm_head, metadata, require_full_vocab=False
    )
    head = draft.lm_head
    exact = (
        hidden_states.double()
        @ (head.weight.double() * head.weight_scale.double()[:, None]).T
    )
    torch.testing.assert_close(logits, exact.bfloat16(), rtol=2**-7, atol=1e-3)
    # The target's verify keeps its BF16 head, untouched.
    assert target.weight.dtype == torch.bfloat16 and torch.equal(target.weight, source)
    with pytest.raises(ValueError, match="unknown draft LM head quantization"):
        draft.quantize_lm_head("fp4")


# ---------------------------------------------------------------------------
# Server arguments
# ---------------------------------------------------------------------------


def test_draft_quantization_serves_only_qwen38_mtp_drafts():
    for arch in ("Qwen4ExpForCausalLM", "Qwen4ExpForConditionalGeneration"):
        assert draft_quantization_refusal([arch], "MTP") is None
    assert "MTP drafts of" in draft_quantization_refusal(
        ["Qwen4ExpForConditionalGeneration"], "EAGLE3"
    )
    assert "MTP drafts of" in draft_quantization_refusal(["Qwen3ForCausalLM"], "MTP")


@pytest.mark.parametrize(
    "flag,field,value",
    [
        (
            "--speculative-draft-lm-head-quantization",
            "speculative_draft_lm_head_quantization",
            "fp8_weight_only",
        ),
        (
            "--speculative-draft-moe-quantization",
            "speculative_draft_moe_quantization",
            "mxfp4",
        ),
    ],
)
def test_draft_quantization_flags_resolve_and_refuse(flag, field, value):
    assert getattr(ServerArgs(model="test-model"), field) == "none"
    assert getattr(ServerArgs(model="test-model", **{field: value}), field) == value
    with pytest.raises(ValueError, match=f"{flag} must be one of"):
        ServerArgs(model="test-model", **{field: "int4"})
    with pytest.raises(ValueError, match=f"{flag} {value} cannot run with"):
        ServerArgs(
            model="test-model",
            model_update_config="{}",
            model_update_sdk_module="test_sdk",
            model_update_engine_type="fluent_llm",
            model_update_draft_weights="retain",
            **{field: value},
        )


def test_mxfp4_draft_experts_refuse_the_decode_monokernel():
    with pytest.raises(ValueError, match="cannot run with --decode-monokernel"):
        ServerArgs(
            model="test-model",
            speculative_draft_moe_quantization="mxfp4",
            decode_monokernel=True,
        )
