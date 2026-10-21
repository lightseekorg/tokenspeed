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

"""Dynamic-activation MXFP4 dense linears: load, prepare, fused SwiGLU MLP."""

from __future__ import annotations

import pytest
import torch
from tokenspeed_kernel.platform import current_platform

if not current_platform().is_amd:
    pytest.skip("Dynamic MXFP4 activations are AMD-only", allow_module_level=True)

from tokenspeed_kernel.ops.activation.triton import silu_and_mul  # noqa: E402
from tokenspeed_kernel.ops.gemm.triton import triton_mm_mxfp4  # noqa: E402
from tokenspeed_kernel.ops.quantization.triton import mxfp4_quantize  # noqa: E402

from tokenspeed.runtime.layers.dense.mxfp4 import (  # noqa: E402
    Mxfp4LinearMethod,
    quantize_lm_head_mxfp4,
)
from tokenspeed.runtime.layers.linear import ReplicatedLinear  # noqa: E402
from tokenspeed.runtime.layers.logits_processor import (  # noqa: E402
    should_apply_lm_head_quant_method,
)
from tokenspeed.runtime.layers.quantization.mxfp4 import Mxfp4Config  # noqa: E402

_HIDDEN = 2560
_INTERMEDIATE = 320


def _linear(input_size: int, output_size: int, prefix: str) -> ReplicatedLinear:
    config = Mxfp4Config(
        is_checkpoint_mxfp4_serialized=True,
        use_dynamic_mxfp4_activations=True,
    )
    with torch.device("cuda"):
        layer = ReplicatedLinear(
            input_size,
            output_size,
            bias=False,
            params_dtype=torch.bfloat16,
            quant_config=config,
            prefix=prefix,
        )
    assert isinstance(layer.quant_method, Mxfp4LinearMethod)
    return layer


def _load(layer: ReplicatedLinear) -> tuple[torch.Tensor, torch.Tensor]:
    n, k_packed = layer.weight.shape
    weight = torch.randint(0, 256, (n, k_packed), device="cuda", dtype=torch.uint8)
    scale = torch.randint(
        118, 124, tuple(layer.weight_scale.shape), device="cuda", dtype=torch.uint8
    )
    layer.weight_loader(layer.weight, weight)
    layer.weight_loader(layer.weight_scale, scale)
    return weight, scale


def _unfused_linear(x, weight, scale) -> torch.Tensor:
    x_fp4, x_scale = mxfp4_quantize(x)
    return triton_mm_mxfp4(x_fp4, weight, x_scale, scale, torch.bfloat16)


@pytest.mark.parametrize("tokens", [3, 64, 700])
def test_shared_expert_mlp_matches_unfused_path(tokens: int) -> None:
    torch.manual_seed(0)
    gate_up = _linear(_HIDDEN, 2 * _INTERMEDIATE, "mlp.shared_expert.gate_up_proj")
    down = _linear(_INTERMEDIATE, _HIDDEN, "mlp.shared_expert.down_proj")
    gate_up.quant_method.enable_swiglu(gate_up)
    gate_up_weight, gate_up_scale = _load(gate_up)
    down_weight, down_scale = _load(down)
    for layer in (gate_up, down):
        layer.quant_method.process_weights_after_loading(layer)

    x = torch.randn((tokens, _HIDDEN), device="cuda", dtype=torch.bfloat16)
    act = gate_up.quant_method.apply_swiglu(gate_up, x)
    out, _ = down(act)

    gate_up_ref = _unfused_linear(x, gate_up_weight, gate_up_scale)
    act_ref = silu_and_mul(
        gate_up_ref,
        torch.empty((tokens, _INTERMEDIATE), device="cuda", dtype=torch.bfloat16),
    )
    out_ref = _unfused_linear(act_ref, down_weight, down_scale)
    # Bit-identical activation quantization; FP32 summation order may differ.
    torch.testing.assert_close(act.float(), act_ref.float(), atol=1e-3, rtol=2**-8)
    torch.testing.assert_close(out.float(), out_ref.float(), atol=1e-3, rtol=2**-8)

    # Plain projection through the same prepared gate_up weight.
    gate_up_out, _ = gate_up(x)
    torch.testing.assert_close(
        gate_up_out.float(), gate_up_ref.float(), atol=1e-3, rtol=2**-8
    )


def test_enable_swiglu_after_processing_is_rejected() -> None:
    layer = _linear(_HIDDEN, 2 * _INTERMEDIATE, "mlp.gate_up_proj")
    _load(layer)
    layer.quant_method.process_weights_after_loading(layer)
    with pytest.raises(RuntimeError, match="precede weight processing"):
        layer.quant_method.enable_swiglu(layer)


def test_qwen_shared_expert_mlp_fuses_swiglu() -> None:
    from tokenspeed.runtime.distributed.mapping import Mapping
    from tokenspeed.runtime.models.qwen3_5_moe import Qwen3_5MoeMLP

    torch.manual_seed(0)
    config = Mxfp4Config(
        is_checkpoint_mxfp4_serialized=True,
        use_dynamic_mxfp4_activations=True,
    )
    with torch.device("cuda"):
        mlp = Qwen3_5MoeMLP(
            hidden_size=_HIDDEN,
            intermediate_size=_INTERMEDIATE,
            hidden_act="silu",
            mapping=Mapping(rank=0, world_size=1),
            quant_config=config,
            reduce_results=False,
            prefix="model.layers.0.mlp.shared_expert",
            parallelism="moe_shared",
        )
    assert mlp.gate_up_proj.mxfp4_swiglu and not mlp.down_proj.mxfp4_swiglu
    gate_up_weight, gate_up_scale = _load(mlp.gate_up_proj)
    down_weight, down_scale = _load(mlp.down_proj)
    for layer in (mlp.gate_up_proj, mlp.down_proj):
        layer.quant_method.process_weights_after_loading(layer)

    x = torch.randn((40, _HIDDEN), device="cuda", dtype=torch.bfloat16)
    act_ref = silu_and_mul(
        _unfused_linear(x, gate_up_weight, gate_up_scale),
        torch.empty((40, _INTERMEDIATE), device="cuda", dtype=torch.bfloat16),
    )
    torch.testing.assert_close(
        mlp(x).float(),
        _unfused_linear(act_ref, down_weight, down_scale).float(),
        atol=1e-3,
        rtol=2**-8,
    )


_E2M1 = (0, 0.5, 1, 1.5, 2, 3, 4, 6, 0, -0.5, -1, -1.5, -2, -3, -4, -6)


def _dequant_weight(weight: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    table = torch.tensor(_E2M1, device=weight.device, dtype=torch.float32)
    codes = torch.stack((weight & 0xF, weight >> 4), dim=-1).flatten(-2)
    return table[codes.long()] * torch.exp2(scale.float() - 127).repeat_interleave(
        32, dim=-1
    )


def _mxfp8_round_trip(x: torch.Tensor) -> torch.Tensor:
    groups = x.float().reshape(x.shape[0], -1, 32)
    ratio = groups.abs().amax(-1) * torch.tensor(1 / 448, dtype=torch.float32)
    bits = ratio.view(torch.int32)
    exponent = ((bits >> 23) & 0xFF) + ((bits & 0x7FFFFF) != 0).to(torch.int32)
    scale = (exponent.clamp(1, 254) << 23).view(torch.float32).unsqueeze(-1)
    return ((groups / scale).to(torch.float8_e4m3fn).float() * scale).reshape(x.shape)


def test_mxfp8_activation_checkpoint_runs_mxfp8_linears() -> None:
    # A Quark W4A8 MX checkpoint's dense linears quantize activations to
    # MXFP8 (E4M3, one E8M0 scale per 32 values) against MXFP4 weights.
    config = Mxfp4Config(
        is_checkpoint_mxfp4_serialized=True,
        use_dynamic_mxfp8_activations=True,
    )
    with torch.device("cuda"):
        layer = ReplicatedLinear(
            _HIDDEN,
            1280,
            bias=False,
            params_dtype=torch.bfloat16,
            quant_config=config,
            prefix="model.layers.0.linear_attn.out_proj",
        )
    assert isinstance(layer.quant_method, Mxfp4LinearMethod)
    assert layer.quant_method.activation_format == "mxfp8"
    weight, scale = _load(layer)
    layer.quant_method.process_weights_after_loading(layer)
    assert layer.mxfp4_linear_weight.activation_format == "mxfp8"

    torch.manual_seed(0)
    x = torch.randn((5, _HIDDEN), device="cuda", dtype=torch.bfloat16)
    out, _ = layer(x)
    expected = _mxfp8_round_trip(x) @ _dequant_weight(weight, scale).T
    torch.testing.assert_close(out.float(), expected, atol=1e-3, rtol=2**-8)


def test_linear_method_needs_dynamic_activations() -> None:
    # Weight-only MXFP4 configs keep dense linears unquantized; building the
    # MXFP4 linear method for them is an error, not a silent format choice.
    with pytest.raises(ValueError, match="MXFP4 or MXFP8"):
        Mxfp4LinearMethod(Mxfp4Config(is_checkpoint_mxfp4_serialized=True))


def test_lm_head_mxfp4_keeps_logits_and_quarters_weight_bytes() -> None:
    torch.manual_seed(0)
    vocab = 4096
    with torch.device("cuda"):
        head = ReplicatedLinear(
            _HIDDEN, vocab, bias=False, params_dtype=torch.bfloat16, prefix="lm_head"
        )
    head.weight.data.normal_(0.0, 0.02)
    x = torch.randn(4, _HIDDEN, device="cuda", dtype=torch.bfloat16)
    reference = (x.float() @ head.weight.float().T).float()
    bf16_bytes = head.weight.numel() * head.weight.element_size()

    quantize_lm_head_mxfp4(head, activation_format="mxfp8")

    assert should_apply_lm_head_quant_method(head, head.quant_method)
    assert head.weight.numel() * head.weight.element_size() == bf16_bytes // 4
    logits = head.quant_method.apply(head, x).float()
    assert logits.shape == (4, vocab)
    # MXFP4 weights and MXFP8 rows: about 10 % relative error per weight,
    # which averages out over the 2560-wide dot products.
    cos = torch.nn.functional.cosine_similarity(logits, reference, dim=-1)
    assert torch.all(cos > 0.99), cos


def _qwen4_exp_draft(head: torch.nn.Module):
    """A Qwen4-Exp MTP draft holding only what sharing the target's head touches."""
    from tokenspeed.runtime.models.qwen4_exp_nextn import Qwen4ExpForCausalLMNextN

    draft = Qwen4ExpForCausalLMNextN.__new__(Qwen4ExpForCausalLMNextN)
    torch.nn.Module.__init__(draft)
    draft.model = torch.nn.Module()
    draft.model.embed_tokens = torch.nn.Embedding(256, _HIDDEN, device="cuda")
    draft.lm_head = head
    return draft


def test_mtp_draft_adopts_the_quantized_target_head() -> None:
    from tokenspeed.runtime.layers.vocab_parallel_embedding import ParallelLMHead

    with torch.device("cuda"):
        target = ParallelLMHead(
            256,
            _HIDDEN,
            params_dtype=torch.bfloat16,
            tp_rank=0,
            tp_size=1,
            tp_group=(0,),
        )
        own = ParallelLMHead(
            256,
            _HIDDEN,
            params_dtype=torch.bfloat16,
            tp_rank=0,
            tp_size=1,
            tp_group=(0,),
        )
    quantize_lm_head_mxfp4(target, activation_format="mxfp8")
    draft = _qwen4_exp_draft(own)
    embed = torch.nn.Parameter(torch.ones(256, _HIDDEN, device="cuda"))
    draft.set_embed_and_head_module(embed, target)
    assert draft.model.embed_tokens.weight is embed
    assert draft.lm_head is target


def test_replicated_mtp_draft_head_shares_only_a_bf16_target_weight() -> None:
    with torch.device("cuda"):
        target, own = (
            ReplicatedLinear(
                _HIDDEN, 256, bias=False, params_dtype=torch.bfloat16, prefix=prefix
            )
            for prefix in ("lm_head", "mtp.lm_head")
        )
    draft = _qwen4_exp_draft(own)
    embed = torch.nn.Parameter(torch.ones(256, _HIDDEN, device="cuda"))
    draft.set_embed_and_head_module(embed, target)
    assert draft.lm_head is own
    assert draft.lm_head.weight is target.weight

    quantize_lm_head_mxfp4(target, activation_format="mxfp8")
    with pytest.raises(ValueError, match="BF16 LM head"):
        _qwen4_exp_draft(own).set_embed_and_head_module(embed, target)
