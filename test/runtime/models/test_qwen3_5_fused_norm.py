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

"""GemmaRMSNorm's quantized copies, where Qwen3.5 layers fuse each add + norm, and how the next projection takes the copy."""

import os
import sys
import weakref
from collections import Counter
from types import SimpleNamespace

import pytest
import torch

sys.path.insert(
    0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
)
from ci_system.ci_register import register_cuda_ci
from tokenspeed_kernel.ops.gemm.fp8_utils import static_quant_fp8
from tokenspeed_kernel.ops.quantization.flashinfer import fp4_quantize
from tokenspeed_kernel.platform import current_platform

from tokenspeed.runtime.configs.qwen3_5_config import Qwen3_5TextConfig
from tokenspeed.runtime.distributed.mapping import Mapping
from tokenspeed.runtime.layers import layernorm
from tokenspeed.runtime.layers.dense import fp8 as dense_fp8
from tokenspeed.runtime.layers.dense import nvfp4 as dense_nvfp4
from tokenspeed.runtime.layers.dense.fp8 import Fp8LinearMethod
from tokenspeed.runtime.layers.dense.nvfp4 import Nvfp4LinearMethod
from tokenspeed.runtime.layers.layernorm import GemmaRMSNorm
from tokenspeed.runtime.layers.quantization.modelopt_mixed import ModelOptMixedConfig
from tokenspeed.runtime.models import qwen3_5_moe
from tokenspeed.runtime.models.qwen3_5 import (
    Qwen3_5AttentionDecoderLayer,
    Qwen3_5LinearDecoderLayer,
    _input_norm,
    _post_attn_norm,
)
from tokenspeed.runtime.models.qwen3_5_nextn import Qwen3_5DraftAttentionDecoderLayer
from tokenspeed.runtime.utils.env import global_server_args_dict

register_cuda_ci(
    est_time=45,
    suite="runtime-1gpu",
    disabled_on_runners=["amd-*"],
    disabled_on_runners_reason="the quantized copies are produced on NVIDIA only",
)

_IS_BLACKWELL = current_platform().is_blackwell
_HIDDEN = 5120


def _norm_and_rows(rows: int, hidden: int, dtype: torch.dtype):
    torch.manual_seed(rows + hidden)
    norm = GemmaRMSNorm(hidden, eps=1e-6).to(device="cuda", dtype=dtype)
    norm.weight.data.copy_(torch.randn(hidden) * 0.3)
    x = torch.randn(rows, hidden, device="cuda", dtype=dtype) * 4
    return norm, x, torch.randn_like(x) * 30


def _add_norm_against_forward(norm, x, residual, add_norm, scale):
    expected, expected_residual = norm(x.clone(), residual.clone())
    normed, copy, new_residual = add_norm(x.clone(), residual.clone(), scale)
    assert torch.equal(new_residual, expected_residual)
    # Same FP32 math; the sum of squares may add in another order, so within one ulp.
    torch.testing.assert_close(normed, expected, atol=0, rtol=2**-7)
    assert (normed != expected).sum().item() <= normed.numel() // 1000
    return normed, copy


@pytest.mark.parametrize("rows", [0, 1, 130])
@pytest.mark.parametrize("hidden", [2560, 5120])
def test_add_norm_with_fp8_matches_forward(rows: int, hidden: int) -> None:
    norm, x, residual = _norm_and_rows(rows, hidden, torch.bfloat16)
    scale = torch.tensor([0.02], device="cuda")

    normed, normed_fp8 = _add_norm_against_forward(
        norm, x, residual, norm.add_norm_with_fp8, scale
    )

    quantized, _ = static_quant_fp8(normed, scale)
    assert torch.equal(normed_fp8.view(torch.uint8), quantized.view(torch.uint8))


@pytest.mark.skipif(not _IS_BLACKWELL, reason="the NVFP4 copy is made on Blackwell")
@pytest.mark.parametrize("rows", [1, 130])
def test_add_norm_with_fp4_matches_forward(rows: int) -> None:
    norm, x, residual = _norm_and_rows(rows, 5120, torch.bfloat16)
    scale = torch.tensor([7.5], device="cuda")

    normed, normed_fp4 = _add_norm_against_forward(
        norm, x, residual, norm.add_norm_with_fp4, scale
    )

    values, scales = fp4_quantize(normed, scale)
    assert torch.equal(normed_fp4[0], values.view(torch.uint8))
    assert torch.equal(normed_fp4[1], scales.view(torch.uint8))


@pytest.mark.parametrize(
    "case",
    [
        "fp16",
        "width",
        "amd",
        "FLASHINFER_DISABLE_FP4_QUANT_FAST_MATH",
        "TRTLLM_DISABLE_FP4_QUANT_FAST_MATH",
        "FLASHINFER_NVFP4_4OVER6",
    ],
)
def test_add_norm_with_fp4_skips_copies_fp4_quantize_would_not_make(
    monkeypatch, case: str
) -> None:
    # FP16 rows (the NVFP4 linear would return BF16), a width without whole chunks, AMD, other recipes.
    if case.isupper():
        monkeypatch.setenv(case, "1")
    if case == "amd":
        monkeypatch.setattr(layernorm, "_is_amd", True)
    dtype = torch.float16 if case == "fp16" else torch.bfloat16
    norm, x, residual = _norm_and_rows(14, 2560 if case == "width" else 5120, dtype)

    _, normed_fp4 = _add_norm_against_forward(
        norm, x, residual, norm.add_norm_with_fp4, torch.tensor([7.5], device="cuda")
    )

    assert normed_fp4 is None


@pytest.mark.parametrize("attn_tp", [False, True])
@pytest.mark.parametrize("unfused", [False, True])
@pytest.mark.parametrize("dense_tp", [False, True])
@pytest.mark.parametrize("first_layer", [False, True])
@pytest.mark.parametrize("with_scales", [False, True])
def test_layer_norms_pick_the_fused_copy(
    attn_tp: bool, unfused: bool, dense_tp: bool, first_layer: bool, with_scales: bool
) -> None:
    calls = []

    def comm_norm(name):
        def run(hidden_states, residual, *ctx):
            calls.append(name)
            return hidden_states, hidden_states if residual is None else residual

        return run

    comm_manager = SimpleNamespace(
        mapping=SimpleNamespace(
            has_attn_tp=attn_tp, dense=SimpleNamespace(has_tp=dense_tp)
        ),
        layer_boundary_norm="unfused" if unfused else "fused",
        input_reduce_norm=comm_norm("input"),
        post_attn_reduce_norm=comm_norm("post_attn"),
    )
    norm = GemmaRMSNorm(_HIDDEN).to(device="cuda", dtype=torch.bfloat16)
    hidden_states = torch.randn(129, _HIDDEN, device="cuda", dtype=torch.bfloat16)
    residual = None if first_layer else torch.randn_like(hidden_states)
    fp8_scale = torch.tensor([0.02], device="cuda") if with_scales else None
    fp4_scale = torch.tensor([7.5], device="cuda") if with_scales else None

    hidden_states, hidden_fp8, residual = _input_norm(
        comm_manager, norm, hidden_states, residual, fp8_scale
    )
    hidden_states, hidden_fp4, residual = _post_attn_norm(
        comm_manager, norm, hidden_states, residual, fp4_scale, None
    )

    fused_input = not (first_layer or attn_tp or unfused)
    assert calls == ([] if fused_input else ["input"]) + (
        ["post_attn"] if attn_tp else []
    )
    assert (hidden_fp8 is not None) == (fused_input and with_scales)
    copy_fp4 = not (attn_tp or dense_tp) and with_scales and _IS_BLACKWELL
    assert (hidden_fp4 is not None) == copy_fp4


class _StubAttention(torch.nn.Module):
    """A deterministic attention core reading q: the projections around it are under test."""

    def forward(self, q, k, v, positions, ctx, **kwargs):
        return torch.tanh(q.reshape(q.shape[0], -1))

    def attend_live_rows(self, q, k, v, positions, ctx):
        return self.forward(q[ctx.gather_ids], k, v, positions, ctx)


class _StubGDNBackend:
    """A deterministic linear-attention core reading the projected values."""

    def forward(self, q, k, v, layer, token_to_kv_pool, forward_mode, bs, **kwargs):
        values = kwargs["mixed_qkv"][:, -kwargs["value_dim"] :]
        return torch.tanh(values).unflatten(1, (-1, kwargs["head_v_dim"]))


@pytest.fixture
def bf16_default_dtype():
    previous = torch.get_default_dtype()
    torch.set_default_dtype(torch.bfloat16)
    yield
    torch.set_default_dtype(previous)


def _decoder_layer(kind: str, split_gdn: bool):
    config = Qwen3_5TextConfig(
        hidden_size=_HIDDEN,
        intermediate_size=1024,
        num_hidden_layers=2,
        num_attention_heads=8,
        num_key_value_heads=2,
        head_dim=128,
        linear_num_key_heads=4,
        linear_num_value_heads=8,
        linear_key_head_dim=128,
        linear_value_head_dim=128,
        layer_types=["linear_attention", "full_attention"],
    )
    config.dtype = torch.bfloat16
    fp8 = ["self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj"]
    fp8 += ["self_attn.o_proj", "linear_attn.out_proj"]
    fp8 += ["linear_attn.in_proj_qkv", "linear_attn.in_proj_z"]
    if not split_gdn:
        fp8 += ["linear_attn.in_proj_b", "linear_attn.in_proj_a"]
    quantized = {f"model.layers.1.{leaf}": "FP8" for leaf in fp8}
    for leaf in ("gate_proj", "up_proj", "down_proj"):
        quantized[f"model.layers.1.mlp.{leaf}"] = "NVFP4"
    layer_cls = {
        "linear": Qwen3_5LinearDecoderLayer,
        "attention": Qwen3_5AttentionDecoderLayer,
        "draft": Qwen3_5DraftAttentionDecoderLayer,
    }[kind]
    attn = "linear_attn" if kind == "linear" else "self_attn"
    layer = layer_cls(
        config,
        Mapping(rank=0, world_size=1),
        1,
        quant_config=ModelOptMixedConfig(quantized_layers=quantized),
        prefix=f"model.layers.1.{attn}",
    ).cuda()

    generator = torch.Generator().manual_seed(0)

    def randn(shape):
        return torch.randn(shape, generator=generator, dtype=torch.float32)

    for norm in (layer.input_layernorm, layer.post_attention_layernorm):
        norm.weight.data.copy_(randn(_HIDDEN) * 0.3)
    # Distinct input scales, so a copy made with another projection's scale shows.
    projections = [(layer.mlp.gate_up_proj, 0.05), (layer.mlp.down_proj, 0.07)]
    if kind == "linear":
        gdn = layer.linear_attn
        assert gdn._split_in_proj == split_gdn
        in_proj = gdn.in_proj_qkvz if split_gdn else gdn.in_proj_qkvzba
        projections += [(in_proj, 0.02), (gdn.out_proj, 0.03)]
        if split_gdn:
            gdn.in_proj_ba.weight.data.copy_(randn(gdn.in_proj_ba.weight.shape) * 0.02)
    else:
        projections += [(layer.qkv_proj, 0.02), (layer.o_proj, 0.03)]
        layer.attn = _StubAttention()
    for proj, input_scale in projections:
        if proj.weight.dtype == torch.uint8:
            proj.weight.data.copy_(
                torch.randint(0, 256, proj.weight.shape, generator=generator)
            )
            proj.weight_scale.data.copy_(randn(proj.weight_scale.shape) * 2 + 0.25)
            proj.weight_scale_2.data.fill_(0.002)
        else:
            proj.weight.data.copy_(randn(proj.weight.shape) * 0.5)
            proj.weight_scale.data.fill_(0.004)
        proj.input_scale.data.fill_(input_scale)
        proj.quant_method.process_weights_after_loading(proj)
    return layer


@pytest.mark.skipif(not _IS_BLACKWELL, reason="NVFP4 GEMMs need Blackwell")
@pytest.mark.parametrize(
    "kind,split_gdn,narrow,fused_swiglu",
    [
        ("attention", False, False, True),
        ("attention", False, False, False),
        ("linear", False, False, True),
        ("linear", True, False, True),
        ("draft", False, False, True),
        ("draft", False, True, True),
    ],
)
@torch.no_grad()
def test_decoder_layer_hands_each_projection_its_own_quant(
    monkeypatch, bf16_default_dtype, kind, split_gdn, narrow, fused_swiglu
) -> None:
    monkeypatch.setitem(global_server_args_dict, "layer_boundary_norm", "fused")
    monkeypatch.setenv(
        "TOKENSPEED_NVFP4_GEMM_SWIGLU_NVFP4_QUANT", str(int(fused_swiglu))
    )
    layer = _decoder_layer(kind, split_gdn)

    # Record each norm's copies, whether an FP8 copy outlives its consumer, and the projections' own quants.
    copies, fp8_copies, fp8_copy_alive, launches = [], [], [], Counter()
    add_rmsnorm = layernorm.add_rmsnorm

    def recording_add_rmsnorm(*args, **kwargs):
        copies.append((kwargs["out_fp8"] is not None, kwargs["out_fp4"] is not None))
        fp8_copy_alive.append(any(copy() is not None for copy in fp8_copies))
        if kwargs["out_fp8"] is not None:
            fp8_copies.append(weakref.ref(kwargs["out_fp8"]))
        return add_rmsnorm(*args, **kwargs)

    def counting(name, quantize):
        def run(*args, **kwargs):
            launches[name] += 1
            return quantize(*args, **kwargs)

        return run

    monkeypatch.setattr(layernorm, "add_rmsnorm", recording_add_rmsnorm)
    fp4_quantize = counting("nvfp4", dense_nvfp4.fp4_quantize)
    monkeypatch.setattr(dense_nvfp4, "fp4_quantize", fp4_quantize)
    monkeypatch.setattr(qwen3_5_moe, "fp4_quantize", fp4_quantize)
    monkeypatch.setattr(
        dense_fp8, "static_quant_fp8", counting("fp8", dense_fp8.static_quant_fp8)
    )

    torch.manual_seed(1)
    hidden_states = torch.randn(129, _HIDDEN, device="cuda") * 4
    # The single-layer MTP draft opens without a residual and may narrow to live rows.
    residual = None if kind == "draft" else torch.randn_like(hidden_states) * 30
    live_rows = torch.arange(0, 129, 3, device="cuda") if narrow else None
    ctx = SimpleNamespace(
        draft_narrowing=None if live_rows is None else object(),
        gather_ids=live_rows,
        query_shard=None,
        collective_global_num_tokens=None,
        global_num_tokens=None,
        collective_num_tokens=None,
        input_num_tokens=129,
        forward_mode=SimpleNamespace(is_idle=lambda: False),
        attn_backend=_StubGDNBackend(),
        token_to_kv_pool=None,
        bs=129,
    )
    positions = {} if kind == "linear" else {"positions": torch.arange(129).cuda()}

    def run():
        launches.clear()
        out = layer(
            hidden_states=hidden_states.clone(),
            residual=None if residual is None else residual.clone(),
            ctx=ctx,
            **positions,
        )
        return out, Counter(launches)

    (out, new_residual), fused_launches = run()
    input_copy = kind != "draft"
    assert copies == ([(True, False)] if input_copy else []) + [(False, True)]
    assert not any(fp8_copy_alive)

    # Without the hooks every projection quantizes the same normed rows itself.
    monkeypatch.setattr(Fp8LinearMethod, "static_fp8_input_scale", lambda *_: None)
    monkeypatch.setattr(Nvfp4LinearMethod, "nvfp4_global_scale", lambda *_: None)
    (expected, expected_residual), own_launches = run()
    assert torch.equal(new_residual, expected_residual)
    assert torch.equal(out, expected)
    assert own_launches["fp8"] - fused_launches["fp8"] == int(input_copy)
    assert own_launches["nvfp4"] - fused_launches["nvfp4"] == 1
