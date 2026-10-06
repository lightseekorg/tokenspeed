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

"""Qwen3.5 layers pick the fused norm only where it can hand its copy to the next projection."""

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
from tokenspeed_kernel.platform import current_platform

from tokenspeed.runtime.configs.qwen3_5_config import Qwen3_5TextConfig
from tokenspeed.runtime.distributed.mapping import Mapping
from tokenspeed.runtime.layers import layernorm
from tokenspeed.runtime.layers.dense import fp8 as dense_fp8
from tokenspeed.runtime.layers.dense import nvfp4 as dense_nvfp4
from tokenspeed.runtime.layers.dense.fp8 import Fp8LinearMethod
from tokenspeed.runtime.layers.dense.nvfp4 import Nvfp4LinearMethod
from tokenspeed.runtime.layers.layernorm import GemmaRMSNorm
from tokenspeed.runtime.layers.linear import ReplicatedLinear
from tokenspeed.runtime.layers.quantization.fp8 import Fp8Config
from tokenspeed.runtime.layers.quantization.modelopt_mixed import ModelOptMixedConfig
from tokenspeed.runtime.layers.quantization.nvfp4 import Nvfp4Config
from tokenspeed.runtime.models import qwen3_5_moe
from tokenspeed.runtime.models.qwen3_5 import (
    Qwen3_5AttentionDecoderLayer,
    Qwen3_5LinearDecoderLayer,
    _input_norm,
    _post_attn_norm,
)
from tokenspeed.runtime.models.qwen3_5_moe import Qwen3_5MoeMLP
from tokenspeed.runtime.models.qwen3_5_nextn import Qwen3_5DraftAttentionDecoderLayer
from tokenspeed.runtime.utils.env import global_server_args_dict

register_cuda_ci(
    est_time=30,
    suite="runtime-1gpu",
    disabled_on_runners=["amd-*"],
    disabled_on_runners_reason="the quantized copies are produced on NVIDIA only",
)

_IS_BLACKWELL = current_platform().is_blackwell
_HIDDEN = 5120


def _norm() -> GemmaRMSNorm:
    norm = GemmaRMSNorm(_HIDDEN, eps=1e-6).to(device="cuda", dtype=torch.bfloat16)
    norm.weight.data.normal_(0, 0.3)
    return norm


def _comm_manager(norm, calls, *, attn_tp, unfused, dense_tp):
    def input_reduce_norm(hidden_states, residual):
        calls.append("input_reduce_norm")
        if residual is None:
            return norm(hidden_states), hidden_states
        return norm(hidden_states, residual)

    def post_attn_reduce_norm(hidden_states, residual, ctx):
        calls.append("post_attn_reduce_norm")
        return norm(hidden_states, residual)

    return SimpleNamespace(
        mapping=SimpleNamespace(
            has_attn_tp=attn_tp, dense=SimpleNamespace(has_tp=dense_tp)
        ),
        layer_boundary_norm="unfused" if unfused else "fused",
        input_reduce_norm=input_reduce_norm,
        post_attn_reduce_norm=post_attn_reduce_norm,
    )


@pytest.mark.parametrize("attn_tp", [False, True])
@pytest.mark.parametrize("unfused", [False, True])
@pytest.mark.parametrize("dense_tp", [False, True])
@pytest.mark.parametrize("first_layer", [False, True])
@pytest.mark.parametrize("with_scales", [False, True])
def test_layer_norms_pick_the_fused_copy(
    attn_tp: bool, unfused: bool, dense_tp: bool, first_layer: bool, with_scales: bool
) -> None:
    torch.manual_seed(0)
    norm = _norm()
    calls = []
    comm_manager = _comm_manager(
        norm, calls, attn_tp=attn_tp, unfused=unfused, dense_tp=dense_tp
    )
    hidden_states = torch.randn(129, _HIDDEN, device="cuda", dtype=torch.bfloat16)
    residual = None if first_layer else torch.randn_like(hidden_states) * 3
    fp8_scale = torch.tensor([0.02], device="cuda") if with_scales else None
    fp4_scale = torch.tensor([7.5], device="cuda") if with_scales else None

    hidden_states, hidden_fp8, residual = _input_norm(
        comm_manager, norm, hidden_states, residual, fp8_scale
    )
    fused_input = not (first_layer or attn_tp or unfused)
    assert calls == ([] if fused_input else ["input_reduce_norm"])
    assert (hidden_fp8 is not None) == (fused_input and with_scales)

    calls.clear()
    hidden_states, hidden_fp4, residual = _post_attn_norm(
        comm_manager, norm, hidden_states, residual, fp4_scale, None
    )
    assert calls == (["post_attn_reduce_norm"] if attn_tp else [])
    copy_fp4 = not (attn_tp or dense_tp) and with_scales and _IS_BLACKWELL
    assert (hidden_fp4 is not None) == copy_fp4


@pytest.mark.parametrize("scheme", ["static", "dynamic", "block", "unquantized"])
def test_static_fp8_input_scale_hook(scheme: str) -> None:
    config = None
    if scheme != "unquantized":
        config = Fp8Config(
            is_checkpoint_fp8_serialized=True,
            activation_scheme="dynamic" if scheme == "block" else scheme,
            weight_block_size=[128, 128] if scheme == "block" else None,
        )
    layer = ReplicatedLinear(
        128,
        128,
        bias=False,
        params_dtype=torch.bfloat16,
        quant_config=config,
        prefix="model.proj",
    )

    scale = layer.quant_method.static_fp8_input_scale(layer)

    if scheme == "static":
        assert scale is layer.input_scale
    else:
        assert scale is None


def _count_quant_launches(monkeypatch) -> Counter:
    launches = Counter()
    static_quant_fp8 = dense_fp8.static_quant_fp8
    fp4_quantize = dense_nvfp4.fp4_quantize

    def counting_static_quant_fp8(*args, **kwargs):
        launches["fp8"] += 1
        return static_quant_fp8(*args, **kwargs)

    def counting_fp4_quantize(*args, **kwargs):
        launches["nvfp4"] += 1
        return fp4_quantize(*args, **kwargs)

    monkeypatch.setattr(dense_fp8, "static_quant_fp8", counting_static_quant_fp8)
    monkeypatch.setattr(dense_nvfp4, "fp4_quantize", counting_fp4_quantize)
    monkeypatch.setattr(qwen3_5_moe, "fp4_quantize", counting_fp4_quantize)
    return launches


def _fill_nvfp4(layer, generator: torch.Generator, input_scale: float) -> None:
    layer.weight.data.copy_(
        torch.randint(
            0, 256, layer.weight.shape, dtype=torch.uint8, generator=generator
        )
    )
    scales = torch.rand(layer.weight_scale.shape, generator=generator) * 2 + 0.25
    layer.weight_scale.data.copy_(scales.to(torch.float8_e4m3fn))
    layer.input_scale.data.fill_(input_scale)
    layer.weight_scale_2.data.fill_(0.002)
    layer.quant_method.process_weights_after_loading(layer)


@pytest.mark.skipif(not _IS_BLACKWELL, reason="NVFP4 GEMMs need Blackwell")
@pytest.mark.parametrize("fused_swiglu", ["0", "1"])
def test_mlp_takes_the_nvfp4_copy_as_its_own_quant(
    monkeypatch, fused_swiglu: str
) -> None:
    monkeypatch.setenv("TOKENSPEED_NVFP4_GEMM_SWIGLU_NVFP4_QUANT", fused_swiglu)
    mlp = Qwen3_5MoeMLP(
        hidden_size=_HIDDEN,
        intermediate_size=1024,
        hidden_act="silu",
        mapping=Mapping(rank=0, world_size=1),
        quant_config=Nvfp4Config(),
        reduce_results=False,
        parallelism="dense",
    ).cuda()
    assert mlp._use_nvfp4_gemm_swiglu_nvfp4_quant == (fused_swiglu == "1")
    generator = torch.Generator().manual_seed(0)
    _fill_nvfp4(mlp.gate_up_proj, generator, 0.05)
    _fill_nvfp4(mlp.down_proj, generator, 0.07)
    norm = _norm()
    launches = _count_quant_launches(monkeypatch)

    for rows in (0, 1, 129):
        torch.manual_seed(rows)
        x = torch.randn(rows, _HIDDEN, device="cuda", dtype=torch.bfloat16)
        residual = torch.randn_like(x) * 4
        normed, normed_fp4, _ = norm.add_norm_with_fp4(
            x, residual, mlp.input_fp4_scale()
        )

        assert normed_fp4 is not None
        launches.clear()
        prequantized = mlp.forward_prequantized(normed, normed_fp4)
        fused_launches = launches["nvfp4"]
        launches.clear()
        expected = mlp(normed)
        torch.testing.assert_close(prequantized, expected, atol=0, rtol=0)
        # The copy stands in for gate_up_proj's own input quant.
        assert launches["nvfp4"] - fused_launches == (1 if rows else 0)


class _StubAttention(torch.nn.Module):
    """A deterministic attention core: the projections around it are under test."""

    def __init__(self, heads: int, kv_heads: int, head_dim: int) -> None:
        super().__init__()
        self.heads = heads
        self.kv_heads = kv_heads
        self.head_dim = head_dim

    def forward(self, q, k, v, positions, ctx, **kwargs):
        rows = q.shape[0]
        q = q.reshape(rows, self.heads, self.head_dim).float()
        k = k.reshape(rows, self.kv_heads, self.head_dim).float()
        k = k.repeat_interleave(self.heads // self.kv_heads, dim=1)
        return torch.tanh(q * 0.05 + k * 0.03).reshape(rows, -1).to(v.dtype)

    def attend_live_rows(self, q, k, v, positions, ctx):
        rows = ctx.gather_ids
        return self.forward(q[rows], k[rows], v[rows], positions[rows], ctx)


class _StubGDNBackend:
    """A deterministic linear-attention core reading the projected q, k, v and z."""

    def forward(self, q, k, v, layer, token_to_kv_pool, forward_mode, bs, **kwargs):
        mixed = kwargs["mixed_qkv"].float()
        key_dim, value_dim = kwargs["key_dim"], kwargs["value_dim"]
        values = mixed[:, 2 * key_dim : 2 * key_dim + value_dim]
        keys = mixed[:, :key_dim].repeat(1, value_dim // key_dim)
        out = torch.tanh(values * 0.05 + keys * 0.02)
        return out.reshape(mixed.shape[0], -1, kwargs["head_v_dim"]).to(
            kwargs["z"].dtype
        )


def _fill_fp8(layer, generator: torch.Generator, input_scale: float) -> None:
    weight = torch.randn(layer.weight.shape, generator=generator, dtype=torch.float32)
    layer.weight.data.copy_((weight * 0.5).to(torch.float8_e4m3fn))
    layer.weight_scale.data.fill_(0.004)
    layer.input_scale.data.fill_(input_scale)
    layer.quant_method.process_weights_after_loading(layer)


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
        rms_norm_eps=1e-6,
        layer_types=["linear_attention", "full_attention"],
    )
    config.dtype = torch.bfloat16
    prefix = "model.layers.1"
    fp8_leaves = ["self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj"]
    fp8_leaves += ["self_attn.o_proj", "linear_attn.out_proj"]
    fp8_leaves += ["linear_attn.in_proj_qkv", "linear_attn.in_proj_z"]
    if not split_gdn:
        fp8_leaves += ["linear_attn.in_proj_b", "linear_attn.in_proj_a"]
    quantized = {f"{prefix}.{leaf}": "FP8" for leaf in fp8_leaves}
    for leaf in ("gate_proj", "up_proj", "down_proj"):
        quantized[f"{prefix}.mlp.{leaf}"] = "NVFP4"
    layer_cls = {
        "linear": Qwen3_5LinearDecoderLayer,
        "attention": Qwen3_5AttentionDecoderLayer,
        "draft": Qwen3_5DraftAttentionDecoderLayer,
    }[kind]
    layer = layer_cls(
        config,
        Mapping(rank=0, world_size=1),
        1,
        quant_config=ModelOptMixedConfig(quantized_layers=quantized),
        prefix=f"{prefix}.linear_attn" if kind == "linear" else f"{prefix}.self_attn",
    ).cuda()

    generator = torch.Generator().manual_seed(0)
    for norm in (layer.input_layernorm, layer.post_attention_layernorm):
        norm.weight.data.copy_(torch.randn(_HIDDEN, generator=generator) * 0.3)
    # Distinct input scales, so a copy made with another projection's scale shows.
    _fill_nvfp4(layer.mlp.gate_up_proj, generator, 0.05)
    _fill_nvfp4(layer.mlp.down_proj, generator, 0.07)
    if kind != "linear":
        _fill_fp8(layer.qkv_proj, generator, 0.02)
        _fill_fp8(layer.o_proj, generator, 0.03)
        layer.attn = _StubAttention(layer.num_heads, layer.num_kv_heads, layer.head_dim)
        return layer
    gdn = layer.linear_attn
    assert gdn._split_in_proj == split_gdn
    _fill_fp8(gdn.out_proj, generator, 0.03)
    if split_gdn:
        _fill_fp8(gdn.in_proj_qkvz, generator, 0.02)
        gdn.in_proj_ba.weight.data.copy_(
            torch.randn(gdn.in_proj_ba.weight.shape, generator=generator) * 0.02
        )
    else:
        _fill_fp8(gdn.in_proj_qkvzba, generator, 0.02)
    return layer


def _run_layer(layer, kind: str, hidden_states, residual, live_rows):
    rows = hidden_states.shape[0]
    ctx = SimpleNamespace(
        draft_narrowing=None if live_rows is None else object(),
        gather_ids=live_rows,
        query_shard=None,
        collective_global_num_tokens=None,
        global_num_tokens=None,
        collective_num_tokens=None,
        input_num_tokens=rows,
        forward_mode=SimpleNamespace(is_idle=lambda: False),
        attn_backend=_StubGDNBackend(),
        token_to_kv_pool=None,
        bs=rows,
    )
    if kind == "linear":
        return layer(hidden_states=hidden_states, residual=residual, ctx=ctx)
    positions = torch.arange(rows, device="cuda")
    return layer(
        positions=positions, hidden_states=hidden_states, residual=residual, ctx=ctx
    )


@pytest.mark.skipif(not _IS_BLACKWELL, reason="NVFP4 GEMMs need Blackwell")
@pytest.mark.parametrize(
    "kind,split_gdn,narrow",
    [
        ("attention", False, False),
        ("linear", False, False),
        ("linear", True, False),
        ("draft", False, False),
        ("draft", False, True),
    ],
)
@torch.no_grad()
def test_decoder_layer_hands_each_projection_its_own_quant(
    monkeypatch, bf16_default_dtype, kind: str, split_gdn: bool, narrow: bool
) -> None:
    monkeypatch.setitem(global_server_args_dict, "layer_boundary_norm", "fused")
    layer = _decoder_layer(kind, split_gdn)
    copies = []
    fp8_copies = []
    fp8_copy_alive = []
    add_rmsnorm = layernorm.add_rmsnorm

    def recording_add_rmsnorm(*args, **kwargs):
        copies.append((kwargs["out_fp8"] is not None, kwargs["out_fp4"] is not None))
        fp8_copy_alive.append(any(copy() is not None for copy in fp8_copies))
        if kwargs["out_fp8"] is not None:
            fp8_copies.append(weakref.ref(kwargs["out_fp8"]))
        return add_rmsnorm(*args, **kwargs)

    monkeypatch.setattr(layernorm, "add_rmsnorm", recording_add_rmsnorm)
    launches = _count_quant_launches(monkeypatch)
    torch.manual_seed(1)
    hidden_states = torch.randn(129, _HIDDEN, device="cuda") * 4
    # The single-layer MTP draft opens without a residual and may narrow to live rows.
    residual = None if kind == "draft" else torch.randn_like(hidden_states) * 30
    live_rows = torch.arange(0, 129, 3, device="cuda") if narrow else None

    def run():
        launches.clear()
        outputs = _run_layer(
            layer,
            kind,
            hidden_states.clone(),
            None if residual is None else residual.clone(),
            live_rows,
        )
        return outputs, Counter(launches)

    (out, new_residual), fused_launches = run()
    input_copy = kind != "draft"
    assert copies == ([(True, False), (False, True)] if input_copy else [(False, True)])
    # The FP8 copy is freed with its consumer, before the post-attention norm runs.
    assert fp8_copy_alive == [False] * len(copies)

    # Without the hooks every projection quantizes the same normed rows itself.
    monkeypatch.setattr(
        Fp8LinearMethod, "static_fp8_input_scale", lambda self, layer: None
    )
    monkeypatch.setattr(
        Nvfp4LinearMethod, "nvfp4_global_scale", lambda self, layer: None
    )
    (expected, expected_residual), own_launches = run()
    assert not any(fp8 or fp4 for fp8, fp4 in copies[2 if input_copy else 1 :])
    assert torch.equal(new_residual, expected_residual)
    assert torch.equal(out, expected)
    # Each copy replaces exactly its consumer's own input quant.
    assert own_launches["fp8"] - fused_launches["fp8"] == int(input_copy)
    assert own_launches["nvfp4"] - fused_launches["nvfp4"] == 1
