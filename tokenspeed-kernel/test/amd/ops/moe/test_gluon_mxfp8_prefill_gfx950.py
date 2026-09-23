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

"""Native contracts for the gfx950 sorted-scale MXFP8 expert pipeline."""

from __future__ import annotations

import pytest
import torch
from kimi3_reference import dequantize_mxfp4, mxfp4_moe_reference
from utils import is_cdna4

if not is_cdna4():
    pytest.skip("AMD CDNA4 is required", allow_module_level=True)

import tokenspeed_kernel  # noqa: E402
from test_gluon_mxfp4_situ_gfx950 import _make_mxfp4_module  # noqa: E402
from tokenspeed_kernel.ops.moe import (  # noqa: E402
    latent_moe_decode_pipeline_available,
    latent_moe_expert_shared,
)
from tokenspeed_kernel_amd._scheduling import (  # noqa: E402
    sched_barrier_compile_options,
)
from tokenspeed_kernel_amd._triton import (  # noqa: E402
    cdna4_async_copy,
    gl,
    gluon,
    triton,
)
from tokenspeed_kernel_amd.ops.gfx950.moe.mxfp4.expert_mesh import (  # noqa: E402
    _build_small_mesh,
    _clear_mesh,
    _count_mesh,
    _fill_mesh,
    _scatter_mesh,
    sort_expert_slots,
)
from tokenspeed_kernel_amd.ops.gfx950.moe.mxfp4.mxfp8_gemm import (  # noqa: E402
    _A,
    _MMA,
    _SHARED,
    _a_offsets,
    _assemble_fragment,
    _copy_a,
    _fragment_k64,
    _load_a,
    _mxfp8_stage1,
    _mxfp8_stage2,
    _publish_a,
    _situ,
)
from tokenspeed_kernel_amd.ops.gfx950.moe.mxfp4.mxfp8_quantize import (  # noqa: E402
    _quantize_mxfp8_kernel,
    _quantize_sorted_mxfp8,
    _sort_mxfp8_scales,
    quantize_mxfp8,
)
from tokenspeed_kernel_amd.ops.gfx950.moe.mxfp4.n16_weights import (  # noqa: E402
    n16_mxfp4_shape,
    preprocess_n16_mxfp4_weights,
)
from tokenspeed_kernel_amd.ops.gfx950.moe.mxfp4.prefill_mxfp8 import (  # noqa: E402
    mxfp8_situ_prefill,
)
from tokenspeed_kernel_amd.ops.gfx950.moe.mxfp4.situ_decode import (  # noqa: E402
    _stage2_a16w4_warp_gemv_combine,
    gluon_a16w4_situ_warp_decode_ep_gfx950,
)

_NAMES = ("w13_weight", "w13_weight_scale", "w2_weight", "w2_weight_scale")
_RAW_NAMES = ("w13_weight", "w13_scale", "w2_weight", "w2_scale")
_A8 = "gluon_mxfp4_a8w4_situ_ep_precomputed_moe_apply"


def _module(e: int, d: int, i: int, topk: int):
    module, raw = _make_mxfp4_module(
        num_experts=e,
        latent_size=d,
        intermediate_size=i,
        top_k=topk,
        generator=torch.Generator(device="cuda").manual_seed(731),
    )
    module.num_local_experts = e
    module.num_experts = e * 8
    module.ep_size, module.ep_rank = 8, 1
    module.w13_input_layout = "concatenated"
    return module, raw


def _plan(policy: str):
    return tokenspeed_kernel.moe_plan(
        "mxfp4",
        input_dtype=torch.bfloat16,
        activation="situ",
        routing_mode="precomputed_topk",
        ep_size=8,
        ispp=3072,
        internal_activation_dtype=policy,
        solution="gluon",
        hidden=None,
        swiglu_form=None,
        activation_clamped=False,
        expert_id_repeats=False,
        fast_math=True,
    )


def _bank(module):
    return tuple(getattr(module, name) for name in _NAMES)


def _q32(x):
    groups = x.float().reshape(x.shape[0], -1, 32)
    magnitude = torch.where(torch.isnan(groups), 0.0, groups.abs())
    amax = magnitude.amax(-1).clamp_min(1.0e-10)
    bits = (amax * (1.0 / 448.0)).view(torch.int32)
    exponent = (bits >> 23) & 255
    exponent = exponent + ((exponent < 255) & ((bits & 0x7FFFFF) != 0))
    scale = (exponent << 23).view(torch.float32)
    q = (groups * scale.reciprocal().unsqueeze(-1)).to(torch.float8_e4m3fn)
    return q.reshape_as(x), exponent.to(torch.uint8)


def _dequant_q(q, scale):
    return (
        q.float().reshape(q.shape[0], -1, 32)
        * torch.exp2(scale.float() - 127).unsqueeze(-1)
    ).reshape(q.shape)


def _situ_math(g, u):
    # Independent FP32 math; hardware exp2/rcp can differ at a BF16 boundary.
    return (
        4
        * torch.tanh(g.float() / 4)
        * torch.sigmoid(g.float())
        * (25 * torch.tanh(u.float() / 25))
    ).bfloat16()


def _expert_math(x, raw, ids, weights, start):
    xq, xs = _q32(x)
    a = _dequant_q(xq, xs)
    result = torch.zeros_like(x, dtype=torch.float32)
    for expert in range(raw["w13_weight"].shape[0]):
        token, slot = (ids == expert + start).nonzero(as_tuple=True)
        if not token.numel():
            continue
        w13 = dequantize_mxfp4(raw["w13_weight"][expert], raw["w13_scale"][expert])
        gu = a[token] @ w13.T
        g, u = gu.chunk(2, -1)
        zq, zs = _q32(_situ_math(g, u))
        w2 = dequantize_mxfp4(raw["w2_weight"][expert], raw["w2_scale"][expert])
        projected = _dequant_q(zq, zs) @ w2.T
        contribution = (projected * weights[token, slot].float()[:, None]).bfloat16()
        result.index_add_(0, token, contribution.float())
    return result.bfloat16()


def _assert_rms(actual, expected):
    assert torch.isfinite(actual).all()
    error = (actual.float() - expected.float()).square().mean().sqrt()
    scale = expected.float().square().mean().sqrt().clamp_min(1e-12)
    assert (error / scale).item() <= 0.04


def _record_compiled_calls(monkeypatch, kernels):
    calls = {kernel.fn.__name__: [] for kernel in kernels}

    def observe(kernel):
        run = kernel.run

        def record(*args, **kwargs):
            compiled = run(*args, **kwargs)
            calls[kernel.fn.__name__].append(compiled)
            return compiled

        monkeypatch.setattr(kernel, "run", record)

    for kernel in kernels:
        observe(kernel)
    return calls


def _assert_compiled_reuse(handles, name, record_property):
    assert handles and handles[0] is not None, name
    assert all(handle is handles[0] for handle in handles), name
    assert len({handle.hash for handle in handles}) == 1, name
    record_property(f"{name}_compiled_hash", handles[0].hash)


def _quantize_probe(x, fused):
    rows, k = x.shape
    q_storage = torch.full((x.numel() + 32,), 0x55, device="cuda", dtype=torch.uint8)
    q = q_storage[:-32].view(torch.float8_e4m3fn).view_as(x)
    if not fused:
        scales = torch.empty((rows, k // 32), device="cuda", dtype=torch.uint8)
        compiled = _quantize_mxfp8_kernel[((scales.numel() + 63) // 64,)](
            x,
            q,
            scales,
            rows,
            k,
            x.stride(0),
            x.stride(1),
            num_warps=1,
            enable_fp_fusion=False,
        )
    else:
        padded = triton.cdiv(rows, 32) * 32
        ids = torch.arange(padded, device="cuda", dtype=torch.int32)
        ids[rows:] = rows | (1 << 24)
        valid = torch.tensor([padded], device="cuda", dtype=torch.int32)
        scale_storage = torch.full(
            (padded * (k // 32) + 32,), 0x55, device="cuda", dtype=torch.uint8
        )
        value_blocks = triton.cdiv(rows * (k // 32), 128)
        scale_blocks = triton.cdiv(padded * (k // 32), 128)
        compiled = _quantize_sorted_mxfp8[(value_blocks + scale_blocks,)](
            x,
            q,
            ids,
            valid,
            scale_storage,
            rows,
            rows,
            value_blocks,
            1,
            k,
            x.stride(0),
            x.stride(1),
            False,
            num_warps=4,
            enable_fp_fusion=False,
        )
        canonical = (
            scale_storage[:-32]
            .reshape(padded // 32, k // 256, 4, 16, 2, 2)
            .permute(0, 5, 3, 1, 4, 2)
            .contiguous()
            .reshape(padded, k // 32)
        )
        torch.testing.assert_close(
            canonical[rows:], torch.full_like(canonical[rows:], 127), atol=0, rtol=0
        )
        torch.testing.assert_close(
            scale_storage[-32:],
            torch.full_like(scale_storage[-32:], 0x55),
            atol=0,
            rtol=0,
        )
        scales = canonical[:rows]
    torch.testing.assert_close(
        q_storage[-32:], torch.full_like(q_storage[-32:], 0x55), atol=0, rtol=0
    )
    return q, scales, compiled


@pytest.mark.parametrize("k", [256, 3072, 3584])
@pytest.mark.parametrize("fused", [False, True])
def test_q32_finite_bits_and_signed_zero(k, fused):
    values = torch.tensor(
        [
            0.0,
            -0.0,
            2**-133,
            -(2**-133),
            2**-126,
            -(2**-126),
            1.0,
            1.0625,
            1.1875,
            448.0,
            450.0,
            464.0,
            torch.finfo(torch.bfloat16).max,
        ],
        device="cuda",
        dtype=torch.bfloat16,
    )
    x = values[:, None].expand(-1, k).clone()
    x[:, 1::32] = -0.0
    x[:, 2::32] = 0.0
    q, scales, _ = _quantize_probe(x, fused)
    expected, expected_s = _q32(x)
    torch.testing.assert_close(
        q.view(torch.uint8), expected.view(torch.uint8), atol=0, rtol=0
    )
    torch.testing.assert_close(scales, expected_s, atol=0, rtol=0)
    assert int(scales[0, 0]) == 85


@pytest.mark.parametrize("fused", [False, True])
def test_q32_nonfinite_classification(fused, record_property):
    x = torch.ones((5, 256), device="cuda", dtype=torch.bfloat16)
    x[0, :] = float("nan")
    x[1, 0::32] = float("nan")
    x[2, 0::32] = float("inf")
    x[3, 0::32] = -float("inf")
    x[4, 0::32], x[4, 1::32] = float("nan"), float("inf")
    x[:, 2::32] = -0.0
    q, scales, _ = _quantize_probe(x, fused)
    expected, expected_s = _q32(x)
    torch.testing.assert_close(scales, expected_s, atol=0, rtol=0)
    torch.testing.assert_close(torch.isnan(q.float()), torch.isnan(expected.float()))
    finite = torch.isfinite(expected.float())
    torch.testing.assert_close(
        q.view(torch.uint8)[finite], expected.view(torch.uint8)[finite], atol=0, rtol=0
    )
    record_property(
        "nan_payload_differences",
        int(((q.view(torch.uint8) != expected.view(torch.uint8)) & ~finite).sum()),
    )


@pytest.mark.parametrize("fused", [False, True])
def test_q32_all_bf16_encodings_use_hardware_scaled_downcast(fused):
    x = torch.arange(2**16, device="cuda", dtype=torch.int32).to(torch.int16)
    x = x.view(torch.bfloat16).reshape(256, 256)
    q, scales, compiled = _quantize_probe(x, fused)
    expected, expected_s = _q32(x)
    torch.testing.assert_close(scales, expected_s, atol=0, rtol=0)
    torch.testing.assert_close(torch.isnan(q.float()), torch.isnan(expected.float()))
    finite = torch.isfinite(expected.float())
    torch.testing.assert_close(
        q.view(torch.uint8)[finite], expected.view(torch.uint8)[finite], atol=0, rtol=0
    )
    assert "v_cvt_scalef32_pk_fp8_f32" in compiled.asm["amdgcn"]


@pytest.mark.parametrize("topk", [1, 16, 32, 33, 65])
@pytest.mark.parametrize("block_m", [32, 128])
@pytest.mark.parametrize("m", [1, 5, 35, 64, 65])
def test_mesh_preserves_repeated_slots_and_remote_routes(topk, block_m, m):
    e, start = 3, 32
    ids = torch.full((m, topk), start, dtype=torch.int64, device="cuda")
    ids[0, :] = -1
    if m > 3:
        ids[2, ::2] = 127
        ids[3, :] = start + 2
    weights = (
        torch.arange(m * topk, device="cuda", dtype=torch.float32).view(m, topk) / 128
    )
    out_storage = torch.full((m, 512), 7.0, dtype=torch.bfloat16, device="cuda")
    out = out_storage[:, :256]
    saved_ids, saved_weights = ids.clone(), weights.clone()
    sorted_ids, sorted_weights, expert_ids, valid = sort_expert_slots(
        ids,
        weights,
        out,
        global_experts=128,
        local_experts=e,
        expert_start=start,
        block_m=block_m,
    )
    count = int(valid[0])
    seen = []
    decoded = sorted_ids[:count].to(torch.int64) & 0xFFFFFFFF
    for index, value in enumerate(decoded.cpu().tolist()):
        token, slot = value & 0xFFFFFF, value >> 24
        expert = int(expert_ids[index // block_m])
        if token == m:
            assert slot == topk
        else:
            assert int(ids[token, slot]) == expert + start
            assert float(sorted_weights[index]) == float(weights[token, slot])
            seen.append((token, slot))
    expected = ((ids >= start) & (ids < start + e)).nonzero().cpu().tolist()
    assert sorted(seen) == sorted(map(tuple, expected))
    assert count == sum(
        triton.cdiv(int((ids == expert + start).sum()), block_m) * block_m
        for expert in range(e)
    )
    torch.testing.assert_close(out, torch.zeros_like(out), atol=0, rtol=0)
    torch.testing.assert_close(
        out_storage[:, 256:], torch.full_like(out, 7), atol=0, rtol=0
    )
    torch.testing.assert_close(ids, saved_ids, atol=0, rtol=0)
    torch.testing.assert_close(weights, saved_weights, atol=0, rtol=0)


@pytest.mark.parametrize("k", [256, 3072, 3584])
@pytest.mark.parametrize("block_m", [32, 128])
@pytest.mark.parametrize("slot_major", [False, True])
def test_sorted_scale_words_follow_token_slot_rows(k, block_m, slot_major):
    m, topk = 17, 4
    ids = (
        torch.tensor([[0, 0, 1, -1]], device="cuda", dtype=torch.int32)
        .expand(m, -1)
        .clone()
    )
    ids[0] = torch.tensor([-1, 0, 0, 1], device="cuda", dtype=torch.int32)
    ids[1].fill_(-1)
    weights = torch.ones_like(ids, dtype=torch.float32)
    weights[0, 1] = 0.0
    out = torch.empty((m, 256), dtype=torch.bfloat16, device="cuda")
    sorted_ids, _, _, valid = sort_expert_slots(
        ids,
        weights,
        out,
        global_experts=4,
        local_experts=2,
        expert_start=0,
        block_m=block_m,
    )
    x = (
        torch.exp2(
            (
                torch.arange(m * topk if slot_major else m, device="cuda")[:, None] % 7
                - 3
            ).float()
        )
        .expand(-1, k)
        .bfloat16()
    )
    q, scales = quantize_mxfp8(
        x,
        sorted_ids,
        valid,
        tokens=m,
        topk=topk,
        slot_major=slot_major,
        block_m=block_m,
    )
    expected_q, raw = _q32(x)
    torch.testing.assert_close(
        q.view(torch.uint8), expected_q.view(torch.uint8), atol=0, rtol=0
    )
    count = int(valid[0])
    encoded = sorted_ids[:count].to(torch.int64) & 0xFFFFFFFF
    token, slot = encoded & 0xFFFFFF, encoded >> 24
    canonical = torch.full((count, k // 32), 127, dtype=torch.uint8, device="cuda")
    live = token < m
    source = token * topk + slot if slot_major else token
    canonical[live] = raw[source[live]]
    packed = (
        canonical.reshape(count // 32, 2, 16, k // 256, 2, 4)
        .permute(0, 3, 5, 2, 4, 1)
        .contiguous()
    )
    torch.testing.assert_close(
        scales.view(-1)[: count * (k // 32)], packed.view(-1), atol=0, rtol=0
    )


@pytest.mark.parametrize("slot_major", [False, True])
def test_fused_quantizer_empty_prefix_preserves_scale_storage(slot_major):
    m, topk, k = 17, 4, 256
    rows = m * topk if slot_major else m
    storage = torch.randn((rows, 2 * k), device="cuda", dtype=torch.bfloat16)
    x = storage[:, ::2]
    saved = storage.clone()
    q = torch.empty_like(x, dtype=torch.float8_e4m3fn)
    ids = torch.full((128,), -1, device="cuda", dtype=torch.int32)
    valid = torch.zeros((2,), device="cuda", dtype=torch.int32)
    scales = torch.full((128, k // 32), 0x55, device="cuda", dtype=torch.uint8)
    value_blocks = triton.cdiv(rows * (k // 32), 128)
    _quantize_sorted_mxfp8[(value_blocks + 8,)](
        x,
        q,
        ids,
        valid,
        scales,
        rows,
        m,
        value_blocks,
        topk,
        k,
        x.stride(0),
        x.stride(1),
        slot_major,
        num_warps=4,
        enable_fp_fusion=False,
    )
    expected, _ = _q32(x)
    torch.testing.assert_close(
        q.view(torch.uint8), expected.view(torch.uint8), atol=0, rtol=0
    )
    torch.testing.assert_close(scales, torch.full_like(scales, 0x55), atol=0, rtol=0)
    torch.testing.assert_close(storage, saved, atol=0, rtol=0)


@pytest.mark.parametrize("k", [256, 3072, 3584])
@pytest.mark.parametrize("slot_major", [False, True])
def test_fused_quantizer_partial_prefix_preserves_scale_storage(k, slot_major):
    m, topk, capacity = 17, 4, 128
    rows = m * topk if slot_major else m
    storage = torch.randn((rows, 2 * k), device="cuda", dtype=torch.bfloat16)
    x, saved = storage[:, ::2], storage.clone()
    q = torch.empty_like(x, dtype=torch.float8_e4m3fn)
    row = torch.arange(capacity, device="cuda", dtype=torch.int32)
    token = torch.where(row % 13 == 2, m, row % m)
    slot = row % topk
    ids = token | (slot << 24)
    valid = torch.zeros((2,), device="cuda", dtype=torch.int32)
    scales = torch.empty((capacity, k // 32), device="cuda", dtype=torch.uint8)
    value_blocks = triton.cdiv(rows * (k // 32), 128)
    scale_blocks = triton.cdiv(capacity * (k // 32), 128)
    expected_q, raw = _q32(x)
    for count in (1, 16, 17, 32):
        valid[0] = count
        q.fill_(0)
        scales.fill_(0x55)
        _quantize_sorted_mxfp8[(value_blocks + scale_blocks,)](
            x,
            q,
            ids,
            valid,
            scales,
            rows,
            m,
            value_blocks,
            topk,
            k,
            x.stride(0),
            x.stride(1),
            slot_major,
            num_warps=4,
            enable_fp_fusion=False,
        )
        canonical = torch.full_like(scales, 0x55)
        canonical[:count] = 127
        live = (row < count) & (token < m)
        source = token * topk + slot if slot_major else token
        canonical[live] = raw[source[live]]
        packed = (
            canonical.reshape(capacity // 32, 2, 16, k // 256, 2, 4)
            .permute(0, 3, 5, 2, 4, 1)
            .contiguous()
        )
        torch.testing.assert_close(
            q.view(torch.uint8), expected_q.view(torch.uint8), atol=0, rtol=0
        )
        torch.testing.assert_close(scales.view(-1), packed.view(-1), atol=0, rtol=0)
    torch.testing.assert_close(storage, saved, atol=0, rtol=0)


@pytest.mark.parametrize("layout", ["concatenated", "interleaved"])
def test_n16_bank_lifecycle_has_no_attribute_dependency(layout):
    module, raw = _module(2, 256, 256, 4)
    module.w13_input_layout = layout
    if layout == "interleaved":
        for name in _NAMES[:2]:
            t = getattr(module, name)
            setattr(
                module,
                name,
                torch.nn.Parameter(
                    t.reshape(2, 2, 256, -1).transpose(1, 2).contiguous().reshape_as(t),
                    requires_grad=False,
                ),
            )
    preprocess_n16_mxfp4_weights(module)
    bank = _bank(module)
    assert n16_mxfp4_shape(*bank) == (2, 256, 256)
    assert set(dict(module.named_parameters())) == set(_NAMES)
    assert sum(t.numel() for t in bank) == sum(t.numel() for t in raw.values())
    for name, t in zip(_NAMES, bank, strict=True):
        copied = t.detach().clone().to("cpu").to("cuda").view_as(t)
        setattr(module, name, torch.nn.Parameter(copied, requires_grad=False))
    assert n16_mxfp4_shape(*_bank(module)) == (2, 256, 256)
    for actual, expected in zip(_bank(module), bank, strict=True):
        torch.testing.assert_close(actual, expected, atol=0, rtol=0)
    with pytest.raises(ValueError, match="already prepared"):
        preprocess_n16_mxfp4_weights(module)
    with pytest.raises(ValueError, match="rank-6"):
        n16_mxfp4_shape(*[t.view(t.shape[0], -1, 1) for t in bank])


@pytest.mark.parametrize("k,m", [(256, 1), (256, 129), (3072, 1), (3072, 129)])
@pytest.mark.parametrize("block_m", [32, 128])
def test_stage2_full_k_scale_fragments_and_fp32_route_product(k, m, block_m):
    _check_stage2_basis(k, m, pitch=None, buffer_safe=True, block_m=block_m)


@pytest.mark.parametrize("block_m", [32, 128])
def test_stage2_odd_k_final_slot_and_deferred_tail(block_m):
    _check_stage2_basis(3328, 129, pitch=None, buffer_safe=True, block_m=block_m)


@pytest.mark.parametrize("block_m", [32, 128])
def test_stage2_full_k_wide_row_byte_offsets(block_m):
    # The third row starts beyond 4 GiB; only the view's live columns are touched.
    _check_stage2_basis(3072, 3, pitch=2**30 + 4096, buffer_safe=False, block_m=block_m)


def _check_stage2_basis(k, m, *, pitch, buffer_safe, block_m):
    d = 3584 if k == 3072 else 256
    module, raw = _module(1, d, k, 1)
    raw["w2_weight"].zero_()
    n = torch.arange(d, device="cuda")
    group = (n * 17) % (k // 32)
    raw["w2_weight"][0, n, group * 16] = 0x22
    raw["w2_scale"].copy_(
        (123 + (n[:, None] + torch.arange(k // 32, device="cuda")[None, :]) % 8).to(
            torch.uint8
        )
    )
    rows = torch.arange(m, device="cuda")
    amplitude = torch.exp2(
        (
            (rows[:, None] // 16 + torch.arange(k // 32, device="cuda")[None, :]) % 5
            - 2
        ).float()
    )
    x = torch.zeros((m, k // 32, 32), device="cuda", dtype=torch.bfloat16)
    x[:, :, 0], x[:, :, 1] = amplitude, amplitude / 256
    x = x.reshape(m, k)
    ids = torch.zeros((m, 1), dtype=torch.int32, device="cuda")
    weights = torch.full((m, 1), 1.00390625, dtype=torch.float32, device="cuda")
    storage = torch.empty_strided(
        (m, d + 64),
        (d + 64 if pitch is None else pitch, 1),
        device="cuda",
        dtype=torch.bfloat16,
    ).fill_(17.0)
    out = storage[:, :d]
    sorted_ids, sorted_weights, experts, valid = sort_expert_slots(
        ids,
        weights,
        out,
        global_experts=8,
        local_experts=1,
        expert_start=0,
        block_m=block_m,
    )
    q, scales = quantize_mxfp8(
        x, sorted_ids, valid, tokens=m, topk=1, slot_major=True, block_m=block_m
    )
    preprocess_n16_mxfp4_weights(module)
    _mxfp8_stage2[(d // 128, experts.numel())](
        q,
        scales,
        module.w2_weight,
        module.w2_weight_scale,
        sorted_ids,
        sorted_weights,
        experts,
        valid,
        out,
        m,
        1,
        k,
        d,
        1,
        out.stride(0),
        buffer_safe,
        buffer_safe,
        BM=block_m,
        num_warps=4,
        num_stages=1,
        enable_fp_fusion=False,
        **sched_barrier_compile_options(),
    )
    scale_w = torch.exp2(raw["w2_scale"][0, n, group].float() - 127)
    expected = (
        amplitude[:, group].double()
        * (257.0 / 256)
        * scale_w.double()[None, :]
        * (257.0 / 256)
    ).bfloat16()
    torch.testing.assert_close(
        out.view(torch.int16), expected.view(torch.int16), atol=0, rtol=0
    )
    torch.testing.assert_close(
        storage[:, d:], torch.full_like(storage[:, d:], 17), atol=0, rtol=0
    )


@pytest.mark.parametrize("block_m", [32, 128])
def test_stage1_full_k_gate_up_basis_and_bf16_intermediate(block_m):
    _check_stage1_basis(3584, False, block_m)


@pytest.mark.parametrize("d", [256, 3328, 3584])
@pytest.mark.parametrize("block_m", [32, 128])
def test_stage1_all_scale_roles_and_odd_even_tail(d, block_m):
    _check_stage1_basis(d, True, block_m)


def _check_stage1_basis(d, varying_scales, block_m):
    m, i = 129, 3072
    module, raw = _module(1, d, i, 1)
    raw["w13_weight"].zero_()
    n = torch.arange(2 * i, device="cuda")
    group = (n * 17 + n // i * 7) % (d // 32)
    raw["w13_weight"][0, n, group * 16] = (2 + n % 4).to(torch.uint8)
    raw["w13_scale"].fill_(125)
    if varying_scales:
        raw["w13_scale"].copy_(
            (123 + (n[:, None] // 16 + torch.arange(d // 32, device="cuda")) % 5).to(
                torch.uint8
            )
        )
    x = torch.exp2(
        (
            (
                torch.arange(m, device="cuda")[:, None] // 16
                + torch.arange(d // 32, device="cuda")[None, :]
            )
            % 7
            - 4
        ).float()
    )
    x = x[:, :, None].expand(-1, -1, 32).reshape(m, d).bfloat16()
    ids = torch.zeros((m, 1), dtype=torch.int32, device="cuda")
    sorted_ids, _, experts, valid = sort_expert_slots(
        ids,
        torch.ones_like(ids, dtype=torch.float32),
        torch.empty_like(x),
        global_experts=8,
        local_experts=1,
        expert_start=0,
        block_m=block_m,
    )
    q, scale = quantize_mxfp8(
        x, sorted_ids, valid, tokens=m, topk=1, slot_major=False, block_m=block_m
    )
    z = torch.empty((m, i), device="cuda", dtype=torch.bfloat16)
    preprocess_n16_mxfp4_weights(module)
    _mxfp8_stage1[(i // 64, sorted_ids.numel() // block_m)](
        q,
        scale,
        module.w13_weight,
        module.w13_weight_scale,
        sorted_ids,
        experts,
        valid,
        z,
        m,
        1,
        d,
        i,
        1,
        4.0,
        25.0,
        True,
        BM=block_m,
        num_warps=4,
        num_stages=1,
        enable_fp_fusion=False,
        **sched_barrier_compile_options(),
    )
    code = torch.tensor([1.0, 1.5, 2.0, 3.0], device="cuda")[n % 4]
    scale_w = torch.exp2(raw["w13_scale"][0, n, group].float() - 127)
    gu = x.float()[:, group * 32] * code[None, :] * scale_w[None, :]
    expected = _situ_math(*gu.chunk(2, -1))
    torch.testing.assert_close(z, expected, atol=2e-4, rtol=8e-3)


@gluon.jit
def _a_tile_probe(
    x,
    ids,
    out,
    K: gl.constexpr,
    SLOT_MAJOR: gl.constexpr,
    BUFFER_SAFE: gl.constexpr,
    BM: gl.constexpr,
):
    offsets = _a_offsets(ids, 0, 17, 4, K, SLOT_MAJOR, BM, not SLOT_MAJOR)
    smem = gl.allocate_shared_memory(gl.uint8, [BM, 256], _SHARED)
    row = gl.arange(0, 16, layout=gl.SliceLayout(1, _A))
    col = gl.arange(0, 128, layout=gl.SliceLayout(0, _A))
    for kt in gl.static_range(K // 256):
        if SLOT_MAJOR:
            a = _load_a(x, offsets, kt, BUFFER_SAFE, 0, 2)
            _publish_a(smem, a, 0, 2)
            if BM == 128:
                a = _load_a(x, offsets, kt, BUFFER_SAFE, 2, 8)
                _publish_a(smem, a, 2, 8)
        else:
            _copy_a(smem, x, offsets, kt, BUFFER_SAFE)
            cdna4_async_copy.wait_group(0)
        gl.barrier()
        for mi in gl.static_range(BM // 16):
            for kh in gl.static_range(2):
                value = _assemble_fragment(
                    _fragment_k64(smem, mi, kh, 0),
                    _fragment_k64(smem, mi, kh, 1),
                ).to(gl.uint8, bitcast=True)
                gl.store(
                    out
                    + (mi * 16 + row[:, None]) * K
                    + kt * 256
                    + kh * 128
                    + col[None, :],
                    value,
                )
        gl.barrier()


@pytest.mark.parametrize("k", [256, 3072, 3584])
@pytest.mark.parametrize("slot_major", [False, True])
@pytest.mark.parametrize("buffer_safe", [False, True])
@pytest.mark.parametrize("block_m", [32, 128])
def test_xor_a_tiles_safe_padding_and_split_fragments(
    k, slot_major, buffer_safe, block_m
):
    row = torch.arange(block_m, device="cuda", dtype=torch.int32)
    token, slot = (row * 7) % 17, row % 4
    encoded = token | (slot << 24)
    encoded[7], encoded[31 if block_m == 128 else 15], encoded[block_m - 1] = (
        17,
        4 << 24,
        -1,
    )
    first_duplicate = 40 if block_m == 128 else 20
    encoded[first_duplicate : first_duplicate + 4] = 9 | (
        torch.arange(4, device="cuda") << 24
    )
    r = torch.arange(17 * 4 if slot_major else 17, device="cuda")[:, None]
    col = torch.arange(k, device="cuda")[None, :]
    x = (r * 37 + col * 17 + col // 256 * 29 + col // 64 * 11).to(torch.uint8)
    saved_x, saved_ids = x.clone(), encoded.clone()
    actual = torch.empty((block_m, k), device="cuda", dtype=torch.uint8)
    compiled = _a_tile_probe[(1,)](
        x,
        encoded,
        actual,
        k,
        slot_major,
        buffer_safe,
        block_m,
        num_warps=4,
        num_stages=1,
    )
    if not slot_major:
        # Both row tiles must bypass VGPR staging while preserving XOR LDS rows.
        assert any(
            "load" in line and "lds" in line and ("dwordx4" in line or "b128" in line)
            for line in compiled.asm["amdgcn"].splitlines()
        )
    bits = encoded.to(torch.int64) & 0xFFFFFFFF
    token, slot = bits & 0xFFFFFF, bits >> 24
    valid = (token < 17) & (slot < 4)
    source = token * 4 + slot if slot_major else token
    source = torch.where(valid, source, 0)
    torch.testing.assert_close(actual, x[source], atol=0, rtol=0)
    torch.testing.assert_close(x, saved_x, atol=0, rtol=0)
    torch.testing.assert_close(encoded, saved_ids, atol=0, rtol=0)


@gluon.jit
def _situ_boundary_probe(g, u, out, SIZE: gl.constexpr):
    index = gl.arange(0, 256, layout=gl.BlockedLayout([1], [64], [4], [0]))
    gate = gl.load(g + index, mask=index < SIZE, other=0.0)
    up = gl.load(u + index, mask=index < SIZE, other=0.0)
    gl.store(out + index, _situ(gate, up, 4.0, 25.0), mask=index < SIZE)


def test_situ_nonfinite_and_signed_zero_classification(record_property):
    values = torch.tensor(
        [
            0.0,
            -0.0,
            2**-133,
            -(2**-133),
            1.0,
            -1.0,
            float("inf"),
            -float("inf"),
            float("nan"),
        ],
        device="cuda",
    )
    g, u = torch.meshgrid(values, values, indexing="ij")
    g, u = g.flatten().contiguous(), u.flatten().contiguous()
    actual = torch.empty_like(g, dtype=torch.bfloat16)
    _situ_boundary_probe[(1,)](
        g, u, actual, g.numel(), num_warps=4, enable_fp_fusion=False
    )

    # Match the source's ordered sign selection, not a fictitious FP64 oracle.
    def source_tanh(x):
        e = torch.exp2(-2.8853900817779268 * x.abs())
        value = (1 - e) * (1 + e).reciprocal()
        return torch.where(x > 0, value, -value)

    expected = (
        4
        * source_tanh(g / 4)
        * (1 + torch.exp2(-1.4426950408889634 * g)).reciprocal()
        * (25 * source_tanh(u / 25))
    ).bfloat16()
    torch.testing.assert_close(torch.isnan(actual), torch.isnan(expected))
    torch.testing.assert_close(torch.isinf(actual), torch.isinf(expected))
    zero_or_inf = (expected == 0) | torch.isinf(expected)
    torch.testing.assert_close(
        torch.signbit(actual[zero_or_inf]), torch.signbit(expected[zero_or_inf])
    )
    finite = torch.isfinite(expected)
    torch.testing.assert_close(actual[finite], expected[finite], atol=2e-4, rtol=8e-3)
    record_property(
        "nan_payload_differences",
        int(
            (
                (actual.view(torch.int16) != expected.view(torch.int16))
                & torch.isnan(expected)
            ).sum()
        ),
    )


@pytest.mark.parametrize("m", [0, 1, 5, 8, 16, 32, 33, 64, 129, 896, 1024, 1025])
@pytest.mark.parametrize("policy", ["input", "fp8"])
def test_public_prefill_strides_duplicates_empty_and_input_preservation(
    m, policy, monkeypatch
):
    from tokenspeed_kernel_amd.ops.gfx950.moe.mxfp4 import prefill_mxfp8

    module, raw = _module(2, 3584, 3072, 16)
    plan = _plan(policy)
    assert plan["apply_kernel_name"] == _A8
    tokenspeed_kernel.moe_process_weights(plan, module)
    arena = torch.empty((m, 3 * 3584), dtype=torch.bfloat16, device="cuda")
    x, output = arena[:, :3584], arena[:, 3584 : 2 * 3584]
    x.normal_(std=0.1)
    id_storage = torch.empty((m, 32), device="cuda", dtype=torch.int32)
    weight_storage = torch.empty((m, 32), device="cuda", dtype=torch.float32)
    ids, weights = id_storage[:, ::2], weight_storage[:, ::2]
    ids.copy_(
        (torch.arange(m, device="cuda")[:, None] + torch.arange(16, device="cuda")) % 16
    )
    weights.fill_(1 / 16)
    if m:
        ids[0].fill_(-1)
    if m > 1:
        ids[1].fill_(2)
    saved = [t.clone() for t in (x, ids, weights)]
    module._situ_output_buffer = output
    if m == 0:

        def fail(*args, **kwargs):
            pytest.fail("empty MoE must not sort or quantize")

        monkeypatch.setattr(prefill_mxfp8, "sort_expert_slots", fail)
        monkeypatch.setattr(prefill_mxfp8, "quantize_mxfp8", fail)
    result = tokenspeed_kernel.moe_apply(
        plan,
        x,
        module,
        torch.empty((m, 0), device="cuda"),
        topk_ids=ids,
        topk_weights=weights,
    )
    assert result is output
    for actual, expected in zip((x, ids, weights), saved, strict=True):
        torch.testing.assert_close(actual, expected, atol=0, rtol=0)
    if m:
        _assert_rms(result, _expert_math(x, raw, ids, weights, 2))
        torch.testing.assert_close(
            result[0], torch.zeros_like(result[0]), atol=0, rtol=0
        )


@pytest.mark.parametrize("policy", ["input", "fp8"])
def test_public_prefill_reuses_compiled_handles_across_rows(
    policy, monkeypatch, record_property
):
    module, raw = _module(2, 3584, 3072, 16)
    plan = _plan(policy)
    assert plan["apply_kernel_name"] == _A8
    tokenspeed_kernel.moe_process_weights(plan, module)
    calls = _record_compiled_calls(
        monkeypatch,
        (
            _build_small_mesh,
            _clear_mesh,
            _fill_mesh,
            _count_mesh,
            _scatter_mesh,
            _quantize_mxfp8_kernel,
            _sort_mxfp8_scales,
            _quantize_sorted_mxfp8,
            _mxfp8_stage1,
            _mxfp8_stage2,
        ),
    )
    rows = (
        5,
        7,
        8,
        9,
        15,
        16,
        17,
        27,
        31,
        32,
        33,
        63,
        64,
        65,
        127,
        128,
        129,
        896,
        1023,
        1024,
        1025,
        1152,
        1792,
    )
    if policy == "fp8":
        rows = (1, 2, 3, 4, *rows)
    grouped = {}
    for m in rows:
        before = {name: len(handles) for name, handles in calls.items()}
        # Views keep identical strides/alignment while bounds, mesh pitch and
        # launch grids cross both the quantizer and expert-tile boundaries.
        arena = torch.empty((m, 3 * 3584), dtype=torch.bfloat16, device="cuda")
        x, output = arena[:, :3584], arena[:, 3584 : 2 * 3584]
        x.normal_(std=0.1)
        ids = torch.empty((m, 32), dtype=torch.int32, device="cuda")[:, ::2]
        weights = torch.empty((m, 32), dtype=torch.float32, device="cuda")[:, ::2]
        ids.copy_(
            (torch.arange(m, device="cuda")[:, None] + torch.arange(16, device="cuda"))
            % 16
        )
        ids[0].fill_(-1)
        if m > 1:
            ids[1].fill_(2)
        weights.fill_(1 / 16)
        saved = [t.clone() for t in (x, ids, weights)]
        module._situ_output_buffer = output
        result = tokenspeed_kernel.moe_apply(
            plan,
            x,
            module,
            torch.empty((m, 0), device="cuda"),
            topk_ids=ids,
            topk_weights=weights,
        )
        assert result is output
        _assert_rms(result, _expert_math(x, raw, ids, weights, 2))
        torch.testing.assert_close(
            result[0], torch.zeros_like(result[0]), atol=0, rtol=0
        )
        for actual, expected in zip((x, ids, weights), saved, strict=True):
            torch.testing.assert_close(actual, expected, atol=0, rtol=0)
        block_m = 32 if m <= 1024 else 128
        for name, handles in calls.items():
            if name == "_build_small_mesh":
                launches = int(m <= 64)
            elif name in ("_clear_mesh", "_fill_mesh", "_count_mesh"):
                launches = int(m > 64)
            elif name == "_quantize_sorted_mxfp8":
                launches = 2 if block_m == 32 else 0
            elif name in ("_quantize_mxfp8_kernel", "_sort_mxfp8_scales"):
                launches = 2 if block_m == 128 else 0
            else:
                launches = 1
            current = handles[before[name] :]
            assert len(current) == launches, (name, m)
            bucket = (
                block_m
                if name in ("_scatter_mesh", "_mxfp8_stage1", "_mxfp8_stage2")
                else 0
            )
            for stage, handle in enumerate(current):
                grouped.setdefault((name, bucket, stage), []).append(handle)

    for (name, bucket, stage), handles in grouped.items():
        _assert_compiled_reuse(handles, f"{name}_bm{bucket}_{stage}", record_property)


@pytest.mark.parametrize("m", [5, 8, 16, 33, 896, 1024, 1025])
def test_public_prefill_graph_replay_changed_routes_and_values(m):
    module, raw = _module(2, 3584, 3072, 16)
    plan = _plan("input")
    tokenspeed_kernel.moe_process_weights(plan, module)
    x = torch.randn((m, 3584), device="cuda", dtype=torch.bfloat16) * 0.1
    ids = torch.full((m, 16), 2, device="cuda", dtype=torch.int32)
    weights = torch.full((m, 16), 1 / 16, device="cuda", dtype=torch.float32)
    logits = torch.empty((m, 0), device="cuda")
    module._situ_output_buffer = torch.empty_like(x)

    def apply():
        return tokenspeed_kernel.moe_apply(
            plan, x, module, logits, topk_ids=ids, topk_weights=weights
        )

    apply()
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        output = apply()
    x.mul_(0.5)
    ids[:, 0].fill_(3)
    ids[0].fill_(-1)
    weights[:, 0].fill_(0.5)
    snapshots = [t.clone() for t in (x, ids, weights)]
    graph.replay()
    _assert_rms(output, _expert_math(x, raw, ids, weights, 2))
    for actual, expected in zip((x, ids, weights), snapshots, strict=True):
        torch.testing.assert_close(actual, expected, atol=0, rtol=0)


def test_prefill_output_validation_and_disjoint_workspace_pitches():
    arena = torch.empty((16, 768), device="cuda", dtype=torch.bfloat16)
    x, out = arena[::2, :256], arena[:8, 256:512]
    x.normal_(std=0.1)
    saved = x.clone()
    module, raw = _module(1, 256, 256, 1)
    preprocess_n16_mxfp4_weights(module)
    ids = torch.zeros((8, 1), device="cuda", dtype=torch.int32)
    weights = torch.ones_like(ids, dtype=torch.float32)

    def apply(output):
        return mxfp8_situ_prefill(
            x,
            *_bank(module),
            ids,
            weights,
            global_experts=8,
            expert_start=0,
            situ_beta=4.0,
            situ_linear_beta=25.0,
            out=output,
        )

    assert apply(out) is out
    _assert_rms(out, _expert_math(x, raw, ids, weights, 0))
    torch.testing.assert_close(x, saved, atol=0, rtol=0)
    aligned = torch.full((8, 260), 17, device="cuda", dtype=torch.bfloat16)
    view = aligned[:, 2:258]
    assert apply(view) is view
    _assert_rms(view, _expert_math(x, raw, ids, weights, 0))
    assert torch.all(aligned[:, :2] == 17) and torch.all(aligned[:, 258:] == 17)
    invalid = (
        out[:, :-1],
        out.float(),
        arena[:8, :512:2],
        arena[0, :256].expand(8, -1),
        torch.empty((8, 257), device="cuda", dtype=torch.bfloat16)[:, :256],
        aligned[:, 1:257],
    )
    for output in invalid:
        with pytest.raises(ValueError, match="output requires"):
            apply(output)


@pytest.fixture(scope="module", params=["concatenated", "interleaved"])
def joint_bank(request):
    module, raw = _module(112, 3584, 3072, 16)
    plan = _plan("input")
    projections = (
        torch.empty((896, 7168), dtype=torch.bfloat16, device="cuda"),
        torch.empty((3584, 7168), dtype=torch.bfloat16, device="cuda"),
        torch.empty((1536, 7168), dtype=torch.bfloat16, device="cuda"),
        torch.empty((7168, 768), dtype=torch.bfloat16, device="cuda"),
    )
    assert latent_moe_decode_pipeline_available(
        *projections,
        *_bank(module),
        plan,
        topk=16,
        linear_clamp=25.0,
    )
    module.w13_input_layout = request.param
    if request.param == "interleaved":
        for name in _NAMES[:2]:
            t = getattr(module, name)
            setattr(
                module,
                name,
                torch.nn.Parameter(
                    t.reshape(112, 2, 3072, -1)
                    .transpose(1, 2)
                    .contiguous()
                    .reshape_as(t),
                    requires_grad=False,
                ),
            )
    tokenspeed_kernel.moe_process_weights(plan, module)
    assert latent_moe_decode_pipeline_available(
        *projections,
        *_bank(module),
        plan,
        topk=16,
        linear_clamp=25.0,
    )
    assert not latent_moe_decode_pipeline_available(
        *projections,
        *_bank(module),
        _plan("fp8"),
        topk=16,
        linear_clamp=25.0,
    )
    del projections
    return module, raw


@pytest.mark.parametrize("m", [1, 2, 3, 4])
def test_n16_joint_decode_retains_raw_bank_arithmetic(joint_bank, m):
    module, raw = joint_bank
    x = torch.randn((m, 3584), dtype=torch.bfloat16, device="cuda") * 0.1
    ids = (torch.arange(16, device="cuda")[None, :].expand(m, -1) + 112).to(torch.int32)
    ids[:, -1] = -1
    weights = torch.full((m, 16), 1 / 16, device="cuda", dtype=torch.float32)
    shared_input = torch.randn((m, 768), dtype=torch.bfloat16, device="cuda") * 0.1
    shared_weight = torch.randn((7168, 768), dtype=torch.bfloat16, device="cuda") * 0.1

    def decode(bank, interleaved):
        return gluon_a16w4_situ_warp_decode_ep_gfx950(
            x,
            *bank,
            weights,
            ids,
            situ_beta=4.0,
            situ_linear_beta=25.0,
            expert_start=112,
            linear_weights=True,
            w13_interleaved=interleaved,
            shared_input=shared_input,
            shared_weight=shared_weight,
            routed_out=torch.empty_like(x),
            shared_out=torch.empty((m, 7168), dtype=torch.bfloat16, device="cuda"),
        )

    expected = decode(tuple(raw[name] for name in _RAW_NAMES), False)
    routed, shared = latent_moe_expert_shared(
        x,
        *_bank(module),
        weights,
        ids,
        shared_input,
        shared_weight,
        activation_clamp=4.0,
        linear_clamp=25.0,
        expert_start=112,
        w13_interleaved=module.w13_input_layout == "interleaved",
        routed_out=torch.empty_like(x),
        shared_out=torch.empty((m, 7168), dtype=torch.bfloat16, device="cuda"),
    )
    torch.testing.assert_close(
        routed.view(torch.int16), expected[0].view(torch.int16), atol=0, rtol=0
    )
    torch.testing.assert_close(
        shared.view(torch.int16), expected[1].view(torch.int16), atol=0, rtol=0
    )
    local_ids = torch.where(ids >= 112, ids - 112, -1)
    oracle = mxfp4_moe_reference(
        x,
        raw["w13_weight"],
        raw["w13_scale"],
        raw["w2_weight"],
        raw["w2_scale"],
        local_ids,
        weights,
        activation_dtype=torch.bfloat16,
        situ_beta=4.0,
        situ_linear_beta=25.0,
    )
    torch.testing.assert_close(routed, oracle, atol=2e-3, rtol=8e-2)


@pytest.mark.parametrize("m", [4, 8])
@pytest.mark.parametrize(
    "d,i",
    [
        pytest.param(256, 128, id="linear-I128-n16-rejected"),
        (256, 256),
        (256, 512),
        (512, 768),
        (768, 768),
        (3584, 3072),
    ],
)
def test_n16_a16_cell_tiles_tails_and_strided_output(m, d, i):
    module, raw = _module(2, d, i, 4)
    if i == 128:
        with pytest.raises(ValueError, match="D and I divisible by 256"):
            preprocess_n16_mxfp4_weights(module)
        return
    preprocess_n16_mxfp4_weights(module)
    generator = torch.Generator(device="cuda").manual_seed(853)
    x = (
        torch.randn((m, d), dtype=torch.bfloat16, device="cuda", generator=generator)
        * 0.1
    )
    ids = torch.tensor([2, 3, 2, -1], device="cuda", dtype=torch.int32).repeat(m, 1)
    ids[0] = -1
    weights = torch.full((m, 4), 0.25, device="cuda", dtype=torch.float32)
    before = [value.clone() for value in (x, ids, weights)]
    workspace = torch.full((m, d + 8), 17.0, dtype=torch.bfloat16, device="cuda")
    output = workspace[:, :d]

    def decode(bank, out):
        return gluon_a16w4_situ_warp_decode_ep_gfx950(
            x,
            *bank,
            weights,
            ids,
            situ_beta=4.0,
            situ_linear_beta=25.0,
            expert_start=2,
            linear_weights=True,
            w13_interleaved=False,
            routed_out=out,
        )

    actual = decode(_bank(module), output)
    assert actual is output
    # The unchanged linear path cannot upcast its minimum packed-K128 tile.
    # Those small-I cases use the independent oracle below, not a new raw path.
    if i % 512 == 0:
        expected = decode(tuple(raw[name] for name in _RAW_NAMES), torch.empty_like(x))
        torch.testing.assert_close(
            actual.view(torch.int16), expected.view(torch.int16), atol=0, rtol=0
        )
    oracle = mxfp4_moe_reference(
        x,
        raw["w13_weight"],
        raw["w13_scale"],
        raw["w2_weight"],
        raw["w2_scale"],
        torch.where(ids >= 2, ids - 2, -1),
        weights,
        activation_dtype=torch.bfloat16,
        situ_beta=4.0,
        situ_linear_beta=25.0,
    )
    torch.testing.assert_close(actual, oracle, atol=2e-3, rtol=8e-2)
    assert torch.isfinite(actual).all() and torch.count_nonzero(actual[0]) == 0
    assert torch.all(workspace[:, d:] == 17)
    for value, saved in zip((x, ids, weights), before, strict=True):
        torch.testing.assert_close(value, saved, atol=0, rtol=0)


@pytest.mark.parametrize("rows", [(1, 3), (2, 4)])
def test_public_a16_reuses_compiled_handles_across_rows(
    joint_bank, monkeypatch, record_property, rows
):
    module, raw = joint_bank
    plan = _plan("input")
    handles = []
    for m in rows:
        x = torch.randn((m, 3584), device="cuda", dtype=torch.bfloat16) * 0.1
        ids = (torch.arange(16, device="cuda")[None, :].repeat(m, 1) + 112).int()
        ids[:, -1] = -1
        weights = torch.full((m, 16), 1 / 16, dtype=torch.float32, device="cuda")
        expected = gluon_a16w4_situ_warp_decode_ep_gfx950(
            x,
            *(raw[name] for name in _RAW_NAMES),
            weights,
            ids,
            situ_beta=4.0,
            situ_linear_beta=25.0,
            expert_start=112,
            linear_weights=True,
            w13_interleaved=False,
        )
        output = torch.empty_like(x)
        monkeypatch.setattr(module, "_situ_output_buffer", output, raising=False)
        saved = [t.clone() for t in (x, ids, weights)]
        with monkeypatch.context() as context:
            calls = _record_compiled_calls(context, (_stage2_a16w4_warp_gemv_combine,))
            result = tokenspeed_kernel.moe_apply(
                plan,
                x,
                module,
                torch.empty((m, 0), device="cuda"),
                topk_ids=ids,
                topk_weights=weights,
            )
        called = calls["_stage2_a16w4_warp_gemv_combine"]
        assert len(called) == 1
        handles.extend(called)
        assert result is output
        torch.testing.assert_close(
            result.view(torch.int16), expected.view(torch.int16), atol=0, rtol=0
        )
        for actual, original in zip((x, ids, weights), saved, strict=True):
            torch.testing.assert_close(actual, original, atol=0, rtol=0)
    _assert_compiled_reuse(handles, "a16_nonfused_stage2", record_property)


def test_explicit_fp8_rejects_unclamped_and_unprepared_banks():
    module, _ = _module(2, 3584, 3072, 16)
    module.activation_situ_linear_beta = None
    with pytest.raises(ValueError, match="explicit FP8"):
        tokenspeed_kernel.moe_process_weights(_plan("fp8"), module)
    module.activation_situ_linear_beta = 25.0
    x = torch.empty((1, 3584), dtype=torch.bfloat16, device="cuda")
    ids = torch.zeros((1, 16), dtype=torch.int32, device="cuda")
    with pytest.raises(ValueError, match="rank-6"):
        tokenspeed_kernel.moe_apply(
            _plan("input"),
            x,
            module,
            torch.empty((1, 0), device="cuda"),
            topk_ids=ids,
            topk_weights=torch.ones_like(ids, dtype=torch.float32),
        )
