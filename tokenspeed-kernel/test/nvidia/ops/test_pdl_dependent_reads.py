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

"""Tests that PDL-enabled kernels wait for upstream writes before reading.

Each must call ``gdc_wait()`` before it reads anything an upstream kernel
wrote, since ``launch_pdl=True`` lets its CTAs be scheduled before that
upstream kernel retires.

Every case below races a synthetic slow producer against the kernel under
test on the same stream, with no host sync in between. Like the production
kernels, the producer releases its dependents at its start, then spins before
copying the real input over a poisoned buffer. The dependent therefore starts
while the producer is still spinning; only its own ``gdc_wait()``, which returns
after the producer has finished and its writes are visible, keeps it from
reading the poison.
"""

from __future__ import annotations

import pytest
import torch
from tokenspeed_kernel._triton import tl, triton
from tokenspeed_kernel.ops.attention.dsv41 import (
    compressor_pool,
    dspark_rows,
    dspark_workspace,
    global_rope_scatter,
    index_key_norm_rope,
    index_key_norm_rope_scatter,
    index_query_rope_quant,
    rope_pad_query,
    rope_quantize_fp8_grouped,
    swa_rope_scatter,
    verify_layout,
)
from tokenspeed_kernel.ops.attention.dsv41._deep_gemm.indexer import (
    clean_logits_and_block_max,
)
from tokenspeed_kernel.ops.attention.dsv41.triton import (
    block_maxima_with_lengths,
    clean_logits,
)
from tokenspeed_kernel.ops.moe._deep_gemm.mega_moe_stage import (
    stage_mxfp4_mega_moe_inputs,
)
from tokenspeed_kernel.ops.quantization.triton import (
    _quantize_fp8_group32_swizzled,
    triton_rmsnorm_fp8_group32_ue8m0,
)
from tokenspeed_kernel.ops.residual.triton import (
    mhc_pre_layer_norm_hc4,
    mhc_pre_mix_hc4,
    triton_mhc_post,
)
from tokenspeed_kernel.platform import pdl_enabled

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")

DEVICE = "cuda"
# Longer than the host takes to launch the kernel under test after the
# producer, including its wrapper's allocations, so a kernel missing its wait
# starts while the producer still spins and reads the poison.
_DELAY_NS = 500_000


@pytest.fixture(autouse=True)
def _require_pdl():
    if not pdl_enabled():
        pytest.skip("PDL is not enabled on this platform")


# ---------------------------------------------------------------------------
# Synthetic slow producer: release dependents, delay, then publish real bytes.
# ---------------------------------------------------------------------------


@triton.jit
def _pdl_spin_wait_ns(delay_ns: tl.constexpr):
    # A self-contained PTX busy loop on %globaltimer (nanoseconds since boot).
    # Written entirely in inline asm so the compiler cannot hoist, skip, or
    # shorten it: there is no data dependency for it to optimize away.
    tl.inline_asm_elementwise(
        f"""
        {{
            .reg .u64 %t0, %t1, %delta;
            .reg .pred %spin;
            mov.u64 %t0, %globaltimer;
        spin_loop:
            mov.u64 %t1, %globaltimer;
            sub.u64 %delta, %t1, %t0;
            setp.lt.u64 %spin, %delta, {delay_ns};
            @%spin bra spin_loop;
            mov.s32 $0, 0;
        }}
        """,
        "=r",
        [],
        dtype=tl.int32,
        is_pure=False,
        pack=1,
    )


@triton.jit
def _pdl_race_producer_kernel(
    src_ptr,
    dst_ptr,
    n_bytes,
    delay_ns: tl.constexpr,
    block: tl.constexpr,
):
    # The source comes from an ordinary kernel, so wait for it like any PDL
    # kernel, then let the kernel under test launch during the delay.
    tl.extra.cuda.gdc_wait()
    tl.extra.cuda.gdc_launch_dependents()
    pid = tl.program_id(0)
    offs = pid * block + tl.arange(0, block)
    mask = offs < n_bytes
    _pdl_spin_wait_ns(delay_ns)
    values = tl.load(src_ptr + offs, mask=mask, other=0)
    tl.store(dst_ptr + offs, values, mask=mask)


def _poison_like(real: torch.Tensor) -> torch.Tensor:
    """A same-shape buffer whose raw bytes differ from ``real``'s values."""
    poisoned = torch.empty_like(real)
    poisoned.view(torch.uint8).fill_(0xAB)
    return poisoned


def _publish_with_delay(real: torch.Tensor, dst: torch.Tensor, delay_ns: int) -> None:
    """Copy ``real``'s bytes into ``dst`` after a delay.

    The next kernel can launch before the copy lands.
    """
    assert real.shape == dst.shape and real.dtype == dst.dtype
    assert real.is_contiguous() and dst.is_contiguous()
    src_bytes = real.view(torch.uint8).reshape(-1)
    dst_bytes = dst.view(torch.uint8).reshape(-1)
    n = src_bytes.numel()
    block = 1024
    grid = (triton.cdiv(n, block),)
    _pdl_race_producer_kernel[grid](
        src_bytes,
        dst_bytes,
        n,
        delay_ns,
        block,
        num_warps=4,
        launch_pdl=True,
    )


def _assert_tuples_close(actual, expected) -> None:
    if not isinstance(actual, tuple):
        actual = (actual,)
    if not isinstance(expected, tuple):
        expected = (expected,)
    assert len(actual) == len(expected)
    for a, e in zip(actual, expected):
        torch.testing.assert_close(
            a.view(torch.uint8), e.view(torch.uint8), rtol=0, atol=0
        )


# ---------------------------------------------------------------------------
# One case per PDL-patched kernel: (real tensor to race, run(raced) -> tuple).
# run() launches only the kernel under test, so nothing separates the producer
# from it.
# ---------------------------------------------------------------------------


def _case_mhc_post():
    torch.manual_seed(101)
    tokens, hidden = 32, 5120
    real = torch.randn(tokens, hidden, dtype=torch.bfloat16, device=DEVICE)
    residual = torch.randn(tokens, 4, hidden, dtype=torch.bfloat16, device=DEVICE)
    post = torch.randn(tokens, 4, dtype=torch.float32, device=DEVICE)
    comb = torch.randn(tokens, 4, 4, dtype=torch.float32, device=DEVICE)

    def run(hidden_states):
        return (triton_mhc_post(hidden_states, residual, post, comb),)

    return real, run


def _case_mhc_pre_mix():
    torch.manual_seed(102)
    num_tokens, n_splits, hidden_size = 64, 1, 5120
    real = torch.randn(n_splits, num_tokens, 24, dtype=torch.float32, device=DEVICE)
    gemm_out_sqrsum = (
        torch.rand(n_splits, num_tokens, dtype=torch.float32, device=DEVICE) + 1.0
    )
    hc_scale = torch.rand(3, dtype=torch.float32, device=DEVICE) + 0.5
    hc_base = torch.randn(24, dtype=torch.float32, device=DEVICE)

    def run(gemm_out_mul):
        pre_mix = torch.empty(num_tokens, 4, dtype=torch.float32, device=DEVICE)
        post_mix = torch.empty(num_tokens, 4, dtype=torch.float32, device=DEVICE)
        comb_mix = torch.empty(num_tokens, 16, dtype=torch.float32, device=DEVICE)
        mhc_pre_mix_hc4(
            gemm_out_mul,
            gemm_out_sqrsum,
            hc_scale,
            hc_base,
            pre_mix,
            post_mix,
            comb_mix,
            hidden_size=hidden_size,
            rms_eps=1e-6,
            hc_eps=1e-5,
            sinkhorn_iters=3,
            n_splits=n_splits,
            num_tokens=num_tokens,
        )
        return (pre_mix, post_mix, comb_mix)

    return real, run


def _case_mhc_pre_layer_norm():
    torch.manual_seed(103)
    tokens, hidden = 32, 5120
    real = torch.randn(tokens, 4, hidden, dtype=torch.bfloat16, device=DEVICE)
    pre_mix = torch.rand(tokens, 4, dtype=torch.float32, device=DEVICE) + 0.1
    weight = torch.randn(hidden, dtype=torch.bfloat16, device=DEVICE)

    def run(residual):
        out = torch.empty(tokens, hidden, dtype=torch.bfloat16, device=DEVICE)
        mhc_pre_layer_norm_hc4(pre_mix, residual, weight, out, eps=1e-6)
        return (out,)

    return real, run


def _case_quantize_fp8_group32_swizzled():
    torch.manual_seed(105)
    rows, hidden = 128, 5120
    real = torch.randn(rows, hidden, dtype=torch.bfloat16, device=DEVICE)

    def run(x):
        return _quantize_fp8_group32_swizzled(x)

    return real, run


def _case_rmsnorm_fp8_group32_swizzled():
    torch.manual_seed(106)
    rows, hidden = 32, 1536
    real = torch.randn(rows, hidden, dtype=torch.bfloat16, device=DEVICE)
    weight = torch.randn(hidden, dtype=torch.bfloat16, device=DEVICE)

    def run(x):
        return triton_rmsnorm_fp8_group32_ue8m0(
            x, weight, 1e-6, scale_layout="swizzled"
        )

    return real, run


def _case_rope_quantize_fp8_grouped():
    torch.manual_seed(107)
    # 37 tokens: programs past T (aligned to 4) only zero padding scale rows,
    # and they must wait too.
    tokens, heads = 37, 64
    real = torch.randn(tokens, heads, 512, dtype=torch.bfloat16, device=DEVICE)
    positions = torch.arange(tokens, device=DEVICE) * 3
    angles = torch.arange(4096, device=DEVICE)[:, None].float() * torch.exp(
        -torch.arange(32, device=DEVICE).float() / 4
    )
    cache = torch.cat((angles.cos(), -angles.sin()), dim=-1)

    def run(values):
        codes, scales = rope_quantize_fp8_grouped(values, positions, cache, 8)
        return (codes, scales.contiguous())

    return real, run


def _case_stage_mxfp4_mega_moe_inputs():
    torch.manual_seed(108)
    tokens, hidden, top_k = 64, 5120, 6
    real = torch.randn(tokens, hidden, dtype=torch.bfloat16, device=DEVICE)
    weights = torch.rand(tokens, top_k, dtype=torch.float32, device=DEVICE)
    ids = torch.randint(0, 384, (tokens, top_k), dtype=torch.int32, device=DEVICE)

    def run(hidden_states):
        out = (
            torch.empty(tokens, hidden, dtype=torch.float8_e4m3fn, device=DEVICE),
            torch.empty(tokens, hidden // 128, dtype=torch.int32, device=DEVICE),
            torch.empty(tokens, top_k, dtype=torch.int64, device=DEVICE),
            torch.empty(tokens, top_k, dtype=torch.float32, device=DEVICE),
        )
        stage_mxfp4_mega_moe_inputs(hidden_states, weights, ids, *out, None, None, None)
        return out

    return real, run


def _dsv41_cos_sin_cache(rotary_dim: int = 64) -> torch.Tensor:
    angles = torch.arange(4096, device=DEVICE)[:, None].float() * torch.exp(
        -torch.arange(rotary_dim // 2, device=DEVICE).float() / 4
    )
    return torch.cat((angles.cos(), -angles.sin()), dim=-1)


def _case_dsv41_index_key_norm_rope():
    torch.manual_seed(109)
    tokens = 32
    real = torch.randn(tokens, 128, dtype=torch.bfloat16, device=DEVICE)
    weight = torch.randn(128, dtype=torch.bfloat16, device=DEVICE)
    positions = torch.arange(tokens, device=DEVICE) * 3
    cache = _dsv41_cos_sin_cache()

    def run(values):
        return (index_key_norm_rope(values, weight, positions, cache, 1e-6),)

    return real, run


def _case_dsv41_index_key_norm_rope_scatter():
    torch.manual_seed(110)
    tokens = 64
    real = torch.randn(tokens, 128, dtype=torch.bfloat16, device=DEVICE)
    weight = torch.randn(128, dtype=torch.bfloat16, device=DEVICE)
    positions = torch.arange(tokens, device=DEVICE)
    slots = torch.arange(tokens, dtype=torch.int32, device=DEVICE)
    cache = _dsv41_cos_sin_cache()

    def run(values):
        field = torch.empty(1, 64, 68, dtype=torch.uint8, device=DEVICE)
        index_key_norm_rope_scatter(
            values, weight, positions, cache, 1e-6, field, slots
        )
        return (field,)

    return real, run


def _case_dsv41_index_query_rope_quant():
    torch.manual_seed(111)
    tokens, heads = 16, 64
    real = torch.randn(tokens, heads, 128, dtype=torch.bfloat16, device=DEVICE)
    head_weights = torch.randn(tokens, heads, dtype=torch.bfloat16, device=DEVICE)
    positions = torch.arange(tokens, device=DEVICE) * 2
    cache = _dsv41_cos_sin_cache()

    def run(query):
        (codes, scales), scaled_weights = index_query_rope_quant(
            query, head_weights, positions, cache, 0.25, torch.float32
        )
        return (codes, scales, scaled_weights)

    return real, run


def _case_dsv41_global_rope_scatter():
    torch.manual_seed(112)
    tokens = 64
    real = torch.randn(tokens, 512, dtype=torch.bfloat16, device=DEVICE)
    positions = torch.arange(tokens, device=DEVICE)
    slots = torch.arange(tokens, dtype=torch.int32, device=DEVICE)
    cache = _dsv41_cos_sin_cache()

    def run(values):
        field = torch.empty(1, 64, 288, dtype=torch.uint8, device=DEVICE)
        global_rope_scatter(values, positions, cache, field, slots)
        return (field,)

    return real, run


def _case_dsv41_swa_rope_scatter():
    torch.manual_seed(113)
    tokens = 64
    real = torch.randn(tokens, 512, dtype=torch.bfloat16, device=DEVICE)
    norm_weight = torch.randn(512, dtype=torch.bfloat16, device=DEVICE)
    positions = torch.arange(tokens, device=DEVICE)
    slots = torch.arange(64, 128, dtype=torch.int32, device=DEVICE)
    cache = _dsv41_cos_sin_cache()

    def run(values):
        field = torch.empty(2, 64, 528, dtype=torch.uint8, device=DEVICE)
        out = torch.empty(tokens, 512, dtype=torch.bfloat16, device=DEVICE)
        swa_rope_scatter(
            values, positions, cache, field, slots, "swa", out, norm_weight, 1e-6
        )
        return (out, field[1].contiguous())

    return real, run


def _case_dsv41_compressor_pool():
    torch.manual_seed(114)
    tokens = 32
    real = torch.randn(tokens, 512, dtype=torch.float32, device=DEVICE)
    scores = torch.randn(tokens, 512, dtype=torch.float32, device=DEVICE)
    previous = torch.arange(-1, tokens - 1, dtype=torch.int32, device=DEVICE)
    tail = torch.randn(4, 2, 2, 512, dtype=torch.float32, device=DEVICE)
    tail_slots = torch.full((tokens,), 2, dtype=torch.int32, device=DEVICE)
    active = torch.ones(tokens, dtype=torch.bool, device=DEVICE)
    norm_weight = torch.randn(512, dtype=torch.bfloat16, device=DEVICE)

    def run(content):
        out = torch.empty(tokens, 512, dtype=torch.bfloat16, device=DEVICE)
        compressor_pool(
            content, scores, previous, tail, tail_slots, active, out, norm_weight, 1e-6
        )
        return (out,)

    return real, run


def _case_dsv41_dspark_rows():
    torch.manual_seed(115)
    tokens = 32
    real = torch.randn(tokens, 512, dtype=torch.bfloat16, device=DEVICE)
    norm_weight = torch.randn(512, dtype=torch.bfloat16, device=DEVICE)
    positions = torch.arange(tokens, device=DEVICE)
    cache = _dsv41_cos_sin_cache()

    def run(values):
        out = torch.empty(tokens, 512, dtype=torch.bfloat16, device=DEVICE)
        dspark_rows(values, norm_weight, 1e-6, positions, cache, None, None, out)
        return (out,)

    return real, run


def _case_dsv41_dspark_workspace():
    torch.manual_seed(116)
    batch, history, block_size = 2, 8, 6
    real = torch.randn(batch * block_size, 512, dtype=torch.bfloat16, device=DEVICE)
    norm_weight = torch.randn(512, dtype=torch.bfloat16, device=DEVICE)
    positions = torch.arange(batch * block_size, device=DEVICE)
    cache = _dsv41_cos_sin_cache()
    window = torch.randn(2, 64, 512, dtype=torch.bfloat16, device=DEVICE)
    page = torch.ones(batch, history, dtype=torch.int64, device=DEVICE)
    row = (
        torch.arange(history, dtype=torch.int64, device=DEVICE)
        .unsqueeze(0)
        .expand(batch, -1)
        .contiguous()
    )

    def run(values):
        out = dspark_workspace(
            values, norm_weight, 1e-6, positions, cache, window, page, row, block_size
        )
        return (out,)

    return real, run


def _case_dsv41_rope_pad_query():
    torch.manual_seed(117)
    tokens, heads = 16, 48
    real = torch.randn(tokens, heads, 512, dtype=torch.bfloat16, device=DEVICE)
    positions = torch.arange(tokens, device=DEVICE)
    cache = _dsv41_cos_sin_cache()

    def run(values):
        return (rope_pad_query(values, positions, cache, None),)

    return real, run


def _case_dsv41_clean_logits():
    torch.manual_seed(118)
    rows, width = 8, 1024
    real = torch.randn(rows, width, dtype=torch.float32, device=DEVICE)
    lengths = torch.full((rows,), 517, dtype=torch.int32, device=DEVICE)
    table = (
        torch.arange(16, dtype=torch.int32, device=DEVICE)
        .remainder(3)
        .unsqueeze(0)
        .expand(rows, -1)
        .contiguous()
    )

    def run(values):
        return (clean_logits(values, lengths, table, 3, width, 64),)

    return real, run


def _case_dsv41_clean_logits_block_max():
    torch.manual_seed(119)
    rows, width = 8, 1024
    real = torch.randn(rows, width, dtype=torch.float32, device=DEVICE)
    lengths = torch.full((rows,), 517, dtype=torch.int32, device=DEVICE)
    table = (
        torch.arange(16, dtype=torch.int32, device=DEVICE)
        .remainder(3)
        .unsqueeze(0)
        .expand(rows, -1)
        .contiguous()
    )

    def run(values):
        return clean_logits_and_block_max(values, lengths, table, 3, width, 64, 512)

    return real, run


def _case_dsv41_block_maxima():
    torch.manual_seed(120)
    rows, width = 8, 1024
    real = torch.randn(rows, width, dtype=torch.float32, device=DEVICE)
    lengths = torch.full((rows,), 517, dtype=torch.int32, device=DEVICE)

    def run(values):
        return block_maxima_with_lengths(values, lengths)

    return real, run


def _case_dsv41_verify_layout():
    torch.manual_seed(121)
    bs, draft_width, capacity = 16, 5, 48
    real = torch.randn(bs, draft_width, dtype=torch.float32, device=DEVICE)

    def run(scores):
        rows_out = torch.empty(capacity, dtype=torch.int64, device=DEVICE)
        indptr_out = torch.empty(bs + 1, dtype=torch.int64, device=DEVICE)
        counts_out = torch.empty(bs, dtype=torch.int64, device=DEVICE)
        valid_out = torch.empty((), dtype=torch.bool, device=DEVICE)
        verify_layout(
            scores,
            query_capacity=capacity,
            rows_out=rows_out,
            indptr_out=indptr_out,
            counts_out=counts_out,
            valid_out=valid_out,
            latest_scores=None,
            request_slots=None,
            active_count=None,
        )
        return (rows_out, indptr_out, counts_out, valid_out)

    return real, run


_CASES = {
    "mhc_post_hc4": _case_mhc_post,
    "mhc_pre_mix_hc4": _case_mhc_pre_mix,
    "mhc_pre_layer_norm_hc4": _case_mhc_pre_layer_norm,
    "fp8_group32_ue8m0_quantize_swizzled": _case_quantize_fp8_group32_swizzled,
    "rmsnorm_fp8_group32_ue8m0": _case_rmsnorm_fp8_group32_swizzled,
    "dsv41_rope_quantize_fp8_grouped": _case_rope_quantize_fp8_grouped,
    "stage_mxfp4_mega_moe_inputs": _case_stage_mxfp4_mega_moe_inputs,
    "dsv41_index_key_norm_rope": _case_dsv41_index_key_norm_rope,
    "dsv41_index_key_norm_rope_scatter": _case_dsv41_index_key_norm_rope_scatter,
    "dsv41_index_query_rope_quant": _case_dsv41_index_query_rope_quant,
    "dsv41_global_rope_scatter": _case_dsv41_global_rope_scatter,
    "dsv41_swa_rope_scatter": _case_dsv41_swa_rope_scatter,
    "dsv41_compressor_pool": _case_dsv41_compressor_pool,
    "dsv41_dspark_rows": _case_dsv41_dspark_rows,
    "dsv41_dspark_workspace": _case_dsv41_dspark_workspace,
    "dsv41_rope_pad_query": _case_dsv41_rope_pad_query,
    "dsv41_clean_logits": _case_dsv41_clean_logits,
    "dsv41_clean_logits_block_max": _case_dsv41_clean_logits_block_max,
    "dsv41_block_maxima": _case_dsv41_block_maxima,
    "dsv41_verify_layout": _case_dsv41_verify_layout,
}


@pytest.mark.parametrize("case_name", list(_CASES))
def test_pdl_kernel_waits_for_upstream_write(case_name):
    real, run = _CASES[case_name]()

    # Reference first: it also compiles the kernel, so the raced launch below
    # reaches the GPU while the producer is still spinning.
    expected = run(real.clone())
    torch.cuda.synchronize()

    # Race path: no host sync between the producer and the kernel under test.
    race_buf = _poison_like(real)
    _publish_with_delay(real, race_buf, _DELAY_NS)
    actual = run(race_buf)
    torch.cuda.synchronize()

    _assert_tuples_close(actual, expected)


@pytest.mark.parametrize("case_name", list(_CASES))
def test_pdl_kernel_waits_for_upstream_write_under_cuda_graph(case_name):
    real, run = _CASES[case_name]()

    # CUDA graphs require a warm-up launch outside the capture on buffers
    # that are not reused by the capture itself.
    warmup_real = real.clone()
    warmup_buf = _poison_like(warmup_real)
    _publish_with_delay(warmup_real, warmup_buf, _DELAY_NS)
    run(warmup_buf)
    torch.cuda.synchronize()

    source = real.clone()
    race_buf = _poison_like(real)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        race_buf.view(torch.uint8).fill_(0xAB)
        _publish_with_delay(source, race_buf, _DELAY_NS)
        captured = run(race_buf)

    for _ in range(3):
        source.normal_()
        graph.replay()
        torch.cuda.synchronize()
        actual = tuple(tensor.clone() for tensor in captured)

        expected = run(source.clone())
        torch.cuda.synchronize()

        _assert_tuples_close(actual, expected)
