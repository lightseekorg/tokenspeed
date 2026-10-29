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

from __future__ import annotations

import math

import pytest
import torch
from tokenspeed_kernel.numerics.torch.attention.mha import torch_dense_attention
from utils import assert_no_triton_compile, is_cdna5

if not is_cdna5():
    pytest.skip(
        "AMD CDNA5 is required for gfx1250 Gluon MHA tests", allow_module_level=True
    )


from tokenspeed_kernel_amd.ops.gfx1250.attention.mha import prefill  # noqa: E402


def _inputs(seqlens, n_q_heads, n_kv_heads, head_dim, device, dtype):
    cu_cpu = [0]
    for s in seqlens:
        cu_cpu.append(cu_cpu[-1] + s)
    cu = torch.tensor(cu_cpu, device=device, dtype=torch.int32)
    total = cu_cpu[-1]
    q = torch.randn((total, n_q_heads, head_dim), device=device, dtype=dtype)
    k = torch.randn((total, n_kv_heads, head_dim), device=device, dtype=dtype)
    v = torch.randn((total, n_kv_heads, head_dim), device=device, dtype=dtype)
    return q, k, v, cu, cu_cpu, max(seqlens)


def _reference(q, k, v, cu_cpu, n_q_heads, n_kv_heads, head_dim, window_left=-1):
    outs = []
    for start, end in zip(cu_cpu[:-1], cu_cpu[1:]):
        out, _ = torch_dense_attention(
            q[start:end],
            k[start:end],
            v[start:end],
            scale=1.0 / math.sqrt(head_dim),
            causal_offset=0,
            window_left=window_left if window_left >= 0 else None,
            sinks=None,
        )
        outs.append(out)
    return torch.cat(outs, dim=0)


@pytest.mark.parametrize(
    "block_m,num_warps", [(128, 4), (256, 8)], ids=["narrow", "wide"]
)
@pytest.mark.parametrize("head_dim", [64, 128], ids=["d64", "d128"])
@pytest.mark.parametrize("window_left", [-1, 64], ids=["full", "sliding"])
def test_mha_prefill_tile_shapes(block_m, num_warps, head_dim, window_left):
    """Both WMMA layouts produce correct causal and windowed attention."""
    q, k, v, cu, cu_cpu, maximum = _inputs(
        [513, 129], 4, 1, head_dim, "cuda", torch.bfloat16
    )
    out = _launch_tile(
        q,
        k,
        v,
        cu,
        cu_cpu,
        maximum,
        block_m=block_m,
        num_warps=num_warps,
        window_left=window_left,
        packed_gqa=False,
        tdm_warp_hint=False,
        reverse_q_blocks=False,
    )
    expected = _reference(q, k, v, cu_cpu, 4, 1, head_dim, window_left)
    torch.testing.assert_close(out.float(), expected, rtol=2e-2, atol=2e-2)


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
@pytest.mark.parametrize("head_dim", [64, 128])
def test_mha_prefill_long_prefill(dtype, head_dim):
    """Long prefill remains correct through automatic launch selection."""
    seqlens = [2048] * 4 if head_dim == 64 else [1024] * 8
    q, k, v, cu, cu_cpu, maximum = _inputs(seqlens, 8, 2, head_dim, "cuda", dtype)
    out = prefill.launch_gluon_mha_prefill_gfx1250(q, k, v, cu, cu_cpu, maximum)
    _check_sampled_fp64(out, q, k, v, cu_cpu)


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
@pytest.mark.parametrize("group_size", [1, 2, 4, 8, 32])
def test_mha_prefill_packed_gqa(dtype, group_size):
    """Packed query rows preserve attention across all supported group sizes."""
    q, k, v, cu, cu_cpu, maximum = _inputs(
        [257, 129], group_size, 1, 128, "cuda", dtype
    )
    out = _launch_tile(
        q,
        k,
        v,
        cu,
        cu_cpu,
        maximum,
        block_m=128,
        num_warps=4,
        window_left=-1,
        packed_gqa=True,
        tdm_warp_hint=False,
        reverse_q_blocks=False,
    )
    _check_sampled_fp64(out, q, k, v, cu_cpu)


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
def test_mha_prefill_packed_window(dtype):
    """The selected packed schedule preserves both edges of a sliding window."""
    torch.manual_seed(41)
    q, k, v, cu, cu_cpu, maximum = _inputs([4096] * 4, 8, 1, 128, "cuda", dtype)
    out = prefill.launch_gluon_mha_prefill_gfx1250(
        q, k, v, cu, cu_cpu, maximum, window_left=512
    )
    _check_sampled_fp64(out, q, k, v, cu_cpu, window_left=512)


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
def test_mha_prefill_wide_transport(dtype):
    """The public launcher computes both BN64 halves of each BN128 transfer."""
    torch.manual_seed(42)
    q, k, v, cu, cu_cpu, maximum = _inputs([1024] * 8, 8, 1, 128, "cuda", dtype)
    out = prefill.launch_gluon_mha_prefill_gfx1250(q, k, v, cu, cu_cpu, maximum)
    _check_sampled_fp64(out, q, k, v, cu_cpu)


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
@pytest.mark.parametrize("fused_tdm", [False, True], ids=["separate", "fused"])
def test_mha_prefill_wide_transport_small(dtype, fused_tdm):
    """Direct launch covers both transfer variants within the simulator budget."""
    torch.manual_seed(75)
    # Five BN64 tiles cover the prologue, loop, and odd/even transfer tails.
    q, k, v, cu, cu_cpu, maximum = _inputs([320], 2, 1, 128, "cuda", dtype)
    out = torch.full_like(q, float("nan"))
    prefill.gluon_mha_prefill_gfx1250_wide_transport[(1, 1, 5)](
        q,
        k,
        v,
        cu,
        out,
        Q_STRIDE_T=q.stride(0),
        Q_STRIDE_H=q.stride(1),
        Q_STRIDE_D=q.stride(2),
        K_STRIDE_T=k.stride(0),
        K_STRIDE_H=k.stride(1),
        K_STRIDE_D=k.stride(2),
        V_STRIDE_T=v.stride(0),
        V_STRIDE_H=v.stride(1),
        V_STRIDE_D=v.stride(2),
        N_HEADS=2,
        N_KV_HEADS=1,
        HEAD_DIM=128,
        SM_SCALE=128**-0.5 * prefill._INV_LN2_VALUE,
        NUM_WARPS=4,
        NUM_BUFFERS=2,
        FUSED_TDM=fused_tdm,
        num_warps=4,
        waves_per_eu=1,
        llvm_fn_attrs="amdgpu-sched-strategy=coexec",
        enable_fp_fusion=True,
        **prefill.sched_compile_options(),
    )
    expected = _reference(q.double(), k.double(), v.double(), cu_cpu, 2, 1, 128)
    tolerance = 2e-2 if dtype == torch.bfloat16 else 3e-3
    torch.testing.assert_close(out.double(), expected, atol=tolerance, rtol=tolerance)
    _check_sampled_fp64(out, q, k, v, cu_cpu)


def test_mha_prefill_addresses_past_four_gib():
    """Sequence and head bases must not wrap through 32-bit buffer offsets."""
    device, dtype = "cuda", torch.bfloat16
    tokens, n_heads, head_dim = 2, 17, 64
    head_stride = 2**27
    boundary = 2**31
    storage_elements = boundary + tokens * head_dim
    storage_bytes = storage_elements * torch.tensor([], dtype=dtype).element_size()
    free_bytes, _ = torch.cuda.mem_get_info(device)
    if free_bytes < storage_bytes + 512 * 1024**2:
        pytest.skip("large-offset regression requires about 4.5 GiB free")

    storage = torch.empty(storage_elements, dtype=dtype, device=device)
    qkv = storage.as_strided(
        (tokens, n_heads, head_dim),
        (head_dim, head_stride, 1),
    )
    generator = torch.Generator(device=device).manual_seed(20261002)
    compact = torch.randn(qkv.shape, dtype=dtype, device=device, generator=generator)
    qkv.copy_(compact)
    cu_cpu = [0, tokens]
    cu = torch.tensor(cu_cpu, dtype=torch.int32, device=device)
    out = prefill.launch_gluon_mha_prefill_gfx1250(qkv, qkv, qkv, cu, cu_cpu, tokens)
    _check_sampled_fp64(out, compact, compact, compact, cu_cpu)

    # Exercise a sequence/output base at exactly 4 GiB with one sparse allocation.
    base_token = boundary // head_dim
    sentinel = torch.full((head_dim,), -7.0, dtype=dtype, device=device)
    value = torch.randn((head_dim,), dtype=dtype, device=device, generator=generator)
    storage[:head_dim].copy_(sentinel)
    storage[boundary : boundary + head_dim].copy_(value)
    large_cu = torch.tensor(
        [base_token, base_token + 1], dtype=torch.int32, device=device
    )
    prefill.gluon_mha_prefill_gfx1250[(1, 1, 1)](
        storage,
        storage,
        storage,
        large_cu,
        storage,
        storage,
        storage,
        Q_STRIDE_T=head_dim,
        Q_STRIDE_H=head_dim,
        Q_STRIDE_D=1,
        K_STRIDE_T=head_dim,
        K_STRIDE_H=head_dim,
        K_STRIDE_D=1,
        V_STRIDE_T=head_dim,
        V_STRIDE_H=head_dim,
        V_STRIDE_D=1,
        N_HEADS=1,
        N_KV_HEADS=1,
        HEAD_DIM=head_dim,
        SM_SCALE=(1.0 / math.sqrt(head_dim)) * prefill._INV_LN2_VALUE,
        BLOCK_M=128,
        BLOCK_N=64,
        IS_FP8=False,
        HAS_SINK=False,
        HAS_LSE=False,
        WINDOW_LEFT=-1,
        TDM_WARP_HINT=False,
        REVERSE_Q_BLOCKS=False,
        DEEP_PIPELINE=False,
        PACKED_GQA=False,
        GUARDED_QUERY_ROWS=False,
        COMPACT_RAGGED=False,
        WIDE_ADDRESSING=True,
        QK_VALU_GROUP_SIZE=0,
        QK_DS_GROUP_STRIDE=0,
        NUM_WARPS=4,
        NUM_BUFFERS=2,
        ONE_KV_TILE=True,
        TWO_KV_TILES=True,
        USE_TDM_Q=False,
        uniform_seqlen=1,
        UNIFORM_SEQLENS=False,
        num_warps=4,
        waves_per_eu=1,
        llvm_fn_attrs="",
        **prefill.sched_compile_options(),
    )
    torch.cuda.synchronize()
    assert torch.equal(storage[:head_dim], sentinel)
    assert torch.equal(storage[boundary : boundary + head_dim], value)


def test_mha_prefill_reverse_counts_live_ragged_workgroups():
    """A long request alongside one-token requests preserves each boundary."""
    q, k, v, cu, cu_cpu, maximum = _inputs(
        [4096] + [1] * 31, 1, 1, 64, "cuda", torch.bfloat16
    )
    out = prefill.launch_gluon_mha_prefill_gfx1250(q, k, v, cu, cu_cpu, maximum)
    _check_sampled_fp64(out, q, k, v, cu_cpu)


# The two-head case is the same schedule at a size the MI450 simulator affords.
@pytest.mark.parametrize("n_q_heads,n_kv_heads", [(8, 2), (2, 1)], ids=["gqa4", "sim"])
def test_mha_prefill_tdm_warp_hint_remainder(n_q_heads, n_kv_heads):
    """Changing scheduling preserves outputs on partial query blocks."""
    q, k, v, cu, cu_cpu, maximum = _inputs(
        [300, 513], n_q_heads, n_kv_heads, 128, "cuda", torch.bfloat16
    )
    options = dict(
        block_m=256,
        num_warps=8,
        window_left=-1,
        packed_gqa=False,
        tdm_warp_hint=False,
        reverse_q_blocks=False,
    )
    control = _launch_tile(q, k, v, cu, cu_cpu, maximum, **options)
    out = _launch_tile(
        q, k, v, cu, cu_cpu, maximum, **(options | {"tdm_warp_hint": True})
    )
    torch.testing.assert_close(out, control, rtol=0, atol=0)
    _check_sampled_fp64(out, q, k, v, cu_cpu)


# The two-head case is the same schedule at a size the MI450 simulator affords.
@pytest.mark.parametrize("n_q_heads,n_kv_heads", [(8, 2), (2, 1)], ids=["gqa4", "sim"])
def test_mha_prefill_reverse_q_blocks_ragged(n_q_heads, n_kv_heads):
    """Changing scheduling preserves outputs on partial query blocks."""
    q, k, v, cu, cu_cpu, maximum = _inputs(
        [300, 513], n_q_heads, n_kv_heads, 128, "cuda", torch.bfloat16
    )
    options = dict(
        block_m=256,
        num_warps=8,
        window_left=-1,
        packed_gqa=False,
        tdm_warp_hint=False,
        reverse_q_blocks=False,
    )
    control = _launch_tile(q, k, v, cu, cu_cpu, maximum, **options)
    out = _launch_tile(
        q, k, v, cu, cu_cpu, maximum, **(options | {"reverse_q_blocks": True})
    )
    torch.testing.assert_close(out, control, rtol=0, atol=0)
    _check_sampled_fp64(out, q, k, v, cu_cpu)


def _offset_inputs(*tensors):
    """Contiguous tensors whose first element is offset by two bytes."""
    return tuple(
        torch.empty(t.numel() + 1, dtype=t.dtype, device=t.device)[1:]
        .view_as(t)
        .copy_(t)
        for t in tensors
    )


def _check_graph_replays(graph, output, q, k, v, cu_cpu, *, repeats):
    output.fill_(float("nan"))
    graph.replay()
    _check_sampled_fp64(output, q, k, v, cu_cpu)
    expected = output.clone()
    for _ in range(repeats):
        output.fill_(float("nan"))
        graph.replay()
        torch.testing.assert_close(output, expected, rtol=0, atol=0)


def _check_sampled_fp64(out, q, k, v, cu_cpu, *, sinks=None, lse=None, window_left=-1):
    # Sample complete chunks at causal/tile boundaries without materializing a
    # quadratic FP64 score tensor for every query in a long prefill.
    assert torch.isfinite(out).all()
    tolerance = 2e-2 if q.dtype == torch.bfloat16 else 3e-3
    for start, end in zip(cu_cpu[:-1], cu_cpu[1:]):
        length = end - start
        if length == 0:
            continue
        for row in sorted(
            {0, min(63, length - 1), min(127, length - 1), max(0, length - 33)}
        ):
            stop = min(length, row + 33)
            expected, expected_lse = torch_dense_attention(
                q[start + row : start + stop].double(),
                k[start:end].double(),
                v[start:end].double(),
                scale=1.0 / math.sqrt(q.shape[-1]),
                causal_offset=row,
                window_left=window_left if window_left >= 0 else None,
                sinks=None if sinks is None else sinks.double(),
            )
            actual = out[start + row : start + stop].double()
            torch.testing.assert_close(actual, expected, atol=tolerance, rtol=tolerance)
            rms_tolerance = 4e-3 if q.dtype == torch.bfloat16 else 8e-4
            error_norm = torch.linalg.vector_norm(actual - expected)
            reference_norm = torch.linalg.vector_norm(expected)
            assert error_norm <= rms_tolerance * reference_norm + 1e-12

            if lse is not None:
                torch.testing.assert_close(
                    lse[start + row : start + stop].double(),
                    expected_lse,
                    atol=3e-5,
                    rtol=3e-5,
                )


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
@pytest.mark.parametrize(
    "n_q_heads,n_kv_heads,tails",
    [
        (8, 8, [31, 32, 33]),
        (32, 8, [7, 8, 9, 31, 32, 33]),
        (8, 1, [3, 4, 5, 31, 32, 33]),
        (32, 1, [1, 2, 31, 32, 33]),
        (12, 4, [1, 31, 32, 33]),
    ],
    ids=["mha", "gqa4", "gqa8", "mqa32", "gqa3"],
)
def test_mha_prefill_small_query_tails(dtype, n_q_heads, n_kv_heads, tails):
    """All heads must be written at both small-tail routing boundaries."""
    torch.manual_seed(47)
    q, k, v, cu, cu_cpu, maximum = _inputs(
        [4096 + tail for tail in tails] + [0],
        n_q_heads,
        n_kv_heads,
        128,
        "cuda",
        dtype,
    )

    q, k, v = _offset_inputs(q, k, v)
    saved = [tensor.clone() for tensor in (q, k, v)]
    prefill.launch_gluon_mha_prefill_gfx1250(q, k, v, cu, cu_cpu, maximum)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        output = prefill.launch_gluon_mha_prefill_gfx1250(q, k, v, cu, cu_cpu, maximum)

    # Retry decisions and the packed tail's head ownership must remain correct
    # when values change in the same capture, including a return to normal data.
    for scale in (1, 8, 1):
        for tensor, original in zip((q, k, v), saved):
            tensor.copy_(original)
        q.mul_(scale)
        k.mul_(scale)
        _check_graph_replays(graph, output, q, k, v, cu_cpu, repeats=20)


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
def test_mha_prefill_query_tail_lengths_do_not_recompile(dtype):
    """Length changes select full, small, or packed tiles within one binary."""
    torch.manual_seed(48)
    shapes = [
        [4097, 3500, 3000],
        [4104, 3501, 3001, 0],
        [4105, 3503, 3003],
        [4128, 3499, 2999],
        [4129, 3498, 2998],
    ]
    for index, seqlens in enumerate(shapes):
        q, k, v, cu, cu_cpu, maximum = _inputs(seqlens, 32, 8, 128, "cuda", dtype)
        if index == 0:
            out = prefill.launch_gluon_mha_prefill_gfx1250(q, k, v, cu, cu_cpu, maximum)
        else:
            with assert_no_triton_compile(prefill.gluon_mha_prefill_gfx1250):
                out = prefill.launch_gluon_mha_prefill_gfx1250(
                    q, k, v, cu, cu_cpu, maximum
                )
        _check_sampled_fp64(out, q, k, v, cu_cpu)


@pytest.mark.parametrize("n_q_heads,n_kv_heads", [(8, 1), (8, 8), (32, 8), (32, 1)])
@pytest.mark.parametrize("input_scale", [1.0, 8.0])
def test_mha_prefill_fp16_full_tiles(n_q_heads, n_kv_heads, input_scale):
    # Exercise automatic selection, including the different packed group sizes.
    torch.manual_seed(42)
    q, k, v, cu, cu_cpu, maximum = _inputs(
        [4096] * 4, n_q_heads, n_kv_heads, 128, "cuda", torch.float16
    )
    q.mul_(input_scale)
    k.mul_(input_scale)
    out = prefill.launch_gluon_mha_prefill_gfx1250(q, k, v, cu, cu_cpu, maximum)
    _check_sampled_fp64(out, q, k, v, cu_cpu)


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
@pytest.mark.parametrize(
    "seqlens",
    [[851, 914, 1053], [1053] * 4, [1152, 1025, 1, 0], [1153, 1025, 1]],
    ids=["ragged", "uniform-tail", "cutoff-empty", "above-cutoff"],
)
def test_mha_prefill_short_partial_tiles(dtype, seqlens):
    torch.manual_seed(43)
    q, k, v, cu, cu_cpu, maximum = _inputs(seqlens, 32, 8, 128, "cuda", dtype)
    out = prefill.launch_gluon_mha_prefill_gfx1250(q, k, v, cu, cu_cpu, maximum)
    _check_sampled_fp64(out, q, k, v, cu_cpu)


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
@pytest.mark.parametrize("input_scale", [1.0, 8.0])
@pytest.mark.parametrize(
    "seqlens,n_q_heads,n_kv_heads",
    [
        ([0, 1, 63, 64, 65, 127, 128, 513], 8, 1),
        ([511, 385, 129], 32, 8),
        ([512, 385, 129], 32, 8),
        ([1536, 1200, 1000], 32, 8),
        ([4096, 3500, 3000], 32, 8),
        ([8193, 4097, 17, 0], 8, 1),
        ([16384, 12289, 8193], 32, 8),
    ],
    ids=[
        "short-tail-empty",
        "below-minimum",
        "minimum",
        "medium",
        "long",
        "long-tail",
        "16k-ragged",
    ],
)
def test_mha_prefill_guarded_pipeline(
    dtype, input_scale, seqlens, n_q_heads, n_kv_heads
):
    # Cover the one/two-tile fallback and the steady-state loop with independent
    # FP64 rows, including sharp softmax distributions and both query tile sizes.
    torch.manual_seed(45)
    q, k, v, cu, cu_cpu, maximum = _inputs(
        seqlens, n_q_heads, n_kv_heads, 128, "cuda", dtype
    )
    q.mul_(input_scale)
    k.mul_(input_scale)
    out = prefill.launch_gluon_mha_prefill_gfx1250(q, k, v, cu, cu_cpu, maximum)
    _check_sampled_fp64(out, q, k, v, cu_cpu)


@pytest.mark.parametrize(
    "seqlens,n_q_heads,n_kv_heads,dtype",
    [
        ([2048] * 4, 8, 1, torch.float16),
        ([4096] * 4, 8, 1, torch.float16),
        ([8192] * 2, 8, 1, torch.float16),
        ([4096] * 4, 8, 8, torch.float16),
        ([4096] * 4, 32, 8, torch.float16),
        ([4096] * 4, 32, 1, torch.float16),
        ([851, 914, 1053], 32, 8, torch.float16),
        ([4096, 3500, 3000], 32, 8, torch.float16),
        ([4096, 3500, 3000], 32, 8, torch.bfloat16),
        ([0, 1, 63, 64, 65, 127, 128, 513], 8, 1, torch.float16),
        ([0, 1, 63, 64, 65, 127, 128, 513], 8, 1, torch.bfloat16),
    ],
    ids=[
        "medium",
        "dense",
        "long",
        "mha",
        "gqa4",
        "mqa",
        "ragged",
        "guarded-long-fp16",
        "guarded-long-bf16",
        "guarded-short-fp16",
        "guarded-short-bf16",
    ],
)
def test_mha_prefill_storage_offset_repeated(seqlens, n_q_heads, n_kv_heads, dtype):
    torch.manual_seed(44)
    q, k, v, cu, cu_cpu, maximum = _inputs(
        seqlens, n_q_heads, n_kv_heads, 128, "cuda", dtype
    )

    q, k, v = _offset_inputs(q, k, v)
    expected = prefill.launch_gluon_mha_prefill_gfx1250(q, k, v, cu, cu_cpu, maximum)
    _check_sampled_fp64(expected, q, k, v, cu_cpu)
    for _ in range(100):
        out = prefill.launch_gluon_mha_prefill_gfx1250(q, k, v, cu, cu_cpu, maximum)
        torch.testing.assert_close(out, expected, rtol=0, atol=0)

    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        captured = prefill.launch_gluon_mha_prefill_gfx1250(
            q, k, v, cu, cu_cpu, maximum
        )
    # Replay with new values to detect stale data and synchronization defects.
    q.mul_(0.5)
    k.mul_(2.0)
    v.neg_()
    for _ in range(20):
        graph.replay()
    _check_sampled_fp64(captured, q, k, v, cu_cpu)


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
@pytest.mark.parametrize(
    "head_dim,seqlen",
    [(64, 8192), (64, 8193), (128, 8192), (128, 8193), (64, 32768), (64, 32769)],
    ids=["d64-full", "d64-tail", "d128-full", "d128-tail", "d64-wide", "d64-wide-tail"],
)
def test_mha_prefill_fixed_shift_retry_graph(dtype, seqlen, head_dim):
    """Changing logits must select a safe numerical path on every replay."""
    torch.manual_seed(46)
    q, k, v, cu, cu_cpu, maximum = _inputs([seqlen], 8, 1, head_dim, "cuda", dtype)

    q, k, v = _offset_inputs(q, k, v)
    original = [tensor.clone() for tensor in (q, k, v)]
    prefill.launch_gluon_mha_prefill_gfx1250(q, k, v, cu, cu_cpu, maximum)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        output = prefill.launch_gluon_mha_prefill_gfx1250(q, k, v, cu, cu_cpu, maximum)

    for case in ("normal", "sharp", "large_denominator", "zero_v", "normal"):
        for tensor, saved in zip((q, k, v), original):
            tensor.copy_(saved)
        if case == "sharp":
            q.mul_(8)
            k.mul_(8)
        elif case in ("large_denominator", "zero_v"):
            # The first KV tile has zero logits; later keys invalidate that
            # shift. BF16 can underflow the reciprocal without overflowing P.
            q.fill_(1)
            k.zero_()
            late_logit = (
                (10 if head_dim == 64 else 7) if case == "large_denominator" else 16
            )
            k[256:].fill_(late_logit)
            if case == "zero_v":
                # Overflowed probabilities times zero must not leak NaNs.
                v.zero_()

        _check_graph_replays(graph, output, q, k, v, cu_cpu, repeats=20)


def _launch_tile(
    q,
    k,
    v,
    cu,
    cu_cpu,
    maximum,
    *,
    block_m,
    num_warps,
    window_left,
    packed_gqa,
    tdm_warp_hint,
    reverse_q_blocks,
):
    """Exercise an explicit kernel tile without replacing production selectors."""
    output = torch.empty_like(q)
    heads, kv_heads, dim = q.shape[1], k.shape[1], q.shape[2]
    group = heads // kv_heads if packed_gqa else 1
    grid = (
        len(cu_cpu) - 1,
        kv_heads if packed_gqa else heads,
        prefill.triton_cdiv(maximum * group, block_m),
    )
    if not packed_gqa and block_m == 256:
        grid = (grid[1], grid[2], grid[0])
    prefill.gluon_mha_prefill_gfx1250[grid](
        q,
        k,
        v,
        cu,
        output,
        q,
        q,
        Q_STRIDE_T=q.stride(0),
        Q_STRIDE_H=q.stride(1),
        Q_STRIDE_D=q.stride(2),
        K_STRIDE_T=k.stride(0),
        K_STRIDE_H=k.stride(1),
        K_STRIDE_D=k.stride(2),
        V_STRIDE_T=v.stride(0),
        V_STRIDE_H=v.stride(1),
        V_STRIDE_D=v.stride(2),
        N_HEADS=heads,
        N_KV_HEADS=kv_heads,
        HEAD_DIM=dim,
        SM_SCALE=math.log2(math.e) / math.sqrt(dim),
        BLOCK_M=block_m,
        BLOCK_N=64,
        NUM_WARPS=num_warps,
        NUM_BUFFERS=2,
        IS_FP8=False,
        HAS_SINK=False,
        HAS_LSE=False,
        WINDOW_LEFT=window_left,
        PACKED_GQA=packed_gqa,
        TDM_WARP_HINT=tdm_warp_hint,
        REVERSE_Q_BLOCKS=reverse_q_blocks,
        DEEP_PIPELINE=False,
        GUARDED_QUERY_ROWS=True,
        COMPACT_RAGGED=False,
        WIDE_ADDRESSING=False,
        QK_VALU_GROUP_SIZE=0,
        QK_DS_GROUP_STRIDE=0,
        ONE_KV_TILE=maximum <= 64,
        TWO_KV_TILES=maximum <= 128,
        USE_TDM_Q=False,
        uniform_seqlen=maximum,
        UNIFORM_SEQLENS=False,
        num_warps=num_warps,
        waves_per_eu=1,
        llvm_fn_attrs="amdgpu-sched-strategy=coexec",
        enable_fp_fusion=True,
        **prefill.sched_compile_options(),
    )
    return output


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
@pytest.mark.parametrize(
    "seqlens,n_q_heads,n_kv_heads,head_dim",
    [
        ([64] * 32, 32, 8, 128),
        ([96] * 32, 32, 8, 128),
        ([127] * 32, 32, 8, 128),
        ([128] * 32, 32, 8, 128),
        ([128] * 32, 8, 1, 128),
        ([128] * 32, 8, 8, 128),
        ([128] * 32, 8, 8, 64),
        ([256] * 32, 8, 8, 128),
        ([129, 127, 0], 32, 8, 128),
        ([257, 65, 1, 0], 8, 1, 64),
        ([512], 32, 8, 128),
        ([1025, 1000, 900], 8, 8, 128),
    ],
    ids=[
        "one_tile",
        "gqa_96",
        "gqa_127",
        "gqa_two_tiles",
        "mqa",
        "mha",
        "mha_d64",
        "two_tiles",
        "ragged",
        "ragged_d64",
        "medium_gqa",
        "medium_ragged_mha",
    ],
)
def test_mha_prefill_short_replay(dtype, seqlens, n_q_heads, n_kv_heads, head_dim):
    """Short-prefill loads and masked stores stay correct on unaligned replay."""
    torch.manual_seed(49)
    q, k, v, cu, cu_cpu, maximum = _inputs(
        seqlens, n_q_heads, n_kv_heads, head_dim, "cuda", dtype
    )

    q, k, v = _offset_inputs(q, k, v)
    saved = [tensor.clone() for tensor in (q, k, v)]
    prefill.launch_gluon_mha_prefill_gfx1250(q, k, v, cu, cu_cpu, maximum)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        output = prefill.launch_gluon_mha_prefill_gfx1250(q, k, v, cu, cu_cpu, maximum)
    for scale in (1, 8, 1):
        for tensor, original in zip((q, k, v), saved):
            tensor.copy_(original)
        q.mul_(scale)
        k.mul_(scale)
        _check_graph_replays(graph, output, q, k, v, cu_cpu, repeats=100)


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
@pytest.mark.parametrize("n_q_heads,n_kv_heads", [(8, 1), (8, 8)])
def test_mha_prefill_uniform_lengths_do_not_recompile(dtype, n_q_heads, n_kv_heads):
    """Different uniform lengths in one tile bucket share a compiled kernel."""
    torch.manual_seed(50)
    for index, length in enumerate((73, 77, 81, 85)):
        q, k, v, cu, cu_cpu, maximum = _inputs(
            [length], n_q_heads, n_kv_heads, 128, "cuda", dtype
        )
        if index == 0:
            output = prefill.launch_gluon_mha_prefill_gfx1250(
                q, k, v, cu, cu_cpu, maximum
            )
        else:
            with assert_no_triton_compile(prefill.gluon_mha_prefill_gfx1250):
                output = prefill.launch_gluon_mha_prefill_gfx1250(
                    q, k, v, cu, cu_cpu, maximum
                )
        _check_sampled_fp64(output, q, k, v, cu_cpu)


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
@pytest.mark.parametrize("head_dim", [64, 128])
@pytest.mark.parametrize("seqlen", [129, 4097])
def test_mha_prefill_large_finite_scores(dtype, head_dim, seqlen):
    """Large finite logits retain a unit maximum weight before normalization."""
    torch.manual_seed(51)
    q, k, v, cu, cu_cpu, maximum = _inputs(
        [seqlen, 1, 0], 8, 1, head_dim, "cuda", dtype
    )

    q, k, v = _offset_inputs(q, k, v)
    prefill.launch_gluon_mha_prefill_gfx1250(q, k, v, cu, cu_cpu, maximum)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        output = prefill.launch_gluon_mha_prefill_gfx1250(q, k, v, cu, cu_cpu, maximum)
    # Equal finite scores have uniform attention over the causal prefix.
    # Non-power-of-two mantissas expose cancellation from scaling before shift.
    values = (
        [
            (-1 * 2.0**100, 1),
            (-3 * 2.0**100, 1),
            (3 * 2.0**100, 1),
            (-5 * 2.0**100, 1),
            (5 * 2.0**100, 1),
        ]
        if dtype == torch.bfloat16
        else [(2047, 2047), (-2047, 2047)]
    )
    for q_value, k_value in values:
        q.fill_(q_value)
        k.fill_(k_value)
        _check_graph_replays(graph, output, q, k, v, cu_cpu, repeats=20)


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
@pytest.mark.parametrize("n_q_heads,n_kv_heads", [(8, 1), (8, 8), (32, 8)])
def test_mha_prefill_compact_batch_counts_do_not_recompile(
    dtype, n_q_heads, n_kv_heads
):
    """Changing request counts preserves compact mapping and its compiled binary."""
    torch.manual_seed(52)
    for index, batch_size in enumerate((3, 6, 7, 17)):
        # Counts include extra grid workgroups, one-token requests and an empty
        # request; none may overwrite a different request's query rows.
        seqlens = [4097] + [1] * (batch_size - 2) + [0]
        q, k, v, cu, cu_cpu, maximum = _inputs(
            seqlens, n_q_heads, n_kv_heads, 128, "cuda", dtype
        )
        if index == 0:
            output = prefill.launch_gluon_mha_prefill_gfx1250(
                q, k, v, cu, cu_cpu, maximum
            )
        else:
            with assert_no_triton_compile(prefill.gluon_mha_prefill_gfx1250):
                output = prefill.launch_gluon_mha_prefill_gfx1250(
                    q, k, v, cu, cu_cpu, maximum
                )
        _check_sampled_fp64(output, q, k, v, cu_cpu)


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
def test_mha_prefill_nonzero_sequence_base(dtype):
    """Uniform requests may start after an unused prefix of the input storage."""
    torch.manual_seed(53)
    q, k, v, cu, cu_cpu, maximum = _inputs([11, 129, 129], 8, 1, 128, "cuda", dtype)
    cu_cpu = cu_cpu[1:]
    cu = cu[1:]
    output = prefill.launch_gluon_mha_prefill_gfx1250(q, k, v, cu, cu_cpu, maximum)
    # Only the rows named by the cumulative boundaries belong to this launch.
    expected, _ = torch_dense_attention(
        q[11:140].double(),
        k[11:140].double(),
        v[11:140].double(),
        scale=1.0 / math.sqrt(128),
        causal_offset=0,
        window_left=None,
        sinks=None,
    )
    tolerance = 2e-2 if dtype == torch.bfloat16 else 3e-3
    torch.testing.assert_close(
        output[11:140].double(), expected, rtol=tolerance, atol=tolerance
    )
    expected, _ = torch_dense_attention(
        q[140:].double(),
        k[140:].double(),
        v[140:].double(),
        scale=1.0 / math.sqrt(128),
        causal_offset=0,
        window_left=None,
        sinks=None,
    )
    torch.testing.assert_close(
        output[140:].double(), expected, rtol=tolerance, atol=tolerance
    )


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
@pytest.mark.parametrize(
    "batch,n_q_heads,n_kv_heads,lengths",
    [
        (1, 16, 4, (80, 96, 112, 128)),
        (1, 32, 8, (80, 96, 112, 128)),
        (1, 64, 16, (80, 96, 112, 128)),
        (1, 32, 16, (96, 128)),
        (1, 32, 1, (80, 96, 112, 128)),
        (1, 32, 32, (128,)),
        (48, 8, 1, (96, 112, 128)),
        (48, 8, 1, (97, 111, 125, 127)),
        (48, 8, 1, (81, 87, 89, 95)),
        (48, 8, 1, (65, 71, 79)),
        (32, 8, 1, (81, 95, 97, 111, 127)),
        (32, 8, 1, (80, 96, 112)),
        (24, 16, 1, (65, 69, 75, 79)),
        (12, 32, 1, (65, 71, 77, 81)),
        (4, 32, 1, (96, 112, 128)),
        (6, 32, 1, (128,)),
        (8, 32, 1, (65, 69, 73, 77, 81)),
        (6, 32, 1, (85, 89, 93, 97)),
        (5, 32, 1, (113, 117, 121, 125, 127)),
        (4, 32, 1, (125, 126, 127)),
        (47, 8, 1, (97, 111, 125, 127)),
        (48, 8, 4, (65, 79, 95, 111, 127)),
        (64, 8, 4, (65, 79, 81, 95, 97, 111, 127)),
        (96, 8, 4, (65, 79, 81, 95, 97, 111, 127)),
        (24, 16, 4, (97, 111, 125, 127)),
        # Keep neighboring lengths inside the eight-wave dispatch bucket.
        (24, 16, 1, (81, 95, 97, 111, 125, 127)),
        (12, 32, 1, (85, 95, 97, 111, 125, 127)),
    ],
)
def test_mha_prefill_short_queries_replay(dtype, batch, n_q_heads, n_kv_heads, lengths):
    torch.manual_seed(71)
    for index, length in enumerate(lengths):
        q, k, v, cu, cu_cpu, maximum = _inputs(
            [length] * batch, n_q_heads, n_kv_heads, 128, "cuda", dtype
        )

        q, k, v = _offset_inputs(q, k, v)
        saved = [tensor.clone() for tensor in (q, k, v)]
        if index == 0:
            prefill.launch_gluon_mha_prefill_gfx1250(q, k, v, cu, cu_cpu, maximum)
        else:
            with assert_no_triton_compile(prefill.gluon_mha_prefill_gfx1250):
                prefill.launch_gluon_mha_prefill_gfx1250(q, k, v, cu, cu_cpu, maximum)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            output = prefill.launch_gluon_mha_prefill_gfx1250(
                q, k, v, cu, cu_cpu, maximum
            )
        for scale in (1, 8, 1):
            for tensor, original in zip((q, k, v), saved):
                tensor.copy_(original)
            q.mul_(scale)
            k.mul_(scale)
            _check_graph_replays(graph, output, q, k, v, cu_cpu, repeats=100)


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
def test_mha_prefill_medium_ragged_lengths_do_not_recompile(dtype):
    """Neighboring ragged lengths in one geometry bucket share the same kernel."""
    torch.manual_seed(72)
    for index, length in enumerate((1023, 1024, 1025)):
        q, k, v, cu, cu_cpu, maximum = _inputs(
            [length, 1000, 900], 8, 8, 128, "cuda", dtype
        )
        if index == 0:
            output = prefill.launch_gluon_mha_prefill_gfx1250(
                q, k, v, cu, cu_cpu, maximum
            )
        else:
            with assert_no_triton_compile(prefill.gluon_mha_prefill_gfx1250):
                output = prefill.launch_gluon_mha_prefill_gfx1250(
                    q, k, v, cu, cu_cpu, maximum
                )
        _check_sampled_fp64(output, q, k, v, cu_cpu)


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
@pytest.mark.parametrize(
    "batch,n_q_heads,n_kv_heads,lengths",
    [
        (1, 32, 16, (513, 514, 520, 537, 545, 569, 577, 633)),
        (1, 32, 8, (513, 514, 520, 521, 537, 545, 552, 553, 577)),
        (1, 32, 4, (513, 514, 516, 521, 525, 529, 532, 541, 553)),
        (1, 32, 2, (513, 514, 515, 517, 519, 521, 522, 531)),
        (1, 32, 1, (513, 515, 517, 519, 521, 523)),
        (4, 8, 1, (513, 514, 516, 521, 525, 529, 532, 541, 553)),
    ],
)
def test_mha_prefill_partial_packed_tiles_replay_without_recompile(
    dtype, batch, n_q_heads, n_kv_heads, lengths
):
    """Partial packed tiles preserve numerics and reuse code across request lengths."""
    torch.manual_seed(73)
    for index, length in enumerate(lengths):
        q, k, v, cu, cu_cpu, maximum = _inputs(
            [length] * batch, n_q_heads, n_kv_heads, 128, "cuda", dtype
        )

        q, k, v = _offset_inputs(q, k, v)
        saved = [tensor.clone() for tensor in (q, k, v)]
        if index == 0:
            prefill.launch_gluon_mha_prefill_gfx1250(q, k, v, cu, cu_cpu, maximum)
        else:
            with assert_no_triton_compile(prefill.gluon_mha_prefill_gfx1250):
                prefill.launch_gluon_mha_prefill_gfx1250(q, k, v, cu, cu_cpu, maximum)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            output = prefill.launch_gluon_mha_prefill_gfx1250(
                q, k, v, cu, cu_cpu, maximum
            )
        for scale in (1, 8, 1):
            for tensor, original in zip((q, k, v), saved):
                tensor.copy_(original)
            q.mul_(scale)
            k.mul_(scale)
            _check_graph_replays(graph, output, q, k, v, cu_cpu, repeats=100)


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
@pytest.mark.parametrize(
    "head_dim,batch,seqlen,n_kv_heads",
    [
        (64, 4, 4096, 2),
        (128, 4, 4096, 2),
        (128, 64, 95, 4),
        (128, 96, 95, 4),
        # Sinks/LSE must route the wide-transport shape to a compatible schedule.
        (128, 8, 1024, 1),
    ],
)
@pytest.mark.parametrize("feature", ["sink", "lse", "both"])
def test_mha_prefill_sink_lse_replay(
    dtype, head_dim, batch, seqlen, n_kv_heads, feature
):
    """Short and long schedules preserve sinks/LSE as graph inputs change."""
    torch.manual_seed(74)
    q, k, v, cu, cu_cpu, maximum = _inputs(
        [seqlen] * batch, 8, n_kv_heads, head_dim, "cuda", dtype
    )
    saved = [tensor.clone() for tensor in (q, k, v)]
    sinks = (
        torch.linspace(-4, 4, 8, device="cuda", dtype=dtype)
        if feature in ("sink", "both")
        else None
    )
    original_sinks = None if sinks is None else sinks.clone()
    return_lse = feature in ("lse", "both")
    prefill.launch_gluon_mha_prefill_gfx1250(
        q, k, v, cu, cu_cpu, maximum, sinks=sinks, return_lse=return_lse
    )
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        result = prefill.launch_gluon_mha_prefill_gfx1250(
            q, k, v, cu, cu_cpu, maximum, sinks=sinks, return_lse=return_lse
        )
    output, lse = result if return_lse else (result, None)
    patterns = ["normal", "sharp", "late_large"]
    if feature == "both" and dtype == torch.bfloat16:
        patterns.extend(["large_value", "small_probability"])
    # 2**125 is finite in BF16; FP16 would store infinity.
    if sinks is not None and dtype == torch.bfloat16:
        patterns.append("large_sink")
    patterns.append("normal")
    for pattern in patterns:
        for tensor, original in zip((q, k, v), saved):
            tensor.copy_(original)
        if sinks is not None:
            sinks.copy_(original_sinks)
        if pattern == "sharp":
            q.mul_(8)
            k.mul_(8)
        elif pattern == "late_large":
            q.fill_(1)
            k.zero_()
            for start, end in zip(cu_cpu[:-1], cu_cpu[1:]):
                k[start + min(256, seqlen // 2) : end].fill_(16)
            if sinks is not None:
                sinks.add_(96)
        elif pattern == "large_sink":
            # Finite sink logits can overflow conversion to unscaled QK units.
            # Keep other heads unchanged to check both normalization regimes.
            sinks[::2] = 2.0**125
        elif pattern == "large_value":
            # Finite BF16 values can overflow the unnormalized FP32 PV sum
            # before the sink makes the final attention output much smaller.
            q.zero_()
            k.zero_()
            v.fill_(2.0**120)
            v[..., 1::2].neg_()
            sinks.fill_(64)
        elif pattern == "small_probability":
            # Scaling P instead of V would underflow small probabilities whose
            # product with a large value still contributes to the output.
            q.fill_(1)
            k.zero_()
            v.zero_()
            sinks.zero_()
            for start, end in zip(cu_cpu[:-1], cu_cpu[1:]):
                k[end - 1].fill_(-80 / math.sqrt(head_dim))
                v[end - 1].fill_(2.0**120)
        output.fill_(float("nan"))
        if lse is not None:
            lse.fill_(float("nan"))
        graph.replay()
        _check_sampled_fp64(output, q, k, v, cu_cpu, sinks=sinks, lse=lse)
        expected = output.clone()
        expected_lse = None if lse is None else lse.clone()
        for _ in range(20):
            output.fill_(float("nan"))
            if lse is not None:
                lse.fill_(float("nan"))
            graph.replay()
            torch.testing.assert_close(output, expected, rtol=0, atol=0)
            if lse is not None:
                torch.testing.assert_close(lse, expected_lse, rtol=0, atol=0)
