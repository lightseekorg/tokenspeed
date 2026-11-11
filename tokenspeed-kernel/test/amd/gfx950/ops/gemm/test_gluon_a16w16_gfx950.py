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

import contextlib

import pytest
import torch
from tokenspeed_kernel.numerics.torch.gemm import torch_mm
from utils import assert_no_triton_compile, assert_warmup_covers, is_cdna4

if not is_cdna4():
    pytest.skip(
        "AMD CDNA4 is required for dense16 Gluon GEMM tests",
        allow_module_level=True,
    )


from tokenspeed_kernel.ops.gemm import bmm, kimi3_router_projection, mm  # noqa: E402
from tokenspeed_kernel.workspace import capture_session, zeroed_state  # noqa: E402
from tokenspeed_kernel_amd.ops.gfx950.gemm.fp16.largem import (  # noqa: E402
    launch_gluon_mm_a16w16_prefill_gfx950,
    launch_gluon_mm_a16w16_prefill_publish_rows_gfx950,
    launch_gluon_mm_a16w16_prefill_skip_rows_gfx950,
    supports_gluon_mm_a16w16_prefill_gfx950,
)
from tokenspeed_kernel_amd.ops.gfx950.gemm.fp16.mm import (  # noqa: E402
    _choose_mfma_lds_mediumm_config,
    _get_partial_scratch,
    _supports_mfma_lds_smallm,
    gluon_grouped_mm_a16w16_decode_num_counters_gfx950,
    gluon_mm_a16w16_decode_num_counters_gfx950,
    gluon_mm_a16w16_medium_gfx950,
    launch_gluon_bmm_a16w16_gfx950,
    launch_gluon_grouped_mm_a16w16_decode_gfx950,
    launch_gluon_mm_a16w16_column_shard_gfx950,
    launch_gluon_mm_a16w16_decode_add3_gfx950,
    launch_gluon_mm_a16w16_decode_gfx950,
    launch_gluon_mm_a16w16_medium_gfx950,
    launch_gluon_mm_a16w16_splitk_gfx950,
    supports_gluon_mm_a16w16_column_shard_gfx950,
    supports_gluon_mm_a16w16_decode_add3_gfx950,
    supports_gluon_mm_a16w16_decode_gfx950,
)

# Kernels and references accumulate in FP32 and round once to BF16, so outputs
# differ by at most one BF16 ULP (2**-7 relative). atol only covers FP32
# accumulation-order noise on near-zero outputs.
_ATOL = 1e-5
_RTOL = 2**-7

_CORRECTNESS_CASES = [
    pytest.param(
        launch_gluon_mm_a16w16_splitk_gfx950,
        (4, 256, 2048),
        id="splitk-smallm",
    ),
    pytest.param(
        launch_gluon_mm_a16w16_medium_gfx950,
        (8, 128, 64),
        id="mediumm",
    ),
    pytest.param(
        launch_gluon_mm_a16w16_prefill_gfx950,
        (256, 256, 256),
        id="largem",
    ),
]


@pytest.mark.parametrize("kernel,shape", _CORRECTNESS_CASES)
def test_dense16_kernel_variant_correctness(
    kernel, shape: tuple[int, int, int]
) -> None:
    torch.manual_seed(0)
    dtype = torch.bfloat16
    m, n, k = shape
    a = torch.randn((m, k), device="cuda", dtype=dtype) * 0.25
    b = torch.randn((n, k), device="cuda", dtype=dtype) * 0.25

    out = kernel(a, b, dtype)
    assert out is not None

    torch.testing.assert_close(out, torch.mm(a, b.T), atol=_ATOL, rtol=_RTOL)


@pytest.mark.parametrize("kernel,shape", _CORRECTNESS_CASES)
def test_dense16_kernel_variant_writes_strided_out(
    kernel, shape: tuple[int, int, int]
) -> None:
    torch.manual_seed(0)
    dtype = torch.bfloat16
    m, n, k = shape
    a = torch.randn((m, k), device="cuda", dtype=dtype) * 0.25
    b = torch.randn((n, k), device="cuda", dtype=dtype) * 0.25
    backing = torch.empty((m, n + 17), device="cuda", dtype=dtype)
    out = backing[:, :n]

    actual = kernel(a, b, dtype, out=out)

    assert actual is out
    torch.testing.assert_close(out, torch.mm(a, b.T), atol=_ATOL, rtol=_RTOL)


@pytest.mark.parametrize("batch", [12, 16])
def test_dense16_bmm_writes_strided_out(batch: int) -> None:
    torch.manual_seed(0)
    dtype = torch.bfloat16
    m, n, k = 1, 512, 128
    a_backing = torch.randn((m, batch, k), device="cuda", dtype=dtype) * 0.25
    a = a_backing.transpose(0, 1)
    weight = torch.randn((batch, k, n), device="cuda", dtype=dtype) * 0.25
    b = weight.transpose(1, 2)
    backing = torch.empty((m, batch, n + 17), device="cuda", dtype=dtype)
    out = backing[..., :n].transpose(0, 1)

    actual = launch_gluon_bmm_a16w16_gfx950(a, b, dtype, out=out)

    assert actual is out
    torch.testing.assert_close(out, torch.bmm(a, weight), atol=_ATOL, rtol=_RTOL)


def test_dense16_bmm_rejects_unsupported_shape() -> None:
    a = torch.empty((12, 2, 128), device="cuda", dtype=torch.bfloat16)
    b = torch.empty((12, 512, 128), device="cuda", dtype=torch.bfloat16)

    assert launch_gluon_bmm_a16w16_gfx950(a, b, torch.bfloat16) is None


def test_splitk_smallm_out_handles_padded_reducer_rows() -> None:
    torch.manual_seed(0)
    dtype = torch.bfloat16
    m, n, k = 2, 256, 2048
    a = torch.randn((m, k), device="cuda", dtype=dtype) * 0.25
    b = torch.randn((n, k), device="cuda", dtype=dtype) * 0.25
    backing = torch.empty((m, n + 17), device="cuda", dtype=dtype)
    out = backing[:, :n]

    actual = launch_gluon_mm_a16w16_splitk_gfx950(a, b, dtype, out=out)

    assert actual is out
    torch.testing.assert_close(out, torch.mm(a, b.T), atol=_ATOL, rtol=_RTOL)


def test_supports_splitk_covers_smallm_high_k_shapes() -> None:
    assert _supports_mfma_lds_smallm(1, 4096, 4096)
    assert _supports_mfma_lds_smallm(2, 4096, 4096)
    assert _supports_mfma_lds_smallm(1, 1280, 2880)
    assert _supports_mfma_lds_smallm(4, 1280, 1024)
    assert _supports_mfma_lds_smallm(4, 2560, 2048)
    assert _supports_mfma_lds_smallm(4, 8192, 8192)


def test_supports_splitk_rejects_non_target_shapes() -> None:
    assert not _supports_mfma_lds_smallm(4, 3968, 4096)
    assert not _supports_mfma_lds_smallm(4, 1280, 960)
    assert not _supports_mfma_lds_smallm(4, 1280, 1216)
    assert not _supports_mfma_lds_smallm(4, 4224, 4096)
    assert not _supports_mfma_lds_smallm(3, 4096, 4096)
    assert not _supports_mfma_lds_smallm(8, 8192, 4096)


def test_splitk_partial_scratch_is_stream_local() -> None:
    device = torch.device("cuda")
    first = _get_partial_scratch(device, 2, 8, 256, 4)
    second = _get_partial_scratch(device, 2, 8, 256, 4)

    other_stream = torch.cuda.Stream()
    with torch.cuda.stream(other_stream):
        other_first = _get_partial_scratch(device, 2, 8, 256, 4)
        other_second = _get_partial_scratch(device, 2, 8, 256, 4)
    other_stream.synchronize()

    assert first.shape == second.shape == other_first.shape == other_second.shape
    assert first.data_ptr() == second.data_ptr()
    assert other_first.data_ptr() == other_second.data_ptr()
    assert first.data_ptr() != other_first.data_ptr()


def test_choose_mfma_lds_mediumm_config_uses_tuned_medium_m_tiles() -> None:
    assert _choose_mfma_lds_mediumm_config(8, 1280, 64) == (16, 32, 64, 2, 2, 1)
    assert _choose_mfma_lds_mediumm_config(8, 1280, 512) == (16, 32, 256, 2, 2, 2)
    assert _choose_mfma_lds_mediumm_config(16, 1280, 768) == (16, 32, 256, 2, 2, 3)
    assert _choose_mfma_lds_mediumm_config(8, 1280, 1024) == (16, 32, 512, 2, 2, 2)
    assert _choose_mfma_lds_mediumm_config(32, 2560, 2048) == (16, 16, 512, 2, 2, 2)
    assert _choose_mfma_lds_mediumm_config(64, 1280, 1024) == (32, 32, 512, 2, 2, 2)
    assert _choose_mfma_lds_mediumm_config(64, 1280, 2048) == (32, 32, 512, 2, 2, 2)
    assert _choose_mfma_lds_mediumm_config(64, 2560, 2048) == (32, 32, 128, 2, 2, 3)
    assert _choose_mfma_lds_mediumm_config(128, 2560, 2048) == (32, 32, 64, 2, 2, 3)
    assert _choose_mfma_lds_mediumm_config(128, 1280, 2880) == (32, 32, 64, 2, 2, 3)
    assert _choose_mfma_lds_mediumm_config(128, 4096, 4096) == (16, 128, 64, 1, 4, 3)
    assert _choose_mfma_lds_mediumm_config(768, 3584, 7168) == (
        128,
        128,
        64,
        2,
        4,
        3,
    )
    assert _choose_mfma_lds_mediumm_config(1024, 3584, 7168) == (
        128,
        128,
        64,
        2,
        4,
        3,
    )
    assert _choose_mfma_lds_mediumm_config(384, 7168, 3584) == (
        128,
        128,
        64,
        2,
        4,
        3,
    )
    assert _choose_mfma_lds_mediumm_config(512, 7168, 3584) == (
        128,
        128,
        64,
        2,
        4,
        3,
    )


def test_choose_mfma_lds_mediumm_config_falls_back_for_slow_shapes() -> None:
    assert _choose_mfma_lds_mediumm_config(16, 1280, 2880) is None
    assert _choose_mfma_lds_mediumm_config(16, 2560, 2048) is None
    assert _choose_mfma_lds_mediumm_config(32, 1280, 1024) is None
    assert _choose_mfma_lds_mediumm_config(16, 1280, 8192) is None
    assert _choose_mfma_lds_mediumm_config(32, 1280, 4096) is None
    assert _choose_mfma_lds_mediumm_config(128, 4096, 64) is None
    assert _choose_mfma_lds_mediumm_config(128, 4096, 512) is None
    assert _choose_mfma_lds_mediumm_config(64, 4096, 2048) is None
    assert _choose_mfma_lds_mediumm_config(256, 1280, 1024) is None
    assert _choose_mfma_lds_mediumm_config(512, 4096, 4096) is None
    assert _choose_mfma_lds_mediumm_config(1024, 8192, 8192) is None
    assert _choose_mfma_lds_mediumm_config(640, 3584, 7168) is None
    assert _choose_mfma_lds_mediumm_config(1152, 3584, 7168) is None
    assert _choose_mfma_lds_mediumm_config(320, 7168, 3584) is None
    assert _choose_mfma_lds_mediumm_config(576, 7168, 3584) is None


@pytest.mark.parametrize("m,k", [(8192, 1536), (4096, 1536), (2048, 768)])
def test_largem_publish_rows_keeps_bits_and_counts_column_tiles(m, k) -> None:
    torch.manual_seed(m + k)
    a = torch.randn((m, k), device="cuda", dtype=torch.bfloat16)
    b = torch.randn((7168, k), device="cuda", dtype=torch.bfloat16) / 40
    full = launch_gluon_mm_a16w16_prefill_gfx950(a, b, torch.bfloat16)
    out = torch.full_like(full, 7.0)
    counters = torch.full((5,), -3, device="cuda", dtype=torch.int32)
    # Every owner's heap is this one, so each owned tile counts all 8 owners.
    heaps = (counters.data_ptr(),) * 8
    for call in (1, 2):
        assert (
            launch_gluon_mm_a16w16_prefill_publish_rows_gfx950(
                a, b, out, counters, heaps, rank=3
            )
            is out
        )
        torch.testing.assert_close(out, full, atol=0, rtol=0)
        tiles = m // 2048
        expected = [-3 + call * 8 * 28] * tiles + [-3] * (5 - tiles)
        assert counters.tolist() == expected
    # Owner ranges of partial tiles, or too few counters, launch nothing.
    for rows, width in ((m - 8, 7168), (m, 7168)):
        operand = a[:rows]
        short = counters[: m // 2048 - 1] if rows == m else counters
        assert (
            launch_gluon_mm_a16w16_prefill_publish_rows_gfx950(
                operand, b, out[:rows, :width], short, heaps, rank=3
            )
            is None
        )
    assert counters.tolist() == expected


@pytest.mark.parametrize(
    "m,skip_start,skip_rows",
    [
        (8192, 0, 1024),
        (8192, 3072, 1024),
        (8192, 7168, 1024),
        (2048, 256, 256),
        (1800, 512, 768),
    ],
)
def test_largem_skip_rows_keeps_the_full_gemm_bits(m, skip_start, skip_rows) -> None:
    torch.manual_seed(m + skip_start)
    a = torch.randn((m, 7168), device="cuda", dtype=torch.bfloat16)
    b = torch.randn((6288, 7168), device="cuda", dtype=torch.bfloat16) / 84
    full = launch_gluon_mm_a16w16_prefill_gfx950(a, b, torch.bfloat16)
    out = torch.full_like(full, 7.0)
    assert (
        launch_gluon_mm_a16w16_prefill_skip_rows_gfx950(
            a, b, out, skip_start=skip_start, skip_rows=skip_rows
        )
        is out
    )
    skipped = slice(skip_start, skip_start + skip_rows)
    assert torch.all(out[skipped] == 7.0)
    out[skipped] = full[skipped]
    torch.testing.assert_close(out, full, atol=0, rtol=0)
    # Row bounds off the 256-row tiles, or skipping every row, launch nothing.
    for start, rows in (
        (skip_start + 1, skip_rows),
        (skip_start, skip_rows + 8),
        (0, m),
    ):
        assert (
            launch_gluon_mm_a16w16_prefill_skip_rows_gfx950(
                a, b, out, skip_start=start, skip_rows=rows
            )
            is None
        )


@pytest.mark.parametrize(
    "shape",
    [
        pytest.param((300, 6288, 7168), id="qkvfab-ragged-mn"),
        pytest.param((77, 100, 512), id="sub-tile"),
        pytest.param((3001, 7168, 4224), id="odd-k-pairs"),
    ],
)
def test_largem_masks_partial_tiles(shape: tuple[int, int, int]) -> None:
    torch.manual_seed(0)
    dtype = torch.bfloat16
    m, n, k = shape
    a = torch.randn((m, k), device="cuda", dtype=dtype) * 0.25
    b = torch.randn((n, k), device="cuda", dtype=dtype) * 0.25
    # A sentinel-filled padded row stride catches stores past column N.
    backing = torch.full((m, n + 16), 7.0, device="cuda", dtype=dtype)
    out = backing[:, :n]

    launch_gluon_mm_a16w16_prefill_gfx950(a, b, dtype, out=out)

    torch.testing.assert_close(out, torch.mm(a, b.T), atol=_ATOL, rtol=_RTOL)
    assert torch.all(backing[:, n:] == 7.0)


def test_prefill_routes_k3_shapes_with_busy_cus() -> None:
    # qkvfab spans 25 workgroups across N. 4096 tokens launch 400 workgroups,
    # busying 78% of 256 CUs over two rounds; 3072 tokens launch 300, which
    # leaves most CUs idle in the second round.
    assert supports_gluon_mm_a16w16_prefill_gfx950(4096, 6288, 7168)
    assert supports_gluon_mm_a16w16_prefill_gfx950(4000, 6288, 7168)
    assert not supports_gluon_mm_a16w16_prefill_gfx950(3072, 6288, 7168)
    assert not supports_gluon_mm_a16w16_prefill_gfx950(1024, 6288, 7168)
    # Past two rounds the long qkvfab reduction needs nearly every CU busy:
    # 8192 tokens busy 78% over four rounds, 12288 tokens 94% over five. The
    # short attention output reduction keeps the base rule.
    assert not supports_gluon_mm_a16w16_prefill_gfx950(8192, 6288, 7168)
    assert supports_gluon_mm_a16w16_prefill_gfx950(12288, 6288, 7168)
    assert supports_gluon_mm_a16w16_prefill_gfx950(8192, 7168, 1536)
    # Unmeasured or losing shapes keep hipBLASLt at any token count.
    assert not supports_gluon_mm_a16w16_prefill_gfx950(4096, 4096, 4096)
    assert not supports_gluon_mm_a16w16_prefill_gfx950(7168, 2304, 1536)


# Kimi K3 TP8 decode shapes covering split-K (router with FP32 output, shared
# expert gate/up), N not a multiple of BLOCK_N (qkvfab), M not a multiple of
# BLOCK_M, and the single-pass path.
_DECODE_CASES = [
    pytest.param((4, 896, 7168), torch.float32, id="router-splitk-fp32"),
    pytest.param((20, 1536, 7168), torch.bfloat16, id="shared-gu-splitk"),
    pytest.param((8, 6288, 7168), torch.bfloat16, id="qkvfab-ragged-n"),
    pytest.param((40, 7168, 768), torch.bfloat16, id="shared-dn-ragged-m"),
    # 16x128 split-K tiles store C through the coalesced row layout.
    pytest.param((12, 8448, 7168), torch.bfloat16, id="dense-gu-coalesced-store"),
]


def _decode(a, b, out_dtype, out=None):
    m, k = a.shape
    counters = zeroed_state(
        gluon_mm_a16w16_decode_num_counters_gfx950(m, b.shape[0], k),
        dtype=torch.int32,
        device=a.device,
    )
    return launch_gluon_mm_a16w16_decode_gfx950(
        a, b, out_dtype, counters=counters, out=out
    )


def _decode_reference(a, b, out_dtype):
    # The product in the output dtype: FP32 accumulates FP32 copies.
    return torch_mm(a.to(out_dtype), b.to(out_dtype), None, None, out_dtype)


@pytest.mark.parametrize("shape,out_dtype", _DECODE_CASES)
def test_decode_gemm_correctness_across_repeated_calls(
    shape: tuple[int, int, int], out_dtype: torch.dtype
) -> None:
    torch.manual_seed(0)
    m, n, k = shape
    b = torch.randn((n, k), device="cuda", dtype=torch.bfloat16) * 0.25
    # Split-K reuses per-tile arrival counters; the last program of each tile
    # resets them, so back-to-back launches must stay correct.
    for _ in range(3):
        a = torch.randn((m, k), device="cuda", dtype=torch.bfloat16) * 0.25
        out = _decode(a, b, out_dtype)
        assert out.dtype == out_dtype
        # FP32 output differs from the reference only by summation order.
        rtol = 1e-5 if out_dtype == torch.float32 else _RTOL
        torch.testing.assert_close(
            out, _decode_reference(a, b, out_dtype), atol=1e-4, rtol=rtol
        )


@pytest.mark.parametrize("session", [True, False], ids=["session", "no-session"])
def test_decode_gemm_graph_replays(session: bool) -> None:
    # The router bucket uses 16x16 tiles with split-K 7. A capture session
    # zeroes the counters once outside the graph, otherwise the graph zeroes
    # them itself; either way every replay leaves them zeroed for the next.
    torch.manual_seed(0)
    m, n, k = 4, 896, 7168
    a = torch.randn((m, k), device="cuda", dtype=torch.bfloat16) * 0.25
    b = torch.randn((n, k), device="cuda", dtype=torch.bfloat16) * 0.25
    out = torch.empty((m, n), device="cuda", dtype=torch.float32)

    stream = torch.cuda.Stream()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.stream(stream):
        _decode(a, b, torch.float32, out=out)
        scope = capture_session(graph) if session else contextlib.nullcontext()
        with scope, torch.cuda.graph(graph, stream=stream):
            _decode(a, b, torch.float32, out=out)
    stream.synchronize()

    for _ in range(3):
        a.copy_(torch.randn_like(a) * 0.25)
        graph.replay()
        torch.testing.assert_close(out, a.float() @ b.float().T, atol=1e-4, rtol=1e-5)


@pytest.mark.parametrize("m", [4, 16])
def test_kimi3_router_gluon_decode_matches_reference(m: int) -> None:
    # The router's split-K decode buckets, through the model-facing op.
    torch.manual_seed(0)
    a = torch.randn((m, 7168), device="cuda", dtype=torch.bfloat16) * 0.25
    b = torch.randn((896, 7168), device="cuda", dtype=torch.bfloat16) * 0.25
    for _ in range(2):
        out = kimi3_router_projection(a, b, solution="gluon_decode")
        torch.testing.assert_close(out, a.float() @ b.float().T, atol=1e-4, rtol=1e-5)


def test_decode_gemm_writes_strided_out() -> None:
    torch.manual_seed(0)
    m, n, k = 4, 896, 7168
    a = torch.randn((m, k), device="cuda", dtype=torch.bfloat16) * 0.25
    b = torch.randn((n, k), device="cuda", dtype=torch.bfloat16) * 0.25
    backing = torch.empty((m, n + 17), device="cuda", dtype=torch.float32)
    out = backing[:, :n]

    actual = _decode(a, b, torch.float32, out=out)

    assert actual is out
    torch.testing.assert_close(out, a.float() @ b.float().T, atol=1e-4, rtol=1e-5)


def test_decode_gemm_supports_only_measured_buckets() -> None:
    # Every measured K3 shape from M=2 up to its fastest bucket.
    assert supports_gluon_mm_a16w16_decode_gfx950(2, 896, 7168)
    assert supports_gluon_mm_a16w16_decode_gfx950(64, 896, 7168)
    assert supports_gluon_mm_a16w16_decode_gfx950(3, 2304, 1536)
    # M=1 keeps the GEMV kernels; larger M and unknown shapes keep torch.
    assert not supports_gluon_mm_a16w16_decode_gfx950(1, 896, 7168)
    assert not supports_gluon_mm_a16w16_decode_gfx950(65, 896, 7168)
    assert not supports_gluon_mm_a16w16_decode_gfx950(4, 4096, 4096)
    # A bucket missing from a shape's table falls back instead of borrowing
    # the next bucket's config: dense gate/up is measured only for 9..32.
    assert not supports_gluon_mm_a16w16_decode_gfx950(8, 8448, 7168)
    assert supports_gluon_mm_a16w16_decode_gfx950(9, 8448, 7168)
    # Past 64 rows only the measured buckets serve: the Muse TP8 pre-expert
    # projection and the K3.1 TP8 verify shapes that beat torch there.
    assert supports_gluon_mm_a16w16_decode_gfx950(256, 512, 8192)
    assert not supports_gluon_mm_a16w16_decode_gfx950(65, 8192, 1024)
    assert not supports_gluon_mm_a16w16_decode_gfx950(257, 512, 8192)
    assert supports_gluon_mm_a16w16_decode_gfx950(128, 1536, 7168)
    assert not supports_gluon_mm_a16w16_decode_gfx950(129, 1536, 7168)
    assert supports_gluon_mm_a16w16_decode_gfx950(192, 2112, 7168)
    assert not supports_gluon_mm_a16w16_decode_gfx950(65, 20480, 7168)


# The K3.1 TP8 speculative-verify buckets at ragged and full row counts, twice
# each so split-K counters reset between calls.
_K31_VERIFY_BUCKETS = [
    ((6288, 7168), (36, 100)),
    ((1536, 7168), (68, 128)),
    ((7168, 768), (100, 132, 192)),
    ((8448, 7168), (36, 64)),
    ((7168, 4224), (44, 64)),
    ((160, 7168), (100, 132, 192)),
    ((6400, 1536), (36, 60)),
    ((1536, 128), (36, 100, 192)),
    ((2112, 7168), (60, 100, 188)),
    ((1536, 1536), (44, 100)),
]


@pytest.mark.parametrize(
    "n,k,rows",
    [pytest.param(n, k, rows, id=f"{n}x{k}") for (n, k), rows in _K31_VERIFY_BUCKETS],
)
def test_decode_gemm_k31_verify_buckets(n: int, k: int, rows: tuple[int, ...]) -> None:
    torch.manual_seed(0)
    b = torch.randn((n, k), device="cuda", dtype=torch.bfloat16) * 0.25
    for m in rows:
        assert supports_gluon_mm_a16w16_decode_gfx950(m, n, k)
        for _ in range(2):
            a = torch.randn((m, k), device="cuda", dtype=torch.bfloat16) * 0.25
            torch.testing.assert_close(
                _decode(a, b, torch.bfloat16),
                _decode_reference(a, b, torch.bfloat16),
                atol=1e-4,
                rtol=_RTOL,
            )


# Kimi K3 shared gate/up, the Muse TP8 pre-expert projection, whose 64-row
# bucket splits K, and the indexer wq_b, which has only a 32-row and a 64-row
# tile.
@pytest.mark.parametrize("n,k", [(1536, 7168), (512, 8192), (4096, 1536)])
def test_decode_gemm_warmup_compiles_every_row_bucket(n: int, k: int) -> None:
    torch.manual_seed(0)
    a = torch.randn((64, k), device="cuda", dtype=torch.bfloat16) * 0.25
    b = torch.randn((n, k), device="cuda", dtype=torch.bfloat16) * 0.25
    counters = zeroed_state(
        gluon_mm_a16w16_decode_num_counters_gfx950(64, n, k),
        dtype=torch.int32,
        device=a.device,
    )

    def run(rows):
        if supports_gluon_mm_a16w16_decode_gfx950(rows, n, k):
            out = _decode(a[:rows], b, torch.bfloat16)
            torch.testing.assert_close(
                out, torch.mm(a[:rows], b.T), atol=1e-4, rtol=_RTOL
            )

    # The row count itself does not specialize, so one launch per bucket
    # covers every count in it, multiples of 16 included.
    assert_warmup_covers(
        (gluon_mm_a16w16_medium_gfx950,),
        lambda: launch_gluon_mm_a16w16_decode_gfx950(
            a, b, torch.bfloat16, counters=counters, warm_variants=True
        ),
        run,
        (2, 6, 7, 12, 14, 16, 20, 27, 32, 40, 48, 61, 64),
    )


@pytest.mark.parametrize("m", [2, 4, 7, 8, 16, 24, 32])
def test_decode_add3_matches_single_rounding_reference(m: int) -> None:
    torch.manual_seed(0)
    n, k = 7168, 3584
    a = torch.randn((m, k), device="cuda", dtype=torch.bfloat16) * 0.25
    b = torch.randn((n, k), device="cuda", dtype=torch.bfloat16) * 0.25
    addend_a = torch.randn((m, n), device="cuda", dtype=torch.bfloat16)
    # A column slice of a wider lane, as the K3 shared-expert output arrives.
    lane = torch.randn((m, n + 3584), device="cuda", dtype=torch.bfloat16)
    addend_b = lane[:, 3584:]

    out = launch_gluon_mm_a16w16_decode_add3_gfx950(a, b, addend_a, addend_b)

    expected = (a.float() @ b.float().T + addend_a.float() + addend_b.float()).to(
        torch.bfloat16
    )
    torch.testing.assert_close(out, expected, atol=1e-4, rtol=_RTOL)


def test_decode_add3_supports_only_single_pass_buckets() -> None:
    assert supports_gluon_mm_a16w16_decode_add3_gfx950(2, 7168, 3584)
    assert supports_gluon_mm_a16w16_decode_add3_gfx950(32, 7168, 3584)
    # lat_up has no M=64 bucket; split-K shapes cannot fuse the addends.
    assert not supports_gluon_mm_a16w16_decode_add3_gfx950(33, 7168, 3584)
    assert not supports_gluon_mm_a16w16_decode_add3_gfx950(16, 3584, 7168)
    assert not supports_gluon_mm_a16w16_decode_add3_gfx950(1, 7168, 3584)


def test_decode_add3_row_count_does_not_recompile() -> None:
    torch.manual_seed(0)
    n, k = 7168, 3584
    a = torch.randn((32, k), device="cuda", dtype=torch.bfloat16) * 0.25
    b = torch.randn((n, k), device="cuda", dtype=torch.bfloat16) * 0.25
    addend = torch.randn((32, n), device="cuda", dtype=torch.bfloat16)

    def run(rows):
        launch_gluon_mm_a16w16_decode_add3_gfx950(
            a[:rows], b, addend[:rows], addend[:rows]
        )

    gluon_mm_a16w16_medium_gfx950.device_caches.clear()
    for rows in (3, 4, 5, 8, 9, 17):
        run(rows)
    with assert_no_triton_compile(gluon_mm_a16w16_medium_gfx950):
        for rows in (2, 6, 7, 12, 14, 16, 20, 27, 31, 32):
            run(rows)


# Qwen3.8-Flash-Next TP2 decode shapes, each at a ragged row count inside
# its bucket: split-K over every K tile in flight (32-row MTP fc, indexer),
# 64-row tiles (router), and split-K with a few buffers.
_QWEN38_DECODE_CASES = [
    pytest.param((20, 2560, 3072), id="out-proj"),
    pytest.param((30, 512, 2560), id="router-all-tiles-in-flight"),
    pytest.param((50, 512, 2560), id="router-64-row-tile"),
    pytest.param((3, 640, 2560), id="indexer"),
    pytest.param((27, 640, 2560), id="indexer-all-tiles-in-flight"),
    pytest.param((16, 2560, 2560), id="mtp-fc"),
    pytest.param((21, 2560, 2560), id="mtp-fc-all-tiles-in-flight"),
]


@pytest.mark.parametrize("shape", _QWEN38_DECODE_CASES)
def test_qwen38_decode_gemm_matches_reference(shape: tuple[int, int, int]) -> None:
    torch.manual_seed(0)
    m, n, k = shape
    assert supports_gluon_mm_a16w16_decode_gfx950(m, n, k)
    b = torch.randn((n, k), device="cuda", dtype=torch.bfloat16) * 0.25
    for _ in range(2):
        a = torch.randn((m, k), device="cuda", dtype=torch.bfloat16) * 0.25
        out = _decode(a, b, torch.bfloat16)
        torch.testing.assert_close(out, torch.mm(a, b.T), atol=1e-4, rtol=_RTOL)


# K3.1 indexer verify row counts between buckets; wq_b up to 16 rows takes the GEMV.
_K31_INDEXER_DECODE_CASES = [
    pytest.param((27, 4096, 1536), id="wq-b-32-row-tile"),
    pytest.param((61, 4096, 1536), id="wq-b-64-row-bucket"),
    pytest.param((4, 160, 7168), id="key-weights-splitk-c1"),
    pytest.param((13, 160, 7168), id="key-weights-splitk-ragged"),
    pytest.param((64, 160, 7168), id="key-weights-splitk-c16"),
]


@pytest.mark.parametrize("shape", _K31_INDEXER_DECODE_CASES)
def test_k31_indexer_decode_gemm_matches_reference(
    shape: tuple[int, int, int],
) -> None:
    torch.manual_seed(0)
    m, n, k = shape
    assert supports_gluon_mm_a16w16_decode_gfx950(m, n, k)
    b = torch.randn((n, k), device="cuda", dtype=torch.bfloat16) * 0.25
    for _ in range(2):
        a = torch.randn((m, k), device="cuda", dtype=torch.bfloat16) * 0.25
        out = _decode(a, b, torch.bfloat16)
        torch.testing.assert_close(out, torch.mm(a, b.T), atol=1e-4, rtol=_RTOL)


def test_k31_indexer_decode_row_count_does_not_recompile() -> None:
    torch.manual_seed(0)
    n, k = 160, 7168
    a = torch.randn((64, k), device="cuda", dtype=torch.bfloat16) * 0.25
    b = torch.randn((n, k), device="cuda", dtype=torch.bfloat16) * 0.25

    def run(rows):
        out = _decode(a[:rows], b, torch.bfloat16)
        torch.testing.assert_close(out, torch.mm(a[:rows], b.T), atol=1e-4, rtol=_RTOL)

    for rows in (3, 4, 5, 8, 9, 16, 17, 32, 33, 64):
        run(rows)
    with assert_no_triton_compile(gluon_mm_a16w16_medium_gfx950):
        for rows in (2, 6, 7, 12, 14, 20, 27, 40, 48, 61):
            run(rows)


def _absorbed_query_operands(rows: int, heads: int = 12):
    """q [rows, heads, 192] and the query buffer [rows, heads, 576] it absorbs into."""
    nope, rope, latent = 128, 64, 512
    q = torch.randn((rows, heads, nope + rope), device="cuda", dtype=torch.bfloat16)
    query = torch.full(
        (rows, heads, latent + rope), 7.0, device="cuda", dtype=torch.bfloat16
    )
    # K-major w_kc: the bmm operand w_kc.transpose(1, 2) is contiguous [H, N, K].
    w_kc_t = torch.randn((heads, latent, nope), device="cuda", dtype=torch.bfloat16)
    return q[..., :nope].transpose(0, 1), w_kc_t * 0.1, query


@pytest.mark.parametrize("rows", [1, 4, 13, 37, 64, 100])
def test_absorbed_query_bmm_fills_only_latent_channels(rows: int) -> None:
    torch.manual_seed(0)
    q_nope, w_kc_t, query = _absorbed_query_operands(rows)

    bmm(q_nope, w_kc_t, out=query[..., :512].transpose(0, 1))

    expected = torch.bmm(q_nope.float(), w_kc_t.float().transpose(1, 2))
    torch.testing.assert_close(
        query[..., :512].transpose(0, 1).float(), expected, atol=1e-3, rtol=_RTOL
    )
    assert torch.all(query[..., 512:] == 7.0)


def test_absorbed_query_bmm_row_count_does_not_recompile() -> None:
    torch.manual_seed(0)
    q_nope, w_kc_t, query = _absorbed_query_operands(64)

    def run(rows):
        out = query[:rows, :, :512].transpose(0, 1)
        bmm(q_nope[:, :rows], w_kc_t, out=out)
        expected = torch.bmm(q_nope[:, :rows].float(), w_kc_t.float().transpose(1, 2))
        torch.testing.assert_close(out.float(), expected, atol=1e-3, rtol=_RTOL)

    for rows in (1, 2, 16, 17, 32):
        run(rows)
    with assert_no_triton_compile(gluon_mm_a16w16_medium_gfx950):
        for rows in (3, 5, 8, 9, 15, 24, 33, 48, 61, 64):
            run(rows)


@pytest.mark.parametrize("rows", [1, 4, 29, 64, 100])
def test_value_projection_bmm_matches_reference(rows: int) -> None:
    torch.manual_seed(0)
    heads, latent, value = 12, 512, 128
    attn = torch.randn((rows, heads, latent), device="cuda", dtype=torch.bfloat16)
    # K-major w_vc: the bmm operand w_vc.transpose(1, 2) is contiguous [H, N, K].
    w_vc_t = (
        torch.randn((heads, value, latent), device="cuda", dtype=torch.bfloat16) * 0.05
    )
    out = torch.empty((rows, heads * value), device="cuda", dtype=torch.bfloat16)

    bmm(
        attn.transpose(0, 1),
        w_vc_t,
        out=out.view(rows, heads, value).transpose(0, 1),
    )

    expected = torch.einsum("mhk,hnk->mhn", attn.float(), w_vc_t.float())
    torch.testing.assert_close(
        out.view(rows, heads, value).float(), expected, atol=1e-3, rtol=_RTOL
    )


def test_value_projection_bmm_row_count_does_not_recompile() -> None:
    torch.manual_seed(0)
    heads, latent, value = 12, 512, 128
    attn = torch.randn((64, heads, latent), device="cuda", dtype=torch.bfloat16)
    w_vc_t = (
        torch.randn((heads, value, latent), device="cuda", dtype=torch.bfloat16) * 0.05
    )
    out = torch.empty((64, heads, value), device="cuda", dtype=torch.bfloat16)

    def run(rows):
        bmm(attn[:rows].transpose(0, 1), w_vc_t, out=out[:rows].transpose(0, 1))
        expected = torch.einsum("mhk,hnk->mhn", attn[:rows].float(), w_vc_t.float())
        torch.testing.assert_close(out[:rows].float(), expected, atol=1e-3, rtol=_RTOL)

    for rows in (1, 2, 16, 17, 32, 33, 64):
        run(rows)
    with assert_no_triton_compile(gluon_mm_a16w16_medium_gfx950):
        for rows in (3, 5, 8, 9, 15, 24, 31, 40, 48, 61):
            run(rows)


def test_grouped_bmm_without_a_config_matches_reference() -> None:
    torch.manual_seed(0)
    a = torch.randn((12, 4, 512), device="cuda", dtype=torch.bfloat16)
    b = torch.randn((12, 512, 512), device="cuda", dtype=torch.bfloat16) * 0.05

    actual = bmm(a, b)

    expected = torch.bmm(a.float(), b.float().transpose(1, 2))
    torch.testing.assert_close(actual.float(), expected, atol=1e-3, rtol=_RTOL)


def _gated_value_operands(rows: int):
    """Attended latent, K-major w_vc [H, latent, value], and a column-slice gate."""
    heads, latent, value = 12, 512, 128
    attn = torch.randn((rows, heads, latent), device="cuda", dtype=torch.bfloat16)
    w_vc = (
        torch.randn((heads, value, latent), device="cuda", dtype=torch.bfloat16) * 0.05
    ).transpose(1, 2)
    # The model's gate is a column slice of the fused a-projection output.
    fused = torch.randn(
        (rows, 704 + heads * value), device="cuda", dtype=torch.bfloat16
    )
    return attn, w_vc, fused[:, 704:]


@pytest.mark.parametrize("rows", [1, 4, 37, 64])
def test_grouped_decode_gemm_gate_matches_gemm_then_sigmoid_mul(rows: int) -> None:
    from tokenspeed_kernel.ops.activation.triton import sigmoid_mul

    torch.manual_seed(0)
    attn, w_vc, gate = _gated_value_operands(rows)
    weight = w_vc.transpose(1, 2)
    counters = torch.zeros(1, device="cuda", dtype=torch.int32)
    expected = torch.empty((rows, 12 * 128), device="cuda", dtype=torch.bfloat16)
    launch_gluon_grouped_mm_a16w16_decode_gfx950(
        attn, weight, counters, False, out=expected.view(rows, 12, 128)
    )
    sigmoid_mul(expected, gate)

    actual = torch.empty_like(expected)
    launch_gluon_grouped_mm_a16w16_decode_gfx950(
        attn,
        weight,
        counters,
        False,
        out=actual.view(rows, 12, 128),
        gate=gate.view(rows, 12, 128),
    )

    assert torch.equal(actual, expected)


@pytest.mark.parametrize("gated", [False, True])
@pytest.mark.parametrize("rows", [1, 4, 29, 64, 100])
def test_k_major_value_projection_matches_reference(rows: int, gated: bool) -> None:
    from tokenspeed_kernel.ops.attention.mla import mla_project_value

    torch.manual_seed(0)
    attn, w_vc, gate = _gated_value_operands(rows)
    gate = gate if gated else None

    out = mla_project_value(attn, w_vc, gate=gate)

    expected = torch.einsum("mhk,hkn->mhn", attn.float(), w_vc.float()).flatten(1)
    expected = expected.to(torch.bfloat16).float()
    if gated:
        expected = expected * torch.sigmoid(gate.float())
    torch.testing.assert_close(out.float(), expected, atol=1e-3, rtol=2**-6)


def test_k_major_value_projection_row_count_does_not_recompile() -> None:
    from tokenspeed_kernel.ops.attention.mla import mla_project_value

    torch.manual_seed(0)
    attn, w_vc, gate = _gated_value_operands(64)
    out = torch.empty((64, 12 * 128), device="cuda", dtype=torch.bfloat16)

    def run(rows):
        mla_project_value(attn[:rows], w_vc, gate=gate[:rows], out=out[:rows])

    for rows in (1, 2, 16, 17, 32, 33, 64):
        run(rows)
    with assert_no_triton_compile(gluon_mm_a16w16_medium_gfx950):
        for rows in (3, 5, 8, 9, 15, 24, 31, 40, 48, 61):
            run(rows)


def test_grouped_decode_gemm_splitk_writes_strided_out() -> None:
    torch.manual_seed(0)
    m, g, n, k = 8, 2, 1024, 4096
    x = torch.randn((m, g, k), device="cuda", dtype=torch.bfloat16) * 0.25
    weight = torch.randn((g, n, k), device="cuda", dtype=torch.bfloat16) * 0.25
    backing = torch.full((m, g, n + 64), 7.0, device="cuda", dtype=torch.bfloat16)
    out = backing[..., :n]
    counters = zeroed_state(
        gluon_grouped_mm_a16w16_decode_num_counters_gfx950(m, g, n, k),
        dtype=torch.int32,
        device=x.device,
    )

    launch_gluon_grouped_mm_a16w16_decode_gfx950(x, weight, counters, False, out=out)

    expected = torch.einsum("mgk,gnk->mgn", x.float(), weight.float())
    torch.testing.assert_close(out.float(), expected, atol=1e-2, rtol=_RTOL)
    assert torch.all(backing[..., n:] == 7.0)


def test_qwen38_decode_row_count_does_not_recompile() -> None:
    torch.manual_seed(0)
    n, k = 512, 2560
    a = torch.randn((64, k), device="cuda", dtype=torch.bfloat16) * 0.25
    b = torch.randn((n, k), device="cuda", dtype=torch.bfloat16) * 0.25

    def run(rows):
        out = _decode(a[:rows], b, torch.bfloat16)
        torch.testing.assert_close(out, torch.mm(a[:rows], b.T), atol=1e-4, rtol=_RTOL)

    for rows in (3, 4, 5, 8, 9, 16, 17, 32, 33, 64):
        run(rows)
    with assert_no_triton_compile(gluon_mm_a16w16_medium_gfx950):
        for rows in (2, 6, 7, 12, 14, 20, 27, 40, 48, 61):
            run(rows)


# A long-N decode projection and a split-K long-K one, at ragged rows.
_DECODE_SHAPES = [(8192, 1024), (512, 8192)]


@pytest.mark.parametrize("n,k", _DECODE_SHAPES)
# 24-64 rows are speculative verify batches of 3-8 requests.
@pytest.mark.parametrize("m", [2, 3, 7, 16, 24, 33, 40, 64])
def test_decode_gemm_is_selected_and_matches_reference(m: int, n: int, k: int) -> None:
    torch.manual_seed(0)
    assert supports_gluon_mm_a16w16_decode_gfx950(m, n, k)
    b = torch.randn((n, k), device="cuda", dtype=torch.bfloat16) * 0.25
    for _ in range(2):
        a = torch.randn((m, k), device="cuda", dtype=torch.bfloat16) * 0.25
        out = _decode(a, b, torch.bfloat16)
        torch.testing.assert_close(out, torch.mm(a, b.T), atol=1e-4, rtol=_RTOL)
        # gemm.mm hands the rows to the same kernel.
        assert torch.equal(mm(a, b), out)


def test_decode_gemm_serves_muse_pre_expert_verify_rows() -> None:
    # Speculative verify sends 65-172 rows (9-21 requests of eight) through the
    # Muse TP8 pre-expert projection; the 128- and 256-row buckets take them,
    # with rows a runtime argument inside each bucket.
    torch.manual_seed(0)
    n, k = 512, 8192
    a = torch.randn((256, k), device="cuda", dtype=torch.bfloat16) * 0.25
    b = torch.randn((n, k), device="cuda", dtype=torch.bfloat16) * 0.25

    def run(rows):
        assert supports_gluon_mm_a16w16_decode_gfx950(rows, n, k)
        out = _decode(a[:rows], b, torch.bfloat16)
        torch.testing.assert_close(out, torch.mm(a[:rows], b.T), atol=1e-4, rtol=_RTOL)
        assert torch.equal(mm(a[:rows], b), out)

    # Warm both integer classes (divisible by 16 or not) of each bucket.
    for rows in (65, 128, 129, 256):
        run(rows)
    with assert_no_triton_compile(gluon_mm_a16w16_medium_gfx950):
        for rows in (72, 100, 172, 200):
            run(rows)


def test_medium_coalesced_store_masks_strided_out() -> None:
    # 128x128 medium tiles store through the coalesced row layout; a padded
    # row stride must stay inside the output view.
    torch.manual_seed(0)
    m, n, k = 768, 3584, 7168
    a = torch.randn((m, k), device="cuda", dtype=torch.bfloat16) * 0.25
    b = torch.randn((n, k), device="cuda", dtype=torch.bfloat16) * 0.25
    backing = torch.full((m, n + 8), 7.0, device="cuda", dtype=torch.bfloat16)
    out = backing[:, :n]

    actual = launch_gluon_mm_a16w16_medium_gfx950(a, b, torch.bfloat16, out=out)

    assert actual is out
    torch.testing.assert_close(out, torch.mm(a, b.T), atol=1e-2, rtol=_RTOL)
    assert torch.all(backing[:, n:] == 7.0)


@pytest.mark.parametrize("m", [9, 13, 16])
def test_column_shard_equals_the_replicated_decode_gemm_columns(m: int) -> None:
    """Each 1024-column shard of the [8192, 4096] decode GEMM is bitwise its columns."""
    torch.manual_seed(60 + m)
    a = torch.randn((m, 4096), device="cuda", dtype=torch.bfloat16) * 0.25
    b = torch.randn((8192, 4096), device="cuda", dtype=torch.bfloat16) * 0.25
    assert supports_gluon_mm_a16w16_column_shard_gfx950(m, 8192, 1024, 4096)
    full = launch_gluon_mm_a16w16_decode_gfx950(
        a, b, torch.bfloat16, counters=torch.zeros(1, dtype=torch.int32, device="cuda")
    )
    for shard in (0, 5):
        out = torch.empty(m, 1024, dtype=torch.bfloat16, device="cuda")
        launch_gluon_mm_a16w16_column_shard_gfx950(
            a, b[shard * 1024 : (shard + 1) * 1024], out
        )
        assert torch.equal(out, full[:, shard * 1024 : (shard + 1) * 1024]), shard


def test_column_shard_row_count_does_not_recompile() -> None:
    torch.manual_seed(77)
    a = torch.randn((16, 4096), device="cuda", dtype=torch.bfloat16) * 0.25
    b = torch.randn((1024, 4096), device="cuda", dtype=torch.bfloat16) * 0.25
    out = torch.empty(16, 1024, dtype=torch.bfloat16, device="cuda")
    # One warm-up per integer specialization class of the row count (16 is
    # divisible by 16, 9 is not); every other row count reuses one of them.
    for m in (9, 16):
        launch_gluon_mm_a16w16_column_shard_gfx950(a[:m].contiguous(), b, out[:m])
    with assert_no_triton_compile(gluon_mm_a16w16_medium_gfx950):
        for m in range(9, 17):
            launch_gluon_mm_a16w16_column_shard_gfx950(a[:m].contiguous(), b, out[:m])
