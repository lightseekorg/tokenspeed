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

from types import SimpleNamespace

import pytest
import torch
from utils import is_cdna4

if not is_cdna4():
    pytest.skip(
        "AMD CDNA4 is required for dense16 Gluon GEMM tests",
        allow_module_level=True,
    )


from tokenspeed_kernel.ops.gemm import mm  # noqa: E402
from tokenspeed_kernel.registry import KernelRegistry  # noqa: E402
from tokenspeed_kernel.selection import select_kernel  # noqa: E402
from tokenspeed_kernel.signature import (  # noqa: E402
    dense_tensor_format,
    format_signature,
)
from tokenspeed_kernel_amd.ops.gfx950.gemm.fp16.largem import (  # noqa: E402
    _largem_launch_metadata,
    _supports_largem_shape,
    gluon_mm_a16w16_prefill_gfx950,
    launch_gluon_mm_a16w16_prefill_gfx950,
)
from tokenspeed_kernel_amd.ops.gfx950.gemm.fp16.mm import (  # noqa: E402
    _choose_mfma_lds_mediumm_config,
    _dense16_bmm_launch_metadata,
    _dense16_mediumm_launch_metadata,
    _dense16_mm_launch_metadata,
    _dense16_splitk_launch_metadata,
    _dense16_splitk_reduce_launch_metadata,
    _get_partial_scratch,
    _mfma_lds_smallm_reduce_kernel,
    _supports_mfma_lds_smallm,
    _use_mfma_lds_largem,
    _use_mfma_lds_mediumm,
    _use_mfma_lds_smallm,
    _use_warp_reduce_smallm,
    gluon_bmm_a16w16_gfx950,
    gluon_mm_a16w16_gfx950,
    gluon_mm_a16w16_medium_gfx950,
    gluon_mm_a16w16_splitk_gfx950,
    gluon_mm_a16w16_warp_gfx950,
    launch_gluon_bmm_a16w16_gfx950,
    launch_gluon_mm_a16w16_medium_gfx950,
    launch_gluon_mm_a16w16_splitk_gfx950,
    launch_gluon_mm_a16w16_warp_gfx950,
)

_REGISTERED_CASES = [
    pytest.param(
        "gluon_mm_a16w16_prefill_gfx950",
        (2816, 3072, 512),
        (
            (2560, 3072, 512),
            (2817, 3072, 512),
            (4352, 3072, 512),
            (2816, 3328, 512),
            (2816, 3072, 640),
        ),
        id="prefill",
    ),
]

_REGISTERED_IMPLEMENTATIONS = {
    "gluon_mm_a16w16_prefill_gfx950": gluon_mm_a16w16_prefill_gfx950,
}

_UNREGISTERED_SMALL_M_KERNELS = (
    "gluon_mm_a16w16_warp_gfx950",
    "gluon_mm_a16w16_splitk_gfx950",
    "gluon_mm_a16w16_medium_gfx950",
)


@pytest.mark.parametrize("kernel_name", _UNREGISTERED_SMALL_M_KERNELS)
def test_dense16_small_m_kernels_are_not_registered(kernel_name: str) -> None:
    assert KernelRegistry.get().get_by_name(kernel_name) is None


@pytest.mark.parametrize("kernel_name,accepted,rejected", _REGISTERED_CASES)
def test_dense16_registration_selects_only_measured_problems(
    kernel_name: str,
    accepted: tuple[int, int, int],
    rejected: tuple[tuple[int, int, int], ...],
) -> None:
    spec = KernelRegistry.get().get_by_name(kernel_name)
    assert spec is not None
    registered_implementation = KernelRegistry.get().get_impl(kernel_name)
    assert registered_implementation is not None
    assert registered_implementation.__name__ == kernel_name
    assert _REGISTERED_IMPLEMENTATIONS[kernel_name].__name__ == kernel_name

    problem_filters = spec.traits["mnk_problem_filter"]
    assert len(problem_filters) == 1
    problem_filter = next(iter(problem_filters))
    assert problem_filter(*accepted)
    assert problem_filter(4096, 3072, 512)
    assert all(not problem_filter(*shape) for shape in rejected)


@pytest.mark.parametrize("kernel_name,shape,_rejected", _REGISTERED_CASES)
def test_dense16_registration_is_selected_and_correct(
    kernel_name: str,
    shape: tuple[int, int, int],
    _rejected: tuple[tuple[int, int, int], ...],
) -> None:
    m, n, k = shape
    signature = format_signature(
        a=dense_tensor_format(torch.bfloat16),
        b=dense_tensor_format(torch.bfloat16),
    )
    selected = select_kernel(
        "gemm",
        "mm",
        signature,
        traits={
            "m": m,
            "n": n,
            "k": k,
            "a_inner_stride_one": True,
            "b_inner_stride_one": True,
            "out_dtype": torch.bfloat16,
            "block_scale_layout": "canonical",
            "pdl_enabled": False,
        },
    )
    assert selected.name == kernel_name

    torch.manual_seed(0)
    a = torch.randn((m, k), device="cuda", dtype=torch.bfloat16) * 0.25
    b = torch.randn((n, k), device="cuda", dtype=torch.bfloat16) * 0.25
    out = torch.empty((m, n), device="cuda", dtype=torch.bfloat16)

    actual = mm(a, b, out=out)

    assert actual is out
    torch.testing.assert_close(
        actual,
        torch.mm(a, b.T),
        atol=2e-2,
        rtol=2e-2,
    )


_CORRECTNESS_CASES = [
    pytest.param(
        launch_gluon_mm_a16w16_warp_gfx950,
        (2, 128, 1024),
        id="warp-reduce",
    ),
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

    torch.testing.assert_close(out, torch.mm(a, b.T), atol=1e-2, rtol=1e-2)


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
    torch.testing.assert_close(out, torch.mm(a, b.T), atol=1e-2, rtol=1e-2)


def test_dense16_mm_launch_metadata_reports_flops_and_tensor_bytes() -> None:
    m, n, k = 2, 128, 512
    a = torch.empty((m, k), device="cuda", dtype=torch.bfloat16)
    b = torch.empty((n, k), device="cuda", dtype=torch.bfloat16)
    c = torch.empty((m, n), device="cuda", dtype=torch.bfloat16)
    args = {"M": m, "N": n, "K": k, "a_ptr": a, "b_ptr": b, "c_ptr": c}

    metadata = _dense16_mm_launch_metadata(
        None,
        SimpleNamespace(name="dense16"),
        args,
    )

    assert metadata == {
        "name": "dense16",
        "flops16": 2 * m * n * k,
        "bytes": (m * k + n * k + m * n) * 2,
    }
    assert gluon_mm_a16w16_warp_gfx950.launch_metadata is _dense16_mm_launch_metadata
    assert gluon_mm_a16w16_prefill_gfx950.launch_metadata is _largem_launch_metadata
    assert _largem_launch_metadata(
        None,
        SimpleNamespace(name="largem"),
        args,
    ) == {
        "name": "largem",
        "flops16": 2 * m * n * k,
        "bytes": (m * k + n * k + m * n) * 2,
    }


def test_dense16_medium_launch_metadata_reports_add3_work() -> None:
    m, n, k = 16, 128, 512
    a = torch.empty((m, k), device="cuda", dtype=torch.bfloat16)
    b = torch.empty((n, k), device="cuda", dtype=torch.bfloat16)
    c = torch.empty((m, n), device="cuda", dtype=torch.bfloat16)
    args = {
        "M": m,
        "N": n,
        "K": k,
        "a_ptr": a,
        "b_ptr": b,
        "c_ptr": c,
        "ADD3": True,
        "addend_a_ptr": c,
        "addend_b_ptr": c,
    }

    metadata = _dense16_mediumm_launch_metadata(
        None,
        SimpleNamespace(name="medium"),
        args,
    )

    assert metadata == {
        "name": "medium",
        "flops16": 2 * m * n * k,
        "flops32": 2 * m * n,
        "bytes": (m * k + n * k + 3 * m * n) * 2,
    }
    assert (
        gluon_mm_a16w16_medium_gfx950.launch_metadata
        is _dense16_mediumm_launch_metadata
    )


def test_dense16_splitk_launch_metadata_reports_scratch_traffic() -> None:
    m, n, k = 2, 7168, 4224
    split_k, reduce_m, block_n = 6, 4, 256
    a = torch.empty((m, k), device="cuda", dtype=torch.bfloat16)
    b = torch.empty((n, k), device="cuda", dtype=torch.bfloat16)
    c = torch.empty((m, n), device="cuda", dtype=torch.bfloat16)
    partial = torch.empty((n * reduce_m * split_k,), device="cuda")

    producer = _dense16_splitk_launch_metadata(
        None,
        SimpleNamespace(name="splitk"),
        {
            "M": m,
            "N": n,
            "a_ptr": a,
            "b_ptr": b,
            "partial_ptr": partial,
            "SPLIT_K": split_k,
            "PARTIAL_M": reduce_m,
            "K_TILES_PER_SPLIT": 11,
            "BLOCK_K": 64,
        },
    )
    reducer = _dense16_splitk_reduce_launch_metadata(
        (n // block_n,),
        SimpleNamespace(name="reduce"),
        {
            "partial_ptr": partial,
            "c_ptr": c,
            "BLOCK_N": block_n,
            "REDUCE_M": reduce_m,
            "OUTPUT_M": m,
            "SPLIT_K": split_k,
        },
    )

    assert producer == {
        "name": "splitk",
        "flops16": 2 * m * n * k,
        "bytes": (m * k + n * k) * 2 + n * reduce_m * split_k * 4,
    }
    assert reducer == {
        "name": "reduce",
        "flops32": m * n * (split_k - 1),
        "bytes": reduce_m * n * split_k * 4 + m * n * 2,
    }
    assert (
        gluon_mm_a16w16_splitk_gfx950.launch_metadata is _dense16_splitk_launch_metadata
    )
    assert (
        _mfma_lds_smallm_reduce_kernel.launch_metadata
        is _dense16_splitk_reduce_launch_metadata
    )


def test_dense16_bmm_launch_metadata_reports_batched_work() -> None:
    batch, n, k, block_n = 12, 512, 128, 32
    a = torch.empty((batch, 1, k), device="cuda", dtype=torch.bfloat16)
    b = torch.empty((batch, n, k), device="cuda", dtype=torch.bfloat16)
    output = torch.empty((batch, 1, n), device="cuda", dtype=torch.bfloat16)

    metadata = _dense16_bmm_launch_metadata(
        (batch * n // block_n,),
        SimpleNamespace(name="bmm"),
        {
            "N": n,
            "K": k,
            "BLOCK_N": block_n,
            "a_ptr": a,
            "b_ptr": b,
            "output_ptr": output,
        },
    )

    assert metadata == {
        "name": "bmm",
        "flops16": 2 * batch * n * k,
        "bytes": (batch * k + batch * n * k + batch * n) * 2,
    }
    assert gluon_bmm_a16w16_gfx950.launch_metadata is _dense16_bmm_launch_metadata


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
    torch.testing.assert_close(out, torch.bmm(a, weight), atol=1e-2, rtol=1e-2)


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
    torch.testing.assert_close(out, torch.mm(a, b.T), atol=1e-2, rtol=1e-2)


def test_use_warp_reduce_covers_small_k_decode_shapes() -> None:
    assert _use_warp_reduce_smallm(1, 1280, 1024)
    assert _use_warp_reduce_smallm(2, 2560, 2048)
    assert _use_warp_reduce_smallm(4, 1280, 512)
    assert _use_warp_reduce_smallm(4, 1280, 1024)


def test_use_warp_reduce_rejects_splitk_or_medium_shapes() -> None:
    assert not _use_warp_reduce_smallm(1, 1280, 2880)
    assert not _use_warp_reduce_smallm(4, 2560, 2048)
    assert not _use_warp_reduce_smallm(8, 1280, 512)


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


def test_use_splitk_is_disabled_for_default_routing() -> None:
    assert not _use_mfma_lds_smallm(1, 4096, 4096)
    assert not _use_mfma_lds_smallm(4, 2560, 2048)
    assert not _use_mfma_lds_smallm(1, 2560, 2048)
    assert not _use_mfma_lds_smallm(2, 2560, 2048)
    assert not _use_mfma_lds_smallm(4, 1280, 1024)


def test_dispatcher_falls_back_for_splitk_shapes() -> None:
    dtype = torch.bfloat16
    a = torch.empty((1, 4096), device="cuda", dtype=dtype)
    b = torch.empty((4096, 4096), device="cuda", dtype=dtype)

    assert _supports_mfma_lds_smallm(1, 4096, 4096)
    assert gluon_mm_a16w16_gfx950(a, b, dtype) is None


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


def test_use_mediumm_routes_configured_shapes() -> None:
    assert _use_mfma_lds_mediumm(8, 1280, 1024)
    assert _use_mfma_lds_mediumm(64, 1280, 2880)
    assert _use_mfma_lds_mediumm(128, 4096, 4096)
    assert _use_mfma_lds_mediumm(768, 3584, 7168)
    assert _use_mfma_lds_mediumm(512, 7168, 3584)
    assert not _use_mfma_lds_mediumm(4, 1280, 1024)
    assert not _use_mfma_lds_mediumm(256, 1280, 1024)
    assert not _use_mfma_lds_mediumm(640, 3584, 7168)


def test_supports_largem_shape_covers_aligned_prefill_tiles() -> None:
    assert _supports_largem_shape(256, 256, 256)
    assert _supports_largem_shape(2048, 8192, 8192)


def test_supports_largem_shape_rejects_unaligned_or_medium_shapes() -> None:
    assert not _supports_largem_shape(128, 4096, 4096)
    assert not _supports_largem_shape(256, 128, 256)
    assert not _supports_largem_shape(256, 256, 128)
    assert not _supports_largem_shape(256, 1280, 2880)
    assert not _supports_largem_shape(384, 4096, 4096)
    assert not _supports_largem_shape(512, 3968, 4096)


def test_use_largem_routes_only_dispatch_target_shapes() -> None:
    assert _use_mfma_lds_largem(2048, 4096, 4096)
    assert not _use_mfma_lds_largem(1024, 8192, 8192)
    assert not _use_mfma_lds_largem(2048, 1280, 2880)
