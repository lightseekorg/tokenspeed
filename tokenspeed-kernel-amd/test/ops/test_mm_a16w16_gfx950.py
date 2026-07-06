from __future__ import annotations

import inspect

import torch
from tokenspeed_kernel_amd.ops.gemm.mm_a16w16_gfx950 import (
    _choose_mfma_lds_mediumm_config,
    _supports_mfma_lds_smallm,
    _use_mfma_lds_largem,
    _use_mfma_lds_mediumm,
    _use_mfma_lds_smallm,
    _use_warp_reduce_smallm,
    gluon_mm_a16w16_gfx950,
    gluon_mm_a16w16_mfma_lds_mediumm_gfx950,
    gluon_mm_a16w16_mfma_lds_smallm_gfx950,
    gluon_mm_a16w16_warp_reduce_smallm_gfx950,
)
from tokenspeed_kernel_amd.ops.gemm.mm_a16w16_largem_gfx950 import (
    _supports_largem_shape,
    gluon_mm_a16w16_largem_gfx950,
)


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


def test_use_mfma_lds_mediumm_rejects_non_target_shapes() -> None:
    assert _use_mfma_lds_mediumm(8, 1280, 512)
    assert _use_mfma_lds_mediumm(8, 1280, 1024)
    assert _use_mfma_lds_mediumm(64, 1280, 2880)
    assert _use_mfma_lds_mediumm(128, 4096, 4096)
    assert not _use_mfma_lds_mediumm(4, 1280, 1024)
    assert not _use_mfma_lds_mediumm(8, 1344, 1024)
    assert not _use_mfma_lds_mediumm(16, 1280, 2880)
    assert not _use_mfma_lds_mediumm(256, 1280, 1024)
    assert not _use_mfma_lds_mediumm(512, 4096, 4096)
    assert not _use_mfma_lds_mediumm(1024, 8192, 8192)


def test_supports_largem_shape_covers_aligned_prefill_tiles() -> None:
    assert _supports_largem_shape(256, 256, 256)
    assert _supports_largem_shape(512, 4096, 4096)
    assert _supports_largem_shape(2048, 8192, 8192)


def test_supports_largem_shape_rejects_unaligned_or_medium_shapes() -> None:
    assert not _supports_largem_shape(128, 4096, 4096)
    assert not _supports_largem_shape(256, 128, 256)
    assert not _supports_largem_shape(256, 256, 128)
    assert not _supports_largem_shape(256, 1280, 2880)
    assert not _supports_largem_shape(384, 4096, 4096)
    assert not _supports_largem_shape(512, 3968, 4096)


def test_use_mfma_lds_largem_routes_only_large_competitive_shapes() -> None:
    assert _use_mfma_lds_largem(2048, 4096, 4096)
    assert _use_mfma_lds_largem(2048, 8192, 8192)
    assert not _use_mfma_lds_largem(256, 1280, 1024)
    assert not _use_mfma_lds_largem(1024, 8192, 8192)
    assert not _use_mfma_lds_largem(2048, 1280, 2880)


def test_use_warp_reduce_covers_small_k_decode_shapes() -> None:
    assert _use_warp_reduce_smallm(1, 1280, 1024)
    assert _use_warp_reduce_smallm(2, 2560, 2048)
    assert _use_warp_reduce_smallm(4, 1280, 512)
    assert _use_warp_reduce_smallm(4, 1280, 1024)


def test_use_warp_reduce_rejects_splitk_target_shapes() -> None:
    assert not _use_warp_reduce_smallm(1, 1280, 2880)
    assert not _use_warp_reduce_smallm(2, 4608, 4096)
    assert not _use_warp_reduce_smallm(4, 2560, 2048)
    assert not _use_warp_reduce_smallm(4, 4608, 4096)
    assert not _use_warp_reduce_smallm(8, 1280, 512)


def test_supports_mfma_lds_covers_splitk_shapes() -> None:
    assert _supports_mfma_lds_smallm(1, 4096, 4096)
    assert _supports_mfma_lds_smallm(2, 4096, 4096)
    assert _supports_mfma_lds_smallm(1, 1280, 2880)
    assert _supports_mfma_lds_smallm(4, 1280, 1024)
    assert _supports_mfma_lds_smallm(4, 1280, 1280)
    assert _supports_mfma_lds_smallm(4, 2560, 2048)
    assert _supports_mfma_lds_smallm(4, 4608, 7168)
    assert _supports_mfma_lds_smallm(4, 8192, 8192)


def test_supports_mfma_lds_rejects_non_target_shapes() -> None:
    assert not _supports_mfma_lds_smallm(4, 3968, 4096)
    assert not _supports_mfma_lds_smallm(4, 1280, 960)
    assert not _supports_mfma_lds_smallm(4, 1280, 1216)
    assert not _supports_mfma_lds_smallm(4, 4224, 4096)
    assert not _supports_mfma_lds_smallm(3, 4096, 4096)
    assert not _supports_mfma_lds_smallm(8, 8192, 4096)


def test_use_mfma_lds_routes_supported_non_warp_shapes() -> None:
    assert _use_mfma_lds_smallm(1, 4096, 4096)
    assert _use_mfma_lds_smallm(2, 4096, 4096)
    assert _use_mfma_lds_smallm(1, 1280, 2880)
    assert _use_mfma_lds_smallm(4, 1280, 1280)
    assert _use_mfma_lds_smallm(4, 2560, 2048)
    assert _use_mfma_lds_smallm(4, 4608, 7168)
    assert _use_mfma_lds_smallm(4, 8192, 8192)
    assert not _use_mfma_lds_smallm(1, 2560, 2048)
    assert not _use_mfma_lds_smallm(2, 2560, 2048)
    assert not _use_mfma_lds_smallm(4, 1280, 512)
    assert not _use_mfma_lds_smallm(4, 1280, 1024)
