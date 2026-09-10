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

"""Blackwell gated-residual mix with separate weight, activation and MMA warps."""

from __future__ import annotations

import dataclasses
import threading
import weakref

import torch
from tokenspeed_kernel.platform import (
    ArchVersion,
    CapabilityRequirement,
    current_platform,
    pdl_enabled,
)
from tokenspeed_kernel.registry import Priority, register_kernel
from tokenspeed_kernel.signature import format_signatures

try:
    from cuda.bindings.driver import CUstream
    from cutlass.cute import experimental as cute_ext
    from cutlass.cute.runtime import from_dlpack
    from tokenspeed_kernel.thirdparty.cute_dsl.hc_splitk import (
        SplitKDenseGemmKernel,
        SplitKTactic,
        _run_mix_gemm,
    )

    _CUTEDSL_AVAILABLE = True
except ImportError:
    _CUTEDSL_AVAILABLE = False

_LOWRANK_ALIGNMENT = 128
_MAX_ROWS = 32
_PRODUCTION_UP_SHAPE = (4 * 2560, 320)


def _round_up(value: int, alignment: int) -> int:
    return (value + alignment - 1) // alignment * alignment


@dataclasses.dataclass
class _CachedPaddedWeight:
    source: weakref.ReferenceType[torch.Tensor]
    padded: torch.Tensor


_PADDED_WEIGHT_LOCK = threading.Lock()
_PADDED_UP_WEIGHTS: dict[tuple[int, int], _CachedPaddedWeight] = {}


def _copy_padded_up_weight(
    padded: torch.Tensor, up_weight: torch.Tensor, lowrank: int
) -> None:
    with torch.no_grad():
        padded.zero_()
        # Adjacent GEMM output columns belong to the same hidden position,
        # allowing one epilogue CTA to reduce all four residual branches.
        padded.view(2560, 4, padded.shape[1])[:, :, :lowrank].copy_(
            up_weight.view(4, 2560, lowrank).permute(1, 0, 2)
        )


def _prepare_padded_up_weight(up_weight: torch.Tensor, lowrank: int) -> bool:
    """Create or refresh a graph-stable padded weight outside forward.

    Args:
        up_weight: Source mix-up weight shaped ``[wide, lowrank]``.
        lowrank: Unpadded rank of the source weight.

    Returns:
        Whether the CuTeDSL backend uses a padded allocation for this weight.
    """
    padded_lowrank = _round_up(lowrank, _LOWRANK_ALIGNMENT)
    if (
        not up_weight.is_cuda
        or padded_lowrank == lowrank
        or tuple(up_weight.shape) != _PRODUCTION_UP_SHAPE
        or up_weight.dtype not in (torch.bfloat16, torch.float16)
        or not up_weight.is_contiguous()
        or not _CUTEDSL_AVAILABLE
        or not current_platform().is_blackwell
    ):
        return False
    if torch.cuda.is_current_stream_capturing():
        raise RuntimeError(
            "gated-residual weight preparation must run outside CUDA Graph capture"
        )
    device_index = up_weight.device.index
    if device_index is None:
        device_index = torch.cuda.current_device()
    key = (device_index, up_weight.data_ptr())
    with _PADDED_WEIGHT_LOCK:
        cached = _PADDED_UP_WEIGHTS.get(key)
        if cached is not None and cached.source() is up_weight:
            _copy_padded_up_weight(cached.padded, up_weight, lowrank)
            return True
        padded = torch.empty(
            (up_weight.shape[0], padded_lowrank),
            dtype=up_weight.dtype,
            device=up_weight.device,
        )
        _copy_padded_up_weight(padded, up_weight, lowrank)
        _PADDED_UP_WEIGHTS[key] = _CachedPaddedWeight(
            source=weakref.ref(up_weight), padded=padded
        )
        return True


def _find_prepared_padded_up_weight(up_weight: torch.Tensor) -> torch.Tensor | None:
    """Look up a prepared weight without allocating or changing cache state."""
    device_index = up_weight.device.index
    if device_index is None:
        device_index = torch.cuda.current_device()
    key = (device_index, up_weight.data_ptr())
    cached = _PADDED_UP_WEIGHTS.get(key)
    if cached is not None and cached.source() is up_weight:
        return cached.padded
    return None


def _get_prepared_padded_up_weight(up_weight: torch.Tensor) -> torch.Tensor:
    padded = _find_prepared_padded_up_weight(up_weight)
    if padded is not None:
        return padded
    raise RuntimeError(
        "CuTeDSL gated-residual weight was not prepared; call "
        "prepare_gated_residual_weight_cache after loading weights and before "
        "forward or CUDA Graph capture"
    )


if _CUTEDSL_AVAILABLE:

    def _cute_tensor(tensor: torch.Tensor, leading_dim: int):
        # Parameter views can retain requires_grad even in inference contexts.
        # Detach only this export view; keep the storage and source flags intact.
        return from_dlpack(tensor.detach(), assumed_align=32).mark_layout_dynamic(
            leading_dim=leading_dim
        )

    def _operands(x: torch.Tensor, weight: torch.Tensor, out: torch.Tensor):
        return (
            _cute_tensor(weight.unsqueeze(0), 2),
            _cute_tensor(x.unsqueeze(0).transpose(-2, -1), 1),
            _cute_tensor(out.unsqueeze(0).transpose(-2, -1), 1),
        )

    _COMPILE_LOCK = threading.Lock()
    _COMPILED: dict[tuple, object] = {}

    def _compiled_gemm(
        device: torch.device,
        dtype: torch.dtype,
        rows: int,
        mode: str,
        enable_pdl: bool,
        scale: float,
    ):
        # Larger row counts remain available for explicit backend comparisons;
        # the automatic policy uses the measured T<=8 region.
        tactic = (
            SplitKTactic(64, 8, 16, 6 if rows <= 8 else 4)
            if mode == "hc_down"
            else SplitKTactic(128, 8, 1, 6 if rows <= 8 else 3)
        )
        key = (device.index, dtype, tactic, mode, enable_pdl, scale)
        cached = _COMPILED.get(key)
        if cached is not None:
            return cached
        with _COMPILE_LOCK:
            cached = _COMPILED.get(key)
            if cached is not None:
                return cached
            with torch.cuda.device(device):
                gemm = SplitKDenseGemmKernel(
                    tactic=tactic,
                    use_pdl=enable_pdl,
                    epilogue_mode=mode,
                    epilogue_scale=scale,
                    epilogue_group=4 if mode == "gate" else 1,
                )
                x = torch.empty((8, 128), device=device, dtype=dtype)
                weight = torch.empty((64, 128), device=device, dtype=dtype)
                result = torch.empty((8, 64), device=device, dtype=dtype)
                aux = torch.empty(
                    (8, 4 if mode == "hc_down" else 16), device=device, dtype=dtype
                )
                cached = cute_ext.compile(
                    _run_mix_gemm,
                    gemm,
                    *_operands(x, weight, result),
                    _cute_tensor(x, 1),
                    _cute_tensor(aux, 1),
                    CUstream(torch.cuda.current_stream(device).cuda_stream),
                )
            _COMPILED[key] = cached
            return cached

    @register_kernel(
        "residual",
        "hyperconnection_mix",
        name="cute_dsl_hyperconnection_mix",
        solution="cute_dsl",
        capability=CapabilityRequirement(
            vendors=frozenset({"nvidia"}),
            min_arch_version=ArchVersion(10, 0),
            max_arch_version=ArchVersion(10, 9),
        ),
        signatures=format_signatures(
            ("normalized", "projection_weight", "up_weight"),
            "dense",
            {torch.bfloat16, torch.float16},
        ),
        traits={
            "num_tokens": frozenset(range(1, 9)),
            "hc_count": frozenset({4}),
            "hidden_size": frozenset({2560}),
            "lowrank": frozenset({320}),
            "has_inject": frozenset({False, True}),
            "contiguous": frozenset({True}),
            "folded_scale": frozenset({False, True}),
            "deterministic": frozenset({False, True}),
            "capturing": frozenset({False, True}),
            "prepared_up_weight": frozenset({True}),
            "tma_aligned": frozenset({True}),
            "pdl": frozenset({True}),
        },
        priority=Priority.SPECIALIZED + 2,
        tags={"cute_dsl", "decode", "determinism", "latency"},
    )
    def cute_dsl_hyperconnection_mix(
        normalized: torch.Tensor,
        projection_weight: torch.Tensor,
        up_weight: torch.Tensor,
        hc_count: int,
        hidden_size: int,
        lowrank: int,
        projection_scale: float,
        weights_independent: bool,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """Run two warp-specialized GEMMs including both fused epilogues.

        Down waits for all preceding input/weight writes and publishes SiLU,
        zero padding and inject logits. Up can stage its independent weights
        while down completes, waiting only in its activation DMA warp. Each
        invocation owns its activation and outputs, including under capture.
        """
        rows = int(normalized.shape[0])
        if (
            not current_platform().is_blackwell
            or not 1 <= rows <= _MAX_ROWS
            or normalized.dtype not in (torch.bfloat16, torch.float16)
            or hc_count != 4
            or hidden_size != 2560
            or lowrank != 320
            or not normalized.is_contiguous()
            or not projection_weight.is_contiguous()
            or not up_weight.is_contiguous()
        ):
            raise ValueError(
                "CuTeDSL HC mix requires Blackwell and contiguous BF16/FP16 "
                "production shape T<=32, hc_count=4, hidden_size=2560, "
                "lowrank=320"
            )
        padded_lowrank = _round_up(lowrank, _LOWRANK_ALIGNMENT)
        padded_up = _get_prepared_padded_up_weight(up_weight)
        if normalized.data_ptr() % 32 or projection_weight.data_ptr() % 32:
            raise ValueError("CuTeDSL HC inputs require 32-byte-aligned base addresses")
        activated = torch.empty(
            (rows, padded_lowrank), dtype=normalized.dtype, device=normalized.device
        )
        inject = torch.empty(
            (rows, hc_count), dtype=normalized.dtype, device=normalized.device
        )
        mixed = torch.empty(
            (rows, hidden_size), dtype=normalized.dtype, device=normalized.device
        )
        enable_pdl = pdl_enabled(None)
        with torch.cuda.device(normalized.device):
            stream = CUstream(torch.cuda.current_stream(normalized.device).cuda_stream)
            down = _compiled_gemm(
                normalized.device,
                normalized.dtype,
                rows,
                "hc_down",
                enable_pdl,
                projection_scale,
            )
            up = _compiled_gemm(
                normalized.device,
                normalized.dtype,
                rows,
                "gate",
                enable_pdl,
                1.0 / hc_count,
            )
            down(
                *_operands(normalized, projection_weight, activated),
                _cute_tensor(normalized, 1),
                _cute_tensor(inject, 1),
                stream,
            )
            up(
                *_operands(activated, padded_up, normalized),
                _cute_tensor(normalized, 1),
                _cute_tensor(mixed, 1),
                stream,
            )
        return mixed, inject if projection_weight.shape[0] != lowrank else None
