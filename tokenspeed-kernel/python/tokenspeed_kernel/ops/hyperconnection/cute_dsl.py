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

"""Blackwell CuTeDSL low-M gated-residual mix."""

from __future__ import annotations

import dataclasses
import threading
import weakref

import torch
from tokenspeed_kernel.ops.hyperconnection.triton import (
    _launch_mix_epilogue,
    _launch_projection_epilogue,
)
from tokenspeed_kernel.platform import (
    ArchVersion,
    CapabilityRequirement,
    current_platform,
)
from tokenspeed_kernel.registry import Priority, register_kernel
from tokenspeed_kernel.signature import format_signatures
from tokenspeed_kernel.thirdparty.cute_dsl.ll_bf16 import ll_bf16_router

_LOWRANK_ALIGNMENT = 128
_MAX_ROWS = 32
_CUTEDSL_AVAILABLE = ll_bf16_router.is_available()


def _round_up(value: int, alignment: int) -> int:
    return (value + alignment - 1) // alignment * alignment


@dataclasses.dataclass
class _CachedPaddedWeight:
    source: weakref.ReferenceType[torch.Tensor]
    version: int
    padded: torch.Tensor


_PADDED_WEIGHT_LOCK = threading.Lock()
_PADDED_UP_WEIGHTS: dict[tuple[int, int], _CachedPaddedWeight] = {}


def _copy_padded_up_weight(
    padded: torch.Tensor, up_weight: torch.Tensor, lowrank: int
) -> None:
    with torch.no_grad():
        padded.zero_()
        padded[:, :lowrank].copy_(up_weight)


def _refresh_padded_up_weight(up_weight: torch.Tensor, lowrank: int) -> bool:
    """Refresh an existing graph-stable padded weight in place.

    Args:
        up_weight: Source mix-up weight shaped ``[wide, lowrank]``.
        lowrank: Unpadded rank of the source weight.

    Returns:
        Whether a cached padded allocation existed and was refreshed.
    """
    padded_lowrank = _round_up(lowrank, _LOWRANK_ALIGNMENT)
    if not up_weight.is_cuda or padded_lowrank == lowrank:
        return False
    device_index = up_weight.device.index
    if device_index is None:
        device_index = torch.cuda.current_device()
    key = (device_index, up_weight.data_ptr())
    with _PADDED_WEIGHT_LOCK:
        cached = _PADDED_UP_WEIGHTS.get(key)
        if cached is None or cached.source() is not up_weight:
            return False
        _copy_padded_up_weight(cached.padded, up_weight, lowrank)
        cached.version = int(up_weight._version)
        return True


def _padded_up_weight(up_weight: torch.Tensor, lowrank: int) -> torch.Tensor:
    """Return a graph-stable, zero-padded CuTe input weight."""
    padded_lowrank = _round_up(lowrank, _LOWRANK_ALIGNMENT)
    if padded_lowrank == lowrank:
        return up_weight
    device_index = up_weight.device.index
    if device_index is None:
        device_index = torch.cuda.current_device()
    key = (device_index, up_weight.data_ptr())
    version = int(up_weight._version)
    cached = _PADDED_UP_WEIGHTS.get(key)
    if (
        cached is not None
        and cached.source() is up_weight
        and cached.version == version
    ):
        return cached.padded
    with _PADDED_WEIGHT_LOCK:
        cached = _PADDED_UP_WEIGHTS.get(key)
        if cached is not None and cached.source() is up_weight:
            if cached.version != version:
                _copy_padded_up_weight(cached.padded, up_weight, lowrank)
                cached.version = version
            return cached.padded
        padded = torch.zeros(
            (up_weight.shape[0], padded_lowrank),
            dtype=up_weight.dtype,
            device=up_weight.device,
        )
        _copy_padded_up_weight(padded, up_weight, lowrank)
        _PADDED_UP_WEIGHTS[key] = _CachedPaddedWeight(
            source=weakref.ref(up_weight), version=version, padded=padded
        )
        return padded


if _CUTEDSL_AVAILABLE:

    @register_kernel(
        "hyperconnection",
        "mix",
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
            {torch.bfloat16},
        ),
        traits={
            "num_tokens": frozenset(range(1, _MAX_ROWS + 1)),
            "hc_count": frozenset({4}),
            "hidden_size": frozenset({2560}),
            "lowrank": frozenset({320}),
            "has_inject": frozenset({False, True}),
            "contiguous": frozenset({True}),
            "folded_scale": frozenset({False, True}),
            "deterministic": frozenset({False, True}),
            "capturing": frozenset({False, True}),
        },
        # The generic CuTe router building blocks keep this backend available
        # for explicit tuning, but the four-launch composition is slower than
        # the persistent Triton path at T<=16 and cuBLAS/Triton above it on
        # current Blackwell systems. Do not make it the heuristic default.
        priority=Priority.PORTABLE + 1,
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
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """Run two low-latency CuTe GEMMs with fused Triton epilogues.

        The first GEMM retains TokenSpeed's combined down/inject projection.
        Its epilogue applies scale and SiLU while padding the rank dimension.
        The second GEMM uses that padded activation; sigmoid, branch weighting,
        and reduction run without materializing their intermediate products.
        """
        rows = int(normalized.shape[0])
        if (
            not current_platform().is_blackwell
            or not 1 <= rows <= _MAX_ROWS
            or normalized.dtype is not torch.bfloat16
            or hc_count != 4
            or hidden_size != 2560
            or lowrank != 320
            or not normalized.is_contiguous()
            or not projection_weight.is_contiguous()
            or not up_weight.is_contiguous()
        ):
            raise ValueError(
                "CuTeDSL HC mix requires Blackwell and contiguous BF16 "
                "production shape T<=32, hc_count=4, hidden_size=2560, "
                "lowrank=320"
            )
        padded_lowrank = _round_up(lowrank, _LOWRANK_ALIGNMENT)
        padded_up = _padded_up_weight(up_weight, lowrank)
        if not ll_bf16_router.supports(normalized, projection_weight, rows):
            raise ValueError(
                "CuTeDSL down/inject projection does not support these operands"
            )
        projected = ll_bf16_router(
            normalized, projection_weight, out_dtype=normalized.dtype
        )
        activated, inject = _launch_projection_epilogue(
            projected,
            lowrank,
            hc_count,
            projection_scale,
            activation_width=padded_lowrank,
        )
        if not ll_bf16_router.supports(activated, padded_up, rows):
            raise ValueError("CuTeDSL up projection does not support these operands")
        gate = ll_bf16_router(activated, padded_up, out_dtype=normalized.dtype)
        mixed = _launch_mix_epilogue(gate, normalized, hc_count, hidden_size)
        return mixed, inject
