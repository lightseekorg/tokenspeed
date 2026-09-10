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

"""Register the fused Blackwell gated-residual decode kernel."""

from __future__ import annotations

import threading
from functools import cache

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
    from cutlass.utils.hardware_info import HardwareInfo
    from tokenspeed_kernel.thirdparty.cute_dsl.hc_fused import FusedGatedResidualKernel

    _AVAILABLE = True
except ImportError:
    _AVAILABLE = False

_COMPILED = {}
_COMPILE_LOCK = threading.Lock()
_WORKSPACES = {}
_WORKSPACE_LOCK = threading.Lock()
_SPLIT_K = 16


@cache
def _resident_clusters(index: int, stream: int) -> int:
    # CuTe queries cuOccupancyMaxActiveClusters with the device's maximum
    # shared-memory allocation. The stream also identifies a green context.
    with torch.cuda.device(index):
        return HardwareInfo(index).get_max_active_clusters(_SPLIT_K, CUstream(stream))


def supports_fused_hc(device: torch.device) -> bool:
    """Whether six 16-CTA clusters fit the current device and stream context."""
    if not _AVAILABLE or not current_platform().is_blackwell:
        return False
    props = torch.cuda.get_device_properties(device)
    if (
        props.multi_processor_count < 96
        or props.shared_memory_per_block_optin < 227 * 1024
    ):
        return False
    index = torch.cuda.current_device() if device.index is None else device.index
    stream = int(torch.cuda.current_stream(device).cuda_stream)
    return _resident_clusters(index, stream) >= 6


def _persistent_workspace(device: torch.device, projection_rows: int):
    index = device.index
    if index is None:
        index = torch.cuda.current_device()
    stream = int(torch.cuda.current_stream(device).cuda_stream)
    key = (index, stream, projection_rows)
    workspace = _WORKSPACES.get(key)
    if workspace is None:
        with _WORKSPACE_LOCK:
            workspace = _WORKSPACES.get(key)
            if workspace is None:
                # Opaque 16-bit storage allows ordered BF16/FP16 calls to reuse
                # one stream workspace. The invocation supplies the typed view.
                # Reducers overwrite every consumed post-SiLU activation.
                activation_storage = torch.empty(
                    (16, 320), device=device, dtype=torch.int16
                )
                epochs = torch.zeros(
                    (projection_rows + 63) // 64, device=device, dtype=torch.int64
                )
                workspace = (activation_storage, epochs)
                _WORKSPACES[key] = workspace
    return workspace


if _AVAILABLE:

    @register_kernel(
        "residual",
        "hyperconnection_mix",
        name="cute_fused_hyperconnection_mix",
        solution="cute_fused",
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
            "num_tokens": frozenset(range(1, 17)),
            "hc_count": frozenset({4}),
            "hidden_size": frozenset({2560}),
            "lowrank": frozenset({320}),
            "contiguous": frozenset({True}),
            "deterministic": frozenset({False, True}),
            "weights_independent": frozenset({True}),
            "fused_tma_aligned": frozenset({True}),
            "fused_grid_supported": frozenset({True}),
        },
        priority=Priority.SPECIALIZED + 3,
        tags={"cute_dsl", "decode", "latency"},
    )
    def cute_fused_hyperconnection_mix(
        normalized: torch.Tensor,
        projection_weight: torch.Tensor,
        up_weight: torch.Tensor,
        hc_count: int,
        hidden_size: int,
        lowrank: int,
        projection_scale: float,
        weights_independent: bool,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """Run down, SiLU, up and gate/output in one cooperative CuTe kernel.

        Args follow the public ``gated_residual_mix`` contract. Independent
        weights must already be ready and remain immutable within forward;
        the conservative value also waits for a PDL weight producer. Returns
        the mixed tensor and optional inject logits, owned by this invocation.
        """
        rows = int(normalized.shape[0])
        tensors = (normalized, projection_weight, up_weight)
        if (
            not supports_fused_hc(normalized.device)
            or (hc_count, hidden_size, lowrank) != (4, 2560, 320)
            or not 1 <= rows <= 16
            or normalized.dtype not in (torch.bfloat16, torch.float16)
            or any(not t.is_contiguous() or t.data_ptr() % 16 for t in tensors)
        ):
            raise ValueError(
                "fused CuTe HC requires six resident Blackwell clusters and 16-byte-aligned "
                "contiguous BF16/FP16 tensors with HC4/H2560/R320/T1..16"
            )
        projection_rows = int(projection_weight.shape[0])
        with torch.cuda.device(normalized.device):
            activation_storage, epochs = _persistent_workspace(
                normalized.device, projection_rows
            )
            out = torch.empty(
                (rows, hidden_size), dtype=normalized.dtype, device=normalized.device
            )
            inject = (
                torch.empty(
                    (rows, hc_count), dtype=normalized.dtype, device=normalized.device
                )
                if projection_rows != lowrank
                else None
            )
            values = (
                normalized.unsqueeze(-1),
                projection_weight.unsqueeze(-1),
                up_weight.unsqueeze(-1),
                activation_storage.view(normalized.dtype),
                epochs,
                out,
                out if inject is None else inject,
            )
            leading_dims = (1, 1, 1, 1, 0, 1, 1)
            operands = tuple(
                from_dlpack(value.detach(), assumed_align=16).mark_layout_dynamic(
                    leading_dim=leading
                )
                for value, leading in zip(values, leading_dims)
            )
            stream = CUstream(torch.cuda.current_stream(normalized.device).cuda_stream)
            enable_pdl = pdl_enabled(None)
            key = (
                normalized.device.index,
                normalized.dtype,
                rows,
                projection_rows,
                _SPLIT_K,
                enable_pdl,
                projection_scale,
                weights_independent,
            )
            compiled = _COMPILED.get(key)
            if compiled is None:
                with _COMPILE_LOCK:
                    compiled = _COMPILED.get(key)
                    if compiled is None:
                        kernel = FusedGatedResidualKernel(
                            rows,
                            projection_rows,
                            _SPLIT_K,
                            enable_pdl,
                            projection_scale,
                            weights_independent,
                        )
                        compiled = cute_ext.compile(kernel, *operands, stream)
                        _COMPILED[key] = compiled
            compiled(*operands, stream)
        return out, inject
