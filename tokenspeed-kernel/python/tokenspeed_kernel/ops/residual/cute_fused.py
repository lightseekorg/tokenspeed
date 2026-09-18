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

"""Register the fused Blackwell gated-residual mix kernel."""

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
    from cuda.bindings import driver as cuda_driver
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
_MAX_TOKENS = 2**31 - 1
_PLANS = {}
_CAPACITIES = {}


@cache
def _resident_clusters(index: int, stream: int) -> int:
    # CuTe queries cuOccupancyMaxActiveClusters with the device's maximum
    # shared-memory allocation. The stream also identifies a green context.
    with torch.cuda.device(index):
        return HardwareInfo(index).get_max_active_clusters(_SPLIT_K, CUstream(stream))


def supports_fused_hc(device: torch.device) -> bool:
    """Whether six 16-CTA clusters fit the current device and stream context.

    The occupancy result is cached permanently by device and stream. Concurrent
    occupancy changes are not re-evaluated; if they invalidate this assumption,
    the cooperative launch fails rather than hanging.
    """
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


def _workspace_for_plan(
    device: torch.device,
    projection_rows: int,
    workers: int,
    clusters: int,
    slot_rows: int,
):
    index = device.index
    if index is None:
        index = torch.cuda.current_device()
    stream = int(torch.cuda.current_stream(device).cuda_stream)
    key = (index, stream, projection_rows, workers, clusters, slot_rows)
    workspace = _WORKSPACES.get(key)
    if workspace is None:
        with _WORKSPACE_LOCK:
            workspace = _WORKSPACES.get(key)
            if workspace is None:
                if torch.cuda.is_current_stream_capturing():
                    raise RuntimeError(
                        "fused CuTe HC workspace is not initialized for CUDA graph "
                        "capture; warm up on the capture stream before capture"
                    )
                # Opaque 16-bit storage allows ordered BF16/FP16 calls to reuse
                # one stream workspace. The invocation supplies the typed view.
                # Reducers overwrite every consumed post-SiLU activation.
                activation_storage = torch.empty(
                    (workers * slot_rows, 320), device=device, dtype=torch.int16
                )
                epochs = torch.zeros(
                    2 * workers * clusters, device=device, dtype=torch.int64
                )
                workspace = (activation_storage, epochs)
                _WORKSPACES[key] = workspace
    return workspace


def _persistent_workspace(device: torch.device, projection_rows: int):
    """Return the original one-worker layout for small-tile diagnostics."""
    return _workspace_for_plan(
        device, projection_rows, 1, (projection_rows + 63) // 64, 16
    )


def _checked(result):
    if result[0] != cuda_driver.CUresult.CUDA_SUCCESS:
        raise RuntimeError(f"CUDA occupancy query failed: {result[0]}")
    return result[1] if len(result) == 2 else result[1:]


def _capacity(compiled, kernel, device: torch.device, stream: CUstream):
    """Query the loaded kernel, not a dummy kernel or SM-count heuristic."""
    compiled.to(device.index)
    libraries = compiled.jit_module.cuda_library
    if len(libraries) != 1:
        raise RuntimeError("Expected one CuTe HC device library")
    library = cuda_driver.CUlibrary(int(libraries[0]))
    handles = _checked(cuda_driver.cuLibraryEnumerateKernels(1, library))
    function = _checked(cuda_driver.cuKernelGetFunction(handles[0]))
    blocks = _checked(
        cuda_driver.cuOccupancyMaxActiveBlocksPerMultiprocessor(
            function, 192, kernel.smem_bytes
        )
    )
    config = cuda_driver.CUlaunchConfig()
    config.blockDimX, config.blockDimY, config.blockDimZ = 192, 1, 1
    config.gridDimX = kernel.clusters
    config.gridDimY = kernel.split_k
    config.gridDimZ = kernel.workers
    config.sharedMemBytes = kernel.smem_bytes
    config.hStream = stream
    attribute = cuda_driver.CUlaunchAttribute()
    attribute.id = cuda_driver.CUlaunchAttributeID.CU_LAUNCH_ATTRIBUTE_CLUSTER_DIMENSION
    attribute.value.clusterDim.x = 1
    attribute.value.clusterDim.y = kernel.split_k
    attribute.value.clusterDim.z = 1
    config.attrs, config.numAttrs = [attribute], 1
    clusters = _checked(cuda_driver.cuOccupancyMaxActiveClusters(function, config))
    registers = _checked(
        cuda_driver.cuFuncGetAttribute(
            cuda_driver.CUfunction_attribute.CU_FUNC_ATTRIBUTE_NUM_REGS, function
        )
    )
    local_bytes = _checked(
        cuda_driver.cuFuncGetAttribute(
            cuda_driver.CUfunction_attribute.CU_FUNC_ATTRIBUTE_LOCAL_SIZE_BYTES,
            function,
        )
    )
    return blocks, clusters, registers, local_bytes


def _tactic(rows: int, projection_rows: int):
    # S, M_proj, B_mma, L_proj, L_batch, stages, F_mma. The single
    # parameterized kernel also supports S=1 and cluster-local projection loops.
    if rows <= 16:
        return 16, 64, 8 if rows <= 8 else 16, 1, 1, 5, 32
    if rows <= 32:
        return 8, 64, 16, 1, 1, 5, 32
    if rows <= 96:
        return 4, 64, 16, 1, 1, 5, 32
    if rows <= 192:
        return 4, 64, 32, 1, 1, 5, 32
    return 4, 128, 32, 1, 1, 4, 32


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
            # Batch size selects a CTA tactic, not a separate implementation.
            "hc_count": frozenset({4}),
            "hidden_size": frozenset({2560}),
            "lowrank": frozenset({320}),
            "contiguous": frozenset({True}),
            "deterministic": frozenset({False, True}),
            "weights_independent": frozenset({True}),
            "fused_tma_aligned": frozenset({True}),
            "fused_grid_supported": frozenset({True}),
        },
        priority=Priority.SPECIALIZED,
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
        """Run down, SiLU, up and gate/output in one tiled CuTe kernel.

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
            or not 1 <= rows <= _MAX_TOKENS
            or normalized.dtype not in (torch.bfloat16, torch.float16)
            or any(not t.is_contiguous() or t.data_ptr() % 16 for t in tensors)
        ):
            raise ValueError(
                "fused CuTe HC requires six resident Blackwell clusters and 16-byte-aligned "
                "contiguous BF16/FP16 tensors with HC4/H2560/R320 and positive T"
            )
        projection_rows = int(projection_weight.shape[0])
        with torch.cuda.device(normalized.device):
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
            stream_id = int(torch.cuda.current_stream(normalized.device).cuda_stream)
            stream = CUstream(stream_id)
            enable_pdl = pdl_enabled()
            key = (
                normalized.device.index,
                stream_id,
                normalized.dtype,
                rows,
                projection_rows,
                enable_pdl,
                projection_scale,
                weights_independent,
            )
            plan = _PLANS.get(key) if key in _COMPILED else None

            def operands_for(kernel):
                activation_storage, epochs = _workspace_for_plan(
                    normalized.device,
                    projection_rows,
                    kernel.workers,
                    kernel.clusters,
                    kernel.slot_rows,
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
                return tuple(
                    from_dlpack(value.detach(), assumed_align=16).mark_layout_dynamic(
                        leading_dim=leading
                    )
                    for value, leading in zip(values, leading_dims)
                )

            if plan is None:
                with _COMPILE_LOCK:
                    plan = _PLANS.get(key) if key in _COMPILED else None
                    if plan is None:
                        if torch.cuda.is_current_stream_capturing():
                            raise RuntimeError(
                                "fused CuTe HC kernel plan is not initialized for CUDA graph "
                                "capture; warm up on the capture stream before capture"
                            )
                        split_k, m, n, lp, lb, stages, f = _tactic(
                            rows, projection_rows
                        )
                        kernel = FusedGatedResidualKernel(
                            rows,
                            projection_rows,
                            split_k,
                            enable_pdl,
                            projection_scale,
                            weights_independent,
                        )
                        kernel.configure(m, n, lp, lb, 1, stages, f)
                        compiled = cute_ext.compile(
                            kernel, *operands_for(kernel), stream
                        )
                        capacity = _capacity(
                            compiled, kernel, normalized.device, stream
                        )
                        if capacity[0] < 1 or capacity[1] < kernel.clusters:
                            raise RuntimeError(
                                "Compiled fused HC group does not fit the device"
                            )
                        jobs = (rows + n * lb - 1) // (n * lb)
                        workers = (
                            jobs
                            if kernel.clusters == 1
                            else min(jobs, capacity[1] // kernel.clusters)
                        )
                        if workers != 1:
                            kernel = FusedGatedResidualKernel(
                                rows,
                                projection_rows,
                                split_k,
                                enable_pdl,
                                projection_scale,
                                weights_independent,
                            )
                            kernel.configure(m, n, lp, lb, workers, stages, f)
                            compiled = cute_ext.compile(
                                kernel, *operands_for(kernel), stream
                            )
                            capacity = _capacity(
                                compiled, kernel, normalized.device, stream
                            )
                            if (
                                kernel.clusters > 1
                                and workers * kernel.clusters > capacity[1]
                            ):
                                raise RuntimeError(
                                    "Compiled fused HC cooperative grid exceeds capacity"
                                )
                        plan = (kernel, compiled)
                        _PLANS[key] = plan
                        _COMPILED[key] = compiled
                        _CAPACITIES[key] = capacity
            kernel, compiled = plan
            compiled(*operands_for(kernel), stream)
        return out, inject
