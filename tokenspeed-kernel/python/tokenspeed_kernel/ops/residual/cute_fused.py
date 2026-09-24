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

_CACHE_LOCK = threading.Lock()
_WORKSPACES = {}
_WORKSPACE_LOCK = threading.Lock()
_PROBE_SPLIT_K = 16
_MAX_OVERRIDE_TOKENS = 1024
_PLANS = {}
_CAPACITIES = {}


@cache
def _resident_clusters(index: int, stream: int) -> int:
    # CuTe queries cuOccupancyMaxActiveClusters with the device's maximum
    # shared-memory allocation. The stream also identifies a green context.
    with torch.cuda.device(index):
        return HardwareInfo(index).get_max_active_clusters(
            _PROBE_SPLIT_K, CUstream(stream)
        )


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
    # The runtime orders model launches on one execution lane. Persistent HC
    # state is therefore owned by its layout, not by the CUDA stream handle.
    key = (index, projection_rows, workers, clusters, slot_rows)
    workspace = _WORKSPACES.get(key)
    if workspace is None:
        with _WORKSPACE_LOCK:
            workspace = _WORKSPACES.get(key)
            if workspace is None:
                if torch.cuda.is_current_stream_capturing():
                    raise RuntimeError(
                        "fused CuTe HC workspace is not initialized for CUDA graph "
                        "capture; warm up this workspace layout before capture"
                    )
                # Ordered BF16/FP16 calls reuse opaque 16-bit storage through
                # an invocation-typed view. Reducers overwrite every consumed
                # post-SiLU activation.
                activation_storage = torch.empty(
                    (workers * slot_rows, 320), device=device, dtype=torch.int16
                )
                epochs = torch.zeros(
                    2 * workers * clusters, device=device, dtype=torch.int64
                )
                workspace = (activation_storage, epochs)
                _WORKSPACES[key] = workspace
    return workspace


def _checked(result):
    if result[0] != cuda_driver.CUresult.CUDA_SUCCESS:
        raise RuntimeError(f"CUDA occupancy query failed: {result[0]}")
    return result[1] if len(result) == 2 else result[1:]


def _capacity(compiled, kernel, device: torch.device, stream: CUstream):
    """Query a conservative cluster capacity from the loaded kernel."""
    compiled.to(device.index)
    libraries = compiled.jit_module.cuda_library
    if len(libraries) != 1:
        raise RuntimeError("Expected one CuTe HC device library")
    library = cuda_driver.CUlibrary(int(libraries[0]))
    kernel_count = _checked(cuda_driver.cuLibraryGetKernelCount(library))
    if kernel_count != 1:
        raise RuntimeError("Expected one CuTe HC kernel in the device library")
    handles = _checked(cuda_driver.cuLibraryEnumerateKernels(kernel_count, library))
    function = _checked(cuda_driver.cuKernelGetFunction(handles[0]))
    smem_limit = _checked(
        cuda_driver.cuFuncGetAttribute(
            cuda_driver.CUfunction_attribute.CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES,
            function,
        )
    )
    config = cuda_driver.CUlaunchConfig()
    config.blockDimX, config.blockDimY, config.blockDimZ = 192, 1, 1
    config.gridDimX = kernel.clusters
    config.gridDimY = kernel.split_k
    config.gridDimZ = kernel.workers
    # CuTe infers exact launch storage from its allocations. Supported tactics
    # consume over half an SM, so the opt-in limit preserves one-CTA-per-SM
    # residency and remains conservative if allocations change.
    config.sharedMemBytes = smem_limit
    config.hStream = stream
    attribute = cuda_driver.CUlaunchAttribute()
    attribute.id = cuda_driver.CUlaunchAttributeID.CU_LAUNCH_ATTRIBUTE_CLUSTER_DIMENSION
    attribute.value.clusterDim.x = 1
    attribute.value.clusterDim.y = kernel.split_k
    attribute.value.clusterDim.z = 1
    config.attrs, config.numAttrs = [attribute], 1
    return _checked(cuda_driver.cuOccupancyMaxActiveClusters(function, config))


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
            # Larger batches use the GEMM/Triton path to avoid fused regressions.
            "num_tokens": frozenset(range(1, 257)),
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

        Automatic selection is limited to 1..256 rows; larger batches use the
        GEMM/Triton implementation. Exact overrides support up to 1024 rows for
        kernel diagnostics.

        Different row counts and streams share a compiled plan within the same
        CTA tactic, resident-worker count and scheduling-round bucket, with
        separate full/tail tile variants. The input layout supplies rows at
        launch; grid sizing and tail masks remain dynamic. Cooperative
        workspaces reserve the tactic's resident worker capacity, with a fixed
        READY/CONSUMED split across row changes. Calls which share a workspace
        must be ordered on the model execution lane. Warm each tactic/round
        bucket and each capture stream's occupancy result before graph capture;
        capture may then use a previously unseen row count in that bucket. Plan
        and occupancy caches live for the process lifetime; diagnostic callers
        must not scan unbounded scale or stream variants.
        """
        rows = int(normalized.shape[0])
        tensors = (normalized, projection_weight, up_weight)
        if (
            not supports_fused_hc(normalized.device)
            or (hc_count, hidden_size, lowrank) != (4, 2560, 320)
            or not 1 <= rows <= _MAX_OVERRIDE_TOKENS
            or normalized.dtype not in (torch.bfloat16, torch.float16)
            or any(not t.is_contiguous() or t.data_ptr() % 16 for t in tensors)
        ):
            raise ValueError(
                "fused CuTe HC requires six resident Blackwell clusters and 16-byte-aligned "
                "contiguous BF16/FP16 tensors with HC4/H2560/R320 and at most 1024 rows"
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
            tactic = _tactic(rows, projection_rows)
            split_k, m, n, lp, lb, stages, f = tactic
            single_tile = rows <= n and lb == 1
            full_tiles = rows % n == 0
            # A stream changes available occupancy (for example in a green
            # context), but it does not change generated code. The final plan
            # key adds the worker count derived from that occupancy below.
            static_key = (
                normalized.device.index,
                normalized.dtype,
                tactic,
                single_tile,
                full_tiles,
                projection_rows,
                enable_pdl,
                projection_scale,
                weights_independent,
            )

            def operands_for(kernel):
                workers = (
                    (rows + n * lb - 1) // (n * lb)
                    if kernel.clusters == 1
                    else kernel.workers
                )
                activation_storage, epochs = _workspace_for_plan(
                    normalized.device,
                    projection_rows,
                    workers,
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

            def compile_plan(workers, rounds):
                if torch.cuda.is_current_stream_capturing():
                    raise RuntimeError(
                        "fused CuTe HC kernel plan is not initialized for CUDA graph "
                        "capture; warm up this variant before capture"
                    )
                kernel = FusedGatedResidualKernel(
                    projection_tile=m,
                    token_tile=n,
                    projection_rows=projection_rows,
                    split_k=split_k,
                    projection_tiles=lp,
                    batch_tiles=lb,
                    workers=workers,
                    stages=stages,
                    final_tile=f,
                    rounds=rounds,
                    use_pdl=enable_pdl,
                    scale=projection_scale,
                    weights_independent=weights_independent,
                    single_tile=single_tile,
                    full_tiles=full_tiles,
                )
                compiled = cute_ext.compile(kernel, *operands_for(kernel), stream)
                return kernel, compiled

            def capacity_for(plan_key, plan):
                kernel, compiled = plan
                capacity_key = (stream_id, plan_key)
                capacity = _CAPACITIES.get(capacity_key)
                if capacity is None:
                    if torch.cuda.is_current_stream_capturing():
                        raise RuntimeError(
                            "fused CuTe HC occupancy is not initialized for CUDA "
                            "graph capture; warm up on the capture stream before "
                            "capture"
                        )
                    with _CACHE_LOCK:
                        capacity = _CAPACITIES.get(capacity_key)
                        if capacity is None:
                            capacity = _capacity(
                                compiled, kernel, normalized.device, stream
                            )
                            _CAPACITIES[capacity_key] = capacity
                required = kernel.clusters * (
                    kernel.workers if kernel.clusters > 1 else 1
                )
                if required > capacity:
                    raise RuntimeError(
                        "Compiled fused HC launch exceeds cluster capacity"
                    )
                return capacity

            def plan_for(workers, rounds):
                plan_key = (*static_key, workers, rounds)
                plan = _PLANS.get(plan_key)
                if plan is None:
                    with _CACHE_LOCK:
                        plan = _PLANS.get(plan_key)
                        if plan is None:
                            plan = compile_plan(workers, rounds)
                            _PLANS[plan_key] = plan
                return plan_key, plan

            probe_key, probe = plan_for(1, 1)
            probe_capacity = capacity_for(probe_key, probe)
            workers = 1 if single_tile else probe_capacity // probe[0].clusters
            plan = probe
            if workers > 1:
                worker_key, plan = plan_for(workers, 1)
                capacity_for(worker_key, plan)
            kernel, compiled = plan
            rounds = (
                (rows + n * lb * kernel.workers - 1) // (n * lb * kernel.workers)
                if kernel.clusters > 1
                else 1
            )
            if rounds > 1:
                # Bucket by scheduling rounds, not exact rows. Static loop bounds
                # preserve register reuse across the pipeline's warp roles.
                round_key, plan = plan_for(kernel.workers, rounds)
                capacity_for(round_key, plan)
                kernel, compiled = plan
            compiled(*operands_for(kernel), stream)
        return out, inject
