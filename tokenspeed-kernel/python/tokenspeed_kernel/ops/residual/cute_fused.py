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
from tokenspeed_kernel.selection import resolve_kernel_override
from tokenspeed_kernel.signature import format_signatures

try:
    from cuda.bindings import driver as cuda_driver
    from cuda.bindings.driver import CUstream
    from cutlass.cute import experimental as cute_ext
    from cutlass.cute.runtime import from_dlpack
    from cutlass.utils.hardware_info import HardwareInfo
    from tokenspeed_kernel.thirdparty.cute_dsl.hc_fused import FusedGatedResidualKernel

    _AVAILABLE = True
except (ImportError, NotImplementedError):
    _AVAILABLE = False

_CACHE_LOCK = threading.Lock()
_WORKSPACES = {}
_WORKSPACE_LOCK = threading.Lock()
_PROBE_SPLIT_K = 16
_MAX_OVERRIDE_TOKENS = 1024
_PLANS = {}
_CAPACITIES = {}
_WARMED = set()
_RESIDENT_CTA_BUDGET = 96
_MAX_AUTO_MMA_STEPS = 640
_EVICT_UP_WEIGHTS_FIRST = True
_EVICT_DOWN_WEIGHTS_FIRST = True
_UP_L2_PREFETCH_K = 256
_USE_UP_TANH = True
_USE_UP_TOKEN_TAIL = True
_USE_NATIVE64_DOWN_X_RETENTION = True


def _up_l2_prefetch_k(split_k: int, reuse_down_storage: bool) -> int:
    # Other tactics lack the DownDone-to-publication overlap used by this hint.
    return _UP_L2_PREFETCH_K if split_k == 1 and reuse_down_storage else 0


def supports_fused_hc_shape(hc_count: int, hidden_size: int, lowrank: int) -> bool:
    """Whether the model has a validated fused HC tactic family.

    Args:
        hc_count: Number of residual branches.
        hidden_size: Width of one branch.
        lowrank: Rank of the mix projection.

    Returns:
        Whether the model dimensions select a validated fused implementation.
        Other model shapes retain the ordinary GEMM/Triton selection.
    """
    return (hc_count, hidden_size, lowrank) in ((4, 2560, 320), (4, 8192, 1024))


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
    partial_stride: int,
    workers: int,
    clusters: int,
    grid_clusters: int,
    slot_rows: int,
    lowrank: int,
    reduction_groups: int,
    split_k: int,
    projection_tiles: int,
):
    index = device.index
    if index is None:
        index = torch.cuda.current_device()
    # The runtime orders model launches on one execution lane. Persistent HC
    # state is therefore owned by its layout, not by the CUDA stream handle.
    key = (
        index,
        projection_rows,
        partial_stride,
        workers,
        clusters,
        grid_clusters,
        slot_rows,
        lowrank,
        reduction_groups,
        split_k,
        projection_tiles,
    )
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
                # Persistent scratch can be reused by calls made outside the
                # inference-mode context that first warmed this layout.
                with torch.inference_mode(False):
                    activation_storage = torch.empty(
                        (workers * slot_rows, lowrank), device=device, dtype=torch.int16
                    )
                    partials = (
                        torch.empty(
                            (workers * slot_rows, partial_stride * reduction_groups),
                            device=device,
                            dtype=torch.float32,
                        )
                        if reduction_groups > 1
                        else activation_storage
                    )
                    epochs = torch.zeros(
                        2 * workers * grid_clusters
                        + workers
                        * (clusters // reduction_groups)
                        * int(reduction_groups > 1),
                        device=device,
                        dtype=torch.int64,
                    )
                workspace = (activation_storage, partials, epochs)
                _WORKSPACES[key] = workspace
    return workspace


def _checked(result):
    if result[0] != cuda_driver.CUresult.CUDA_SUCCESS:
        raise RuntimeError(f"CUDA occupancy query failed: {result[0]}")
    return result[1] if len(result) == 2 else result[1:]


def _capacity(compiled, kernel, device: torch.device, stream: CUstream):
    """Query cluster capacity for the compiled function's exact launch storage."""
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
    config.blockDimX, config.blockDimY, config.blockDimZ = kernel.threads, 1, 1
    config.gridDimX = kernel.physical_clusters
    config.gridDimY = kernel.split_k
    config.gridDimZ = kernel.workers
    if kernel.cta_shared_bytes > smem_limit:
        raise ValueError("HC shared storage exceeds the compiled opt-in limit")
    config.sharedMemBytes = kernel.cta_shared_bytes
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
        return 8, 64, 32, 1, 1, 5, 32
    if rows <= 96:
        return 4, 64, 16, 1, 1, 5, 32
    if rows <= 192:
        return 4, 64, 32, 1, 1, 5, 32
    return 4, 128, 32, 1, 1, 4, 32


def _epilogue_threads(token_tile: int, down_token_parts: int) -> int:
    return (
        min(512, 128 * (token_tile // down_token_parts // 8))
        if token_tile >= 32
        else 128
    )


_RG_OVERRIDE = {}


def _reduction_groups(hc_count: int, hidden_size: int, lowrank: int) -> int:
    return 16 if lowrank > 6 * 64 and hc_count * hidden_size >= 4 * lowrank else 1


@cache
def _cluster_capacity_hint(index: int, stream: int, split_k: int) -> int:
    with torch.cuda.device(index):
        return HardwareInfo(index).get_max_active_clusters(split_k, CUstream(stream))


def _evict_down_weights_first_tactic(
    hc_count: int,
    hidden_size: int,
    lowrank: int,
    token_tile: int,
    split_k: int,
    projection_tile: int,
    pipelines_per_cta: int,
    reduction_groups: int,
    workers: int,
    rounds: int,
    projection_tiles: int,
    batch_tiles: int,
    token_tile_batches: int,
    up_clusters: int,
) -> bool:
    if not _EVICT_DOWN_WEIGHTS_FIRST:
        return False
    if not (
        hc_count == 4
        and hidden_size == 8192
        and lowrank == 1024
        and split_k == 1
        and reduction_groups == 16
        and workers
        == rounds
        == projection_tiles
        == batch_tiles
        == token_tile_batches
        == 1
        and up_clusters == 0
    ):
        return False
    # Cold measurements support these finite buckets; other native plans retain
    # their existing policy, including the configured rows96 warm case.
    return (
        pipelines_per_cta == 2
        and projection_tile == 64
        and token_tile in (8, 16, 24, 32, 128)
    )


def _up_sigmoid_mode_tactic(
    hc_count: int,
    hidden_size: int,
    lowrank: int,
    token_tile: int,
    split_k: int,
    projection_tile: int,
    pipelines_per_cta: int,
    reduction_groups: int,
    workers: int,
    rounds: int,
    projection_tiles: int,
    batch_tiles: int,
    token_tile_batches: int,
    up_clusters: int,
    full_tiles: bool,
) -> str:
    if (
        _USE_UP_TANH
        and hc_count == 4
        and hidden_size == 8192
        and lowrank == 1024
        and split_k == 1
        and reduction_groups == 16
        and (
            (
                token_tile in (64, 192)
                and projection_tile == 64
                and pipelines_per_cta == 2
            )
            or (token_tile == 96 and projection_tile == 128 and pipelines_per_cta == 1)
        )
        and workers
        == rounds
        == projection_tiles
        == batch_tiles
        == token_tile_batches
        == 1
        and up_clusters == 0
        and full_tiles
    ):
        return "tanh"
    return "exp_rcp"


def _up_token_tail_overlap_tactic(
    hc_count: int,
    hidden_size: int,
    lowrank: int,
    token_tile: int,
    split_k: int,
    projection_tile: int,
    pipelines_per_cta: int,
    reduction_groups: int,
    workers: int,
    rounds: int,
    projection_tiles: int,
    batch_tiles: int,
    token_tile_batches: int,
    up_clusters: int,
    full_tiles: bool,
) -> bool:
    return (
        _USE_UP_TOKEN_TAIL
        and hc_count == 4
        and hidden_size == 8192
        and lowrank == 1024
        and token_tile == 256
        and split_k == 1
        and reduction_groups == 16
        and projection_tile == 128
        and pipelines_per_cta == 1
        and workers
        == rounds
        == projection_tiles
        == batch_tiles
        == token_tile_batches
        == 1
        and up_clusters == 0
        and full_tiles
    )


def _legacy_up_tactic(
    hc_count: int, hidden_size: int, lowrank: int, token_tile: int, split_k: int
) -> bool:
    # K64 covers rank 320 without padding. Keep its weights cached across
    # token workers while reusing Down's operand arena.
    return (
        (hc_count, hidden_size, lowrank) == (4, 2560, 320)
        and token_tile == 16
        and split_k == 4
    )


def _evict_up_weights_first_tactic(
    hc_count: int, hidden_size: int, lowrank: int, token_tile: int, split_k: int
) -> bool:
    return _EVICT_UP_WEIGHTS_FIRST and not _legacy_up_tactic(
        hc_count, hidden_size, lowrank, token_tile, split_k
    )


def _up_k_tactic(
    hc_count: int, hidden_size: int, lowrank: int, token_tile: int, split_k: int
) -> int:
    if _legacy_up_tactic(hc_count, hidden_size, lowrank, token_tile, split_k):
        return 64
    if (
        hc_count == 4
        and hidden_size == 8192
        and lowrank == 1024
        and token_tile in (8, 16, 64)
    ):
        return 64
    return 64 if token_tile > 64 else 128


def _down_x_cache_mode_tactic(
    hc_count: int,
    hidden_size: int,
    lowrank: int,
    token_tile: int,
    split_k: int,
    workers: int,
    rounds: int,
    projection_tiles: int,
    batch_tiles: int,
    token_tile_batches: int,
    up_clusters: int,
    full_tiles: bool,
) -> str:
    if hc_count == 4 and hidden_size == 8192 and lowrank == 1024:
        if token_tile in (96, 128, 192, 256):
            return "evict_last"
        # A large generic batch can also use N64, with several workers.
        if (
            _USE_NATIVE64_DOWN_X_RETENTION
            and token_tile == 64
            and full_tiles
            and split_k == 1
            and workers
            == rounds
            == projection_tiles
            == batch_tiles
            == token_tile_batches
            == 1
            and up_clusters == 0
        ):
            return "evict_last"
    return "native"


def _up_output_parts_tactic(
    hc_count: int, hidden_size: int, lowrank: int, token_tile: int
) -> int:
    return 1


def _up_stage_counts_tactic(
    hc_count: int, hidden_size: int, lowrank: int, token_tile: int
) -> tuple[int, int] | None:
    if hc_count == 4 and hidden_size == 8192 and lowrank == 1024 and token_tile == 192:
        return (3, 2)
    return None


def _down_k_order_tactic(
    hc_count: int, hidden_size: int, lowrank: int, token_tile: int, split_k: int
) -> str:
    if _legacy_up_tactic(hc_count, hidden_size, lowrank, token_tile, split_k):
        return "paired"
    if hc_count == 4 and hidden_size == 8192 and lowrank == 1024:
        if token_tile in (96, 128, 192):
            return "paired"
        if token_tile == 256:
            return "contiguous"
    return "interleaved"


def _rotate_up_k_tactic(
    hc_count: int, hidden_size: int, lowrank: int, token_tile: int
) -> bool:
    return (
        hc_count == 4
        and hidden_size == 8192
        and lowrank == 1024
        and token_tile in (8, 16, 24, 32, 64, 96, 128)
    )


def _token_major_up_tactic(
    hc_count: int, hidden_size: int, lowrank: int, token_tile: int
) -> bool:
    return (
        hc_count == 4
        and hidden_size == 8192
        and lowrank == 1024
        and token_tile in (96, 128, 256)
    )


def _up_token_parts_tactic(
    hc_count: int, hidden_size: int, lowrank: int, token_tile: int
) -> int:
    return (
        2
        if hc_count == 4
        and hidden_size == 8192
        and lowrank == 1024
        and token_tile == 256
        else 1
    )


@cache
def _local_pipeline_parameters(
    token_tile: int, hc_count: int, hidden_size: int, lowrank: int
):
    """Choose shared and TMEM capacity for finite native tile buckets."""
    if (
        token_tile in (96, 256)
        and hc_count == 4
        and hidden_size == 8192
        and lowrank == 1024
    ):
        return 1, 256, 227 * 1024
    if token_tile == 8 and hc_count == 4 and hidden_size == 8192 and lowrank == 1024:
        return 2, 128, 111 * 1024
    groups = _reduction_groups(hc_count, hidden_size, lowrank)
    branch_tile = min(4, hc_count & -hc_count)
    if token_tile >= 16 and groups > 1 and branch_tile == hc_count:
        return 2, 128, 111 * 1024
    return 1, 256, 227 * 1024


@cache
def _activation_tactic(
    hc_count, hidden_size, lowrank, projection_rows, tactic, tile_batches, rows
):
    """Choose fixed operand lifetimes within the native tactic's shared budget."""
    split_k, m, n, lp, lb, stages, f = tactic
    pipes, up_columns, shared_budget = _local_pipeline_parameters(
        n, hc_count, hidden_size, lowrank
    )
    reuse = (
        n > 16
        or _reduction_groups(hc_count, hidden_size, lowrank) > 1
        or _legacy_up_tactic(hc_count, hidden_size, lowrank, n, split_k)
    )
    if not _AVAILABLE:
        return (*tactic, False, reuse)

    def geometry(down_stages, pin, reuse_storage):
        return FusedGatedResidualKernel(
            hc_count=hc_count,
            hidden_size=hidden_size,
            lowrank=lowrank,
            up_k=_up_k_tactic(hc_count, hidden_size, lowrank, n, split_k),
            down_x_cache_mode=_down_x_cache_mode_tactic(
                hc_count,
                hidden_size,
                lowrank,
                n,
                split_k,
                1,
                1,
                lp,
                lb,
                tile_batches,
                0,
                False,
            ),
            down_k_order=_down_k_order_tactic(
                hc_count, hidden_size, lowrank, n, split_k
            ),
            projection_tile=m,
            token_tile=n,
            token_major_up=_token_major_up_tactic(hc_count, hidden_size, lowrank, n),
            up_output_parts=_up_output_parts_tactic(hc_count, hidden_size, lowrank, n),
            up_token_parts=_up_token_parts_tactic(hc_count, hidden_size, lowrank, n),
            up_stage_counts=_up_stage_counts_tactic(hc_count, hidden_size, lowrank, n),
            rotate_up_k=_rotate_up_k_tactic(hc_count, hidden_size, lowrank, n),
            epilogue_threads=min(_epilogue_threads(n, 1), 256 if pipes > 1 else 512),
            down_token_parts=1,
            projection_rows=projection_rows,
            partial_stride=((projection_rows + 31) // 32) * 32,
            split_k=split_k,
            reduction_groups=(
                4
                if (hc_count, hidden_size, lowrank) == (4, 8192, 1024)
                and rows > 128
                and n <= 64
                else _reduction_groups(hc_count, hidden_size, lowrank)
            ),
            projection_tiles=lp,
            batch_tiles=lb,
            token_tile_batches=tile_batches,
            workers=1,
            up_clusters=0,
            stages=down_stages,
            pin_up_activation=pin,
            reuse_down_storage=reuse_storage,
            final_tile=f,
            rounds=1,
            use_pdl=False,
            scale=1.0,
            weights_independent=True,
            evict_up_weights_first=_evict_up_weights_first_tactic(
                hc_count, hidden_size, lowrank, n, split_k
            ),
            evict_down_weights_first=_evict_down_weights_first_tactic(
                hc_count,
                hidden_size,
                lowrank,
                n,
                split_k,
                m,
                pipes,
                (
                    4
                    if (hc_count, hidden_size, lowrank) == (4, 8192, 1024)
                    and rows > 128
                    and n <= 64
                    else _reduction_groups(hc_count, hidden_size, lowrank)
                ),
                1,
                1,
                lp,
                lb,
                tile_batches,
                0,
            ),
            single_tile=False,
            full_tiles=False,
            pipelines_per_cta=pipes,
            up_columns=up_columns,
            pipeline_shared_budget=shared_budget,
            down_pair_reduction="auto",
            down_producer_mode="auto",
            up_activation_ready_mode="auto",
            up_l2_prefetch_k=_up_l2_prefetch_k(split_k, reuse_storage),
            up_sigmoid_mode="exp_rcp",
            up_token_tail_overlap=False,
        )

    # The preferred Down depth must leave room for at least one Up tile.
    # Model geometry can make the initial depth too large, especially when
    # several branch groups require an extra mix buffer.
    for down_stages in range(stages, 0, -1):
        try:
            baseline = geometry(down_stages, False, reuse)
        except ValueError as error:
            if str(error) not in (
                "HC tactic leaves no shared memory for an Up K tile",
                "Actual pipeline layout exceeds its explicit shared budget",
            ):
                raise
            if down_stages == 1:
                raise
            continue
        stages = down_stages
        tactic = (split_k, m, n, lp, lb, stages, f)
        break
    if baseline.up_steps <= 1 or baseline.up_k_tiles <= 1:
        return (*tactic, False, reuse)
    required_weights = (
        min(stages, baseline.up_stages)
        if n <= 16 and _reduction_groups(hc_count, hidden_size, lowrank) > 1
        else baseline.up_stages if n <= 16 else min(2, baseline.up_stages)
    )
    # Streaming ranks benefit from a complete retained activation and a deeper
    # Down pipe. Reuse only after the final Down epilogue acknowledges its arena.
    retained_reuse = reuse or baseline.up_k_tiles > baseline.up_stages
    choices = (stages, max(1, stages - 1)) if n <= 16 else (stages,)
    for down_stages in choices:
        try:
            retained = geometry(down_stages, True, retained_reuse)
        except ValueError as error:
            if str(error) not in (
                "HC tactic leaves no shared memory for an Up K tile",
                "Actual pipeline layout exceeds its explicit shared budget",
            ):
                raise
            continue
        if retained.up_stages >= required_weights:
            return split_k, m, n, lp, lb, down_stages, f, True, retained_reuse
    return (*tactic, False, reuse)


def _model_tactic(
    rows: int, hc_count: int, hidden_size: int, lowrank: int, projection_rows: int
):
    """Apply model geometry to the finite native CTA tactics."""
    wide_native = (
        hc_count == 4 and hidden_size == 8192 and lowrank == 1024 and 65 <= rows <= 256
    )
    split_k, m, n, lp, lb, stages, f = _tactic(rows, projection_rows)
    if (
        (hc_count, hidden_size, lowrank) == (4, 2560, 320)
        and 32 < rows <= 64
        and (split_k, m, n, lp, lb, stages, f) == (4, 64, 16, 1, 1, 5, 32)
    ):
        # S8/N32 keeps this model's token groups in one resident round.
        split_k, n = 8, 32
    if projection_rows > 6 * m:
        # Four-CTA clusters expose more Up output work at the resident limit.
        # Keep enough independent projection tiles while buffering their K loop.
        split_k, m = 4, 64
        if rows >= 64:
            n, stages = 64, 3
    reduction_groups = (
        4
        if (hc_count, hidden_size, lowrank) == (4, 8192, 1024)
        and rows > 128
        and not wide_native
        else _reduction_groups(hc_count, hidden_size, lowrank)
    )
    if wide_native:
        n = 96 if rows <= 96 else 128 if rows <= 128 else 192 if rows <= 192 else 256
        stages = 8 if n == 96 else 6 if n == 128 else 4
    if hc_count == 4 and hidden_size == 8192 and lowrank == 1024 and 16 < rows <= 24:
        # A finite24-token bucket avoids the eight padded rows ofN32.
        n = 24
    pipes, _, _ = _local_pipeline_parameters(n, hc_count, hidden_size, lowrank)
    if reduction_groups > 1:
        split_k, m = 1, 64 if pipes > 1 else 128
    k_blocks = (hc_count * hidden_size + 127) // 128
    split_k = min(
        split_k, 1 << ((max(1, k_blocks // reduction_groups)).bit_length() - 1)
    )
    stages = min(
        stages,
        (k_blocks + split_k * reduction_groups - 1) // (split_k * reduction_groups),
    )
    cluster_budget = _RESIDENT_CTA_BUDGET // split_k
    if reduction_groups > 1:
        index = torch.cuda.current_device()
        stream = int(torch.cuda.current_stream(index).cuda_stream)
        cluster_budget = (
            _cluster_capacity_hint(index, stream, split_k) * pipes // reduction_groups
        )
    lp = max(lp, (projection_rows + cluster_budget * m - 1) // (cluster_budget * m))
    tactic = _activation_tactic(
        hc_count,
        hidden_size,
        lowrank,
        projection_rows,
        (split_k, m, n, lp, lb, stages, f),
        (
            2
            if (hc_count, hidden_size, lowrank) == (4, 8192, 1024)
            and 96 < rows <= 128
            and n <= 64
            else 1
        ),
        rows,
    )
    return _down_tactic(hc_count, hidden_size, lowrank, projection_rows, tactic)


@cache
def _down_tactic(hc_count, hidden_size, lowrank, projection_rows, tactic):
    """Pair a larger Down K tile with stages that fit the existing arena."""
    split_k, m, n, lp, lb, stages, f, pin, reuse = tactic
    groups = _reduction_groups(hc_count, hidden_size, lowrank)
    pipes, up_columns, shared_budget = _local_pipeline_parameters(
        n, hc_count, hidden_size, lowrank
    )
    if (
        (hc_count == 4 and hidden_size == 8192 and lowrank == 1024 and n > 64)
        or pipes > 1
        or not _AVAILABLE
        or groups == 1
        or not reuse
        or hc_count * hidden_size % (split_k * groups * 256)
    ):
        return tactic
    down_stages = min(stages, 3 if n <= 16 else 2)
    if down_stages == stages:
        return tactic
    # Preserve the parent's activation lifetime and Up stages. A real geometry
    # check accounts for the extra branch-mix storage of other G values.
    probe = FusedGatedResidualKernel(
        hc_count=hc_count,
        hidden_size=hidden_size,
        lowrank=lowrank,
        up_k=_up_k_tactic(hc_count, hidden_size, lowrank, n, split_k),
        down_x_cache_mode=_down_x_cache_mode_tactic(
            hc_count, hidden_size, lowrank, n, split_k, 1, 1, lp, lb, 1, 0, False
        ),
        down_k_order=_down_k_order_tactic(hc_count, hidden_size, lowrank, n, split_k),
        projection_tile=m,
        token_tile=n,
        token_major_up=_token_major_up_tactic(hc_count, hidden_size, lowrank, n),
        up_output_parts=_up_output_parts_tactic(hc_count, hidden_size, lowrank, n),
        up_token_parts=_up_token_parts_tactic(hc_count, hidden_size, lowrank, n),
        up_stage_counts=_up_stage_counts_tactic(hc_count, hidden_size, lowrank, n),
        rotate_up_k=_rotate_up_k_tactic(hc_count, hidden_size, lowrank, n),
        epilogue_threads=min(_epilogue_threads(n, 1), 256 if pipes > 1 else 512),
        down_token_parts=1,
        projection_rows=projection_rows,
        partial_stride=((projection_rows + 31) // 32) * 32,
        split_k=split_k,
        reduction_groups=groups,
        projection_tiles=lp,
        batch_tiles=lb,
        token_tile_batches=1,
        workers=1,
        up_clusters=0,
        stages=down_stages,
        pin_up_activation=pin,
        reuse_down_storage=reuse,
        final_tile=f,
        rounds=1,
        use_pdl=False,
        scale=1.0,
        weights_independent=True,
        evict_up_weights_first=_evict_up_weights_first_tactic(
            hc_count, hidden_size, lowrank, n, split_k
        ),
        evict_down_weights_first=_evict_down_weights_first_tactic(
            hc_count,
            hidden_size,
            lowrank,
            n,
            split_k,
            m,
            pipes,
            groups,
            1,
            1,
            lp,
            lb,
            1,
            0,
        ),
        single_tile=False,
        full_tiles=False,
        pipelines_per_cta=pipes,
        up_columns=up_columns,
        pipeline_shared_budget=shared_budget,
        down_pair_reduction="auto",
        down_producer_mode="auto",
        up_activation_ready_mode="auto",
        up_l2_prefetch_k=_up_l2_prefetch_k(split_k, reuse),
        up_sigmoid_mode="exp_rcp",
        up_token_tail_overlap=False,
    )
    if probe.down_k != 256:
        return tactic
    return split_k, m, n, lp, lb, down_stages, f, pin, reuse


def prefers_fused_hc(rows: int, hc_count: int, hidden_size: int, lowrank: int) -> bool:
    """Bound serialized MMA work before preferring fusion over library GEMMs.

    Count K16 instructions in M64 units using the conservative resident budget.
    The existing decode tactics fit this budget; wider shapes can still select
    CuTe explicitly while their cooperative pipeline is tuned.

    Args:
        rows: Number of token rows in this invocation.
        hc_count: Number of residual branches.
        hidden_size: Width of one branch.
        lowrank: Rank of the mix projection.

    Returns:
        Whether conservative per-worker work fits the automatic selection budget.
    """
    if rows < 1 or not supports_fused_hc_shape(hc_count, hidden_size, lowrank):
        return False
    if (hc_count, hidden_size, lowrank) == (4, 2560, 320):
        # Preserve the existing model's validated automatic row bound.
        return rows <= 256
    projection_rows = lowrank + hc_count
    s, m, n, lp, lb, stages, f, pin, reuse = _model_tactic(
        rows, hc_count, hidden_size, lowrank, projection_rows
    )
    clusters = (projection_rows + m * lp - 1) // (m * lp)
    ctas = clusters * s
    workers = max(1, _RESIDENT_CTA_BUDGET // ctas)
    rounds = (rows + n * lb * workers - 1) // (n * lb * workers)
    branches = min(4, hc_count & -hc_count)
    final = max(f, 64 // branches)
    up_steps = ((hidden_size // final + ctas - 1) // ctas) * (hc_count // branches)
    down_k = (hc_count * hidden_size + s * 128 - 1) // (s * 128)
    down_work = down_k * 8 * lp * (m // 64)
    up_work = up_steps * ((lowrank + 15) // 16) * (branches * final // 64)
    return rounds * lb * (down_work + up_work) <= _MAX_AUTO_MMA_STEPS


def warmup(
    normalized: torch.Tensor,
    projection_weight: torch.Tensor,
    up_weight: torch.Tensor,
    hc_count: int,
    hidden_size: int,
    lowrank: int,
    *,
    projection_rows: int,
    projection_scale: float,
    weights_independent: bool,
    override: str | None,
    solution: str | None,
) -> None:
    """Warm eligible fused mix tactics, including tails and scheduling rounds.

    Args:
        normalized: Prototype supplying device, dtype and residual width.
        projection_weight: Loaded down/inject projection weight.
        up_weight: Loaded mix-up projection weight.
        hc_count: Number of residual branches.
        hidden_size: Width of each branch.
        lowrank: Rank of the mix gate bottleneck.
        projection_rows: Logical down/inject rows before storage padding.
        projection_scale: Scale applied to the down/inject projection.
        weights_independent: Whether weights are ready before the PDL wait.
        override: Call-site kernel override. Global overrides follow dispatch
            precedence for eligibility and the row bound.
        solution: Explicit solution choice, preserved during selection.

    Returns:
        None. Eligible plans and workspaces remain cached for serving.
    """
    effective_override = resolve_kernel_override(
        "residual", "hyperconnection_mix", override
    )
    if (
        effective_override not in (None, "cute_fused", "cute_fused_hyperconnection_mix")
        or (effective_override is None and solution not in (None, "cute_fused"))
        or not supports_fused_hc_shape(hc_count, hidden_size, lowrank)
        or not weights_independent
        or not supports_fused_hc(normalized.device)
        or normalized.dtype not in (torch.bfloat16, torch.float16)
    ):
        return
    from tokenspeed_kernel.ops.residual import select_gated_residual_mix_kernel

    max_rows = 256 if effective_override is None else _MAX_OVERRIDE_TOKENS
    key = (
        normalized.device.index,
        normalized.dtype,
        hc_count,
        hidden_size,
        lowrank,
        projection_rows,
        projection_scale,
        weights_independent,
        pdl_enabled(),
        max_rows,
        int(torch.cuda.current_stream(normalized.device).cuda_stream),
    )
    if key in _WARMED:
        return
    sample = normalized.new_zeros((max_rows, normalized.shape[1]))
    selected, _ = select_gated_residual_mix_kernel(
        sample[:1],
        projection_weight,
        up_weight,
        hc_count,
        hidden_size,
        lowrank,
        weights_independent=weights_independent,
        override=override,
        solution=solution,
    )
    if selected.name != "cute_fused_hyperconnection_mix":
        return
    # Row counts also select full/tail tiles and occupancy-dependent rounds.
    # Walking the small supported range covers these without duplicating the
    # cooperative scheduler's capacity calculation in a second planner.
    for rows in range(1, max_rows + 1):
        selected(
            sample[:rows],
            projection_weight,
            up_weight,
            hc_count,
            hidden_size,
            lowrank,
            projection_rows,
            projection_scale,
            weights_independent,
        )
    _WARMED.add(key)


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
            "fused_shape_supported": frozenset({True}),
            "fused_workload_supported": frozenset({True}),
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
        projection_rows: int,
        projection_scale: float,
        weights_independent: bool,
        packed_up_weight: torch.Tensor | None = None,
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
        # Native MMA tiles already cover the projection tail. TMA should see
        # only logical rows instead of fetching model-loading padding.
        projection_weight = projection_weight[:projection_rows]
        tensors = (normalized, projection_weight, up_weight)
        # These model dimensions and the row bound keep byte offsets below 2**31.
        if (
            not supports_fused_hc(normalized.device)
            or not supports_fused_hc_shape(hc_count, hidden_size, lowrank)
            or not 1 <= rows <= _MAX_OVERRIDE_TOKENS
            or normalized.dtype not in (torch.bfloat16, torch.float16)
            or any(not t.is_contiguous() or t.data_ptr() % 16 for t in tensors)
        ):
            raise ValueError(
                "fused CuTe HC requires six resident Blackwell clusters and 16-byte-aligned "
                "contiguous BF16/FP16 tensors with validated model dimensions "
                "(4, 2560, 320) or (4, 8192, 1024) and at most 1024 rows"
            )
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
            tactic = _model_tactic(
                rows, hc_count, hidden_size, lowrank, projection_rows
            )
            split_k, m, n, lp, lb, stages, f, pin, reuse = tactic
            pipes, up_columns, shared_budget = _local_pipeline_parameters(
                n, hc_count, hidden_size, lowrank
            )
            single_tile = rows <= n and lb == 1 and pipes == 1
            full_tiles = rows % n == 0
            reduction_groups = (
                4
                if (hc_count, hidden_size, lowrank) == (4, 8192, 1024)
                and rows > 128
                and n <= 64
                else _reduction_groups(hc_count, hidden_size, lowrank)
            )
            token_tile_batches = (
                2
                if (hc_count, hidden_size, lowrank) == (4, 8192, 1024)
                and 96 < rows <= 128
                and n <= 64
                else 1
            )
            # A stream changes available occupancy (for example in a green
            # context), but it does not change generated code. The final plan
            # key adds the worker count derived from that occupancy below.
            static_key = (
                normalized.device.index,
                normalized.dtype,
                hc_count,
                hidden_size,
                lowrank,
                tactic,
                reduction_groups,
                token_tile_batches,
                single_tile,
                full_tiles,
                projection_rows,
                ((projection_rows + 31) // 32) * 32,
                enable_pdl,
                projection_scale,
                weights_independent,
                _evict_up_weights_first_tactic(
                    hc_count, hidden_size, lowrank, n, split_k
                ),
                _USE_UP_TANH,
                _USE_UP_TOKEN_TAIL,
                # The final plan key also includes actual worker count.
                _EVICT_DOWN_WEIGHTS_FIRST,
                min(_epilogue_threads(n, 1), 256 if pipes > 1 else 512),
                pipes,
                up_columns,
                shared_budget,
                _up_k_tactic(hc_count, hidden_size, lowrank, n, split_k),
                _token_major_up_tactic(hc_count, hidden_size, lowrank, n),
                _up_output_parts_tactic(hc_count, hidden_size, lowrank, n),
                _up_token_parts_tactic(hc_count, hidden_size, lowrank, n),
                _up_stage_counts_tactic(hc_count, hidden_size, lowrank, n),
                _rotate_up_k_tactic(hc_count, hidden_size, lowrank, n),
                _down_k_order_tactic(hc_count, hidden_size, lowrank, n, split_k),
                _USE_NATIVE64_DOWN_X_RETENTION,
            )

            def operands_for(kernel):
                workers = (
                    (rows + n * lb - 1) // (n * lb)
                    if kernel.clusters == 1
                    else kernel.workers
                )
                activation_storage, partials, epochs = _workspace_for_plan(
                    normalized.device,
                    projection_rows,
                    kernel.partial_stride,
                    workers,
                    kernel.clusters,
                    kernel.grid_clusters,
                    kernel.slot_rows,
                    lowrank,
                    kernel.reduction_groups,
                    kernel.split_k,
                    kernel.projection_tiles,
                )
                values = (
                    normalized.unsqueeze(-1),
                    projection_weight.unsqueeze(-1),
                    up_weight.unsqueeze(-1),
                    activation_storage.view(normalized.dtype),
                    partials,
                    epochs,
                    out,
                    out if inject is None else inject,
                )
                leading_dims = (1, 1, 1, 1, 1, 0, 1, 1)
                return tuple(
                    from_dlpack(value.detach(), assumed_align=16).mark_layout_dynamic(
                        leading_dim=leading
                    )
                    for value, leading in zip(values, leading_dims)
                )

            def compile_plan(workers, rounds, up_clusters, down_token_parts):
                token_tail_overlap = _up_token_tail_overlap_tactic(
                    hc_count,
                    hidden_size,
                    lowrank,
                    n,
                    split_k,
                    m,
                    pipes,
                    reduction_groups,
                    workers,
                    rounds,
                    lp,
                    lb,
                    token_tile_batches,
                    up_clusters,
                    full_tiles,
                )
                if torch.cuda.is_current_stream_capturing():
                    raise RuntimeError(
                        "fused CuTe HC kernel plan is not initialized for CUDA graph "
                        "capture; warm up this variant before capture"
                    )
                kernel = FusedGatedResidualKernel(
                    hc_count=hc_count,
                    hidden_size=hidden_size,
                    lowrank=lowrank,
                    up_k=_up_k_tactic(hc_count, hidden_size, lowrank, n, split_k),
                    down_x_cache_mode=_down_x_cache_mode_tactic(
                        hc_count,
                        hidden_size,
                        lowrank,
                        n,
                        split_k,
                        workers,
                        rounds,
                        lp,
                        lb,
                        token_tile_batches,
                        up_clusters,
                        full_tiles,
                    ),
                    down_k_order=_down_k_order_tactic(
                        hc_count, hidden_size, lowrank, n, split_k
                    ),
                    projection_tile=m,
                    token_tile=n,
                    token_major_up=_token_major_up_tactic(
                        hc_count, hidden_size, lowrank, n
                    ),
                    up_output_parts=_up_output_parts_tactic(
                        hc_count, hidden_size, lowrank, n
                    ),
                    up_token_parts=_up_token_parts_tactic(
                        hc_count, hidden_size, lowrank, n
                    ),
                    up_stage_counts=_up_stage_counts_tactic(
                        hc_count, hidden_size, lowrank, n
                    ),
                    rotate_up_k=_rotate_up_k_tactic(hc_count, hidden_size, lowrank, n),
                    epilogue_threads=min(
                        _epilogue_threads(n, down_token_parts),
                        256 if pipes > 1 else 512,
                    ),
                    down_token_parts=down_token_parts,
                    projection_rows=projection_rows,
                    partial_stride=((projection_rows + 31) // 32) * 32,
                    split_k=split_k,
                    reduction_groups=reduction_groups,
                    projection_tiles=lp,
                    batch_tiles=lb,
                    token_tile_batches=token_tile_batches,
                    workers=workers,
                    up_clusters=up_clusters,
                    stages=stages,
                    pin_up_activation=pin,
                    reuse_down_storage=reuse,
                    final_tile=f,
                    rounds=rounds,
                    use_pdl=enable_pdl,
                    scale=projection_scale,
                    weights_independent=weights_independent,
                    evict_up_weights_first=_evict_up_weights_first_tactic(
                        hc_count, hidden_size, lowrank, n, split_k
                    ),
                    evict_down_weights_first=_evict_down_weights_first_tactic(
                        hc_count,
                        hidden_size,
                        lowrank,
                        n,
                        split_k,
                        m,
                        pipes,
                        reduction_groups,
                        workers,
                        rounds,
                        lp,
                        lb,
                        token_tile_batches,
                        up_clusters,
                    ),
                    single_tile=single_tile,
                    full_tiles=full_tiles,
                    pipelines_per_cta=pipes,
                    up_columns=up_columns,
                    pipeline_shared_budget=shared_budget,
                    down_pair_reduction="auto",
                    down_producer_mode="auto",
                    up_activation_ready_mode="auto",
                    up_l2_prefetch_k=_up_l2_prefetch_k(split_k, reuse),
                    up_sigmoid_mode=_up_sigmoid_mode_tactic(
                        hc_count,
                        hidden_size,
                        lowrank,
                        n,
                        split_k,
                        m,
                        pipes,
                        reduction_groups,
                        workers,
                        rounds,
                        lp,
                        lb,
                        token_tile_batches,
                        up_clusters,
                        full_tiles,
                    ),
                    up_token_tail_overlap=token_tail_overlap,
                )
                compiled = cute_ext.compile(kernel, *operands_for(kernel), stream)
                return kernel, compiled

            def capacity_for(plan_key, plan, validate_grid):
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
                required = kernel.physical_clusters * (
                    kernel.workers if kernel.clusters > 1 else 1
                )
                if validate_grid and required > capacity:
                    raise RuntimeError(
                        "Compiled fused HC launch exceeds cluster capacity"
                    )
                return capacity

            def plan_for(workers, rounds, up_clusters, down_token_parts):
                plan_key = (*static_key, workers, rounds, up_clusters, down_token_parts)
                plan = _PLANS.get(plan_key)
                if plan is None:
                    with _CACHE_LOCK:
                        plan = _PLANS.get(plan_key)
                        if plan is None:
                            plan = compile_plan(
                                workers, rounds, up_clusters, down_token_parts
                            )
                            _PLANS[plan_key] = plan
                return plan_key, plan

            probe_key, probe = plan_for(1, 1, 0, 1)
            probe_capacity = capacity_for(probe_key, probe, True)
            down_token_parts = 1
            if (
                single_tile
                and n >= 16
                and projection_rows > 6 * m
                and 2 * probe[0].clusters <= probe_capacity
            ):
                # Partition only Down's independent token outputs. Up keeps the
                # full native token tile and its weight/activation reuse.
                partition_key, partition = plan_for(1, 1, 0, 2)
                partition_capacity = capacity_for(partition_key, partition, False)
                if partition[0].grid_clusters <= partition_capacity:
                    probe_key, probe = partition_key, partition
                    probe_capacity = partition_capacity
                    down_token_parts = 2
            workers = 1 if single_tile else probe_capacity // probe[0].physical_clusters
            plan = probe
            if single_tile and probe[0].clusters > 1:
                geometry = probe[0]
                max_clusters = min(
                    probe_capacity,
                    (geometry.hidden_tiles + split_k - 1) // split_k,
                )
                old_steps = (
                    geometry.hidden_tiles + geometry.group_ctas - 1
                ) // geometry.group_ctas
                new_ctas = max_clusters * split_k
                new_steps = (geometry.hidden_tiles + new_ctas - 1) // new_ctas
                if max_clusters > geometry.clusters and new_steps < old_steps:
                    extra = max_clusters - geometry.clusters
                    extra_key, plan = plan_for(1, 1, extra, down_token_parts)
                    capacity_for(extra_key, plan, True)
            if workers > 1:
                worker_key, plan = plan_for(workers, 1, 0, down_token_parts)
                capacity_for(worker_key, plan, True)
            kernel, compiled = plan
            rows_per_round = n * lb * kernel.token_tile_batches * kernel.workers
            rounds = (
                (rows + rows_per_round - 1) // rows_per_round
                if kernel.clusters > 1
                else 1
            )
            if rounds > 1:
                # Bucket by scheduling rounds, not exact rows. Static loop bounds
                # preserve register reuse across the pipeline's warp roles.
                round_key, plan = plan_for(
                    kernel.workers, rounds, kernel.up_clusters, kernel.down_token_parts
                )
                capacity_for(round_key, plan, True)
                kernel, compiled = plan
            compiled(*operands_for(kernel), stream)
        return out, inject
