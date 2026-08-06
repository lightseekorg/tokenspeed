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

"""``triton_shmem`` fused all-reduce + residual + RMSNorm backend.

The fused kernels are embedded as private siblings of this backend and run over
PyTorch symmetric memory. Coarse data buffers use HIP IPC, while the small
signal pad remains a symmetric-memory allocation for system-scope atomics. This
keeps the backend dependency-free and graph-capture safe.

It exposes the same shim contract as ``communication.iris``
(``create_*_state`` + ``*_allreduce_residual_rmsnorm`` + a ``*_STATES`` cache) so
it drops into the ``TS_ARNORM_BACKEND`` switch as the ``triton_shmem`` backend.

Substrate mapping (canonical backend design):

* Allocation: ``symm_mem.empty`` + ``rendezvous`` (via :func:`._alloc_symm`).
* Pointer translation: a **per-tensor** ``buffer_ptrs_dev`` table (via
  :func:`._peer_ptrs_dev`). The one-shot kernels translate only ``input`` (one
  table); the two-shot kernel pushes into peers' ``output`` and ``residual_out``
  too, so it takes three tables.
* Barriers: a single-block signal-pad barrier kernel
  (:func:`._triton_shmem_kernels.symm_grid_barrier_kernel`).
  We issue a
  **leading** barrier (all peers' inputs visible before any pull) and a
  **trailing** barrier (all peer pushes visible + all peer reads done) for
  **every** variant. The trailing barrier is required even for the one-shot
  kernels -- although their outputs are purely local, the persistent symmetric
  ``input`` buffer is reused across calls (repeated captured decode graphs), so
  peers must finish reading it before the next call overwrites it. This matches
  the native ``amd_allreduce_residual_rmsnorm_kernel`` (entry + exit barrier) and
  Iris (``device_barrier`` before + after).

PERFORMANCE: torch symm_mem on ROCm allocates
**fine-grained** memory (HIP VMM, no coherence knob) which bypasses L2 and
delivers only ~105 GB/s for bulk local access vs. ~3200 GB/s coarse-grained --
a ~30x penalty on copy-in/out and the kernel's local reads/writes. This is fixed
by ``_coarse_shmem.py``: the *data* buffers are coarse-grained ``torch.empty``
tensors shared peer-to-peer via HIP IPC (like the rocSHMEM heap), leaving only
the signal pad fine-grained. On by default (``TS_TRITON_SHMEM_COARSE``; set ``0``
for the legacy fine-grained path). Coarse is 5-38x faster (ws=8) and lands at
0.5-0.83x RCCL, competitive with the rocSHMEM reference. Remote (xGMI) access is
fabric-bound either way; the win is entirely on local bandwidth.

Scope (matches the Iris/native contract): AMD ROCm, bf16, 2-D
``(num_tokens, hidden)`` input, ``weight`` of shape ``(hidden,)``. The reduction
spans ``group.size()`` ranks; symm_mem rendezvous accepts any process group, so
this is not restricted to the whole world, but the current dispatch only
exercises whole-world TP.
"""

import logging
import os

import torch
import torch.distributed as dist
import torch.distributed._symmetric_memory as symm_mem
from tokenspeed_kernel._triton import triton
from tokenspeed_kernel.platform import current_platform
from tokenspeed_kernel.profiling import kernel_scope

from . import _triton_shmem_kernels as _k
from . import _triton_shmem_profile as _p
from ._coarse_shmem import CoarseSymmBuffer, alloc_coarse_symm
from .triton import _alloc_symm, _peer_ptrs_dev

logger = logging.getLogger(__file__)

_platform = current_platform()
_STATE_ENV_KEYS = (
    "TS_TRITON_SHMEM_COARSE",
    "TS_TRITON_SHMEM_INKERNEL_BARRIER",
    "TS_TRITON_SHMEM_FOLD_COPYIN",
    "TS_TRITON_SHMEM_DOUBLE_BUFFER_INPUT",
    "TS_TRITON_SHMEM_BORROW_TWOSHOT_OUTPUT",
    "TS_TRITON_SHMEM_INPUT_SITE_RING",
    "TS_TRITON_SHMEM_OUTPUT_RING",
    "TS_TRITON_SHMEM_WORKGROUP_SYNC",
    "TS_TRITON_SHMEM_FOLD_NUM_WARPS",
    "TS_TRITON_SHMEM_ONESHOT_BLOCK_N",
    "TS_TRITON_SHMEM_ONESHOT_VARIANT",
    "TS_TRITON_SHMEM_ONESHOT_NUM_WARPS",
    "TS_TRITON_SHMEM_PADDED_MAX_M",
    "TS_TRITON_SHMEM_TWOSHOT_BLOCK_N",
    "TS_TRITON_SHMEM_GRID_CAP",
    "TS_TRITON_SHMEM_GRID_CAP_MIN_M",
    "TS_TRITON_SHMEM_BARRIER_GRID",
    "TS_TRITON_SHMEM_ONESHOT_MAX_M",
)


def _coarse_enabled() -> bool:
    """Whether to back the *data* buffers with coarse-grained HBM + HIP IPC
    instead of fine-grained symm_mem. Default ON: the
    fine-grained substrate is ~30x slower for all local access. Set
    ``TS_TRITON_SHMEM_COARSE=0`` to force the legacy symm_mem-only path."""
    return os.environ.get("TS_TRITON_SHMEM_COARSE", "1") not in ("0", "false", "False")


def _inkernel_barrier_enabled() -> bool:
    """Fold the leading+trailing signal-pad barriers into the fused kernels
    instead of launching two separate barrier kernels — removes the barrier-launch
    staging overhead that dominates small-M/decode latency.
    **Default OFF:** the generic backend must remain safe when ranks can diverge
    in M. Qualified pure-TP profiles may set
    ``TS_TRITON_SHMEM_INKERNEL_BARRIER=1`` after transition testing.

    CAVEAT — under configs that can diverge M across TP ranks (DP,
    ``overlap_schedule_depth>1``, speculative decode) the M-dependent in-kernel
    barrier DEADLOCKS if two ranks run it over different M simultaneously (its slot
    range is M-dependent). Pure TP (dp=1, overlap_schedule_depth=1) never diverges
    M, so the default is safe there. For divergent configs, set the fixed-grid
    barrier (``TS_TRITON_SHMEM_BARRIER_GRID>0``, lever B) -- divergence-safe and
    still faster than falling back to ``INKERNEL_BARRIER=0``. Repro:
    ``benchmark/probe_inkernel_barrier_graph.py`` (PROBE_MODE=multigraph)."""
    return os.environ.get("TS_TRITON_SHMEM_INKERNEL_BARRIER", "0") not in (
        "0",
        "false",
        "False",
    )


def _fold_copyin_enabled() -> bool:
    """Fold the input copy-in into the one-shot fused kernel: each rank writes its
    local input into its symmetric buffer in a phase-0 pass, ordered before the
    pull by the in-kernel leading barrier, instead of a separate ``copy_`` launch
    Removes the ~0.012 ms/op copy-in launch that is an
    untouched floor at decode M. Requires the in-kernel barrier (the phase-0 write
    must precede the leading barrier); no-op on the two-shot / separate-barrier
    path. Default OFF: serving qualification reproduced a graph-to-eager small-M
    fault despite the narrower synthetic transition probe passing. Re-enable
    only after the producer-to-system-release publication contract is proven."""
    return os.environ.get("TS_TRITON_SHMEM_FOLD_COPYIN", "0") not in (
        "0",
        "false",
        "False",
    )


def _double_buffer_input_enabled() -> bool:
    """Use two symmetric input slots and omit the one-shot exit barrier.

    Diagnostic only. The next call's leading barrier proves peers completed the
    prior slot before it is reused two calls later. Captured graphs must contain
    an even number of fused calls (GPT-OSS has 72); odd-call graph replay is not
    covered by this lifetime contract.
    """
    return os.environ.get("TS_TRITON_SHMEM_DOUBLE_BUFFER_INPUT", "0") not in (
        "0",
        "false",
        "False",
    )


def _borrow_twoshot_output_enabled() -> bool:
    """Return state-owned symmetric two-shot outputs instead of copying them.

    This is restricted to eager calls with backend-owned outputs. Two symmetric
    output pairs are ping-ponged so the next fused site never overwrites the
    residual tensor it is simultaneously reading.
    """
    return os.environ.get("TS_TRITON_SHMEM_BORROW_TWOSHOT_OUTPUT", "0") not in (
        "0",
        "false",
        "False",
    )


def _input_site_ring_size() -> int:
    """Number of graph-stable one-shot input sites reserved by the profile."""
    value = int(os.environ.get("TS_TRITON_SHMEM_INPUT_SITE_RING", "0"))
    if value < 0 or value == 1:
        raise ValueError("TS_TRITON_SHMEM_INPUT_SITE_RING must be 0 or at least 2")
    return value


def _workgroup_sync_enabled() -> bool:
    """Bracket each scalar cross-rank signal barrier with a workgroup barrier.

    Required whenever a Triton program uses multiple wavefronts: the scalar
    system-scope release/acquire must represent all wavefronts' preceding stores
    or reads. The opt-out exists only to reproduce the pre-fix race."""
    return os.environ.get("TS_TRITON_SHMEM_WORKGROUP_SYNC", "1") not in (
        "0",
        "false",
        "False",
    )


def _fold_num_warps() -> int:
    """Wavefront count for the folded copy-in specialization.

    A single wavefront makes the scalar system-scope signal barrier order the
    entire program's phase-0 stores and peer reads. Multi-wave folded kernels
    remain available for diagnostics while their system-fence semantics are
    qualified."""
    value = int(os.environ.get("TS_TRITON_SHMEM_FOLD_NUM_WARPS", "1"))
    if value not in (1, 2, 4, 8):
        raise ValueError("TS_TRITON_SHMEM_FOLD_NUM_WARPS must be 1, 2, 4, or 8")
    return value


def _oneshot_block_n() -> int:
    """Optional diagnostic override for the blocked one-shot tile width.

    ``0`` keeps the architecture recommendation. A narrow override lets the
    profiled small-M path be tuned without perturbing the two-shot kernel.
    """
    value = int(os.environ.get("TS_TRITON_SHMEM_ONESHOT_BLOCK_N", "0"))
    if value < 0 or (value and value & (value - 1)):
        raise ValueError(
            "TS_TRITON_SHMEM_ONESHOT_BLOCK_N must be 0 or a positive power of two"
        )
    return value


def _oneshot_variant() -> str:
    value = os.environ.get("TS_TRITON_SHMEM_ONESHOT_VARIANT", "auto")
    if value not in ("auto", "blocked", "padded"):
        raise ValueError(
            "TS_TRITON_SHMEM_ONESHOT_VARIANT must be auto, blocked, or padded"
        )
    return value


def _oneshot_num_warps() -> int:
    value = int(os.environ.get("TS_TRITON_SHMEM_ONESHOT_NUM_WARPS", "0"))
    if value not in (0, 1, 2, 4, 8):
        raise ValueError("TS_TRITON_SHMEM_ONESHOT_NUM_WARPS must be 0, 1, 2, 4, or 8")
    return value


def _padded_max_m() -> int:
    value = int(os.environ.get("TS_TRITON_SHMEM_PADDED_MAX_M", "64"))
    if value < 0:
        raise ValueError("TS_TRITON_SHMEM_PADDED_MAX_M must be non-negative")
    return value


def _twoshot_block_n() -> int:
    """Optional diagnostic override for the blocked two-shot tile width."""
    value = int(os.environ.get("TS_TRITON_SHMEM_TWOSHOT_BLOCK_N", "0"))
    if value < 0 or (value and value & (value - 1)):
        raise ValueError(
            "TS_TRITON_SHMEM_TWOSHOT_BLOCK_N must be 0 or a positive power of two"
        )
    return value


def _dynamic_grid_cap() -> int:
    """Optional cap on the normal M-dependent compute grid.

    Unlike ``BARRIER_GRID``, this does not add zero-row participants or change
    divergence semantics; it only limits compute/barrier parallelism. The
    generic default is uncapped; model-specific profiles own tuned caps."""
    return int(os.environ.get("TS_TRITON_SHMEM_GRID_CAP", "0"))


def _dynamic_grid_cap_min_m() -> int:
    return int(os.environ.get("TS_TRITON_SHMEM_GRID_CAP_MIN_M", "0"))


def _barrier_grid() -> int:
    """Fixed in-kernel-barrier participant/grid width `G`.
    `0` (DEFAULT) = M-dependent grid (`min(cap, M, num_cus)`); `>0` = launch
    exactly `min(G, num_cus)` blocks for EVERY call regardless of M, striding rows
    over G.

    A fixed `G` makes the barrier participant set **M-independent** (block index
    i in [0,G) present on all ranks, zero-row blocks still barrier), which is
    **divergence-safe** under cross-rank M-divergence (DP / overlap>1 /
    spec-decode) -- validated: it flips the multigraph-divergent probe HANG->PASS.
    Correctness relies on the one-shot pull's disjoint row partition (block i owns
    rows {i, i+G, ...}; no cross-block data dep → only a per-block cross-rank
    barrier is needed, never a within-GPU grid sync). `G` MUST be <= num_cus with
    headroom so all G blocks stay co-resident (the spin barrier deadlocks else).

    **DEFAULT OFF (measured):** a fixed `G` is a net PERFORMANCE LOSS vs the
    M-dependent grid -- the barrier cost scales with participant count, so no fixed
    `G` is perf-neutral (small `G` starves large-M parallelism; large `G` makes the
    small-M barrier expensive; the M=256 dip is not fixed). Use `>0` **only**
    for M-diverging configs, where it beats the alternative (INKERNEL_BARRIER=0,
    separate-barrier path). Tune `G` to the config's decode-M range."""
    return int(os.environ.get("TS_TRITON_SHMEM_BARRIER_GRID", "0"))


def _oneshot_max_m() -> int:
    """At ws>=4 (two-shot state), route calls with ``M <= this`` through the
    one-shot pull kernel instead of two-shot. One-shot writes the output locally,
    so it skips the two-shot symmetric copy-out, and with in-kernel barriers has
    minimal fixed overhead -- decisive in the small-M/decode regime where
    two-shot's bandwidth advantage does not apply. Two-shot's
    bandwidth-optimality still wins at large M. The conservative default remains
    256; isolated width-specific crossovers require model-level serving gates."""
    return int(os.environ.get("TS_TRITON_SHMEM_ONESHOT_MAX_M", "256"))


def _configuration_errors(
    *,
    arch: str,
    world_size: int,
    max_token_num: int,
    hidden_dim: int,
    dtype: torch.dtype,
) -> list[str]:
    """Return unsupported backend properties before allocating communication state."""
    errors: list[str] = []
    if arch != "gfx950":
        errors.append(f"arch={arch!r} is unsupported; expected 'gfx950'")
    if world_size not in {1, 2, 4, 8}:
        errors.append(f"world_size={world_size} is unsupported")
    if max_token_num <= 0:
        errors.append(f"max_token_num={max_token_num} must be positive")
    if hidden_dim <= 0:
        errors.append(f"hidden_dim={hidden_dim} must be positive")
    if dtype != torch.bfloat16:
        errors.append(f"dtype={dtype} is unsupported; expected torch.bfloat16")
    return errors


def _configuration_is_eligible(
    *,
    group: dist.ProcessGroup,
    max_token_num: int,
    hidden_dim: int,
    dtype: torch.dtype,
    device: torch.device,
) -> bool:
    """Validate the supported hardware and shape contract on every rank."""
    local_errors = _configuration_errors(
        arch=_p.detect_arch(device.index),
        world_size=group.size(),
        max_token_num=max_token_num,
        hidden_dim=hidden_dim,
        dtype=dtype,
    )
    reports: list = [None] * group.size()
    dist.all_gather_object(reports, local_errors, group=group)
    errors = [
        f"rank {rank}: {error}"
        for rank, rank_errors in enumerate(reports)
        for error in rank_errors
    ]
    if errors:
        logger.warning(
            "triton_shmem declined before state creation: %s",
            "; ".join(errors),
        )
        return False
    logger.info(
        "triton_shmem configuration resolved: arch=%s ws=%d hidden=%d "
        "dtype=%s max_tokens=%d",
        _p.detect_arch(device.index),
        group.size(),
        hidden_dim,
        dtype,
        max_token_num,
    )
    return True


def triton_shmem_state_cache_key(
    group: dist.ProcessGroup,
    max_token_num: int,
    hidden_dim: int,
    dtype: torch.dtype,
) -> tuple:
    """Key state by allocation shape and every environment-owned policy."""
    return (
        id(group),
        max_token_num,
        hidden_dim,
        dtype,
        torch.cuda.current_device(),
        tuple((name, os.environ.get(name)) for name in _STATE_ENV_KEYS),
    )


__all__ = [
    "TritonShmemAllReduceResidualRMSNorm",
    "create_triton_shmem_ar_rmsnorm_state",
    "triton_shmem_allreduce_residual_rmsnorm",
    "is_available",
    "TRITON_SHMEM_AR_RMSNORM_STATES",
    "_configuration_errors",
    "triton_shmem_can_run",
    "triton_shmem_state_cache_key",
]


# State cache includes all policy that affects allocation, synchronization, or
# graph lifetime. Diagnostic overrides therefore cannot reuse an incompatible
# state created earlier in the process.
TRITON_SHMEM_AR_RMSNORM_STATES: dict = {}


def is_available() -> bool:
    """Whether the triton_shmem (symm_mem) fused backend can run here."""
    return _platform.is_amd


def _num_cus(device: torch.device) -> int:
    # A single-node MI350X system is homogeneous, so the local CU count is
    # rank-consistent (no cross-rank MIN needed).
    return torch.cuda.get_device_properties(device).multi_processor_count


class TritonShmemAllReduceResidualRMSNorm:
    """symm_mem-backed fused all-reduce + residual-add + RMSNorm.

    Holds persistent symmetric ``(max_token_num, hidden_dim)`` input (and, for
    the two-shot kernel, output + residual_out) buffers plus their per-tensor
    peer-pointer tables and a local fp32 scratch, and dispatches to the tuned
    kernel variant per call.
    """

    def __init__(
        self,
        group: dist.ProcessGroup,
        rank_in_group: int,
        max_token_num: int,
        hidden_dim: int,
        dtype: torch.dtype = torch.bfloat16,
        device: torch.device | None = None,
    ) -> None:
        assert _platform.is_amd, (
            "TritonShmemAllReduceResidualRMSNorm targets AMD ROCm; "
            f"got non-AMD platform: {_platform}"
        )
        assert dist.is_initialized(), (
            "torch.distributed must be initialized before constructing "
            "TritonShmemAllReduceResidualRMSNorm."
        )

        self.group = group
        self.rank_in_group = rank_in_group
        self.max_token_num = max_token_num
        self.hidden_dim = hidden_dim
        self.dtype = dtype
        self.device = device or torch.device(f"cuda:{torch.cuda.current_device()}")
        self.world_size = group.size()

        # The (ws, hidden) pair is fixed for this state's lifetime, so the tuned
        # kernel variant is chosen once here.
        self.kernel = _p.recommended_kernel(self.world_size, hidden_dim)
        self._is_twoshot = self.kernel == "twoshot_blocked"
        self._num_cus = _num_cus(self.device)

        # Reserve the signal pad before the first symm_mem.empty (the pad size is
        # baked into the allocation). The whole-grid barrier kernel launches at
        # most `num_cus` blocks (recommended_grid caps grid_sms at num_cus) and
        # indexes the pad at block_id * ws + rank, so num_cus * ws uint32 slots
        # suffice for any grid we launch. max() never shrinks another module's pad.
        pad_bytes = self._num_cus * self.world_size * 4
        symm_mem.set_signal_pad_size(max(symm_mem.get_signal_pad_size(), pad_bytes))

        shape = (max_token_num, hidden_dim)
        itemsize = torch.empty((), dtype=dtype).element_size()
        buf_bytes = max_token_num * hidden_dim * itemsize

        # Data buffers can be coarse-grained HBM (full bandwidth) shared via HIP
        # IPC, with only the signal pad left fine-grained.
        # This recovers the ~30x local-bandwidth penalty of fine-grained symm_mem.
        self._coarse = _coarse_enabled()
        self._double_buffer_input = _double_buffer_input_enabled()
        self._borrow_twoshot_output = _borrow_twoshot_output_enabled()
        self._coarse_buffers: list = []
        self._opened_cache: dict = {}
        self._coarse_fallback = False

        if self._coarse:
            # Tiny dedicated symm_mem allocation carries the fine-grained signal
            # pad and provides the rank/world identity used by the barrier.
            self._pad_tensor, pad_hdl = _alloc_symm((1,), dtype, self.device, group)
            self._signal_pad = pad_hdl.signal_pad_ptrs_dev
            self.my_pe = pad_hdl.rank
            ws_check = pad_hdl.world_size
            xb = self._alloc_data(shape, dtype, group)
            self._x, self._input_bases = xb.tensor, xb.peer_ptrs_dev
            if self._double_buffer_input:
                xb_alt = self._alloc_data(shape, dtype, group)
                self._x_alt = xb_alt.tensor
                self._input_bases_alt = xb_alt.peer_ptrs_dev
        else:
            self._pad_tensor = None
            # Input is always pulled by peers -> must be symmetric.
            self._x, x_hdl = _alloc_symm(shape, dtype, self.device, group)
            self._signal_pad = x_hdl.signal_pad_ptrs_dev
            self.my_pe = x_hdl.rank
            ws_check = x_hdl.world_size
            self._input_bases = _peer_ptrs_dev(
                x_hdl, shape, dtype, self.world_size, self.device
            )
            if self._double_buffer_input:
                self._x_alt, x_alt_hdl = _alloc_symm(shape, dtype, self.device, group)
                self._input_bases_alt = _peer_ptrs_dev(
                    x_alt_hdl,
                    shape,
                    dtype,
                    self.world_size,
                    self.device,
                )
        assert (
            self.my_pe == rank_in_group
        ), f"rank mismatch: rank_in_group={rank_in_group}, symm_mem rank={self.my_pe}"
        assert (
            ws_check == self.world_size
        ), f"symm_mem world {ws_check} != group size {self.world_size}"

        self._input_ring = [self._x]
        self._input_bases_ring = [self._input_bases]
        if self._double_buffer_input:
            self._input_ring.append(self._x_alt)
            self._input_bases_ring.append(self._input_bases_alt)
        self._input_ring_index = 0

        n_symm = len(self._input_ring)
        if self._is_twoshot:
            # Two-shot pushes normalized output and pre-norm residual_out into
            # peers, so both must be peer-accessible and each needs its OWN
            # pointer table (offsets are not shared across allocations, §4).
            if self._coarse:
                yb = self._alloc_data(shape, dtype, group)
                rb = self._alloc_data(shape, dtype, group)
                self._y, self._output_bases = yb.tensor, yb.peer_ptrs_dev
                self._residual_out, self._residual_out_bases = (
                    rb.tensor,
                    rb.peer_ptrs_dev,
                )
            else:
                self._y, y_hdl = _alloc_symm(shape, dtype, self.device, group)
                self._residual_out, r_hdl = _alloc_symm(
                    shape, dtype, self.device, group
                )
                self._output_bases = _peer_ptrs_dev(
                    y_hdl, shape, dtype, self.world_size, self.device
                )
                self._residual_out_bases = _peer_ptrs_dev(
                    r_hdl, shape, dtype, self.world_size, self.device
                )
            n_symm += 2
            self._twoshot_y_ring = [self._y]
            self._twoshot_residual_ring = [self._residual_out]
            self._twoshot_output_bases_ring = [self._output_bases]
            self._twoshot_residual_bases_ring = [self._residual_out_bases]
            if self._borrow_twoshot_output:
                if self._coarse:
                    yb_alt = self._alloc_data(shape, dtype, group)
                    rb_alt = self._alloc_data(shape, dtype, group)
                    y_alt, y_alt_bases = yb_alt.tensor, yb_alt.peer_ptrs_dev
                    r_alt, r_alt_bases = rb_alt.tensor, rb_alt.peer_ptrs_dev
                else:
                    y_alt, y_alt_hdl = _alloc_symm(shape, dtype, self.device, group)
                    r_alt, r_alt_hdl = _alloc_symm(shape, dtype, self.device, group)
                    y_alt_bases = _peer_ptrs_dev(
                        y_alt_hdl, shape, dtype, self.world_size, self.device
                    )
                    r_alt_bases = _peer_ptrs_dev(
                        r_alt_hdl, shape, dtype, self.world_size, self.device
                    )
                self._twoshot_y_ring.append(y_alt)
                self._twoshot_residual_ring.append(r_alt)
                self._twoshot_output_bases_ring.append(y_alt_bases)
                self._twoshot_residual_bases_ring.append(r_alt_bases)
                n_symm += 2
            self._twoshot_output_index = 0
        else:
            self._y = None
            self._residual_out = None
            self._output_bases = None
            self._residual_out_bases = None
            self._twoshot_y_ring = []
            self._twoshot_residual_ring = []
            self._twoshot_output_bases_ring = []
            self._twoshot_residual_bases_ring = []
            self._twoshot_output_index = 0

        # Local fp32 scratch: two-shot owns cdiv(M, ws) shard rows; the one-shot
        # blocked kernel reuses one row-slot per persistent program (<= #CUs).
        # oneshot_wholerow needs no scratch.
        if self._is_twoshot:
            m_shard_max = triton.cdiv(max_token_num, self.world_size)
            self._scratch = torch.empty(
                (m_shard_max, hidden_dim), dtype=torch.float32, device=self.device
            )
        elif self.kernel == "oneshot_blocked":
            self._scratch = torch.empty(
                (self._num_cus, hidden_dim), dtype=torch.float32, device=self.device
            )
        else:  # oneshot_wholerow
            self._scratch = None

        # Small-M one-shot path: at ws>=4 the dispatched kernel
        # is two-shot, but small-M/decode calls route through one-shot pull, which
        # writes output locally (skips the two-shot symmetric copy-out) and folds
        # its barriers in-kernel -- both dominate small-M latency. Reuses the
        # (already symmetric) input buffer; needs its own fp32 scratch.
        self._inkernel = _inkernel_barrier_enabled()
        self._fold_copyin = _fold_copyin_enabled()
        self._workgroup_sync = _workgroup_sync_enabled()
        self._fold_num_warps = _fold_num_warps()
        self._oneshot_variant = _oneshot_variant()
        self._padded_max_m = _padded_max_m()
        configured_oneshot_block_n = _oneshot_block_n()
        self._oneshot_block_n = configured_oneshot_block_n or _p.recommended_block_n(
            self.dtype, hidden_dim
        )
        configured_twoshot_block_n = _twoshot_block_n()
        self._twoshot_block_n = configured_twoshot_block_n or _p.recommended_block_n(
            self.dtype, hidden_dim
        )
        self._dynamic_grid_cap = max(0, _dynamic_grid_cap())
        self._dynamic_grid_cap_min_m = max(0, _dynamic_grid_cap_min_m())
        self._barrier_grid = _barrier_grid()
        self._oneshot_max_m = max(0, _oneshot_max_m())
        self._input_site_ring_size = _input_site_ring_size()
        if self._input_site_ring_size and self._double_buffer_input:
            raise ValueError(
                "TS_TRITON_SHMEM_INPUT_SITE_RING and "
                "TS_TRITON_SHMEM_DOUBLE_BUFFER_INPUT are mutually exclusive"
            )
        self._input_site_ring_index = 0
        self._input_site_max_m = 0
        self._input_site_tensor = None
        self._input_site_bases = None
        input_site_bytes = 0
        if self._input_site_ring_size:
            if not self._inkernel or self._oneshot_max_m <= 0:
                raise ValueError(
                    "TS_TRITON_SHMEM_INPUT_SITE_RING requires the in-kernel "
                    "barrier and a positive TS_TRITON_SHMEM_ONESHOT_MAX_M"
                )
            self._input_site_max_m = min(max_token_num, self._oneshot_max_m)
            site_shape = (
                self._input_site_ring_size * self._input_site_max_m,
                hidden_dim,
            )
            if self._coarse:
                site_buffer = self._alloc_data(site_shape, dtype, group)
                self._input_site_tensor = site_buffer.tensor
                self._input_site_bases = site_buffer.peer_ptrs_dev
            else:
                self._input_site_tensor, site_hdl = _alloc_symm(
                    site_shape, dtype, self.device, group
                )
                self._input_site_bases = _peer_ptrs_dev(
                    site_hdl, site_shape, dtype, self.world_size, self.device
                )
            input_site_bytes = (
                self._input_site_ring_size
                * self._input_site_max_m
                * hidden_dim
                * itemsize
            )
        if self._is_twoshot:
            self._oneshot_kernel = (
                "oneshot_wholerow"
                if (hidden_dim & (hidden_dim - 1)) == 0
                else "oneshot_blocked"
            )
            if self._oneshot_variant == "blocked":
                self._oneshot_kernel = "oneshot_blocked"
            elif self._oneshot_variant == "padded":
                self._oneshot_kernel = "oneshot_wholerow_padded"
            self._oneshot_scratch = torch.empty(
                (self._num_cus, hidden_dim), dtype=torch.float32, device=self.device
            )
        else:
            self._oneshot_kernel = self.kernel  # ws<=2 is already one-shot
            if self._oneshot_variant == "blocked":
                self._oneshot_kernel = "oneshot_blocked"
            elif self._oneshot_variant == "padded":
                self._oneshot_kernel = "oneshot_wholerow_padded"
            if self._oneshot_kernel == "oneshot_blocked" and self._scratch is None:
                self._oneshot_scratch = torch.empty(
                    (self._num_cus, hidden_dim),
                    dtype=torch.float32,
                    device=self.device,
                )
            else:
                self._oneshot_scratch = self._scratch
        configured_oneshot_num_warps = _oneshot_num_warps()
        self._oneshot_num_warps = configured_oneshot_num_warps or (
            8
            if self._oneshot_kernel == "oneshot_wholerow"
            else (
                4
                if self._oneshot_kernel == "oneshot_wholerow_padded"
                else _p.recommended_num_warps(self._oneshot_kernel)
            )
        )
        self._oneshot_padded_block_n = triton.next_power_of_2(hidden_dim)

        # Optional model-profile lifetime contract for outputs referenced by a
        # captured custom kernel. Allocate before capture warmup so graph replay
        # never depends on transient graph-pool output storage. The configured
        # size must cover every simultaneously live fused site; generic default
        # stays off because that count is model/control-flow specific.
        self._output_ring_size = int(os.environ.get("TS_TRITON_SHMEM_OUTPUT_RING", "0"))
        if self._output_ring_size < 0:
            raise ValueError("TS_TRITON_SHMEM_OUTPUT_RING must be non-negative")
        if self._output_ring_size:
            self._output_ring_max_m = min(
                max_token_num,
                self._oneshot_max_m if self._oneshot_max_m > 0 else max_token_num,
            )
            self._norm_output_ring = torch.empty(
                (
                    self._output_ring_size,
                    self._output_ring_max_m,
                    hidden_dim,
                ),
                dtype=dtype,
                device=self.device,
            )
            self._residual_output_ring = torch.empty_like(self._norm_output_ring)
            self._output_ring_index = 0
        else:
            self._output_ring_max_m = 0
            self._norm_output_ring = None
            self._residual_output_ring = None
            self._output_ring_index = 0

        logger.info(
            "triton_shmem AR+RMSNorm state: kernel=%s ws=%d max_tokens=%d hidden=%d "
            "substrate=%s data=%.1f MiB/rank inkernel_barrier=%s fold_copyin=%s "
            "workgroup_sync=%s fold_num_warps=%d oneshot_block_n=%d "
            "twoshot_block_n=%d oneshot_variant=%s padded_max_m=%d "
            "oneshot_num_warps=%d "
            "grid_cap=%d grid_cap_min_m=%d "
            "oneshot_default=%s oneshot_max_m=%d input_ring=%d output_ring=%d "
            "input_site_ring=%d "
            "borrow_twoshot_output=%s twoshot_output_ring=%d "
            "oneshot_exit_barrier=%s",
            self.kernel,
            self.world_size,
            max_token_num,
            hidden_dim,
            (
                "coarse+ipc"
                if self._coarse
                else (
                    "mixed/fine-fallback" if self._coarse_fallback else "symm_mem(fine)"
                )
            ),
            (n_symm * buf_bytes + input_site_bytes) / 1024**2,
            self._inkernel,
            self._fold_copyin and self._inkernel,
            self._workgroup_sync,
            self._fold_num_warps,
            self._oneshot_block_n,
            self._twoshot_block_n,
            self._oneshot_variant,
            self._padded_max_m,
            self._oneshot_num_warps,
            self._dynamic_grid_cap,
            self._dynamic_grid_cap_min_m,
            self._oneshot_kernel if self._is_twoshot else "n/a",
            self._oneshot_max_m if self._is_twoshot else 0,
            len(self._input_ring),
            self._output_ring_size,
            self._input_site_ring_size,
            self._borrow_twoshot_output,
            len(self._twoshot_y_ring),
            not (self._double_buffer_input or self._input_site_ring_size > 0),
        )

    def _alloc_data(self, shape, dtype, group):
        """Allocate peer data, falling back collectively when HIP IPC declines."""
        if not self._coarse:
            tensor, handle = _alloc_symm(shape, dtype, self.device, group)
            return CoarseSymmBuffer(
                tensor=tensor,
                peer_ptrs_dev=_peer_ptrs_dev(
                    handle,
                    shape,
                    dtype,
                    self.world_size,
                    self.device,
                ),
            )
        try:
            buf = alloc_coarse_symm(
                shape, dtype, self.device, group, _opened_cache=self._opened_cache
            )
        except RuntimeError as exc:
            # alloc_coarse_symm exchanges export/open status before raising, so
            # every rank enters this fallback together.
            logger.warning(
                "triton_shmem coarse HIP-IPC allocation declined; using "
                "fine-grained symmetric memory for this state: %s",
                exc,
            )
            self._coarse = False
            self._coarse_fallback = True
            tensor, handle = _alloc_symm(shape, dtype, self.device, group)
            return CoarseSymmBuffer(
                tensor=tensor,
                peer_ptrs_dev=_peer_ptrs_dev(
                    handle,
                    shape,
                    dtype,
                    self.world_size,
                    self.device,
                ),
            )
        self._coarse_buffers.append(buf)
        return buf

    def __del__(self) -> None:
        for buf in getattr(self, "_coarse_buffers", []):
            try:
                buf.close()
            except Exception:  # noqa: BLE001 - best-effort teardown
                pass

    def _barrier(self) -> None:
        # Global signal-pad barrier at the kernel-launch boundary. A single
        # block per rank is sufficient: each rank's block signals every peer and
        # waits for every peer (all-to-all), and kernel-launch serialization does
        # the rest. Using 1 block (vs. the fused grid width) keeps the barrier
        # cheap -- it dominates the small-M decode regime otherwise.
        _k.symm_grid_barrier_kernel[(1,)](
            self._signal_pad,
            RANK=self.my_pe,
            WORLD_SIZE=self.world_size,
            num_warps=1,
        )

    def _grid_width(self, kern, ws, work_rows, inkernel):
        """Persistent-grid width. When the in-kernel
        barrier is active and ``_barrier_grid>0``, use a FIXED ``min(G, num_cus)``
        blocks for every call (M-independent → the barrier participant set matches
        across ranks regardless of M, and its cost is decoupled from M). Otherwise
        the legacy M-dependent ``min(cap, work_rows, num_cus)``."""
        if inkernel and self._barrier_grid > 0:
            return max(1, min(self._barrier_grid, self._num_cus))
        grid = _p.recommended_grid(kern, ws, work_rows, self._num_cus)
        if (
            self._dynamic_grid_cap > 0
            and kern.startswith("oneshot")
            and work_rows >= self._dynamic_grid_cap_min_m
        ):
            grid = min(grid, self._dynamic_grid_cap)
        return max(1, grid)

    def _oneshot_kernel_for_m(self, m: int) -> str:
        if (
            self._oneshot_kernel == "oneshot_wholerow_padded"
            and self._padded_max_m > 0
            and m > self._padded_max_m
        ):
            return "oneshot_blocked"
        return self._oneshot_kernel

    def _run_oneshot(
        self,
        x,
        input_bases,
        local_src,
        residual,
        weight,
        eps,
        m,
        n,
        ws,
        norm_out,
        residual_out,
        fold,
        exit_barrier,
    ):
        """One-shot pull into the caller's *local* output (no copy-out). Barriers
        are folded into the kernel when ``self._inkernel`` (default), else issued
        as the legacy separate launches. Reads the symmetric input ``x``; writes
        ``norm_out``/``residual_out`` directly. When ``fold`` (lever A), the kernel
        also writes ``local_src`` into ``x`` (symmetric) in a phase-0 pass before
        the leading barrier, so the caller skipped the separate ``copy_``."""
        kern = self._oneshot_kernel_for_m(m)
        inkernel = self._inkernel
        grid_sms = self._grid_width(kern, ws, m, inkernel)
        grid = (grid_sms,)
        num_warps = self._oneshot_num_warps
        if fold:
            num_warps = self._fold_num_warps
        if not inkernel:
            self._barrier()  # leading (legacy separate-barrier path)
        if kern == "oneshot_wholerow":
            _k.fused_ar_rmsnorm_oneshot_wholerow_kernel[grid](
                x,
                norm_out,
                eps,
                weight,
                self.my_pe,
                input_bases,
                residual,
                residual_out,
                norm_out,  # add_in placeholder (HAS_ADD=False)
                m,
                self._signal_pad,
                local_src,
                N=n,
                ws=ws,
                NUM_SMS=grid_sms,
                HAS_RESIDUAL=True,
                HAS_ADD=False,
                RANK=self.my_pe,
                INKERNEL_BARRIER=inkernel,
                FOLD_COPYIN=fold,
                EXIT_BARRIER=exit_barrier,
                WORKGROUP_SYNC=self._workgroup_sync,
                num_warps=num_warps,
            )
        elif kern == "oneshot_wholerow_padded":
            _k.fused_ar_rmsnorm_oneshot_wholerow_padded_kernel[grid](
                x,
                norm_out,
                eps,
                weight,
                self.my_pe,
                input_bases,
                residual,
                residual_out,
                norm_out,  # add_in placeholder (HAS_ADD=False)
                m,
                n,
                self._signal_pad,
                local_src,
                BLOCK_N=self._oneshot_padded_block_n,
                ws=ws,
                NUM_SMS=grid_sms,
                HAS_RESIDUAL=True,
                HAS_ADD=False,
                RANK=self.my_pe,
                INKERNEL_BARRIER=inkernel,
                FOLD_COPYIN=fold,
                EXIT_BARRIER=exit_barrier,
                WORKGROUP_SYNC=self._workgroup_sync,
                num_warps=num_warps,
            )
        else:  # oneshot_blocked
            _k.fused_ar_rmsnorm_oneshot_blocked_kernel[grid](
                x,
                norm_out,
                self._oneshot_scratch,
                eps,
                weight,
                self.my_pe,
                input_bases,
                residual,
                residual_out,
                norm_out,  # add_in placeholder (HAS_ADD=False)
                m,
                n,
                self._signal_pad,
                local_src,
                BLOCK_N=self._oneshot_block_n,
                ws=ws,
                NUM_SMS=grid_sms,
                HAS_RESIDUAL=True,
                HAS_ADD=False,
                RANK=self.my_pe,
                INKERNEL_BARRIER=inkernel,
                FOLD_COPYIN=fold,
                EXIT_BARRIER=exit_barrier,
                WORKGROUP_SYNC=self._workgroup_sync,
                num_warps=num_warps,
            )
        if not inkernel:
            self._barrier()  # trailing

    def _should_borrow_twoshot_output(
        self,
        use_oneshot: bool,
        norm_out: torch.Tensor | None,
        residual_out: torch.Tensor | None,
    ) -> bool:
        return (
            self._borrow_twoshot_output
            and not use_oneshot
            and norm_out is None
            and residual_out is None
            and not torch.cuda.is_current_stream_capturing()
        )

    def fused(
        self,
        input_tensor: torch.Tensor,
        residual: torch.Tensor,
        weight: torch.Tensor,
        eps: float,
        norm_out: torch.Tensor | None = None,
        residual_out: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        assert input_tensor.dtype == self.dtype
        assert input_tensor.dim() == 2 and input_tensor.shape == residual.shape
        assert input_tensor.shape[1] == self.hidden_dim
        assert weight.shape == (self.hidden_dim,)
        assert input_tensor.is_contiguous() and residual.is_contiguous()

        m = input_tensor.shape[0]
        n = self.hidden_dim
        ws = self.world_size
        assert m <= self.max_token_num

        # Dispatch is resolved before output allocation because the eager
        # two-shot borrowed-output path returns its symmetric destination views
        # directly and therefore must not allocate throw-away local outputs.
        use_oneshot = (not self._is_twoshot) or (
            self._oneshot_max_m > 0 and m <= self._oneshot_max_m
        )
        borrow_twoshot_output = self._should_borrow_twoshot_output(
            use_oneshot, norm_out, residual_out
        )

        if (
            not borrow_twoshot_output
            and norm_out is None
            and residual_out is None
            and self._output_ring_size
            and m <= self._output_ring_max_m
        ):
            # Capture freezes each selected view's pointer. GPT-OSS executes 72
            # unconditional fused sites per forward and configures 72 slots, so
            # warmup/capture and every eager small-M forward return this host
            # phase to zero before another graph or forward can use the ring.
            output_slot = self._output_ring_index
            self._output_ring_index = (output_slot + 1) % self._output_ring_size
            norm_out = self._norm_output_ring[output_slot, :m]
            residual_out = self._residual_output_ring[output_slot, :m]
        else:
            if not borrow_twoshot_output:
                if norm_out is None:
                    norm_out = torch.empty_like(input_tensor)
                if residual_out is None:
                    residual_out = torch.empty_like(residual)

        # Fold the input copy-in into the one-shot
        # kernel (phase-0 write ordered by the in-kernel leading barrier). Requires
        # the in-kernel barrier; two-shot keeps the explicit copy_.
        fold = self._fold_copyin and self._inkernel and use_oneshot

        use_input_site = (
            use_oneshot
            and self._input_site_ring_size > 0
            and m <= self._input_site_max_m
        )
        if use_input_site:
            input_slot = self._input_site_ring_index
            self._input_site_ring_index = (input_slot + 1) % self._input_site_ring_size
            row_start = input_slot * self._input_site_max_m
            x = self._input_site_tensor[row_start : row_start + m]
            input_bases = self._input_site_bases
        else:
            input_slot = self._input_ring_index
            x = self._input_ring[input_slot][:m]
            input_bases = self._input_bases_ring[input_slot]
            self._input_ring_index = (input_slot + 1) % len(self._input_ring)
        if not fold:
            x.copy_(input_tensor)

        if use_oneshot:
            self._run_oneshot(
                x,
                input_bases,
                input_tensor,
                residual,
                weight,
                eps,
                m,
                n,
                ws,
                norm_out,
                residual_out,
                fold,
                not (self._double_buffer_input or use_input_site),
            )
            return norm_out, residual_out

        # Two-shot push (ws>=4, large M). In-kernel barriers are re-enabled here
        # ONLY under lever B (fixed grid): the in-kernel barrier's cost scales with
        # grid width, and two-shot's legacy grid is large (up to num_cus), so with
        # the M-dependent grid folding it in was a net loss (measured reclaim
        # -0.005..-0.048 ms at M=512..1024). The fixed grid (`_barrier_grid`) caps
        # the participant count so the barrier is cheap + constant → two-shot in-
        # kernel becomes viable. Falls back to separate barriers when barrier_grid=0.
        work_rows = triton.cdiv(m, ws)
        inkernel = self._inkernel and self._barrier_grid > 0
        grid_sms = self._grid_width("twoshot_blocked", ws, work_rows, inkernel)
        num_warps = _p.recommended_num_warps("twoshot_blocked")
        grid = (grid_sms,)
        if borrow_twoshot_output:
            output_slot = self._twoshot_output_index
            self._twoshot_output_index = (output_slot + 1) % len(self._twoshot_y_ring)
            y = self._twoshot_y_ring[output_slot][:m]
            res_out = self._twoshot_residual_ring[output_slot][:m]
            output_bases = self._twoshot_output_bases_ring[output_slot]
            residual_out_bases = self._twoshot_residual_bases_ring[output_slot]
            norm_out = y
            residual_out = res_out
        else:
            y = self._y[:m]
            res_out = self._residual_out[:m]
            output_bases = self._output_bases
            residual_out_bases = self._residual_out_bases
        if not inkernel:
            self._barrier()  # leading: peers' copy-in visible before any pull
        _k.fused_ar_rmsnorm_twoshot_blocked_kernel[grid](
            x,
            y,
            self._scratch,
            eps,
            weight,
            self.my_pe,
            input_bases,
            output_bases,
            residual_out_bases,
            residual,
            res_out,
            y,  # add_in placeholder (HAS_ADD=False)
            m,
            n,
            self._signal_pad,
            BLOCK_N=self._twoshot_block_n,
            ws=ws,
            NUM_SMS=grid_sms,
            HAS_RESIDUAL=True,
            HAS_ADD=False,
            RANK=self.my_pe,
            INKERNEL_BARRIER=inkernel,
            WORKGROUP_SYNC=self._workgroup_sync,
            num_warps=num_warps,
        )
        if not inkernel:
            self._barrier()  # trailing: peers' pushes into our output/residual_out visible
        if not borrow_twoshot_output:
            norm_out.copy_(y)
            residual_out.copy_(res_out)
        return norm_out, residual_out


def create_triton_shmem_ar_rmsnorm_state(
    group: dist.ProcessGroup,
    rank_in_group: int,
    max_token_num: int,
    hidden_dim: int,
    dtype: torch.dtype = torch.bfloat16,
    device: torch.device | None = None,
) -> "TritonShmemAllReduceResidualRMSNorm | None":
    """Create a triton_shmem (symm_mem) fused AR+RMSNorm state, or ``None``.

    Returns ``None`` (so the caller can fall back) when the backend can't run on
    this platform or state construction fails.
    """
    if not is_available():
        return None
    resolved_device = device or torch.device(f"cuda:{torch.cuda.current_device()}")
    if not _configuration_is_eligible(
        group=group,
        max_token_num=max_token_num,
        hidden_dim=hidden_dim,
        dtype=dtype,
        device=resolved_device,
    ):
        return None
    try:
        return TritonShmemAllReduceResidualRMSNorm(
            group=group,
            rank_in_group=rank_in_group,
            max_token_num=max_token_num,
            hidden_dim=hidden_dim,
            dtype=dtype,
            device=resolved_device,
        )
    except Exception as exc:  # noqa: BLE001 - decline rather than crash forward
        logger.warning("triton_shmem AR+RMSNorm state creation failed: %s", exc)
        return None


def triton_shmem_can_run(
    state: "TritonShmemAllReduceResidualRMSNorm",
    input_tensor: torch.Tensor,
) -> bool:
    """Decline captured calls without persistent profile-owned outputs.

    An input-site ring is optional: states without one retain the generic
    one-shot exit barrier, so their single symmetric input remains safe.
    """
    if not torch.cuda.is_current_stream_capturing():
        return True
    eligible = (
        state._output_ring_size > 0
        and input_tensor.shape[0] <= state._output_ring_max_m
    )
    if not eligible and not getattr(state, "_captured_output_decline_logged", False):
        logger.info(
            "triton_shmem captured call declined: M=%d persistent_output_max_m=%d; "
            "capturing complete ordinary fallback",
            input_tensor.shape[0],
            state._output_ring_max_m,
        )
        state._captured_output_decline_logged = True
    return eligible


def triton_shmem_allreduce_residual_rmsnorm(
    state: "TritonShmemAllReduceResidualRMSNorm",
    input_tensor: torch.Tensor,
    residual: torch.Tensor,
    weight: torch.Tensor,
    eps: float = 1e-6,
    norm_out: torch.Tensor | None = None,
    residual_out: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    m, n = input_tensor.shape
    use_oneshot = (not state._is_twoshot) or (
        state._oneshot_max_m > 0 and m <= state._oneshot_max_m
    )
    use_input_site = (
        use_oneshot and state._input_site_ring_size > 0 and m <= state._input_site_max_m
    )
    path = state._oneshot_kernel_for_m(m) if use_oneshot else "twoshot_blocked"
    borrow_twoshot_output = state._should_borrow_twoshot_output(
        use_oneshot, norm_out, residual_out
    )
    with kernel_scope(
        "communication",
        "allreduce_residual_rmsnorm",
        input_tensor.dtype,
        kernel_name=path,
        M=m,
        N=n,
        world_size=state.world_size,
        fold_copyin=int(state._fold_copyin and state._inkernel and use_oneshot),
        input_ring=len(state._input_ring),
        input_site_ring=state._input_site_ring_size,
        input_site_slot=(state._input_site_ring_index if use_input_site else -1),
        borrow_twoshot_output=int(borrow_twoshot_output),
        twoshot_output_slot=(
            state._twoshot_output_index if borrow_twoshot_output else -1
        ),
        exit_barrier=int(
            not (
                (state._double_buffer_input or use_input_site)
                and state._inkernel
                and use_oneshot
            )
        ),
    ):
        return state.fused(
            input_tensor=input_tensor,
            residual=residual,
            weight=weight,
            eps=eps,
            norm_out=norm_out,
            residual_out=residual_out,
        )
