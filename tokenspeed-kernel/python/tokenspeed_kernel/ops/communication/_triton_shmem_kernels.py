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

"""Embedded fused all-reduce (+ add + residual) + RMSNorm Triton kernels.

The TokenSpeed implementation uses PyTorch symmetric memory instead of a
rocSHMEM heap. The two-shot kernel therefore takes three per-tensor peer-pointer
tables (input / output / residual_out), while the one-shot kernels only
translate the input pointer.

``triton``/``tl`` are imported from ``tokenspeed_kernel._triton`` (the vendored
``tokenspeed_triton`` distribution) so these run under the same Triton as the
rest of the package. There are **no** ``rocshmem4py`` / ``triton_shmem`` imports.

Host-side barriers (rocSHMEM ``barrier_all``) are replaced by the symm_mem
signal-pad CAS barrier (``symm_mem_barrier`` from ``triton.py``), issued from a
dedicated single-block barrier kernel by the caller (see ``triton_shmem.py``).
"""

from __future__ import annotations

from tokenspeed_kernel._triton import tl, triton
from tokenspeed_kernel.ops.communication.triton import symm_mem_barrier


# ---------------------------------------------------------------------------
# Device-side symmetric pointer translation. Given a per-tensor peer-pointer
# table ``bases`` (rocSHMEM ``heap_bases`` OR symm_mem ``buffer_ptrs_dev`` -- the math
# is identical; see the canonical backend design), translate ``local_ptr`` from my rank's
# address space into ``peer``'s.
# ---------------------------------------------------------------------------
@triton.jit
def symmetric_ptr(local_ptr, my_pe, peer, bases):
    local_int = tl.cast(local_ptr, tl.uint64)
    my_base = tl.load(bases + my_pe)
    peer_base = tl.load(bases + peer)
    offset = local_int - my_base
    peer_byte = tl.cast(peer_base, tl.pointer_type(tl.int8))
    return tl.cast(peer_byte + offset, local_ptr.dtype)


# ---------------------------------------------------------------------------
# Whole-grid symm_mem barrier kernel. Each of the ``grid_sms`` blocks performs a
# signal-pad CAS barrier against the same block index on every peer, so the
# kernel-launch boundary is a global barrier (all peers reached it). Requires
# ``grid_sms <= num_cus`` so every block is resident (the spin-wait would
# otherwise deadlock); the caller guarantees this via ``recommended_grid``.
# ---------------------------------------------------------------------------
@triton.jit
def symm_grid_barrier_kernel(
    signal_pad_ptrs_dev,
    RANK: tl.constexpr,
    WORLD_SIZE: tl.constexpr,
):
    symm_mem_barrier(signal_pad_ptrs_dev, tl.program_id(0), RANK, WORLD_SIZE)


# ===========================================================================
# Kernels. HAS_ADD / HAS_RESIDUAL are constexpr flags; the fused ordering is:
#   x = all_reduce_sum(input); if HAS_ADD: x += add_in;
#   if HAS_RESIDUAL: x += residual; residual_out = x
#   norm_out = x * rsqrt(mean(x**2) + eps) * gamma
# ===========================================================================
@triton.jit
def fused_ar_rmsnorm_oneshot_wholerow_kernel(
    input,
    output,
    epsilon,
    gamma,
    my_pe,
    heap_bases,
    residual,
    residual_out,
    add_in,
    M,
    signal_pad,
    local_src,
    N: tl.constexpr,
    ws: tl.constexpr,
    NUM_SMS: tl.constexpr,
    HAS_RESIDUAL: tl.constexpr,
    HAS_ADD: tl.constexpr,
    RANK: tl.constexpr = 0,
    INKERNEL_BARRIER: tl.constexpr = False,
    FOLD_COPYIN: tl.constexpr = False,
    EXIT_BARRIER: tl.constexpr = True,
    WORKGROUP_SYNC: tl.constexpr = True,
):
    """One-shot pull, whole-row. Every PE reduces all rows by pulling each peer's
    ``input`` and writes only its own local ``output`` / ``residual_out`` (no
    push). ``N`` must be a power of two. Only ``input`` is symmetric, so a single
    peer-pointer table (``heap_bases``) is used.

    ``INKERNEL_BARRIER`` folds the leading + trailing signal-pad barriers into the
    kernel (per-block at index ``pid``), removing two separate barrier-kernel
    launches — the dominant small-M/decode overhead. Safe here
    because this kernel is pull-only (reads peers, writes local): entry acquire
    makes peers' copy-in visible before the pull; exit release lets peers know we
    finished reading our symmetric input before the next call overwrites it. Same
    pattern as the native ``amd_allreduce_residual_rmsnorm_kernel``.

    ``FOLD_COPYIN`` (requires ``INKERNEL_BARRIER``) writes this rank's local input
    (``local_src``) into its symmetric ``input`` buffer in a phase-0 pass *before*
    the leading barrier, replacing the separate ``copy_`` launch. The leading
    barrier then orders the write before any peer pull, so
    the same barrier that already existed does double duty. ``EXIT_BARRIER`` may
    be disabled only when a qualified symmetric input ring delays slot reuse."""
    tl.static_assert(
        (N & (N - 1)) == 0,
        "fused_ar_rmsnorm_oneshot_wholerow_kernel requires N to be a power of two; "
        "use fused_ar_rmsnorm_oneshot_blocked_kernel for arbitrary N",
    )
    pid = tl.program_id(0)
    offsets_n = tl.max_contiguous(tl.multiple_of(tl.arange(0, N), N), N)

    if FOLD_COPYIN:
        for row_id in range(pid, M, NUM_SMS):
            offsets_io = offsets_n + (N * row_id)
            tl.store(input + offsets_io, tl.load(local_src + offsets_io))

    if INKERNEL_BARRIER:
        # The signal CAS is issued by a scalar lane. Synchronize all wavefronts
        # before its system-scope release so every phase-0 store is ordered, and
        # after its acquire before any wavefront starts pulling peer data.
        if WORKGROUP_SYNC:
            tl.debug_barrier()
        symm_mem_barrier(signal_pad, pid, RANK, ws)
        if WORKGROUP_SYNC:
            tl.debug_barrier()

    gamma_row = tl.load(gamma + offsets_n).to(tl.float32)

    for row_id in range(pid, M, NUM_SMS):
        offsets_io = offsets_n + (N * row_id)
        acc = tl.zeros((N,), tl.float32)
        for peer in tl.static_range(0, ws):
            peer_ptr = symmetric_ptr(input, my_pe, peer, heap_bases)
            acc += tl.load(peer_ptr + offsets_io).to(tl.float32)

        if HAS_ADD:
            acc += tl.load(add_in + offsets_io).to(tl.float32)
        if HAS_RESIDUAL:
            acc += tl.load(residual + offsets_io).to(tl.float32)
            tl.store(residual_out + offsets_io, acc.to(residual_out.dtype.element_ty))

        sum_squares = tl.sum(acc * acc)
        norm_factor = tl.rsqrt((sum_squares / N) + epsilon)
        rms_norm = (acc * norm_factor * gamma_row).to(output.dtype.element_ty)
        tl.store(output + offsets_io, rms_norm)

    if INKERNEL_BARRIER and EXIT_BARRIER:
        # No peer may reuse its persistent input until every wavefront in this
        # workgroup has finished the pull.
        if WORKGROUP_SYNC:
            tl.debug_barrier()
        symm_mem_barrier(signal_pad, pid, RANK, ws)
        if WORKGROUP_SYNC:
            tl.debug_barrier()


@triton.jit
def fused_ar_rmsnorm_twoshot_blocked_kernel(
    input,
    output,
    scratch,
    epsilon,
    gamma,
    my_pe,
    input_bases,
    output_bases,
    residual_out_bases,
    residual,
    residual_out,
    add_in,
    M,
    N,
    signal_pad,
    BLOCK_N: tl.constexpr,
    ws: tl.constexpr,
    NUM_SMS: tl.constexpr,
    HAS_RESIDUAL: tl.constexpr,
    HAS_ADD: tl.constexpr,
    RANK: tl.constexpr = 0,
    INKERNEL_BARRIER: tl.constexpr = False,
    WORKGROUP_SYNC: tl.constexpr = True,
):
    """Two-pass, N-blocked. Contiguous row ownership: PE ``my_pe`` owns rows
    ``[my_pe*M_shard, (my_pe+1)*M_shard)``, reduces them by pulling every peer's
    ``input``, and pushes the pre-norm ``residual_out`` and normalized ``output``
    into every peer. ``input``, ``output`` and ``residual_out`` are all symmetric
    and each is translated with **its own** peer-pointer table (the sole device
    change vs. the rocSHMEM upstream, which shared one ``heap_bases``).

    ``INKERNEL_BARRIER`` folds the leading + trailing signal-pad barriers into the
    kernel (per-block at ``pid``), removing two separate barrier-kernel launches.
    The leading barrier makes peers' copy-in visible before the pull; the trailing
    barrier makes peers' pushes into our ``output``/``residual_out`` visible before
    the caller's copy-out. Two-shot serves M>oneshot_max_m (eager prefill only; not
    inside a decode graph), where TP ranks always share M -- so the barrier
    participant set (``NUM_SMS`` blocks) matches across ranks. Same M-divergence
    caveat as the one-shot path."""
    tl.static_assert(
        (BLOCK_N & (BLOCK_N - 1)) == 0,
        "fused_ar_rmsnorm_twoshot_blocked_kernel requires BLOCK_N to be a power of two",
    )
    pid = tl.program_id(0)
    if INKERNEL_BARRIER:
        if WORKGROUP_SYNC:
            tl.debug_barrier()
        symm_mem_barrier(signal_pad, pid, RANK, ws)
        if WORKGROUP_SYNC:
            tl.debug_barrier()
    M_shard = tl.cdiv(M, ws)
    shard_row_offset = M_shard * my_pe
    n_blocks = tl.cdiv(N, BLOCK_N)
    col = tl.arange(0, BLOCK_N)

    for shard_row_id in range(pid, M_shard, NUM_SMS):
        global_row = shard_row_id + shard_row_offset
        if global_row < M:
            row_io_off = global_row * N  # into the symmetric (M, N) tensors
            scratch_off = shard_row_id * N  # into the local (M_shard, N) scratch

            sum_squares = tl.zeros((), tl.float32)
            for blk in range(0, n_blocks):
                cols = blk * BLOCK_N + col
                mask = cols < N
                offs = row_io_off + cols
                acc = tl.zeros((BLOCK_N,), tl.float32)
                for peer in tl.static_range(0, ws):
                    peer_ptr = symmetric_ptr(input, my_pe, peer, input_bases)
                    acc += tl.load(peer_ptr + offs, mask=mask, other=0.0).to(tl.float32)
                if HAS_ADD:
                    acc += tl.load(add_in + offs, mask=mask, other=0.0).to(tl.float32)
                if HAS_RESIDUAL:
                    acc += tl.load(residual + offs, mask=mask, other=0.0).to(tl.float32)
                    res = acc.to(residual_out.dtype.element_ty)
                    tl.store(residual_out + offs, res, mask=mask)
                    for peer in tl.static_range(0, ws):
                        if peer != my_pe:
                            peer_ptr = symmetric_ptr(
                                residual_out, my_pe, peer, residual_out_bases
                            )
                            tl.store(peer_ptr + offs, res, mask=mask)
                sum_squares += tl.sum(acc * acc, axis=0)
                tl.store(scratch + scratch_off + cols, acc, mask=mask)

            norm_factor = tl.rsqrt((sum_squares / N) + epsilon)

            for blk in range(0, n_blocks):
                cols = blk * BLOCK_N + col
                mask = cols < N
                offs = row_io_off + cols
                reduced = tl.load(scratch + scratch_off + cols, mask=mask, other=0.0)
                block_g = tl.load(gamma + cols, mask=mask, other=0.0).to(tl.float32)
                rms_norm = (reduced * norm_factor * block_g).to(output.dtype.element_ty)
                tl.store(output + offs, rms_norm, mask=mask)
                for peer in tl.static_range(0, ws):
                    if peer != my_pe:
                        peer_ptr = symmetric_ptr(output, my_pe, peer, output_bases)
                        tl.store(peer_ptr + offs, rms_norm, mask=mask)

    if INKERNEL_BARRIER:
        if WORKGROUP_SYNC:
            tl.debug_barrier()
        symm_mem_barrier(signal_pad, pid, RANK, ws)
        if WORKGROUP_SYNC:
            tl.debug_barrier()


@triton.jit(do_not_specialize=["M"])
def fused_ar_rmsnorm_oneshot_wholerow_padded_kernel(
    input,
    output,
    epsilon,
    gamma,
    my_pe,
    heap_bases,
    residual,
    residual_out,
    add_in,
    M,
    N: tl.constexpr,
    signal_pad,
    local_src,
    BLOCK_N: tl.constexpr,
    ws: tl.constexpr,
    NUM_SMS: tl.constexpr,
    HAS_RESIDUAL: tl.constexpr,
    HAS_ADD: tl.constexpr,
    RANK: tl.constexpr = 0,
    INKERNEL_BARRIER: tl.constexpr = False,
    FOLD_COPYIN: tl.constexpr = False,
    EXIT_BARRIER: tl.constexpr = True,
    WORKGROUP_SYNC: tl.constexpr = True,
):
    """Scratch-free whole-row one-shot for arbitrary hidden widths.

    ``BLOCK_N`` is the next power of two at least as large as ``N``. Masked
    lanes contribute zero to the RMS reduction, matching Iris's fused decode
    structure while retaining the triton_shmem pointer/barrier substrate.
    """
    tl.static_assert(
        (BLOCK_N & (BLOCK_N - 1)) == 0,
        "BLOCK_N must be a power of two",
    )
    pid = tl.program_id(0)
    cols = tl.max_contiguous(
        tl.multiple_of(tl.arange(0, BLOCK_N), BLOCK_N),
        BLOCK_N,
    )
    mask = cols < N

    if FOLD_COPYIN:
        for row_id in range(pid, M, NUM_SMS):
            offsets = row_id * N + cols
            tl.store(
                input + offsets,
                tl.load(local_src + offsets, mask=mask, other=0.0),
                mask=mask,
            )

    if INKERNEL_BARRIER:
        if WORKGROUP_SYNC:
            tl.debug_barrier()
        symm_mem_barrier(signal_pad, pid, RANK, ws)
        if WORKGROUP_SYNC:
            tl.debug_barrier()

    gamma_row = tl.load(gamma + cols, mask=mask, other=0.0).to(tl.float32)
    for row_id in range(pid, M, NUM_SMS):
        offsets = row_id * N + cols
        acc = tl.zeros((BLOCK_N,), tl.float32)
        for peer in tl.static_range(0, ws):
            peer_ptr = symmetric_ptr(input, my_pe, peer, heap_bases)
            acc += tl.load(
                peer_ptr + offsets,
                mask=mask,
                other=0.0,
            ).to(tl.float32)

        if HAS_ADD:
            acc += tl.load(add_in + offsets, mask=mask, other=0.0).to(tl.float32)
        if HAS_RESIDUAL:
            acc += tl.load(residual + offsets, mask=mask, other=0.0).to(tl.float32)
            tl.store(
                residual_out + offsets,
                acc.to(residual_out.dtype.element_ty),
                mask=mask,
            )

        sum_squares = tl.sum(acc * acc)
        norm_factor = tl.rsqrt((sum_squares / N) + epsilon)
        rms_norm = (acc * norm_factor * gamma_row).to(output.dtype.element_ty)
        tl.store(output + offsets, rms_norm, mask=mask)

    if INKERNEL_BARRIER and EXIT_BARRIER:
        if WORKGROUP_SYNC:
            tl.debug_barrier()
        symm_mem_barrier(signal_pad, pid, RANK, ws)
        if WORKGROUP_SYNC:
            tl.debug_barrier()


@triton.jit
def fused_ar_rmsnorm_oneshot_blocked_kernel(
    input,
    output,
    scratch,
    epsilon,
    gamma,
    my_pe,
    heap_bases,
    residual,
    residual_out,
    add_in,
    M,
    N,
    signal_pad,
    local_src,
    BLOCK_N: tl.constexpr,
    ws: tl.constexpr,
    NUM_SMS: tl.constexpr,
    HAS_RESIDUAL: tl.constexpr,
    HAS_ADD: tl.constexpr,
    RANK: tl.constexpr = 0,
    INKERNEL_BARRIER: tl.constexpr = False,
    FOLD_COPYIN: tl.constexpr = False,
    EXIT_BARRIER: tl.constexpr = True,
    WORKGROUP_SYNC: tl.constexpr = True,
):
    """One-shot pull, two-pass, N-blocked (arbitrary ``N``). No row ownership, no
    peer push. Every PE reduces all rows by pulling each peer's ``input`` and
    writes the full result to its own local ``output`` / ``residual_out``.
    ``scratch`` is a local fp32 ``(NUM_SMS, N)`` buffer (one slot per program).
    Only ``input`` is symmetric, so a single peer-pointer table is used.

    ``INKERNEL_BARRIER`` folds the leading + trailing signal-pad barriers into the
    kernel (per-block at ``pid``), removing two separate barrier-kernel launches.
    Pull-only ⇒ same proven pattern as the native
    ``amd_allreduce_residual_rmsnorm_kernel`` (no cross-thread push to order).

    ``FOLD_COPYIN`` (requires ``INKERNEL_BARRIER``) writes this rank's local input
    (``local_src``) into its symmetric ``input`` buffer in a phase-0 pass *before*
    the leading barrier, replacing the separate ``copy_`` launch.
    ``EXIT_BARRIER`` may be disabled only with qualified delayed slot reuse."""
    tl.static_assert(
        (BLOCK_N & (BLOCK_N - 1)) == 0,
        "fused_ar_rmsnorm_oneshot_blocked_kernel requires BLOCK_N to be a power of two",
    )
    pid = tl.program_id(0)
    n_blocks = tl.cdiv(N, BLOCK_N)
    col = tl.arange(0, BLOCK_N)
    scratch_off = pid * N

    if FOLD_COPYIN:
        for row_id in range(pid, M, NUM_SMS):
            row_io_off = row_id * N
            for blk in range(0, n_blocks):
                cols = blk * BLOCK_N + col
                mask = cols < N
                offs = row_io_off + cols
                tl.store(
                    input + offs,
                    tl.load(local_src + offs, mask=mask, other=0.0),
                    mask=mask,
                )

    if INKERNEL_BARRIER:
        # See the whole-row variant: the scalar system release/acquire must be
        # bracketed by workgroup barriers so it represents every wavefront's
        # global stores/loads rather than only the issuing lane's operations.
        if WORKGROUP_SYNC:
            tl.debug_barrier()
        symm_mem_barrier(signal_pad, pid, RANK, ws)
        if WORKGROUP_SYNC:
            tl.debug_barrier()

    for row_id in range(pid, M, NUM_SMS):
        row_io_off = row_id * N

        sum_squares = tl.zeros((), tl.float32)
        for blk in range(0, n_blocks):
            cols = blk * BLOCK_N + col
            mask = cols < N
            offs = row_io_off + cols
            acc = tl.zeros((BLOCK_N,), tl.float32)
            for peer in tl.static_range(0, ws):
                peer_ptr = symmetric_ptr(input, my_pe, peer, heap_bases)
                acc += tl.load(peer_ptr + offs, mask=mask, other=0.0).to(tl.float32)
            if HAS_ADD:
                acc += tl.load(add_in + offs, mask=mask, other=0.0).to(tl.float32)
            if HAS_RESIDUAL:
                acc += tl.load(residual + offs, mask=mask, other=0.0).to(tl.float32)
                tl.store(
                    residual_out + offs,
                    acc.to(residual_out.dtype.element_ty),
                    mask=mask,
                )
            sum_squares += tl.sum(acc * acc, axis=0)
            tl.store(scratch + scratch_off + cols, acc, mask=mask)

        norm_factor = tl.rsqrt((sum_squares / N) + epsilon)

        for blk in range(0, n_blocks):
            cols = blk * BLOCK_N + col
            mask = cols < N
            offs = row_io_off + cols
            reduced = tl.load(scratch + scratch_off + cols, mask=mask, other=0.0)
            block_g = tl.load(gamma + cols, mask=mask, other=0.0).to(tl.float32)
            rms_norm = (reduced * norm_factor * block_g).to(output.dtype.element_ty)
            tl.store(output + offs, rms_norm, mask=mask)

    if INKERNEL_BARRIER and EXIT_BARRIER:
        if WORKGROUP_SYNC:
            tl.debug_barrier()
        symm_mem_barrier(signal_pad, pid, RANK, ws)
        if WORKGROUP_SYNC:
            tl.debug_barrier()
