"""Grid and cross-rank synchronization for tensor Gluon kernels."""

import triton.experimental.gluon as g
from lib.gemm.rocm.intrinsics import (
    BufferResource,
    _native_call,
    amdgcn_ballot,
    amdgcn_s_waitcnt,
)
from lib.tal.tensor_ops import atomic_add, first, load_words, wave_id
from triton.experimental.gluon import language as l


@g.jit
def agent_fence_release():
    """Order earlier memory accesses before a following publication on this GPU."""
    _native_call("fence.release.agent", "void", (), (), False)


@g.jit
def agent_fence_acquire():
    """Order later memory accesses after an observed publication on this GPU."""
    _native_call("fence.acquire.agent", "void", (), (), False)


@g.jit
def system_fence_release():
    """Order earlier memory accesses before a following publication to peer GPUs."""
    _native_call("fence.release.system", "void", (), (), False)


@g.jit
def system_fence_acquire():
    """Order later payload accesses after observing a peer's release publication."""
    _native_call("fence.acquire.system", "void", (), (), False)


@g.jit
def wave_barrier():
    """Synchronize participating lanes within one wave, not other waves or CTAs.

    This is the LLVM wave barrier; payload memory ordering is handled separately.
    """
    _native_call("llvm.amdgcn.wave.barrier", "void", (), (), False)


@g.jit
def buffer_wbl2_sc0_sc1():
    """Issue an L2 writeback with SC0/SC1 cache controls.

    This emits ``buffer_wbl2 sc0 sc1`` without an accompanying wait or collective
    barrier; callers supply completion and synchronization as required.
    """
    _native_call("buffer.wbl2.sc0.sc1", "void", (), (), False)


@g.jit
def compiler_memory_barrier():
    """Prevent compiler movement of memory accesses across this point.

    Emits an empty side-effecting assembly block with a memory clobber, without
    a hardware fence or wait instruction.
    """
    _native_call("compiler.memory.barrier", "void", (), (), False)


@g.jit
def complete_scoped_vmem():
    """Wait for this wave's VMEM counter to reach zero, with compiler barriers.

    Only VMEM is explicitly drained; this does not wait for other waves or replace
    an acquire/release fence when publishing or consuming a payload.
    """
    compiler_memory_barrier()
    amdgcn_s_waitcnt(0)
    compiler_memory_barrier()


@g.jit
def tensor_grid_sync(
    Workspace: l.constexpr,
    workspace,
    sm_idx,
    kNumSMs: l.constexpr,
    kGridSyncIndex: l.constexpr,
    kAcquirePayload: l.constexpr,
    kSystemScope: l.constexpr,
):
    """Join all resident CTAs using a reusable counter with a toggled epoch bit.

    Wave zero publishes one arrival per CTA after releasing earlier writes.
    kSystemScope selects agent or system ordering; kAcquirePayload requests
    matching acquire ordering before the CTA resumes. All CTAs must call the
    same counter slot and remain resident until the barrier completes.
    """
    l.barrier()
    if wave_id() == 0:
        lane = l.arange(0, 64, layout=l.BlockedLayout([1], [64], [l.num_warps()], [0]))
        count_offset = Workspace.GridSyncBarrierOffset() + kGridSyncIndex * 4
        first_sm = (sm_idx.to(l.uint32) - 1) >> 31
        delta = 1 + first_sm * (0x80000000 - kNumSMs)
        if kSystemScope:
            system_fence_release()
        else:
            agent_fence_release()
        old = first(
            atomic_add(
                workspace.br_,
                count_offset,
                0,
                delta,
                (
                    BufferResource.kAtomicScopeSystem
                    if kSystemScope
                    else BufferResource.kAtomicScopeAgent
                ),
                lane == 0,
            )
        ).to(l.uint32)
        pending = l.full((), True, l.int1)
        while pending:
            observed = BufferResource.LoadU32(
                workspace.br_,
                count_offset,
                0,
                BufferResource.kSC0Bit | BufferResource.kSC1Bit,
            )
            pending = ((observed ^ old) & 0x80000000) == 0
            if pending:
                _native_call("s.sleep.1", "void", (), (), False)
        if kAcquirePayload:
            if kSystemScope:
                system_fence_acquire()
            else:
                agent_fence_acquire()
        else:
            compiler_memory_barrier()
    l.barrier()


@g.jit
def wait_tensor_signal(workspace, offset, expected, mask, epoch: l.constexpr):
    """Poll active lanes in wave zero until their signal reaches the target.

    With epoch enabled, use signed modular subtraction to accept later epochs;
    otherwise require equality. This polls coherent signals but does not acquire
    payload writes: the caller must issue the appropriate acquire fence.
    """
    # The caller assigns this polling tensor to wave zero. Its reduction must
    # stay within that wave, since the other waves can have independent work.
    l.static_assert(offset.numel <= 64)
    pending = mask
    while first(amdgcn_ballot(pending)) != 0:
        compiler_memory_barrier()
        observed = load_words(
            workspace.br_,
            offset,
            0,
            1,
            BufferResource.kSC0Bit | BufferResource.kSC1Bit,
            pending,
        )
        if epoch:
            reached = (observed - expected).to(l.int32) >= 0
        else:
            reached = observed == expected
        pending = pending & ~reached


@g.jit
def wait_xgpu_signal_relaxed(workspace, signal_offset, target):
    """Wait for exact signed-int32 equality using SC0/SC1 buffer loads.

    Each poll includes a compiler memory barrier. There is no payload acquire
    or timeout; the producer must not skip the target value.
    """
    pending = l.full((), True, l.int1)
    while pending:
        compiler_memory_barrier()
        observed = BufferResource.LoadU32(
            workspace.br_,
            signal_offset,
            0,
            BufferResource.kSC0Bit | BufferResource.kSC1Bit,
        ).to(l.int32)
        pending = observed != target
