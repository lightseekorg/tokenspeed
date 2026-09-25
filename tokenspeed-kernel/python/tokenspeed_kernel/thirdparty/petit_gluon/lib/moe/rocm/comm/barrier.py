"""Native grid and cross-rank barriers, called in a per-thread Gluon body."""

import triton.experimental.gluon as g
from lib.gemm.rocm.amd_intrinsics import (
    BufferResource,
    _native_call,
    amdgcn_s_waitcnt,
    amdgcn_shuffle,
)
from lib.tal.device import DeviceTemplate, device_method
from triton.experimental.gluon import language as l


@g.jit
def agent_fence_release():
    _native_call("fence.release.agent", "void", (), (), False)


@g.jit
def agent_fence_acquire():
    _native_call("fence.acquire.agent", "void", (), (), False)


@g.jit
def system_fence_release():
    _native_call("fence.release.system", "void", (), (), False)


@g.jit
def system_fence_acquire():
    _native_call("fence.acquire.system", "void", (), (), False)


@g.jit
def wave_barrier():
    _native_call("llvm.amdgcn.wave.barrier", "void", (), (), False)


@g.jit
def buffer_wbl2_sc0_sc1():
    _native_call("buffer.wbl2.sc0.sc1", "void", (), (), False)


@g.jit
def compiler_memory_barrier():
    _native_call("compiler.memory.barrier", "void", (), (), False)


@g.jit
def complete_scoped_vmem():
    compiler_memory_barrier()
    amdgcn_s_waitcnt(0)
    compiler_memory_barrier()


@g.jit
def store_xgpu_epoch_relaxed(workspace, signal_offset, epoch):
    BufferResource.StoreU32(
        workspace.br_,
        signal_offset,
        0,
        epoch,
        BufferResource.kSC0Bit | BufferResource.kSC1Bit,
    )


@g.jit
def store_xgpu_epoch_release(workspace, signal_offset, epoch):
    system_fence_release()
    store_xgpu_epoch_relaxed(workspace, signal_offset, epoch)


@g.jit
def wait_xgpu_signal(workspace, signal_offset, target, kProfile: l.constexpr = False):
    kNumTimeoutCycles: l.constexpr = 300 * 2000000000
    start_clock = l.full((), 0, l.uint64)
    if kProfile:
        start_clock = _native_call("llvm.readcyclecounter", "i64", (), (), False)
    while (
        BufferResource.LoadU32(
            workspace.br_, signal_offset, 0, BufferResource.kAtomicScopeSystem
        ).to(l.int32)
        != target
    ):
        if kProfile:
            if (
                _native_call("llvm.readcyclecounter", "i64", (), (), False)
                - start_clock
                >= kNumTimeoutCycles
            ):
                _native_call("llvm.debugtrap", "void", (), (), False)


@g.jit
def wait_xgpu_signal_relaxed(workspace, signal_offset, target):
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


@g.jit
def wait_xgpu_epoch_relaxed(workspace, signal_offset, expected):
    pending = l.full((), True, l.int1)
    while pending:
        compiler_memory_barrier()
        observed = BufferResource.LoadU32(
            workspace.br_,
            signal_offset,
            0,
            BufferResource.kSC0Bit | BufferResource.kSC1Bit,
        )
        pending = (observed - expected).to(l.int32) < 0


@g.jit
def xgpu_epoch_reached(observed, expected):
    return (observed - expected).to(l.int32) >= 0


@g.jit
def wait_xgpu_epoch(workspace, signal_offset, expected, kProfile: l.constexpr = False):
    kNumTimeoutCycles: l.constexpr = 300 * 2000000000
    start_clock = l.full((), 0, l.uint64)
    if kProfile:
        start_clock = _native_call("llvm.readcyclecounter", "i64", (), (), False)
    pending = l.full((), True, l.int1)
    while pending:
        observed = BufferResource.LoadU32(
            workspace.br_,
            signal_offset,
            0,
            BufferResource.kSC0Bit | BufferResource.kSC1Bit,
        )
        pending = not xgpu_epoch_reached(observed, expected)
        if kProfile:
            if (
                pending
                and _native_call("llvm.readcyclecounter", "i64", (), (), False)
                - start_clock
                >= kNumTimeoutCycles
            ):
                _native_call("llvm.debugtrap", "void", (), (), False)


@g.jit
def grid_sync(
    Workspace: l.constexpr,
    workspace,
    sm_idx,
    thread_idx,
    sync_scope: l.constexpr,
    kNumSMs: l.constexpr,
    kGridSyncIndex: l.constexpr = 0,
    kAcquirePayload: l.constexpr = True,
    kSystemScope: l.constexpr = False,
):
    if kNumSMs == 1:
        sync_scope()
        return
    kFinishSumTag: l.constexpr = 0x80000000
    sync_scope()
    if thread_idx == 0:
        count_offset = Workspace.GridSyncBarrierOffset() + kGridSyncIndex * 4
        l.static_assert(kNumSMs <= 0x80000000)
        is_first_sm = (sm_idx.to(l.uint32) - 1) >> 31
        arrival_delta = 1 + is_first_sm * (kFinishSumTag - kNumSMs)
        if kSystemScope:
            system_fence_release()
        else:
            agent_fence_release()
        if kSystemScope:
            old_value = BufferResource.AtomicAddI32(
                workspace.br_,
                count_offset,
                0,
                arrival_delta.to(l.int32),
                BufferResource.kAtomicScopeSystem,
            ).to(l.uint32)
        else:
            old_value = BufferResource.AtomicAddI32(
                workspace.br_,
                count_offset,
                0,
                arrival_delta.to(l.int32),
                BufferResource.kAtomicScopeAgent,
            ).to(l.uint32)
        new_value = l.full((), 0, l.uint32)
        pending = l.full((), True, l.int1)
        while pending:
            new_value = BufferResource.LoadU32(
                workspace.br_,
                count_offset,
                0,
                BufferResource.kSC0Bit | BufferResource.kSC1Bit,
            )
            pending = ((new_value ^ old_value) & kFinishSumTag) == 0
            if pending:
                _native_call("s.sleep.1", "void", (), (), False)
        if kAcquirePayload:
            if kSystemScope:
                system_fence_acquire()
            else:
                agent_fence_acquire()
        else:
            compiler_memory_barrier()
    sync_scope()


@g.jit
def xgpu_barrier(
    Workspace: l.constexpr,
    workspace,
    sm_idx,
    thread_idx,
    sync_scope: l.constexpr,
    kNumRanks: l.constexpr,
    kNumSMs: l.constexpr,
    kNumThreads: l.constexpr,
    kGridSyncIndex: l.constexpr,
    kAcquireProloguePayload: l.constexpr = True,
    kAcquireEpiloguePayload: l.constexpr = True,
    kProfile: l.constexpr = False,
    sync_prologue=True,
    sync_epilogue=True,
):
    l.static_assert(kNumRanks <= kNumThreads)
    if sync_prologue:
        grid_sync(
            Workspace,
            workspace,
            sm_idx,
            thread_idx,
            sync_scope,
            kNumSMs,
            kGridSyncIndex,
            kAcquireProloguePayload,
        )
    if kNumRanks == 1:
        if sync_epilogue and not sync_prologue:
            grid_sync(
                Workspace,
                workspace,
                sm_idx,
                thread_idx,
                sync_scope,
                kNumSMs,
                kGridSyncIndex,
                kAcquireEpiloguePayload,
            )
        return
    if sm_idx == 0:
        if thread_idx < 64:
            counter_offset = Workspace.XGpuBarrierCounterOffset(workspace.rank_id_)
            status = l.full((), 0, l.uint32)
            if thread_idx == 0:
                status = (
                    BufferResource.LoadU32(
                        workspace.br_, counter_offset, 0, BufferResource.kNone
                    )
                    & 3
                )
            status = amdgcn_shuffle(status, 0)
            system_fence_release()
            signal_phase, signal_sign = status & 1, status >> 1
            signal_delta = l.where(signal_sign != 0, -1, 1)
            if thread_idx < kNumRanks:
                BufferResource.AtomicAddI32(
                    workspace.br_,
                    Workspace.XGpuBarrierSignalOffset(thread_idx, signal_phase),
                    0,
                    signal_delta,
                    BufferResource.kAtomicScopeSystem,
                )
            wave_barrier()
            amdgcn_s_waitcnt(0, -1, 0)
            if thread_idx == 0:
                BufferResource.StoreU32(
                    workspace.br_, counter_offset, 0, status + 1, BufferResource.kNone
                )
                target = l.where(signal_sign != 0, 0, kNumRanks)
                wait_xgpu_signal(
                    workspace,
                    Workspace.XGpuBarrierSignalOffset(workspace.rank_id_, signal_phase),
                    target,
                    kProfile,
                )
                system_fence_acquire()
        sync_scope()
    if sync_epilogue:
        grid_sync(
            Workspace,
            workspace,
            sm_idx,
            thread_idx,
            sync_scope,
            kNumSMs,
            kGridSyncIndex,
            kAcquireEpiloguePayload,
        )


class LegacyXGpuSync(DeviceTemplate):
    def __init__(self, Config, Workspace):
        self._key = (Config.cache_key, Workspace.cache_key)
        self.kNumSMs, self.kNumRanks, self.kThreads = (
            Config.kNumSMs,
            Config.kNumRanks,
            Config.kThreads,
        )
        self.Workspace = Workspace

    @device_method
    def Begin(
        self,
        workspace,
        sm_idx,
        thread_idx,
        sync_scope: l.constexpr,
        kPrologueGridSyncIndex: l.constexpr,
        kProfile: l.constexpr = False,
    ):
        grid_sync(
            self.Workspace,
            workspace,
            sm_idx,
            thread_idx,
            sync_scope,
            self.kNumSMs,
            kPrologueGridSyncIndex,
            False,
        )
        return ()

    @device_method
    def Finish(
        self,
        workspace,
        sm_idx,
        thread_idx,
        ticket,
        sync_scope: l.constexpr,
        kEpilogueGridSyncIndex: l.constexpr,
        kProfile: l.constexpr = False,
    ):
        xgpu_barrier(
            self.Workspace,
            workspace,
            sm_idx,
            thread_idx,
            sync_scope,
            self.kNumRanks,
            self.kNumSMs,
            self.kThreads,
            kEpilogueGridSyncIndex,
            False,
            True,
            kProfile,
            False,
            True,
        )


class EpochXGpuSync(LegacyXGpuSync):
    @device_method
    def Begin(
        self,
        workspace,
        sm_idx,
        thread_idx,
        sync_scope: l.constexpr,
        kPrologueGridSyncIndex: l.constexpr,
        kProfile: l.constexpr = False,
    ):
        next_epoch = l.full((), 0, l.uint32)
        if thread_idx == 0:
            next_epoch = 1 + BufferResource.LoadU32(
                workspace.br_,
                self.Workspace.XGpuEpochCounterOffset(workspace.rank_id_),
                0,
                BufferResource.kNone,
            )
        grid_sync(
            self.Workspace,
            workspace,
            sm_idx,
            thread_idx,
            sync_scope,
            self.kNumSMs,
            kPrologueGridSyncIndex,
            True,
        )
        return next_epoch

    @device_method
    def Finish(
        self,
        workspace,
        sm_idx,
        thread_idx,
        next_epoch,
        sync_scope: l.constexpr,
        kEpilogueGridSyncIndex: l.constexpr,
        kProfile: l.constexpr = False,
    ):
        l.static_assert(self.kNumRanks <= 64)
        if thread_idx < 64:
            next_epoch = amdgcn_shuffle(next_epoch, 0)
            if sm_idx == 0:
                if thread_idx == 0:
                    BufferResource.StoreU32(
                        workspace.br_,
                        self.Workspace.XGpuEpochCounterOffset(workspace.rank_id_),
                        0,
                        next_epoch,
                        BufferResource.kNone,
                    )
                if thread_idx < self.kNumRanks:
                    system_fence_release()
                    BufferResource.StoreU32(
                        workspace.br_,
                        self.Workspace.XGpuEpochSignalOffset(
                            thread_idx, workspace.rank_id_
                        ),
                        0,
                        next_epoch,
                        BufferResource.kSC0Bit | BufferResource.kSC1Bit,
                    )
                wave_barrier()
                amdgcn_s_waitcnt(0, -1, 0)
            if thread_idx < self.kNumRanks:
                wait_xgpu_epoch(
                    workspace,
                    self.Workspace.XGpuEpochSignalOffset(
                        workspace.rank_id_, thread_idx
                    ),
                    next_epoch,
                    kProfile,
                )
                system_fence_acquire()
        sync_scope()
