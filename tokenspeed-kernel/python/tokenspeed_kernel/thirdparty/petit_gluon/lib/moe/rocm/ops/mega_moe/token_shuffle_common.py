"""Routing-counter initialization and cleanup for direct push."""

from lib.gemm.rocm.amd_intrinsics import BufferResource
from lib.moe.rocm.comm.barrier import system_fence_release
from lib.moe.rocm.mega_moe.workspace import MegaMoEWorkspace
from lib.tal.device import DeviceTemplate, device_method
from triton.experimental.gluon import language as l


class TokenShuffleCommon(DeviceTemplate):
    def __init__(self, Config):
        self._key = Config.cache_key
        self.Workspace = MegaMoEWorkspace(Config)
        self.kThreads, self.kTopK = Config.kThreads, Config.kTopK
        self.kNumExperts, self.kNumRanks = Config.kNumExperts, Config.kNumRanks
        self.kExpertsPerRank = self.kNumExperts // self.kNumRanks

    @device_method
    def ClearExpertCounts(self, expert_count, tid):
        kIterations: l.constexpr = (
            self.kNumExperts + self.kThreads - 1
        ) // self.kThreads
        for i in l.static_range(kIterations):
            expert = tid + i * self.kThreads
            if expert < self.kNumExperts:
                l.store(expert_count + expert, 0)
        l.barrier()

    @device_method
    def ResetRoutingCounters(self, workspace, sm_id, tid):
        if sm_id != 0:
            return
        kExpertIterations: l.constexpr = (
            self.kNumExperts + self.kThreads - 1
        ) // self.kThreads
        for i in l.static_range(kExpertIterations):
            expert = tid + i * self.kThreads
            if expert < self.kNumExperts:
                BufferResource.StoreU64(
                    workspace.br_,
                    self.Workspace.SendCounterOffset(expert),
                    0,
                    (0, 0),
                    BufferResource.kNone,
                )
        for i in range(tid, self.kNumRanks * self.kExpertsPerRank, self.kThreads):
            BufferResource.StoreU64(
                workspace.br_,
                self.Workspace.RecvCounterOffset(
                    workspace.rank_id_,
                    i // self.kExpertsPerRank,
                    i % self.kExpertsPerRank,
                ),
                0,
                (0, 0),
                BufferResource.kAtomicScopeSystem,
            )
        for local_expert in range(tid, self.kExpertsPerRank, self.kThreads):
            BufferResource.StoreU64(
                workspace.br_,
                self.Workspace.RecvSumCounterOffset(local_expert),
                0,
                (0, 0),
                BufferResource.kAtomicScopeSystem,
            )
        system_fence_release()
