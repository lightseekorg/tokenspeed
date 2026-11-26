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

"""Device-side safety checks for the native Ascend memory saver."""

from collections.abc import Sequence

from tokenspeed.runtime.cache.transfer.ops import RestoreOp, SnapshotOp
from tokenspeed.runtime.engine.io_struct import UpdateWeightFromDiskReqInput
from tokenspeed.runtime.execution.device import DeviceHandle
from tokenspeed.runtime.execution.types import (
    DpForwardMetadata,
    ModelExecutionResult,
    PlannedForward,
)


class AscendMemorySaverDeviceHandle(DeviceHandle):
    """Additional launch and weight-update checks for managed Ascend pools."""

    _additional_discard_reload_types = (UpdateWeightFromDiskReqInput,)
    _stale_kv_backup = False

    def commit_weight_update(self, success: bool) -> None:
        super().commit_weight_update(success)
        # A partial load or a repeated version string can change the target too.
        # The host flushed its prefix index before loading. Never expose these
        # backed-up bytes through a new transfer registration.
        self._stale_kv_backup = self._weight_memory.kv_released

    def _restore_kv_contents(self) -> None:
        if self._weight_memory.policy == "discard" or self._stale_kv_backup:
            super()._restore_kv_contents()
        self._stale_kv_backup = False

    _TRAINER_WEIGHT_UPDATE_UNSUPPORTED = (
        "This trainer weight-update path uses CUDA tensors and is not implemented "
        "for NPU, including backend='hccl'. Use disk or Model Updater SDK updates."
    )

    def _require_active_pools(self) -> None:
        # A failed allocator transition can leave pages unmapped before the
        # control plane records a successful release or wake.
        self._executor.model_runner.memory_saver_adapter.assert_active()

    def execute(
        self,
        execution_plan,
        planned: PlannedForward | None,
        *,
        cache_ops: Sequence,
        retirement_slots: Sequence[int] = (),
        retirement_id: int | None = None,
    ) -> ModelExecutionResult | None:
        # Snapshots, PD slot setup and recurrent-state work also use model-owned
        # state. Ordinary KV copies and page maintenance only need the cache pools.
        if (
            planned is not None
            or retirement_slots
            or execution_plan.remote_prefill is not None
            or any(isinstance(op, (SnapshotOp, RestoreOp)) for op in cache_ops)
            or any(len(pairs) for pairs in execution_plan.state_folds_arrays().values())
        ):
            self._require_active_pools()
        else:
            self._executor.model_runner.memory_saver_adapter.assert_cache_active()
        return super().execute(
            execution_plan,
            planned,
            cache_ops=cache_ops,
            retirement_slots=retirement_slots,
            retirement_id=retirement_id,
        )

    def run_idle_forward(self, dp_metadata: DpForwardMetadata) -> None:
        self._require_active_pools()
        super().run_idle_forward(dp_metadata)

    def check_weight_update(self, req) -> tuple[bool, str]:
        """Validate Ascend saver updates before the replica starts a load."""
        from tokenspeed.runtime.engine.io_struct import (
            UpdateWeightsFromDistributedReqInput,
        )

        if type(req) is UpdateWeightsFromDistributedReqInput:
            return False, self._TRAINER_WEIGHT_UPDATE_UNSUPPORTED
        ready, message = super().check_weight_update(req)
        if not ready:
            return ready, message
        # Suspend clears device KV only. The host must invalidate old L2/L3
        # prefixes before any rank writes new weights, including draft inputs.
        if not req.flush_cache:
            return False, "Ascend Memory Saver weight updates require flush_cache=true"
        if (
            type(req) is UpdateWeightFromDiskReqInput
            and self._weight_memory is not None
            and self._weight_memory.policy == "discard"
            and len(self._weight_memory.models) > 1
        ):
            return (
                False,
                "Disk discard restore updates the target only; restore the discarded "
                "target and refresh draft together with Model Updater SDK",
            )
        return ready, message

    def update_weights(self, req) -> tuple[bool, str]:
        from tokenspeed.runtime.engine.io_struct import (
            DestroyWeightsUpdateGroupReqInput,
            InitWeightsUpdateGroupReqInput,
        )

        if type(req) in (
            DestroyWeightsUpdateGroupReqInput,
            InitWeightsUpdateGroupReqInput,
        ):
            return False, self._TRAINER_WEIGHT_UPDATE_UNSUPPORTED
        return super().update_weights(req)
