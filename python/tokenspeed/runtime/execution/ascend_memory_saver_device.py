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

from tokenspeed.runtime.execution.device import DeviceHandle
from tokenspeed.runtime.execution.types import (
    DpForwardMetadata,
    ModelExecutionResult,
    PlannedForward,
)


class AscendMemorySaverDeviceHandle(DeviceHandle):
    """Additional launch and weight-update checks for managed Ascend pools."""

    _NCCL_UNSUPPORTED = "Ascend weight updates support disk and Model Updater SDK"

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
    ) -> ModelExecutionResult | None:
        # A round can still touch KV storage when it has no model forward.
        self._require_active_pools()
        return super().execute(execution_plan, planned, cache_ops=cache_ops)

    def run_idle_forward(self, dp_metadata: DpForwardMetadata) -> None:
        self._require_active_pools()
        super().run_idle_forward(dp_metadata)

    def _is_complete_weight_update(self, req) -> bool:
        from tokenspeed.runtime.engine.io_struct import UpdateWeightFromDiskReqInput

        if type(req) is UpdateWeightFromDiskReqInput:
            return True
        return super()._is_complete_weight_update(req)

    def check_weight_update(self, req) -> tuple[bool, str]:
        """Validate Ascend saver updates before the replica starts a load."""
        from tokenspeed.runtime.engine.io_struct import (
            UpdateWeightFromDiskReqInput,
            UpdateWeightsFromDistributedReqInput,
            UpdateWeightsFromMooncakeReqInput,
        )

        if type(req) is UpdateWeightsFromDistributedReqInput:
            return False, self._NCCL_UNSUPPORTED
        ready, message = super().check_weight_update(req)
        if not ready:
            return ready, message
        # Suspend clears device KV only. The host must invalidate old L2/L3
        # prefixes before any rank writes new weights, including draft inputs.
        if not req.flush_cache:
            return False, "Ascend Memory Saver weight updates require flush_cache=true"
        if type(req) is UpdateWeightsFromMooncakeReqInput:
            return self._executor.model_runner.check_ascend_mooncake_update(
                self._mooncake_update_models()
            )
        if (
            type(req) is UpdateWeightFromDiskReqInput
            and self._weight_memory is not None
            and self._weight_memory.policy == "discard"
            and len(self._weight_memory.models) > 1
        ):
            return False, "Use Model Updater SDK to replace discarded draft weights"
        if type(req) is UpdateWeightFromDiskReqInput:
            # The host agrees admission before cache invalidation, and the
            # local update checks again before beginning a weight transaction.
            return self._executor.model_runner.check_ascend_disk_update(req)
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
            return False, self._NCCL_UNSUPPORTED
        return super().update_weights(req)
