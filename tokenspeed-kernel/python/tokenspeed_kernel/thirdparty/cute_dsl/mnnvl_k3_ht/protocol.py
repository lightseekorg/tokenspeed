# Copyright (c) 2026 by FlashInfer team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""FlashInfer HT protocol adapter for native Kimi K3 latent width."""

from flashinfer.comm.mnnvl_cutedsl.kernel_ht.protocol import (
    HTAllReduceTuning,
    HTFinalizeTuning,
    HTProtocol,
)
from tokenspeed_kernel.thirdparty.cute_dsl.mnnvl_k3_ht.device_kernel import (
    K3H3584MoeFinalizeAllReduceRMSNormHTDeviceKernel,
)

K3_LATENT_SIZE = 3584
K3_TOP_K = 16
K3_TP_SIZE = 8
K3_HT_FINALIZE_GB300_TP8_H3584_K16 = HTFinalizeTuning(
    persistent_ctas=None,
    consumer_threads=448,
    vectors_per_thread=1,
    stages=7,
    reduction_warps=2,
    reduction_cta_groups=None,
    rms_token_groups=2,
    rms_pipeline_stages=3,
    rms_shard_major=False,
    enable_pdl=True,
)
K3_HT_ALL_REDUCE_GB300_TP8_H3584 = HTAllReduceTuning(
    persistent_ctas=None,
    consumer_threads=448,
    vectors_per_thread=1,
    stages=2,
    reduction_warps=2,
    reduction_cta_groups=None,
    rms_token_groups=2,
    rms_pipeline_stages=1,
    rms_shard_major=False,
    enable_pdl=True,
)


class K3H3584HTProtocol(HTProtocol):
    """HT protocol using a tail-predicated 56-pack TP8 reduction shard."""

    def _validate_k3_geometry(self) -> None:
        if self.hidden_size != K3_LATENT_SIZE:
            raise ValueError(f"hidden_size must be {K3_LATENT_SIZE}")
        if self.tp_size != K3_TP_SIZE:
            raise ValueError(f"tp_size must be {K3_TP_SIZE}")
        if self.top_k not in (1, K3_TOP_K):
            raise ValueError(f"top_k must be 1 or {K3_TOP_K}")

    def _compile_finalize(self, tuning: HTFinalizeTuning):
        self._validate_k3_geometry()
        active_ctas = self._resolve_ctas(tuning.persistent_ctas)
        groups = tuning.reduction_cta_groups or active_ctas // self.tp_size
        kernel = K3H3584MoeFinalizeAllReduceRMSNormHTDeviceKernel(
            hidden=self.hidden_size,
            top_k=self.top_k,
            tp=self.tp_size,
            rank=self.rank,
            active_ctas=active_ctas,
            stages=tuning.stages,
            consumer_threads=tuning.consumer_threads,
            vectors_per_thread=tuning.vectors_per_thread,
            reduction_warps=tuning.reduction_warps,
            reduction_cta_groups=groups,
            rms_token_groups=tuning.rms_token_groups,
            rms_pipeline_stages=tuning.rms_pipeline_stages,
            rms_shard_major=tuning.rms_shard_major,
            rms_epsilon=self.rms_epsilon,
            routed_scaling_factor=self.routed_scaling_factor,
            weight_bias=self.weight_bias,
            include_shared_expert=self.include_shared_expert,
            add_residual=self.add_residual,
            write_residual_output=self.write_residual_output,
            enable_pdl=tuning.enable_pdl,
        )
        return self._compile(kernel, top_k=self.top_k)

    def _compile_all_reduce(self, tuning: HTAllReduceTuning):
        self._validate_k3_geometry()
        active_ctas = self._resolve_ctas(tuning.persistent_ctas)
        groups = tuning.reduction_cta_groups or active_ctas // self.tp_size
        kernel = K3H3584MoeFinalizeAllReduceRMSNormHTDeviceKernel(
            hidden=self.hidden_size,
            top_k=0,
            tp=self.tp_size,
            rank=self.rank,
            active_ctas=active_ctas,
            stages=tuning.stages,
            consumer_threads=tuning.consumer_threads,
            vectors_per_thread=tuning.vectors_per_thread,
            reduction_warps=tuning.reduction_warps,
            reduction_cta_groups=groups,
            rms_token_groups=tuning.rms_token_groups,
            rms_pipeline_stages=tuning.rms_pipeline_stages,
            rms_shard_major=tuning.rms_shard_major,
            rms_epsilon=self.rms_epsilon,
            routed_scaling_factor=1.0,
            weight_bias=self.weight_bias,
            include_shared_expert=True,
            add_residual=self.add_residual,
            write_residual_output=self.write_residual_output,
            enable_pdl=tuning.enable_pdl,
        )
        return self._compile(kernel, top_k=0)


__all__ = [
    "K3H3584HTProtocol",
    "K3_HT_ALL_REDUCE_GB300_TP8_H3584",
    "K3_HT_FINALIZE_GB300_TP8_H3584_K16",
    "K3_LATENT_SIZE",
    "K3_TOP_K",
    "K3_TP_SIZE",
]
