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

"""FlashInfer LL/BT/HT behind one workspace.

The pinned FlashInfer version includes the H3584 HT kernel and tuning presets.
LL/BT routes cap at 1024 tokens; HT covers the full serving capacity.
"""

from __future__ import annotations

import torch
import torch.distributed as dist


def allreduce_fusion_support_error(group, hidden_size, top_k, max_num_tokens, dtype):
    """Return a local incompatibility without allocating collective state."""
    if type(max_num_tokens) is not int or max_num_tokens <= 0:
        return "allreduce fusion capacity must be positive"
    if not dist.is_initialized() or dist.get_world_size(group) not in (4, 8, 16):
        return "allreduce fusion requires an initialized TP4, TP8, or TP16 group"
    if hidden_size != 3584 or top_k != 16 or dtype != torch.bfloat16:
        return "allreduce fusion currently supports BF16 H3584/top-k16"
    if not torch.cuda.is_available():
        return "CUDA is unavailable"
    device = torch.device("cuda", torch.cuda.current_device())
    if not (10, 0) <= torch.cuda.get_device_capability(device) <= (10, 3):
        return "allreduce fusion requires data-center Blackwell"
    try:
        import torch.distributed._symmetric_memory as symm_mem
        from flashinfer.comm import allreduce_fusion
        from flashinfer.comm.mnnvl import is_multicast_supported
        from flashinfer.comm.mnnvl_cutedsl.kernel_ht import protocol as ht_protocol
        from flashinfer.comm.mnnvl_cutedsl.kernel_ll.protocol import LLAllReduceTuning
        from flashinfer.comm.mnnvl_cutedsl_ar import (
            MNNVLCuteDSLAllReduceFusionWorkspace,
        )

        if not all(
            hasattr(ht_protocol, name)
            for name in ("HT_FINALIZE_GB300_H3584_K16", "HT_ALL_REDUCE_GB300_H3584")
        ):
            return "FlashInfer H3584 HT presets are unavailable"

        if symm_mem.get_backend(device) is None or not is_multicast_supported(
            device.index
        ):
            return "NVLink multicast symmetric memory is unavailable"
        if not all(
            callable(value)
            for value in (
                allreduce_fusion,
                MNNVLCuteDSLAllReduceFusionWorkspace,
                LLAllReduceTuning,
                ht_protocol.HTProtocol,
            )
        ):
            return "FlashInfer allreduce fusion interfaces are unavailable"
    except Exception as error:
        return f"allreduce fusion support probe raised {type(error).__name__}: {error}"
    return None


class MNNVLAllReduceFusionBackend:
    """Own both input forms and all three protocol workspaces before capture."""

    def __init__(self, group, hidden_size, top_k, max_num_tokens, rms_eps):
        from flashinfer.comm.mnnvl_cutedsl.config import (
            KernelTarget,
            MNNVLCuteDSLConfig,
            MRangeDispatch,
            ProtocolKind,
            StaticProfile,
        )
        from flashinfer.comm.mnnvl_cutedsl.kernel_bt.protocol import (
            BTAllReduceTuning,
            BTCollectiveTuning,
            BTFinalizeTuning,
        )
        from flashinfer.comm.mnnvl_cutedsl.kernel_ht.protocol import (
            HT_ALL_REDUCE_GB300_H3584,
            HT_FINALIZE_GB300_H3584_K16,
        )
        from flashinfer.comm.mnnvl_cutedsl.kernel_ll.protocol import (
            LLAllReduceTuning,
            LLCollectiveTuning,
            LLFinalizeTuning,
        )
        from flashinfer.comm.mnnvl_cutedsl_ar import (
            MNNVLCuteDSLAllReduceFusionWorkspace,
        )

        ll_collective = LLCollectiveTuning(
            cluster_size=8, rank_lanes=1, threads=128, enable_pdl=True
        )
        bt_collective = BTCollectiveTuning(
            reduction_threads=224, rms_threads=448, enable_pdl=True
        )
        profile = StaticProfile(
            tp_size=group.size(),
            hidden_size=hidden_size,
            top_k=top_k,
            dtype=torch.bfloat16,
            finalize_routes=MRangeDispatch(
                upper_bounds=(32, 1024, None),
                targets=(
                    KernelTarget(
                        ProtocolKind.LL,
                        LLFinalizeTuning(
                            elements_per_thread=4,
                            threads=128,
                            prefetch_group=16,
                            load_shared_expert_before_pdl=False,
                            collective=ll_collective,
                        ),
                    ),
                    KernelTarget(
                        ProtocolKind.BT,
                        BTFinalizeTuning(
                            elements_per_thread=2,
                            threads=256,
                            prefetch_group=1,
                            load_shared_expert_before_pdl=False,
                            collective=bt_collective,
                        ),
                    ),
                    KernelTarget(ProtocolKind.HT, HT_FINALIZE_GB300_H3584_K16),
                ),
            ),
            all_reduce_routes=MRangeDispatch(
                upper_bounds=(32, 1024, None),
                targets=(
                    KernelTarget(
                        ProtocolKind.LL,
                        LLAllReduceTuning(
                            publish_elements_per_thread=8,
                            publish_threads=128,
                            publish_release_before_store=False,
                            collective=ll_collective,
                        ),
                    ),
                    KernelTarget(
                        ProtocolKind.BT,
                        BTAllReduceTuning(
                            publish_threads=256,
                            publish_vectors_per_thread=1,
                            collective=bt_collective,
                        ),
                    ),
                    KernelTarget(ProtocolKind.HT, HT_ALL_REDUCE_GB300_H3584),
                ),
            ),
        )
        self._workspace = MNNVLCuteDSLAllReduceFusionWorkspace(
            tp_size=group.size(),
            tp_rank=group.rank(),
            max_token_num=max_num_tokens,
            hidden_dim=hidden_size,
            dtype=torch.bfloat16,
            group=group,
            top_k=top_k,
            rms_eps=rms_eps,
            routed_scaling_factor=1.0,
            weight_bias=0.0,
            include_shared_expert=False,
            add_residual=False,
            write_residual_output=False,
            config=MNNVLCuteDSLConfig(profiles=(profile,)),
        )
        self.device = torch.device("cuda", torch.cuda.current_device())
        self.output = torch.empty(
            (max_num_tokens, hidden_size), device=self.device, dtype=torch.bfloat16
        )
        self._empty_routed = torch.zeros(
            (1, hidden_size), device=self.device, dtype=torch.bfloat16
        )
        self.rms_eps = rms_eps

    def run(self, input, gamma, num_tokens, finalize, expert_weights, expanded_idx):
        from flashinfer.comm import allreduce_fusion
        from flashinfer.comm.allreduce import AllReduceFusionPattern

        output = self.output[:num_tokens]
        if finalize and input.shape[0] == 0:
            input = self._empty_routed
        pattern = (
            AllReduceFusionPattern.kMoEFinalizeARResidualRMSNorm
            if finalize
            else AllReduceFusionPattern.kARResidualRMSNorm
        )
        return allreduce_fusion(
            input=input,
            workspace=self._workspace,
            pattern=pattern,
            launch_with_pdl=True,
            trigger_completion_at_end=True,
            fp32_acc=False,
            residual_out=None,
            norm_out=output,
            residual_in=None,
            rms_gamma=gamma,
            rms_eps=self.rms_eps,
            expanded_idx_to_permuted_idx=expanded_idx,
            expert_scale_factor=expert_weights,
            shared_expert_output=None,
            weight_bias=0.0,
        )
