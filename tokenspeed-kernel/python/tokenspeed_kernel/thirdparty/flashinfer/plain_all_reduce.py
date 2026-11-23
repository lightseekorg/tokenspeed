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

"""FlashInfer's BT owner-reduce protocol as a plain all-reduce.

Uses FlashInfer's GB300 TP8 H7168 BT presets without norm or residual. Callers
import this module where they use it, so a forward reload of it re-executes
nothing else.
"""

from __future__ import annotations

import torch
import torch.distributed as dist

# Plain all-reduces of at least this many bytes per rank take the BT kernels;
# smaller ones keep the Lamport one-shot and two-shot (GB300, TP8, H7168:
# 64 rows break even, 128 rows are 6% faster, 512 rows 22%).
PLAIN_ALL_REDUCE_MIN_BYTES = 1 << 20
# FlashInfer's TP8 H7168 BT split: its first preset serves through 256 rows.
_PLAIN_ALL_REDUCE_SPLIT_ROWS = 256


def plain_all_reduce_support_error(group, hidden_size, max_num_tokens, dtype):
    """Return why this rank cannot build the plain BT all-reduce, or None."""
    if type(max_num_tokens) is not int or max_num_tokens <= 0:
        return "plain all-reduce capacity must be positive"
    if not dist.is_initialized() or dist.get_world_size(group) != 8:
        return "the plain BT all-reduce presets are measured for TP8"
    if hidden_size != 7168 or dtype != torch.bfloat16:
        return "the plain BT all-reduce presets cover BF16 H7168"
    if not torch.cuda.is_available():
        return "CUDA is unavailable"
    device = torch.device("cuda", torch.cuda.current_device())
    if torch.cuda.get_device_capability(device) < (10, 0):
        return "the plain BT all-reduce requires SM100 or newer"
    try:
        import torch.distributed._symmetric_memory as symm_mem
        from flashinfer.comm import allreduce_fusion
        from flashinfer.comm.mnnvl import is_multicast_supported
        from flashinfer.comm.mnnvl_cutedsl.kernel_bt import protocol as bt_protocol

        if not all(
            hasattr(bt_protocol, name)
            for name in (
                "BT_ALL_REDUCE_GB300_TP8_H7168_PRESET_0",
                "BT_ALL_REDUCE_GB300_TP8_H7168_PRESET_1",
                "BT_FINALIZE_GB300_TP8_H7168_K16_PRESET_0",
            )
        ):
            return "FlashInfer TP8 H7168 BT presets are unavailable"
        if symm_mem.get_backend(device) is None or not is_multicast_supported(
            device.index
        ):
            return "NVLink multicast symmetric memory is unavailable"
        if not callable(allreduce_fusion):
            return "FlashInfer allreduce fusion interfaces are unavailable"
    except Exception as error:
        return f"plain all-reduce support probe raised {type(error).__name__}: {error}"
    return None


class MNNVLPlainAllReduceBackend:
    """FlashInfer's BT owner-reduce all-reduce without norm or residual.

    Each rank publishes every token's slice to the token's owner rank, the
    owner sums the ranks in order 0..7 in fp32 and rounds once, and multicasts
    the sum. Built collectively before graph capture; calls are sequential.
    """

    def __init__(self, group, hidden_size, max_num_tokens):
        from flashinfer.comm.mnnvl_cutedsl.config import (
            KernelTarget,
            MNNVLCuteDSLConfig,
            MRangeDispatch,
            ProtocolKind,
            StaticProfile,
        )
        from flashinfer.comm.mnnvl_cutedsl.kernel_bt.protocol import (
            BT_ALL_REDUCE_GB300_TP8_H7168_PRESET_0,
            BT_ALL_REDUCE_GB300_TP8_H7168_PRESET_1,
            BT_FINALIZE_GB300_TP8_H7168_K16_PRESET_0,
        )
        from flashinfer.comm.mnnvl_cutedsl_ar import (
            MNNVLCuteDSLAllReduceFusionWorkspace,
        )

        bt = ProtocolKind.BT
        profile = StaticProfile(
            tp_size=group.size(),
            hidden_size=hidden_size,
            top_k=16,
            dtype=torch.bfloat16,
            # The profile requires a finalize route; the plain path never takes it.
            finalize_routes=MRangeDispatch(
                upper_bounds=(None,),
                targets=(KernelTarget(bt, BT_FINALIZE_GB300_TP8_H7168_K16_PRESET_0),),
            ),
            all_reduce_routes=MRangeDispatch(
                upper_bounds=(_PLAIN_ALL_REDUCE_SPLIT_ROWS, None),
                targets=(
                    KernelTarget(bt, BT_ALL_REDUCE_GB300_TP8_H7168_PRESET_0),
                    KernelTarget(bt, BT_ALL_REDUCE_GB300_TP8_H7168_PRESET_1),
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
            top_k=16,
            include_shared_expert=False,
            add_residual=False,
            write_residual_output=True,
            apply_rms_norm=False,
            config=MNNVLCuteDSLConfig(profiles=(profile,), applies_rms_norm=False),
        )
        self.hidden_size = hidden_size
        self.max_num_tokens = max_num_tokens

    def serves(self, num_tokens, hidden_size, dtype):
        """Whether a plain all-reduce of this shape takes the BT kernels."""
        return (
            dtype == torch.bfloat16
            and hidden_size == self.hidden_size
            and 0 < num_tokens <= self.max_num_tokens
            and num_tokens * hidden_size * 2 >= PLAIN_ALL_REDUCE_MIN_BYTES
        )

    def run(self, input, output):
        from flashinfer.comm import allreduce_fusion
        from flashinfer.comm.allreduce import AllReduceFusionPattern

        return allreduce_fusion(
            input=input,
            workspace=self._workspace,
            pattern=AllReduceFusionPattern.kAllReduce,
            launch_with_pdl=True,
            trigger_completion_at_end=True,
            output=output,
        )
