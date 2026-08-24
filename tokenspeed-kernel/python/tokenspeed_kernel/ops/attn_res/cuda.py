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
#
# Blackwell (sm_100a/sm_103a) TMA Attention-Residual forward kernel.
import torch
from tokenspeed_kernel.platform import (
    ArchVersion,
    CapabilityRequirement,
    current_platform,
)
from tokenspeed_kernel.registry import Priority, register_kernel
from tokenspeed_kernel.signature import format_signatures

platform = current_platform()

# Register only when the compiled kernel is actually loadable, so a Blackwell box
# with a missing/failed build degrades to the torch fallback via select_kernel
# instead of crashing on the first call.
_HAS_CUDA_KERNEL = False
if platform.is_nvidia and platform.is_blackwell:
    from tokenspeed_kernel.thirdparty.cuda.attn_res import (
        attn_res_fwd_packed,
        has_attn_res_fwd,
    )

    _HAS_CUDA_KERNEL = has_attn_res_fwd()

if _HAS_CUDA_KERNEL:

    @register_kernel(
        "attn_res",
        "fwd",
        name="cuda_attn_res_fwd",
        solution="cuda",
        capability=CapabilityRequirement(
            min_arch_version=ArchVersion(10, 0),
            vendors=frozenset({"nvidia"}),
        ),
        signatures=format_signatures(
            ("layer_residual", "block_residual"), "dense", {torch.bfloat16}
        ),
        priority=Priority.SPECIALIZED,
        traits={
            "has_delta": frozenset({False, True}),
            # The kernel is instantiated at H=7168 only and checks it; without
            # this the registry picks it for any H and the check aborts.
            "hidden_size": frozenset({7168}),
            "inputs_on_same_gpu": frozenset({True}),
            "partial_block_storage": frozenset({False, True}),
            "separate_output_eps": frozenset({False}),
            "writes_block": frozenset({False, True}),
        },
        tags={"latency", "throughput"},
    )
    def cuda_attn_res_fwd(
        *,
        layer_residual,
        block_residual,
        res_weight,
        rms_weight,
        eps,
        out_norm_weight=None,
        out_norm_eps=None,
        delta=None,
        num_valid_blocks=None,
        block_write_idx=-1,
    ) -> torch.Tensor:
        if (
            out_norm_weight is not None
            and out_norm_eps is not None
            and out_norm_eps != eps
        ):
            raise ValueError("CUDA AttnRes requires matching RMSNorm epsilons")
        # `.contiguous()` materializes a private copy when the caller passes a
        # non-contiguous view. The kernel writes the snapshot through
        # block_residual and, under delta, the accumulated prefix through
        # layer_residual; both writes would land in the temporary and the
        # caller's tensor would stay stale, since only the mixed output is
        # returned. Detect the copy and write the affected rows back.
        packed_block = block_residual.unsqueeze(2)
        snapshot_copied = block_write_idx >= 0 and not packed_block.is_contiguous()
        packed_block = packed_block.contiguous()
        packed_layer = layer_residual.unsqueeze(1)
        prefix_copied = delta is not None and not packed_layer.is_contiguous()
        packed_layer = packed_layer.contiguous()
        out = attn_res_fwd_packed(
            packed_layer,
            packed_block,
            res_weight.contiguous(),
            rms_weight.contiguous(),
            eps,
            out_norm_weight=(
                None if out_norm_weight is None else out_norm_weight.contiguous()
            ),
            delta=None if delta is None else delta.unsqueeze(1).contiguous(),
            num_blocks=num_valid_blocks,
            block_write_idx=block_write_idx,
        )
        if snapshot_copied:
            block_residual[block_write_idx].copy_(
                packed_block[block_write_idx].squeeze(1)
            )
        if prefix_copied:
            layer_residual.copy_(packed_layer.squeeze(1))
        return out.squeeze(1)  # [T, H]
