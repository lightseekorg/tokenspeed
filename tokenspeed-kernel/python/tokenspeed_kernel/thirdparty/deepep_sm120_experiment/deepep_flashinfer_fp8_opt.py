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

from tokenspeed_kernel.thirdparty.deepep_sm120_experiment.route_mask import mask_routes

"""Single-node prototype: FP8 DeepEP dispatch and SM120 FlashInfer experts."""
import torch
from tokenspeed_kernel.thirdparty.deepep_sm120_experiment.prequant_fused_moe_opt import (
    apply_sm12x_fused_moe,
)
from tokenspeed_kernel.thirdparty.deepep_sm120_experiment.q0_pack import pack_q0


def apply_deepep_flashinfer(
    buffer,
    x,
    state,
    topk_ids,
    topk_weights,
    *,
    num_experts,
    num_sms,
    num_qps,
    num_max_tokens_per_rank,
    communication_capacity,
):
    """Return source-local outputs; the caller owns the prepared buffer/weights.

    Logical and communication capacities are explicit and identical across
    ranks. Each graph captures both; valid receive counts remain device-side.
    """
    from tokenspeed_kernel.thirdparty.deepep_sm120_experiment.low_latency_wire1 import (
        LowLatencyWire1Buffer,
    )

    if (
        not 0 < num_max_tokens_per_rank <= communication_capacity
        or x.shape[0] > num_max_tokens_per_rank
    ):
        raise ValueError("Source rows exceed the logical or communication capacity")
    if isinstance(buffer, LowLatencyWire1Buffer):
        if num_experts != buffer.experts:
            raise ValueError("Expert partition differs from initialized buffer")
        return buffer.apply(
            x,
            state,
            topk_ids,
            topk_weights,
            capacity=num_max_tokens_per_rank,
            layout_capacity=communication_capacity,
        )
    packed = pack_q0(x)
    received, ids, weights, handle, _ = buffer.dispatch(
        packed,
        topk_ids,
        topk_weights,
        num_experts=num_experts,
        num_sms=num_sms,
        num_qps=num_qps,
        do_cpu_sync=False,
        num_max_tokens_per_rank=communication_capacity,
    )
    rx, scales = received
    # Capacity tails are not initialized by asynchronous dispatch. Consumers
    # must mask them before routing/quantization, including when no token arrives.
    local_ids, safe_weights = mask_routes(
        ids, weights, handle.psum_num_recv_tokens_per_scaleup_rank, state.w2.shape[0]
    )
    from tokenspeed_kernel.thirdparty.flashinfer.sm12x_nvfp4_moe import (
        Sm12xNvfp4MoEWeights,
        apply_sm12x_nvfp4,
    )

    if isinstance(state, Sm12xNvfp4MoEWeights):
        from tokenspeed_kernel.thirdparty.deepep_sm120_experiment.nvfp4_receive import (
            decode_elastic_fp8,
        )

        rows = decode_elastic_fp8(
            rx, scales, handle.psum_num_recv_tokens_per_scaleup_rank
        )
        value = apply_sm12x_nvfp4(rows, state, local_ids, safe_weights, False)
    else:
        value = apply_sm12x_fused_moe(rx, scales, state, local_ids, safe_weights, False)
    # FlashInfer already applies routing weights; combine must not reapply them.
    out, _, _ = buffer.combine(value, handle, num_sms=num_sms, num_qps=num_qps)
    return out
