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

"""Ascend MoE kernel registrations."""

import torch
from tokenspeed_kernel.platform import CapabilityRequirement, current_platform
from tokenspeed_kernel.registry import Priority, register_kernel
from tokenspeed_kernel.signature import format_signatures

if current_platform().is_npu:
    from tokenspeed_kernel_npu.ops.moe import (
        ascend_bf16_distributed_precomputed_moe_apply as _distributed_moe_apply,
    )
    from tokenspeed_kernel_npu.ops.moe import (
        ascend_bf16_precomputed_moe_apply as _moe_apply,
    )
    from tokenspeed_kernel_npu.ops.moe import (
        ascend_bf16_process_moe_weights as _process_weights,
    )
    from tokenspeed_kernel_npu.ops.moe import (
        ascend_int8_precomputed_moe_apply as _int8_moe_apply,
    )
    from tokenspeed_kernel_npu.ops.moe import (
        ascend_int8_process_moe_weights as _int8_process_weights,
    )
    from tokenspeed_kernel_npu.ops.moe import (
        ascend_softmax_bias_topk as _softmax_bias_topk,
    )
    from tokenspeed_kernel_npu.ops.w4a4 import (
        ascend_int4_precomputed_moe_apply as _int4_moe_apply,
    )
    from tokenspeed_kernel_npu.ops.w4a4 import (
        ascend_int4_process_moe_weights as _int4_process_weights,
    )

    _CAPABILITY = CapabilityRequirement(vendors=frozenset({"ascend"}))

    @register_kernel(
        "moe",
        "softmax_bias_topk",
        name="ascend_softmax_bias_topk",
        solution="torch_npu",
        capability=_CAPABILITY,
        signatures=format_signatures(
            "router_logits", "dense", {torch.float16, torch.bfloat16, torch.float32}
        ),
        priority=Priority.SPECIALIZED,
        features={"ascend", "latency"},
    )
    def softmax_bias_topk(**kwargs):
        return _softmax_bias_topk(**kwargs)

    @register_kernel(
        "moe",
        "apply",
        name="ascend_bf16_precomputed_moe_apply",
        solution="torch_npu",
        capability=_CAPABILITY,
        signatures=format_signatures("x", "dense", {torch.bfloat16}),
        traits={
            "weight_dtype": frozenset({"unquant"}),
            "activation": frozenset({"silu", "swiglu"}),
            "routing_mode": frozenset({"precomputed_topk"}),
            "supports_deferred_finalize": frozenset({False}),
            "supports_ep": frozenset({True}),
            "supports_all_to_all_ep": frozenset({False}),
            "ispp_alignment": frozenset({32}),
            "internal_activation_dtype": frozenset({"input"}),
            "supports_bias": frozenset({False}),
        },
        priority=Priority.SPECIALIZED,
        features={"ascend", "cuda_graph"},
        weight_preprocessor=_process_weights,
    )
    def moe_apply(**kwargs):
        return _moe_apply(**kwargs)

    @register_kernel(
        "moe",
        "apply",
        name="ascend_int8_precomputed_moe_apply",
        solution="torch_npu",
        capability=_CAPABILITY,
        signatures=format_signatures("x", "dense", {torch.bfloat16}),
        traits={
            "weight_dtype": frozenset({"int8"}),
            "activation": frozenset({"silu", "swiglu"}),
            "routing_mode": frozenset({"precomputed_topk"}),
            "supports_deferred_finalize": frozenset({False}),
            "supports_ep": frozenset({True}),
            "supports_all_to_all_ep": frozenset({False}),
            "ispp_alignment": frozenset({32}),
            "internal_activation_dtype": frozenset({"int8"}),
            "supports_bias": frozenset({False}),
        },
        priority=Priority.SPECIALIZED,
        features={"ascend", "npu_graph", "w8a8", "nz"},
        weight_preprocessor=_int8_process_weights,
    )
    def int8_moe_apply(**kwargs):
        return _int8_moe_apply(**kwargs)

    @register_kernel(
        "moe",
        "apply",
        name="ascend_int4_precomputed_moe_apply",
        solution="flash_npu_chain",
        capability=_CAPABILITY,
        signatures=format_signatures("x", "dense", {torch.bfloat16}),
        traits={
            "weight_dtype": frozenset({"int4"}),
            "activation": frozenset({"silu", "swiglu"}),
            "routing_mode": frozenset({"precomputed_topk"}),
            "supports_deferred_finalize": frozenset({False}),
            "supports_ep": frozenset({True}),
            "supports_all_to_all_ep": frozenset({False}),
            "ispp_alignment": frozenset({64}),
            "internal_activation_dtype": frozenset({"int4"}),
            "supports_bias": frozenset({False}),
        },
        priority=Priority.SPECIALIZED,
        features={"ascend", "npu_graph", "w4a4", "flatquant", "nz"},
        weight_preprocessor=_int4_process_weights,
    )
    def int4_moe_apply(**kwargs):
        return _int4_moe_apply(**kwargs)

    @register_kernel(
        "moe",
        "apply",
        name="ascend_bf16_distributed_precomputed_moe_apply",
        solution="torch_npu",
        capability=_CAPABILITY,
        signatures=format_signatures("x", "dense", {torch.bfloat16}),
        traits={
            "weight_dtype": frozenset({"unquant"}),
            "activation": frozenset({"silu", "swiglu"}),
            "routing_mode": frozenset({"precomputed_topk"}),
            "supports_deferred_finalize": frozenset({False}),
            "supports_ep": frozenset({True}),
            "supports_all_to_all_ep": frozenset({True}),
            "ispp_alignment": frozenset({32}),
            "internal_activation_dtype": frozenset({"input"}),
            "supports_bias": frozenset({False}),
        },
        priority=Priority.SPECIALIZED,
        features={"ascend", "npu_graph", "all_to_all"},
        weight_preprocessor=_process_weights,
    )
    def distributed_moe_apply(**kwargs):
        return _distributed_moe_apply(**kwargs)


__all__ = ["moe_apply", "softmax_bias_topk"]
