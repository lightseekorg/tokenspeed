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

"""Initialization-time selection for Ascend grouped INT4 GEMMs."""

import logging

import torch
from tokenspeed_kernel.platform import CapabilityRequirement
from tokenspeed_kernel.registry import Priority, register_kernel
from tokenspeed_kernel.selection import KernelPlan, resolve_kernel_override
from tokenspeed_kernel.signature import dense_tensor_format, format_signature
from tokenspeed_kernel_npu.ops import dsv41_w4a4_native

logger = logging.getLogger(__name__)
_SIGNATURE = format_signature(
    quantized=dense_tensor_format(torch.int32),
    weight=dense_tensor_format(torch.int32),
)
_CAPABILITY = CapabilityRequirement(vendors=frozenset({"ascend"}))
_CONTRACT = {
    "weight_layout": frozenset({"NZ"}),
    "out_dtype": frozenset({torch.bfloat16}),
}


@register_kernel(
    "gemm",
    "grouped_w4a4",
    name="cann_grouped_w4a4",
    solution="cann",
    capability=_CAPABILITY,
    signatures=frozenset({_SIGNATURE}),
    traits=_CONTRACT,
    features={"npu_graph"},
    priority=Priority.PORTABLE,
)
def cann_grouped_w4a4(quantized, weight, weight_scale, token_scale, counts):
    import torch_npu

    return torch_npu.npu_grouped_matmul(
        [quantized],
        [weight.transpose(1, 2)],
        scale=[weight_scale],
        per_token_scale=[token_scale],
        group_list=counts,
        split_item=3,
        group_type=0,
        group_list_type=1,
        output_dtype=torch.bfloat16,
    )[0]


@register_kernel(
    "gemm",
    "grouped_w4a4",
    name="ascend_c_grouped_w4a4",
    solution="ascend_c",
    capability=_CAPABILITY,
    signatures=frozenset({_SIGNATURE}),
    traits={
        **_CONTRACT,
        "projection_shape": dsv41_w4a4_native.SUPPORTED_PROJECTIONS,
        "native_available": frozenset({True}),
    },
    features={"npu_graph"},
    priority=Priority.SPECIALIZED,
)
def ascend_c_grouped_w4a4(quantized, weight, weight_scale, token_scale, counts):
    out = quantized.new_empty(
        (quantized.shape[0], weight.shape[1]), dtype=torch.bfloat16
    )
    return dsv41_w4a4_native.grouped_w4a4(
        quantized, weight, weight_scale, token_scale, counts, out
    )


def plan_grouped_w4a4(
    input_size: int, output_size: int, *, solution: str | None
) -> KernelPlan:
    """Choose a fixed GMM for packed INT4 activations and NZ expert weights.

    The sizes are the unpacked K and N dimensions. ``solution=None`` permits
    automatic selection; ``cann`` and ``ascend_c`` require that implementation.
    Standard kernel overrides are honored, including capability checks. Token
    and expert counts remain device inputs and never participate in selection.
    """
    requested = resolve_kernel_override("gemm", "grouped_w4a4", None) or solution
    native_available = (
        (input_size, output_size) in dsv41_w4a4_native.SUPPORTED_PROJECTIONS
        and requested not in ("cann", "cann_grouped_w4a4")
        and dsv41_w4a4_native.available()
    )
    plan = KernelPlan.select(
        "gemm",
        "grouped_w4a4",
        _SIGNATURE,
        features=frozenset({"npu_graph"}),
        static_traits={
            "projection_shape": (input_size, output_size),
            "native_available": native_available,
            "weight_layout": "NZ",
            "out_dtype": torch.bfloat16,
        },
        variable_traits=frozenset({"m", "experts"}),
        declared_traits=frozenset(_CONTRACT),
        solution=solution,
        ignore_layout=False,
    )
    logger.info(
        f"Ascend W4A4 GMM: K={input_size}, N={output_size}, kernel={plan.kernel.name}"
    )
    return plan
