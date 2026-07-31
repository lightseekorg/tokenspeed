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

"""Registration shim for AMD Gluon GEMM kernels.

Dense16 Gluon GEMM registration is intentionally disabled so dense GEMM
dispatch falls back to the portable torch GEMM implementation.
"""

from __future__ import annotations

import torch
from tokenspeed_kernel.ops.gemm.moe_input_projections import fused_weight_view
from tokenspeed_kernel.platform import (
    ArchVersion,
    CapabilityRequirement,
    current_platform,
)
from tokenspeed_kernel.registry import Priority, register_kernel
from tokenspeed_kernel.signature import dense_tensor_format, format_signature

if current_platform().is_amd:
    from tokenspeed_kernel_amd.ops.gfx950.gemm.fp16.fused_moe_input_projections_gfx950 import (
        gluon_fused_moe_input_projections_gfx950 as _fused_moe_input_projections_impl,
    )
    from tokenspeed_kernel_amd.ops.gfx950.gemm.fp16.moe_input_projections_gfx950 import (
        gluon_moe_input_projections_gfx950 as _moe_input_projections_impl,
    )

    @register_kernel(
        "gemm",
        "moe_input_projections",
        name="gluon_moe_input_projections_gfx950",
        solution="gluon",
        capability=CapabilityRequirement(
            min_arch_version=ArchVersion(9, 5),
            max_arch_version=ArchVersion(9, 5),
            vendors=frozenset({"amd"}),
        ),
        signatures=frozenset(
            {
                format_signature(
                    hidden_states=dense_tensor_format(torch.bfloat16),
                    router_weight=dense_tensor_format(torch.bfloat16),
                    routed_weight=dense_tensor_format(torch.bfloat16),
                    shared_gate_up_weight=dense_tensor_format(torch.bfloat16),
                )
            }
        ),
        priority=Priority.SPECIALIZED,
        traits={
            "tokens": frozenset({1}),
            "hidden_size": frozenset({7168}),
            "num_experts": frozenset({896}),
            "latent_size": frozenset({3584}),
            "shared_size": frozenset({768}),
            "inputs_contiguous": frozenset({True}),
        },
    )
    def gluon_moe_input_projections_gfx950(**kwargs):
        return _moe_input_projections_impl(
            kwargs["hidden_states"],
            kwargs["router_weight"],
            kwargs["routed_weight"],
            kwargs["shared_gate_up_weight"],
            beta=kwargs["gate_clamp"],
            linear_beta=kwargs["up_clamp"],
        )

    @register_kernel(
        "gemm",
        "moe_input_projections",
        name="gluon_fused_moe_input_projections_gfx950",
        solution="gluon",
        capability=CapabilityRequirement(
            min_arch_version=ArchVersion(9, 5),
            max_arch_version=ArchVersion(9, 5),
            vendors=frozenset({"amd"}),
        ),
        signatures=frozenset(
            {
                format_signature(
                    hidden_states=dense_tensor_format(torch.bfloat16),
                    router_weight=dense_tensor_format(torch.bfloat16),
                    routed_weight=dense_tensor_format(torch.bfloat16),
                    shared_gate_up_weight=dense_tensor_format(torch.bfloat16),
                )
            }
        ),
        # Below the single-token specialist, which still wins at one token,
        # and above the portable Triton kernel. Past ~128 tokens the tiling
        # here loses to that kernel, so the trait stops short of it.
        priority=Priority.SPECIALIZED - 1,
        traits={
            "weights_fused": frozenset({True}),
            "inputs_contiguous": frozenset({True}),
            "tokens": frozenset(range(1, 129)),
        },
    )
    def gluon_fused_moe_input_projections_gfx950(**kwargs):
        weights = (
            kwargs["router_weight"],
            kwargs["routed_weight"],
            kwargs["shared_gate_up_weight"],
        )
        return _fused_moe_input_projections_impl(
            kwargs["hidden_states"],
            *weights,
            fused_weight_view(*weights),
            beta=kwargs["gate_clamp"],
            linear_beta=kwargs["up_clamp"],
        )
