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

"""The host guard of Beam's one-launch route (``router_route_rejection``), which
decides whether ``beam_route_weights`` runs the Triton route or the Torch
statements, and the FP32 rounding of the routed scale the kernel holds.
``test/nvidia/ops/moe/test_router_select_cuda.py`` compares the two routes' bytes.
"""

from __future__ import annotations

import torch
from tokenspeed_kernel.ops.moe._triton.route_epilogue import route_scale_fp32
from tokenspeed_kernel.ops.moe._triton.router_select import (
    router_route_rejection,
    router_select_rejection,
)

# Beam's routing: 256 experts in 8 groups, group score over the top 2, 4 groups kept,
# top-8.
EXPERTS, NUM_GROUPS, GROUP_TOPK, TOPK_GROUPS, TOPK = 256, 8, 2, 4, 8


def test_host_guards_name_the_first_refused_clause():
    """The rule that gives a call to the one-launch Triton route: metadata only, so
    each clause is reached on CPU; the device clause is last."""

    def select(rows=4, experts=EXPERTS, num_groups=NUM_GROUPS, **layout):
        if layout.get("strided"):
            logits = torch.zeros(rows, 2 * experts)[:, ::2]
        else:
            logits = torch.zeros(rows, experts)
        bias = torch.zeros(EXPERTS, dtype=layout.get("bias_dtype", torch.float32))
        return logits, bias, num_groups, GROUP_TOPK, TOPK_GROUPS, TOPK

    route = select()
    for reason, needle in (
        (router_select_rejection(*select(rows=0)), "logits: rows=0"),
        (
            router_select_rejection(*select(experts=128)),
            "logits: choices must have shape (rows, 256)",
        ),
        (router_select_rejection(*select(num_groups=4)), "logits: num_groups must"),
        (
            router_select_rejection(*select(bias_dtype=torch.bfloat16)),
            "bias must be torch.float32",
        ),
        (
            router_select_rejection(*select(strided=True)),
            "logits: choices must be contiguous",
        ),
        (router_select_rejection(*route), "logits: choices must be a CUDA tensor"),
        (router_route_rejection(*route, 1, None), "renormalize must be a bool"),
        (router_route_rejection(*route, True, "2.5"), "scale must be a real number"),
        (router_route_rejection(*route, True, 1e-40), "scale must round to a finite"),
        (router_route_rejection(*route, True, 2.5), "logits: choices must be a CUDA"),
    ):
        assert reason.startswith(needle), (reason, needle)


def test_route_scale_is_rounded_to_fp32_as_torch_rounds_a_scalar():
    for value in (2.5, 0.1, 1 / 3, 3, 2.0**-126, 1e-40):
        weights = torch.ones(4, dtype=torch.float32)
        weights *= value
        assert route_scale_fp32(value) == weights[0].item(), value
