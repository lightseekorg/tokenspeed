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

"""Shared pieces of the monokernel MoE stream tests."""

from types import SimpleNamespace

import torch
from tokenspeed_kernel_amd.ops.gfx950.moe.mxfp4.weight_preprocess import (
    preprocess_gluon_mxfp4_gfx950_moe_weights,
)

HIDDEN, INTER = 4096, 512

# The physical widths and the integer arguments the codegen test compiles as
# multiples of 16 (Triton specializes them at launch when they are).
PHYS = dict(N_PHYS1=2 * INTER, N_PHYS2=HIDDEN)
DIVISIBLE = {
    "stride_w13",
    "stride_s13",
    "stride_s13nb",
    "stride_w2",
    "stride_s2",
    "stride_s2nb",
    "stride_xsnb",
}


def make_experts(num_experts, device, seed):
    """A processed MXFP4 MoE layer as Muse's EP1 decode reads it (random
    weights, random E8M0 scales), and its gate/up scales before the swizzle
    (the input is interleaved, so their rows are the physical columns)."""
    generator = torch.Generator(device=device).manual_seed(seed)

    def draw(low, high, *shape):
        return torch.randint(
            low, high, shape, dtype=torch.uint8, device=device, generator=generator
        )

    module = torch.nn.Module()
    module.w13_input_layout = "interleaved"
    module.quant_config = SimpleNamespace(use_dynamic_mxfp4_activations=True)
    w13_scale = draw(118, 131, num_experts, 2 * INTER, HIDDEN // 32)
    module.w13_weight = torch.nn.Parameter(
        draw(0, 256, num_experts, 2 * INTER, HIDDEN // 2), requires_grad=False
    )
    module.w13_weight_scale = torch.nn.Parameter(w13_scale.clone(), requires_grad=False)
    module.w2_weight = torch.nn.Parameter(
        draw(0, 256, num_experts, HIDDEN, INTER // 2), requires_grad=False
    )
    module.w2_weight_scale = torch.nn.Parameter(
        draw(118, 131, num_experts, HIDDEN, INTER // 32), requires_grad=False
    )
    preprocess_gluon_mxfp4_gfx950_moe_weights({}, module)
    return module, w13_scale
