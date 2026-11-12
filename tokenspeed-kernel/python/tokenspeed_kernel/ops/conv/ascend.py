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

"""Ascend causal conv1d kernel registrations.

The leaves live in ``tokenspeed_kernel_npu.ops.conv`` and drive CANN's native
``npu_fused_causal_conv1d``; see that module for the window contract and for
what the native op cannot express.

``causal_conv1d_prefill_qkv`` is not registered: the native op writes one
packed output, and the portable Triton leaf already serves the fused split.
``causal_conv1d_update`` is registered with ``num_accepted_tokens`` and
``intermediate_conv_window`` excluded by the leaf, so selection keeps those
call sites on the portable solution.
"""

from __future__ import annotations

import torch
from tokenspeed_kernel.platform import CapabilityRequirement, current_platform
from tokenspeed_kernel.registry import Priority, register_kernel
from tokenspeed_kernel.signature import format_signatures

if current_platform().is_npu:
    from tokenspeed_kernel_npu.ops.conv import causal_conv1d_prefill as _prefill
    from tokenspeed_kernel_npu.ops.conv import causal_conv1d_update as _update

    _CAPABILITY = CapabilityRequirement(vendors=frozenset({"ascend"}))
    # The native op takes bfloat16 activations and a bfloat16 window.
    _DTYPES = {torch.bfloat16}

    @register_kernel(
        "conv",
        "causal_conv1d_prefill",
        name="ascend_causal_conv1d_prefill",
        solution="torch_npu",
        capability=_CAPABILITY,
        signatures=format_signatures("x", "dense", _DTYPES),
        priority=Priority.SPECIALIZED,
        traits={
            # Qwen3.5/3.6 use four taps; the native op supplies three and the
            # leaf adds the rest. Anything narrower than the native width has
            # no native span at all.
            "width": frozenset({3, 4}),
            "state_layout": frozenset({"channel_major"}),
            "channel_contiguous": frozenset({True}),
            "weight_dtype": frozenset({torch.bfloat16}),
            # The leaf applies the bias itself; the native op has no bias input.
            "bias_dtype": frozenset({None, torch.bfloat16}),
            "state_dtype": frozenset({torch.bfloat16}),
            "indexed_state": frozenset({True}),
        },
        features={"ascend"},
    )
    def ascend_causal_conv1d_prefill(**kwargs):
        return _prefill(**kwargs)

    @register_kernel(
        "conv",
        "causal_conv1d_update",
        name="ascend_causal_conv1d_update",
        solution="torch_npu",
        capability=_CAPABILITY,
        signatures=format_signatures("x", "dense", _DTYPES),
        priority=Priority.SPECIALIZED,
        traits={
            "width": frozenset({3, 4}),
            "state_layout": frozenset({"channel_major"}),
        },
        features={"ascend"},
    )
    def ascend_causal_conv1d_update(**kwargs):
        return _update(**kwargs)


__all__: list[str] = []
