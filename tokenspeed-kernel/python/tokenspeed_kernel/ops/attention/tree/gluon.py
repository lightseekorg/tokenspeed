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

"""Registration shim for the AMD Gluon draft-tree decode."""

from __future__ import annotations

import torch
from tokenspeed_kernel.platform import (
    ArchVersion,
    CapabilityRequirement,
    current_platform,
)
from tokenspeed_kernel.registry import Priority, register_kernel
from tokenspeed_kernel.signature import (
    dense_tensor_format,
    format_signature,
    format_signatures,
)

if current_platform().is_amd:
    from tokenspeed_kernel_amd.ops.gfx950.attention.mha.tree import (
        launch_gluon_mha_tree_decode_gfx950 as _tree_decode_impl,
    )

    @register_kernel(
        "attention",
        "tree_decode",
        name="gluon_mha_tree_decode_gfx950",
        solution="gluon",
        capability=CapabilityRequirement(
            min_arch_version=ArchVersion(9, 5),
            max_arch_version=ArchVersion(9, 5),
            vendors=frozenset({"amd"}),
        ),
        # Matching dtypes, or a 16-bit query over a unit-scale FP8 E4M3 cache
        # (its K/V tiles widen after the LDS read).
        signatures=format_signatures(
            ("q", "k_cache", "v_cache"),
            "dense",
            {torch.float16, torch.bfloat16, torch.float8_e4m3fn},
        )
        | {
            format_signature(
                q=dense_tensor_format(dtype),
                k_cache=dense_tensor_format(torch.float8_e4m3fn),
                v_cache=dense_tensor_format(torch.float8_e4m3fn),
            )
            for dtype in (torch.float16, torch.bfloat16)
        },
        priority=Priority.SPECIALIZED,
        traits={
            "head_dim": frozenset({64, 128, 256}),
            "page_size": frozenset({64}),
            "sliding_window": frozenset({False, True}),
        },
    )
    def gluon_mha_tree_decode_gfx950(*args, **kwargs):
        return _tree_decode_impl(*args, **kwargs)
