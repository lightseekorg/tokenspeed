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

"""Registration shims for AMD Gluon DSV4 attention kernels."""

from __future__ import annotations

import torch
from tokenspeed_kernel.platform import (
    ArchVersion,
    CapabilityRequirement,
    current_platform,
)
from tokenspeed_kernel.registry import Priority, register_kernel
from tokenspeed_kernel.signature import dense_tensor_format, format_signature

if current_platform().is_amd:
    from tokenspeed_kernel_amd.ops.gfx950.attention.dsv4 import (
        launch_gluon_dsv4_prefill_gfx950 as _dsv4_prefill_impl,
    )
    from tokenspeed_kernel_amd.ops.gfx1250.attention.dsv4 import (
        launch_gluon_dsv4_prefill_gfx1250 as _dsv4_prefill_gfx1250_impl,
    )

    @register_kernel(
        "attention",
        "dsv4_prefill",
        name="gluon_dsv4_prefill_gfx950",
        solution="gluon",
        capability=CapabilityRequirement(
            min_arch_version=ArchVersion(9, 5),
            max_arch_version=ArchVersion(9, 5),
            vendors=frozenset({"amd"}),
        ),
        signatures=frozenset(
            {
                format_signature(
                    q=dense_tensor_format(torch.bfloat16),
                    kv=dense_tensor_format(torch.bfloat16),
                )
            }
        ),
        priority=Priority.SPECIALIZED,
        traits={
            "head_dim": frozenset({512}),
            "selected_width": frozenset({128, 384, 512, 640, 768, 1024, 1152}),
            "cache_layout": frozenset({"dense_workspace"}),
            "metadata_dtypes": frozenset({torch.int32}),
            "sinks": frozenset({True}),
        },
    )
    def gluon_dsv4_prefill_gfx950(*args, **kwargs):
        return _dsv4_prefill_impl(*args, **kwargs)

    @register_kernel(
        "attention",
        "dsv4_prefill",
        name="gluon_dsv4_prefill_gfx1250",
        solution="gluon",
        capability=CapabilityRequirement(
            min_arch_version=ArchVersion(12, 5),
            max_arch_version=ArchVersion(12, 5),
            vendors=frozenset({"amd"}),
        ),
        signatures=frozenset(
            {
                format_signature(
                    q=dense_tensor_format(torch.bfloat16),
                    kv=dense_tensor_format(torch.bfloat16),
                )
            }
        ),
        priority=Priority.SPECIALIZED,
        traits={
            "head_dim": frozenset({512}),
            # 133: DSpark draft attention, a 128-row window plus the 5-row block.
            "selected_width": frozenset({128, 133, 384, 512, 640, 768, 1024, 1152}),
            "cache_layout": frozenset({"dense_workspace"}),
            "metadata_dtypes": frozenset({torch.int32}),
            "sinks": frozenset({True}),
        },
    )
    def gluon_dsv4_prefill_gfx1250(*args, **kwargs):
        return _dsv4_prefill_gfx1250_impl(*args, **kwargs)
