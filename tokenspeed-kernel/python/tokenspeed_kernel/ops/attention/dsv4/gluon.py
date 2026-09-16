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
        gluon_dsv4_decode_split_gfx950 as _dsv4_decode_split_impl,
    )
    from tokenspeed_kernel_amd.ops.gfx950.attention.dsv4 import (
        gluon_dsv4_decode_topk_mxfp4_gfx950 as _dsv4_decode_topk_impl,
    )
    from tokenspeed_kernel_amd.ops.gfx950.attention.dsv4 import (
        gluon_dsv4_plan_gfx950 as _dsv4_plan_impl,
    )
    from tokenspeed_kernel_amd.ops.gfx950.attention.dsv4 import (
        gluon_dsv4_prefill_gfx950 as _dsv4_prefill_impl,
    )
    from tokenspeed_kernel_amd.ops.gfx950.attention.dsv4 import (
        gluon_dsv4_prefill_topk_mxfp4_gfx950 as _dsv4_prefill_topk_impl,
    )

    _DSV4_MXFP4_SIGNATURE = format_signature(
        q=dense_tensor_format(torch.uint8),
        weights=dense_tensor_format(torch.float32),
        index_k_cache=dense_tensor_format(torch.uint8),
    )
    _DSV4_MXFP4_TRAITS = {
        "index_heads": frozenset({32, 64}),
        "head_dim": frozenset({128}),
        "topk": frozenset({512, 1024, 2048}),
        "page_size": frozenset({64}),
        "index_k_format": frozenset({"mxfp4"}),
    }

    @register_kernel(
        "attention",
        "dsv4_prefill_topk",
        name="gluon_dsv4_prefill_topk_mxfp4_gfx950",
        solution="gluon",
        capability=CapabilityRequirement(
            min_arch_version=ArchVersion(9, 5),
            max_arch_version=ArchVersion(9, 5),
            vendors=frozenset({"amd"}),
        ),
        signatures=frozenset({_DSV4_MXFP4_SIGNATURE}),
        traits=_DSV4_MXFP4_TRAITS,
        priority=Priority.SPECIALIZED,
        tags={"amd", "gfx950", "mxfp4", "sparse", "radix_topk"},
    )
    def gluon_dsv4_prefill_topk_mxfp4_gfx950(*args, **kwargs):
        return _dsv4_prefill_topk_impl(*args, **kwargs)

    @register_kernel(
        "attention",
        "dsv4_decode_topk",
        name="gluon_dsv4_decode_topk_mxfp4_gfx950",
        solution="gluon",
        capability=CapabilityRequirement(
            min_arch_version=ArchVersion(9, 5),
            max_arch_version=ArchVersion(9, 5),
            vendors=frozenset({"amd"}),
        ),
        signatures=frozenset({_DSV4_MXFP4_SIGNATURE}),
        traits=_DSV4_MXFP4_TRAITS,
        priority=Priority.SPECIALIZED,
        tags={"amd", "gfx950", "mxfp4", "sparse", "radix_topk"},
    )
    def gluon_dsv4_decode_topk_mxfp4_gfx950(*args, **kwargs):
        return _dsv4_decode_topk_impl(*args, **kwargs)

    @register_kernel(
        "attention",
        "dsv4_plan",
        name="gluon_dsv4_plan_gfx950",
        solution="gluon",
        capability=CapabilityRequirement(
            min_arch_version=ArchVersion(9, 5),
            max_arch_version=ArchVersion(9, 5),
            vendors=frozenset({"amd"}),
        ),
        signatures=frozenset({format_signature()}),
        traits={"page_size": frozenset({64})},
        priority=Priority.SPECIALIZED,
        tags={"amd", "gfx950", "cuda_graph"},
    )
    def gluon_dsv4_plan_gfx950(**kwargs):
        return _dsv4_plan_impl(**kwargs)

    @register_kernel(
        "attention",
        "dsv4_decode",
        name="gluon_dsv4_decode_split_gfx950",
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
                    swa_kv_cache=dense_tensor_format(torch.uint8),
                )
            }
        ),
        priority=Priority.SPECIALIZED,
        traits={
            "tokens": frozenset({1, 2, 3, 4, 5, 6}),
            "head_dim": frozenset({512}),
            "num_heads": frozenset({16, 32}),
            "cache_layout": frozenset({"fp8_swa_page_planar"}),
            "topk_layout": frozenset({"global_slots"}),
            "support_sink": frozenset({True}),
            "has_extra": frozenset({True}),
            "has_extra_segment": frozenset({True}),
            "swa_selected_width": frozenset({128}),
            "extra_selected_width": frozenset({1024}),
            "swa_page_size": frozenset({64}),
            "extra_page_size": frozenset({64}),
            "metadata_dtypes": frozenset({torch.int32}),
        },
        tags={"amd", "gfx950", "paged_cache", "selected_attention"},
    )
    def gluon_dsv4_decode_split_gfx950(*args, **kwargs):
        return _dsv4_decode_split_impl(*args, **kwargs)

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
            "cache_layout": frozenset({"dense_workspace"}),
            "support_sink": frozenset({True}),
            "selected_width": frozenset({384, 512, 640, 768, 1024, 1152}),
            "metadata_dtypes": frozenset({torch.int32}),
        },
        tags={"amd", "gfx950", "selected_attention"},
    )
    def gluon_dsv4_prefill_gfx950(*args, **kwargs):
        return _dsv4_prefill_impl(*args, **kwargs)
