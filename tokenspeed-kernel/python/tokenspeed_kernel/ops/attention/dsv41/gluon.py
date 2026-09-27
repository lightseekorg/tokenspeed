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

"""Registration shims for AMD Gluon DeepSeek V4.1 attention kernels."""

from __future__ import annotations

import torch
from tokenspeed_kernel.ops.attention.dsv41._gluon.indexer import (
    launch_gfx950_logits,
    launch_gfx1250_logits,
    run_dsv41_csa2_index_topk,
)
from tokenspeed_kernel.platform import (
    ArchVersion,
    CapabilityRequirement,
    current_platform,
)
from tokenspeed_kernel.registry import Priority, register_kernel
from tokenspeed_kernel.signature import dense_tensor_format, format_signature

if current_platform().is_amd:
    from tokenspeed_kernel_amd.ops.gfx950.attention.dsv41 import (
        launch_gluon_dsv41_selected_attention_gfx950 as _dsv41_selected_gfx950,
    )
    from tokenspeed_kernel_amd.ops.gfx1250.attention.dsv41 import (
        launch_gluon_dsv41_selected_attention_gfx1250 as _dsv41_selected_gfx1250,
    )

    _SIGNATURES = frozenset({format_signature(x=dense_tensor_format(torch.bfloat16))})
    _INDEX_TRAITS = {
        "index_heads": frozenset(range(1, 33)),
        "index_k_format": frozenset({"mxfp4"}),
        "index_shards": frozenset({1}),
        "native_indexer": frozenset({False}),
    }

    @register_kernel(
        "attention",
        "dsv41_selected_attention",
        name="gluon_dsv41_selected_attention_gfx950",
        solution="gluon",
        capability=CapabilityRequirement(
            min_arch_version=ArchVersion(9, 5),
            max_arch_version=ArchVersion(9, 5),
            vendors=frozenset({"amd"}),
        ),
        signatures=_SIGNATURES,
        priority=Priority.SPECIALIZED,
    )
    def gluon_dsv41_selected_attention_gfx950(*args, **kwargs):
        return _dsv41_selected_gfx950(*args, **kwargs)

    @register_kernel(
        "attention",
        "dsv41_index_topk",
        name="gluon_dsv41_index_topk_gfx950",
        solution="gluon",
        capability=CapabilityRequirement(
            min_arch_version=ArchVersion(9, 5),
            max_arch_version=ArchVersion(9, 5),
            vendors=frozenset({"amd"}),
        ),
        signatures=_SIGNATURES,
        traits=_INDEX_TRAITS,
        priority=Priority.SPECIALIZED,
    )
    def gluon_dsv41_index_topk_gfx950(*args, **kwargs):
        return run_dsv41_csa2_index_topk(
            *args, **kwargs, launch_logits=launch_gfx950_logits
        )

    @register_kernel(
        "attention",
        "dsv41_selected_attention",
        name="gluon_dsv41_selected_attention_gfx1250",
        solution="gluon",
        capability=CapabilityRequirement(
            min_arch_version=ArchVersion(12, 5),
            max_arch_version=ArchVersion(12, 5),
            vendors=frozenset({"amd"}),
        ),
        signatures=_SIGNATURES,
        priority=Priority.SPECIALIZED,
    )
    def gluon_dsv41_selected_attention_gfx1250(*args, **kwargs):
        return _dsv41_selected_gfx1250(*args, **kwargs)

    @register_kernel(
        "attention",
        "dsv41_index_topk",
        name="gluon_dsv41_index_topk_gfx1250",
        solution="gluon",
        capability=CapabilityRequirement(
            min_arch_version=ArchVersion(12, 5),
            max_arch_version=ArchVersion(12, 5),
            vendors=frozenset({"amd"}),
        ),
        signatures=_SIGNATURES,
        traits=_INDEX_TRAITS,
        priority=Priority.SPECIALIZED,
    )
    def gluon_dsv41_index_topk_gfx1250(*args, **kwargs):
        return run_dsv41_csa2_index_topk(
            *args, **kwargs, launch_logits=launch_gfx1250_logits
        )
