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

"""Ascend partial attention state merge registration."""

import torch
from tokenspeed_kernel.platform import CapabilityRequirement, current_platform
from tokenspeed_kernel.registry import Priority, register_kernel
from tokenspeed_kernel.signature import format_signatures

if current_platform().is_npu:
    from tokenspeed_kernel_npu.ops.mla import attn_merge_state

    register_kernel(
        "attention",
        "attn_merge_state",
        name="ascend_attn_merge_state",
        solution="torch_npu",
        capability=CapabilityRequirement(vendors=frozenset({"ascend"})),
        signatures=format_signatures(("out_a", "out_b"), "dense", {torch.bfloat16}),
        priority=Priority.PORTABLE,
    )(attn_merge_state)

    @register_kernel(
        "attention",
        "merge_normalized_partials",
        name="ascend_merge_normalized_partials",
        solution="ascend",
        capability=CapabilityRequirement(vendors=frozenset({"ascend"})),
        signatures=format_signatures("input", "dense", {torch.bfloat16, torch.float32}),
        priority=Priority.PERFORMANT,
    )
    def ascend_merge_normalized_partials(output, lse):
        from tokenspeed_kernel_npu.ops.dcp_tensors import merge_normalized_partials

        return merge_normalized_partials(output, lse)

    @register_kernel(
        "attention",
        "prefill_cache_layout",
        name="ascend_prefill_cache_layout",
        solution="ascend",
        capability=CapabilityRequirement(vendors=frozenset({"ascend"})),
        signatures=format_signatures("input", "dense", {torch.int32}),
        priority=Priority.PERFORMANT,
    )
    def ascend_prefill_cache_layout(
        table,
        cumulative,
        prefixes,
        *,
        first,
        rows,
        total,
        per_block,
        degree,
        local_pages,
        send_rows,
        page_size,
        active_requests,
        history_table=None,
        patch_slots=None,
        history_pages=None
    ):
        from tokenspeed_kernel_npu.ops.dsa import prefill_cache_layout

        return prefill_cache_layout(
            table,
            cumulative,
            prefixes,
            first=first,
            rows=rows,
            total=total,
            per_block=per_block,
            degree=degree,
            local_pages=local_pages,
            send_rows=send_rows,
            page_size=page_size,
            active_requests=active_requests,
            history_table=history_table,
            patch_slots=patch_slots,
            history_pages=history_pages,
        )
