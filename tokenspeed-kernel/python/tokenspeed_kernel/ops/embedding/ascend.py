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

"""Registration shim for Ascend NPU RoPE embedding kernel."""

from __future__ import annotations

import torch
from tokenspeed_kernel.platform import (
    ArchVersion,
    CapabilityRequirement,
    current_platform,
)
from tokenspeed_kernel.registry import Priority, register_kernel
from tokenspeed_kernel.signature import format_signatures

if current_platform().is_ascend:
    from tokenspeed_kernel_ascend.ops.embedding import (
        torch_npu_embedding_rope as _rope_impl,
    )

    @register_kernel(
        "embedding",
        "rope",
        name="torch_npu_embedding_rope",
        solution="torch_npu",
        capability=CapabilityRequirement(
            min_arch_version=ArchVersion(1, 0),
            vendors=frozenset({"ascend"}),
        ),
        signatures=format_signatures(
            ("query", "key"), "dense", {torch.float16, torch.bfloat16}
        ),
        priority=Priority.PERFORMANT,
        traits={
            "partial_rotary": frozenset({True, False}),
            "is_neox": frozenset({True, False}),
            "has_fused_kv": frozenset({False}),
            "has_q_out": frozenset({True, False}),
            "has_k_out": frozenset({True, False}),
        },
    )
    def torch_npu_embedding_rope(*args, **kwargs):
        return _rope_impl(*args, **kwargs)
