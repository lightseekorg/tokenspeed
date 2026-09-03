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

"""Registration shim for Ascend rotary embedding."""

from typing import Any

import torch
from tokenspeed_kernel.platform import CapabilityRequirement, current_platform
from tokenspeed_kernel.registry import Priority, register_kernel
from tokenspeed_kernel.signature import format_signatures

if current_platform().is_npu:
    from tokenspeed_kernel_npu.ops.rotary_embedding import apply_rope as _apply_rope

    @register_kernel(
        "embedding",
        "rope",
        name="ascend_embedding_rope",
        solution="torch_npu",
        capability=CapabilityRequirement(vendors=frozenset({"ascend"})),
        signatures=format_signatures(
            ("q", "k"), "dense", {torch.float16, torch.bfloat16}
        ),
        priority=Priority.PERFORMANT,
        traits={
            "partial_rotary": frozenset({True, False}),
            "is_neox": frozenset({True, False}),
            "has_fused_kv": frozenset({False}),
            "has_fused_mla_kv": frozenset({False}),
            "fused_mla_full_query": frozenset({False}),
            "has_q_out": frozenset({True, False}),
            "has_k_out": frozenset({True, False}),
        },
        tags={"portability"},
    )
    def ascend_embedding_rope(
        *,
        positions: torch.Tensor,
        q: torch.Tensor,
        k: torch.Tensor,
        head_size: int,
        cos_sin_cache: torch.Tensor,
        is_neox: bool = True,
        fused_set_kv_buffer_arg: Any = None,
        fused_mla_set_kv_buffer_arg: Any = None,
        q_rope_out: torch.Tensor | None = None,
        k_rope_out: torch.Tensor | None = None,
        enable_pdl: bool = False,
    ) -> None:
        del enable_pdl
        if (
            fused_set_kv_buffer_arg is not None
            or fused_mla_set_kv_buffer_arg is not None
        ):
            raise ValueError("Ascend RoPE does not support fused KV writes")
        _apply_rope(
            positions=positions,
            q=q,
            k=k,
            head_size=head_size,
            cos_sin_cache=cos_sin_cache,
            is_neox=is_neox,
            q_rope_out=q_rope_out,
            k_rope_out=k_rope_out,
        )
