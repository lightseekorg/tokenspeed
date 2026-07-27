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

"""Intel XPU rotary embedding (vllm-xpu-kernels backend).

Registers an XPU ``embedding.rope`` kernel that delegates to the vllm-xpu-kernels
rotary-embedding Torch op. Registered with conservative traits (no fused KV-cache
write) so the selector falls back to the portable Triton kernel for fused paths.
"""

from __future__ import annotations

import torch

from tokenspeed_kernel.platform import CapabilityRequirement
from tokenspeed_kernel.registry import Priority, register_kernel
from tokenspeed_kernel.signature import format_signatures

# vllm-xpu-kernels registers rotary_embedding into the PyTorch dispatcher when
# `vllm_xpu_kernels._C` is imported (done by the package __init__).
_XPU_ROPE_OP = getattr(getattr(torch.ops, "_C", None), "rotary_embedding", None)


if _XPU_ROPE_OP is not None:

    @register_kernel(
        "embedding",
        "rope",
        name="xpu_embedding_rope",
        solution="xpu",
        capability=CapabilityRequirement(vendors=frozenset({"intel"})),
        signatures=format_signatures(
            ("q", "k"), "dense", {torch.float16, torch.bfloat16}
        ),
        priority=Priority.PERFORMANT,
        # Conservative: no fused KV-cache write, no separate out buffers. Those
        # cases fall back to the Triton kernel.
        traits={
            "partial_rotary": frozenset({True, False}),
            "is_neox": frozenset({True, False}),
            "has_fused_kv": frozenset({False}),
            "has_fused_mla_kv": frozenset({False}),
            "has_q_out": frozenset({False}),
            "has_k_out": frozenset({False}),
        },
        tags={"xpu"},
    )
    def xpu_embedding_rope(
        *,
        positions: torch.Tensor,
        q: torch.Tensor,
        k: torch.Tensor,
        head_size: int,
        cos_sin_cache: torch.Tensor,
        is_neox: bool = True,
        fused_set_kv_buffer_arg: object | None = None,
        fused_mla_set_kv_buffer_arg: object | None = None,
        q_rope_out: torch.Tensor | None = None,
        k_rope_out: torch.Tensor | None = None,
        enable_pdl: bool = False,
    ) -> None:
        """Apply rotary embedding in place on ``q`` and ``k`` (XPU).

        TODO(intel): verify the vllm rotary op signature/layout vs v0.1.7,
        especially cos_sin_cache packing (concat(cos, sin) on last dim) and the
        expected [num_tokens, num_heads * head_size] q/k layout.
        """
        # vllm's rotary op requires cos_sin_cache to match the q/k dtype (it is
        # commonly stored as float32 in the runtime), so cast when needed.
        if cos_sin_cache.dtype != q.dtype:
            cos_sin_cache = cos_sin_cache.to(q.dtype)
        # vllm rotary_embedding rotates query/key in place.
        _XPU_ROPE_OP(positions, q, k, head_size, cos_sin_cache, is_neox)
