# Copyright (c) 2026 LightSeek Foundation
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

"""CUDA rotary embedding kernels."""

from typing import Any

import torch
from tokenspeed_kernel.platform import CapabilityRequirement, current_platform
from tokenspeed_kernel.registry import Priority, register_kernel
from tokenspeed_kernel.signature import format_signatures

platform = current_platform()

if platform.is_nvidia:
    from tokenspeed_kernel.thirdparty.cuda.rope import (
        apply_rope_with_cos_sin_cache_inplace,
    )

    @register_kernel(
        "embedding",
        "rope",
        name="cuda_embedding_rope",
        solution="cuda",
        capability=CapabilityRequirement(vendors=frozenset({"nvidia"})),
        signatures=format_signatures(
            ("q", "k"), "dense", {torch.float16, torch.bfloat16}
        ),
        priority=Priority.PERFORMANT,
        traits={
            "head_size": frozenset({64, 128, 256, 512}),
            "partial_rotary": frozenset({True, False}),
            "is_neox": frozenset({True, False}),
            "has_fused_kv": frozenset({True, False}),
            "has_fused_mla_kv": frozenset({False}),
            "has_q_out": frozenset({True, False}),
            "has_k_out": frozenset({True, False}),
        },
        tags={"latency"},
    )
    def cuda_embedding_rope(
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
        if fused_mla_set_kv_buffer_arg is not None:
            raise ValueError("CUDA RoPE does not support MLA fused KV write")
        apply_rope_with_cos_sin_cache_inplace(
            positions=positions,
            query=q,
            key=k,
            head_size=head_size,
            cos_sin_cache=cos_sin_cache,
            is_neox=is_neox,
            fused_set_kv_buffer_arg=fused_set_kv_buffer_arg,
            output_q_rope=q_rope_out,
            output_k_rope=k_rope_out,
            enable_pdl=enable_pdl,
        )
