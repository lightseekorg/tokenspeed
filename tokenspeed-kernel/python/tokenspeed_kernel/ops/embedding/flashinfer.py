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

"""FlashInfer rotary embedding kernels."""

import torch
from tokenspeed_kernel.platform import CapabilityRequirement, current_platform
from tokenspeed_kernel.registry import Priority, register_kernel
from tokenspeed_kernel.signature import format_signatures

platform = current_platform()


if platform.is_nvidia:
    from flashinfer.rope import mla_rope_quantize_fp8

    @register_kernel(
        "embedding",
        "rope_mla",
        name="flashinfer_embedding_rope_mla",
        solution="flashinfer",
        capability=CapabilityRequirement(vendors=frozenset({"nvidia"})),
        signatures=format_signatures(
            ("q_rope", "k_rope", "q_nope", "k_nope"),
            "dense",
            {torch.float16, torch.bfloat16},
        ),
        priority=Priority.SPECIALIZED,
        traits={
            "is_neox": frozenset({True, False}),
            "quantize_dtype": frozenset({torch.float8_e4m3fn}),
            "has_scale_q_tensor": frozenset({False}),
            "has_scale_kv_tensor": frozenset({False}),
        },
    )
    def flashinfer_embedding_rope_mla(
        *,
        positions: torch.Tensor,
        q_rope: torch.Tensor,
        k_rope: torch.Tensor,
        q_nope: torch.Tensor,
        k_nope: torch.Tensor,
        cos_sin_cache: torch.Tensor,
        q_rope_out: torch.Tensor,
        k_rope_out: torch.Tensor,
        q_nope_out: torch.Tensor,
        k_nope_out: torch.Tensor,
        is_neox: bool = True,
        quant_scale_q: float = 1.0,
        quant_scale_kv: float = 1.0,
        enable_pdl: bool = False,
    ) -> None:
        mla_rope_quantize_fp8(
            q_rope=q_rope,
            k_rope=k_rope,
            q_nope=q_nope,
            k_nope=k_nope,
            cos_sin_cache=cos_sin_cache,
            pos_ids=positions,
            is_neox=is_neox,
            quantize_dtype=torch.float8_e4m3fn,
            q_rope_out=q_rope_out,
            k_rope_out=k_rope_out,
            q_nope_out=q_nope_out,
            k_nope_out=k_nope_out,
            quant_scale_q=quant_scale_q,
            quant_scale_kv=quant_scale_kv,
            enable_pdl=enable_pdl,
        )
