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


"""MLA prefix K/V expansion on the TRT-LLM CuTe DSL block-FP8 GEMM."""

from __future__ import annotations

from importlib.util import find_spec

import torch
from tokenspeed_kernel.platform import (
    ArchVersion,
    CapabilityRequirement,
    current_platform,
)
from tokenspeed_kernel.registry import Priority, register_kernel
from tokenspeed_kernel.signature import ScaleFormat, format_signatures

if current_platform().is_nvidia and find_spec("cutlass") is not None:

    @register_kernel(
        "attention",
        "mla_expand_kv",
        name="trtllm_mla_expand_kv_fp8",
        solution="trtllm",
        capability=CapabilityRequirement(
            min_arch_version=ArchVersion(10, 0),
            max_arch_version=ArchVersion(10, 3),
            vendors=frozenset({"nvidia"}),
        ),
        signatures=format_signatures(
            ("a", "b"),
            "mxfp8",
            {torch.float8_e4m3fn},
            scale=ScaleFormat(
                storage_dtype=torch.float32, granularity="block", block_shape=(128, 128)
            ),
        ),
        traits={
            "nope_head_dim": frozenset({128}),
            "value_head_dim": frozenset({128}),
        },
        priority=Priority.SPECIALIZED,
    )
    def trtllm_mla_expand_kv_fp8(
        latent, latent_scales, kv_b_weight, kv_b_scales, k_pe, key, value
    ):
        """Project the latent straight into key and value, one GEMM per half of
        each head's kv_b rows, then broadcast the RoPE keys.

        Arguments match tokenspeed_kernel.ops.attention.mla.mla_expand_kv_fp8,
        with 128 key and 128 value rows per head.
        """
        from tokenspeed_kernel.ops.attention.mla._triton.kv_pack import (
            triton_mla_rope_broadcast_fp8,
        )
        from tokenspeed_kernel.thirdparty.trtllm_blockwise import (
            blockwise_gemm_heads,
        )

        nope = key.shape[-1] - k_pe.shape[-1]
        head_rows = nope + value.shape[-1]
        if kv_b_weight.shape[0] != key.shape[1] * head_rows:
            raise ValueError("kv_b rows must hold one key and value block per head")
        blockwise_gemm_heads(
            latent,
            kv_b_weight,
            latent_scales,
            kv_b_scales,
            key[..., :nope],
            row_offset=0,
            head_rows=head_rows,
        )
        blockwise_gemm_heads(
            latent,
            kv_b_weight,
            latent_scales,
            kv_b_scales,
            value,
            row_offset=nope,
            head_rows=head_rows,
        )
        triton_mla_rope_broadcast_fp8(k_pe, key[..., nope:])
