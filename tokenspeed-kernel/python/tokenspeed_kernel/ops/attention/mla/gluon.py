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

"""Registration shims for AMD Gluon MLA attention kernels."""

from __future__ import annotations

import torch
from tokenspeed_kernel.platform import (
    ArchVersion,
    CapabilityRequirement,
    current_platform,
)
from tokenspeed_kernel.registry import Priority, register_kernel
from tokenspeed_kernel.signature import (
    dense_tensor_format,
    format_signature,
    format_signatures,
)

if current_platform().is_amd:
    from tokenspeed_kernel_amd.ops.gfx950.attention.mla.decode import (
        gluon_mla_decode_bf16xbf16_gfx950_bh16_multiblock as _mla_decode_bf16xbf16_bh16_multiblock_impl,
    )
    from tokenspeed_kernel_amd.ops.gfx950.attention.mla.decode import (
        gluon_mla_decode_bf16xbf16_gfx950_bh16bn64 as _mla_decode_bf16xbf16_bh16bn64_impl,
    )
    from tokenspeed_kernel_amd.ops.gfx950.attention.mla.decode import (
        gluon_mla_decode_bf16xbf16_gfx950_bh64 as _mla_decode_bf16xbf16_bh64_impl,
    )
    from tokenspeed_kernel_amd.ops.gfx950.attention.mla.decode import (
        gluon_mla_decode_bf16xbf16_gfx950_bh64_small as _mla_decode_bf16xbf16_bh64_small_impl,
    )
    from tokenspeed_kernel_amd.ops.gfx950.attention.mla.decode import (
        gluon_mla_decode_bf16xfp8_gfx950 as _mla_decode_bf16xfp8_impl,
    )
    from tokenspeed_kernel_amd.ops.gfx950.attention.mla.decode import (
        gluon_mla_decode_fp8xfp8_gfx950 as _mla_decode_fp8xfp8_impl,
    )
    from tokenspeed_kernel_amd.ops.gfx950.attention.mla.decode import (
        gluon_mla_decode_projected_value_gfx950 as _mla_decode_projected_value_impl,
    )
    from tokenspeed_kernel_amd.ops.gfx950.attention.mla.normalize_project_query import (
        gluon_mla_normalize_project_query_gfx950 as _mla_normalize_project_query_impl,
    )
    from tokenspeed_kernel_amd.ops.gfx950.attention.mla.prefill import (
        gluon_mla_prefill_gfx950 as _mla_prefill_gfx950_impl,
    )
    from tokenspeed_kernel_amd.ops.gfx950.attention.mla.project_value import (
        gluon_mla_project_value_gfx950 as _mla_project_value_impl,
    )
    from tokenspeed_kernel_amd.ops.gfx1250.attention.mla.decode import (
        gluon_mla_decode_gfx1250 as _mla_decode_gfx1250_impl,
    )
    from tokenspeed_kernel_amd.ops.gfx1250.attention.mla.decode import (
        gluon_mla_decode_projected_value_gfx1250 as _mla_decode_projected_value_gfx1250_impl,
    )
    from tokenspeed_kernel_amd.ops.gfx1250.attention.mla.extend import (
        gluon_mla_extend_gfx1250 as _mla_extend_gfx1250_impl,
    )
    from tokenspeed_kernel_amd.ops.gfx1250.attention.mla.normalize_project_query import (
        gluon_mla_normalize_project_query_gfx1250 as _mla_normalize_project_query_gfx1250_impl,
    )
    from tokenspeed_kernel_amd.ops.gfx1250.attention.mla.prefill import (
        gluon_mla_prefill_gfx1250 as _mla_prefill_gfx1250_impl,
    )
    from tokenspeed_kernel_amd.ops.gfx1250.attention.mla.project_value import (
        gluon_mla_project_value_gfx1250 as _mla_project_value_gfx1250_impl,
    )

    @register_kernel(
        "attention",
        "mla_decode_with_kvcache",
        name="gluon_mla_decode_bf16xbf16_gfx950_bh16bn64",
        solution="gluon",
        capability=CapabilityRequirement(
            min_arch_version=ArchVersion(9, 5),
            max_arch_version=ArchVersion(9, 5),
            vendors=frozenset({"amd"}),
        ),
        signatures=format_signatures(
            ("q", "kv_cache"),
            "dense",
            {torch.bfloat16},
        ),
        priority=Priority.SPECIALIZED,
        traits={
            "q_len": frozenset({1}),
            "num_q_heads": frozenset(range(1, 17)),
            "page_size": frozenset({64}),
            "kv_lora_rank": frozenset({512}),
            "qk_rope_head_dim": frozenset({64}),
            "support_logit_cap": frozenset({False}),
            "return_lse": frozenset({False, True}),
        },
    )
    def gluon_mla_decode_bf16xbf16_gfx950_bh16bn64(*args, **kwargs):
        return _mla_decode_bf16xbf16_bh16bn64_impl(*args, **kwargs)

    @register_kernel(
        "attention",
        "mla_decode_with_kvcache",
        name="gluon_mla_decode_bf16xbf16_gfx950_bh16_multiblock",
        solution="gluon",
        capability=CapabilityRequirement(
            min_arch_version=ArchVersion(9, 5),
            max_arch_version=ArchVersion(9, 5),
            vendors=frozenset({"amd"}),
        ),
        signatures=format_signatures(
            ("q", "kv_cache"),
            "dense",
            {torch.bfloat16},
        ),
        priority=Priority.SPECIALIZED,
        traits={
            "batch_size": frozenset({1}),
            "q_len": frozenset({1}),
            "num_q_heads": frozenset({64}),
            "batch_size_div_64": frozenset({False}),
            "page_size": frozenset({64}),
            "kv_lora_rank": frozenset({512}),
            "qk_rope_head_dim": frozenset({64}),
            "support_logit_cap": frozenset({False}),
            "return_lse": frozenset({False, True}),
        },
    )
    def gluon_mla_decode_bf16xbf16_gfx950_bh16_multiblock(*args, **kwargs):
        return _mla_decode_bf16xbf16_bh16_multiblock_impl(*args, **kwargs)

    @register_kernel(
        "attention",
        "mla_decode_with_kvcache",
        name="gluon_mla_decode_bf16xbf16_gfx950_bh64_small",
        solution="gluon",
        capability=CapabilityRequirement(
            min_arch_version=ArchVersion(9, 5),
            max_arch_version=ArchVersion(9, 5),
            vendors=frozenset({"amd"}),
        ),
        signatures=format_signatures(
            ("q", "kv_cache"),
            "dense",
            {torch.bfloat16},
        ),
        priority=Priority.SPECIALIZED,
        traits={
            "batch_size": frozenset({2, 4}),
            "q_len": frozenset({1}),
            "num_q_heads": frozenset({64}),
            "batch_size_div_64": frozenset({False}),
            "page_size": frozenset({64}),
            "kv_lora_rank": frozenset({512}),
            "qk_rope_head_dim": frozenset({64}),
            "support_logit_cap": frozenset({False}),
            "return_lse": frozenset({False, True}),
        },
    )
    def gluon_mla_decode_bf16xbf16_gfx950_bh64_small(*args, **kwargs):
        return _mla_decode_bf16xbf16_bh64_small_impl(*args, **kwargs)

    @register_kernel(
        "attention",
        "mla_decode_with_kvcache",
        name="gluon_mla_decode_bf16xfp8_gfx950_bh16bn128",
        solution="gluon",
        capability=CapabilityRequirement(
            min_arch_version=ArchVersion(9, 5),
            max_arch_version=ArchVersion(9, 5),
            vendors=frozenset({"amd"}),
        ),
        signatures=frozenset(
            format_signature(
                q=dense_tensor_format(q_dtype),
                kv_cache=dense_tensor_format(kv_dtype),
            )
            for q_dtype, kv_dtype in (
                (torch.bfloat16, torch.float8_e4m3fn),
                (torch.bfloat16, torch.float8_e5m2),
            )
        ),
        priority=Priority.SPECIALIZED,
        traits={
            "q_len": frozenset({1}),
            "num_q_heads": frozenset(range(1, 17)),
            "page_size": frozenset({64}),
            "kv_lora_rank": frozenset({512}),
            "qk_rope_head_dim": frozenset({64}),
            "support_logit_cap": frozenset({False}),
            "return_lse": frozenset({False, True}),
        },
    )
    def gluon_mla_decode_bf16xfp8_gfx950_bh16bn128(*args, **kwargs):
        return _mla_decode_bf16xfp8_impl(*args, **kwargs)

    @register_kernel(
        "attention",
        "mla_decode_with_kvcache",
        name="gluon_mla_decode_fp8xfp8_gfx950_bh16bn128",
        solution="gluon",
        capability=CapabilityRequirement(
            min_arch_version=ArchVersion(9, 5),
            max_arch_version=ArchVersion(9, 5),
            vendors=frozenset({"amd"}),
        ),
        signatures=frozenset(
            {
                format_signature(
                    q=dense_tensor_format(torch.float8_e4m3fn),
                    kv_cache=dense_tensor_format(torch.float8_e4m3fn),
                )
            }
        ),
        priority=Priority.SPECIALIZED,
        traits={
            "q_len": frozenset({1}),
            "num_q_heads": frozenset(range(1, 17)),
            "page_size": frozenset({64}),
            "kv_lora_rank": frozenset({512}),
            "qk_rope_head_dim": frozenset({64}),
            "support_logit_cap": frozenset({False}),
            "return_lse": frozenset({False, True}),
        },
    )
    def gluon_mla_decode_fp8xfp8_gfx950_bh16bn128(*args, **kwargs):
        return _mla_decode_fp8xfp8_impl(*args, **kwargs)

    register_kernel(
        "attention",
        "mla_decode_projected_value",
        name="gluon_mla_decode_projected_value_gfx950",
        solution="gluon",
        capability=CapabilityRequirement(
            min_arch_version=ArchVersion(9, 5),
            max_arch_version=ArchVersion(9, 5),
            vendors=frozenset({"amd"}),
        ),
        signatures=frozenset(
            {
                format_signature(
                    q=dense_tensor_format(torch.float8_e4m3fn),
                    kv_cache=dense_tensor_format(torch.float8_e4m3fn),
                    value_weight=dense_tensor_format(torch.bfloat16),
                    out=dense_tensor_format(torch.bfloat16),
                ),
                format_signature(
                    q=dense_tensor_format(torch.float8_e5m2),
                    kv_cache=dense_tensor_format(torch.float8_e5m2),
                    value_weight=dense_tensor_format(torch.bfloat16),
                    out=dense_tensor_format(torch.bfloat16),
                ),
            }
        ),
        priority=Priority.SPECIALIZED,
        traits={
            "batch_size": frozenset({1, 2, 4}),
            "q_len": frozenset({1}),
            "num_q_heads": frozenset({12, 16}),
            "page_size": frozenset({64}),
            "kv_lora_rank": frozenset({512}),
            "qk_rope_head_dim": frozenset({64}),
            "value_head_dim": frozenset({128}),
            "gate_kind": frozenset({"none", "sigmoid"}),
            "support_logit_cap": frozenset({False}),
        },
    )(_mla_decode_projected_value_impl)

    register_kernel(
        "attention",
        "mla_project_value",
        name="gluon_mla_project_value_gfx950",
        solution="gluon",
        capability=CapabilityRequirement(
            min_arch_version=ArchVersion(9, 5),
            max_arch_version=ArchVersion(9, 5),
            vendors=frozenset({"amd"}),
        ),
        signatures=frozenset(
            {
                format_signature(
                    attention=dense_tensor_format(torch.bfloat16),
                    weight=dense_tensor_format(torch.bfloat16),
                    out=dense_tensor_format(torch.bfloat16),
                )
            }
        ),
        priority=Priority.SPECIALIZED,
        traits={
            "batch_size": frozenset({1, 2, 4}),
            "num_heads": frozenset({12, 16}),
            "latent_dim": frozenset({512}),
            "value_dim": frozenset({128}),
            "gate_kind": frozenset({"none", "sigmoid"}),
            "inputs_contiguous": frozenset({True}),
        },
    )(_mla_project_value_impl)

    register_kernel(
        "attention",
        "mla_normalize_project_query",
        name="gluon_mla_normalize_project_query_gfx950",
        solution="gluon",
        capability=CapabilityRequirement(
            min_arch_version=ArchVersion(9, 5),
            max_arch_version=ArchVersion(9, 5),
            vendors=frozenset({"amd"}),
        ),
        signatures=frozenset(
            {
                format_signature(
                    query=dense_tensor_format(torch.bfloat16),
                    kv=dense_tensor_format(torch.bfloat16),
                    projection_weight=dense_tensor_format(torch.bfloat16),
                    out=dense_tensor_format(torch.bfloat16),
                ),
                format_signature(
                    query=dense_tensor_format(torch.bfloat16),
                    kv=dense_tensor_format(torch.bfloat16),
                    projection_weight=dense_tensor_format(torch.bfloat16),
                    out=dense_tensor_format(torch.bfloat16),
                    tail_out=dense_tensor_format(torch.bfloat16),
                ),
            }
        ),
        priority=Priority.SPECIALIZED,
        traits={
            "num_tokens": frozenset({1}),
            "query_width": frozenset({1536}),
            "kv_width": frozenset({512}),
            "output_width": frozenset({2304, 3072}),
            "output_prefix_width": frozenset({128, 2304, 3072}),
            "output_tail_width": frozenset({0, 64}),
            "split_output": frozenset({False, True}),
            "inputs_contiguous": frozenset({True}),
            "outputs_inner_contiguous": frozenset({True}),
        },
    )(_mla_normalize_project_query_impl)

    register_kernel(
        "attention",
        "mla_normalize_project_query",
        name="gluon_mla_normalize_project_query_gfx1250",
        solution="gluon",
        capability=CapabilityRequirement(
            min_arch_version=ArchVersion(12, 5),
            max_arch_version=ArchVersion(12, 5),
            vendors=frozenset({"amd"}),
        ),
        signatures=frozenset(
            {
                format_signature(
                    query=dense_tensor_format(torch.bfloat16),
                    kv=dense_tensor_format(torch.bfloat16),
                    projection_weight=dense_tensor_format(torch.bfloat16),
                    out=dense_tensor_format(torch.bfloat16),
                ),
                format_signature(
                    query=dense_tensor_format(torch.bfloat16),
                    kv=dense_tensor_format(torch.bfloat16),
                    projection_weight=dense_tensor_format(torch.bfloat16),
                    out=dense_tensor_format(torch.bfloat16),
                    tail_out=dense_tensor_format(torch.bfloat16),
                ),
            }
        ),
        priority=Priority.SPECIALIZED,
        traits={
            "num_tokens": frozenset({1}),
            "query_width": frozenset({1536}),
            "kv_width": frozenset({512}),
            "output_width": frozenset({2304, 3072}),
            "output_prefix_width": frozenset({128, 2304, 3072}),
            "output_tail_width": frozenset({0, 64}),
            "split_output": frozenset({False, True}),
            "inputs_contiguous": frozenset({True}),
            "outputs_inner_contiguous": frozenset({True}),
        },
    )(_mla_normalize_project_query_gfx1250_impl)

    @register_kernel(
        "attention",
        "mla_decode_with_kvcache",
        name="gluon_mla_decode_bf16xbf16_gfx950_bh64",
        solution="gluon",
        capability=CapabilityRequirement(
            min_arch_version=ArchVersion(9, 5),
            max_arch_version=ArchVersion(9, 5),
            vendors=frozenset({"amd"}),
        ),
        signatures=format_signatures(
            ("q", "kv_cache"),
            "dense",
            {torch.bfloat16},
        ),
        priority=Priority.SPECIALIZED,
        traits={
            "q_len": frozenset({1}),
            "num_q_heads": frozenset({64, 128}),
            "batch_size_div_64": frozenset({True}),
            "page_size": frozenset({64}),
            "kv_lora_rank": frozenset({512}),
            "qk_rope_head_dim": frozenset({64}),
            "support_logit_cap": frozenset({False}),
            "return_lse": frozenset({False, True}),
        },
    )
    def gluon_mla_decode_bf16xbf16_gfx950_bh64(*args, **kwargs):
        return _mla_decode_bf16xbf16_bh64_impl(*args, **kwargs)

    @register_kernel(
        "attention",
        "mla_decode_with_kvcache",
        name="gluon_mla_decode_gfx1250",
        solution="gluon",
        capability=CapabilityRequirement(
            min_arch_version=ArchVersion(12, 5),
            max_arch_version=ArchVersion(12, 5),
            vendors=frozenset({"amd"}),
        ),
        signatures=format_signatures(
            ("q", "kv_cache"),
            "dense",
            {
                torch.float16,
                torch.bfloat16,
                torch.float8_e4m3fn,
                torch.float8_e5m2,
            },
        ),
        priority=Priority.SPECIALIZED,
        traits={
            "q_len": frozenset({1}),
            "num_q_heads": frozenset(range(1, 129)),
            "page_size": frozenset({64}),
            "kv_lora_rank": frozenset({512}),
            "qk_rope_head_dim": frozenset({64}),
            "support_logit_cap": frozenset({False}),
            "return_lse": frozenset({False, True}),
        },
    )
    def gluon_mla_decode_gfx1250(*args, **kwargs):
        return _mla_decode_gfx1250_impl(*args, **kwargs)

    register_kernel(
        "attention",
        "mla_decode_projected_value",
        name="gluon_mla_decode_projected_value_gfx1250",
        solution="gluon",
        capability=CapabilityRequirement(
            min_arch_version=ArchVersion(12, 5),
            max_arch_version=ArchVersion(12, 5),
            vendors=frozenset({"amd"}),
        ),
        signatures=frozenset(
            {
                format_signature(
                    q=dense_tensor_format(torch.float8_e4m3fn),
                    kv_cache=dense_tensor_format(torch.float8_e4m3fn),
                    value_weight=dense_tensor_format(torch.bfloat16),
                    out=dense_tensor_format(torch.bfloat16),
                ),
                format_signature(
                    q=dense_tensor_format(torch.float8_e5m2),
                    kv_cache=dense_tensor_format(torch.float8_e5m2),
                    value_weight=dense_tensor_format(torch.bfloat16),
                    out=dense_tensor_format(torch.bfloat16),
                ),
            }
        ),
        priority=Priority.SPECIALIZED,
        traits={
            "batch_size": frozenset({1, 2, 4, 8}),
            "q_len": frozenset({1}),
            "num_q_heads": frozenset({12, 16}),
            "page_size": frozenset({64}),
            "kv_lora_rank": frozenset({512}),
            "qk_rope_head_dim": frozenset({64}),
            "value_head_dim": frozenset({128}),
            "gate_kind": frozenset({"none", "sigmoid"}),
            "support_logit_cap": frozenset({False}),
        },
    )(_mla_decode_projected_value_gfx1250_impl)

    register_kernel(
        "attention",
        "mla_project_value",
        name="gluon_mla_project_value_gfx1250",
        solution="gluon",
        capability=CapabilityRequirement(
            min_arch_version=ArchVersion(12, 5),
            max_arch_version=ArchVersion(12, 5),
            vendors=frozenset({"amd"}),
        ),
        signatures=frozenset(
            {
                format_signature(
                    attention=dense_tensor_format(torch.bfloat16),
                    weight=dense_tensor_format(torch.bfloat16),
                    out=dense_tensor_format(torch.bfloat16),
                )
            }
        ),
        priority=Priority.SPECIALIZED,
        traits={
            "batch_size": frozenset({1, 2, 4, 8}),
            "num_heads": frozenset({12, 16}),
            "latent_dim": frozenset({512}),
            "value_dim": frozenset({128}),
            "gate_kind": frozenset({"none", "sigmoid"}),
            "inputs_contiguous": frozenset({True}),
        },
    )(_mla_project_value_gfx1250_impl)

    @register_kernel(
        "attention",
        "mla_extend_with_kvcache",
        name="gluon_mla_extend_gfx1250",
        solution="gluon",
        capability=CapabilityRequirement(
            min_arch_version=ArchVersion(12, 5),
            max_arch_version=ArchVersion(12, 5),
            vendors=frozenset({"amd"}),
        ),
        signatures=format_signatures(
            ("q", "kv_cache"),
            "dense",
            {
                torch.float16,
                torch.bfloat16,
                torch.float8_e4m3fn,
                torch.float8_e5m2,
            },
        ),
        priority=Priority.SPECIALIZED,
        traits={
            "num_q_heads": frozenset(range(1, 129)),
            # Absorbed attention wins only for short cached extends; longer
            # queries should use expanded prefix replay.
            "max_seqlen_q": frozenset(range(1, 257)),
            "page_size": frozenset({64}),
            "qk_nope_head_dim": frozenset({128}),
            "kv_lora_rank": frozenset({512}),
            "qk_rope_head_dim": frozenset({64}),
            "is_causal": frozenset({True}),
            "support_logit_cap": frozenset({False}),
            "return_lse": frozenset({False}),
        },
    )
    def gluon_mla_extend_gfx1250(*args, **kwargs):
        return _mla_extend_gfx1250_impl(*args, **kwargs)

    @register_kernel(
        "attention",
        "mla_prefill",
        name="gluon_mla_prefill_gfx950",
        solution="gluon",
        capability=CapabilityRequirement(
            min_arch_version=ArchVersion(9, 5),
            max_arch_version=ArchVersion(9, 5),
            vendors=frozenset({"amd"}),
        ),
        signatures=format_signatures(
            ("q", "k", "v"),
            "dense",
            {
                torch.float16,
                torch.bfloat16,
                torch.float8_e4m3fn,
                torch.float8_e5m2,
            },
        ),
        priority=Priority.SPECIALIZED,
        traits={
            "qk_head_dim": frozenset({192}),
            "v_head_dim": frozenset({128}),
            "is_causal": frozenset({False, True}),
            "support_logit_cap": frozenset({False}),
            "return_lse": frozenset({False, True}),
        },
    )
    def gluon_mla_prefill_gfx950(*args, **kwargs):
        return _mla_prefill_gfx950_impl(*args, **kwargs)

    @register_kernel(
        "attention",
        "mla_prefill",
        name="gluon_mla_prefill_gfx1250",
        solution="gluon",
        capability=CapabilityRequirement(
            min_arch_version=ArchVersion(12, 5),
            max_arch_version=ArchVersion(12, 5),
            vendors=frozenset({"amd"}),
        ),
        signatures=format_signatures(
            ("q", "k", "v"),
            "dense",
            {
                torch.float16,
                torch.bfloat16,
                torch.float8_e4m3fn,
                torch.float8_e5m2,
            },
        ),
        priority=Priority.SPECIALIZED,
        traits={
            "qk_head_dim": frozenset({192}),
            "v_head_dim": frozenset({128}),
            "is_causal": frozenset({False, True}),
            "support_logit_cap": frozenset({False}),
            "return_lse": frozenset({False, True}),
        },
    )
    def gluon_mla_prefill_gfx1250(*args, **kwargs):
        return _mla_prefill_gfx1250_impl(*args, **kwargs)
