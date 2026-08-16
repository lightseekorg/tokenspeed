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

"""Registration shims for AMD Gluon attention kernels."""

from __future__ import annotations

import torch
from tokenspeed_kernel.ops.attention.kda_utils import KdaPrefillResult
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
    _DSA_FULL_TOPK_WIDTHS = frozenset({512, 1024, 2048})
    _DSA_PREFILL_TOPK_WIDTHS = _DSA_FULL_TOPK_WIDTHS

    from tokenspeed_kernel_amd.ops.gfx950.attention.dsa.attention import (
        gluon_dsa_decode_gfx950 as _dsa_decode_impl,
    )
    from tokenspeed_kernel_amd.ops.gfx950.attention.dsa.attention import (
        gluon_dsa_prefill_gfx950 as _dsa_prefill_impl,
    )
    from tokenspeed_kernel_amd.ops.gfx950.attention.dsa.sparse_mla import (
        gluon_dsa_decode_topk_fp8_gfx950 as _dsa_decode_topk_impl,
    )
    from tokenspeed_kernel_amd.ops.gfx950.attention.dsa.sparse_mla import (
        gluon_dsa_prefill_topk_fp8_gfx950 as _dsa_prefill_topk_impl,
    )
    from tokenspeed_kernel_amd.ops.gfx950.attention.kda.decode import (
        gluon_kda_fused_decode_gfx950 as _kda_fused_decode_impl,
    )
    from tokenspeed_kernel_amd.ops.gfx950.attention.kda.decode import (
        gluon_kda_recurrent_decode_gfx950 as _kda_decode_impl,
    )
    from tokenspeed_kernel_amd.ops.gfx950.attention.kda.prefill import (
        gluon_kda_paged_prefill_gfx950 as _kda_prefill_impl,
    )
    from tokenspeed_kernel_amd.ops.gfx950.attention.mha.decode import (
        gluon_mha_decode_gfx950 as _decode_impl,
    )
    from tokenspeed_kernel_amd.ops.gfx950.attention.mha.extend import (
        gluon_mha_extend_gfx950 as _extend_impl,
    )
    from tokenspeed_kernel_amd.ops.gfx950.attention.mha.prefill import (
        gluon_mha_prefill_gfx950 as _prefill_impl,
    )
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
    from tokenspeed_kernel_amd.ops.gfx950.attention.rmha.decode import (
        gluon_rel_mha_decode_gfx950 as _rel_decode_impl,
    )
    from tokenspeed_kernel_amd.ops.gfx950.attention.rmha.extend import (
        gluon_rel_mha_extend_gfx950 as _rel_extend_impl,
    )
    from tokenspeed_kernel_amd.ops.gfx950.attention.rmha.prefill import (
        gluon_rel_mha_prefill_gfx950 as _rel_prefill_impl,
    )
    from tokenspeed_kernel_amd.ops.gfx1250.attention.dsa.attention import (
        gluon_dsa_decode_gfx1250 as _dsa_decode_gfx1250_impl,
    )
    from tokenspeed_kernel_amd.ops.gfx1250.attention.dsa.attention import (
        gluon_dsa_prefill_gfx1250 as _dsa_prefill_gfx1250_impl,
    )
    from tokenspeed_kernel_amd.ops.gfx1250.attention.dsa.sparse_mla import (
        gluon_dsa_decode_topk_fp8_gfx1250 as _dsa_decode_topk_gfx1250_impl,
    )
    from tokenspeed_kernel_amd.ops.gfx1250.attention.dsa.sparse_mla import (
        gluon_dsa_prefill_topk_fp8_gfx1250 as _dsa_prefill_topk_gfx1250_impl,
    )
    from tokenspeed_kernel_amd.ops.gfx1250.attention.kda.decode import (
        gluon_kda_recurrent_decode_gfx1250 as _kda_decode_gfx1250_impl,
    )
    from tokenspeed_kernel_amd.ops.gfx1250.attention.kda.prefill import (
        gluon_kda_paged_prefill_gfx1250 as _kda_prefill_gfx1250_impl,
    )
    from tokenspeed_kernel_amd.ops.gfx1250.attention.mha.decode import (
        gluon_mha_decode_gfx1250 as _decode_gfx1250_impl,
    )
    from tokenspeed_kernel_amd.ops.gfx1250.attention.mha.prefill import (
        gluon_mha_prefill_gfx1250 as _prefill_gfx1250_impl,
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
        "kda_paged_prefill",
        name="gluon_kda_paged_prefill_gfx950",
        solution="gluon",
        capability=CapabilityRequirement(
            min_arch_version=ArchVersion(9, 5),
            max_arch_version=ArchVersion(9, 5),
            vendors=frozenset({"amd"}),
        ),
        signatures=format_signatures(
            ("q", "k", "v"),
            "dense",
            {torch.float16, torch.bfloat16},
        ),
        priority=Priority.SPECIALIZED,
        tags={"amd", "gfx950", "paged_cache"},
    )
    def gluon_kda_paged_prefill_gfx950(**kwargs) -> KdaPrefillResult:
        """Run specialized gfx950 KDA prefill with canonical K-major state."""
        output, final_state = _kda_prefill_impl(**kwargs)
        return KdaPrefillResult(out=output, final_state=final_state)

    @register_kernel(
        "attention",
        "kda_paged_prefill",
        name="gluon_kda_paged_prefill_gfx1250",
        solution="gluon",
        capability=CapabilityRequirement(
            min_arch_version=ArchVersion(12, 5),
            max_arch_version=ArchVersion(12, 5),
            vendors=frozenset({"amd"}),
        ),
        signatures=format_signatures(
            ("q", "k", "v"),
            "dense",
            {torch.float16, torch.bfloat16},
        ),
        priority=Priority.SPECIALIZED,
        tags={"amd", "gfx1250", "paged_cache"},
    )
    def gluon_kda_paged_prefill_gfx1250(**kwargs) -> KdaPrefillResult:
        """Run specialized gfx1250 KDA prefill with canonical K-major state."""
        output, final_state = _kda_prefill_gfx1250_impl(**kwargs)
        return KdaPrefillResult(out=output, final_state=final_state)

    @register_kernel(
        "attention",
        "kda_paged_decode",
        name="gluon_kda_paged_decode_gfx950",
        solution="gluon",
        capability=CapabilityRequirement(
            min_arch_version=ArchVersion(9, 5),
            max_arch_version=ArchVersion(9, 5),
            vendors=frozenset({"amd"}),
        ),
        signatures=format_signatures(
            ("q", "k", "v"),
            "dense",
            {torch.float16, torch.bfloat16},
        ),
        priority=Priority.SPECIALIZED,
        traits={
            "indexed_state": frozenset({True}),
            "single_token": frozenset({True}),
        },
        tags={"amd", "gfx950", "paged_cache", "cuda_graph"},
    )
    def gluon_kda_paged_decode_gfx950(**kwargs):
        """Run specialized gfx950 KDA decode against the canonical K-major pool."""
        return _kda_decode_impl(**kwargs)

    @register_kernel(
        "attention",
        "kda_fused_paged_decode",
        name="gluon_kda_fused_paged_decode_gfx950",
        solution="gluon",
        capability=CapabilityRequirement(
            min_arch_version=ArchVersion(9, 5),
            max_arch_version=ArchVersion(9, 5),
            vendors=frozenset({"amd"}),
        ),
        signatures=format_signatures(
            ("q", "k", "v"),
            "dense",
            {torch.bfloat16},
        ),
        priority=Priority.SPECIALIZED,
        traits={
            "paged_state": frozenset({True}),
            "fused_output_norm": frozenset({True}),
            "num_heads": frozenset({12}),
            "head_dim": frozenset({128}),
            "conv_kernel_size": frozenset({4}),
        },
        tags={"amd", "gfx950", "paged_cache", "cuda_graph", "fusion"},
    )
    def gluon_kda_fused_paged_decode_gfx950(
        mixed_qkv: torch.Tensor,
        conv_weights: torch.Tensor,
        conv_states: torch.Tensor,
        f_a_out: torch.Tensor,
        f_b_weight: torch.Tensor,
        beta_logits: torch.Tensor,
        A_log: torch.Tensor,
        dt_bias: torch.Tensor,
        *,
        state_pool: torch.Tensor,
        read_indices: torch.Tensor,
        write_indices: torch.Tensor,
        num_heads: int,
        head_dim: int,
        cu_seqlens: torch.Tensor,
        lower_bound: float | None,
        output_gate: torch.Tensor | None,
        norm_weight: torch.Tensor | None,
        norm_eps: float | None,
    ):
        """Run the decay projection and fused gfx950 KDA decode epilogue."""
        if output_gate is None or norm_weight is None or norm_eps is None:
            raise ValueError("gfx950 fused KDA decode requires output normalization")
        raw_g = torch.nn.functional.linear(f_a_out, f_b_weight)
        return _kda_fused_decode_impl(
            mixed_qkv=mixed_qkv,
            conv_weights=conv_weights,
            conv_states=conv_states,
            raw_g=raw_g,
            beta_logits=beta_logits,
            A_log=A_log,
            dt_bias=dt_bias,
            output_gate=output_gate,
            norm_weight=norm_weight,
            norm_eps=norm_eps,
            state_pool=state_pool,
            read_indices=read_indices,
            write_indices=write_indices,
            num_heads=num_heads,
            head_dim=head_dim,
            cu_seqlens=cu_seqlens,
            lower_bound=lower_bound,
        )

    @register_kernel(
        "attention",
        "kda_paged_decode",
        name="gluon_kda_paged_decode_gfx1250",
        solution="gluon",
        capability=CapabilityRequirement(
            min_arch_version=ArchVersion(12, 5),
            max_arch_version=ArchVersion(12, 5),
            vendors=frozenset({"amd"}),
        ),
        signatures=format_signatures(
            ("q", "k", "v"),
            "dense",
            {torch.float16, torch.bfloat16},
        ),
        priority=Priority.SPECIALIZED,
        traits={
            "indexed_state": frozenset({True}),
            "single_token": frozenset({True}),
        },
        tags={"amd", "gfx1250", "paged_cache", "cuda_graph"},
    )
    def gluon_kda_paged_decode_gfx1250(**kwargs):
        """Run specialized gfx1250 KDA decode against the canonical K-major pool."""
        return _kda_decode_gfx1250_impl(**kwargs)

    @register_kernel(
        "attention",
        "mha_decode_with_kvcache",
        name="gluon_mha_decode_gfx950",
        solution="gluon",
        capability=CapabilityRequirement(
            min_arch_version=ArchVersion(9, 5),
            max_arch_version=ArchVersion(9, 5),
            vendors=frozenset({"amd"}),
        ),
        signatures=format_signatures(
            ("q", "k_cache", "v_cache"),
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
            "head_dim": frozenset({64, 128}),
            "page_size": frozenset({64}),
            "sliding_window": frozenset({False, True}),
            "support_sinks": frozenset({False, True}),
            "support_logit_cap": frozenset({False}),
            "return_lse": frozenset({False}),
        },
    )
    def gluon_mha_decode_gfx950(*args, **kwargs):
        return _decode_impl(*args, **kwargs)

    @register_kernel(
        "attention",
        "mha_decode_with_kvcache",
        name="gluon_mha_decode_gfx1250",
        solution="gluon",
        capability=CapabilityRequirement(
            min_arch_version=ArchVersion(12, 5),
            max_arch_version=ArchVersion(12, 5),
            vendors=frozenset({"amd"}),
        ),
        signatures=format_signatures(
            ("q", "k_cache", "v_cache"),
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
            "head_dim": frozenset({64, 128}),
            "page_size": frozenset({64, 128}),
            "sliding_window": frozenset({False}),
            "support_sinks": frozenset({False}),
            "support_logit_cap": frozenset({False}),
            "return_lse": frozenset({False}),
        },
    )
    def gluon_mha_decode_gfx1250(*args, **kwargs):
        return _decode_gfx1250_impl(*args, **kwargs)

    @register_kernel(
        "attention",
        "mha_prefill",
        name="gluon_mha_prefill_gfx950",
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
            "head_dim": frozenset({64, 128}),
            "sliding_window": frozenset({False, True}),
            "support_sinks": frozenset({False, True}),
            "support_logit_cap": frozenset({False}),
            "return_lse": frozenset({False, True}),
        },
    )
    def gluon_mha_prefill_gfx950(*args, **kwargs):
        return _prefill_impl(*args, **kwargs)

    @register_kernel(
        "attention",
        "mha_prefill",
        name="gluon_mha_prefill_gfx1250",
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
            "head_dim": frozenset({64, 128}),
            "sliding_window": frozenset({False, True}),
            "support_sinks": frozenset({False, True}),
            "support_logit_cap": frozenset({False}),
            "return_lse": frozenset({False, True}),
        },
    )
    def gluon_mha_prefill_gfx1250(*args, **kwargs):
        return _prefill_gfx1250_impl(*args, **kwargs)

    @register_kernel(
        "attention",
        "mha_extend_with_kvcache",
        name="gluon_mha_extend_gfx950",
        solution="gluon",
        capability=CapabilityRequirement(
            min_arch_version=ArchVersion(9, 5),
            max_arch_version=ArchVersion(9, 5),
            vendors=frozenset({"amd"}),
        ),
        signatures=format_signatures(
            ("q", "k_cache", "v_cache"),
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
            "head_dim": frozenset({64, 128}),
            "page_size": frozenset({64}),
            "is_causal": frozenset({False, True}),
            "sliding_window": frozenset({False, True}),
            "support_sinks": frozenset({False, True}),
            "support_logit_cap": frozenset({False}),
            "return_lse": frozenset({False, True}),
        },
    )
    def gluon_mha_extend_gfx950(*args, **kwargs):
        return _extend_impl(*args, **kwargs)

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
            "batch_size": frozenset({1}),
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
            "batch_size": frozenset({1}),
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
            "batch_size": frozenset({1}),
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
            "batch_size": frozenset({1}),
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

    @register_kernel(
        "attention",
        "dsa_decode_topk",
        name="gluon_dsa_decode_topk_fp8_gfx950",
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
                    weights=dense_tensor_format(torch.float32),
                ),
                format_signature(
                    q=dense_tensor_format(torch.bfloat16),
                    weights=dense_tensor_format(torch.bfloat16),
                ),
            }
        ),
        priority=Priority.SPECIALIZED,
        traits={
            "head_dim": frozenset({128}),
            "topk": frozenset({512, 1024, 2048}),
            "page_size": frozenset({64}),
            "q_len_per_req": frozenset({1, 2, 3, 4, 5, 6}),
            "index_k_format": frozenset({"fp8_scaled"}),
        },
    )
    def gluon_dsa_decode_topk_fp8_gfx950(*args, **kwargs):
        return _dsa_decode_topk_impl(*args, **kwargs)

    @register_kernel(
        "attention",
        "dsa_prefill_topk",
        name="gluon_dsa_prefill_topk_fp8_gfx950",
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
                    weights=dense_tensor_format(torch.float32),
                ),
                format_signature(
                    q=dense_tensor_format(torch.bfloat16),
                    weights=dense_tensor_format(torch.bfloat16),
                ),
            }
        ),
        priority=Priority.SPECIALIZED,
        traits={
            "head_dim": frozenset({128}),
            "topk": frozenset({512, 1024, 2048}),
            "page_size": frozenset({64}),
            "index_k_format": frozenset({"fp8_scaled"}),
        },
    )
    def gluon_dsa_prefill_topk_fp8_gfx950(*args, **kwargs):
        return _dsa_prefill_topk_impl(*args, **kwargs)

    @register_kernel(
        "attention",
        "dsa_decode",
        name="gluon_dsa_decode_gfx950",
        solution="gluon",
        capability=CapabilityRequirement(
            min_arch_version=ArchVersion(9, 5),
            max_arch_version=ArchVersion(9, 5),
            vendors=frozenset({"amd"}),
        ),
        signatures=frozenset(
            {
                format_signature(q=dense_tensor_format(torch.bfloat16)),
                format_signature(q=dense_tensor_format(torch.float8_e4m3fn)),
                format_signature(q=dense_tensor_format(torch.float8_e5m2)),
            }
        ),
        priority=Priority.SPECIALIZED,
        traits={
            "page_size": frozenset({64}),
            "q_len_per_req": frozenset({1, 2, 3, 4, 5, 6}),
            "qk_nope_head_dim": frozenset({128, 192}),
            "kv_lora_rank": frozenset({128, 512}),
            "qk_rope_head_dim": frozenset({64}),
            "topk": _DSA_FULL_TOPK_WIDTHS,
            "kv_cache_available": frozenset({False, True}),
            "sparse_kv_cache_available": frozenset({False, True}),
            "topk_layout": frozenset({"global_slots"}),
            "support_logit_cap": frozenset({False}),
            "return_lse": frozenset({False}),
        },
    )
    def gluon_dsa_decode_gfx950(*args, **kwargs):
        return _dsa_decode_impl(*args, **kwargs)

    @register_kernel(
        "attention",
        "dsa_prefill",
        name="gluon_dsa_prefill_gfx950",
        solution="gluon",
        capability=CapabilityRequirement(
            min_arch_version=ArchVersion(9, 5),
            max_arch_version=ArchVersion(9, 5),
            vendors=frozenset({"amd"}),
        ),
        signatures=frozenset(
            {
                format_signature(q=dense_tensor_format(torch.bfloat16)),
            }
        ),
        priority=Priority.SPECIALIZED,
        traits={
            "page_size": frozenset({64}),
            "q_len_per_req": frozenset({1}),
            "qk_nope_head_dim": frozenset({128, 192}),
            "kv_lora_rank": frozenset({128, 512}),
            "qk_rope_head_dim": frozenset({64}),
            "topk": _DSA_PREFILL_TOPK_WIDTHS,
            "kv_cache_available": frozenset({False, True}),
            "sparse_kv_cache_available": frozenset({False, True}),
            "topk_layout": frozenset({"global_slots"}),
            "support_logit_cap": frozenset({False}),
            "return_lse": frozenset({False}),
        },
    )
    def gluon_dsa_prefill_gfx950(*args, **kwargs):
        return _dsa_prefill_impl(*args, **kwargs)

    @register_kernel(
        "attention",
        "dsa_prefill",
        name="gluon_dsa_prefill_fp8_dense_gfx950",
        solution="gluon",
        capability=CapabilityRequirement(
            min_arch_version=ArchVersion(9, 5),
            max_arch_version=ArchVersion(9, 5),
            vendors=frozenset({"amd"}),
        ),
        signatures=frozenset(
            {
                format_signature(q=dense_tensor_format(torch.float8_e4m3fn)),
                format_signature(q=dense_tensor_format(torch.float8_e5m2)),
            }
        ),
        priority=Priority.SPECIALIZED,
        traits={
            "page_size": frozenset({64}),
            "q_len_per_req": frozenset({1}),
            "qk_nope_head_dim": frozenset({128, 192}),
            "kv_lora_rank": frozenset({512}),
            "qk_rope_head_dim": frozenset({64}),
            "topk": _DSA_PREFILL_TOPK_WIDTHS,
            "kv_cache_available": frozenset({True}),
            "sparse_kv_cache_available": frozenset({False}),
            "topk_layout": frozenset({"global_slots"}),
            "support_logit_cap": frozenset({False}),
            "return_lse": frozenset({False}),
        },
    )
    def gluon_dsa_prefill_fp8_dense_gfx950(*args, **kwargs):
        return _dsa_prefill_impl(*args, **kwargs)

    @register_kernel(
        "attention",
        "dsa_decode_topk",
        name="gluon_dsa_decode_topk_fp8_gfx1250",
        solution="gluon",
        capability=CapabilityRequirement(
            min_arch_version=ArchVersion(12, 5),
            max_arch_version=ArchVersion(12, 5),
            vendors=frozenset({"amd"}),
        ),
        signatures=frozenset(
            {
                format_signature(
                    q=dense_tensor_format(torch.bfloat16),
                    weights=dense_tensor_format(torch.float32),
                ),
                format_signature(
                    q=dense_tensor_format(torch.bfloat16),
                    weights=dense_tensor_format(torch.bfloat16),
                ),
            }
        ),
        priority=Priority.SPECIALIZED,
        traits={
            "head_dim": frozenset({128}),
            "topk": _DSA_FULL_TOPK_WIDTHS,
            "page_size": frozenset({64}),
            "q_len_per_req": frozenset({1, 2, 3, 4, 5, 6}),
            "index_k_format": frozenset({"fp8_scaled"}),
        },
        tags={"amd", "gfx1250"},
    )
    def gluon_dsa_decode_topk_fp8_gfx1250(*args, **kwargs):
        return _dsa_decode_topk_gfx1250_impl(*args, **kwargs)

    @register_kernel(
        "attention",
        "dsa_prefill_topk",
        name="gluon_dsa_prefill_topk_fp8_gfx1250",
        solution="gluon",
        capability=CapabilityRequirement(
            min_arch_version=ArchVersion(12, 5),
            max_arch_version=ArchVersion(12, 5),
            vendors=frozenset({"amd"}),
        ),
        signatures=frozenset(
            {
                format_signature(
                    q=dense_tensor_format(torch.bfloat16),
                    weights=dense_tensor_format(torch.float32),
                ),
                format_signature(
                    q=dense_tensor_format(torch.bfloat16),
                    weights=dense_tensor_format(torch.bfloat16),
                ),
            }
        ),
        priority=Priority.SPECIALIZED,
        traits={
            "head_dim": frozenset({128}),
            "topk": _DSA_PREFILL_TOPK_WIDTHS,
            "page_size": frozenset({64}),
            "index_k_format": frozenset({"fp8_scaled"}),
        },
        tags={"amd", "gfx1250"},
    )
    def gluon_dsa_prefill_topk_fp8_gfx1250(*args, **kwargs):
        return _dsa_prefill_topk_gfx1250_impl(*args, **kwargs)

    @register_kernel(
        "attention",
        "dsa_decode",
        name="gluon_dsa_decode_gfx1250",
        solution="gluon",
        capability=CapabilityRequirement(
            min_arch_version=ArchVersion(12, 5),
            max_arch_version=ArchVersion(12, 5),
            vendors=frozenset({"amd"}),
        ),
        signatures=frozenset(
            {
                format_signature(q=dense_tensor_format(torch.bfloat16)),
                format_signature(q=dense_tensor_format(torch.float8_e4m3fn)),
                format_signature(q=dense_tensor_format(torch.float8_e5m2)),
            }
        ),
        priority=Priority.SPECIALIZED,
        traits={
            "page_size": frozenset({64}),
            "q_len_per_req": frozenset({1, 2, 3, 4, 5, 6}),
            "qk_nope_head_dim": frozenset({128, 192}),
            "kv_lora_rank": frozenset({128, 512}),
            "qk_rope_head_dim": frozenset({64}),
            "topk": _DSA_FULL_TOPK_WIDTHS,
            "kv_cache_available": frozenset({False, True}),
            "sparse_kv_cache_available": frozenset({False, True}),
            "topk_layout": frozenset({"global_slots"}),
            "support_logit_cap": frozenset({False}),
            "return_lse": frozenset({False}),
        },
        tags={"amd", "gfx1250"},
    )
    def gluon_dsa_decode_gfx1250(*args, **kwargs):
        return _dsa_decode_gfx1250_impl(*args, **kwargs)

    @register_kernel(
        "attention",
        "dsa_prefill",
        name="gluon_dsa_prefill_gfx1250",
        solution="gluon",
        capability=CapabilityRequirement(
            min_arch_version=ArchVersion(12, 5),
            max_arch_version=ArchVersion(12, 5),
            vendors=frozenset({"amd"}),
        ),
        signatures=frozenset(
            {
                format_signature(q=dense_tensor_format(torch.bfloat16)),
            }
        ),
        priority=Priority.SPECIALIZED,
        traits={
            "page_size": frozenset({64}),
            "q_len_per_req": frozenset({1}),
            "qk_nope_head_dim": frozenset({128, 192}),
            "kv_lora_rank": frozenset({128, 512}),
            "qk_rope_head_dim": frozenset({64}),
            "topk": _DSA_PREFILL_TOPK_WIDTHS,
            "kv_cache_available": frozenset({False, True}),
            "sparse_kv_cache_available": frozenset({False, True}),
            "topk_layout": frozenset({"global_slots"}),
            "support_logit_cap": frozenset({False}),
            "return_lse": frozenset({False}),
        },
        tags={"amd", "gfx1250"},
    )
    def gluon_dsa_prefill_gfx1250(*args, **kwargs):
        return _dsa_prefill_gfx1250_impl(*args, **kwargs)

    @register_kernel(
        "attention",
        "dsa_prefill",
        name="gluon_dsa_prefill_fp8_dense_gfx1250",
        solution="gluon",
        capability=CapabilityRequirement(
            min_arch_version=ArchVersion(12, 5),
            max_arch_version=ArchVersion(12, 5),
            vendors=frozenset({"amd"}),
        ),
        signatures=frozenset(
            {
                format_signature(q=dense_tensor_format(torch.float8_e4m3fn)),
                format_signature(q=dense_tensor_format(torch.float8_e5m2)),
            }
        ),
        priority=Priority.SPECIALIZED,
        traits={
            "page_size": frozenset({64}),
            "q_len_per_req": frozenset({1}),
            "qk_nope_head_dim": frozenset({128, 192}),
            "kv_lora_rank": frozenset({512}),
            "qk_rope_head_dim": frozenset({64}),
            "topk": _DSA_PREFILL_TOPK_WIDTHS,
            "kv_cache_available": frozenset({True}),
            "sparse_kv_cache_available": frozenset({False}),
            "topk_layout": frozenset({"global_slots"}),
            "support_logit_cap": frozenset({False}),
            "return_lse": frozenset({False}),
        },
        tags={"amd", "gfx1250"},
    )
    def gluon_dsa_prefill_fp8_dense_gfx1250(*args, **kwargs):
        return _dsa_prefill_gfx1250_impl(*args, **kwargs)

    @register_kernel(
        "attention",
        "rel_mha_prefill",
        name="gluon_rel_mha_prefill_gfx950",
        solution="gluon",
        capability=CapabilityRequirement(
            min_arch_version=ArchVersion(9, 5),
            max_arch_version=ArchVersion(9, 5),
            vendors=frozenset({"amd"}),
        ),
        signatures=format_signatures(("q", "k", "v"), "dense", {torch.bfloat16}),
        priority=Priority.SPECIALIZED,
        traits={
            "head_dim": frozenset({64, 128}),
            "sliding_window": frozenset({False, True}),
            "return_lse": frozenset({False, True}),
        },
    )
    def gluon_rel_mha_prefill_gfx950(*args, **kwargs):
        kwargs.pop("enable_pdl", None)
        tau = kwargs.pop("tau", None)
        if tau is not None:
            # No fused per-row logit scale in the gluon backend; fold tau
            # into q and the rel bias: tau*(scale*qk + rel).
            kwargs["q"] = kwargs["q"] * tau[:, None, None].to(kwargs["q"].dtype)
            kwargs["rel_logits"] = kwargs["rel_logits"] * tau[:, None, None].to(
                kwargs["rel_logits"].dtype
            )
        return _rel_prefill_impl(*args, **kwargs)

    @register_kernel(
        "attention",
        "rel_mha_extend_with_kvcache",
        name="gluon_rel_mha_extend_gfx950",
        solution="gluon",
        capability=CapabilityRequirement(
            min_arch_version=ArchVersion(9, 5),
            max_arch_version=ArchVersion(9, 5),
            vendors=frozenset({"amd"}),
        ),
        signatures=format_signatures(
            ("q", "k_cache", "v_cache"), "dense", {torch.bfloat16}
        ),
        priority=Priority.SPECIALIZED,
        traits={
            "head_dim": frozenset({64, 128}),
            "page_size": frozenset({64, 128, 256}),
            "sliding_window": frozenset({False, True}),
            "return_lse": frozenset({False, True}),
        },
    )
    def gluon_rel_mha_extend_gfx950(*args, **kwargs):
        kwargs.pop("enable_pdl", None)
        tau = kwargs.pop("tau", None)
        if tau is not None:
            # No fused per-row logit scale in the gluon backend; fold tau
            # into q and the rel bias: tau*(scale*qk + rel).
            kwargs["q"] = kwargs["q"] * tau[:, None, None].to(kwargs["q"].dtype)
            kwargs["rel_logits"] = kwargs["rel_logits"] * tau[:, None, None].to(
                kwargs["rel_logits"].dtype
            )
        return _rel_extend_impl(*args, **kwargs)

    @register_kernel(
        "attention",
        "rel_mha_decode_with_kvcache",
        name="gluon_rel_mha_decode_gfx950",
        solution="gluon",
        capability=CapabilityRequirement(
            min_arch_version=ArchVersion(9, 5),
            max_arch_version=ArchVersion(9, 5),
            vendors=frozenset({"amd"}),
        ),
        signatures=format_signatures(
            ("q", "k_cache", "v_cache"), "dense", {torch.bfloat16}
        ),
        priority=Priority.SPECIALIZED,
        traits={
            "head_dim": frozenset({64, 128}),
            "page_size": frozenset({64, 128, 256}),
            "sliding_window": frozenset({False, True}),
            "return_lse": frozenset({False}),
        },
    )
    def gluon_rel_mha_decode_gfx950(*args, **kwargs):
        kwargs.pop("enable_pdl", None)
        tau = kwargs.pop("tau", None)
        if tau is not None:
            # No fused per-row logit scale in the gluon backend; fold tau
            # into q and the rel bias: tau*(scale*qk + rel).
            kwargs["q"] = kwargs["q"] * tau[:, None, None].to(kwargs["q"].dtype)
            kwargs["rel_logits"] = kwargs["rel_logits"] * tau[:, None, None].to(
                kwargs["rel_logits"].dtype
            )
        return _rel_decode_impl(*args, **kwargs)
