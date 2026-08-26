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
        gluon_dsa_decode_topk_standard_gfx950 as _dsa_decode_topk_standard_impl,
    )
    from tokenspeed_kernel_amd.ops.gfx950.attention.dsa.sparse_mla import (
        gluon_dsa_prefill_topk_fp8_gfx950 as _dsa_prefill_topk_impl,
    )
    from tokenspeed_kernel_amd.ops.gfx950.attention.dsa.sparse_mla import (
        gluon_dsa_prefill_topk_standard_gfx950 as _dsa_prefill_topk_standard_impl,
    )
    from tokenspeed_kernel_amd.ops.gfx950.attention.dsv4 import (
        gluon_dsv4_indexer_decode_topk_mxfp4_gfx950 as _dsv4_indexer_decode_topk_impl,
    )
    from tokenspeed_kernel_amd.ops.gfx950.attention.dsv4 import (
        gluon_dsv4_indexer_prefill_topk_mxfp4_gfx950 as _dsv4_indexer_prefill_topk_impl,
    )
    from tokenspeed_kernel_amd.ops.gfx950.attention.dsv4 import (
        gluon_dsv4_paged_selected_attention_split_gfx950 as _dsv4_paged_selected_attention_split_impl,
    )
    from tokenspeed_kernel_amd.ops.gfx950.attention.dsv4 import (
        gluon_dsv4_plan_gfx950 as _dsv4_plan_impl,
    )
    from tokenspeed_kernel_amd.ops.gfx950.attention.dsv4 import (
        gluon_dsv4_selected_attention_gfx950 as _dsv4_selected_attention_impl,
    )
    from tokenspeed_kernel_amd.ops.gfx950.attention.kda.decode import (
        gluon_kda_fused_decode_gfx950 as _kda_fused_decode_impl,
    )
    from tokenspeed_kernel_amd.ops.gfx950.attention.kda.decode import (
        gluon_kda_fused_replay_gfx950 as _kda_fused_replay_impl,
    )
    from tokenspeed_kernel_amd.ops.gfx950.attention.kda.decode import (
        gluon_kda_fused_verify_gfx950 as _kda_fused_verify_impl,
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
        gluon_kda_fused_decode_gfx1250 as _kda_fused_decode_gfx1250_impl,
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

    _DSV4_MXFP4_SIGNATURE = format_signature(
        q=dense_tensor_format(torch.uint8),
        weights=dense_tensor_format(torch.float32),
        index_k_cache=dense_tensor_format(torch.uint8),
    )
    _DSV4_MXFP4_TRAITS = {
        "index_heads": frozenset({32, 64}),
        "head_dim": frozenset({128}),
        "topk": frozenset({512, 1024, 2048}),
        "page_size": frozenset({64}),
        "index_k_format": frozenset({"mxfp4"}),
    }

    @register_kernel(
        "attention",
        "dsv4_indexer_prefill_topk",
        name="gluon_dsv4_indexer_prefill_topk_mxfp4_gfx950",
        solution="gluon",
        capability=CapabilityRequirement(
            min_arch_version=ArchVersion(9, 5),
            max_arch_version=ArchVersion(9, 5),
            vendors=frozenset({"amd"}),
        ),
        signatures=frozenset({_DSV4_MXFP4_SIGNATURE}),
        traits=_DSV4_MXFP4_TRAITS,
        priority=Priority.SPECIALIZED,
        tags={"amd", "gfx950", "mxfp4", "sparse", "radix_topk"},
    )
    def gluon_dsv4_indexer_prefill_topk_mxfp4_gfx950(*args, **kwargs):
        return _dsv4_indexer_prefill_topk_impl(*args, **kwargs)

    @register_kernel(
        "attention",
        "dsv4_indexer_decode_topk",
        name="gluon_dsv4_indexer_decode_topk_mxfp4_gfx950",
        solution="gluon",
        capability=CapabilityRequirement(
            min_arch_version=ArchVersion(9, 5),
            max_arch_version=ArchVersion(9, 5),
            vendors=frozenset({"amd"}),
        ),
        signatures=frozenset({_DSV4_MXFP4_SIGNATURE}),
        traits=_DSV4_MXFP4_TRAITS,
        priority=Priority.SPECIALIZED,
        tags={"amd", "gfx950", "mxfp4", "sparse", "radix_topk"},
    )
    def gluon_dsv4_indexer_decode_topk_mxfp4_gfx950(*args, **kwargs):
        return _dsv4_indexer_decode_topk_impl(*args, **kwargs)

    @register_kernel(
        "attention",
        "dsv4_plan",
        name="gluon_dsv4_plan_gfx950",
        solution="gluon",
        capability=CapabilityRequirement(
            min_arch_version=ArchVersion(9, 5),
            max_arch_version=ArchVersion(9, 5),
            vendors=frozenset({"amd"}),
        ),
        signatures=frozenset({format_signature()}),
        traits={"page_size": frozenset({64})},
        priority=Priority.SPECIALIZED,
        tags={"amd", "gfx950", "cuda_graph"},
    )
    def gluon_dsv4_plan_gfx950(**kwargs):
        return _dsv4_plan_impl(**kwargs)

    @register_kernel(
        "attention",
        "dsv4_paged_selected_attention",
        name="gluon_dsv4_paged_selected_attention_split_gfx950",
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
                    swa_kv_cache=dense_tensor_format(torch.uint8),
                )
            }
        ),
        priority=Priority.SPECIALIZED,
        traits={
            "tokens": frozenset({1, 2, 3, 4, 5, 6}),
            "head_dim": frozenset({512}),
            "num_heads": frozenset({16, 32}),
            "cache_layout": frozenset({"fp8_swa_page_planar"}),
            "topk_layout": frozenset({"global_slots"}),
            "support_sink": frozenset({True}),
            "has_extra": frozenset({True}),
            "has_extra_segment": frozenset({True}),
            "swa_selected_width": frozenset({128}),
            "extra_selected_width": frozenset({1024}),
            "swa_page_size": frozenset({64}),
            "extra_page_size": frozenset({64}),
            "metadata_dtypes": frozenset({torch.int32}),
        },
        tags={"amd", "gfx950", "paged_cache", "selected_attention"},
    )
    def gluon_dsv4_paged_selected_attention_split_gfx950(*args, **kwargs):
        return _dsv4_paged_selected_attention_split_impl(*args, **kwargs)

    @register_kernel(
        "attention",
        "dsv4_selected_attention",
        name="gluon_dsv4_selected_attention_gfx950",
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
                    kv=dense_tensor_format(torch.bfloat16),
                )
            }
        ),
        priority=Priority.SPECIALIZED,
        traits={
            "head_dim": frozenset({512}),
            "cache_layout": frozenset({"dense_workspace"}),
            "support_sink": frozenset({True}),
            "selected_width": frozenset({384, 512, 640, 768, 1024, 1152}),
            "metadata_dtypes": frozenset({torch.int32}),
        },
        tags={"amd", "gfx950", "selected_attention"},
    )
    def gluon_dsv4_selected_attention_gfx950(*args, **kwargs):
        return _dsv4_selected_attention_impl(*args, **kwargs)

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
        """Run specialized gfx950 KDA prefill with V-major state."""
        # Host-boundary hint is consumed only by the CuteDSL wrapper.
        kwargs.pop("cu_seqlens_cpu", None)
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
        """Run specialized gfx1250 KDA prefill with V-major state."""
        # Host-boundary hint is consumed only by the CuteDSL wrapper.
        kwargs.pop("cu_seqlens_cpu", None)
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
            "recurrent_layout": frozenset({"v_major"}),
        },
        tags={"amd", "gfx950", "paged_cache", "cuda_graph"},
    )
    def gluon_kda_paged_decode_gfx950(**kwargs):
        """Run specialized gfx950 KDA decode against the physical V-major pool."""
        return _kda_decode_impl(**kwargs)

    @register_kernel(
        "attention",
        "kda_fused_paged_decode",
        name="gluon_kda_fused_paged_decode_vmajor_gfx950",
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
            "recurrent_layout": frozenset({"v_major"}),
        },
        tags={"amd", "gfx950", "paged_cache", "cuda_graph", "fusion"},
    )
    def gluon_kda_fused_paged_decode_vmajor_gfx950(
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
        """Run the decay projection and V-major gfx950 fused decode."""
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

    def _gluon_kda_fused_paged_verify_vmajor_gfx950(
        mixed_qkv: torch.Tensor,
        conv_weights: torch.Tensor,
        conv_states: torch.Tensor,
        conv_scratch: torch.Tensor,
        f_a_out: torch.Tensor,
        f_b_weight: torch.Tensor,
        beta_logits: torch.Tensor,
        A_log: torch.Tensor,
        dt_bias: torch.Tensor,
        *,
        state_pool: torch.Tensor,
        state_scratch: torch.Tensor | None,
        read_indices: torch.Tensor,
        write_indices: torch.Tensor,
        num_heads: int,
        head_dim: int,
        draft_token_num: int,
        lower_bound: float | None,
        replay_mixed_qkv: torch.Tensor | None = None,
        replay_gate: torch.Tensor | None = None,
        replay_beta: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Run target verify and optionally capture a complete raw-g replay tape."""
        raw_g = torch.nn.functional.linear(f_a_out, f_b_weight)
        return _kda_fused_verify_impl(
            mixed_qkv=mixed_qkv,
            conv_weights=conv_weights,
            conv_pool=conv_states,
            conv_scratch=conv_scratch,
            raw_g=raw_g,
            beta_logits=beta_logits,
            A_log=A_log,
            dt_bias=dt_bias,
            state_pool=state_pool,
            state_scratch=state_scratch,
            read_indices=read_indices,
            write_indices=write_indices,
            num_heads=num_heads,
            head_dim=head_dim,
            draft_token_num=draft_token_num,
            lower_bound=lower_bound,
            replay_mixed_qkv=replay_mixed_qkv,
            replay_gate=replay_gate,
            replay_beta=replay_beta,
        )

    @register_kernel(
        "attention",
        "kda_fused_paged_verify",
        name="gluon_kda_fused_paged_verify_nostore_vmajor_gfx950",
        solution="gluon",
        capability=CapabilityRequirement(
            min_arch_version=ArchVersion(9, 5),
            max_arch_version=ArchVersion(9, 5),
            vendors=frozenset({"amd"}),
        ),
        signatures=format_signatures(("q", "k", "v"), "dense", {torch.bfloat16}),
        priority=Priority.SPECIALIZED,
        traits={
            "paged_state": frozenset({True}),
            "store_states": frozenset({False}),
            "recurrent_layout": frozenset({"v_major"}),
            "num_heads": frozenset({12}),
            "head_dim": frozenset({128}),
        },
        tags={
            "amd",
            "gfx950",
            "paged_cache",
            "cuda_graph",
            "fusion",
            "speculative",
            "replay",
        },
    )
    def gluon_kda_fused_paged_verify_nostore_vmajor_gfx950(*args, **kwargs):
        return _gluon_kda_fused_paged_verify_vmajor_gfx950(*args, **kwargs)

    @register_kernel(
        "attention",
        "kda_replay_commit",
        name="gluon_kda_fused_replay_gfx950",
        solution="gluon",
        capability=CapabilityRequirement(
            min_arch_version=ArchVersion(9, 5),
            max_arch_version=ArchVersion(9, 5),
            vendors=frozenset({"amd"}),
        ),
        signatures=format_signatures(("q", "k", "v"), "dense", {torch.bfloat16}),
        priority=Priority.SPECIALIZED,
        traits={
            "flat_state": frozenset({True}),
            "batched_layers": frozenset({True}),
            "recurrent_layout": frozenset({"v_major"}),
            "replay_raw_gate": frozenset({True}),
            "num_heads": frozenset({12}),
            "head_dim": frozenset({128}),
        },
        tags={
            "amd",
            "gfx950",
            "paged_cache",
            "cuda_graph",
            "speculative",
            "replay",
            "batched_layers",
            "raw_gate",
        },
    )
    def gluon_kda_fused_replay_gfx950(
        descriptors: torch.Tensor,
        *,
        read_indices: torch.Tensor,
        write_indices: torch.Tensor,
        accepted_length: torch.Tensor,
        draft_token_num: int,
        num_heads: int,
        head_dim: int,
        f_a_dim: int,
        qkv_stride: int,
        conv_stride: int,
        f_a_stride: int,
        beta_stride: int,
        state_stride: int,
        gate_stride: int,
        conv_width: int,
        layers_per_group: int,
        lower_bound: float,
    ) -> None:
        """Replay all gfx950 layers from persistent BF16 raw-g descriptors."""
        _kda_fused_replay_impl(
            descriptors,
            read_indices,
            write_indices,
            accepted_length,
            draft_token_num=draft_token_num,
            num_heads=num_heads,
            head_dim=head_dim,
            f_a_dim=f_a_dim,
            qkv_stride=qkv_stride,
            conv_stride=conv_stride,
            f_a_stride=f_a_stride,
            beta_stride=beta_stride,
            state_stride=state_stride,
            gate_stride=gate_stride,
            conv_width=conv_width,
            layers_per_group=layers_per_group,
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
            "recurrent_layout": frozenset({"v_major"}),
        },
        tags={"amd", "gfx1250", "paged_cache", "cuda_graph"},
    )
    def gluon_kda_paged_decode_gfx1250(**kwargs):
        """Run specialized gfx1250 KDA decode against the physical V-major pool."""
        return _kda_decode_gfx1250_impl(**kwargs)

    @register_kernel(
        "attention",
        "kda_fused_paged_decode",
        name="gluon_kda_fused_paged_decode_vmajor_gfx1250",
        solution="gluon",
        capability=CapabilityRequirement(
            min_arch_version=ArchVersion(12, 5),
            max_arch_version=ArchVersion(12, 5),
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
            "recurrent_layout": frozenset({"v_major"}),
        },
        tags={"amd", "gfx1250", "paged_cache", "cuda_graph", "fusion"},
    )
    def gluon_kda_fused_paged_decode_vmajor_gfx1250(
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
        """Run the decay projection and V-major gfx1250 fused decode."""
        if output_gate is None or norm_weight is None or norm_eps is None:
            raise ValueError("gfx1250 fused KDA decode requires output normalization")
        raw_g = torch.nn.functional.linear(f_a_out, f_b_weight)
        return _kda_fused_decode_gfx1250_impl(
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
    def gluon_mha_decode_gfx950(*args, enable_pdl: bool = False, **kwargs):
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
    def gluon_mha_decode_gfx1250(*args, enable_pdl: bool = False, **kwargs):
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
    def gluon_mha_extend_gfx950(*args, enable_pdl: bool = False, **kwargs):
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
        name="gluon_dsa_decode_topk_standard_gfx950",
        solution="gluon",
        capability=CapabilityRequirement(
            min_arch_version=ArchVersion(9, 5),
            max_arch_version=ArchVersion(9, 5),
            vendors=frozenset({"amd"}),
        ),
        signatures=frozenset(
            {
                format_signature(
                    q=dense_tensor_format(q_dtype),
                    weights=dense_tensor_format(weight_dtype),
                )
                for q_dtype in (torch.bfloat16, torch.float8_e4m3fn)
                for weight_dtype in (torch.bfloat16, torch.float32)
            }
        ),
        priority=Priority.SPECIALIZED + 1,
        traits={
            "index_heads": frozenset({32, 64}),
            "head_dim": frozenset({128}),
            "topk": frozenset({512, 1024, 2048}),
            "page_size": frozenset({64}),
            "q_len_per_req": frozenset({1, 2, 3, 4, 5, 6}),
            "index_k_format": frozenset({"fp8_scaled"}),
            "index_k_layout": frozenset({"packed", "page_planar"}),
        },
    )
    def gluon_dsa_decode_topk_standard_gfx950(*args, **kwargs):
        return _dsa_decode_topk_standard_impl(*args, **kwargs)

    @register_kernel(
        "attention",
        "dsa_prefill_topk",
        name="gluon_dsa_prefill_topk_standard_gfx950",
        solution="gluon",
        capability=CapabilityRequirement(
            min_arch_version=ArchVersion(9, 5),
            max_arch_version=ArchVersion(9, 5),
            vendors=frozenset({"amd"}),
        ),
        signatures=frozenset(
            {
                format_signature(
                    q=dense_tensor_format(q_dtype),
                    weights=dense_tensor_format(weight_dtype),
                )
                for q_dtype in (torch.bfloat16, torch.float8_e4m3fn)
                for weight_dtype in (torch.bfloat16, torch.float32)
            }
        ),
        priority=Priority.SPECIALIZED + 1,
        traits={
            "index_heads": frozenset({32, 64}),
            "head_dim": frozenset({128}),
            "topk": frozenset({512, 1024, 2048}),
            "page_size": frozenset({64}),
            "index_k_format": frozenset({"fp8_scaled"}),
            "index_k_layout": frozenset({"packed", "page_planar"}),
        },
    )
    def gluon_dsa_prefill_topk_standard_gfx950(*args, **kwargs):
        return _dsa_prefill_topk_standard_impl(*args, **kwargs)

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
            "index_k_layout": frozenset({"packed", "page_planar"}),
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
            "index_k_layout": frozenset({"packed", "page_planar"}),
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
    def gluon_dsa_decode_gfx950(*args, enable_pdl: bool = False, **kwargs):
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
    def gluon_dsa_prefill_gfx950(*args, enable_pdl: bool = False, **kwargs):
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
    def gluon_dsa_prefill_fp8_dense_gfx950(*args, enable_pdl: bool = False, **kwargs):
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
            "index_k_layout": frozenset({"packed", "page_planar"}),
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
            "index_k_layout": frozenset({"packed", "page_planar"}),
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
    def gluon_dsa_decode_gfx1250(*args, enable_pdl: bool = False, **kwargs):
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
    def gluon_dsa_prefill_gfx1250(*args, enable_pdl: bool = False, **kwargs):
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
    def gluon_dsa_prefill_fp8_dense_gfx1250(*args, enable_pdl: bool = False, **kwargs):
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
