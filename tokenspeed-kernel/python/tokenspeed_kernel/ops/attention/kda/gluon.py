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

"""Registration shims for AMD Gluon KDA attention kernels."""

from __future__ import annotations

import torch
from tokenspeed_kernel.ops.attention.kda import KdaPrefillResult
from tokenspeed_kernel.platform import (
    ArchVersion,
    CapabilityRequirement,
    current_platform,
)
from tokenspeed_kernel.registry import Priority, register_kernel
from tokenspeed_kernel.signature import format_signatures

if current_platform().is_amd:
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
    from tokenspeed_kernel_amd.ops.gfx1250.attention.kda.decode import (
        gluon_kda_fused_decode_gfx1250 as _kda_fused_decode_gfx1250_impl,
    )
    from tokenspeed_kernel_amd.ops.gfx1250.attention.kda.decode import (
        gluon_kda_fused_replay_gfx1250 as _kda_fused_replay_gfx1250_impl,
    )
    from tokenspeed_kernel_amd.ops.gfx1250.attention.kda.decode import (
        gluon_kda_fused_verify_gfx1250 as _kda_fused_verify_gfx1250_impl,
    )
    from tokenspeed_kernel_amd.ops.gfx1250.attention.kda.decode import (
        gluon_kda_recurrent_decode_gfx1250 as _kda_decode_gfx1250_impl,
    )
    from tokenspeed_kernel_amd.ops.gfx1250.attention.kda.prefill import (
        gluon_kda_paged_prefill_gfx1250 as _kda_prefill_gfx1250_impl,
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
        group_indices: torch.Tensor,
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
        lower_bound: float,
    ) -> None:
        """Replay all gfx950 layers from persistent BF16 raw-g descriptors."""
        _kda_fused_replay_impl(
            descriptors,
            group_indices,
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

    def _gluon_kda_fused_paged_verify_vmajor_gfx1250(
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
        return _kda_fused_verify_gfx1250_impl(
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
        name="gluon_kda_fused_paged_verify_nostore_vmajor_gfx1250",
        solution="gluon",
        capability=CapabilityRequirement(
            min_arch_version=ArchVersion(12, 5),
            max_arch_version=ArchVersion(12, 5),
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
            "gfx1250",
            "paged_cache",
            "cuda_graph",
            "fusion",
            "speculative",
            "replay",
        },
    )
    def gluon_kda_fused_paged_verify_nostore_vmajor_gfx1250(*args, **kwargs):
        return _gluon_kda_fused_paged_verify_vmajor_gfx1250(*args, **kwargs)

    @register_kernel(
        "attention",
        "kda_replay_commit",
        name="gluon_kda_fused_replay_gfx1250",
        solution="gluon",
        capability=CapabilityRequirement(
            min_arch_version=ArchVersion(12, 5),
            max_arch_version=ArchVersion(12, 5),
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
            "gfx1250",
            "paged_cache",
            "cuda_graph",
            "speculative",
            "replay",
            "batched_layers",
            "raw_gate",
        },
    )
    def gluon_kda_fused_replay_gfx1250(
        descriptors: torch.Tensor,
        *,
        group_indices: torch.Tensor,
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
        lower_bound: float,
    ) -> None:
        """Replay all gfx1250 layers from persistent BF16 raw-g descriptors."""
        _kda_fused_replay_gfx1250_impl(
            descriptors,
            group_indices,
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
            lower_bound=lower_bound,
        )
