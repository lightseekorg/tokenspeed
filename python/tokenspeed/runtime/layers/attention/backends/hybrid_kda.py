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

"""KDA (Kimi Delta Attention) backends: the scan seams KDA overrides on the
shared linear-attention machinery, and the composite wrapper KDA hybrids use.
See ``KdaAttnBackend`` for what separates the family from GDN."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from tokenspeed_kernel.ops.activation.triton import rmsnorm_gated_sigmoid
from tokenspeed_kernel.ops.attention import (
    kda_paged_decode,
    kda_paged_prefill,
    try_kda_fused_paged_decode,
    try_kda_fused_paged_verify,
)
from typing_extensions import override

from tokenspeed.runtime.layers.attention.backends.hybrid_linear_attn import (
    HybridLinearAttnBackend,
    MambaAttnBackend,
    logger,
)

if TYPE_CHECKING:
    from tokenspeed.runtime.layers.attention.configs.base import BaseAttnConfig


KDA_PREFILL_BACKENDS = ("auto", "fla", "flashkda", "cutedsl_kda")


def _cu_seqlens_cpu_hint(
    extend_seq_lens_cpu: torch.Tensor | None, expected_len: int
) -> tuple[int, ...] | None:
    """Host prefix sum of the scheduler's extend lengths, or ``None``.

    The tuple must equal the contents of ``query_start_loc`` — a wrong hint
    silently corrupts the CuteDSL host chunk plan — so any absence or length
    misalignment returns ``None`` (the wrapper then falls back to its own
    boundary read).

    Args:
        extend_seq_lens_cpu: CPU per-sequence extend lengths, or ``None``.
        expected_len: ``query_start_loc.numel()`` of the batch.

    Returns:
        ``(0, lens[0], lens[0]+lens[1], ...)`` when it has exactly
        ``expected_len`` entries; ``None`` otherwise.
    """
    if extend_seq_lens_cpu is None:
        return None
    bounds = [0]
    for n in extend_seq_lens_cpu.tolist():
        bounds.append(bounds[-1] + int(n))
    if len(bounds) != expected_len:
        return None
    return tuple(bounds)


def _slice_kda_prefill_inputs(
    num_real_tokens: int,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    gate: torch.Tensor,
    beta: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Slice packed KDA inputs to the real-token prefix along their token axis."""
    return (
        query[:, :num_real_tokens],
        key[:, :num_real_tokens],
        value[:, :num_real_tokens],
        gate[:, :num_real_tokens],
        beta[:, :num_real_tokens],
    )


class KdaAttnBackend(MambaAttnBackend):
    """Attention backend for KDA linear attention layers (Kimi-K3).

    Everything generic to linear attention -- state paging, cache groups,
    cuda-graph buffers, the verify scratch and its commit -- is inherited
    unchanged. KDA only replaces the scan seams: its decay gate is
    per-channel (a low-rank ``f_a``/``f_b`` projection plus raw beta logits)
    where GDN's is scalar per head, and its decode/verify kernels can fuse
    the conv, the gate GEMV and the recurrence into a single launch.
    """

    def __init__(self, config: BaseAttnConfig, kda_backend: str = "auto") -> None:
        super().__init__(config)
        self.kda_backend = (kda_backend or "auto").strip().lower()
        if self.kda_backend not in KDA_PREFILL_BACKENDS:
            raise ValueError(
                f"--kda-backend must be one of {', '.join(KDA_PREFILL_BACKENDS)}; "
                f"got {self.kda_backend!r}"
            )
        logger.info(
            "KDA prefill routes through %s; decode remains on the "
            "platform-selected kernels",
            self.kda_backend,
        )

    def _kda_gate(
        self,
        g_raw: torch.Tensor | None,
        f_a_out: torch.Tensor | None,
        f_b_weight: torch.Tensor | None,
    ) -> torch.Tensor | None:
        """Per-channel decay gate for the multi-token extend paths.

        The fused decode/verify kernels absorb ``f_b`` into the scan; the
        remaining paths need the plain GEMV, computed on first use.

        Args:
            g_raw: Gate the model already materialized, when it did.
            f_a_out: Low-rank gate activation feeding the GEMV.
            f_b_weight: Second gate projection.

        Returns:
            The gate, or None when the model supplied neither form.
        """
        if g_raw is None and f_a_out is not None:
            return torch.nn.functional.linear(f_a_out, f_b_weight)
        else:
            return g_raw

    @override
    def _decode(
        self,
        mixed_qkv: torch.Tensor,
        conv_weights: torch.Tensor,
        conv_states: torch.Tensor,
        ssm_states: torch.Tensor,
        read_indices: torch.Tensor,
        write_indices: torch.Tensor,
        *,
        f_a_out: torch.Tensor | None,
        f_b_weight: torch.Tensor | None,
        beta_raw: torch.Tensor | None,
        A_log: torch.Tensor,
        dt_bias: torch.Tensor,
        value_dim: int,
        attn_tp_size: int,
        head_v_dim: int,
        lower_bound: float | None,
        output_gate: torch.Tensor | None,
        norm_weight: torch.Tensor | None,
        norm_eps: float | None,
    ) -> torch.Tensor | None:
        if output_gate is not None and (norm_weight is None or norm_eps is None):
            raise ValueError(
                "norm_weight and norm_eps are required with a KDA output gate"
            )
        if f_a_out is None:
            return None

        num_value_heads = value_dim // attn_tp_size // head_v_dim
        result = try_kda_fused_paged_decode(
            mixed_qkv,
            conv_weights,
            conv_states,
            f_a_out,
            f_b_weight,
            beta_raw,
            A_log,
            dt_bias,
            state_pool=ssm_states,
            read_indices=read_indices,
            write_indices=write_indices,
            num_heads=num_value_heads,
            head_dim=head_v_dim,
            cu_seqlens=self.forward_metadata.query_start_loc,
            lower_bound=lower_bound,
            output_gate=output_gate,
            norm_weight=norm_weight,
            norm_eps=norm_eps,
        )
        if result is None:
            return None
        if result.output_norm_applied or output_gate is None:
            return result.out
        return rmsnorm_gated_sigmoid(
            result.out.reshape(-1, num_value_heads * head_v_dim).contiguous(),
            output_gate.contiguous(),
            norm_weight,
            norm_eps,
            num_value_heads,
            head_v_dim,
        ).view(1, -1, num_value_heads, head_v_dim)

    @override
    def _decode_scan(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        ssm_states: torch.Tensor,
        read_indices: torch.Tensor,
        write_indices: torch.Tensor,
        *,
        A_log: torch.Tensor,
        dt_bias: torch.Tensor,
        a: torch.Tensor | None,
        b: torch.Tensor | None,
        g_raw: torch.Tensor | None,
        f_a_out: torch.Tensor | None,
        f_b_weight: torch.Tensor | None,
        beta_raw: torch.Tensor | None,
        lower_bound: float | None,
        output_gate: torch.Tensor | None,
        norm_weight: torch.Tensor | None,
        norm_eps: float | None,
    ) -> torch.Tensor:
        seq_len = query.shape[0]
        num_heads = query.shape[2]
        head_k_dim = query.shape[3]
        num_value_heads = value.shape[2]
        head_v_dim = value.shape[3]

        query_start_loc = self.forward_metadata.query_start_loc
        g_raw = self._kda_gate(g_raw, f_a_out, f_b_weight)
        query = query.view(1, seq_len, num_heads, head_k_dim)
        key = key.view(1, seq_len, num_heads, head_k_dim)
        value = value.view(1, seq_len, num_value_heads, head_v_dim)
        g_kda = g_raw.view(1, seq_len, num_value_heads, head_k_dim)
        beta_kda = beta_raw.view(1, seq_len, num_value_heads)

        core_attn_out = kda_paged_decode(
            query,
            key,
            value,
            g_kda,
            beta_kda,
            A_log,
            dt_bias,
            state_pool=ssm_states,
            read_indices=read_indices,
            write_indices=write_indices,
            cu_seqlens=query_start_loc,
            lower_bound=lower_bound,
        )
        if output_gate is not None:
            core_attn_out = rmsnorm_gated_sigmoid(
                core_attn_out.reshape(-1, num_value_heads * head_v_dim).contiguous(),
                output_gate.contiguous(),
                norm_weight,
                norm_eps,
                num_value_heads,
                head_v_dim,
            ).view(1, -1, num_value_heads, head_v_dim)
        return core_attn_out.squeeze(0)

    @override
    def _verify(
        self,
        mixed_qkv: torch.Tensor,
        conv_weights: torch.Tensor,
        conv_comp: torch.Tensor,
        conv_scratch: torch.Tensor,
        ssm_comp: torch.Tensor,
        ssm_scratch: torch.Tensor,
        state_in_blocks: torch.Tensor,
        output_indices: torch.Tensor,
        *,
        bias: torch.Tensor | None,
        f_a_out: torch.Tensor | None,
        f_b_weight: torch.Tensor | None,
        beta_raw: torch.Tensor | None,
        A_log: torch.Tensor,
        dt_bias: torch.Tensor,
        batch_size: int,
        draft_token_num: int,
        value_dim: int,
        attn_tp_size: int,
        head_v_dim: int,
        lower_bound: float | None,
    ) -> torch.Tensor | None:

        if f_a_out is None or bias is not None:
            return None
        else:
            num_value_heads = value_dim // attn_tp_size // head_v_dim
            return try_kda_fused_paged_verify(
                mixed_qkv,
                conv_weights,
                conv_comp,
                conv_scratch,
                f_a_out,
                f_b_weight,
                beta_raw,
                A_log,
                dt_bias,
                state_pool=ssm_comp,
                state_scratch=ssm_scratch,
                read_indices=state_in_blocks[:batch_size],
                write_indices=output_indices[:batch_size],
                num_heads=num_value_heads,
                head_dim=head_v_dim,
                draft_token_num=draft_token_num,
                lower_bound=lower_bound,
            )

    @override
    def _verify_scan(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        ssm_comp: torch.Tensor,
        ssm_scratch: torch.Tensor,
        state_in_blocks: torch.Tensor,
        output_indices: torch.Tensor,
        *,
        A_log: torch.Tensor,
        dt_bias: torch.Tensor,
        a: torch.Tensor | None,
        b: torch.Tensor | None,
        g_raw: torch.Tensor | None,
        f_a_out: torch.Tensor | None,
        f_b_weight: torch.Tensor | None,
        beta_raw: torch.Tensor | None,
        batch_size: int,
        draft_token_num: int,
        seq_len: int,
        lower_bound: float | None,
    ) -> torch.Tensor:

        from tokenspeed_kernel.thirdparty.triton.fla_kda_recurrent import (
            fused_recurrent_kda_mtp,
        )

        num_heads = query.shape[2]
        head_k_dim = query.shape[3]
        num_value_heads = value.shape[2]
        head_v_dim = value.shape[3]

        query_b = query.view(batch_size, draft_token_num, num_heads, head_k_dim)
        key_b = key.view(batch_size, draft_token_num, num_heads, head_k_dim)
        value_b = value.view(batch_size, draft_token_num, num_value_heads, head_v_dim)

        g_b = self._kda_gate(g_raw, f_a_out, f_b_weight).view(
            batch_size, draft_token_num, num_value_heads, head_k_dim
        )

        beta_b = beta_raw.view(batch_size, draft_token_num, num_value_heads)
        grid = output_indices[:batch_size]

        return fused_recurrent_kda_mtp(
            query_b,
            key_b,
            value_b,
            g_b,
            beta_b,
            A_log,
            dt_bias,
            ssm_comp,
            state_in_blocks[:batch_size].to(torch.int64),
            grid,
            h_pool_out=ssm_scratch,
            lower_bound=lower_bound,
        ).reshape(1, seq_len, num_value_heads, head_v_dim)

    @override
    def _prefill_scan(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        recurrent_state: torch.Tensor,
        query_start_loc: torch.Tensor,
        *,
        A_log: torch.Tensor,
        dt_bias: torch.Tensor,
        a: torch.Tensor | None,
        b: torch.Tensor | None,
        g_raw: torch.Tensor | None,
        f_a_out: torch.Tensor | None,
        f_b_weight: torch.Tensor | None,
        beta_raw: torch.Tensor | None,
        seq_len: int,
        num_real_tokens: int,
        lower_bound: float | None,
        extend_seq_lens_cpu: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Run only the real-token prefix through the KDA prefill kernel.

        FlashKDA sizes its output from the input shape but tiles from
        ``cu_seqlens``: bucket-padded inputs would leave the output tail
        unwritten and feed padding into its final full-tile loads. The graph
        handoff clears and restores the bucket tail afterward.

        ``extend_seq_lens_cpu`` is the scheduler's host-side per-sequence
        extend lengths; its prefix sum equals the contents of
        ``query_start_loc``. Forwarding it as ``cu_seqlens_cpu`` lets the
        CuteDSL wrapper plan on the host without a stream-synchronizing D2H
        read of the boundaries — otherwise that read recurs on every KDA
        layer of every prefill chunk (the wrapper's identity memo cannot hit
        across layers because the op casts ``cu_seqlens`` to a fresh int64
        tensor per call).
        """
        head_k_dim = query.shape[3]
        num_value_heads = value.shape[2]

        g_kda = self._kda_gate(g_raw, f_a_out, f_b_weight).view(
            1, seq_len, num_value_heads, head_k_dim
        )

        beta_kda = beta_raw.view(1, seq_len, num_value_heads)

        query, key, value, g_kda, beta_kda = _slice_kda_prefill_inputs(
            num_real_tokens, query, key, value, g_kda, beta_kda
        )

        cu_seqlens_cpu = _cu_seqlens_cpu_hint(
            extend_seq_lens_cpu, query_start_loc.numel()
        )

        kda_result = kda_paged_prefill(
            query,
            key,
            value,
            g_kda,
            beta_kda,
            A_log,
            dt_bias,
            initial_state=recurrent_state,
            cu_seqlens=query_start_loc,
            cu_seqlens_cpu=cu_seqlens_cpu,
            lower_bound=lower_bound,
            solution=None if self.kda_backend == "auto" else self.kda_backend,
        )

        return kda_result.out.squeeze(0), kda_result.final_state


class HybridKDABackend(HybridLinearAttnBackend):
    """Composite backend for KDA hybrid models (full attention + KDA layers).

    Identical to ``HybridLinearAttnBackend`` today; it exists so KDA-only
    composite surface (deferred-commit settlement, lifecycle hooks) has a
    home that other linear hybrids never inherit.
    """
