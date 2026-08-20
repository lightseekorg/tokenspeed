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
    kda_replay_commit_supported,
    resolve_kda_batched_replay_commit,
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

    Everything generic to linear attention -- state paging and cache groups --
    is inherited. When replay commit is available, speculative verify captures
    its compact projection payload and writes no per-position state tape;
    commit immediately replays the accepted prefix from the committed page.
    KDA also replaces the scan seams: its decay gate is
    per-channel (a low-rank ``f_a``/``f_b`` projection plus raw beta logits)
    where GDN's is scalar per head, and its decode/verify kernels can fuse
    the conv, the gate GEMV and the recurrence into a single launch.
    """

    def __init__(self, config: BaseAttnConfig, kda_backend: str = "auto") -> None:
        super().__init__(config)
        self.max_bs = config.max_bs
        self._replay_active = kda_replay_commit_supported(self.dtype)
        self._batched_replay_kernel = resolve_kda_batched_replay_commit(self.dtype)
        self._replay_payloads: tuple[torch.Tensor, ...] | None = None
        self._replay_weights: dict[int, tuple] = {}
        self._replay_descriptors = None
        self._batched_replay_launch = None
        self._batched_replay_ready = False
        self._replay_descriptor_bound: set[int] = set()
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

    @override
    def set_kv_pool(self, kv_pool) -> None:
        super().set_kv_pool(kv_pool)
        if self._replay_active and self.speculative_num_draft_tokens > 1:
            rows = self.max_bs * self.speculative_num_draft_tokens
            layer_ids = tuple(self._state_layer_ids())
            self._descriptor_row_by_layer = {
                layer_id: row for row, layer_id in enumerate(layer_ids)
            }
            self._replay_group_ids = tuple(self._state_groups())
            self._replay_group_rows = {
                group_id: row for row, group_id in enumerate(self._replay_group_ids)
            }
            with torch.inference_mode(False):
                addresses = torch.zeros(
                    (len(layer_ids), 10), dtype=torch.uint64, device=self.device
                )
                first_conv, first_ssm = self._state_components(layer_ids[0])
                hv, head_dim = first_ssm.shape[1:3]
                for layer_id in layer_ids[1:]:
                    conv, ssm = self._state_components(layer_id)
                    if (
                        conv.shape[1:] != first_conv.shape[1:]
                        or ssm.shape[1:] != first_ssm.shape[1:]
                    ):
                        raise RuntimeError(
                            "batched KDA replay requires uniform layer pools"
                        )
                payload_shape = (len(layer_ids), rows)
                self._replay_payloads = (
                    torch.empty(
                        (*payload_shape, first_conv.shape[1]),
                        dtype=self.dtype,
                        device=self.device,
                    ),
                    torch.empty(
                        (*payload_shape, head_dim), dtype=self.dtype, device=self.device
                    ),
                    torch.empty(
                        (*payload_shape, hv), dtype=self.dtype, device=self.device
                    ),
                    torch.empty(
                        (*payload_shape, hv * head_dim),
                        dtype=torch.float32,
                        device=self.device,
                    ),
                )
                self._replay_descriptors = addresses
                self._replay_descriptor_bound.clear()
                self._replay_weights.clear()
                self._batched_replay_launch = None
                self._batched_replay_ready = False

    def _replay_payload(self, layer_id: int) -> tuple[torch.Tensor, ...]:
        """Return one layer row from the stacked replay workspaces."""
        assert self._replay_payloads is not None
        row = self._descriptor_row_by_layer[layer_id]
        return tuple(payload[row] for payload in self._replay_payloads)

    def _bind_replay_descriptor(self, layer_id: int, weights: tuple) -> None:
        """Publish one layer's stable pointers after its weights are observed."""
        if layer_id in self._replay_descriptor_bound:
            return
        assert self._replay_descriptors is not None
        conv_w, f_b, A_log, dt_bias, _num_heads, _head_dim, lower_bound = weights
        if dt_bias is None or lower_bound is None:
            return
        conv, state = self._state_components(layer_id)
        qkv, f_a, beta, gate = self._replay_payload(layer_id)
        row = self._descriptor_row_by_layer[layer_id]
        addresses = self._replay_descriptors
        addresses[row].copy_(
            torch.tensor(
                [
                    qkv.data_ptr(),
                    conv_w.data_ptr(),
                    conv.data_ptr(),
                    f_a.data_ptr(),
                    f_b.data_ptr(),
                    beta.data_ptr(),
                    A_log.data_ptr(),
                    dt_bias.data_ptr(),
                    state.data_ptr(),
                    gate.data_ptr(),
                ],
                dtype=torch.uint64,
                device=self.device,
            )
        )
        self._replay_descriptor_bound.add(layer_id)
        if len(self._replay_descriptor_bound) == len(self._descriptor_row_by_layer):
            first = next(iter(self._replay_weights.values()))
            first_conv_w, first_f_b, _, _, first_heads, first_dim, _ = first
            geometry = (
                first_heads,
                first_dim,
                first_f_b.shape[1],
                first_conv_w.shape[1],
            )
            first_layer = next(iter(self._replay_weights))
            first_conv, first_state = self._state_components(first_layer)
            first_qkv, first_fa, first_beta, first_gate = self._replay_payload(
                first_layer
            )
            strides = (
                first_qkv.stride(0),
                first_conv.stride(0),
                first_fa.stride(0),
                first_beta.stride(0),
                first_state.stride(0),
                first_gate.stride(0),
            )
            lower_bounds = set()
            for current_layer, layer_weights in self._replay_weights.items():
                layer_conv_w, layer_f_b, _, _, layer_heads, layer_dim, _ = layer_weights
                if (
                    layer_heads,
                    layer_dim,
                    layer_f_b.shape[1],
                    layer_conv_w.shape[1],
                ) != geometry:
                    raise RuntimeError("batched KDA replay requires uniform geometry")
                layer_conv, layer_state = self._state_components(current_layer)
                layer_qkv, layer_fa, layer_beta, layer_gate = self._replay_payload(
                    current_layer
                )
                if (
                    layer_qkv.stride(0),
                    layer_conv.stride(0),
                    layer_fa.stride(0),
                    layer_beta.stride(0),
                    layer_state.stride(0),
                    layer_gate.stride(0),
                ) != strides:
                    raise RuntimeError("batched KDA replay requires uniform strides")
                lower_bounds.add(layer_weights[-1])
            if len(lower_bounds) != 1:
                raise RuntimeError("batched KDA replay requires one lower bound")
            layers_per_group = len(self._descriptor_row_by_layer) // len(
                self._replay_group_ids
            )
            expected_groups = tuple(
                group_id
                for group_id in self._replay_group_ids
                for _ in range(layers_per_group)
            )
            actual_groups = tuple(
                self._state_group_for(layer_id)
                for layer_id in self._descriptor_row_by_layer
            )
            if actual_groups != expected_groups:
                raise RuntimeError(
                    "batched KDA replay requires equal contiguous layer groups"
                )
            conv_width = geometry[3]
            if self._batched_replay_kernel is not None:
                descriptors = self._replay_descriptors
                draft_tokens = self.speculative_num_draft_tokens

                def launch(read_indices, write_indices, accepted_length):
                    self._batched_replay_kernel(
                        descriptors=descriptors,
                        read_indices=read_indices,
                        write_indices=write_indices,
                        accepted_length=accepted_length,
                        draft_token_num=draft_tokens,
                        num_heads=geometry[0],
                        head_dim=geometry[1],
                        f_a_dim=geometry[2],
                        qkv_stride=strides[0],
                        conv_stride=strides[1],
                        f_a_stride=strides[2],
                        beta_stride=strides[3],
                        state_stride=strides[4],
                        gate_stride=strides[5],
                        conv_width=conv_width,
                        layers_per_group=layers_per_group,
                        lower_bound=next(iter(lower_bounds)),
                    )

                self._batched_replay_launch = launch
                self._batched_replay_ready = True

    @override
    def _ensure_verify_scratch(self, bs: int, draft_token_num: int) -> None:
        if not self._replay_active:
            return super()._ensure_verify_scratch(bs, draft_token_num)
        rows = max(len(self.query_start_loc_list), bs)
        scratch = self._verify_scratch
        if scratch is not None:
            # Allocated once at its maximum; graphs may hold its addresses, so
            # an overrun is an invariant violation, never a resize.
            capacity = next(iter(scratch.values()))[0].shape[0]
            if capacity < rows:
                raise RuntimeError(
                    f"KDA verify needs {rows} transient conv rows but the "
                    f"preallocated scratch holds {capacity}"
                )
            return
        self._verify_scratch = {}
        for layer_id in self._state_layer_ids():
            conv, _ = self._state_components(layer_id)
            self._verify_scratch[layer_id] = (
                torch.empty(
                    (rows, *conv.shape[1:]), dtype=conv.dtype, device=conv.device
                ),
                None,
            )

    @override
    def preallocate_verify_workspace(self, max_bs: int, draft_token_num: int) -> int:
        if not self._replay_active:
            return super().preallocate_verify_workspace(max_bs, draft_token_num)
        self._ensure_verify_scratch(max_bs, draft_token_num)
        conv_bytes = sum(pair[0].nbytes for pair in self._verify_scratch.values())
        payload_bytes = sum(payload.nbytes for payload in (self._replay_payloads or ()))
        return conv_bytes + payload_bytes

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
        layer_id: int,
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
        if self._replay_active:
            if f_a_out is None or bias is not None:
                raise RuntimeError(
                    "KDA eager replay requires f_a_out and a bias-free convolution"
                )
            qkv, f_a, beta, _ = self._replay_payload(layer_id)
            rows = batch_size * draft_token_num
            qkv[:rows, : mixed_qkv.shape[-1]].copy_(mixed_qkv[:rows])
            f_a[:rows, : f_a_out.shape[-1]].copy_(f_a_out[:rows])
            beta[:rows, : beta_raw.shape[-1]].copy_(beta_raw[:rows])
            num_value_heads = value_dim // attn_tp_size // head_v_dim
            if layer_id not in self._replay_weights:
                # Parameters are stable objects; model weight updates copy into
                # their storage, so binding their pointers once cannot stale.
                self._replay_weights[layer_id] = (
                    conv_weights,
                    f_b_weight,
                    A_log,
                    dt_bias,
                    num_value_heads,
                    head_v_dim,
                    lower_bound,
                )
                self._bind_replay_descriptor(layer_id, self._replay_weights[layer_id])
            fused_out = try_kda_fused_paged_verify(
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
                store_states=False,
            )
            if fused_out is None:
                raise RuntimeError(
                    "KDA fused paged verify kernel vanished after the replay "
                    "capability probe reported it available"
                )
            return fused_out
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
        initial_pool = ssm_scratch
        initial_rows = output_indices[:batch_size, 0] - 1
        write_rows = output_indices[:batch_size]
        state_out = ssm_scratch

        return fused_recurrent_kda_mtp(
            query_b,
            key_b,
            value_b,
            g_b,
            beta_b,
            A_log,
            dt_bias,
            initial_pool,
            initial_rows,
            write_rows,
            h_pool_out=state_out,
            lower_bound=lower_bound,
        ).reshape(1, seq_len, num_value_heads, head_v_dim)

    @override
    def commit_verified_state(self, accepted_length: torch.Tensor) -> None:
        """Replay and eagerly commit this round's accepted KDA prefix."""
        if not self._replay_active:
            return super().commit_verified_state(accepted_length)
        ctx = self._verify_commit_ctx
        if ctx is None:
            return
        from tokenspeed_kernel.ops.attention import try_kda_replay_commit

        committed, tables, draft_token_num, read_pages_by_group = ctx
        bs = accepted_length.shape[0]
        # Runtime accept lengths count draft matches; the target token itself
        # always advances state, matching the established scratch commit.
        steps = accepted_length.to(torch.int64).clamp(min=1, max=draft_token_num)
        new_last = committed[:bs] + steps - 1
        slot = torch.div(
            new_last.clamp_min(0),
            self._checkpoint_granularity,
            rounding_mode="floor",
        )
        pages_by_group = {}
        for group_id in self._state_groups():
            rows = tables[group_id]
            safe = slot.clamp(max=rows.shape[1] - 1)
            pages_by_group[group_id] = (
                rows[:bs].gather(1, safe.unsqueeze(1)).squeeze(1).to(torch.int32)
            )
        rows = bs * draft_token_num
        if self._batched_replay_ready:
            read_pages = torch.stack(
                [
                    read_pages_by_group[group_id][:bs]
                    for group_id in self._replay_group_ids
                ]
            ).to(torch.int32)
            write_pages = torch.stack(
                [pages_by_group[group_id] for group_id in self._replay_group_ids]
            )
            self._batched_replay_launch(read_pages, write_pages, steps.to(torch.int32))
            self._verify_commit_ctx = None
            return
        for layer_id, weights in self._replay_weights.items():
            conv_w, f_b, A_log, dt_bias, num_heads, head_dim, lower_bound = weights
            group_id = self._state_group_for(layer_id)
            conv, state = self._state_components(layer_id)
            qkv, f_a, beta, gate = self._replay_payload(layer_id)
            if not try_kda_replay_commit(
                qkv[:rows, : conv_w.shape[0]],
                conv_w,
                conv,
                conv,
                f_a[:rows, : f_b.shape[1]],
                f_b,
                beta[:rows, :num_heads],
                A_log,
                dt_bias,
                state_pool=state,
                state_out=state,
                read_indices=read_pages_by_group[group_id][:bs],
                write_indices=pages_by_group[group_id],
                accepted_length=steps.to(torch.int32),
                num_heads=num_heads,
                head_dim=head_dim,
                draft_token_num=draft_token_num,
                lower_bound=lower_bound,
                gate_scratch=gate[:rows, : num_heads * head_dim],
            ):
                raise RuntimeError(
                    "KDA replay commit kernel vanished after capability probing"
                )
        self._verify_commit_ctx = None

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
