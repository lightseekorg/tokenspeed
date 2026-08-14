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
    try_kda_fused_paged_decode,
    try_kda_fused_paged_verify,
)
from typing_extensions import override

from tokenspeed.runtime.execution.forward_batch_info import ForwardMode
from tokenspeed.runtime.layers.attention.backends.hybrid_linear_attn import (
    HybridLinearAttnBackend,
    MambaAttnBackend,
    logger,
)
from tokenspeed.runtime.utils.pdl import pdl_enabled

if TYPE_CHECKING:
    from tokenspeed.runtime.layers.attention.configs.base import BaseAttnConfig


KDA_PREFILL_BACKENDS = ("auto", "fla", "flashkda", "cutedsl_kda")


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
        self.max_bs = config.max_bs
        self._kda_pending: dict | None = None
        self._kda_replay_staged = False
        self._kda_overflow_round = False
        self._kda_overflow_payload: dict = {}
        self._kda_lazy_bufs: dict | None = None
        self._replay_active: bool = kda_replay_commit_supported()
        self._replay_payload_cache: dict | None = None
        self._replay_layer_weights: dict | None = None
        self._verify_scratch: dict | None = None
        self._kda_slot_table = torch.full(
            (self.max_bs + 8,), -1, dtype=torch.int32, device=self.device
        )
        self._kda_table_drop_staging = torch.empty(
            self.max_bs + 8, dtype=torch.int64, pin_memory=True
        )
        self._payload_parity = 0
        self._staged_parity = 0
        self._graphs_captured = False
        self._payload_half_rows: int | None = None
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
        """Bind state storage and allocate replay buffers from its geometry."""
        super().set_kv_pool(kv_pool)
        if self._replay_active and self.speculative_num_draft_tokens > 1:
            self._preallocate_kda_replay_buffers(self.max_bs)

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
        conv_scratch: torch.Tensor | None,
        ssm_comp: torch.Tensor,
        ssm_scratch: torch.Tensor | None,
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
            return self._replay_verify(
                mixed_qkv,
                conv_weights,
                conv_comp,
                ssm_comp,
                layer_id=layer_id,
                bias=bias,
                f_a_out=f_a_out,
                f_b_weight=f_b_weight,
                beta_raw=beta_raw,
                A_log=A_log,
                dt_bias=dt_bias,
                batch_size=batch_size,
                draft_token_num=draft_token_num,
                value_dim=value_dim,
                attn_tp_size=attn_tp_size,
                head_v_dim=head_v_dim,
                lower_bound=lower_bound,
            )
        else:
            return None

    def _replay_verify(
        self,
        mixed_qkv: torch.Tensor,
        conv_weights: torch.Tensor,
        conv_comp: torch.Tensor,
        ssm_comp: torch.Tensor,
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
    ) -> torch.Tensor:
        """Run fused verify with replay capture and lazy commit.

        The registry probe that enables replay also registers this fused
        verify. A decline therefore signals a registry/platform mismatch;
        replay deliberately omitted scratch, so fallback would only fail later
        with an obscure missing-scratch error.
        """
        if f_a_out is None or bias is not None:
            raise RuntimeError(
                "KDA verify fell through to the per-position scratch path "
                "while replay commit is active; the fused KDA verify kernel "
                "must handle every verify batch"
            )

        num_value_heads = value_dim // attn_tp_size // head_v_dim
        gid = self.kv_pool.group_id_for_layer(layer_id)
        bufs = self._kda_lazy_buffers(min_slots=batch_size)
        max_rows = max(len(self.query_start_loc_list), batch_size) * draft_token_num
        widths = (mixed_qkv.shape[-1], f_a_out.shape[-1], beta_raw.shape[-1])
        qkv_buf, f_a_buf, beta_buf = self._replay_payload(
            layer_id, max_rows, widths, mixed_qkv.dtype
        )
        fused_out = try_kda_fused_paged_verify(
            mixed_qkv,
            conv_weights,
            conv_comp,
            f_a_out,
            f_b_weight,
            beta_raw,
            A_log,
            dt_bias,
            state_pool=ssm_comp,
            num_heads=num_value_heads,
            head_dim=head_v_dim,
            draft_token_num=draft_token_num,
            lower_bound=lower_bound,
            enable_pdl=pdl_enabled(),
            read_indices=bufs["anchor"][gid][:batch_size],
            prev_qkv=qkv_buf,
            prev_f_a=f_a_buf,
            prev_beta=beta_buf,
            prev_base=bufs["base"][:batch_size],
            prev_steps=bufs["steps"][:batch_size],
            commit_indices=bufs["commit"][gid][:batch_size],
            capture=(f_a_buf, beta_buf, bufs["capture_base"]),
            gate_scratch=self._kda_gate_scratch(
                2 * batch_size * draft_token_num,
                num_value_heads * head_v_dim,
            ),
        )
        if fused_out is None:
            raise RuntimeError(
                "KDA verify fell through to the per-position scratch path while "
                "replay commit is active; the fused KDA verify kernel must handle "
                "every verify batch"
            )
        self._capture_replay_payload(
            layer_id,
            mixed_qkv,
            f_a_out,
            beta_raw,
            batch_size=batch_size,
            draft_token_num=draft_token_num,
            weights=(
                conv_weights,
                f_b_weight,
                A_log,
                dt_bias,
                num_value_heads,
                head_v_dim,
                lower_bound,
            ),
        )
        return fused_out

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
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Run only the real-token prefix through the KDA prefill kernel.

        FlashKDA sizes its output from the input shape but tiles from
        ``cu_seqlens``: bucket-padded inputs would leave the output tail
        unwritten and feed padding into its final full-tile loads. The graph
        handoff clears and restores the bucket tail afterward.
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
            lower_bound=lower_bound,
            solution=None if self.kda_backend == "auto" else self.kda_backend,
        )

        return kda_result.out.squeeze(0), kda_result.final_state

    def _kda_lazy_buffers(self, min_slots: int = 1) -> dict:
        """Stable composed control buffers for the fused lazy commit.

        CUDA-graph capture records these tensors' storage. They are sized
        ONCE at the engine's max batch (``config.max_bs``, a hard startup
        bound on any batch, eager or graphed) and refreshed in place at
        every verify metadata prep -- so there is no growth, no post-capture
        seal, and no second buffer set for eager batches above the graph
        ceiling. Contents per slot: the payload base row and step count of
        that request's pending window (base ``-1`` = no pending, plain
        verify), the anchor page the fused kernel reads (pre-pending-window
        for pendings, the committed page for fresh requests), and the page
        the deferred commit stores to (``-1`` skips).
        """
        bufs = self._kda_lazy_bufs
        if bufs is None:
            # Round up so every row of the int32 buffers starts 16B-aligned,
            # once and forever: the compose kernel takes row slices, and
            # Triton keys its compiled variants on pointer alignment.
            cap = (max(self.max_bs, len(self.query_start_loc_list), 1) + 3) & ~3
            assert not torch.cuda.is_current_stream_capturing()
            with torch.inference_mode(False):
                gids = self._state_group_ids
                # One storage so the per-round neutralize is two fills, not 2+2G.
                flat = torch.full(
                    (2 + 2 * len(gids), cap), -1, dtype=torch.int32, device=self.device
                )
                bufs = self._kda_lazy_bufs = {
                    "flat": flat,
                    "base": flat[0],
                    "steps": flat[1],
                    "anchor": {g: flat[2 + i] for i, g in enumerate(gids)},
                    "commit": {g: flat[2 + len(gids) + i] for i, g in enumerate(gids)},
                    "capture_base": torch.zeros(
                        1, dtype=torch.int64, device=self.device
                    ),
                }
        assert min_slots <= bufs["base"].shape[0], (
            f"KDA control buffers hold {bufs['base'].shape[0]} slots but the "
            f"round needs {min_slots}; config.max_bs no longer bounds the batch"
        )
        return bufs

    def _kda_table(self) -> torch.Tensor:
        """The device slot table, sized to the request pool.

        ``table[rpi]`` is the pending payload slot that request owns (``-1``
        = none). Written at record (fill + scatter), by the compose kernel
        (condemning gate-failed rows in place), and by the host lifecycle
        screens (``_kda_table_drop``); read by the compose and by the
        flush's write gate. Everything is stream-ordered; the host never
        reads it back, which is what deleted the verdict round-trip this
        replaces.
        """
        return self._kda_slot_table

    def _kda_table_drop(self, rpis) -> None:
        """Clear the table rows of dropped requests, without a stream sync.

        The pinned staging is allocated at construction because its maximum
        width is fixed by the request-pool bound.
        """
        if self._kda_slot_table is None or not rpis:
            return
        host = self._kda_table_drop_staging[: len(rpis)]
        host.copy_(torch.tensor(rpis, dtype=torch.int64))
        dev = torch.empty_like(host, device=self.device)
        dev.copy_(host, non_blocking=True)
        self._kda_slot_table.index_fill_(0, dev, -1)

    def _kda_table_release(self) -> None:
        """Neutralize the whole table (the pending is gone as a unit)."""
        if self._kda_slot_table is not None:
            self._kda_slot_table.fill_(-1)

    def _stage_pending_replay(
        self,
        real_bs: int,
        padded_bs: int,
        kwargs: dict,
        state_in_by_group: dict[str, torch.Tensor] | None,
        committed: torch.Tensor | None = None,
        rpis_dev: torch.Tensor | None = None,
    ) -> None:
        """Compose this verify round's lazy-commit inputs from the pending
        record, flushing any pending request that left the verify batch.

        Runs on every target-verify metadata prep through the
        ``_stage_verify_round`` seam, before the forward launches. Staging
        writes the pending replay parameters into the device control rows the
        next fused verify reads. Must refresh the
        buffers on EVERY verify round -- a stale base would make the graphed
        kernel re-replay an already-committed prefix onto its own result,
        which double-applies the accepted tokens when the commit was in
        place. ``real_bs == 0`` (idle replay) neutralizes the buffers and
        LEAVES the pending: an empty batch is no evidence about residency.
        """
        bufs = self._kda_lazy_buffers(min_slots=padded_bs)
        pend = self._kda_pending
        # A round whose token rows exceed the frozen graph ring runs eagerly
        # on the free-growing overflow payload. The fused launch selects by
        # max(ceiling, batch)*T > ring rows as well (its batch equals
        # padded_bs on every call path), so control values and payload
        # always travel together.
        cache = self._replay_payload_cache
        overflow = (
            self._graphs_captured
            and cache is not None
            and max(len(self.query_start_loc_list), padded_bs)
            * int(self.speculative_num_draft_tokens)
            > cache["rows"]
        )
        self._kda_overflow_round = overflow
        if overflow:
            # Single-half overflow ring: capture at row 0, and the graph
            # ring's parity chain stays untouched -- no pending will be
            # recorded against this round (the commit runs standalone the
            # moment acceptance is known).
            bufs["capture_base"].fill_(0)
        # Payload-ring parity: this round's capture must write the half the
        # pending's replay does NOT read. Derived, not toggled -- an abandoned
        # prep (staged, forward never ran) must not flip the target half.
        elif self._payload_half_rows is None:
            # First round: the ring is allocated inside the forward, after this
            # stage, so capture_base is still its zero init -- half 0 it is.
            self._staged_parity = 0
        else:
            par = pend["parity"] if pend is not None else self._payload_parity
            self._staged_parity = 1 - par
            bufs["capture_base"].fill_(self._staged_parity * self._payload_half_rows)
        n = padded_bs
        # pad_slot_id and every other neutral value is -1; steps rides row 1.
        bufs["flat"][:, :n].fill_(-1)
        bufs["steps"][:n].fill_(0)
        if state_in_by_group is not None and real_bs > 0:
            for gid in self._state_group_ids:
                bufs["anchor"][gid][:real_bs].copy_(
                    state_in_by_group[gid][:real_bs].to(torch.int32)
                )
        if real_bs <= 0:
            # Fully padded round: nothing to fuse into, and an empty batch is
            # no evidence the owners left -- a capacity retraction produces
            # exactly this round with every other decoder still resident.
            # Leave the pending for a round that carries its owners.
            return
        pending = self._kda_pending
        if pending is None:
            return
        batch = kwargs.get("forward_batch")
        if batch is None or not batch.request_pool_indices:
            # No request identity: cannot fuse safely; commit eagerly, run plain.
            self._flush_kda_pending()
            return
        # Screen corpses and departures BEFORE any flush: a growth flush that
        # still holds a departed request would write its reclaimed pages.
        rpis = list(batch.request_pool_indices)[:real_bs]
        staged_req_ids = list(batch.request_ids)[:real_bs]
        _, corpses = self._kda_pending_owner_mask(rpis, staged_req_ids)
        if corpses:
            self._drop_kda_pending(corpses)
            pending = self._kda_pending
            if pending is None:
                return
        current = set(rpis)
        departed = [r for r in pending["slot_by_rpi"] if r not in current]
        if departed:
            self._drop_kda_pending(departed)
            pending = self._kda_pending
            if pending is None:
                return
        cache = self._replay_payload_cache
        needed_rows = max(len(self.query_start_loc_list), real_bs) * int(
            self.speculative_num_draft_tokens
        )
        if cache is not None and cache["rows"] < needed_rows:
            # The pending's ring and this round's ring diverge here -- eager
            # growth rebuilds the buffers the replay reads, and an overflow
            # round writes a different set entirely. Commit before staging.
            self._flush_kda_pending()
            return
        t_prev = pending["draft_token_num"]
        t_now = int(
            kwargs.get("tokens_per_req", 0)
            or self.speculative_num_draft_tokens
            or t_prev
        )
        if t_prev != t_now:
            # Payload is in t_prev units: a different window size would mis-replay.
            self._flush_kda_pending()
            return
        from tokenspeed_kernel.ops.metadata import kda_arm_compose

        # One launch for every control row. Ownership comes from the device
        # slot table; a row failing the identity or causal gate is condemned
        # in the table by the compose itself, so its CPU dict entry can
        # linger harmlessly (every later write gates on the table). The
        # anchor rows arrive holding this round's committed pages, which the
        # compose reads as both the identity reference and the no-pending
        # fallback before overwriting them.
        assert rpis_dev is not None and rpis_dev.shape[0] >= real_bs
        n_groups = len(self._state_group_ids)
        kda_arm_compose(
            bufs["flat"],
            self._kda_table(),
            rpis_dev[:real_bs],
            pending["steps"],
            pending["expect"],
            pending["anchor_stack"],
            pending["commit_stack"],
            None if committed is None else committed[:real_bs],
            bufs["flat"][2 : 2 + n_groups],
            base_offset=pending["parity"] * self._payload_half_rows,
            draft_token_num=t_prev,
            num_groups=n_groups,
        )
        # The record stays live until the forward it was staged for is issued:
        # an abandoned prep still owes the window a flush. `notify_forward_issued`
        # releases it -- NOT the next round's record, which never arrives when
        # the round dies between the KDA layers and the accept sampling.
        self._kda_replay_staged = True

    def _resolve_kda_pending_before_forward(
        self, bs: int, req_pool_indices, num_extends: int, kwargs: dict
    ) -> None:
        """Resolve pending windows whose owners sit in a non-verify batch.

        Owners among the extend rows are having their state rebuilt from
        scratch -- a retract-resume, or a fresh request on a recycled pool
        slot that kept the request id. The recorded window belongs to the
        state being replaced, so it is dropped, never written: this is the
        one credential that separates "retracted, pages freed" from "still
        decoding", which no host-side residency check can supply. Every
        other owner is a resident decoder and commits here, before the
        forward touches its state.
        """
        batch = kwargs.get("forward_batch")
        batch_ids = list(batch.request_ids)[:bs] if batch is not None else []
        # request_pool_indices is host-side on the op; the device tensor's
        # .tolist() would drain the stream (and the overlapped prior step).
        batch_rpis = (
            list(batch.request_pool_indices)[:bs]
            if batch is not None
            else req_pool_indices[:bs].tolist()
        )
        owners, corpses = self._kda_pending_owner_mask(batch_rpis, batch_ids)
        if corpses:
            self._drop_kda_pending(corpses)
        if not owners or self._kda_pending is None:
            return
        # Extend requests are the first num_extends rows of the batch.
        rebuilt = set(batch_rpis[:num_extends])
        stale = [r for r in owners if r in rebuilt]
        if stale:
            self._drop_kda_pending(stale)
            owners = [r for r in owners if r not in rebuilt]
        if owners and self._kda_pending is not None:
            gate = self._kda_owner_commit_gate(owners, batch_rpis, kwargs)
            self._flush_kda_pending(only_rpis=owners, commit_gate_by_group=gate)

    @override
    def _resolve_pending_before_forward(
        self,
        bs: int,
        req_pool_indices: torch.Tensor,
        num_extends: int,
        kwargs: dict,
    ) -> None:
        """Resolve KDA's cross-round replay window before shared page setup."""
        if self._kda_pending is not None:
            self._resolve_kda_pending_before_forward(
                bs, req_pool_indices, num_extends, kwargs
            )

    @override
    def _stage_verify_round(
        self,
        real_bs: int,
        padded_bs: int,
        kwargs: dict,
        state_in_blocks_by_group: dict[str, torch.Tensor] | None,
        verify_committed: torch.Tensor | None,
        req_pool_indices: torch.Tensor,
    ) -> None:
        """Stage KDA's fused deferred replay after shared verify page setup."""
        if self._replay_active:
            self._stage_pending_replay(
                real_bs,
                padded_bs,
                kwargs,
                state_in_blocks_by_group,
                verify_committed,
                rpis_dev=req_pool_indices,
            )

    def _kda_pending_owner_mask(self, rpis, req_ids):
        """Split batch rpis into (owners, corpses) against the pending.

        The engine request id is the one credential pool-index recycling
        cannot forge. ``None`` ids (no identity available) count as owners,
        matching the pre-identity behavior.
        """
        pending = self._kda_pending
        if pending is None:
            return [], []
        id_by_rpi = pending.get("id_by_rpi", {})
        owners, corpses = [], []
        for i, r in enumerate(rpis):
            if r not in pending["slot_by_rpi"]:
                continue
            rec = id_by_rpi.get(r)
            cur = req_ids[i] if i < len(req_ids) else None
            if rec is None or cur is None or rec == cur:
                owners.append(r)
            else:
                corpses.append(r)
        return owners, corpses

    def _drop_kda_pending(self, only_rpis) -> None:
        """Discard pending windows for requests that left the engine.

        A request absent from the batch has been finished or retracted, and
        on the LCM pool that means its state pages are already reclaimed --
        possibly re-assigned and re-written by a NEW request's prefill.
        Replaying the dead window would overwrite the new owner's state
        (observed as all-[PAD] outputs after client aborts), and a retracted
        request resumes with a full prefill, so its state is rebuilt from
        scratch anyway. Dropping, never writing, is the only safe commit.
        """
        pending = self._kda_pending
        if pending is None:
            return
        dropped = [r for r in only_rpis if r in pending["slot_by_rpi"]]
        self._kda_table_drop(dropped)
        for r in dropped:
            pending["slot_by_rpi"].pop(r, None)
        if not pending["slot_by_rpi"]:
            self._kda_pending = None

    def _kda_owner_commit_gate(self, owners, batch_rpis, kwargs):
        """Per-group device mask: the recorded commit page must still sit in
        its owner's current page-table row.

        Same-rpi + same-request-id is not proof of page ownership: a
        retract-resume frees the pages and re-prefills, and the freed page
        may already hold another request's state. Membership in the owner's
        CURRENT table row is the one page-level credential available without
        a stream sync; absent rows are dropped by the flush, never written.
        Returns None (no gating) when the batch carries no cache metadata.
        """
        pending = self._kda_pending
        cache_metadata = kwargs.get("cache_metadata")
        batch = kwargs.get("forward_batch")
        if cache_metadata is None or batch is None:
            return None
        old_bs = pending["steps"].shape[0]
        dev = self.device
        # Fresh pinned staging per call (never persistent, never pageable):
        # a pageable copy here would synchronize and drain the overlapped
        # previous step.
        host = torch.empty(2, len(owners), dtype=torch.int64, pin_memory=True)
        host[0].copy_(torch.tensor([batch_rpis.index(r) for r in owners]))
        host[1].copy_(torch.tensor([pending["slot_by_rpi"][r] for r in owners]))
        staged = torch.empty_like(host, device=dev)
        staged.copy_(host, non_blocking=True)
        pos, slots = staged[0], staged[1]
        gate_by_group = {}
        for gid in self._state_group_ids:
            rows = cache_metadata.require_table(gid, active_forward_op=batch)
            commit = pending["commit_by_group"][gid].to(torch.int64).gather(0, slots)
            present = (rows[pos].to(torch.int64) == commit[:, None]).any(dim=1)
            gate = torch.zeros(old_bs, dtype=torch.bool, device=dev)
            gate[slots] = present
            gate_by_group[gid] = gate
        return gate_by_group

    def _flush_kda_pending(self, only_rpis=None, commit_gate_by_group=None) -> None:
        """Commit pending windows with the standalone replay kernels.

        The escape hatch for a pending that cannot ride the next fused
        verify while its owner is still resident: identity-less batches,
        payload-capacity growth, mode switches, pause/weight-update
        freezes. Requests that LEFT the engine must go through
        ``_drop_kda_pending`` instead -- their pages may already belong to
        someone else. ``only_rpis`` restricts the flush; the rest stay
        pending for fusion. ``commit_gate_by_group`` (device masks from
        ``_kda_owner_commit_gate``) turns rows whose pages moved into
        drops -- consumed, never written.
        """
        # Unreachable during startup capture today; keep it loud, not silent.
        assert (
            not torch.cuda.is_current_stream_capturing()
        ), "KDA pending flush must not run inside CUDA graph capture"
        pending = self._kda_pending
        if pending is None:
            return
        from tokenspeed_kernel.ops.attention import try_kda_replay_commit

        slot_by_rpi = pending["slot_by_rpi"]
        targets = set(slot_by_rpi if only_rpis is None else only_rpis) & set(
            slot_by_rpi
        )
        if not targets:
            return
        old_bs = pending["steps"].shape[0]
        t_prev = pending["draft_token_num"]
        weights = self._replay_layer_weights
        if not weights:
            raise RuntimeError("KDA pending flush has no captured verify projections")
        mask = torch.zeros(old_bs, dtype=torch.bool, device=self.device)
        flush_slots = torch.tensor(
            [slot_by_rpi[r] for r in targets], dtype=torch.int64, device=self.device
        )
        mask[flush_slots] = True
        # The slot table gates every write: a row the compose condemned (or
        # a drop cleared) since this pending was recorded self-masks here,
        # with no verdict ever crossing back to the host.
        table = self._kda_table()
        owner_rows = pending["rpi_by_slot"]
        mask &= table.gather(0, owner_rows) == torch.arange(
            old_bs, dtype=torch.int32, device=self.device
        )
        rows = old_bs * t_prev
        for layer_id in self._state_layer_ids():
            if layer_id not in weights:
                # A layer absent from the capture never advanced state: nothing to commit.
                continue
            gid = self.kv_pool.group_id_for_layer(layer_id)
            (
                conv_w,
                f_b_weight,
                A_log,
                dt_bias,
                num_heads,
                head_dim,
                lower_bound,
            ) = weights[layer_id]
            if pending["overflow"]:
                qkv_buf, f_a_buf, beta_buf = self._kda_overflow_payload[layer_id]
            else:
                qkv_buf, f_a_buf, beta_buf = self._replay_payload_cache["buffers"][
                    layer_id
                ]
                po = pending["parity"] * self._payload_half_rows
                qkv_buf, f_a_buf, beta_buf = qkv_buf[po:], f_a_buf[po:], beta_buf[po:]
            conv_comp = self.kv_pool.get_component(layer_id, "conv_state")
            ssm_comp = self.kv_pool.get_component(layer_id, "recurrent_state")
            row_mask = mask
            if commit_gate_by_group is not None:
                row_mask = mask & commit_gate_by_group[gid]
            write = torch.where(
                row_mask,
                pending["commit_by_group"][gid].to(torch.int64),
                torch.full((old_bs,), -1, dtype=torch.int64, device=self.device),
            ).to(torch.int32)
            ok = try_kda_replay_commit(
                qkv_buf[:rows],
                conv_w,
                conv_comp,
                conv_comp,
                f_a_buf[:rows],
                f_b_weight,
                beta_buf[:rows],
                A_log,
                dt_bias,
                state_pool=ssm_comp,
                state_out=ssm_comp,
                read_indices=pending["anchor_by_group"][gid][:old_bs],
                write_indices=write,
                accepted_length=pending["steps"],
                num_heads=num_heads,
                head_dim=head_dim,
                draft_token_num=t_prev,
                lower_bound=lower_bound,
                gate_scratch=self._kda_gate_scratch(rows, num_heads * head_dim),
            )
            if not ok:
                raise RuntimeError(
                    "KDA replay commit kernel vanished after the capability "
                    "probe reported it available"
                )
        # Committed rows leave the table so nothing can write them twice.
        table.index_fill_(0, owner_rows.gather(0, flush_slots), -1)
        remaining = {r: i for r, i in slot_by_rpi.items() if r not in targets}
        if remaining:
            pending["slot_by_rpi"] = remaining
        else:
            self._kda_pending = None

    def has_pending(self) -> bool:
        """A lazy KDA window is still awaiting its commit."""
        return self._kda_pending is not None

    def notify_forward_issued(self) -> None:
        """Release the record the just-issued forward commits device-side.

        The fused verify applies the pending window inside the KDA layers,
        far upstream of the accept sampling whose result records the next
        one. Ownership has to end where the work lands on the device, not
        where the next record happens to arrive: any round that reaches the
        layers but not ``commit_verified_state`` would otherwise leave an
        already-applied window live, and with the anchor and commit pages
        equal -- as they are for every window that stays inside one state
        page -- the next round replays it onto its own result.
        """
        if self._kda_replay_staged:
            self._kda_replay_staged = False
            self._kda_pending = None
            self._kda_table_release()

    def flush_pending(self, resident_request_ids: set[str]) -> None:
        """Resolve every pending KDA window now (lifecycle escape hatch).

        Must run before anything invalidates the replay inputs: a weight
        update (replay uses the layer weights), a pause writeback, or any
        teardown that releases state pages.

        There is no batch here, so the page-table credential a forward round
        uses is unavailable and the engine's residency set is the only one
        left. A window whose owner has left is dropped, never written: its
        state pages are reclaimed, and the common case is exactly this --
        every request's last verify records a window for a request that has
        already finished.

        Known gap: a capacity-retracted owner stays in the residency set
        (it will resume) while its pages are already freed. Closing it
        needs the scheduler to expose its Decoding set; until then a pause
        landing between a retraction and that owner's resume can commit
        onto a reassigned page.
        """
        pending = self._kda_pending
        if pending is None:
            return
        departed = [
            r
            for r, rid in pending["id_by_rpi"].items()
            if rid is not None and rid not in resident_request_ids
        ]
        if departed:
            self._drop_kda_pending(departed)
        self._flush_kda_pending()

    def _replay_payload(
        self,
        layer_id: int,
        rows: int,
        widths: tuple[int, int, int],
        dtype: torch.dtype,
    ):
        """Per-layer capture buffers for the replay payload of one window.

        Replay needs the projections the TARGET model computed while
        verifying: the pre-convolution packed ``q|k|v``, the low-rank gate
        input, and the raw beta logits. Those are ~9.3 KiB per token per
        layer at the K3 TP8 geometry, against ~795 KiB for the recurrent
        state and conv window a per-position scratch would have held -- 86x
        less, which is the whole point of replaying.

        They are copied rather than referenced because the layer forward is
        captured into a CUDA graph and does not re-run on replay: a stashed
        reference would point at whichever bs bucket was captured last, while
        a copy into a stable buffer replays correctly for every bucket.

        Args:
            layer_id: KDA layer to get buffers for.
            rows: capacity in tokens (``max_bs * draft_token_num``).
            widths: ``(3*P, D_FA, HV)`` channel counts of the three payloads.
            dtype: activation dtype of the captured projections.

        Returns:
            ``(qkv_raw, f_a, beta)`` buffers, each ``[rows, width]``.
        """
        cache = self._replay_payload_cache
        if (
            self._graphs_captured
            and cache is not None
            and rows > cache["rows"]
            and widths == cache["widths"]
        ):
            # Eager round above the graph ceiling: the graph ring is frozen,
            # so the payload lands in the free-growing overflow set. Its
            # window never crosses a round (committed standalone at record),
            # so a single half suffices and no graph ever reads it.
            return self._kda_overflow_payload_buffers(layer_id, rows, widths, dtype)
        if cache is None or cache["rows"] < rows or cache["widths"] != widths:
            assert not self._graphs_captured, (
                f"KDA payload ring grew to {rows} rows after graph capture; "
                "captured graphs still hold the old address"
            )
            # A graphed forward can never rebuild: warmup pre-sizes the ring.
            assert (
                not torch.cuda.is_current_stream_capturing()
            ), "KDA payload ring must be pre-sized before graph capture"
            # Rebuild would replace the tensors a live replay reads: commit first.
            if cache is not None and self._kda_pending is not None:
                self._flush_kda_pending()
            cache = self._replay_payload_cache = {
                "rows": rows,
                "widths": widths,
                "buffers": {},
            }
            # Ring rows double for the round-parity halves; base offsets and
            # capture_base are parity * rows, kept coherent by this one field.
            self._payload_half_rows = rows
            bufs = self._kda_lazy_bufs
            if bufs is not None:
                # The stage filled these against the OLD half size; recompute so
                # this round's capture and next round's replay agree on rows.
                bufs["capture_base"].fill_(self._staged_parity * rows)
                bufs["base"].fill_(-1)
            # The startup overflow preallocation guessed its widths from the
            # pool geometry; this rebuild carries the model's true widths, so
            # re-drive it here (still pre-capture) or the guess would be
            # discarded by the width check during an overflow round -- the
            # mid-serving cudaMalloc the preallocation exists to avoid.
            for stale_id in [
                lid
                for lid, entry in self._kda_overflow_payload.items()
                if tuple(t.shape[1] for t in entry) != tuple(widths)
            ]:
                overflow_rows = self._kda_overflow_payload[stale_id][0].shape[0]
                del self._kda_overflow_payload[stale_id]
                self._kda_overflow_payload_buffers(
                    stale_id, overflow_rows, widths, dtype
                )
        buffers = cache["buffers"]
        entry = buffers.get(layer_id)
        if entry is None:
            # Fallback for enforce-eager / odd f_a widths; must never fire in capture.
            assert (
                not torch.cuda.is_current_stream_capturing()
            ), "KDA replay payload buffers must exist before graph capture"
            # Outside inference mode: the flush path feeds these to kernels from prep.
            with torch.inference_mode(False):
                entry = buffers[layer_id] = tuple(
                    torch.zeros(
                        (2 * cache["rows"], width), dtype=dtype, device=self.device
                    )
                    for width in widths
                )
        return entry

    def _kda_overflow_payload_buffers(
        self,
        layer_id: int,
        rows: int,
        widths: tuple[int, int, int],
        dtype: torch.dtype,
    ):
        """Per-layer payload buffers for an eager round above the graph
        ceiling; see ``_replay_payload`` for the buffers' role."""
        assert not torch.cuda.is_current_stream_capturing()
        store = self._kda_overflow_payload
        entry = store.get(layer_id)
        # Width equality matters beyond capacity: the kernels take the row
        # stride as a constexpr, so a differently-wide buffer would compile
        # (and module-load) fresh variants mid-decode.
        if (
            entry is None
            or entry[0].shape[0] < rows
            or tuple(t.shape[1] for t in entry) != tuple(widths)
        ):
            with torch.inference_mode(False):
                entry = store[layer_id] = tuple(
                    torch.zeros((rows, width), dtype=dtype, device=self.device)
                    for width in widths
                )
        return entry

    def _capture_replay_payload(
        self,
        layer_id: int,
        mixed_qkv: torch.Tensor,
        f_a_out: torch.Tensor,
        beta_raw: torch.Tensor,
        *,
        batch_size: int,
        draft_token_num: int,
        weights: tuple,
    ) -> None:
        """Record one KDA layer's replay weights and payload bookkeeping.

        The projections themselves are staged in-kernel by the conv-window
        commit (CAPTURE mode). ``weights`` holds the layer's model-lifetime
        tensors plus its head geometry; those are bs-independent, so
        recording them by reference is safe across CUDA-graph buckets in a
        way the projections themselves are not.
        """
        max_rows = max(len(self.query_start_loc_list), batch_size) * draft_token_num
        widths = (mixed_qkv.shape[-1], f_a_out.shape[-1], beta_raw.shape[-1])
        qkv_buf, f_a_buf, beta_buf = self._replay_payload(
            layer_id, max_rows, widths, mixed_qkv.dtype
        )
        # Staged in-kernel by the conv-window commit (CAPTURE mode).
        state = self._replay_layer_weights
        if state is None:
            state = self._replay_layer_weights = {}
        state[layer_id] = weights

    def _ensure_verify_scratch(self, bs: int, draft_token_num: int) -> None:
        """Lazily allocate per-group verify scratch: one init row plus
        ``draft_token_num`` per-position rows per request, for both the conv
        window and the recurrent state (rollback source for partial accepts).
        Sized once at the max the backend can see; graph warmup runs this
        path eagerly before capture.

        Skipped entirely on the replay path, which commits from the committed
        page instead of from a per-position row."""
        if self._replay_active:
            return
        max_bs = max(len(self.query_start_loc_list), bs)
        rows_needed = max_bs * (draft_token_num + 1)
        scratch = self._verify_scratch
        if scratch is not None and next(iter(scratch.values()))[0].shape[0] >= (
            rows_needed
        ):
            return
        self._verify_scratch = {}
        self._verify_copy_tables = None
        for layer_id in self._state_layer_ids():
            conv, ssm = self._state_components(layer_id)
            self._verify_scratch[layer_id] = (
                torch.zeros(
                    (rows_needed, *conv.shape[1:]),
                    dtype=conv.dtype,
                    device=conv.device,
                ),
                torch.zeros(
                    (rows_needed, *ssm.shape[1:]),
                    dtype=ssm.dtype,
                    device=ssm.device,
                ),
            )

    def preallocate_verify_workspace(self, max_bs: int, draft_token_num: int) -> int:
        """Allocate graph-stable verify state and return its byte size."""
        if not self.state_paging_active or self.is_draft:
            return 0
        self._ensure_verify_scratch(max_bs, draft_token_num)
        # The replay path deliberately allocates no scratch.
        scratch = self._verify_scratch
        if scratch is None:
            return 0
        return sum(
            tensor.nbytes
            for layer_scratch in scratch.values()
            for tensor in layer_scratch
        )

    def commit_verified_state(self, accepted_length: torch.Tensor) -> None:
        """Commit the state for the accepted draft prefix into each group's
        state slab at the new committed page. All device-side; graph-safe.

        On the replay path the accepted prefix is re-run from the committed
        page (see ``_replay_active``); otherwise the accepted position's
        row is copied out of the per-position verify scratch."""
        ctx = getattr(self, "_verify_commit_ctx", None)
        if ctx is None:
            return
        committed, tables, draft_token_num, read_pages_by_group = ctx[:4]
        bs = accepted_length.shape[0]
        k = accepted_length.to(torch.int64).clamp(min=1, max=draft_token_num)
        new_last = committed[:bs] + k - 1
        slot = torch.div(new_last, self._checkpoint_granularity, rounding_mode="floor")
        stride = draft_token_num + 1
        src_rows = (
            torch.arange(bs, dtype=torch.int64, device=accepted_length.device) * stride
            + k
        )
        pages_by_group: dict[str, torch.Tensor] = {}
        for group_id in self._state_groups():
            rows_tbl = tables[group_id]
            slot_safe = slot.clamp(min=0, max=rows_tbl.shape[1] - 1)
            pages_by_group[group_id] = (
                rows_tbl[:bs]
                .gather(1, slot_safe.unsqueeze(1))
                .squeeze(1)
                .to(torch.int64)
                .clamp_min(0)
            )
        if self._replay_active:
            # Record what a replay needs; the NEXT verify's fused kernel commits it.
            rpis = ctx[4] if len(ctx) > 4 else []
            req_ids = ctx[5] if len(ctx) > 5 else []
            overflow = self._kda_overflow_round
            if not overflow:
                self._payload_parity = self._staged_parity
            # Group-major stacks: the next stage's compose reads all groups in
            # one launch, and the per-group views below share their storage.
            gids = self._state_group_ids
            anchor_stack = torch.stack([read_pages_by_group[g][:bs] for g in gids])
            commit_stack = torch.stack([pages_by_group[g] for g in gids])
            # The record replaces the pending wholesale, table included: rows
            # a stale generation owned are cleared before the new owners are
            # scattered in, and ``rpi_by_slot`` (the inverse map) is cloned
            # because the rpi input buffer is rewritten by the next prep.
            # Only REAL rows scatter: under graph replay ``bs`` is padded and
            # the pad rows' pool index is a live table row -- an entry there
            # would stage the padding as a replay next round.
            real = min(bs, len(rpis))
            rpis_dev = ctx[6][:real].to(torch.int64)
            table = self._kda_table()
            table.fill_(-1)
            # Same bounds contract as the compose: an out-of-range pool
            # index degrades that row to never-fuse (the sentinel row stays
            # -1) instead of a device-side assert poisoning the context.
            in_range = (rpis_dev >= 0) & (rpis_dev < table.shape[0])
            safe_rpis = torch.where(
                in_range, rpis_dev, torch.full_like(rpis_dev, table.shape[0] - 1)
            )
            slots = torch.arange(real, dtype=torch.int32, device=self.device)
            table.scatter_(0, safe_rpis, torch.where(in_range, slots, -1))
            # Pad slots point at the table's last row, which nothing ever
            # scatters a live slot into (legal pool indices stop at
            # max_bs + 1): the flush gate reads -1 there and they self-mask.
            rpi_by_slot = torch.full(
                (bs,), table.shape[0] - 1, dtype=torch.int64, device=self.device
            )
            rpi_by_slot[:real] = safe_rpis
            self._kda_pending = dict(
                parity=0 if overflow else self._staged_parity,
                overflow=overflow,
                rpi_by_slot=rpi_by_slot,
                slot_by_rpi={r: i for i, r in enumerate(rpis[:bs])},
                steps=k.to(torch.int32),
                expect=(committed[:bs].to(torch.int64) + k.to(torch.int64)),
                id_by_rpi={
                    r: (req_ids[i] if i < len(req_ids) else None)
                    for i, r in enumerate(rpis[:bs])
                },
                anchor_stack=anchor_stack,
                commit_stack=commit_stack,
                anchor_by_group={g: anchor_stack[i] for i, g in enumerate(gids)},
                commit_by_group={g: commit_stack[i] for i, g in enumerate(gids)},
                draft_token_num=draft_token_num,
            )
            self._verify_commit_ctx = None
            if not rpis:
                # A slotless pending can never fuse or flush; recording it
                # would silently lose the accepted window.
                raise RuntimeError(
                    "KDA replay commit needs request identities on the batch"
                )
            if overflow:
                # An overflow window never crosses a round: no captured graph
                # can read its ring, so the next verify could not fuse it and
                # a later flush might find the buffers regrown. Acceptance is
                # known right here (device-side), so commit standalone now.
                self._kda_overflow_round = False
                self._flush_kda_pending()
            return
        # Batched commit: scratch row -> committed page, one launch per state kind.
        from tokenspeed_kernel.ops.kvcache.triton import copy_state_rows

        copy_tables = self._verify_copy_tables_get()
        pages_stack = torch.stack(
            [pages_by_group[group_id] for group_id in self._state_groups()]
        )
        dst_rows = pages_stack.index_select(0, copy_tables["group_sel"]).reshape(-1)
        src_tiled = src_rows.repeat(copy_tables["num_layers"])
        copy_state_rows(
            copy_tables["conv_scratch"],
            copy_tables["conv_comp"],
            src_tiled,
            dst_rows,
            row_bytes=copy_tables["conv_bytes"],
            src_row_strides=copy_tables["conv_scratch_stride"],
            dst_row_strides=copy_tables["conv_comp_stride"],
        )
        copy_state_rows(
            copy_tables["ssm_scratch"],
            copy_tables["ssm_comp"],
            src_tiled,
            dst_rows,
            row_bytes=copy_tables["ssm_bytes"],
            src_row_strides=copy_tables["ssm_scratch_stride"],
            dst_row_strides=copy_tables["ssm_comp_stride"],
        )
        self._verify_commit_ctx = None

    def init_forward_metadata(
        self,
        bs: int,
        req_pool_indices: torch.Tensor,
        seq_lens: torch.Tensor,
        forward_mode: ForwardMode = ForwardMode.DECODE,
        **kwargs,
    ):
        self._kda_replay_staged = False
        return super().init_forward_metadata(
            bs, req_pool_indices, seq_lens, forward_mode, **kwargs
        )

    def init_cuda_graph_state(self, max_num_tokens: int):
        for i in range(max_num_tokens):
            self.query_start_loc_list.append(
                torch.empty((i + 2,), dtype=torch.int32, device=self.device)
            )
            # Keep one graph-stable dual-index buffer pair per state group.
            for gid in self._state_group_ids:
                self.state_in_by_group.setdefault(gid, []).append(
                    torch.full(
                        (i + 1,),
                        self.pad_slot_id,
                        dtype=torch.int32,
                        device=self.device,
                    )
                )
                self.state_out_by_group.setdefault(gid, []).append(
                    torch.full(
                        (i + 1,),
                        self.pad_slot_id,
                        dtype=torch.int32,
                        device=self.device,
                    )
                )
        self.cached_cuda_graph_decode_query_start_loc = torch.arange(
            0, max_num_tokens + 1, dtype=torch.int32, device=self.device
        )
        if self.speculative_num_draft_tokens > 0:
            # Need max_num_tokens+1 entries (one per request + sentinel).
            # Each entry is request_index * spec_num_draft_tokens.
            self.cached_cuda_graph_verify_query_start_loc = torch.arange(
                0,
                (max_num_tokens + 1) * self.speculative_num_draft_tokens,
                step=self.speculative_num_draft_tokens,
                dtype=torch.int32,
                device=self.device,
            )
        self._qsl_dirty = [False] * max_num_tokens
        self._qsl_last_mode = [None] * max_num_tokens

    def _preallocate_kda_replay_buffers(self, max_bs: int) -> None:
        """Allocate every lazy-commit buffer before any forward runs.

        Called from ``set_kv_pool``, when pool geometry first becomes known and
        before any warmup, capture, or inference-mode context. Keeping the
        forward path allocation-free avoids inference-mode tensor restrictions
        and any chance of a first touch landing inside a capture mempool.

        The f_a payload width is not in the pool geometry; KDA's low-rank
        gate uses the state head dim (K3: 128 == K). If a model ever
        disagrees, the width check in ``_replay_payload`` rebuilds the
        buffers during the eager warmup forward, before anything captures.
        """
        rows = max(max_bs, 1) * int(self.speculative_num_draft_tokens)
        hv = k = None
        for layer_id in self._state_layer_ids():
            conv = self.kv_pool.get_component(layer_id, "conv_state")
            ssm = self.kv_pool.get_component(layer_id, "recurrent_state")
            hv, k = ssm.shape[1], ssm.shape[2]
            self._replay_payload(
                layer_id,
                rows,
                (conv.shape[1], k, hv),
                self.dtype,
            )
        if hv is not None:
            # Warm the shared workspace to the gate scratch's true peak: the
            # engine's max batch, not the graph ceiling, because batches above
            # the ceiling run eagerly against the same frozen pool. Two
            # halves -- this round's tokens and the pending window's.
            self._kda_gate_scratch(2 * self.max_bs * self.spec_num_tokens, hv * k)
            overflow_rows = self.max_bs * int(self.speculative_num_draft_tokens)
            if overflow_rows > rows:
                # The engine can schedule eager batches above the graph ring:
                # claim their payload at startup, inside the memory budget,
                # instead of cudaMalloc-ing (a device sync) mid-serving under
                # the very load spike that triggers the overflow.
                for layer_id in self._state_layer_ids():
                    conv = self.kv_pool.get_component(layer_id, "conv_state")
                    self._kda_overflow_payload_buffers(
                        layer_id, overflow_rows, (conv.shape[1], k, hv), self.dtype
                    )
        self._kda_lazy_buffers(min_slots=max_bs)
        self._kda_table()

    def _kda_gate_scratch(self, rows: int, width: int) -> torch.Tensor:
        """This round's gate scratch, carved from the shared workspace pool.

        The gate is produced and consumed inside one layer's launch chain, so
        the per-op pool fits; re-fetched every use per the pool contract (the
        bytes are shared with every other op on the stream, the address is
        only stable once the pool freezes).

        PDL invariant: sharing is safe because no programmatic-dependent
        launch chain crosses from this op into another pool consumer -- the
        KDA chain ends at its rmsnorm and the next op launches normally,
        which serializes. A future consumer that PDL-chains across ops could
        start writing the block while the window kernel still reads the
        gate; it must bring its own buffer.
        """
        from tokenspeed.runtime.execution.workspace import workspace_pool

        (gate,) = workspace_pool(torch.device(self.device)).allocate(
            ((rows, width), torch.float32)
        )
        return gate

    def init_forward_metadata_capture_cuda_graph(
        self,
        bs: int,
        req_pool_indices: torch.Tensor,
        seq_lens: torch.Tensor,
        forward_mode: ForwardMode,
        **kwargs,
    ):
        if not self._graphs_captured and self._replay_active:
            self._graphs_captured = True
        self._kda_replay_staged = False
        return super().init_forward_metadata_capture_cuda_graph(
            bs, req_pool_indices, seq_lens, forward_mode, **kwargs
        )

    def init_forward_metadata_replay_cuda_graph(
        self,
        bs: int,
        req_pool_indices: torch.Tensor,
        seq_lens: torch.Tensor,
        forward_mode: ForwardMode = None,
        page_table: torch.Tensor = None,
        **kwargs,
    ):
        self._kda_replay_staged = False
        return super().init_forward_metadata_replay_cuda_graph(
            bs, req_pool_indices, seq_lens, forward_mode, page_table, **kwargs
        )


class HybridKDABackend(HybridLinearAttnBackend):
    """Composite backend for KDA hybrid models (full attention + KDA layers).

    Identical to ``HybridLinearAttnBackend`` today; it exists so KDA-only
    composite surface (deferred-commit settlement, lifecycle hooks) has a
    home that other linear hybrids never inherit.
    """

    @override
    def has_pending(self) -> bool:
        return self.linear_attn_backend.has_pending()

    @override
    def flush_pending(self, resident_request_ids: set[str]) -> None:
        self.linear_attn_backend.flush_pending(resident_request_ids)

    @override
    def settle_deferred_state(self, accepted_length):
        """Release issued replay work, then record a verified state window."""
        self.linear_attn_backend.notify_forward_issued()
        if accepted_length is not None:
            self.linear_attn_backend.commit_verified_state(accepted_length)
