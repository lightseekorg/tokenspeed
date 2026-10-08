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

"""Mamba2 (SSD) layers on the shared recurrent-state backend."""

from __future__ import annotations

from collections import deque
from typing import TYPE_CHECKING, Any

import torch
from tokenspeed_kernel.ops.attention.mamba2 import (
    Mamba2ChunkMetadata,
    build_mamba2_chunk_metadata,
    mamba2_chunk_scan,
    mamba2_replay_commit,
    mamba2_state_update,
    mamba2_verify_scan,
)

from tokenspeed.runtime.layers.attention.backends.state.prefill_capacity import (
    CapacityPrefillBackend,
    CapacityPrefillMetadata,
)
from tokenspeed.runtime.layers.attention.configs.linear_attn import Mamba2Config
from tokenspeed.runtime.utils.tensor import upload_packed

if TYPE_CHECKING:
    from tokenspeed.runtime.layers.attention.configs.base import (
        AttnConfig,
        SoftmaxAttnConfig,
    )

_UNCLAMPED_DT = (0.0, float("inf"))


def _require_mamba2_inputs(
    D: torch.Tensor | None,
    b: torch.Tensor | None,
    g_raw: torch.Tensor | None,
    f_a_out: torch.Tensor | None,
    f_b_weight: torch.Tensor | None,
    beta_raw: torch.Tensor | None,
    lower_bound: float | None,
) -> torch.Tensor:
    """Return ``D``, refusing the GDN/KDA gate inputs the SSD scan has no use for."""
    if D is None:
        raise ValueError("Mamba2 layers must pass their D skip coefficient")
    unused = {
        "b": b,
        "g_raw": g_raw,
        "f_a_out": f_a_out,
        "f_b_weight": f_b_weight,
        "beta_raw": beta_raw,
        "lower_bound": lower_bound,
    }
    passed = sorted(name for name, value in unused.items() if value is not None)
    if passed:
        raise ValueError(f"Mamba2 layers take no {', '.join(passed)}")
    return D


class Mamba2AttnBackend(CapacityPrefillBackend):
    """Mamba2 SSD layers, e.g. Nemotron-H, on the shared recurrent-state flow.

    The model packs the conv channels as ``[C | B | x]``, so the shared split
    yields ``query = C`` and ``key = B`` with the B/C groups as key heads and
    the SSM state size as the key dim, and ``value = x``. ``a`` carries the
    raw per-head ``dt`` and ``D`` the skip coefficient; ``b`` is unused. Only
    the scan seams and the replay recurrence differ from GDN.
    """

    # The conv update and the state update both read the projection view by stride.
    _decode_packed_qkv_views = True
    # Chunk plans need only live bounds, so packing would add work to eager forwards.
    _capacity_layout_when_uncaptured = False

    def __init__(self, config: AttnConfig, spec: SoftmaxAttnConfig):
        super().__init__(config, spec)
        mamba2 = config.component(Mamba2Config)
        if mamba2 is None:
            raise ValueError("Mamba2AttnBackend requires a Mamba2Config component")
        if mamba2.dt_limit != _UNCLAMPED_DT:
            raise NotImplementedError(
                f"the Mamba2 state update does not clamp dt; got {mamba2.dt_limit}"
            )
        self._chunk_size = mamba2.chunk_size
        # Every layer of a forward passes the same fresh host bounds tensor; keep its plan.
        self._chunk_plans: deque[tuple[torch.Tensor, Mamba2ChunkMetadata]] = deque(
            maxlen=3
        )
        # Retained body/tail bounds are rewritten in place, so their plans are too.
        self._capacity_plans: list[tuple[torch.Tensor, Mamba2ChunkMetadata]] = []

    def _admits_capacity_prefill(self) -> bool:
        return True

    def _reset_prefill_metadata(self) -> None:
        super()._reset_prefill_metadata()
        self._capacity_plans.clear()

    def _refresh_captured_prefill(self, metadata: CapacityPrefillMetadata) -> None:
        """Rewrite the body and tail chunk plans of a retained shape in one upload.

        Each plan reserves ``extent // chunk_size + num_seqs`` chunks; the
        unused tail repeats the last offset, so those chunks are empty.
        """
        batch = metadata.prefill_checkpoint_batch
        device = metadata.query_start_loc.device
        parts: list[torch.Tensor] = []
        plans: list[Mamba2ChunkMetadata] = []
        for bounds, extent in (
            (batch.body_cu_seqlens_cpu, batch.body_token_indices.numel()),
            (batch.tail_cu_seqlens_cpu, batch.tail_token_indices.numel()),
        ):
            plan = self._capacity_plan(bounds, extent, device)
            live = build_mamba2_chunk_metadata(
                bounds, self._chunk_size, torch.device("cpu")
            )
            pad = plan.seq_idx.numel() - live.seq_idx.numel()
            parts += [
                torch.cat(
                    (live.cu_chunk_seqlens, live.cu_chunk_seqlens[-1:].expand(pad))
                ),
                torch.cat((live.seq_idx, live.seq_idx[-1:].expand(pad))),
                live.last_chunk_indices,
            ]
            plans.append(plan)
        uploaded = upload_packed(tuple(parts), device)
        for i, plan in enumerate(plans):
            targets = (plan.cu_chunk_seqlens, plan.seq_idx, plan.last_chunk_indices)
            for target, source in zip(targets, uploaded[3 * i : 3 * i + 3]):
                target.copy_(source)

    def _capacity_plan(
        self, bounds: torch.Tensor, extent: int, device: torch.device
    ) -> Mamba2ChunkMetadata:
        """The persistent plan of one retained bounds tensor, allocated on first use."""
        for known, plan in self._capacity_plans:
            if known is bounds:
                return plan
        num_seqs = bounds.numel() - 1
        max_chunks = extent // self._chunk_size + num_seqs
        plan = Mamba2ChunkMetadata(
            chunk_size=self._chunk_size,
            cu_chunk_seqlens=torch.zeros(
                max_chunks + 1, dtype=torch.int32, device=device
            ),
            last_chunk_indices=torch.zeros(num_seqs, dtype=torch.int32, device=device),
            seq_idx=torch.zeros(max_chunks, dtype=torch.int32, device=device),
        )
        self._capacity_plans.append((bounds, plan))
        return plan

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
        D: torch.Tensor | None,
        a: torch.Tensor | None,
        b: torch.Tensor | None,
        g_raw: torch.Tensor | None,
        f_a_out: torch.Tensor | None,
        f_b_weight: torch.Tensor | None,
        beta_raw: torch.Tensor | None,
        seq_len: int,
        num_real_tokens: int,
        lower_bound: float | None,
        inputs_packed: bool,
        cu_seqlens_cpu: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Chunked SSD scan from the gathered states; see the base seam.

        Returns ``([1, T, H, P]`` output, ``[bs, H, P, N]`` final states)``.
        Rows past the live boundaries stay zero.
        """
        del seq_len, num_real_tokens, inputs_packed
        D = _require_mamba2_inputs(
            D, b, g_raw, f_a_out, f_b_weight, beta_raw, lower_bound
        )
        if cu_seqlens_cpu is None:
            raise RuntimeError("the Mamba2 prefill scan needs host sequence bounds")
        x = value[0]
        out = torch.zeros_like(x)
        final_states = mamba2_chunk_scan(
            x,
            a,
            A_log,
            key[0],
            query[0],
            D,
            dt_bias,
            dt_limit=_UNCLAMPED_DT,
            initial_states=recurrent_state,
            cu_seqlens=query_start_loc,
            chunk_metadata=self._chunk_plan(cu_seqlens_cpu, x.device),
            out=out,
        )
        return out.unsqueeze(0), final_states

    def _chunk_plan(
        self, cu_seqlens_cpu: torch.Tensor, device: torch.device
    ) -> Mamba2ChunkMetadata:
        """Build a forward's chunk plan once and share it across its layers.

        The metadata builder makes the host bounds fresh per batch and never
        writes them in place, so tensor identity names one forward's bounds.
        A forward uses at most three (whole batch, checkpoint body and tail).
        Retained capacity bounds are refreshed in place and keep their own plans.
        """
        for bounds, plan in self._capacity_plans:
            if bounds is cu_seqlens_cpu:
                return plan
        for bounds, plan in self._chunk_plans:
            if bounds is cu_seqlens_cpu:
                return plan
        plan = build_mamba2_chunk_metadata(cu_seqlens_cpu, self._chunk_size, device)
        self._chunk_plans.append((cu_seqlens_cpu, plan))
        return plan

    def _verify_scan(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        ssm_comp: torch.Tensor,
        ssm_scratch: torch.Tensor | None,
        state_in_blocks: torch.Tensor,
        output_indices: torch.Tensor,
        *,
        A_log: torch.Tensor,
        dt_bias: torch.Tensor,
        D: torch.Tensor | None,
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
        """SSD steps through each request's verify window; see the base seam.

        ReplaySSM starts from the committed page and writes no state, leaving
        the accepted prefix to the replay commit. The scratch fallback starts
        from the seeded scratch row and writes one state per position.
        """
        D = _require_mamba2_inputs(
            D, b, g_raw, f_a_out, f_b_weight, beta_raw, lower_bound
        )
        groups, d_state = query.shape[2], query.shape[3]
        heads, head_dim = value.shape[2], value.shape[3]
        x = value.view(batch_size, draft_token_num, heads, head_dim)
        out = torch.empty_like(x)
        if self.replay_ssm:
            state, reads, writes = ssm_comp, state_in_blocks[:batch_size], None
        else:
            state = ssm_scratch
            reads = self._verify_scratch_base_rows(batch_size, draft_token_num)
            writes = output_indices
        mamba2_verify_scan(
            state,
            x,
            a.view(batch_size, draft_token_num, heads),
            A_log,
            key.view(batch_size, draft_token_num, groups, d_state),
            query.view(batch_size, draft_token_num, groups, d_state),
            D,
            dt_bias,
            state_indices=reads,
            dst_state_indices=writes,
            parent_indices=self._tree_parents(batch_size),
            null_slot=self.pad_slot_id,
            out=out,
        )
        return out.view(1, seq_len, heads, head_dim)

    def _replay_commit(
        self, payload: torch.Tensor, parameters: torch.Tensor, **tables: Any
    ) -> None:
        """Replay the SSD recurrence instead of the gated delta rule."""
        mamba2_replay_commit(payload, parameters, **tables)

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
        D: torch.Tensor | None,
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
        """One SSD step per request between paged states; see the base seam.

        Returns the ``[1, B, H, P]`` output. Padding rows (index ``-1``)
        neither read nor write state.
        """
        D = _require_mamba2_inputs(
            D, b, g_raw, f_a_out, f_b_weight, beta_raw, lower_bound
        )
        if output_gate is not None or norm_weight is not None or norm_eps is not None:
            raise ValueError("Mamba2 layers apply their gated norm outside the scan")
        x = value[:, 0]
        out = torch.empty_like(x)
        mamba2_state_update(
            ssm_states,
            x,
            a,
            A_log,
            key[:, 0],
            query[:, 0],
            D,
            dt_bias,
            state_indices=read_indices,
            dst_state_indices=write_indices,
            null_slot=self.pad_slot_id,
            out=out,
        )
        return out.unsqueeze(0)
