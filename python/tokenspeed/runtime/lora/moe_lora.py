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

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import torch

from tokenspeed.runtime.lora.lora_batch import NO_LORA_SLOT, LoraBatchInfo

try:
    from tokenspeed_kernel.ops.moe_lora import (
        gate_up_b_expand,
        per_expert_a_shrink,
        per_expert_b_down_expand,
        per_expert_gate_up_b_expand,
        shared_a_shrink,
        shared_b_down_expand,
        sorted_a_down_shrink,
        sorted_gate_up_b_expand,
    )

    _FUSED_MOE_LORA_AVAILABLE = True
except Exception:
    _FUSED_MOE_LORA_AVAILABLE = False

MoeLayerSlotWeights = dict[int, dict[str, torch.Tensor]]
MoeWeightsByLayer = dict[int, MoeLayerSlotWeights]


@dataclass(frozen=True)
class MoeLoraContext:
    """Narrow per-forward view of MoE LoRA state consumed by MoE backends."""

    weights_by_layer: MoeWeightsByLayer
    batch_info: LoraBatchInfo
    scalings: torch.Tensor
    has_active_lora: bool
    # Per-layer buffer lists for CUDA-graph-compatible dynamic slot indexing.
    # When set, _apply_*_slot uses GPU tensor indexing via batch_info.weight_indices
    # instead of Python dict lookup, so the CUDA graph can replay with any adapter.
    w13_A_buffers: list | None
    w13_B_buffers: list | None
    down_A_buffers: list | None
    down_B_buffers: list | None
    # Multi-stream prefetch: secondary stream + pre-allocated output buffers.
    # Shrink ops run on _lora_stream concurrently with the base MoE GEMMs.
    _lora_stream: object | None = None  # torch.cuda.Stream
    _lora_a_m_buf: torch.Tensor | None = None  # (max_bs, 2*max_r)
    _lora_a_flat_buf: torch.Tensor | None = None  # (max_bs*max_topk, max_r)
    # Mutable flags (list elements are mutable even in frozen dataclass):
    #   _prefetch_flags[0] = gate_up shrink pending; [1] = down shrink pending.
    _prefetch_flags: list | None = None

    def apply_gate_up_lora(
        self,
        layer_id: int,
        hidden_states: torch.Tensor,
        topk_ids: torch.Tensor,
        gate_up_output: torch.Tensor,
        *,
        sorted_token_ids: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Apply expert-scoped LoRA to routed MoE gate/up output."""
        if hidden_states.shape[0] == 0 or topk_ids.numel() == 0:
            return gate_up_output
        slots, single_slot = self._token_slots(hidden_states.shape[0])
        if single_slot == NO_LORA_SLOT and slots is None:
            return gate_up_output
        if single_slot != NO_LORA_SLOT:
            self._apply_gate_up_slot(
                layer_id,
                single_slot,
                hidden_states,
                topk_ids,
                gate_up_output,
                sorted_token_ids=sorted_token_ids,
            )
            return gate_up_output
        assert slots is not None
        for slot_t in torch.unique(slots):
            slot = int(slot_t.item())
            if slot == NO_LORA_SLOT:
                continue
            self._apply_gate_up_slot(
                layer_id,
                slot,
                hidden_states,
                topk_ids,
                gate_up_output,
                token_mask=slots == slot,
                sorted_token_ids=sorted_token_ids,
            )
        return gate_up_output

    def apply_down_lora(
        self,
        layer_id: int,
        intermediate: torch.Tensor,
        topk_ids: torch.Tensor,
        topk_weights: torch.Tensor,
        down_output: torch.Tensor,
        *,
        sorted_token_ids: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Apply expert-scoped LoRA to routed MoE down output."""
        if intermediate.shape[0] == 0 or topk_ids.numel() == 0:
            return down_output
        num_tokens = topk_ids.shape[0]
        slots, single_slot = self._token_slots(num_tokens)
        if single_slot == NO_LORA_SLOT and slots is None:
            return down_output
        # Sorted-space fast path: work directly on sorted intermediate, skipping
        # _route_rows_from_cache. Only applies when sorted dispatch is active (TMA
        # config), since the fused shrink kernel has poor utilization for small
        # flat-pair batches.
        if (
            _FUSED_MOE_LORA_AVAILABLE
            and sorted_token_ids is not None
            and single_slot != NO_LORA_SLOT
            and self.down_A_buffers is not None
            and self.batch_info.single_lora_slot != -1
        ):
            if self._apply_down_sorted(
                layer_id,
                single_slot,
                intermediate,
                topk_ids,
                topk_weights,
                down_output,
                sorted_token_ids,
            ):
                return down_output
        route_input = self._route_rows_from_cache(
            intermediate,
            topk_ids.numel(),
            sorted_token_ids=sorted_token_ids,
        ).view(topk_ids.shape[0], topk_ids.shape[1], -1)
        if single_slot != NO_LORA_SLOT:
            self._apply_down_slot(
                layer_id,
                single_slot,
                route_input,
                topk_ids,
                topk_weights,
                down_output,
            )
            return down_output
        assert slots is not None
        for slot_t in torch.unique(slots):
            slot = int(slot_t.item())
            if slot == NO_LORA_SLOT:
                continue
            self._apply_down_slot(
                layer_id,
                slot,
                route_input,
                topk_ids,
                topk_weights,
                down_output,
                token_mask=slots == slot,
            )
        return down_output

    def _token_slots(self, num_tokens: int) -> tuple[torch.Tensor | None, int]:
        bi = self.batch_info
        if bi.bs == 0 or not self.has_active_lora:
            return None, NO_LORA_SLOT
        if bi.single_lora_slot != NO_LORA_SLOT:
            return None, bi.single_lora_slot
        slots = torch.repeat_interleave(
            bi.weight_indices[: bi.bs], bi.seg_lens[: bi.bs]
        )
        if slots.numel() != num_tokens:
            # Token ownership changed under TP/EP communication. Mixed LoRA
            # cannot be applied safely without transforming the slot map too.
            return None, NO_LORA_SLOT
        return slots, NO_LORA_SLOT

    # ── Multi-stream prefetch API ──────────────────────────────────────────────
    # Called from triton_common.py BEFORE each base GEMM to overlap the LoRA
    # shrink kernel with the base model's gate_up / down computation.

    def launch_gate_up_shrink(
        self,
        layer_id: int,
        hidden_states: torch.Tensor,
        topk_ids: torch.Tensor,
    ) -> None:
        """Fork: launch gate_up LoRA A-shrink on secondary stream.

        Must be called immediately BEFORE gate_up_gemm so that shared_a_shrink
        (torch.mm) runs concurrently on _lora_stream while gate_up_gemm runs
        on the main stream.  apply_gate_up_lora will join the stream and use
        the pre-filled _lora_a_m_buf instead of recomputing.
        """
        if self._prefetch_flags is None:
            return
        self._prefetch_flags[0] = False  # default: no prefetch
        bi = self.batch_info
        if (
            not self.has_active_lora
            or bi.single_lora_slot == NO_LORA_SLOT
            or self.w13_A_buffers is None
            or self._lora_stream is None
            or self._lora_a_m_buf is None
        ):
            return
        m = hidden_states.shape[0]
        w13_A_buf = self.w13_A_buffers[layer_id]
        if w13_A_buf.shape[1] != 1:  # only sglang_shared format (shared A)
            return
        if m > self._lora_a_m_buf.shape[0]:
            return  # prefill with too many tokens — skip prefetch to avoid OOB
        # Fork to secondary stream: launch torch.mm concurrently with gate_up_gemm.
        self._lora_stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(self._lora_stream):
            torch.mm(hidden_states, w13_A_buf[0, 0].T, out=self._lora_a_m_buf[:m])
        self._prefetch_flags[0] = True

    def launch_down_shrink(
        self,
        layer_id: int,
        intermediate: torch.Tensor,
        topk_ids: torch.Tensor,
        m_k: int,
    ) -> None:
        """Fork: launch down LoRA A-shrink on secondary stream.

        Must be called immediately BEFORE down_gemm so that per_expert_a_shrink
        runs concurrently on _lora_stream while down_gemm runs on main stream.
        intermediate is intermediate_cache2 (silu output), shape (m*topk, INTER).
        m_k is m_tokens * top_k (non-padded).
        """
        if self._prefetch_flags is None:
            return
        self._prefetch_flags[1] = False
        bi = self.batch_info
        down_A_buf = self.down_A_buffers[layer_id] if self.down_A_buffers else None
        down_B_buf = self.down_B_buffers[layer_id] if self.down_B_buffers else None
        if (
            not self.has_active_lora
            or bi.single_lora_slot == NO_LORA_SLOT
            or down_A_buf is None
            or down_B_buf is None
            or self._lora_stream is None
            or self._lora_a_flat_buf is None
            or not _FUSED_MOE_LORA_AVAILABLE
            or down_A_buf.shape[1] <= 1  # per-expert A only
            or down_B_buf.shape[1] != 1  # shared B only
            or not down_A_buf.is_contiguous()
        ):
            return
        if m_k > self._lora_a_flat_buf.shape[0]:
            return  # prefill with too many tokens — skip prefetch to avoid OOB
        ri_flat = intermediate[:m_k].view(m_k, -1)
        safe_ids = topk_ids.clamp(0, down_A_buf.shape[1] - 1).to(torch.long)
        slot_idx = bi.weight_indices[:1].clamp(0)
        self._lora_stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(self._lora_stream):
            per_expert_a_shrink(
                ri_flat,
                down_A_buf,
                slot_idx,
                safe_ids,
                out=self._lora_a_flat_buf[:m_k],
            )
        self._prefetch_flags[1] = True

    def _apply_gate_up_slot(
        self,
        layer_id: int,
        slot: int,
        hidden_states: torch.Tensor,
        topk_ids: torch.Tensor,
        gate_up_output: torch.Tensor,
        *,
        token_mask: torch.Tensor | None = None,
        sorted_token_ids: torch.Tensor | None = None,
    ) -> None:
        # For the single-slot case (all tokens same adapter), use dynamic GPU tensor
        # indexing so the CUDA graph can replay with any loaded adapter.
        # For multi-slot batches, fall back to Python dict lookup (eager only).
        bi = self.batch_info
        # Determine if we're on the CUDA-graph buffer path (single slot, all tokens
        # same adapter). In this path we keep slot_idx as a GPU tensor so the CUDA
        # graph can replay with any loaded adapter without re-capture.
        _use_buffer_path = self.w13_A_buffers is not None and bi.single_lora_slot != -1
        slot_idx = None
        if _use_buffer_path:
            slot_idx = bi.weight_indices[:1].clamp(0)
            w13_B_buf = self.w13_B_buffers[layer_id]  # (n_slots, E, I2, MAX_R)
            w13_B = None
            # For the sglang_shared fast path (shared A, per-expert B) with fused kernels
            # available, skip the w13_A gather entirely — shared_a_shrink reads directly from
            # the buffer.  For all other paths, gather as before.
            _w13_A_buf = self.w13_A_buffers[layer_id]
            _skip_a_gather = (
                _FUSED_MOE_LORA_AVAILABLE
                and _w13_A_buf.shape[1] == 1  # shared outer (sglang_shared)
                and w13_B_buf.shape[1] > 1  # per-expert B
            )
            # Also skip the buffer copy for per_expert format (both A and B per-expert):
            # per_expert_a_shrink + per_expert_gate_up_b_expand read the full buffer
            # directly, making the 32MB w13_A buffer copy unnecessary.
            _skip_a_gather_per_expert = (
                _FUSED_MOE_LORA_AVAILABLE
                and _w13_A_buf.shape[1] > 1  # per-expert A
                and w13_B_buf.shape[1] > 1  # per-expert B
                and token_mask is None
                and _w13_A_buf.is_contiguous()
                and w13_B_buf.is_contiguous()
            )
            if _skip_a_gather or _skip_a_gather_per_expert:
                # Use slot-0 view (Python int index = no copy) — correct shape for checks.
                # Actual compute reads from the full buffers directly.
                w13_A = _w13_A_buf[0]  # view: (1_or_E, R, H) — no copy!
            else:
                w13_A = _w13_A_buf[slot_idx].squeeze(0)
        else:
            weights = self.weights_by_layer.get(layer_id, {}).get(slot)
            if weights is None:
                return
            w13_A = weights["w13_A"]
            w13_B = weights["w13_B"]
            w13_B_buf = None

        # Determine shapes without materialising w13_B when on the buffer path.
        if _use_buffer_path:
            w13_A_experts = w13_A.shape[0]
            w13_B_experts = w13_B_buf.shape[1]  # E dimension of buffer
        else:
            w13_A_experts = w13_A.shape[0]
            w13_B_experts = w13_B.shape[0]
        num_experts = max(w13_A_experts, w13_B_experts)
        safe_ids = topk_ids.clamp(0, num_experts - 1).to(torch.long)
        m, k = safe_ids.shape
        # Build the validity mask without torch.any() to avoid GPU→CPU synchronisation.
        if token_mask is not None:
            valid = (topk_ids >= 0) & (topk_ids < num_experts) & token_mask[:, None]
        else:
            valid = None

        # Check if per_expert fast path is available (avoids the 32MB+16MB gather copies).
        # Must be determined before the A-shrink so we can skip the expensive gather+einsum.
        _use_flat_per_expert = (
            w13_A.shape[0] > 1  # per-expert A
            and w13_B_buf is not None
            and w13_B_experts > 1  # per-expert B
            and _FUSED_MOE_LORA_AVAILABLE
            and _use_buffer_path
            and token_mask is None
            and self.w13_A_buffers[layer_id].is_contiguous()
            and w13_B_buf.is_contiguous()
        )

        # Shared A (sglang_shared format): one matmul for all tokens.
        # lora_a_m (m, r) is only computed here; lora_a (m, k, r) is deferred until
        # actually needed (not needed for the all-experts or shared-B paths).
        lora_a_m = None
        if w13_A.shape[0] == 1:
            # Skip cuBLAS GEMM when shared_a_shrink will compute it without the gather.
            if _use_buffer_path and _skip_a_gather:
                lora_a_m = None  # computed by shared_a_shrink in the fused branch below
            else:
                lora_a_m = hidden_states @ w13_A[0].T
            lora_a = None  # computed lazily below only if per-expert B path is taken
        elif _use_flat_per_expert:
            lora_a = (
                None  # computed inline by per_expert_a_shrink + per_expert_*_b_expand
            )
        else:
            selected_A = self._select_expert_weights(w13_A, safe_ids)
            lora_a = torch.einsum("mh,mkrh->mkr", hidden_states, selected_A)

        # Compute lora_a only when needed (per-expert B path).
        # For shared-A + all-experts or shared-A + shared-B, lora_a_m is used directly.
        # Lazily materialise w13_B for non-fused fallback paths on the buffer path.
        def _get_w13_B():
            nonlocal w13_B
            if w13_B is None:
                w13_B = w13_B_buf[slot_idx].squeeze(0)
            return w13_B

        if w13_B_experts == 1:
            # Shared B: expand lora_a_m to (m*k, r) via repeat_interleave (no contiguous copy).
            w13_B_local = _get_w13_B()
            r = lora_a_m.shape[-1] if lora_a_m is not None else lora_a.shape[-1]
            la_flat = (
                lora_a_m.repeat_interleave(k, dim=0)
                if lora_a_m is not None
                else lora_a.reshape(-1, r)
            )
            delta = la_flat @ w13_B_local[0].T  # (m*k, n)
            delta = delta.view(m, k, -1)
        elif w13_A.shape[0] == 1:
            # Shared-A + per-expert B.
            if _FUSED_MOE_LORA_AVAILABLE and token_mask is None:
                if sorted_token_ids is not None:
                    # TMA sorted path: write to sorted output positions (SCATTER=False).
                    _scaling = (
                        self.scalings[slot_idx]
                        if _use_buffer_path
                        else self.scalings[slot]
                    )
                    w13_B_local = _get_w13_B()
                    assert w13_B_local.is_contiguous(), "w13_B must be contiguous"
                    sorted_gate_up_b_expand(
                        lora_a_m,
                        w13_B_local,
                        safe_ids,
                        sorted_token_ids,
                        gate_up_output,
                        _scaling,
                        m * k,
                        k,
                    )
                elif _use_buffer_path:
                    # Decode path (buffer path): use pre-fetched lora_a_m if available
                    # (launched on secondary stream before gate_up_gemm), else compute inline.
                    _gu_prefetched = (
                        self._prefetch_flags is not None
                        and self._prefetch_flags[0]
                        and self._lora_a_m_buf is not None
                        and self._lora_stream is not None
                    )
                    if _gu_prefetched:
                        # Join secondary stream: wait for torch.mm to complete.
                        torch.cuda.current_stream().wait_stream(self._lora_stream)
                        lora_a_m = self._lora_a_m_buf[: hidden_states.shape[0]]
                        self._prefetch_flags[0] = False
                    else:
                        lora_a_m = shared_a_shrink(
                            hidden_states, self.w13_A_buffers[layer_id], slot_idx
                        )
                    gate_up_b_expand(
                        lora_a_m,
                        w13_B_buf,
                        slot_idx,
                        safe_ids,
                        gate_up_output,
                        self.scalings,  # full buffer; kernel loads scalings[slot]
                    )
                else:
                    # Non-buffer decode path (multi-slot eager).
                    w13_B_local = _get_w13_B()
                    assert w13_B_local.is_contiguous(), "w13_B must be contiguous"
                    gate_up_b_expand(
                        lora_a_m,
                        w13_B_local.unsqueeze(0),
                        torch.zeros(1, dtype=torch.int32, device=w13_B_local.device),
                        safe_ids,
                        gate_up_output,
                        self.scalings[slot].unsqueeze(0),  # (1,) for slot 0 of fake buf
                    )
                return
            # Fallback: all-experts GEMM + gather (no expand+copy needed).
            w13_B_local = _get_w13_B()
            E_fb, n_out, r = w13_B_local.shape
            candidates = (
                lora_a_m @ w13_B_local.permute(2, 0, 1).reshape(r, E_fb * n_out)
            ).view(m, E_fb, n_out)
            delta = candidates.gather(1, safe_ids.unsqueeze(-1).expand(-1, -1, n_out))
        else:
            # Per-expert A + per-expert B.
            if _use_flat_per_expert:
                # Fast flat path: avoid two buffer gather copies (w13_A_buf[slot] = 32MB,
                # w13_B_buf[slot] = 16MB) by reading directly from the full buffers.
                # per_expert_a_shrink reused: treats w13_A (n_slots, E, 2r, H) as
                # down_A (n_slots, E, MAX_R, INTER) with MAX_R=2r, INTER=H.
                hidden_flat = hidden_states.repeat_interleave(k, dim=0)  # (m*k, H)
                lora_a_flat = per_expert_a_shrink(
                    hidden_flat,
                    self.w13_A_buffers[layer_id],
                    slot_idx,
                    safe_ids,
                )  # (m*k, 2r)
                per_expert_gate_up_b_expand(
                    lora_a_flat,
                    w13_B_buf,
                    slot_idx,
                    safe_ids,
                    gate_up_output,
                    self.scalings,
                )
                return
            # Fallback: gather + einsum for non-buffer or masked paths.
            w13_B_local = _get_w13_B()
            if lora_a is None:
                lora_a = lora_a_m.unsqueeze(1).expand(-1, k, -1).contiguous()
            selected_B = self._select_expert_weights(w13_B_local, safe_ids)
            delta = torch.einsum("mkr,mknr->mkn", lora_a, selected_B)

        # Reuse slot_idx already computed above (avoid extra clamp+gather for scalings).
        scaling = self.scalings[slot_idx] if _use_buffer_path else self.scalings[slot]
        delta = delta * scaling
        if valid is not None:
            delta = delta.masked_fill(~valid[:, :, None], 0.0)
        self._add_route_delta(
            gate_up_output,
            delta.reshape(-1, delta.shape[-1]),
            sorted_token_ids=sorted_token_ids,
        )

    def _apply_down_sorted(
        self,
        layer_id: int,
        slot: int,
        intermediate: torch.Tensor,
        topk_ids: torch.Tensor,
        topk_weights: torch.Tensor,
        down_output: torch.Tensor,
        sorted_token_ids: torch.Tensor,
    ) -> bool:
        """Sorted-space down LoRA: skip route_from_cache, fuse per-expert shrink.

        Returns True if the fast path was taken (per-expert A + shared B format),
        False if the format requires the generic path.
        """
        bi = self.batch_info
        slot_idx = bi.weight_indices[:1].clamp(0)
        down_A = self.down_A_buffers[layer_id][slot_idx].squeeze(0)
        down_B = self.down_B_buffers[layer_id][slot_idx].squeeze(0)
        # Only handles per-expert A + shared B (sglang_shared format for down).
        if down_A.shape[0] <= 1 or down_B.shape[0] != 1:
            return False
        if not down_A.is_contiguous():
            return False

        m, k = topk_ids.shape
        num_experts = down_A.shape[0]
        safe_ids = topk_ids.clamp(0, num_experts - 1).to(torch.long)
        route_count = m * k
        r = down_A.shape[1]

        # moe_dispatch pre-allocates sorted_token_ids for all potential experts, which
        # can exceed the intermediate cache size.  All valid entries (≥0) lie within
        # the first intermediate.shape[0] rows (bound = m*k + max_active*(BM-1)).
        inter_flat = intermediate.reshape(intermediate.shape[0], -1)
        padded = inter_flat.shape[0]
        sti = sorted_token_ids[:padded]  # truncate to intermediate size

        # Fused per-expert shrink: lora_a[s] = intermediate[s] @ down_A[exp[s]].T
        lora_a_sorted = sorted_a_down_shrink(
            inter_flat,  # (padded, INTER)
            down_A,  # (E, r, INTER)
            safe_ids,
            sti,
            route_count=route_count,
            K=k,
        )

        # Shared B GEMM: (padded, r) @ (r, h) → (padded, h)
        delta = lora_a_sorted @ down_B[0].T

        # Scale each sorted position by its topk_weight * adapter scaling.
        valid = (sti >= 0) & (sti < route_count)
        # Clamp to [0, route_count-1]: sorted_token_ids may contain route_count as
        # a sentinel value, which would be OOB without the upper bound.
        flat_j_safe = sti.clamp(0, route_count - 1)
        weights_sorted = topk_weights.reshape(-1)[flat_j_safe].to(delta.dtype)
        scaling_t = self.scalings[slot_idx].to(delta.dtype)
        delta = delta * (weights_sorted * scaling_t * valid.to(delta.dtype)).unsqueeze(
            -1
        )

        # Scatter-add to token-ordered down_output.
        h = delta.shape[-1]
        down_output.view(route_count, h).scatter_add_(
            0, flat_j_safe.unsqueeze(-1).expand(-1, h), delta
        )
        return True

    def _apply_down_slot(
        self,
        layer_id: int,
        slot: int,
        route_input: torch.Tensor,
        topk_ids: torch.Tensor,
        topk_weights: torch.Tensor,
        down_output: torch.Tensor,
        *,
        token_mask: torch.Tensor | None = None,
    ) -> None:
        bi = self.batch_info
        # Determine if we're on the CUDA-graph buffer path (single slot, all tokens
        # same adapter). In this path we keep slot_idx as a GPU tensor so the CUDA
        # graph can replay with any loaded adapter without re-capture.
        _use_buffer_path = self.down_A_buffers is not None and bi.single_lora_slot != -1
        slot_idx = None
        if _use_buffer_path:
            # (1,) GPU tensor — changes at CUDA-graph replay without re-capture.
            slot_idx = bi.weight_indices[:1].clamp(0)
            # Keep references to the full buffers; slicing is done lazily or inside kernels.
            down_A_buf = self.down_A_buffers[layer_id]  # (n_slots, E, MAX_R, INTER)
            down_B_buf = self.down_B_buffers[layer_id]  # (n_slots, 1_or_E, H, MAX_R)
            # Sliced views are populated lazily to avoid redundant gathers.
            down_A = None
            down_B = None
        else:
            weights = self.weights_by_layer.get(layer_id, {}).get(slot)
            if weights is None:
                return
            down_A = weights["down_A"]
            down_B = weights["down_B"]
            down_A_buf = None
            down_B_buf = None

        # Determine shapes without materialising tensors when on the buffer path.
        if _use_buffer_path:
            down_A_experts = down_A_buf.shape[1]  # E dimension of buffer
            down_B_experts = down_B_buf.shape[1]  # 1 for shared-B
        else:
            down_A_experts = down_A.shape[0]
            down_B_experts = down_B.shape[0]
        num_experts = max(down_A_experts, down_B_experts)
        safe_ids = topk_ids.clamp(0, num_experts - 1).to(torch.long)
        m, k = safe_ids.shape
        if token_mask is not None:
            valid = (topk_ids >= 0) & (topk_ids < num_experts) & token_mask[:, None]
        else:
            valid = None

        # Helpers to lazily materialise sliced tensors for fallback paths.
        def _get_down_A():
            nonlocal down_A
            if down_A is None:
                down_A = down_A_buf[slot_idx].squeeze(0)
            return down_A

        def _get_down_B():
            nonlocal down_B
            if down_B is None:
                down_B = down_B_buf[slot_idx].squeeze(0)
            return down_B

        # Fast fused path: per-expert A + shared B on the CUDA-graph buffer path.
        # Eliminates both gather copies (down_A gather + down_B gather) and the
        # separate GEMM + scale + add chain.
        if (
            _FUSED_MOE_LORA_AVAILABLE
            and _use_buffer_path
            and token_mask is None
            and down_A_experts > 1
            and down_B_experts == 1
            and down_A_buf.is_contiguous()
            and down_B_buf.is_contiguous()
        ):
            _down_prefetched = (
                self._prefetch_flags is not None
                and self._prefetch_flags[1]
                and self._lora_a_flat_buf is not None
                and self._lora_stream is not None
            )
            if _down_prefetched:
                # Join secondary stream: wait for per_expert_a_shrink to complete.
                torch.cuda.current_stream().wait_stream(self._lora_stream)
                lora_a_flat = self._lora_a_flat_buf[: m * k]
                self._prefetch_flags[1] = False
            else:
                ri_flat = route_input.reshape(m * k, -1)  # (m*k, INTER)
                lora_a_flat = per_expert_a_shrink(
                    ri_flat, down_A_buf, slot_idx, safe_ids
                )
            shared_b_down_expand(
                lora_a_flat,
                down_B_buf,
                slot_idx,
                down_output.view(m, k, -1),
                topk_weights,
                self.scalings,  # full buffer; kernel loads scalings[slot]
                k,
            )
            return

        # Shared A (sglang_shared down_proj): one matmul per token-topk group.
        if down_A_experts == 1:
            down_A_local = _get_down_A()
            ri = route_input.reshape(m * k, -1)  # (m*k, i)
            lora_a = (ri @ down_A_local[0].T).view(m, k, -1)  # (m, k, r)
        elif _FUSED_MOE_LORA_AVAILABLE and token_mask is None:
            # Flat per-expert shrink: avoids the (m*k, r, INTER) gather intermediate
            # and replaces the batched einsum with a single fused Triton kernel.
            if _use_buffer_path:
                # Buffer path: pass full buffer + slot_idx to avoid gather.
                lora_a = per_expert_a_shrink(
                    route_input.reshape(m * k, -1), down_A_buf, slot_idx, safe_ids
                ).view(m, k, -1)
            else:
                down_A_local = _get_down_A()
                assert down_A_local.is_contiguous(), "down_A must be contiguous"
                lora_a = per_expert_a_shrink(
                    route_input.reshape(m * k, -1),
                    down_A_local.unsqueeze(0),  # fake (1, E, MAX_R, INTER) buffer
                    torch.zeros(1, dtype=torch.int32, device=down_A_local.device),
                    safe_ids,
                ).view(m, k, -1)
        else:
            down_A_local = _get_down_A()
            selected_A = self._select_expert_weights(down_A_local, safe_ids)
            lora_a = torch.einsum("mki,mkri->mkr", route_input, selected_A)

        # Shared B (sglang_shared down_proj): one batched matmul.
        if down_B_experts == 1:
            down_B_local = _get_down_B()
            r = lora_a.shape[-1]
            delta = lora_a.reshape(-1, r) @ down_B_local[0].T  # (m*k, h)
            delta = delta.view(m, k, -1)
        elif (
            _FUSED_MOE_LORA_AVAILABLE
            and _use_buffer_path
            and token_mask is None
            and down_B_buf.is_contiguous()
        ):
            # Per-expert B fast path: avoid the 16MB buffer copy + gather.
            # lora_a computed via per_expert_a_shrink is already (m*k, r); reshape to flat.
            lora_a_flat = lora_a.reshape(m * k, -1)
            per_expert_b_down_expand(
                lora_a_flat,
                down_B_buf,
                slot_idx,
                safe_ids,
                down_output.view(m, k, -1),
                topk_weights,
                self.scalings,
                k,
            )
            return  # accumulation already done inside the kernel
        else:
            down_B_local = _get_down_B()
            selected_B = self._select_expert_weights(down_B_local, safe_ids)
            delta = torch.einsum("mkr,mkhr->mkh", lora_a, selected_B)

        delta = delta * topk_weights[:, :, None].to(delta.dtype)
        # Reuse slot_idx computed above for scalings (avoid extra clamp+gather).
        scaling = self.scalings[slot_idx] if _use_buffer_path else self.scalings[slot]
        delta = delta * scaling
        if valid is not None:
            delta = delta.masked_fill(~valid[:, :, None], 0.0)
        down_output.view(topk_ids.shape[0], topk_ids.shape[1], -1).add_(delta)

    @staticmethod
    def _select_expert_weights(
        weights: torch.Tensor,
        safe_ids: torch.Tensor,
    ) -> torch.Tensor:
        if weights.shape[0] == 1:
            return weights[0].expand(*safe_ids.shape, *weights.shape[1:])
        return weights[safe_ids]

    @staticmethod
    def _add_route_delta(
        output: torch.Tensor,
        route_delta: torch.Tensor,
        *,
        sorted_token_ids: torch.Tensor | None,
    ) -> None:
        if sorted_token_ids is None:
            output.view(route_delta.shape[0], -1).add_(route_delta)
            return
        # moe_dispatch may pre-allocate sorted_token_ids larger than output.
        # Truncate: all valid entries lie within the first output.shape[0] rows.
        padded = output.shape[0]
        sti = sorted_token_ids[:padded]
        # Gather route_delta into output-layout, zero invalid (padding) entries,
        # then add in one vectorised kernel — avoids boolean-index tensor creation.
        route_count = route_delta.shape[0]
        clipped = sti.clamp(0, route_count - 1).to(torch.long)
        reordered = route_delta[clipped]  # (padded, n)
        invalid = (sti < 0) | (sti >= route_count)
        reordered.masked_fill_(invalid.unsqueeze(-1), 0)
        output.add_(reordered)

    @staticmethod
    def _route_rows_from_cache(
        cache: torch.Tensor,
        route_count: int,
        *,
        sorted_token_ids: torch.Tensor | None,
    ) -> torch.Tensor:
        if sorted_token_ids is None:
            return cache.view(route_count, -1)
        # moe_dispatch may pre-allocate sorted_token_ids larger than cache.
        # Truncate: all valid entries lie within the first cache.shape[0] rows.
        sti = sorted_token_ids[: cache.shape[0]]
        # Use scatter_ with an extra dummy row (index 0) for padding positions.
        # Avoids boolean-index tensor creation; one scatter_ + one slice.
        n = cache.shape[-1]
        rows = torch.zeros((route_count + 1, n), dtype=cache.dtype, device=cache.device)
        # Shift: -1 (padding) → 0 (dummy), valid 0..route_count-1 → 1..route_count.
        clipped = (sti.clamp(-1, route_count - 1) + 1).to(torch.long)
        rows.scatter_(0, clipped.unsqueeze(-1).expand(-1, n), cache)
        return rows[1:]  # drop dummy row → (route_count, n)


class MoeLoraBuffers:
    """Own expert-scoped MoE LoRA weights independently from dense buffers."""

    def __init__(
        self,
        *,
        n_layers: int,
        n_slots: int,
        max_lora_rank: int,
        num_experts: int,
        hidden_size: int,
        intermediate_per_tp: int,
        dtype: torch.dtype,
        device: torch.device,
        shard_weights: Callable[
            [str, torch.Tensor, torch.Tensor], tuple[torch.Tensor, torch.Tensor]
        ],
        enabled: bool = True,
        compressed_shared_outer: bool = False,
    ) -> None:
        self.n_layers = n_layers
        self.n_slots = n_slots
        self.max_lora_rank = max_lora_rank
        self.num_experts = num_experts
        self.hidden_size = hidden_size
        self.intermediate_per_tp = intermediate_per_tp
        self.dtype = dtype
        self.device = device
        self._shard_weights = shard_weights
        self.enabled = enabled
        self.compressed_shared_outer = compressed_shared_outer
        self.weights_by_layer: MoeWeightsByLayer = {}
        self.w13_A_buffers: list[torch.Tensor] = []
        self.w13_B_buffers: list[torch.Tensor] = []
        self.down_A_buffers: list[torch.Tensor] = []
        self.down_B_buffers: list[torch.Tensor] = []
        self._alloc()
        # Multi-stream prefetch: overlap LoRA shrink ops with base MoE GEMMs.
        # Shrink kernels run on a secondary stream in parallel with gate_up/down GEMMs.
        # Pre-allocated output buffers avoid torch.empty inside CUDA graphs.
        _max_bs = 128
        _max_topk = 8
        self._lora_stream: torch.cuda.Stream | None = (
            torch.cuda.Stream() if torch.cuda.is_available() else None
        )
        self._lora_a_m_buf: torch.Tensor | None = None
        self._lora_a_flat_buf: torch.Tensor | None = None
        if self.enabled and torch.cuda.is_available():
            # gate/up shrink: (m, 2*r); down shrink: (m*topk, r)
            self._lora_a_m_buf = torch.zeros(
                _max_bs, 2 * max_lora_rank, dtype=dtype, device=device
            )
            self._lora_a_flat_buf = torch.zeros(
                _max_bs * _max_topk, max_lora_rank, dtype=dtype, device=device
            )
            # Pre-warm cuBLAS and Triton kernels on _lora_stream before any CUDA graph
            # capture. torch.mm (cuBLAS) requires its handle to be initialized on each
            # stream; failing to do so causes CUBLAS_STATUS_NOT_INITIALIZED during capture.
            if self._lora_stream is not None:
                _d = torch.zeros(1, dtype=dtype, device=device)
                with torch.cuda.stream(self._lora_stream):
                    torch.mm(_d.unsqueeze(0), _d.unsqueeze(1))
                del _d
                torch.cuda.synchronize()
        # Mutable flags shared between MoeLoraBuffers and MoeLoraContext instances:
        # [0] = gate_up shrink launched; [1] = down shrink launched.
        self._prefetch_flags: list[bool] = [False, False]

    def _alloc(self) -> None:
        if not self.enabled:
            return
        n = self.n_slots
        e = max(self.num_experts, 0)
        r = self.max_lora_rank
        h = self.hidden_size
        i = self.intermediate_per_tp
        w13_a_experts = 1 if self.compressed_shared_outer else e
        w13_b_experts = e
        down_a_experts = e
        down_b_experts = 1 if self.compressed_shared_outer else e
        for _ in range(self.n_layers):
            self.w13_A_buffers.append(
                torch.zeros(
                    (n, w13_a_experts, 2 * r, h),
                    dtype=self.dtype,
                    device=self.device,
                )
            )
            self.w13_B_buffers.append(
                torch.zeros(
                    (n, w13_b_experts, 2 * i, 2 * r),
                    dtype=self.dtype,
                    device=self.device,
                )
            )
            self.down_A_buffers.append(
                torch.zeros(
                    (n, down_a_experts, r, i), dtype=self.dtype, device=self.device
                )
            )
            self.down_B_buffers.append(
                torch.zeros(
                    (n, down_b_experts, h, r), dtype=self.dtype, device=self.device
                )
            )

    def load_adapter_to_slot(self, cpu_weights, slot: int, rank: int) -> None:
        has_moe = any(
            mod.startswith("experts.")
            for modules in cpu_weights.values()
            for mod in modules
        )
        if has_moe and not self.enabled:
            raise ValueError(
                "Adapter contains MoE LoRA weights, but LoRA buffer group 'moe' "
                "is disabled."
            )
        if self.num_experts <= 0:
            if has_moe:
                raise ValueError(
                    "MoE LoRA adapter requires model_config.num_experts or "
                    "model_config.num_local_experts."
                )
            return
        rank = min(rank, self.max_lora_rank)
        for layer_id, modules in cpu_weights.items():
            if not any(mod.startswith("experts.") for mod in modules):
                continue
            self._clear_layer_slot(layer_id, slot)
            if any(
                mod in modules for mod in ("experts.w1", "experts.w2", "experts.w3")
            ):
                self._load_3d_adapter_layer(layer_id, modules, slot, rank)
            else:
                self._load_2d_adapter_layer(layer_id, modules, slot, rank)

    def _load_2d_adapter_layer(self, layer_id: int, modules, slot: int, rank: int):
        expert_ids = [
            int(mod.split(".")[1]) for mod in modules if mod.startswith("experts.")
        ]
        if not expert_ids:
            return
        if self.compressed_shared_outer:
            raise ValueError(
                "Compressed MoE shared-outer storage only supports 3D "
                "experts.w1/w2/w3 adapters."
            )
        num_experts = max(expert_ids) + 1
        self._check_num_experts(layer_id, num_experts)
        w13_A, w13_B, down_A, down_B = self._slot_layer_tensors(layer_id, slot)
        r = rank
        for mod, (lora_A_full, lora_B_full) in modules.items():
            if not mod.startswith("experts."):
                continue
            _, expert_id_s, module = mod.split(".", 2)
            expert_id = int(expert_id_s)
            # Normalize A/B convention: standard PEFT stores A as (rank, in_features)
            # and B as (out_features, rank).  Some adapters use the transposed layout
            # (in_features, rank) and (rank, out_features).  Detect by comparing dims:
            # if the first dim is larger than the second, A is in (in, rank) format.
            if lora_A_full.dim() == 2 and lora_A_full.shape[0] > lora_A_full.shape[1]:
                lora_A_full = lora_A_full.T  # (in, rank) → (rank, in)
            if lora_B_full.dim() == 2 and lora_B_full.shape[0] < lora_B_full.shape[1]:
                lora_B_full = lora_B_full.T  # (rank, out) → (out, rank)
            lora_A_shard_cpu, lora_B_shard_cpu = self._shard_weights(
                module, lora_A_full, lora_B_full
            )
            actual_rank = min(lora_A_shard_cpu.shape[0], r)
            lora_A_shard = lora_A_shard_cpu[:actual_rank].to(
                device=self.device,
                dtype=self.dtype,
                non_blocking=True,
            )
            lora_B_shard = lora_B_shard_cpu[:, :actual_rank].to(
                device=self.device,
                dtype=self.dtype,
                non_blocking=True,
            )
            self._copy_projection(
                module,
                expert_id,
                actual_rank,
                lora_A_shard,
                lora_B_shard,
                w13_A,
                w13_B,
                down_A,
                down_B,
                rank=r,
            )
        self.weights_by_layer.setdefault(layer_id, {})[slot] = {
            "w13_A": w13_A,
            "w13_B": w13_B,
            "down_A": down_A,
            "down_B": down_B,
        }

    def _load_3d_adapter_layer(self, layer_id: int, modules, slot: int, rank: int):
        required = ("experts.w1", "experts.w2", "experts.w3")
        missing = [name for name in required if name not in modules]
        if missing:
            raise ValueError(
                f"3D MoE LoRA layer {layer_id} is missing modules: {missing}"
            )
        w1_A, w1_B = modules["experts.w1"]
        w2_A, w2_B = modules["experts.w2"]
        w3_A, w3_B = modules["experts.w3"]
        num_experts = self._infer_3d_num_experts((w1_A, w1_B, w2_A, w2_B, w3_A, w3_B))
        self._check_num_experts(layer_id, num_experts)
        if self.compressed_shared_outer:
            self._check_shared_outer_layer(layer_id, modules, num_experts)
        w13_A, w13_B, down_A, down_B = self._slot_layer_tensors(layer_id, slot)
        self._copy_3d_projection(
            "gate_proj", w1_A, w1_B, w13_A, w13_B, down_A, down_B, rank
        )
        self._copy_3d_projection(
            "up_proj", w3_A, w3_B, w13_A, w13_B, down_A, down_B, rank
        )
        self._copy_3d_projection(
            "down_proj", w2_A, w2_B, w13_A, w13_B, down_A, down_B, rank
        )
        E_b, I2, R = w13_B.shape
        w13_B_T = w13_B.permute(2, 0, 1).reshape(R, E_b * I2).contiguous()
        self.weights_by_layer.setdefault(layer_id, {})[slot] = {
            "w13_A": w13_A,
            "w13_B": w13_B,
            "w13_B_T": w13_B_T,
            "down_A": down_A,
            "down_B": down_B,
        }

    def _check_num_experts(self, layer_id: int, adapter_num_experts: int) -> None:
        if adapter_num_experts > self.num_experts:
            raise ValueError(
                f"MoE LoRA layer {layer_id} has {adapter_num_experts} experts, "
                f"but the model has {self.num_experts}."
            )

    def _slot_layer_tensors(self, layer_id: int, slot: int):
        return (
            self.w13_A_buffers[layer_id][slot],
            self.w13_B_buffers[layer_id][slot],
            self.down_A_buffers[layer_id][slot],
            self.down_B_buffers[layer_id][slot],
        )

    def _clear_layer_slot(self, layer_id: int, slot: int) -> None:
        self.w13_A_buffers[layer_id][slot].zero_()
        self.w13_B_buffers[layer_id][slot].zero_()
        self.down_A_buffers[layer_id][slot].zero_()
        self.down_B_buffers[layer_id][slot].zero_()

    @staticmethod
    def _check_shared_outer_layer(
        layer_id: int,
        modules,
        num_experts: int,
    ) -> None:
        expected = {
            "experts.w1": (1, num_experts),
            "experts.w2": (num_experts, 1),
            "experts.w3": (1, num_experts),
        }
        for module, (expected_a, expected_b) in expected.items():
            lora_A, lora_B = modules[module]
            if lora_A.shape[0] != expected_a or lora_B.shape[0] != expected_b:
                raise ValueError(
                    "Compressed MoE shared-outer storage expects "
                    f"{module} A/B dim0=({expected_a}, {expected_b}) for "
                    f"layer {layer_id}; got {tuple(lora_A.shape)}, "
                    f"{tuple(lora_B.shape)}."
                )

    @staticmethod
    def _infer_3d_num_experts(tensors: tuple[torch.Tensor, ...]) -> int:
        num_experts = 0
        for tensor in tensors:
            if tensor.dim() != 3:
                raise ValueError(
                    f"3D MoE LoRA tensors must be rank-3, got shape {tuple(tensor.shape)}"
                )
            if tensor.shape[0] != 1:
                num_experts = max(num_experts, int(tensor.shape[0]))
        if num_experts <= 0:
            raise ValueError("3D MoE LoRA layer has no per-expert tensor dimension")
        for tensor in tensors:
            if tensor.shape[0] not in (1, num_experts):
                raise ValueError(
                    "3D MoE LoRA dim0 must be either 1 (shared) or num_experts "
                    f"({num_experts}); got {tuple(tensor.shape)}"
                )
        return num_experts

    def _copy_3d_projection(
        self,
        module: str,
        lora_A_full: torch.Tensor,
        lora_B_full: torch.Tensor,
        w13_A: torch.Tensor,
        w13_B: torch.Tensor,
        down_A: torch.Tensor,
        down_B: torch.Tensor,
        rank: int,
    ) -> None:
        num_experts = max(
            w13_A.shape[0], w13_B.shape[0], down_A.shape[0], down_B.shape[0]
        )
        if self.compressed_shared_outer:
            self._copy_3d_projection_compressed(
                module,
                lora_A_full,
                lora_B_full,
                w13_A,
                w13_B,
                down_A,
                down_B,
                rank,
                num_experts,
            )
            return
        for expert_id in range(num_experts):
            lora_A = self._select_3d_expert_tensor(lora_A_full, expert_id)
            lora_B = self._select_3d_expert_tensor(lora_B_full, expert_id)
            lora_A_shard_cpu, lora_B_shard_cpu = self._shard_weights(
                module, lora_A, lora_B
            )
            actual_rank = min(lora_A_shard_cpu.shape[0], rank)
            lora_A_shard = lora_A_shard_cpu[:actual_rank].to(
                device=self.device,
                dtype=self.dtype,
                non_blocking=True,
            )
            lora_B_shard = lora_B_shard_cpu[:, :actual_rank].to(
                device=self.device,
                dtype=self.dtype,
                non_blocking=True,
            )
            self._copy_projection(
                module,
                expert_id,
                actual_rank,
                lora_A_shard,
                lora_B_shard,
                w13_A,
                w13_B,
                down_A,
                down_B,
                rank=rank,
                a_expert_id=self._dst_expert_id(module, "A", expert_id),
                b_expert_id=self._dst_expert_id(module, "B", expert_id),
            )

    def _copy_3d_projection_compressed(
        self,
        module: str,
        lora_A_full: torch.Tensor,
        lora_B_full: torch.Tensor,
        w13_A: torch.Tensor,
        w13_B: torch.Tensor,
        down_A: torch.Tensor,
        down_B: torch.Tensor,
        rank: int,
        num_experts: int,
    ) -> None:
        if module in ("gate_proj", "up_proj"):
            shared_A = self._select_3d_expert_tensor(lora_A_full, 0)
            first_B = self._select_3d_expert_tensor(lora_B_full, 0)
            lora_A_shard_cpu, _ = self._shard_weights(module, shared_A, first_B)
            actual_rank = min(lora_A_shard_cpu.shape[0], rank)
            lora_A_shard = lora_A_shard_cpu[:actual_rank].to(
                device=self.device,
                dtype=self.dtype,
                non_blocking=True,
            )
            if module == "gate_proj":
                w13_A[0, :actual_rank, :].copy_(lora_A_shard, non_blocking=True)
            else:
                w13_A[0, rank : rank + actual_rank, :].copy_(
                    lora_A_shard, non_blocking=True
                )
            for expert_id in range(num_experts):
                expert_B = self._select_3d_expert_tensor(lora_B_full, expert_id)
                _, lora_B_shard_cpu = self._shard_weights(module, shared_A, expert_B)
                b_rank = min(lora_B_shard_cpu.shape[1], rank)
                lora_B_shard = lora_B_shard_cpu[:, :b_rank].to(
                    device=self.device,
                    dtype=self.dtype,
                    non_blocking=True,
                )
                if module == "gate_proj":
                    w13_B[expert_id, : self.intermediate_per_tp, :b_rank].copy_(
                        lora_B_shard, non_blocking=True
                    )
                else:
                    w13_B[
                        expert_id,
                        self.intermediate_per_tp : 2 * self.intermediate_per_tp,
                        rank : rank + b_rank,
                    ].copy_(lora_B_shard, non_blocking=True)
            return

        if module == "down_proj":
            first_A = self._select_3d_expert_tensor(lora_A_full, 0)
            shared_B = self._select_3d_expert_tensor(lora_B_full, 0)
            _, lora_B_shard_cpu = self._shard_weights(module, first_A, shared_B)
            b_rank = min(lora_B_shard_cpu.shape[1], rank)
            lora_B_shard = lora_B_shard_cpu[:, :b_rank].to(
                device=self.device,
                dtype=self.dtype,
                non_blocking=True,
            )
            down_B[0, :, :b_rank].copy_(lora_B_shard, non_blocking=True)
            for expert_id in range(num_experts):
                expert_A = self._select_3d_expert_tensor(lora_A_full, expert_id)
                lora_A_shard_cpu, _ = self._shard_weights(module, expert_A, shared_B)
                actual_rank = min(lora_A_shard_cpu.shape[0], rank)
                lora_A_shard = lora_A_shard_cpu[:actual_rank].to(
                    device=self.device,
                    dtype=self.dtype,
                    non_blocking=True,
                )
                down_A[expert_id, :actual_rank, :].copy_(
                    lora_A_shard, non_blocking=True
                )
            return

        raise ValueError(f"Unsupported MoE LoRA projection: {module}")

    @staticmethod
    def _select_3d_expert_tensor(tensor: torch.Tensor, expert_id: int) -> torch.Tensor:
        return tensor[0 if tensor.shape[0] == 1 else expert_id]

    def _copy_projection(
        self,
        module: str,
        expert_id: int,
        actual_rank: int,
        lora_A_shard: torch.Tensor,
        lora_B_shard: torch.Tensor,
        w13_A: torch.Tensor,
        w13_B: torch.Tensor,
        down_A: torch.Tensor,
        down_B: torch.Tensor,
        *,
        rank: int,
        a_expert_id: int | None = None,
        b_expert_id: int | None = None,
    ) -> None:
        a_expert_id = expert_id if a_expert_id is None else a_expert_id
        b_expert_id = expert_id if b_expert_id is None else b_expert_id
        if module == "gate_proj":
            w13_A[a_expert_id, :actual_rank, :].copy_(lora_A_shard, non_blocking=True)
            w13_B[
                b_expert_id,
                : self.intermediate_per_tp,
                :actual_rank,
            ].copy_(lora_B_shard, non_blocking=True)
        elif module == "up_proj":
            w13_A[a_expert_id, rank : rank + actual_rank, :].copy_(
                lora_A_shard, non_blocking=True
            )
            w13_B[
                b_expert_id,
                self.intermediate_per_tp : 2 * self.intermediate_per_tp,
                rank : rank + actual_rank,
            ].copy_(lora_B_shard, non_blocking=True)
        elif module == "down_proj":
            down_A[a_expert_id, :actual_rank, :].copy_(lora_A_shard, non_blocking=True)
            down_B[b_expert_id, :, :actual_rank].copy_(lora_B_shard, non_blocking=True)
        else:
            raise ValueError(f"Unsupported MoE LoRA projection: {module}")

    def _dst_expert_id(self, module: str, side: str, expert_id: int) -> int:
        if not self.compressed_shared_outer:
            return expert_id
        if module in ("gate_proj", "up_proj") and side == "A":
            return 0
        if module == "down_proj" and side == "B":
            return 0
        return expert_id

    def clear_slot(self, slot: int) -> None:
        if not self.enabled:
            return
        for layer_id in range(self.n_layers):
            self._clear_layer_slot(layer_id, slot)
        for layer_slots in self.weights_by_layer.values():
            layer_slots.pop(slot, None)

    def build_context(
        self,
        *,
        batch_info: LoraBatchInfo,
        scalings: torch.Tensor,
        has_active_lora: bool,
    ) -> "MoeLoraContext":
        return MoeLoraContext(
            weights_by_layer=self.weights_by_layer,
            batch_info=batch_info,
            scalings=scalings,
            has_active_lora=has_active_lora,
            w13_A_buffers=self.w13_A_buffers if self.enabled else None,
            w13_B_buffers=self.w13_B_buffers if self.enabled else None,
            down_A_buffers=self.down_A_buffers if self.enabled else None,
            down_B_buffers=self.down_B_buffers if self.enabled else None,
            _lora_stream=self._lora_stream,
            _lora_a_m_buf=self._lora_a_m_buf,
            _lora_a_flat_buf=self._lora_a_flat_buf,
            _prefetch_flags=self._prefetch_flags,
        )
