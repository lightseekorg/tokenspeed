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

"""Hy4-flash DFly2 block-parallel drafter sharing target DSA KV and index picks.

Lifecycle per speculative round:
1. Extract each request's anchor token ``x_a`` (bonus token just sampled by the
   target), the target's post-norm final hidden ``h_{a-1}`` at the last accepted
   position, its 1D/3D M-RoPE position, and its committed history length ``a``.
2. For each of the 5 DFly2 layers ``i``, gather context KV slots strictly before
   ``a`` from the target's shared ``HybridHy4FlashTokenToKVPool``:
   - If ``a <= target_index_topk`` (2048) or no sparse selection was recorded
     for ``target_index_layer[i]``, materialize all ``a`` slots from the DSA
     page table.
   - Otherwise, reuse the top-2048 global KV slots recorded by the target's
     ``Hy4FlashCacheGroupRouter`` at ``target_index_layer[i]`` for the last
     accepted token row.
3. Run the 5 DFly2 layers (``mtp_fusion(slot0)`` + dual-source Gated MLA +
   Dense SwiGLU MLP + ``final_norm``) in one block forward.
4. Walk ``hidden_correction`` sequentially across draft slots ``1..block_size-1``
   using the shared target FP32 ``lm_head``, supporting both greedy argmax and
   ``--enable-speculative-sampling``.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from typing_extensions import override

from tokenspeed.runtime.configs.hy4_flash_config import HYV4FlashDFlyConfig
from tokenspeed.runtime.distributed.comm_ops import all_gather
from tokenspeed.runtime.execution.drafter.base import BaseDrafter
from tokenspeed.runtime.execution.forward_batch_info import ForwardMode
from tokenspeed.runtime.layers.attention.backends.hy4_flash import (
    Hy4FlashCacheGroupRouter,
)
from tokenspeed.runtime.layers.attention.kv_cache.hybrid_hy4_flash import (
    HybridHy4FlashTokenToKVPool,
)
from tokenspeed.runtime.layers.logits_processor import LogitsMetadata
from tokenspeed.runtime.layers.vocab_parallel_embedding import VocabParallelEmbedding
from tokenspeed.runtime.models.hy4_flash_dfly import HYV4FlashForCausalLMDFly
from tokenspeed.runtime.utils import get_colorful_logger
from tokenspeed.runtime.utils.nvtx import nvtx_range
from tokenspeed.runtime.utils.spec_block_geometry import (
    read_checkpoint_block_size,
    validate_block_widths,
)
from tokenspeed.runtime.utils.torch_memory_saver_adapter import (
    TorchMemorySaverAdapter,
)

if TYPE_CHECKING:
    from tokenspeed.runtime.execution.context import ForwardContext
    from tokenspeed.runtime.execution.input_buffer import InputBuffers
    from tokenspeed.runtime.execution.model_runner import ModelRunner
    from tokenspeed.runtime.execution.runtime_states import RuntimeStates
    from tokenspeed.runtime.layers.attention.backends.base import AttentionBackend
    from tokenspeed.runtime.layers.attention.kv_cache.base import CachePool
    from tokenspeed.runtime.layers.logits_processor import LogitsProcessorOutput

logger = get_colorful_logger(__name__)


def _resolve_dfly_block_geometry(
    cfg: HYV4FlashDFlyConfig, spec_num_tokens: int
) -> tuple[int, int]:
    """Validate and return ``(verify_width, draft_block_size)`` for DFly2.

    DFly2 uses ``sample_from_anchor=False``: slot 0 holds the target's bonus
    token ``x_a`` and slots ``1..verify_width-1`` predict ``draft_block_size =
    verify_width - 1`` draft tokens.
    """
    verify_width = int(spec_num_tokens)
    if verify_width < 2:
        raise ValueError(
            "DFLY block drafting requires speculative_num_draft_tokens >= 2 "
            f"(anchor + at least one draft token); got {verify_width}."
        )
    draft_block_size = verify_width - 1
    ckpt_block_size = read_checkpoint_block_size(cfg)
    if ckpt_block_size is not None:
        validate_block_widths(
            "DFLY",
            ckpt_block_size,
            draft_block_size,
            verify_width,
            allow_shorter=True,
        )
    return verify_width, draft_block_size


def materialize_dense_context_slots(
    page_table: torch.Tensor,
    ctx_lens: torch.Tensor,
    *,
    page_size: int,
    max_ctx_len: int,
) -> torch.Tensor:
    """Expand paged ``page_table [B, max_pages]`` into ``[B, max_ctx_len]`` int32 slots.

    Positions ``>= ctx_lens[b]`` are filled with ``-1``.
    """
    bsz = ctx_lens.shape[0]
    device = ctx_lens.device
    if max_ctx_len <= 0:
        return torch.empty((bsz, 0), dtype=torch.int32, device=device)
    pos = torch.arange(max_ctx_len, dtype=torch.int64, device=device)
    page_idx = torch.div(pos, page_size, rounding_mode="floor").clamp(
        max=page_table.shape[1] - 1
    )
    page_off = torch.remainder(pos, page_size)
    pages = page_table[:, page_idx].to(torch.int64)
    slots = (pages * page_size + page_off.unsqueeze(0)).to(torch.int32)
    valid = pos.unsqueeze(0) < ctx_lens.to(torch.int64).unsqueeze(1)
    return torch.where(valid, slots, torch.full_like(slots, -1))


class DFly(BaseDrafter):
    """Hy4-flash DFly2 drafter sharing target DSA KV cache and top-k index picks."""

    spec_algorithm = "DFLY"
    shares_target_embed_head = True
    supports_speculative_sampling = True
    drafts_from_vocab_shard = False

    def __init__(
        self,
        spec_num_tokens: int,
        spec_num_steps: int,
        draft_model_runner: ModelRunner,
        attn_backend: AttentionBackend | None = None,
        token_to_kv_pool: CachePool | None = None,
        runtime_states: RuntimeStates | None = None,
        input_buffers: InputBuffers | None = None,
        vocab_size: int | None = None,
    ) -> None:
        super().__init__(
            spec_num_tokens,
            spec_num_steps,
            draft_model_runner,
            runtime_states=runtime_states,
            input_buffers=input_buffers,
            attn_backend=attn_backend,
            token_to_kv_pool=token_to_kv_pool,
            vocab_size=vocab_size,
        )
        draft_model = draft_model_runner.model
        if not isinstance(draft_model, HYV4FlashForCausalLMDFly):
            raise TypeError(
                f"DFLY drafter requires HYV4FlashForCausalLMDFly, got {type(draft_model).__name__}"
            )
        self.draft_model: HYV4FlashForCausalLMDFly = draft_model
        self.model = draft_model.model
        self.dfly_config: HYV4FlashDFlyConfig = draft_model.config
        self.device = draft_model_runner.device

        self.verify_width, self.draft_block_size = _resolve_dfly_block_geometry(
            self.dfly_config, spec_num_tokens
        )
        self.target_kv_layers = tuple(int(x) for x in self.dfly_config.target_kv_layers)
        self.target_index_layers = tuple(
            int(x) for x in self.dfly_config.target_index_layer
        )
        self.target_index_topk = int(self.dfly_config.target_index_topk)
        self.target_model: torch.nn.Module | None = None
        self.lm_head: torch.nn.Module | None = draft_model.lm_head
        self.logits_processor = draft_model.logits_processor
        self.is_mrope = True

        max_bs = int(self.input_buffers.max_bs) if self.input_buffers is not None else 1
        self._memory_saver = getattr(
            draft_model_runner, "memory_saver_adapter", None
        ) or TorchMemorySaverAdapter.create(enable=False)
        with self._memory_saver.named_region("executor_inputs"):
            self.next_tokens_buf = torch.zeros(
                (max_bs, self.verify_width), dtype=torch.int32, device=self.device
            )

    @override
    def wire_target(self, target_model: torch.nn.Module) -> None:
        self.target_model = target_model
        self.lm_head = target_model.lm_head
        self.logits_processor = target_model.logits_processor
        if self.draft_model.lm_head is not target_model.lm_head:
            embed, _ = target_model.get_embed_and_head()
            self.draft_model.set_embed_and_head_module(embed, target_model.lm_head)

    @override
    def require_full_draft_logits(self) -> None:
        if self.logits_processor is not None:
            self.logits_processor.require_full_vocab_logits()

    @override
    def idle_forward_global_num_tokens(
        self,
        global_num_tokens: list[int],
        global_bs: list[int],
        *,
        global_num_decodes: list[int],
    ) -> list[list[int]]:
        del global_num_tokens, global_num_decodes
        # DFly2 runs one block forward of width verify_width per request.
        return [[bs * self.verify_width for bs in global_bs]]

    def _project_step_logits(self, corrected_hidden: torch.Tensor) -> torch.Tensor:
        """Return one column per valid global token for both DFly selection modes."""
        head = self.lm_head
        if head is None:
            raise RuntimeError("DFLY drafter has no bound target lm_head")
        processor = self.logits_processor
        if processor is not None:
            # Reusing the plain TP gather buffer across slots requires every
            # peer to have consumed its previous logits. With hidden correction,
            # the embedding all-reduce on the same group orders those reads.
            require_full_vocab = (
                processor.do_argmax
                or self.draft_sampler is not None
                or (
                    processor.tp_size > 1
                    and (
                        self.model.hidden_correction is None
                        or self.model.embed_tokens.tp_group != processor.tp_group
                    )
                )
            )
            metadata = LogitsMetadata(forward_mode=ForwardMode.DECODE, query_shard=None)
            logits = processor._get_logits(
                corrected_hidden,
                head,
                metadata,
                require_full_vocab=require_full_vocab,
            )
        else:
            weight = head.weight
            logits = torch.nn.functional.linear(
                corrected_hidden.to(weight.dtype), weight
            )
            if isinstance(head, VocabParallelEmbedding):
                if head.tp_size > 1:
                    logits = all_gather(logits, head.tp_group, dim=-1)
                if not head.num_added_embeddings:
                    logits = logits[:, : head.org_vocab_size]
                else:
                    shards = logits.unflatten(
                        1, (head.tp_size, head.num_embeddings_per_partition)
                    )
                    base_width = head.shard_indices.num_org_elements_padded
                    base = shards[:, :, :base_width].flatten(1)[
                        :, : head.org_vocab_size
                    ]
                    added = shards[:, :, base_width:].flatten(1)[
                        :, : head.num_added_embeddings
                    ]
                    logits = torch.cat((base, added), dim=1)
        if self.vocab_size is not None and self.vocab_size < logits.shape[1]:
            return logits[:, : self.vocab_size]
        return logits

    def propose_chain(
        self,
        draft_hidden: torch.Tensor,
        anchor_tokens: torch.Tensor,
        *,
        out_tokens: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Run DFly2 sequential ``hidden_correction`` chain walk over ``[B, block, D]``.

        Slot 0 is populated with ``anchor_tokens`` (``x_a``). For each subsequent
        slot ``j = 1..block_size-1``, ``hidden_correction`` conditions on the
        embedding of the preceding token (``x_a`` for ``j=1``, or proposal
        ``j-1`` for ``j>=2``), projects through the shared target FP32 LM head,
        and selects either the greedy argmax or a speculative sample.
        """
        bsz, block_size, _ = draft_hidden.shape
        proposals = (
            out_tokens
            if out_tokens is not None
            else torch.empty(
                (bsz, block_size), dtype=torch.int32, device=draft_hidden.device
            )
        )
        prev_token = anchor_tokens.to(torch.int32)
        max_token_id = int(self.vocab_size) - 1 if self.vocab_size is not None else None
        proposals[:, 0] = (
            prev_token.clamp(0, max_token_id)
            if max_token_id is not None
            else prev_token
        )

        sampler = self.draft_sampler
        for slot in range(1, block_size):
            corrected = self.model.correct_hidden(draft_hidden[:, slot, :], prev_token)
            logits = self._project_step_logits(corrected)
            if sampler is not None:
                # FlashInfer softmax requires dense rows after vocabulary slicing.
                next_tok = sampler.propose(logits.contiguous(), step=slot - 1).to(
                    torch.int32
                )
            else:
                next_tok = logits.argmax(dim=-1).to(torch.int32)
                if max_token_id is not None:
                    next_tok = next_tok.clamp(0, max_token_id)
            proposals[:, slot] = next_tok
            prev_token = next_tok

        if sampler is not None and max_token_id is not None:
            proposals.clamp_(0, max_token_id)
        return proposals

    def _build_round_inputs(
        self,
        base_ctx: ForwardContext,
        hidden_states: torch.Tensor,
        output_tokens: torch.Tensor,
        accept_lengths: torch.Tensor,
    ) -> tuple[
        torch.Tensor,  # anchor_tokens [bs]
        torch.Tensor,  # prev_hidden [bs, D]
        torch.Tensor,  # last_row_indices [bs]
        torch.Tensor,  # ctx_lens [bs] (number of committed target KV tokens < anchor)
        torch.Tensor,  # block_positions [3, bs, verify_width]
    ]:
        """Extract anchor tokens, last-accepted hidden rows, ctx lens, and 3D block positions."""
        bs = base_ctx.bs
        num_extends = base_ctx.num_extends
        num_prefill_outputs = base_ctx.output_layout.num_prefill_outputs
        num_decodes = bs - num_extends
        verify_width = self.verify_width
        device = output_tokens.device

        anchor_tokens = torch.zeros((bs,), dtype=torch.int32, device=device)
        last_rows = torch.zeros((bs,), dtype=torch.int64, device=device)
        ctx_lens = torch.zeros((bs,), dtype=torch.int32, device=device)

        seq_lens = self.input_buffers.seq_lens_buf[:bs].to(torch.int32)

        if num_prefill_outputs > 0:
            anchor_tokens[:num_prefill_outputs] = output_tokens[
                :num_prefill_outputs
            ].to(torch.int32)
            if base_ctx.gather_ids is None:
                raise RuntimeError(
                    "DFLY extend round requires base_ctx.gather_ids for last-token rows"
                )
            last_rows[:num_prefill_outputs] = base_ctx.gather_ids[
                :num_prefill_outputs
            ].to(torch.int64)
            ctx_lens[:num_prefill_outputs] = seq_lens[:num_prefill_outputs]

        if num_decodes > 0:
            dec_accept = (
                accept_lengths[num_extends:bs]
                .to(torch.int64)
                .clamp(min=1, max=verify_width)
            )
            dec_arange = torch.arange(num_decodes, dtype=torch.int64, device=device)
            out_idx = num_prefill_outputs + dec_arange * verify_width + (dec_accept - 1)
            anchor_tokens[num_extends:bs] = output_tokens[out_idx].to(torch.int32)

            prefill_tokens = base_ctx.input_num_tokens - num_decodes * verify_width
            last_rows[num_extends:bs] = (
                prefill_tokens + dec_arange * verify_width + (dec_accept - 1)
            )
            ctx_lens[num_extends:bs] = (
                seq_lens[num_extends:bs] - verify_width + dec_accept.to(torch.int32)
            ).clamp_min(0)

        safe_rows = last_rows.clamp(min=0, max=max(hidden_states.shape[0] - 1, 0))
        prev_hidden = hidden_states.index_select(0, safe_rows)

        # Build 3D M-RoPE positions for the verify_width draft block slots:
        # slot k sits at last_accepted_pos + (k + 1) across all 3 M-RoPE axes.
        offsets = torch.arange(
            1, verify_width + 1, dtype=torch.int64, device=device
        ).view(1, 1, verify_width)
        if self.is_mrope and self.input_buffers.mrope_positions_buf is not None:
            pos_buf = self.input_buffers.mrope_positions_buf
            pos_rows = safe_rows.clamp(max=pos_buf.shape[1] - 1)
            last_pos_3d = pos_buf[:, pos_rows].to(torch.int64)  # [3, bs]
        else:
            pos_buf = self.input_buffers.positions_buf
            pos_rows = safe_rows.clamp(max=pos_buf.shape[0] - 1)
            last_pos_1d = pos_buf[pos_rows].to(torch.int64)
            last_pos_3d = last_pos_1d.unsqueeze(0).expand(3, -1)
        block_positions = last_pos_3d.unsqueeze(-1) + offsets  # [3, bs, verify_width]
        return anchor_tokens, prev_hidden, safe_rows, ctx_lens, block_positions

    def _resolve_context_slots_per_layer(
        self,
        base_ctx: ForwardContext,
        last_rows: torch.Tensor,
        ctx_lens: torch.Tensor,
    ) -> tuple[list[torch.Tensor], list[torch.Tensor]]:
        """Build per-layer ``(ctx_slots [B, K], ctx_lens [B])`` against target DSA KV.

        During decode (including CUDA-graph capture and replay), every target DSA
        layer unconditionally runs ``CompressedIndexer.select_decode`` and records
        fixed-width ``[num_tokens, index_topk]`` global slots (``-1`` padded). We
        consume those directly with zero host synchronization so decode CUDA graphs
        capture cleanly. Only eager extend rounds where a short prompt took the
        dense prefill path (no recorded ``shared_topk``) expand from ``page_table``.
        """
        router = base_ctx.attn_backend
        if not isinstance(router, Hy4FlashCacheGroupRouter):
            raise TypeError(
                "DFLY requires Hy4FlashCacheGroupRouter as target attn_backend, "
                f"got {type(router).__name__}"
            )
        bs = base_ctx.bs
        dense_slots: torch.Tensor | None = None
        ctx_slots_per_layer: list[torch.Tensor] = []
        ctx_lens_per_layer: list[torch.Tensor] = []

        for idx_layer in self.target_index_layers:
            shared_topk = router.get_shared_topk_slots(idx_layer)
            if shared_topk is not None and shared_topk.shape[0] > 0:
                safe_idx = last_rows.clamp(max=shared_topk.shape[0] - 1)
                selected = shared_topk.index_select(0, safe_idx).to(torch.int32)
                valid_counts = (selected >= 0).sum(dim=-1, dtype=torch.int32)
                ctx_slots_per_layer.append(selected)
                ctx_lens_per_layer.append(valid_counts)
            else:
                # Eager extend fallback when prompt length <= index_topk took dense prefill.
                if dense_slots is None:
                    dsa_view = router.group_view("dsa", bs)
                    host_max_ctx = (
                        router.dsa_prefill_max_context_len(base_ctx.num_extends)
                        if base_ctx.num_extends == bs
                        else None
                    )
                    if host_max_ctx is not None:
                        capped = min(host_max_ctx, self.target_index_topk)
                    elif ctx_lens.device.type == "cpu":
                        capped = min(
                            int(ctx_lens.max().item()) if bs > 0 else 0,
                            self.target_index_topk,
                        )
                    else:
                        # Avoid GPU->CPU sync when host lengths are unavailable;
                        # materialize_dense_context_slots masks slots >= ctx_lens to -1.
                        capped = min(
                            int(dsa_view.page_table.shape[1])
                            * int(dsa_view.kernel_page_size),
                            self.target_index_topk,
                        )
                    dense_slots = materialize_dense_context_slots(
                        dsa_view.page_table,
                        ctx_lens.clamp(max=capped),
                        page_size=dsa_view.kernel_page_size,
                        max_ctx_len=capped,
                    )
                ctx_slots_per_layer.append(dense_slots)
                ctx_lens_per_layer.append(
                    ctx_lens.clamp(max=dense_slots.shape[1]).to(torch.int32)
                )
        return ctx_slots_per_layer, ctx_lens_per_layer

    @override
    @nvtx_range("drafter:dfly", color="purple")
    def run(
        self,
        base_ctx: ForwardContext,
        logits_output: LogitsProcessorOutput,
        output_tokens: torch.Tensor,
        accept_lengths: torch.Tensor,
    ) -> torch.Tensor:
        if self.target_model is None:
            raise RuntimeError("DFLY drafter is not bound to the target model.")
        if logits_output.hidden_states is None:
            raise RuntimeError(
                "DFLY requires target post-norm final hidden states in logits_output.hidden_states."
            )
        pool = base_ctx.token_to_kv_pool
        if not isinstance(pool, HybridHy4FlashTokenToKVPool):
            raise TypeError(
                "DFLY requires HybridHy4FlashTokenToKVPool as target token_to_kv_pool, "
                f"got {type(pool).__name__}"
            )
        if self.draft_sampler is not None:
            self.draft_sampler.prepare_round(
                accept_lengths, num_extends=base_ctx.num_extends
            )

        bs = base_ctx.bs
        next_tokens = self.next_tokens_buf[:bs]
        if bs == 0:
            return next_tokens

        (
            anchor_tokens,
            prev_hidden,
            last_rows,
            ctx_lens,
            block_positions,
        ) = self._build_round_inputs(
            base_ctx, logits_output.hidden_states, output_tokens, accept_lengths
        )
        ctx_slots_per_layer, ctx_lens_per_layer = self._resolve_context_slots_per_layer(
            base_ctx, last_rows, ctx_lens
        )
        draft_hidden = self.model.forward_with_target_kv(
            anchor_tokens=anchor_tokens,
            prev_hidden=prev_hidden,
            draft_positions=block_positions,
            target_kv_pool=pool,
            ctx_slots_per_layer=ctx_slots_per_layer,
            ctx_lens_per_layer=ctx_lens_per_layer,
            block_size=self.verify_width,
        )
        return self.propose_chain(draft_hidden, anchor_tokens, out_tokens=next_tokens)

    @override
    def draft(self, *args, **kwargs) -> torch.Tensor | None:
        raise RuntimeError(
            "DFLY drafts through run() with target KV and hidden captures."
        )


__all__ = [
    "DFly",
    "materialize_dense_context_slots",
]
