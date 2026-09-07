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

"""Backend-owned QSA layout and speculative verification lifecycle.

The ordinary cache-group router owns the page tables. This runtime consumes
its resolved views and owns shared verify staging; the paged QSA leaf only
executes attention, and the model indexer owns weights and top-k selection.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch
from tokenspeed_kernel.ops.attention.triton.qwen4_exp_qsa import (
    qwen4_exp_qsa_commit_verify_layers,
    qwen4_exp_qsa_prepare_metadata,
)

from tokenspeed.runtime.layers.attention.kv_cache.qwen4_exp import (
    QWEN4_EXP_QSA_CACHE_GROUP,
    QWEN4_EXP_QSA_RECENT_CACHE_GROUP,
    qsa_rope_position_field,
)
from tokenspeed.runtime.layers.attention.kv_cache.recipes.plan import (
    cache_field_layer_id,
)
from tokenspeed.runtime.layers.attention.kv_cache.recipes.spec import FULL_ATTENTION

if TYPE_CHECKING:
    from tokenspeed.runtime.execution.context import ForwardContext
    from tokenspeed.runtime.layers.attention.backends.base import AttentionBackend
    from tokenspeed.runtime.layers.attention.backends.paged.mha import (
        MHADecodeMetadata,
        MHAExtendMetadata,
    )
    from tokenspeed.runtime.layers.attention.backends.paged.router import (
        CacheGroupRouter,
    )
    from tokenspeed.runtime.layers.attention.configs.base import AttnConfig


def decode_query_lengths(
    ctx: ForwardContext, total_tokens: int, *, force_uniform: bool
) -> int | None:
    """Return the uniform decode query width, or None for a ragged extend.

    Args:
        ctx: The live model forward context.
        total_tokens: Query rows, after any draft step-0 narrowing.
        force_uniform: Whether a narrowed draft requires uniform rows even
            when its original context still names an extend forward.

    Returns:
        Query rows per request for uniform forwards, otherwise None.
    """
    if not ctx.bs or (
        not force_uniform
        and (ctx.forward_mode is None or not ctx.forward_mode.is_decode())
    ):
        return None
    if total_tokens % ctx.bs:
        raise RuntimeError("Qwen4-Exp QSA decode rows must be divisible by batch size")
    return total_tokens // ctx.bs


@dataclass(frozen=True)
class QSALayout:
    """Layer-invariant cache geometry for one QSA model forward."""

    seq_lens: torch.Tensor
    logical_positions: torch.Tensor
    request_indices: torch.Tensor
    qsa_locs: torch.Tensor
    recent_locs: torch.Tensor
    complete_blocks: torch.Tensor
    qsa_page_table: torch.Tensor
    qsa_page_expansion: int
    full_page_table: torch.Tensor
    full_kernel_page_size: int
    reset_draft_tags: torch.Tensor | None


@dataclass(frozen=True)
class _QSAVerifyWorkspace:
    """Capacity-sized staging and cache addresses shared by every QSA layer."""

    token_k: torch.Tensor
    position_values: torch.Tensor
    logical_positions: torch.Tensor
    recent_locs: torch.Tensor
    raw_addresses: torch.Tensor
    position_addresses: torch.Tensor
    raw_cache: torch.Tensor
    position_cache: torch.Tensor

    @property
    def nbytes(self) -> int:
        # The cache views already belong to the LCM arena's budget.
        return sum(
            tensor.nbytes
            for tensor in (
                self.token_k,
                self.position_values,
                self.logical_positions,
                self.recent_locs,
                self.raw_addresses,
                self.position_addresses,
            )
        )


class QSARuntime:
    """Shared QSA indexing layout and target-verify staging for one router."""

    def __init__(self, config: AttnConfig, router: CacheGroupRouter) -> None:
        self.router = router
        self.dtype = config.dtype
        self.device = config.device
        self._slots: dict[int, int] = {}
        self._verify_workspace: _QSAVerifyWorkspace | None = None
        self._verify_staged = False

    @property
    def is_draft(self) -> bool:
        return self.router.is_draft

    @property
    def spec_num_tokens(self) -> int:
        return self.router.spec_num_tokens

    @property
    def sparse_topk(self):
        return self.router.sparse_topk

    @property
    def cache_pool(self):
        return self.router.cache_pool

    def _metadata(self, ctx: ForwardContext) -> MHAExtendMetadata | MHADecodeMetadata:
        leaf = self.router.leaves[FULL_ATTENTION]
        metadata = (
            leaf.forward_extend_metadata
            if ctx.forward_mode.is_extend_or_mixed()
            else leaf.forward_decode_metadata
        )
        if metadata is None:
            raise RuntimeError(
                f"QSA found no {ctx.forward_mode} metadata on the full-attention leaf"
            )
        return metadata

    def qsa_forward_layout(
        self,
        ctx: ForwardContext,
        total_tokens: int,
        *,
        compressed_token_page_size: int,
        recent_page_size: int,
        compress_ratio: int,
        reset_draft_tags: torch.Tensor | None,
    ) -> QSALayout:
        """Build or reuse the QSA row layout shared by every local QSA layer."""

        cached = self.sparse_topk.qsa_metadata
        if cached is not None:
            if not isinstance(cached, QSALayout):
                raise RuntimeError("invalid QSA per-forward metadata memo")
            if (
                cached.logical_positions.shape[0] != total_tokens
                or cached.seq_lens.shape[0] < ctx.bs
            ):
                raise RuntimeError("stale QSA per-forward metadata memo")
            if (
                reset_draft_tags is not None
                and cached.reset_draft_tags is not reset_draft_tags
            ):
                reset_draft_tags.fill_(torch.iinfo(torch.int64).min)
            return cached

        metadata = self._metadata(ctx)
        query_lengths = decode_query_lengths(
            ctx,
            total_tokens,
            force_uniform=False,
        )
        if query_lengths is None:
            query_lengths = metadata.extend_seq_lens[: ctx.bs]
        qsa = self.router.group_view(QWEN4_EXP_QSA_CACHE_GROUP, ctx.bs)
        recent = self.router.group_view(QWEN4_EXP_QSA_RECENT_CACHE_GROUP, ctx.bs)
        full = self.router.group_view(FULL_ATTENTION, ctx.bs)
        qsa_page_table, qsa_expansion = qsa.page_table, qsa.pages_per_block
        recent_page_table, recent_expansion = recent.page_table, recent.pages_per_block
        seq_lens = metadata.seq_lens[: ctx.bs]
        logical, requests, qsa_locs, recent_locs, complete_blocks = (
            qwen4_exp_qsa_prepare_metadata(
                seq_lens,
                query_lengths,
                total_tokens,
                qsa_page_table,
                qsa_expansion,
                compressed_token_page_size,
                recent_page_table,
                recent_expansion,
                recent_page_size,
                compress_ratio,
                draft_logical_positions=reset_draft_tags,
            )
        )
        layout = QSALayout(
            seq_lens=seq_lens,
            logical_positions=logical,
            request_indices=requests,
            qsa_locs=qsa_locs,
            recent_locs=recent_locs,
            complete_blocks=complete_blocks,
            qsa_page_table=qsa_page_table,
            qsa_page_expansion=qsa_expansion,
            full_page_table=full.page_table,
            full_kernel_page_size=full.kernel_page_size,
            reset_draft_tags=reset_draft_tags,
        )
        self.sparse_topk.qsa_metadata = layout
        return layout

    def preallocate_verify_workspace(self, max_bs: int, draft_token_num: int) -> int:
        """Allocate staging and commit addresses from the bound cache plan."""
        if self.is_draft or draft_token_num <= 1:
            return 0
        if self._verify_workspace is not None:
            return self._verify_workspace.nbytes

        pool = self.cache_pool
        fields = [
            field
            for field in pool.arena.plan.fields
            if field.group_id == QWEN4_EXP_QSA_RECENT_CACHE_GROUP
            and field.field_id.endswith(".qsa.raw_key")
            and cache_field_layer_id(field.field_id) in pool.field_layer_range
        ]
        if not fields:
            raise RuntimeError("QSA cache view has no raw-key fields")
        self._slots = {
            cache_field_layer_id(field.field_id): slot
            for slot, field in enumerate(fields)
        }
        raw_fields = [pool.arena.field(field.field_id) for field in fields]
        position_fields = [
            pool.arena.field(qsa_rope_position_field(layer_id))
            for layer_id in self._slots
        ]
        raw, positions = raw_fields[0], position_fields[0]
        for tensors in (raw_fields, position_fields):
            first = tensors[0]
            if any(
                (tensor.shape, tensor.stride(), tensor.dtype)
                != (first.shape, first.stride(), first.dtype)
                for tensor in tensors[1:]
            ):
                raise RuntimeError("QSA verify cache fields must have uniform geometry")

        shape = (max_bs, draft_token_num)
        self._verify_workspace = _QSAVerifyWorkspace(
            token_k=torch.empty(
                (len(fields), *shape, *raw.shape[2:]),
                dtype=self.dtype,
                device=self.device,
            ),
            position_values=positions.new_empty((*shape, *positions.shape[1:])),
            logical_positions=torch.empty(shape, dtype=torch.int64, device=self.device),
            recent_locs=torch.empty(shape, dtype=torch.int32, device=self.device),
            raw_addresses=torch.tensor(
                [tensor.data_ptr() for tensor in raw_fields],
                dtype=torch.uint64,
                device=self.device,
            ),
            position_addresses=torch.tensor(
                [tensor.data_ptr() for tensor in position_fields],
                dtype=torch.uint64,
                device=self.device,
            ),
            raw_cache=raw,
            position_cache=positions,
        )
        return self._verify_workspace.nbytes

    def verify_staging_buffers(
        self, layer_id: int, bs: int
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return preallocated destinations for a local layer's verify rows."""
        workspace = self._verify_workspace
        if workspace is None:
            raise RuntimeError(
                "QSA verify workspace must be preallocated before forward"
            )
        capacity, width = workspace.token_k.shape[1:3]
        if not 0 < bs <= capacity or width != self.spec_num_tokens:
            raise RuntimeError(
                f"QSA verify batch ({bs}, {self.spec_num_tokens}) exceeds or differs "
                f"from the preallocated shape ({capacity}, {width})"
            )
        slot = self._slots[self.cache_pool._field_layer_id(layer_id)]
        self._verify_staged = True
        return (
            workspace.token_k[slot, :bs],
            workspace.position_values[:bs],
            workspace.logical_positions[:bs],
            workspace.recent_locs[:bs],
        )

    def commit_after_mtp_verify(
        self, accepted_lengths: torch.Tensor, *, num_extends: int
    ) -> None:
        """Commit accepted target-verify candidates for all QSA layers once."""
        if num_extends < 0 or num_extends > accepted_lengths.shape[0]:
            raise ValueError(
                "QSA verify commit received an invalid extend prefix: "
                f"{num_extends} for {accepted_lengths.shape[0]} requests"
            )
        verify_lengths = accepted_lengths[num_extends:]
        bs = verify_lengths.shape[0]
        if bs == 0 or not self._verify_staged:
            return
        workspace = self._verify_workspace
        qwen4_exp_qsa_commit_verify_layers(
            workspace.raw_addresses,
            workspace.position_addresses,
            workspace.token_k,
            workspace.logical_positions[:bs].reshape(-1),
            workspace.recent_locs[:bs].reshape(-1),
            workspace.position_values[:bs].flatten(0, 1),
            verify_lengths,
            workspace.raw_cache,
            workspace.position_cache,
            self.router.geometry.granularity_of(QWEN4_EXP_QSA_RECENT_CACHE_GROUP),
            workspace.raw_cache.shape[1],
            verify_width=self.spec_num_tokens,
        )


def require_qsa_runtime(attn_backend: AttentionBackend) -> QSARuntime:
    """Return the registered router runtime used by a target or draft model.

    Args:
        attn_backend: The model's router or outer hybrid backend.

    Returns:
        The QSA runtime created by the backend registry at construction.
    """
    router = getattr(attn_backend, "full_attn_backend", attn_backend)
    runtime = getattr(router, "runtime", None)
    if not isinstance(runtime, QSARuntime):
        raise RuntimeError("Qwen4-Exp QSA indexers require a registered QSA runtime")
    return runtime


__all__ = ["QSARuntime", "QSALayout", "require_qsa_runtime"]
