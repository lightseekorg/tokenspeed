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

"""Execution-owned verification state shared by a model's QSA indexers.

Persistent request caches belong to the LCM arena. This runtime binds only
its side's cache fields and owns the temporary, capacity-sized verify
workspace. The executor commits acceptance after eager execution or graph
replay; paged attention backends have no reference to this object.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch
from tokenspeed_kernel.ops.attention.triton.qwen4_exp_qsa import (
    qwen4_exp_qsa_commit_verify_layers,
)

from tokenspeed.runtime.layers.attention.kv_cache.qwen4_exp import (
    QWEN4_EXP_QSA_RECENT_CACHE_GROUP,
    qsa_rope_position_field,
)
from tokenspeed.runtime.layers.attention.kv_cache.recipes.plan import (
    cache_field_layer_id,
)

if TYPE_CHECKING:
    from tokenspeed.runtime.layers.attention.configs.base import AttnConfig
    from tokenspeed.runtime.layers.attention.kv_cache.base import CachePool


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


class QSAIndexerRuntime:
    """One side's shared QSA staging and batched post-verification commit."""

    def __init__(self, config: AttnConfig, cache_pool: CachePool) -> None:
        self.cache_pool = cache_pool
        self.dtype = config.dtype
        self.device = config.device
        self.is_draft = bool(config.is_draft)
        self.spec_num_tokens = max(int(config.speculative_num_draft_tokens or 1), 1)
        self._recent_page_size = next(
            spec.block_granularity
            for spec in cache_pool.arena.cache_group_specs
            if spec.group_id == QWEN4_EXP_QSA_RECENT_CACHE_GROUP
        )
        self._slots: dict[int, int] = {}
        self._verify_workspace: _QSAVerifyWorkspace | None = None
        self._verify_staged = False

    def bind_indexers(self, model: torch.nn.Module) -> None:
        """Share this runtime with the loaded model's indexers at startup.

        Args:
            model: This side's loaded model, restricted to the local PP stage.
                Binding never derives cache addresses from model modules.
        """
        from tokenspeed.runtime.layers.attention.qsa.indexer import QSAIndexer

        for module in model.modules():
            if isinstance(module, QSAIndexer):
                module.runtime = self

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

    def commit_after_verify(
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
            self._recent_page_size,
            workspace.raw_cache.shape[1],
            verify_width=self.spec_num_tokens,
        )


__all__ = ["QSAIndexerRuntime"]
