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

"""Qwen4-Exp extensions for the hybrid GDN attention backend."""

from __future__ import annotations

from collections.abc import Iterable
from typing import TYPE_CHECKING

import torch

from tokenspeed.runtime.layers.attention.backends.base import CudaGraphSupport
from tokenspeed.runtime.layers.attention.backends.specific.qsa import bind_qsa_indexers
from tokenspeed.runtime.layers.attention.backends.state.mamba import (
    MambaAttnBackend,
    _row_stride_i32,
)
from tokenspeed.runtime.layers.attention.kv_cache.qwen4_exp import (
    QWEN4_EXP_PLE_CACHE_GROUP,
    qwen4_exp_ple_conv_field,
)

if TYPE_CHECKING:
    from tokenspeed.runtime.layers.attention.backends.base import AttentionBackend
    from tokenspeed.runtime.layers.attention.configs.base import (
        AttnConfig,
        SoftmaxAttnConfig,
    )
    from tokenspeed.runtime.layers.attention.qsa.indexer import QSAIndexer
    from tokenspeed.runtime.layers.qwen4_exp_ple import Qwen4ExpPLELayer


def qwen4_exp_linear_backend(
    attn_backend: AttentionBackend,
) -> Qwen4ExpMambaAttnBackend:
    """Resolve and validate the Qwen4-Exp linear-attention backend."""

    backend = getattr(attn_backend, "linear_attn_backend", attn_backend)
    if not isinstance(backend, Qwen4ExpMambaAttnBackend):
        raise RuntimeError("Qwen4-Exp PLE requires its model-specific GDN backend")
    return backend


class Qwen4ExpMambaAttnBackend(MambaAttnBackend):
    """GDN backend with Qwen4-Exp PLE verification state."""

    # Qwen4-Exp's PLE/QSA modules own token-indexed side-state writes; prefill
    # graph replay pads token rows to a bucket while their cache metadata
    # remains real-token shaped. Keep prefills eager so padding can never
    # advance n-gram, short-conv, or compressed-key state.
    cuda_graph_support = CudaGraphSupport(prefill_graph=False)

    def __init__(self, config: AttnConfig, spec: SoftmaxAttnConfig) -> None:
        super().__init__(config, spec)
        self._ple_layers: tuple[Qwen4ExpPLELayer, ...] = ()
        self._ple_verify_scratch: dict[str, torch.Tensor] = {}
        self._ple_verify_tables: dict | None = None
        self._ple_verify_tables_key: tuple | None = None
        self._ple_rows_cache: dict[
            tuple[int, int], tuple[torch.Tensor, torch.Tensor]
        ] = {}

    def _preallocate_aux_verify_workspace(
        self, max_bs: int, draft_token_num: int
    ) -> int:
        self._ensure_ple_verify_scratch(max_bs, draft_token_num)
        return sum(tensor.nbytes for tensor in self._ple_verify_scratch.values())

    def _ensure_ple_verify_scratch(self, max_bs: int, draft_token_num: int) -> None:
        """Allocate graph-stable PLE context and convolution rollback rows."""
        arena = getattr(self.kv_pool, "arena", None)
        plan = getattr(arena, "plan", None)
        if plan is None:
            return
        fields = [
            field
            for field in plan.fields
            if field.group_id == QWEN4_EXP_PLE_CACHE_GROUP
        ]
        if not fields:
            return
        rows = max_bs * (draft_token_num + 1)
        if self._ple_verify_scratch and all(
            tensor.shape[0] >= rows for tensor in self._ple_verify_scratch.values()
        ):
            return
        scratch: dict[str, torch.Tensor] = {}
        for field in fields:
            cache_field = arena.field(field.field_id)
            scratch[field.field_id] = cache_field.new_zeros(
                (rows, *cache_field.shape[1:])
            )
        self._ple_verify_tables = None
        self._ple_verify_tables_key = None
        self._ple_rows_cache = {}
        self._ple_verify_scratch = scratch

    def ple_verify_scratch(
        self, context_field_id: str, layer_id: int
    ) -> tuple[torch.Tensor, torch.Tensor] | None:
        """Return shared context and per-layer PLE convolution verify rows."""
        context = self._ple_verify_scratch.get(context_field_id)
        conv = self._ple_verify_scratch.get(qwen4_exp_ple_conv_field(layer_id))
        if context is None or conv is None:
            return None
        return context, conv

    def bind_ple_layers(self, layers: Iterable[Qwen4ExpPLELayer]) -> None:
        """Bind the stable model-owned PLE layers used by verify commits."""

        bound = tuple(layers)
        if self._ple_layers is bound:
            return
        if self._ple_layers:
            same = len(self._ple_layers) == len(bound) and all(
                previous is current
                for previous, current in zip(self._ple_layers, bound, strict=True)
            )
            if not same:
                raise RuntimeError("PLE backend cannot be rebound to another model")
        self._ple_layers = bound

    @staticmethod
    def _u64(values: list[int], device: torch.device) -> torch.Tensor:
        return torch.tensor(values, dtype=torch.uint64, device=device)

    @staticmethod
    def _i64(values: list[int], device: torch.device) -> torch.Tensor:
        return torch.tensor(values, dtype=torch.int64, device=device)

    def _ple_verify_tables_get(self) -> dict:
        """Build pointer and stride tables for the batched PLE commit."""

        arena = self.kv_pool.arena
        first_layer = self._ple_layers[0]
        context_field = arena.field(first_layer.context_field_id)
        conv_fields = [
            arena.field(qwen4_exp_ple_conv_field(layer.layer_id))
            for layer in self._ple_layers
        ]
        context_scratch = self._ple_verify_scratch[first_layer.context_field_id]
        conv_scratches = [
            self._ple_verify_scratch[qwen4_exp_ple_conv_field(layer.layer_id)]
            for layer in self._ple_layers
        ]
        key = (
            tuple(layer.layer_id for layer in self._ple_layers),
            arena.buffer.data_ptr(),
        )
        if self._ple_verify_tables is not None and self._ple_verify_tables_key == key:
            return self._ple_verify_tables

        conv_shape = tuple(conv_fields[0].shape[1:])
        conv_dtype = conv_fields[0].dtype
        conv_dst_stride = _row_stride_i32(conv_fields[0])
        for layer, field in zip(self._ple_layers, conv_fields, strict=True):
            if tuple(field.shape[1:]) != conv_shape or field.dtype != conv_dtype:
                raise RuntimeError(
                    f"PLE layer {layer.layer_id} convolution cache geometry "
                    f"{tuple(field.shape[1:])}/{field.dtype} differs from layer "
                    f"{first_layer.layer_id} {conv_shape}/{conv_dtype}"
                )
            if _row_stride_i32(field) != conv_dst_stride:
                raise RuntimeError(
                    f"PLE layer {layer.layer_id} convolution page stride "
                    f"{_row_stride_i32(field)} must match layer "
                    f"{first_layer.layer_id} {conv_dst_stride}"
                )
        for layer, scratch in zip(self._ple_layers, conv_scratches, strict=True):
            if tuple(scratch.shape[1:]) != conv_shape or scratch.dtype != conv_dtype:
                raise RuntimeError(
                    f"PLE layer {layer.layer_id} convolution verify scratch "
                    f"geometry {tuple(scratch.shape[1:])}/{scratch.dtype} differs "
                    f"from its cache field {conv_shape}/{conv_dtype}"
                )
        if (
            tuple(context_scratch.shape[1:]) != tuple(context_field.shape[1:])
            or context_scratch.dtype != context_field.dtype
        ):
            raise RuntimeError(
                "PLE context verify scratch geometry differs from its cache field"
            )

        device = context_field.device
        tables = {
            "context_src": self._u64([context_scratch.data_ptr()], device),
            "context_dst": self._u64([context_field.data_ptr()], device),
            "context_src_stride": self._i64([_row_stride_i32(context_scratch)], device),
            "context_dst_stride": self._i64([_row_stride_i32(context_field)], device),
            "context_row_bytes": context_field[0].numel()
            * context_field.element_size(),
            "conv_src": self._u64(
                [scratch.data_ptr() for scratch in conv_scratches], device
            ),
            "conv_dst": self._u64([field.data_ptr() for field in conv_fields], device),
            "conv_src_stride": self._i64(
                [_row_stride_i32(scratch) for scratch in conv_scratches], device
            ),
            "conv_dst_stride": self._i64([conv_dst_stride] * len(conv_fields), device),
            "conv_row_bytes": conv_fields[0][0].numel() * conv_fields[0].element_size(),
            "num_layers": len(conv_fields),
        }
        self._ple_verify_tables = tables
        self._ple_verify_tables_key = key
        return tables

    def _ple_verify_rows(
        self, bs: int, num_layers: int
    ) -> tuple[torch.Tensor, torch.Tensor]:
        key = (bs, num_layers)
        rows = self._ple_rows_cache.get(key)
        if rows is None:
            rows = (
                torch.empty(num_layers * bs, dtype=torch.int64, device=self.device),
                torch.empty(num_layers * bs, dtype=torch.int64, device=self.device),
            )
            self._ple_rows_cache[key] = rows
        return rows

    def _commit_aux_verified_state(
        self,
        accepted_length: torch.Tensor,
        pages_by_group: dict[str, torch.Tensor],
    ) -> None:
        pages = pages_by_group.get(QWEN4_EXP_PLE_CACHE_GROUP)
        if pages is None or not self._ple_layers:
            return
        bs = accepted_length.shape[0]
        bucket = None
        for layer in self._ple_layers:
            selected = layer.verify_scratch_bucket(bs)
            if selected is None:
                return
            if bucket is None:
                bucket = selected
            elif bucket[0] != selected[0]:
                raise RuntimeError(
                    "Qwen4-Exp PLE layers disagree on the verify scratch "
                    f"bucket: {bucket[0]} vs {selected[0]}"
                )
        width = bucket[0][1]
        tables = self._ple_verify_tables_get()
        num_layers = tables["num_layers"]
        src_rows, dst_rows = self._ple_verify_rows(bs, num_layers)
        from tokenspeed_kernel.ops.kvcache.triton import (
            copy_state_rows,
            state_verify_commit_rows,
        )

        state_verify_commit_rows(
            accepted_length,
            pages,
            src_rows,
            dst_rows,
            verify_width=width,
            num_layers=num_layers,
        )
        if tables["context_row_bytes"]:
            copy_state_rows(
                tables["context_src"],
                tables["context_dst"],
                src_rows[:bs],
                dst_rows[:bs],
                row_bytes=tables["context_row_bytes"],
                src_row_strides=tables["context_src_stride"],
                dst_row_strides=tables["context_dst_stride"],
            )
        copy_state_rows(
            tables["conv_src"],
            tables["conv_dst"],
            src_rows,
            dst_rows,
            row_bytes=tables["conv_row_bytes"],
            src_row_strides=tables["conv_src_stride"],
            dst_row_strides=tables["conv_dst_stride"],
        )


def bind_qwen4_exp_side_state(
    attn_backend: AttentionBackend,
    ple_layers: Iterable[Qwen4ExpPLELayer],
    qsa_indexers: Iterable[QSAIndexer],
) -> None:
    """Bind Qwen4-Exp PLE and QSA state to their owning backends."""

    ple_layers = tuple(ple_layers)
    qsa_indexers = tuple(qsa_indexers)
    if ple_layers:
        qwen4_exp_linear_backend(attn_backend).bind_ple_layers(ple_layers)
    if qsa_indexers:
        bind_qsa_indexers(attn_backend, qsa_indexers)


__all__ = [
    "Qwen4ExpMambaAttnBackend",
    "bind_qwen4_exp_side_state",
    "qwen4_exp_linear_backend",
]
