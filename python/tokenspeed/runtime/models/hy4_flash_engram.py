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

"""Hy4-flash Engram hashed n-gram memory with GPU-sharded or host-resident tables."""

from __future__ import annotations

import logging
import weakref
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from tokenspeed_kernel.ops.embedding import host_gather
from tokenspeed_kernel.platform import current_platform
from torch import nn

from tokenspeed.runtime.configs.hy4_flash_config import HYV4FlashTextConfig
from tokenspeed.runtime.distributed.comm_ops import all_reduce
from tokenspeed.runtime.distributed.mapping import Mapping
from tokenspeed.runtime.execution.request_token_history import RequestTokenHistoryView
from tokenspeed.runtime.utils.env import global_server_args_dict

logger = logging.getLogger(__name__)


def build_hy4_ngram_tokens_from_positions(
    input_ids: torch.Tensor,
    positions: torch.Tensor | None,
    *,
    max_ngram_size: int,
    pad_id: int,
) -> torch.Tensor:
    """Build ``[T, max_ngram_size]`` n-gram ids from packed 1D/3D positions.

    Matches ``HYV4FlashTextModel.build_ngram_tokens`` when no runtime token
    history buffer is active (e.g. standalone prefill tests).
    """
    ids = input_ids.to(torch.int64)
    num_tokens = ids.shape[0]
    if positions is None:
        doc_offsets = torch.arange(num_tokens, device=ids.device, dtype=torch.int64)
    elif positions.ndim == 2 and positions.shape[0] == 3:
        doc_offsets = positions[0].to(torch.int64)
    else:
        doc_offsets = positions.reshape(-1).to(torch.int64)

    shifted = [ids]
    for offset in range(1, max_ngram_size):
        prev = F.pad(ids, (offset, 0), value=pad_id)[:num_tokens]
        shifted.append(torch.where(doc_offsets < offset, pad_id, prev))
    return torch.stack(shifted, dim=-1)


def build_hy4_engram_inputs(
    input_ids: torch.Tensor,
    positions: torch.Tensor | None,
    *,
    config: HYV4FlashTextConfig,
    request_token_history: RequestTokenHistoryView | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Build ``(ngram_tokens [T, max_ngram_size], engram_mask [T])`` graph-safely.

    When ``request_token_history`` is supplied by TokenSpeed's executor, this
    appends the active requests' ``input_ids`` past ``committed_lengths`` in
    place and gathers lookback tokens without host synchronization.
    """
    pad_id = config.engram_pad_id
    max_ngram_size = config.engram_max_ngram_size

    if request_token_history is None:
        ngram_tokens = build_hy4_ngram_tokens_from_positions(
            input_ids,
            positions,
            max_ngram_size=max_ngram_size,
            pad_id=pad_id,
        )
    else:
        history = request_token_history.history_token_ids
        committed = request_token_history.committed_lengths
        req_slots = request_token_history.req_pool_indices
        starts = request_token_history.input_start_offsets
        active_reqs = request_token_history.active_request_mask

        num_tokens = input_ids.shape[0]
        device = input_ids.device
        bs = req_slots.shape[0]
        pad_slot = history.shape[0] - 1
        cap = history.shape[1]

        if num_tokens == 0 or bs == 0:
            ngram_tokens = torch.empty(
                (num_tokens, max_ngram_size), dtype=torch.int64, device=device
            )
        else:
            token_idx = torch.arange(num_tokens, device=device, dtype=starts.dtype)
            req_row = torch.searchsorted(starts[1:], token_idx, right=True).clamp_max(
                bs - 1
            )
            local_j = token_idx - starts[req_row]
            slot = req_slots[req_row].to(torch.int64)
            active = active_reqs[req_row]
            hist_pos = committed[slot].to(torch.int64) + local_j.to(torch.int64)

            safe_slot = torch.where(active, slot, pad_slot)
            safe_write_pos = torch.where(active, hist_pos.clamp(min=0, max=cap - 1), 0)
            history[safe_slot, safe_write_pos] = input_ids.to(history.dtype)

            ids64 = input_ids.to(torch.int64)
            cols = [torch.where(active, ids64, pad_id)]
            for offset in range(1, max_ngram_size):
                read_pos = (hist_pos - offset).clamp(min=0, max=cap - 1)
                fetched = history[safe_slot, read_pos].to(torch.int64)
                cols.append(torch.where(active & (hist_pos >= offset), fetched, pad_id))
            ngram_tokens = torch.stack(cols, dim=-1)

    engram_mask = (
        (input_ids != config.image_token_id)
        & (input_ids != config.video_token_id)
        & (input_ids != pad_id)
    )
    if request_token_history is not None and input_ids.shape[0] > 0:
        bs = request_token_history.req_pool_indices.shape[0]
        if bs > 0:
            starts = request_token_history.input_start_offsets
            token_idx = torch.arange(
                input_ids.shape[0], device=input_ids.device, dtype=starts.dtype
            )
            req_row = torch.searchsorted(starts[1:], token_idx, right=True).clamp_max(
                bs - 1
            )
            engram_mask = (
                engram_mask & request_token_history.active_request_mask[req_row]
            )
    return ngram_tokens, engram_mask


def _fp8_round_trip(x: torch.Tensor) -> torch.Tensor:
    """Per-vector FP8 E4M3 round-trip matching ``HYV4FlashEngramLayer``."""
    fmax = torch.finfo(torch.float8_e4m3fn).max
    scale = x.detach().abs().amax(dim=-1, keepdim=True).clamp(min=1e-12) / fmax
    quantized = (x / scale).clamp(-fmax, fmax).to(torch.float8_e4m3fn).to(
        x.dtype
    ) * scale
    return quantized


class HYV4FlashEngramBranchScales(nn.Module):
    """Per-hyper-connection branch scales matching checkpoint key layout ``.{0..hc_mult-1}``."""

    def __init__(self, hc_mult: int, hidden_size: int) -> None:
        super().__init__()
        self.hc_mult = int(hc_mult)
        for idx in range(self.hc_mult):
            self.register_parameter(str(idx), nn.Parameter(torch.empty(hidden_size)))

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return torch.stack(
            [getattr(self, str(i)) * hidden_states for i in range(self.hc_mult)],
            dim=1,
        )


class HYV4FlashEngramEmbedding(nn.Module):
    """Hash-head embedding table supporting GPU TP row-sharding or host-resident storage."""

    def __init__(
        self,
        head_vocab_sizes: list[int],
        embedding_dim: int,
        *,
        mapping: Mapping | None = None,
        host_table: bool = False,
        layer_id: int = 1,
    ) -> None:
        super().__init__()
        self.head_vocab_sizes = [int(s) for s in head_vocab_sizes]
        self.total_num_embeddings = sum(self.head_vocab_sizes)
        self.embedding_dim = int(embedding_dim)
        self.layer_id = int(layer_id)
        self.host_table = bool(host_table)

        offsets = torch.tensor(
            [0] + self.head_vocab_sizes[:-1], dtype=torch.int64
        ).cumsum(0)
        self.register_buffer("offsets", offsets, persistent=False)

        if mapping is not None and mapping.attn.tp_size > 1:
            tp_size = mapping.attn.tp_size
            tp_rank = mapping.attn.tp_rank
            self.tp_group = mapping.attn.tp_group
            self.row_sharded = True
            self.part_num_embeddings = (
                self.total_num_embeddings + tp_size - 1
            ) // tp_size
            self.row_start = tp_rank * self.part_num_embeddings
            self.row_end = min(
                self.row_start + self.part_num_embeddings,
                self.total_num_embeddings,
            )
            local_rows = max(self.row_end - self.row_start, 1)
        else:
            self.tp_group = None
            self.row_sharded = False
            self.part_num_embeddings = self.total_num_embeddings
            self.row_start = 0
            self.row_end = self.total_num_embeddings
            local_rows = self.total_num_embeddings

        self._host_gpu_registered = False
        self._host_u8: torch.Tensor | None = None
        if self.host_table:
            # Allocate explicitly on CPU even inside a `with torch.device("cuda"):`
            # loader scope, then register once for UVA `host_gather.uint8_row_gather`
            # so CUDA Graph capture runs with zero host synchronization.
            host_weight = torch.empty(
                (local_rows, self.embedding_dim),
                device="cpu",
                dtype=torch.bfloat16,
            )
            self.embedding = nn.Embedding(
                local_rows,
                self.embedding_dim,
                _weight=host_weight,
                _freeze=True,
            )
            self._register_host_for_gpu()
        else:
            self.embedding = nn.Embedding(local_rows, self.embedding_dim)

        self.embedding.weight.weight_loader = self.weight_loader

    def _register_host_for_gpu(self) -> None:
        if not self.host_table or self._host_gpu_registered:
            return
        weight = self.embedding.weight.data
        u8 = weight.view(torch.uint8)
        self._host_u8 = u8
        if weight.device.type == "cpu" and torch.cuda.is_available():
            platform = current_platform()
            if not u8.is_pinned():
                platform.register_host_tensor_for_gpu_access(u8)
                weakref.finalize(
                    self, platform.unregister_host_tensor_for_gpu_access, u8
                )
        self._host_gpu_registered = True

    def _apply(self, fn, recurse=True):
        if self.host_table:
            # Keep the multi-gigabyte host embedding table on CPU while moving
            # small index buffers (offsets) with the surrounding model.
            self.offsets = fn(self.offsets)
            return self
        return super()._apply(fn, recurse)

    def weight_loader(self, param: nn.Parameter, loaded_weight: torch.Tensor) -> None:
        """Load full or row-sharded Engram embedding weight directly into ``param``."""
        if loaded_weight.shape[0] != self.total_num_embeddings:
            raise ValueError(
                f"Engram layer {self.layer_id} embedding rows mismatch: "
                f"checkpoint has {loaded_weight.shape[0]}, expected "
                f"{self.total_num_embeddings}"
            )
        shard = loaded_weight[self.row_start : self.row_end]
        param.data[: shard.shape[0]].copy_(
            shard.to(device=param.device, dtype=param.dtype)
        )

    def load_sharded(self, weight_slice: Any) -> None:
        """Stream only ``[row_start:row_end]`` from a safetensors slice on CPU."""
        chunk_rows = 65536
        target = self.embedding.weight.data
        dst_offset = 0
        for start in range(self.row_start, self.row_end, chunk_rows):
            end = min(start + chunk_rows, self.row_end)
            chunk = weight_slice[start:end]
            target[dst_offset : dst_offset + chunk.shape[0]].copy_(
                chunk.to(device=target.device, dtype=target.dtype),
                non_blocking=target.is_cuda,
            )
            dst_offset += chunk.shape[0]

    def forward(self, hash_ids: torch.Tensor) -> torch.Tensor:
        """Lookup ``[T, num_heads]`` hash ids -> ``[T, num_heads, embedding_dim]``."""
        valid_mask = hash_ids >= 0
        global_ids = hash_ids.clamp_min(0) + self.offsets.to(hash_ids.device)
        weight = self.embedding.weight

        if self.host_table:
            local_mask = (
                valid_mask
                & (global_ids >= self.row_start)
                & (global_ids < self.row_end)
            )
            local_ids = (global_ids - self.row_start).masked_fill(~local_mask, -1)
            table_u8 = (
                self._host_u8
                if self._host_u8 is not None
                else weight.data.view(torch.uint8)
            )
            rows_u8 = host_gather.uint8_row_gather(table_u8, local_ids, None)
            gathered = rows_u8.view(weight.dtype)
            if self.row_sharded and self.tp_group is not None:
                gathered = all_reduce(gathered, group=self.tp_group)
            return gathered

        if not self.row_sharded:
            gathered = F.embedding(global_ids.to(weight.device), weight).to(
                hash_ids.device
            )
            return gathered.masked_fill(~valid_mask.unsqueeze(-1), 0.0)

        local_mask = (
            valid_mask & (global_ids >= self.row_start) & (global_ids < self.row_end)
        )
        local_ids = (global_ids - self.row_start).clamp(min=0, max=weight.shape[0] - 1)
        gathered = F.embedding(local_ids.to(weight.device), weight).to(hash_ids.device)
        gathered = gathered.masked_fill(~local_mask.unsqueeze(-1), 0.0)
        if self.tp_group is not None:
            gathered = all_reduce(gathered, group=self.tp_group)
        return gathered


class HYV4FlashEngramLayer(nn.Module):
    """Engram hashed n-gram memory added to the ``[T, hc_mult, hidden_size]`` streams."""

    def __init__(
        self,
        config: HYV4FlashTextConfig,
        layer_id: int,
        *,
        mapping: Mapping | None = None,
        host_table: bool | None = None,
    ) -> None:
        super().__init__()
        self.layer_id = int(layer_id)
        self.hidden_size = config.hidden_size
        self.hc_mult = config.hc_mult
        self.max_ngram_size = config.engram_max_ngram_size
        self.n_head_per_ngram = config.engram_n_head_per_ngram
        self.fp8_embedding = config.engram_fp8_embedding
        self.pad_id = config.engram_pad_id
        self.seed = config.engram_seed

        if host_table is None:
            host_table = bool(global_server_args_dict.get("engram_host_table", False))
        self.host_table = bool(host_table)

        head_vocab_sizes = [
            int(size)
            for order in config.engram_layer_head_vocab_sizes[str(self.layer_id)]
            for size in order
        ]
        self.config_head_vocab_sizes = head_vocab_sizes
        self.num_hash_heads = len(head_vocab_sizes)
        head_dim = config.engram_n_embed_per_ngram // self.n_head_per_ngram
        engram_size = (self.max_ngram_size - 1) * config.engram_n_embed_per_ngram

        self.register_buffer(
            "lookup_table",
            torch.arange(config.engram_lookup_table_size, dtype=torch.int32),
            persistent=True,
        )
        self.register_buffer(
            "head_vocab_sizes",
            torch.tensor(head_vocab_sizes, dtype=torch.int64),
            persistent=False,
        )
        self.register_buffer(
            "hash_multipliers",
            self.draw_hash_multipliers(config.engram_lookup_table_size),
            persistent=False,
        )

        self.multi_head_embedding = HYV4FlashEngramEmbedding(
            head_vocab_sizes,
            head_dim,
            mapping=mapping,
            host_table=self.host_table,
            layer_id=self.layer_id,
        )
        self.lookup_per_head_norms = nn.ModuleList(
            [nn.RMSNorm(head_dim) for _ in range(self.num_hash_heads)]
        )
        self.value_proj = nn.Linear(engram_size, self.hidden_size, bias=True)
        self.key_projs = nn.ModuleList(
            [
                nn.Linear(engram_size, self.hidden_size, bias=True)
                for _ in range(self.hc_mult)
            ]
        )
        self.norm_key = nn.ModuleList(
            [nn.RMSNorm(self.hidden_size) for _ in range(self.hc_mult)]
        )
        self.norm_query = nn.ModuleList(
            [nn.RMSNorm(self.hidden_size) for _ in range(self.hc_mult)]
        )
        self.unified_rmsnorm = nn.RMSNorm(self.hidden_size, elementwise_affine=False)
        self.unified_pre_gamma = nn.Parameter(torch.empty(self.hidden_size))
        self.unified_per_hc_scales = HYV4FlashEngramBranchScales(
            self.hc_mult, self.hidden_size
        )

    def draw_hash_multipliers(self, compressed_vocab_size: int) -> torch.Tensor:
        """Draw the exact odd int64 multipliers used by official Hy4-flash."""
        half_bound = max(1, np.iinfo(np.int64).max // compressed_vocab_size // 2)
        generator = np.random.default_rng(self.seed + 10007 * self.layer_id)
        drawn = generator.integers(
            0, half_bound, size=self.max_ngram_size, dtype=np.int64
        )
        return torch.tensor((drawn * 2 + 1).tolist(), dtype=torch.int64)

    def refresh_hash_multipliers(self) -> None:
        """Recompute ``hash_multipliers`` from the loaded ``lookup_table`` range."""
        compressed_vocab_size = max(1, int(self.lookup_table.max().item()) + 1)
        drawn = self.draw_hash_multipliers(compressed_vocab_size)
        self.hash_multipliers.copy_(
            drawn.to(
                device=self.hash_multipliers.device,
                dtype=self.hash_multipliers.dtype,
            )
        )

    def hash_ngrams(self, ngram_tokens: torch.Tensor) -> torch.Tensor:
        """Mix ``[T, max_ngram_size]`` shifted token ids into ``[T, num_hash_heads]``."""
        table_size = self.lookup_table.shape[0]
        lookup = self.lookup_table.to(ngram_tokens.device)
        clamped = ngram_tokens.clamp(min=0, max=table_size - 1)
        compressed = lookup[clamped]
        valid_id = (ngram_tokens >= 0) & (ngram_tokens < table_size)
        compressed = torch.where(valid_id, compressed, lookup[self.pad_id])

        multipliers = self.hash_multipliers.to(ngram_tokens.device)
        mixed = compressed.to(torch.int64) * multipliers
        moduli_all = self.head_vocab_sizes.to(ngram_tokens.device)

        hash_ids = []
        for ngram_size in range(2, self.max_ngram_size + 1):
            mix = mixed[..., 0]
            for offset in range(1, ngram_size):
                mix = torch.bitwise_xor(mix, mixed[..., offset])
            order_start = (ngram_size - 2) * self.n_head_per_ngram
            moduli = moduli_all[order_start : order_start + self.n_head_per_ngram]
            hash_ids.append(mix.unsqueeze(-1) % moduli)
        return torch.cat(hash_ids, dim=-1)

    def forward(
        self,
        hidden_states: torch.Tensor,
        ngram_tokens: torch.Tensor,
        engram_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Compute ``[T, hc_mult, hidden_size]`` Engram residual for packed tokens."""
        hash_ids = self.hash_ngrams(ngram_tokens)
        embeddings = self.multi_head_embedding(hash_ids).to(hidden_states.dtype)
        if self.fp8_embedding:
            embeddings = _fp8_round_trip(embeddings)

        embeddings = torch.stack(
            [
                norm(embeddings[:, head])
                for head, norm in enumerate(self.lookup_per_head_norms)
            ],
            dim=1,
        )
        embeddings = embeddings.flatten(start_dim=-2)

        gates = []
        inv_sqrt_d = self.hidden_size**-0.5
        for hc_index in range(self.hc_mult):
            key = self.norm_key[hc_index](self.key_projs[hc_index](embeddings))
            query = self.norm_query[hc_index](hidden_states[:, hc_index])
            gate = (key * query).sum(dim=-1) * inv_sqrt_d
            gate = gate.abs().clamp_min(1e-6).sqrt() * gate.sign()
            gates.append(gate.sigmoid())
        gates_tensor = torch.stack(gates, dim=1).unsqueeze(-1)

        main_signal = self.unified_pre_gamma * self.unified_rmsnorm(
            self.value_proj(embeddings)
        )
        branch = self.unified_per_hc_scales(main_signal)
        engram_output = gates_tensor * main_signal.unsqueeze(1) + branch
        return engram_output.masked_fill(~engram_mask[:, None, None], 0.0)


__all__ = [
    "HYV4FlashEngramBranchScales",
    "HYV4FlashEngramEmbedding",
    "HYV4FlashEngramLayer",
    "build_hy4_engram_inputs",
    "build_hy4_ngram_tokens_from_positions",
]
