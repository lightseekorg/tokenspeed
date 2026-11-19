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

import math
from contextlib import nullcontext
from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch
from torch import nn

if TYPE_CHECKING:
    from tokenspeed_kernel.ops.ple.host_cache import PLEHostCache

    from tokenspeed.runtime.distributed.mapping import Mapping
    from tokenspeed.runtime.layers.vocab_parallel_embedding import (
        VocabParallelEmbedding,
    )


def _all_gather_ids(output, input_ids, group):
    from tokenspeed.runtime.distributed.comm_ops import all_gather_single

    all_gather_single(output, input_ids, group)


def _all_reduce(values, group):
    from tokenspeed.runtime.distributed.comm_ops import all_reduce

    return all_reduce(values, group)


@dataclass(frozen=True)
class LookupTokenLayout:
    physical_extents: tuple[int, ...]

    def __post_init__(self):
        if not isinstance(self.physical_extents, tuple) or any(
            not isinstance(size, int) or isinstance(size, bool) or size < 0
            for size in self.physical_extents
        ):
            raise ValueError(
                "PLE physical extents must be a tuple of nonnegative integers"
            )


@dataclass(eq=False)
class PendingLookup:
    owner: PLELookup
    embeddings: torch.Tensor
    local_slice: slice
    execution_stream: torch.cuda.Stream | None
    completion: torch.cuda.Event | None
    references: tuple[torch.Tensor, ...]
    consumed: bool = False


_PLE_FP8_MAX = 448.0  # torch.float8_e4m3fn finite maximum
_PLE_QUANT_CHUNK_ELEMENTS = 1 << 20


def materialize_ngram_table_on_host(embedding: VocabParallelEmbedding) -> None:
    """Give a meta-constructed n-gram table page-locked host storage."""

    source = embedding.weight
    host_weight = nn.Parameter(
        torch.empty(source.shape, dtype=source.dtype, device="cpu", pin_memory=True),
        requires_grad=False,
    )
    for name, value in vars(source).items():
        setattr(host_weight, name, value)
    del embedding.weight
    embedding.register_parameter("weight", host_weight)


def host_gather_ngram_rows(
    embedding: VocabParallelEmbedding,
    ids: torch.Tensor,
    scale: float | None,
    out: torch.Tensor,
) -> torch.Tensor:
    """Gather local host rows and apply the table's FP8 scale when present."""

    from tokenspeed_kernel.ops.ple import ple_host_gather

    return ple_host_gather(
        embedding.weight,
        ids,
        out,
        embedding.shard_indices.org_vocab_start_index,
        embedding.shard_indices.org_vocab_end_index,
        scale,
        None,
        table_shards=embedding.tp_size,
    )


class PLELookup(nn.Module):
    def __init__(
        self,
        mapping: Mapping,
        *,
        vocab_size: int,
        ngram_heads: int,
        head_dim: int,
        storage_dtype: torch.dtype | None,
        output_dtype: torch.dtype,
        offload: bool,
        enable_hot_cache: bool,
        prefix: str,
    ):
        super().__init__()
        if enable_hot_cache and not offload:
            raise ValueError(
                "--enable-ple-hot-cache requires ple_offload_embedding=true"
            )
        from tokenspeed.runtime.layers.vocab_parallel_embedding import (
            VocabParallelEmbedding,
        )

        if min(vocab_size, ngram_heads, head_dim) <= 0:
            raise ValueError("PLE lookup dimensions must be positive")
        if storage_dtype not in (None, torch.float8_e4m3fn):
            raise ValueError("Unsupported PLE storage dtype")
        if output_dtype not in (torch.float16, torch.bfloat16, torch.float32):
            raise ValueError("Unsupported PLE compute dtype")
        if offload and not torch.cuda.is_available():
            raise RuntimeError("PLE host offloading requires CUDA UVA")
        self.has_dp = mapping.attn.has_dp
        self.dp_group = mapping.attn.dp_group
        self.dp_rank = self.dp_group.index(mapping.rank)
        self.table_group = (
            mapping.attn.world_group if self.has_dp else mapping.attn.tp_group
        )
        self.table_rank = self.table_group.index(mapping.rank)
        self.ngram_heads = ngram_heads
        self.head_dim = head_dim
        self.embedding_dim = ngram_heads * head_dim
        self.embed_store_dtype = storage_dtype
        self.embed_output_dtype = output_dtype
        self.offload_embedding = offload
        self._checkpoint_weight_scale = 1.0
        self._gather_stream: torch.cuda.Stream | None = None
        self._host_cache: PLEHostCache | None = None
        self._pending: PendingLookup | None = None
        with torch.device("meta") if offload else nullcontext():
            self.ngram_embedding = VocabParallelEmbedding(
                vocab_size,
                head_dim,
                org_num_embeddings=vocab_size,
                params_dtype=storage_dtype or output_dtype,
                prefix=prefix,
                tp_rank=self.table_rank,
                tp_size=len(self.table_group),
                tp_group=self.table_group,
            )
        if offload:
            materialize_ngram_table_on_host(self.ngram_embedding)
        self._compute_device = (
            torch.device("cuda", torch.cuda.current_device())
            if offload
            else self.ngram_embedding.weight.device
        )
        if enable_hot_cache:
            from tokenspeed_kernel.ops.ple.host_cache import PLEHostCache

            from tokenspeed.runtime.utils.env import envs

            shard = self.ngram_embedding.shard_indices
            self._host_cache = PLEHostCache(
                slot_count=envs.TOKENSPEED_PLE_HOST_CACHE_SLOTS.get(),
                table=self.ngram_embedding.weight,
                ngram_heads=ngram_heads,
                output_dtype=output_dtype,
                vocab_start=shard.org_vocab_start_index,
                vocab_end=shard.org_vocab_end_index,
                device=self._compute_device,
            )

    def make_layout(self, global_num_tokens, local_tokens: int) -> LookupTokenLayout:
        if not self.has_dp:
            return LookupTokenLayout((local_tokens,))
        if global_num_tokens is None or len(global_num_tokens) <= max(self.dp_group):
            raise ValueError("Global PLE lookup requires global-rank token extents")
        return LookupTokenLayout(
            tuple(global_num_tokens[rank] for rank in self.dp_group)
        )

    def _gather_ids(self, ids, layout):
        sizes = layout.physical_extents
        if len(sizes) != (len(self.dp_group) if self.has_dp else 1):
            raise ValueError("PLE token layout does not match the DP group")
        rank = self.dp_rank if self.has_dp else 0
        if ids.shape[0] > sizes[rank]:
            raise ValueError("PLE query rows exceed the DP token layout")
        if not self.has_dp:
            return ids, slice(0, ids.shape[0])
        start = sum(sizes[:rank])
        local_slice = slice(start, start + ids.shape[0])
        capacity = max(sizes)
        if capacity == 0:
            return ids, local_slice
        padded = ids.new_zeros((capacity, self.ngram_heads))
        padded[: ids.shape[0]].copy_(ids)
        gathered = ids.new_empty((capacity * len(sizes), self.ngram_heads))
        _all_gather_ids(gathered, padded, self.dp_group)
        return (
            torch.cat(
                [
                    gathered[i * capacity : i * capacity + size]
                    for i, size in enumerate(sizes)
                ]
            ),
            local_slice,
        )

    def start(self, ids: torch.Tensor, layout: LookupTokenLayout) -> PendingLookup:
        if self._pending is not None:
            raise RuntimeError("PLE lookup already has an in-flight request")
        if ids.ndim != 2 or ids.shape[1] != self.ngram_heads:
            raise ValueError("PLE IDs must have shape [tokens, ngram_heads]")
        if ids.dtype != torch.int64 or not ids.is_contiguous():
            raise ValueError("PLE IDs must be contiguous int64")
        if not isinstance(layout, LookupTokenLayout):
            raise TypeError("PLE lookup requires LookupTokenLayout")
        if self.offload_embedding:
            if ids.device != self._compute_device:
                raise ValueError("PLE host gather requires IDs on its compute device")
        elif ids.device != self.ngram_embedding.weight.device:
            raise ValueError("PLE IDs and device table must be on the same device")
        execution_stream = (
            torch.cuda.current_stream(ids.device) if ids.is_cuda else None
        )
        gathered_ids, local_slice = self._gather_ids(ids, layout)
        completion = None
        if self.offload_embedding:
            out = self.allocate_lookup_buffer(gathered_ids.shape[0], ids.device)
            cache = self._host_cache
            use_cache = cache is not None and cache.can_gather(gathered_ids)
            if self._gather_stream is None:
                self._gather_stream = torch.cuda.Stream(device=ids.device)
            stream = self._gather_stream
            stream.wait_stream(execution_stream)
            gathered_ids.record_stream(stream)
            out.record_stream(stream)
            with torch.cuda.stream(stream):
                if use_cache:
                    cache.gather(
                        gathered_ids,
                        out.view(-1, self.head_dim),
                        scale=(
                            self._checkpoint_weight_scale
                            if self.embed_store_dtype is not None
                            else None
                        ),
                    )
                else:
                    self.gather_host(gathered_ids, out)
                completion = torch.cuda.Event()
                completion.record(stream)
        else:
            out = self.gather_device(gathered_ids)
            if ids.is_cuda:
                completion = torch.cuda.Event()
                completion.record(torch.cuda.current_stream(ids.device))
        pending = PendingLookup(
            self, out, local_slice, execution_stream, completion, (ids, gathered_ids)
        )
        self._pending = pending
        return pending

    def finish(self, pending: PendingLookup) -> torch.Tensor:
        if pending.owner is not self:
            raise ValueError("PLE handle belongs to another lookup")
        if pending.consumed or self._pending is not pending:
            raise RuntimeError("PLE lookup handle has already been consumed")
        if pending.completion is not None:
            consumer = torch.cuda.current_stream(pending.embeddings.device)
            stream = pending.execution_stream
            with torch.cuda.stream(stream):
                stream.wait_event(pending.completion)
                pending.embeddings.record_stream(stream)
                for tensor in pending.references:
                    tensor.record_stream(stream)
                values = self.reduce_lookup(pending.embeddings)[pending.local_slice]
                if consumer != stream:
                    ready = torch.cuda.Event()
                    ready.record(stream)
                    consumer.wait_event(ready)
            values.record_stream(consumer)
        else:
            values = self.reduce_lookup(pending.embeddings)[pending.local_slice]
        pending.consumed = True
        pending.references = ()
        self._pending = None
        return values

    def allocate_lookup_buffer(self, tokens: int, device: torch.device) -> torch.Tensor:
        """Destination for a host gather, shaped like a flattened lookup."""

        return torch.empty(
            (tokens, self.embedding_dim),
            dtype=self.embed_output_dtype,
            device=device,
        )

    def warmup_host_cache(self) -> None:
        cache = self._host_cache
        if cache is not None:
            cache.warmup(
                scale=(
                    self._checkpoint_weight_scale
                    if self.embed_store_dtype is not None
                    else None
                )
            )

    def gather_host(self, ids: torch.Tensor, out: torch.Tensor) -> torch.Tensor:
        """Read the ``ids`` rows from host memory into ``out``.

        ``out`` is ``[tokens, ngram_heads * head_dim]``. Row-major order makes
        the per-head rows land contiguously, so the flatten that the device
        path applies afterwards is already implicit here, and the FP8 dequant
        happens inside the kernel instead of materializing an fp32 temporary.
        """
        scale = self._checkpoint_weight_scale if self.embed_store_dtype else None
        host_gather_ngram_rows(
            self.ngram_embedding,
            ids,
            scale,
            out.view(-1, self.head_dim),
        )
        return out

    def reduce_lookup(self, embeddings: torch.Tensor) -> torch.Tensor:
        """Combine per-rank shard contributions of a gathered lookup."""

        if self.ngram_embedding.tp_size > 1 and embeddings.numel():
            return _all_reduce(embeddings, self.ngram_embedding.tp_group)
        return embeddings

    def gather_device(self, ids: torch.Tensor) -> torch.Tensor:
        """Read this rank's device table without reducing other ranks yet."""
        embeddings = self.ngram_embedding(ids, reduce_results=False)
        if self.embed_store_dtype is not None:
            embeddings = (embeddings.float() * self._checkpoint_weight_scale).to(
                self.embed_output_dtype
            )
        return embeddings.flatten(-2)

    def set_online_scale(self, local_amax: float) -> None:
        """Derive the shared BF16 scale once for the logical PLE table.

        Every rank of the table group must call it: the per-rank maxima are
        all-reduced into the table-wide maximum, and the resulting BF16 scale
        is stored on the lookup as ``_checkpoint_weight_scale``.
        """
        peak = torch.tensor(
            local_amax, dtype=torch.float32, device=self._compute_device
        )
        if len(self.table_group) > 1:
            from tokenspeed.runtime.distributed.process_group_manager import (
                process_group_manager,
            )

            torch.distributed.all_reduce(
                peak,
                op=torch.distributed.ReduceOp.MAX,
                group=process_group_manager.get_device_process_group(self.table_group),
            )
        self._checkpoint_weight_scale = (peak / _PLE_FP8_MAX).to(torch.bfloat16).item()

    @staticmethod
    def shard_amax(source: torch.Tensor) -> float:
        lo, hi = torch.aminmax(source)
        return max(abs(lo.item()), abs(hi.item()))

    def quantize_shard(self, source: torch.Tensor, target: torch.Tensor) -> None:
        """Quantize source rows into target with the stored checkpoint scale.

        Rows are quantized in chunks bounded by ``_PLE_QUANT_CHUNK_ELEMENTS``;
        a zero checkpoint scale writes zeros.
        """
        scale = self._checkpoint_weight_scale
        chunk_rows = max(1, _PLE_QUANT_CHUNK_ELEMENTS // source.shape[1])
        for start in range(0, len(source), chunk_rows):
            chunk = source[start : start + chunk_rows].float()
            if scale:
                chunk = (chunk / scale).clamp_(-_PLE_FP8_MAX, _PLE_FP8_MAX)
            else:
                chunk = torch.zeros_like(chunk)
            target[start : start + chunk_rows].copy_(chunk.to(torch.float8_e4m3fn))

    def _checkpoint_shard_size(self, split_parts: int) -> int:
        return (self.ngram_embedding.org_vocab_size + split_parts - 1) // split_parts

    def shard_slices(
        self, row_count: int, shard_index: int, split_parts: int
    ) -> tuple[slice, slice]:
        """Return the (checkpoint source, table target) row slices where the
        given shard overlaps this rank's partition.
        """
        embedding = self.ngram_embedding
        row_start = shard_index * self._checkpoint_shard_size(split_parts)
        row_end = row_start + row_count
        tp_start = embedding.shard_indices.org_vocab_start_index
        tp_end = embedding.shard_indices.org_vocab_end_index
        overlap_start = max(row_start, tp_start)
        overlap_end = min(row_end, tp_end)
        rows = max(0, overlap_end - overlap_start)
        source = overlap_start - row_start
        target = overlap_start - tp_start
        return slice(source, source + rows), slice(target, target + rows)

    def shard_rows(
        self, weight: torch.Tensor, shard_index: int, split_parts: int
    ) -> tuple[torch.Tensor, torch.Tensor] | None:
        """Return the (checkpoint weight rows, table target rows) pair for the
        shard, or ``None`` when the shard does not overlap this rank.
        """
        src, dst = self.shard_slices(len(weight), shard_index, split_parts)
        if src.stop > src.start:
            return weight[src], self.ngram_embedding.weight.data[dst]
        return None

    def needs_online_quantization(self, weight: torch.Tensor) -> bool:
        return (
            self.embed_store_dtype == torch.float8_e4m3fn
            and weight.dtype != torch.float8_e4m3fn
        )

    def load_shard(
        self,
        weight: torch.Tensor,
        shard_index: int,
        split_parts: int,
        *,
        prescanned: bool,
    ) -> None:
        """Copy or online-quantize one checkpoint shard into the table.

        ``prescanned`` must be true once :meth:`set_online_scale` has run for
        the table, which is required when ``split_parts`` is greater than 1;
        raises ``ValueError`` when a split table was not prescanned.
        """
        rows = self.shard_rows(weight, shard_index, split_parts)
        if rows is not None and self._host_cache is not None:
            self._host_cache.invalidate()
        if self.needs_online_quantization(weight):
            if not prescanned:
                if split_parts != 1:
                    raise ValueError(
                        "Split online PLE quantization requires a checkpoint prescan"
                    )
                self.set_online_scale(0.0 if rows is None else self.shard_amax(rows[0]))
            if rows is not None:
                self.quantize_shard(*rows)
        elif rows is not None:
            source, target = rows
            if weight.dtype == torch.float8_e4m3fn and target.dtype != weight.dtype:
                source = source.float() * self._checkpoint_weight_scale
            target.copy_(source)

    def load_scale(self, loaded_weight):
        if loaded_weight.numel() != 1:
            raise ValueError(
                f"Qwen4-Exp PLE weight scale must be scalar, got "
                f"{tuple(loaded_weight.shape)}"
            )
        scale = float(loaded_weight.to(torch.float32).item())
        if not math.isfinite(scale) or scale <= 0:
            raise ValueError(
                f"Qwen4-Exp PLE weight scale must be finite and positive, got "
                f"{scale}"
            )

        if self.embed_store_dtype is None:
            # Compute-dtype target: rescale any raw FP8 rows copied before the
            # scale tensor. Future shard copies multiply by the new value.
            if self._host_cache is not None:
                self._host_cache.invalidate()
            self.ngram_embedding.weight.data.mul_(scale / self._checkpoint_weight_scale)
        self._checkpoint_weight_scale = scale
