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


def quantize_ple_embedding_rows(
    rows: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Online per-row FP8 quantization for the n-gram table.

    Checkpoint shards stream in row ranges, so each row is quantized
    independently (``scale = amax / 448``) -- no whole-table amax prescan is
    needed and no clipping can occur. Returns the FP8 rows and their fp32
    dequant scales.
    """

    values = rows.to(torch.float32)
    scale = (values.abs().amax(dim=1) / _PLE_FP8_MAX).clamp_min(1e-12)
    quantized = (values / scale.unsqueeze(1)).to(torch.float8_e4m3fn)
    return quantized, scale


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
    row_scale: torch.Tensor | None = None,
) -> torch.Tensor:
    """Fill ``out`` with the ``ids`` rows of a host-resident n-gram table.

    The dequant factor follows the checkpoint format (see
    :func:`_ple_host_gather_kernel`): ``row_scale`` is the device per-row buffer
    for an online-quantized table, ``scale`` the per-tensor scalar for an
    offline FP8 one, and both ``None`` for a compute-dtype table that needs
    none.
    """

    from tokenspeed_kernel.ops.ple import ple_host_gather

    return ple_host_gather(
        embedding.weight,
        ids,
        out,
        embedding.shard_indices.org_vocab_start_index,
        embedding.shard_indices.org_vocab_end_index,
        scale,
        row_scale,
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
        prefix: str,
    ):
        super().__init__()
        from tokenspeed.runtime.layers.vocab_parallel_embedding import (
            VocabParallelEmbedding,
        )

        if mapping.attn.has_dp and mapping.attn.has_cp:
            raise ValueError("Global PLE lookup with DP requires attention CP=1")
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
        self.register_buffer(
            "ngram_embedding_scale",
            (
                torch.ones(
                    self.ngram_embedding.num_embeddings_per_partition,
                    dtype=torch.float32,
                    device=self._compute_device,
                )
                if storage_dtype is not None and not offload
                else None
            ),
            persistent=False,
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
            if self._gather_stream is None:
                self._gather_stream = torch.cuda.Stream(device=ids.device)
            stream = self._gather_stream
            stream.wait_stream(torch.cuda.current_stream(ids.device))
            gathered_ids.record_stream(stream)
            out.record_stream(stream)
            with torch.cuda.stream(stream):
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

    def gather_host(self, ids: torch.Tensor, out: torch.Tensor) -> torch.Tensor:
        """Read the ``ids`` rows from host memory into ``out``.

        ``out`` is ``[tokens, ngram_heads * head_dim]``. Row-major order makes
        the per-head rows land contiguously, so the flatten that the device
        path applies afterwards is already implicit here, and the FP8 dequant
        happens inside the kernel instead of materializing an fp32 temporary.
        The dequant factor follows the checkpoint format: a per-row device
        buffer for an online-quantized table, the per-tensor scalar for an
        offline FP8 one (see :meth:`__init__`), and none for a compute-dtype
        table.
        """

        row_scale = self.ngram_embedding_scale
        scale = (
            self._checkpoint_weight_scale
            if row_scale is None and self.embed_store_dtype is not None
            else None
        )
        host_gather_ngram_rows(
            self.ngram_embedding,
            ids,
            scale,
            out.view(-1, self.head_dim),
            row_scale=row_scale,
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
            embeddings = self._dequant(embeddings, ids)
        return embeddings.flatten(-2)

    def _dequant(self, raw: torch.Tensor, ids: torch.Tensor) -> torch.Tensor:
        """Cast the FP8 lookup to compute dtype and apply per-row scales.

        Must run before the TP all-reduce: FP8 payloads cannot be reduced and
        each row's scale lives only on its owning rank. Shard-masked rows were
        zero-filled by the embedding forward, so gathering their folded row-0
        scale is harmless (0 * finite = 0).
        """

        from tokenspeed.runtime.layers.vocab_parallel_embedding import (
            get_masked_input_and_mask,
        )

        if self.ngram_embedding_scale is None:
            return (raw.float() * self._checkpoint_weight_scale).to(
                self.embed_output_dtype
            )
        module = self.ngram_embedding
        if module.tp_size > 1:
            local_ids, _ = get_masked_input_and_mask(
                ids,
                module.shard_indices.org_vocab_start_index,
                module.shard_indices.org_vocab_end_index,
                module.shard_indices.num_org_vocab_padding,
                module.shard_indices.added_vocab_start_index,
                module.shard_indices.added_vocab_end_index,
            )
        else:
            local_ids = ids.clamp(min=0, max=module.num_embeddings_padded - 1)
        scale = self.ngram_embedding_scale[local_ids]
        return (raw.to(torch.float32) * scale.unsqueeze(-1)).to(self.embed_output_dtype)

    def load_shard(self, loaded_weight, shard_index: int, split_parts: int):
        embedding = self.ngram_embedding
        shard_size = (embedding.org_vocab_size + split_parts - 1) // split_parts
        row_start = shard_index * shard_size
        row_end = row_start + loaded_weight.shape[0]
        tp_start = embedding.shard_indices.org_vocab_start_index
        tp_end = embedding.shard_indices.org_vocab_end_index
        overlap_start = max(row_start, tp_start)
        overlap_end = min(row_end, tp_end)
        if overlap_start < overlap_end:
            destination = overlap_start - tp_start
            source = overlap_start - row_start
            rows = overlap_end - overlap_start
            source_rows = loaded_weight[source : source + rows]
            target_rows = embedding.weight.data[destination : destination + rows]
            scale_buffer = self.ngram_embedding_scale
            source_is_fp8 = source_rows.dtype == torch.float8_e4m3fn
            target_is_fp8 = target_rows.dtype == torch.float8_e4m3fn
            # Matching formats are copied unchanged. In particular, FP8-to-FP8
            # preserves the checkpoint payload; its global scale is loaded by
            # _load_ple_weight_scale independently of checkpoint weight ordering.
            if target_is_fp8 and not source_is_fp8:
                if scale_buffer is None:
                    if not self.offload_embedding:
                        raise RuntimeError(
                            "FP8 PLE embedding is missing its scale buffer"
                        )
                    # A compute-dtype checkpoint under offload is quantized online,
                    # one streamed shard at a time, and its FP8 payload lands on the
                    # host table. The offline FP8 path offloading targets carries
                    # only a per-tensor scale, so the per-row buffer is not built at
                    # construction; allocate it lazily here, on the device the
                    # gather reads scales from.
                    scale_buffer = torch.ones(
                        embedding.num_embeddings_per_partition,
                        device=self._compute_device,
                        dtype=torch.float32,
                    )
                    self.ngram_embedding_scale = scale_buffer
                # Quantize compute-dtype checkpoint rows for FP8 storage and retain
                # their independently derived dequant scales.
                source_rows, scale = quantize_ple_embedding_rows(source_rows)
                scale_buffer[destination : destination + rows].copy_(
                    scale.to(scale_buffer.device, scale_buffer.dtype)
                )
            elif source_is_fp8 and not target_is_fp8:
                source_rows = (
                    source_rows.to(torch.float32) * self._checkpoint_weight_scale
                )
            target_rows.copy_(source_rows.to(target_rows.device, target_rows.dtype))

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

        scale_buffer = self.ngram_embedding_scale
        if scale_buffer is not None:
            # The checkpoint scale is shared by every pre-quantized row, so one
            # fill handles both scale-before-shards and scale-after-shards order.
            scale_buffer.fill_(scale)
        elif self.embed_store_dtype is not None:
            # Offloaded FP8: the payload stays FP8 on the host and this per-tensor
            # scalar is applied by the gather kernel, so there is no per-row buffer
            # to fill and no payload to rescale -- recording it below is enough.
            pass
        else:
            # Compute-dtype target: rescale any raw FP8 rows copied before the
            # scale tensor. Future shard copies multiply by the new value.
            self.ngram_embedding.weight.data.mul_(scale / self._checkpoint_weight_scale)
        self._checkpoint_weight_scale = scale
