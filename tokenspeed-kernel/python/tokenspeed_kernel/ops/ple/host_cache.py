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

"""Opt-in HBM hot cache for offloaded PLE embedding table shards.

Eligible eager lookups (offloaded BF16 or scalar-scaled FP8 tables on NVIDIA
Hopper and Blackwell, at least 256 tokens) read recently gathered rows from
GPU memory instead of the host; smaller lookups and CUDA graph capture and
replay use the regular host gather. The staged gather and fill kernels live
in :mod:`tokenspeed_kernel.ops.ple._host_cache_triton`.
"""

from __future__ import annotations

import logging

import torch
from tokenspeed_kernel._triton import triton
from tokenspeed_kernel.ops.ple._host_cache_triton import (
    _ple_fill_staged_kernel,
    _ple_gather_staged_tail_kernel,
)
from tokenspeed_kernel.platform import current_platform
from tokenspeed_kernel.thirdparty.cuda.ple_host_gather import (
    ple_staged_gather,
    validate_ple_hot_cache_geometry,
)

logger = logging.getLogger(__name__)


class PLEHostCache:
    """Read a rank's FP8 or BF16 PLE table shard through its own HBM cache.

    Construction validates the supported table layout and device. The
    table bytes must remain immutable between gathers, or the owner must call
    :meth:`invalidate` before updating the table.
    """

    HEADS = 16
    TILE_TOKENS = 64
    MIN_TOKENS = 256
    MAX_TOKENS = ((1 << 31) - 1) // HEADS
    _FILL_ROWS = 16

    def __init__(
        self,
        *,
        slot_count: int,
        table: torch.Tensor,
        ngram_heads: int,
        output_dtype: torch.dtype,
        vocab_start: int,
        vocab_end: int,
        device: torch.device,
    ) -> None:
        """Select a cache for a pinned FP8 or BF16 table shard.

        Args:
            slot_count: Number of cache slots, a power of two greater than one.
            table: Pinned host table of shape ``[local_vocab_rows, head_dim]``.
            ngram_heads: Number of row IDs per input token.
            output_dtype: Compute dtype of the gather output.
            vocab_start: Inclusive global ID of the first local row.
            vocab_end: Exclusive global ID of the last local row.
            device: GPU consuming the table and owning the cache.

        Raises:
            ValueError: Invalid slot count or unsupported dtype.
            RuntimeError: Unsupported GPU.
        """
        if slot_count < 2 or slot_count & (slot_count - 1):
            raise ValueError("PLE cache slots must be a power of two greater than one")
        platform = current_platform()
        if not platform.is_nvidia or device.type != "cuda":
            raise RuntimeError("PLE hot cache requires an NVIDIA CUDA device")
        capability = torch.cuda.get_device_capability(device)
        if capability[0] not in (9, 10):
            raise RuntimeError(
                f"PLE hot cache requires Hopper or Blackwell, got SM{capability[0]}{capability[1]}"
            )
        if (
            table.dtype not in (torch.float8_e4m3fn, torch.bfloat16)
            or output_dtype != torch.bfloat16
        ):
            raise ValueError(
                f"PLE hot cache requires FP8 E4M3FN or BF16 storage and BF16 output, "
                f"got {table.dtype} storage and {output_dtype} output"
            )
        head_dim = table.shape[1]
        validate_ple_hot_cache_geometry(ngram_heads, head_dim, vocab_start, vocab_end)
        launch = ple_staged_gather()
        logger.info(
            f"PLE hot cache enabled: slots={slot_count}, storage={table.dtype}, "
            f"heads={ngram_heads}, head_dim={table.shape[1]}; "
            f"eager lookups with {self.MIN_TOKENS}..{self.MAX_TOKENS} tokens use the cache; "
            "other sizes and CUDA graph capture/replay use regular host gather"
        )
        self.slot_count = slot_count
        self._slot_bits = slot_count.bit_length() - 1
        self._device = device
        self._head_dim = head_dim
        self._block_dim = triton.next_power_of_2(self._head_dim)
        self._vocab_start = vocab_start
        self._vocab_end = vocab_end
        self._table = table
        self._storage_dtype = table.dtype
        self._is_fp8 = table.dtype == torch.float8_e4m3fn
        self._launch = launch
        self._tags: torch.Tensor | None = None
        self._hot: torch.Tensor | None = None
        self._winners: torch.Tensor | None = None
        self._fill_stream: torch.cuda.Stream | None = None
        self._fill_done: torch.cuda.Event | None = None
        self._pending = False

    def can_gather(self, ids: torch.Tensor) -> bool:
        """Select cache-safe sizes; flat winner indices must fit int32."""
        return (
            self.MIN_TOKENS <= ids.shape[0] <= self.MAX_TOKENS
            and not torch.cuda.is_current_stream_capturing()
        )

    def warmup(self, *, scale: float | None) -> None:
        """Compile the tail gather and fill for this cache before serving."""
        ids = torch.full(
            (self.MIN_TOKENS + 1, self.HEADS),
            self._vocab_start,
            dtype=torch.int64,
            device=self._device,
        )
        out = torch.empty(
            (ids.numel(), self._head_dim), dtype=torch.bfloat16, device=self._device
        )
        self.gather(ids, out, scale=scale)
        self.invalidate()

    def invalidate(self) -> None:
        """Finish the in-flight fill and invalidate the current table contents."""
        if self._pending:
            self._fill_done.synchronize()
        self._pending = False
        self._tags = self._hot = self._winners = None

    def gather(
        self,
        ids: torch.Tensor,
        out: torch.Tensor,
        *,
        scale: float | None,
    ) -> torch.Tensor:
        """Fill ``out`` with a BF16 lookup or a scalar-scaled FP8 lookup.

        Args:
            ids: Contiguous CUDA int64 global IDs of shape ``[tokens, 16]``.
                The caller must select this path with :meth:`can_gather`.
            out: Contiguous CUDA BF16 output of shape ``[tokens * 16, head_dim]``.
            scale: FP32 dequantization factor for an FP8 table, or None for BF16.

        Returns:
            The supplied output, with rows outside the shard zeroed, ready on
            the current stream. A side stream copies staged misses into the
            cache for the next cached lookup.
        """
        if self._is_fp8 != (scale is not None):
            raise ValueError("PLE host cache requires a scale only for FP8")
        scale_value = scale if scale is not None else 1.0
        device = ids.device
        stream = torch.cuda.current_stream(device)
        if self._fill_stream is None:
            self._fill_stream = torch.cuda.Stream(device=device)
            self._fill_done = torch.cuda.Event()
        if self._pending:
            stream.wait_event(self._fill_done)
        if self._tags is None:
            self._tags = torch.full(
                (self.slot_count,), -1, dtype=torch.int32, device=device
            )
            self._hot = torch.empty(
                (self.slot_count, self._head_dim),
                dtype=self._storage_dtype,
                device=device,
            )
            self._winners = torch.empty_like(self._tags)
            for buf in (self._tags, self._hot, self._winners):
                buf.record_stream(self._fill_stream)

        query = ids.view(-1)
        rows = query.numel()
        staged = torch.empty_like(out, dtype=self._storage_dtype)
        misses = torch.empty_like(query, dtype=torch.int32)
        self._winners.fill_(-1)
        address = current_platform().device_visible_data_ptr(self._table)
        aligned_tokens = ids.shape[0] // self.TILE_TOKENS * self.TILE_TOKENS
        self._launch(
            address,
            ids,
            self._tags,
            self._hot.view(torch.uint8),
            out,
            misses,
            scale_value,
            self._is_fp8,
            aligned_tokens,
            self._vocab_start,
            self._vocab_end,
            staged.view(torch.uint8),
            self._winners,
        )
        aligned_rows = aligned_tokens * self.HEADS
        tail = rows - aligned_rows
        if tail:
            _ple_gather_staged_tail_kernel[(triton.cdiv(tail, self._FILL_ROWS),)](
                address,
                query[aligned_rows:],
                self._tags,
                self._hot,
                out[aligned_rows:],
                staged[aligned_rows:],
                misses[aligned_rows:],
                self._winners,
                tail,
                self._vocab_start,
                self._vocab_end,
                aligned_rows,
                scale_value,
                self._slot_bits,
                R=self._FILL_ROWS,
                D=self._head_dim,
                C=self._block_dim,
                num_warps=4,
            )

        self._fill_stream.wait_stream(stream)
        for buf in (query, staged, misses):
            buf.record_stream(self._fill_stream)
        with torch.cuda.stream(self._fill_stream):
            _ple_fill_staged_kernel[(triton.cdiv(rows, self._FILL_ROWS),)](
                query,
                staged,
                misses,
                self._winners,
                self._tags,
                self._hot,
                rows,
                self._vocab_start,
                D=self._head_dim,
                R=self._FILL_ROWS,
                BLOCK=self._block_dim,
                num_warps=4,
            )
            self._fill_done.record(self._fill_stream)
        self._pending = True
        return out
