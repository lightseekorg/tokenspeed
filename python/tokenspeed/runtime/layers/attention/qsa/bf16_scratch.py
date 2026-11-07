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

"""BF16 KV scratch for large QSA prefill calls over an FP8 KV cache.

With ``--enable-qsa-prefill-bf16-scratch``, a large target prefill/extend call
of a QSA sparse-attention layer over an FP8 E4M3 cache copies every cache page
its requests' contexts span into a BF16 scratch, rewrites its selected slots to
the scratch (``qsa_bf16_kv_scratch``) and attends there with the direct-slot
CuTe kernel's BF16 path, one query row at a time, instead of the FP8 prefill
kernel ``qsa_sparse_attention`` selects: FlashInfer FA2 by default on NVIDIA,
where the CuTe kernel is registered for decode only, or the CuTe kernel's FP8
path where a kernel override selects it for prefill too. The conversion is
exact and the FP8 path reads the cache at unit scale, so both read the same
K/V values and differ only in rounding. The cache itself stays FP8.

A call keeps the FP8 path unless it is a pure extend of the target with
``MIN_ROWS`` to ``max_rows`` query rows whose context fits the scratch and
needs at most ``TOKENS_PER_ROW`` cache tokens per row (for a longer context
the copy costs more than the BF16 kernel saves). Decode, verify, mixed
batches and drafts always keep it.

The cache recipe reserves the scratch (``QSABf16ScratchPlan.nbytes``) from the
KV cache budget before the pool is sized, and the target's QSA leaf allocates
exactly that plan at startup. Serving takes views of it only: BF16 K and V of
``capacity_pages`` pages, and the remapped slots and attention output of
``max_rows`` rows. The page map from cache pages to scratch pages lives in the
output buffer: a call's copy writes every entry its remap reads (the device
``seq_lens`` it copies by match the host lengths it was sized from), and the
attention writes the output after the remap has read them.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import torch
from tokenspeed_kernel.ops.attention.qsa import (
    qsa_bf16_kv_scratch,
    qsa_bf16_scratch_pages,
    select_qsa_sparse_attention_kernel,
)
from tokenspeed_kernel.ops.tuning import is_warming_up
from tokenspeed_kernel.platform import pdl_enabled
from tokenspeed_kernel.selection import NoKernelFoundError

from tokenspeed.runtime.utils import get_colorful_logger

logger = get_colorful_logger(__name__)

# Query rows below which a call keeps the FP8 path: a smaller call does not
# amortize the copy. Every call of at least 256 rows runs the one BF16 launch
# configuration startup warm-up compiles.
MIN_ROWS = 2048
# Cache tokens a call may copy per query row; a call whose context is longer
# for its rows keeps the FP8 path.
TOKENS_PER_ROW = 128
# Requests whose partial last pages the default capacity absorbs beyond one
# maximum-length context plus one prefill chunk.
SLACK_REQUESTS = 64
# The CuTe kernel's selected-slot width and head dimension.
SELECTED_WIDTH = 2051
HEAD_DIM = 256


@dataclass(frozen=True)
class QSABf16ScratchPlan:
    """Geometry of the scratch an engine reserves; independent of the KV pool.

    Attributes:
        page_size: Tokens per page of the full-attention page table.
        kv_heads: KV heads per rank.
        head_dim: Head dimension of K, V and the query.
        q_heads: Query heads per rank.
        capacity_pages: Scratch pages, the reserved page 0 included.
        max_rows: Query rows of the largest call the scratch serves.
    """

    page_size: int
    kv_heads: int
    head_dim: int
    q_heads: int
    capacity_pages: int
    max_rows: int

    @property
    def nbytes(self) -> int:
        """Bytes of BF16 K and V, the int32 remapped slots and the BF16 output."""
        kv_bytes = 2 * self.capacity_pages * self.page_size * self.kv_heads * 2
        slot_bytes = self.max_rows * SELECTED_WIDTH * 4
        output_bytes = self.max_rows * self.q_heads * 2
        return (kv_bytes + output_bytes) * self.head_dim + slot_bytes


def plan_qsa_bf16_scratch(
    *,
    context_len: int,
    max_rows: int,
    page_size: int,
    kv_heads: int,
    head_dim: int,
    q_heads: int,
) -> QSABf16ScratchPlan:
    """Size the scratch from the model length, the prefill chunk and the page.

    The scratch holds what one call can need: one maximum-length context
    plus one chunk of new tokens, with room for the partial last pages of
    ``SLACK_REQUESTS`` requests, but never more pages than a call of
    ``max_rows`` rows may copy (``TOKENS_PER_ROW``).

    Args:
        context_len: Per-request KV extent (``AttnConfig.context_len``).
        max_rows: Query rows of the largest prefill call (the chunk size).
        page_size: Tokens per full-attention kernel page.
        kv_heads: KV heads per rank.
        head_dim: Head dimension.
        q_heads: Query heads per rank.

    Returns:
        The plan the cache recipe reserves and the attention leaf allocates.

    Raises:
        ValueError: The geometry cannot use the scratch.
    """
    if max_rows < MIN_ROWS:
        raise ValueError(
            f"--enable-qsa-prefill-bf16-scratch serves calls of at least "
            f"{MIN_ROWS} rows, but a prefill call holds at most {max_rows}; "
            "raise --chunked-prefill-size"
        )
    if head_dim != HEAD_DIM or kv_heads < 1 or q_heads % kv_heads:
        raise ValueError(
            "--enable-qsa-prefill-bf16-scratch needs 256-wide heads and whole "
            f"query groups, got head_dim={head_dim}, {q_heads} query and "
            f"{kv_heads} KV heads per rank"
        )
    if page_size <= 0:
        raise ValueError("QSA BF16 scratch page size must be positive")
    context_pages = -(-(context_len + max_rows) // page_size) + SLACK_REQUESTS
    row_pages = max_rows * TOKENS_PER_ROW // page_size
    return QSABf16ScratchPlan(
        page_size=page_size,
        kv_heads=kv_heads,
        head_dim=head_dim,
        q_heads=q_heads,
        capacity_pages=1 + min(context_pages, row_pages),
        max_rows=max_rows,
    )


@dataclass(frozen=True)
class QSAExtendLengths:
    """Host lengths of a pure extend forward.

    Attributes:
        seq_lens: Cached prefix plus new tokens of every request, in batch
            order.
        rows: New tokens, the query rows of the forward.
    """

    seq_lens: tuple[int, ...]
    rows: int


def extend_lengths(
    bs: int,
    num_extends: int,
    is_extend: bool,
    extend_prefix_lens_cpu: torch.Tensor,
    extend_seq_lens_cpu: torch.Tensor,
) -> QSAExtendLengths | None:
    """The host lengths of a pure extend batch, or None.

    Args:
        bs: Requests in the batch.
        num_extends: Leading extend requests.
        is_extend: Whether the forward mode is EXTEND (not MIXED or IDLE).
        extend_prefix_lens_cpu: Host cached-prefix lengths ``[>= num_extends]``.
        extend_seq_lens_cpu: Host new-token lengths ``[>= num_extends]``.

    Returns:
        The lengths when every request extends; None for mixed, idle or
        decode batches, whose decode requests have no host lengths.
    """
    if not is_extend or bs <= 0 or num_extends != bs:
        return None
    prefix = extend_prefix_lens_cpu[:bs].tolist()
    extend = extend_seq_lens_cpu[:bs].tolist()
    return QSAExtendLengths(
        seq_lens=tuple(int(p) + int(e) for p, e in zip(prefix, extend, strict=True)),
        rows=sum(int(e) for e in extend),
    )


class QSABf16PrefillScratch:
    """The reserved BF16 scratch and the gate that sends a call to it."""

    def __init__(self, plan: QSABf16ScratchPlan, *, device: torch.device | str) -> None:
        """Allocate ``plan`` once; serving takes views of these tensors only.

        Args:
            plan: The plan the cache recipe reserved.
            device: The attention leaf's device.

        Raises:
            ValueError: This platform has no CuTe DSL QSA attention kernel.
        """
        self.plan = plan
        scratch_slots = plan.capacity_pages * plan.page_size
        self.k = torch.empty(
            (scratch_slots, plan.kv_heads, plan.head_dim),
            dtype=torch.bfloat16,
            device=device,
        )
        self.v = torch.empty_like(self.k)
        # The copy never writes page 0. Padded slots keep their ids, and the
        # BF16 reads that still address slot 1 for them need finite values.
        self.k[: plan.page_size].zero_()
        self.v[: plan.page_size].zero_()
        self.slots = torch.empty(
            (plan.max_rows, SELECTED_WIDTH), dtype=torch.int32, device=device
        )
        self.output = torch.empty(
            (plan.max_rows, plan.q_heads, plan.head_dim),
            dtype=torch.bfloat16,
            device=device,
        )
        # The page map needs one int32 per cache page of a pool sized after
        # this scratch, so it borrows the output (max_rows * q_heads *
        # head_dim / 2 entries): the remap reads it before attention writes.
        self.page_map = self.output.view(torch.int32).view(-1)
        try:
            # The copy pays off through the CuTe kernel's BF16 path, which
            # also writes into ``self.output``; no other solution is wanted.
            self._attention: Callable[..., torch.Tensor] = (
                select_qsa_sparse_attention_kernel(
                    self.output[:1],
                    self.k,
                    self.v,
                    self.slots[:1],
                    max_seqlen_q=1,
                    k_scale=None,
                    v_scale=None,
                    sparse_metadata=None,
                    override=None,
                    solution="cute_dsl",
                )
            )
        except NoKernelFoundError as error:
            raise ValueError(
                "--enable-qsa-prefill-bf16-scratch needs the CuTe DSL QSA "
                f"sparse-attention kernel (SM100 to SM107): {error}"
            ) from error
        logger.info(
            f"QSA BF16 prefill scratch: {plan.capacity_pages} pages x "
            f"{plan.page_size} tokens, {plan.max_rows} rows, "
            f"{self.nbytes / (1 << 20):.1f} MiB reserved from the KV cache budget"
        )

    @property
    def nbytes(self) -> int:
        return sum(
            tensor.nbytes for tensor in (self.k, self.v, self.slots, self.output)
        )

    def attention(
        self,
        q: torch.Tensor,
        k_cache: torch.Tensor,
        v_cache: torch.Tensor,
        selected_slots: torch.Tensor,
        *,
        page_table: torch.Tensor,
        seq_lens: torch.Tensor,
        lengths: QSAExtendLengths | None,
        scale: float,
    ) -> torch.Tensor | None:
        """Attend over the scratch, or return None for the FP8 path.

        Args:
            q: BF16 queries ``[rows, q_heads, head_dim]``, real rows only.
            k_cache: The layer's FP8 E4M3 key cache ``[slots, kv_heads,
                head_dim]``.
            v_cache: Its value cache.
            selected_slots: Int32 physical slots ``[rows, 2051]``.
            page_table: The forward's full-attention page table.
            seq_lens: Its device lengths, which ``lengths`` mirrors.
            lengths: The forward's host lengths; None outside a pure extend.
            scale: Softmax scale.

        Returns:
            A view of the scratch output holding the attention, or None. The
            next call (the next QSA layer) overwrites the view, so it must be
            consumed first. Under startup warm-up a call that qualifies runs
            this path, compiling its kernels, and still returns None, so the
            FP8 path that serves the other calls warms on it too.
        """
        plan = self.plan
        rows = q.shape[0]
        if (
            lengths is None
            or lengths.rows != rows
            or not MIN_ROWS <= rows <= plan.max_rows
            or not q.is_contiguous()
            or (q.is_cuda and torch.cuda.is_current_stream_capturing())
        ):
            return None
        pages = qsa_bf16_scratch_pages(lengths.seq_lens, plan.page_size)
        if (
            pages > plan.capacity_pages
            or (pages - 1) * plan.page_size > rows * TOKENS_PER_ROW
        ):
            return None
        remapped = self.slots[:rows]
        qsa_bf16_kv_scratch(
            k_cache,
            v_cache,
            selected_slots,
            page_table,
            seq_lens,
            num_requests=len(lengths.seq_lens),
            page_size=plan.page_size,
            scratch_pages=pages,
            k_scratch=self.k,
            v_scratch=self.v,
            page_map=self.page_map,
            out_slots=remapped,
            enable_pdl=pdl_enabled(),
            override=None,
            solution=None,
        )
        output = self._attention(
            q,
            self.k,
            self.v,
            remapped,
            scale=scale,
            max_seqlen_q=1,
            metadata_capacity_rows=None,
            k_scale=None,
            v_scale=None,
            out=self.output[:rows],
        )
        return None if is_warming_up() else output


__all__ = [
    "MIN_ROWS",
    "TOKENS_PER_ROW",
    "QSABf16PrefillScratch",
    "QSABf16ScratchPlan",
    "QSAExtendLengths",
    "extend_lengths",
    "plan_qsa_bf16_scratch",
]
