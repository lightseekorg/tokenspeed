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

"""The paged-KV attention leaf: the kernel-facing contract.

A ``PagedAttentionBackend`` serves the layers of exactly one cache group and
perceives exactly what a paged attention kernel perceives — a kernel page
table, cache sequence lengths and write slots. It never sees cache groups,
scheduler block tables, the pool's contract or which side (target/draft)
it runs on beyond the ``is_draft`` verify-floor geometry. The
``CacheGroupRouter`` owns all of that and hands every leaf a fully resolved
``page_table`` (kernel pages, batch-ordered, padded to ``[bs,
max_num_pages]``) on every metadata call; the leaf copies it into its own
graph-recorded buffers. This is the interface the backends had before cache
groups existed, minus the scheduler-table gather the router now performs.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from contextlib import contextmanager
from typing import TYPE_CHECKING

import torch

from tokenspeed.runtime.layers.attention.backends.support import CudaGraphSupport
from tokenspeed.runtime.utils.common import ceil_div

if TYPE_CHECKING:
    from tokenspeed.runtime.execution.forward_batch_info import ForwardMode
    from tokenspeed.runtime.layers.attention.configs.base import (
        AttnConfig,
        SoftmaxAttnConfig,
    )
    from tokenspeed.runtime.layers.attention.kv_cache.base import CachePool
    from tokenspeed.runtime.layers.paged_attention import PagedAttention


class PagedAttentionBackend(ABC):
    """One cache group's paged attention kernels and their metadata.

    Persistent decode state is leaf-owned: ``init_cuda_graph_state`` allocates
    the ``[rows, max_num_pages]`` page-table buffer and the ``[rows]``
    cache-seqlens buffer a captured graph records, and every
    ``refresh_decode_metadata`` copies the router-supplied table and lengths
    into them in place. Extend metadata is rebuilt per round from the
    supplied table (never graph-recorded; the prefill graph breaks at
    attention).
    """

    # Static CUDA-graph capability of this leaf class (see support.py).
    cuda_graph_support: CudaGraphSupport = CudaGraphSupport()
    # Whether a block draft (DFLASH/DSpark) expands decode rows to
    # spec_num_tokens per request; set from config by leaves that support it.
    draft_block_decode: bool = False
    # PD layerwise transfer counter, installed by the router; the MLA
    # leaves record the step inside their chunked prefill.
    step_counter = None
    # The kernel's page span when the config does not override it
    # (kernel_page_sizes.py). None means the kernel is page-size flexible and
    # views the cache at its group's own block granularity (no expansion).
    default_kernel_page_size: int | None = None

    @classmethod
    def resolve_kernel_page_size(
        cls, config: AttnConfig, block_granularity: int
    ) -> int:
        """The page span this leaf class runs at for a group of
        ``block_granularity`` tokens per scheduler block: the config override
        when given, else the class default, else the group's own grain."""
        if config.kernel_page_size is not None:
            return int(config.kernel_page_size)
        if cls.default_kernel_page_size is not None:
            return int(cls.default_kernel_page_size)
        return int(block_granularity)

    def __init__(
        self, config: AttnConfig, spec: SoftmaxAttnConfig, *, kernel_page_size: int
    ) -> None:
        """Args:
        config: The side's attention config (device, dtypes, spec width).
        spec: The softmax attention component this leaf serves.
        kernel_page_size: Tokens per kernel page, resolved by the caller
            from the kernel registry (``kernel_page_sizes.py``) and the
            group's block granularity.
        """
        self.device = config.device
        self.dtype = config.dtype
        self.is_draft = bool(config.is_draft)
        self.spec_num_tokens = max(int(config.speculative_num_draft_tokens or 1), 1)
        self.max_context_len = int(config.context_len)
        if kernel_page_size <= 0:
            raise ValueError(
                f"kernel_page_size must be positive, got {kernel_page_size}"
            )
        self.kernel_page_size = int(kernel_page_size)
        self.max_num_pages = ceil_div(self.max_context_len, self.kernel_page_size)
        self.num_qo_heads = spec.num_attention_heads // spec.attn_tp_size
        self.num_kv_heads = max(spec.num_kv_heads // spec.attn_tp_size, 1)
        self.head_dim = spec.head_dim
        self.cache_pool: CachePool | None = None

    # ------------------------------------------------------------------
    # Static shape / lifecycle
    # ------------------------------------------------------------------

    def set_cache_pool(self, cache_pool: CachePool) -> None:
        """Remember the pool whose buffers this leaf's kernels read."""
        self.cache_pool = cache_pool

    def child_backends(self) -> tuple:
        """Leaves compose nothing (the DSA leaf overrides for its dense child)."""
        return ()

    def configure_runtime(self, **kwargs) -> None:
        """Post-load configuration hook (e.g. sliding window sizes)."""

    def init_prefill_graph_state(self, max_num_tokens: int, max_bs: int) -> None:
        """Allocate static buffers the breakable prefill graphs bake; most
        leaves keep attention eager at the break points and need none."""

    @property
    def verify_floor(self) -> int:
        """Minimum per-row cache seq_len decode metadata must present.

        The target's verify window spans ``seq - N .. seq - 1`` so its rows
        need ``seq_len >= N``; plain decode and drafts have floor 1, where the
        clamp is the identity.
        """
        return self.spec_num_tokens if not self.is_draft else 1

    @property
    def block_decode_active(self) -> bool:
        """DFLASH/DSpark block draft: spec_num_tokens decode rows per request."""
        return self.draft_block_decode and self.spec_num_tokens > 1

    # ------------------------------------------------------------------
    # Metadata
    # ------------------------------------------------------------------

    @abstractmethod
    def init_cuda_graph_state(self, max_bs: int) -> None:
        """Allocate the persistent decode buffers for up to ``max_bs`` rows
        (``max_bs * spec_num_tokens`` under block decode). Runs
        unconditionally at wrapper construction, enforce-eager included: the
        unified refresh serves eager decode from the same buffers."""

    @abstractmethod
    def init_forward_metadata(
        self,
        bs: int,
        num_extends: int,
        seq_lens: torch.Tensor,
        page_table: torch.Tensor,
        forward_mode: ForwardMode,
        *,
        extend_seq_lens: torch.Tensor,
        extend_seq_lens_cpu: torch.Tensor,
        extend_prefix_lens: torch.Tensor,
        extend_prefix_lens_cpu: torch.Tensor,
        extend_with_prefix: bool,
        **kwargs,
    ) -> None:
        """Build extend/mixed (or idle warmup) metadata.

        Args:
            bs: Rows in the batch (extend rows first, then any decode rows).
            num_extends: Leading extend rows; ``bs`` for a pure extend.
            seq_lens: ``[>= bs]`` total cache lengths after this step.
            page_table: ``[bs, max_num_pages]`` kernel page table,
                batch-ordered and padded (row i is batch position i).
            forward_mode: EXTEND, MIXED or IDLE; a DECODE call is a contract
                violation (decode metadata is :meth:`refresh_decode_metadata`).
            extend_*: ``[>= num_extends]`` per-request new-token / prefix
                lengths with their pinned host mirrors (empty on idle warmup).
            extend_with_prefix: Whether any extend row continues a cached or
                chunked prefix (some ``extend_prefix_lens`` entry is non-zero);
                leaves that size ragged-vs-paged prefill metadata read it.
        """

    @abstractmethod
    def refresh_decode_metadata(
        self,
        bs: int,
        actual_bs: int,
        seq_lens: torch.Tensor,
        page_table: torch.Tensor,
        *,
        num_extends: int = 0,
        for_graph_replay: bool = False,
    ) -> None:
        """The single decode metadata path — eager decode and graph replay.

        Copies ``seq_lens`` (clamped to :attr:`verify_floor`) and
        ``page_table`` into the persistent buffers in place and points
        ``forward_decode_metadata`` at the pointer-stable per-bs views over
        them. Rows in ``[actual_bs, bs)`` are padding (already null pages in
        ``page_table``, seq_len 1 in ``seq_lens``); ``actual_bs == 0`` is the
        idle replay and the capture seeding.

        Args:
            bs: Rows to prepare (the padded graph batch under replay).
            actual_bs: Live rows.
            seq_lens: ``[>= bs]`` live cache lengths.
            page_table: ``[bs, max_num_pages]`` kernel page table for these
                rows, batch-ordered and padded.
            num_extends: Leading extend rows of a MIXED round whose decode
                half this refresh describes; 0 for pure decode.
            for_graph_replay: A graph is in play (live replay or the capture
                seeding). Branch on it only for graph-mechanics asymmetries
                a shared in-place refresh cannot express.
        """

    def init_forward_metadata_capture_cuda_graph(
        self, bs: int, seq_lens: torch.Tensor, page_table: torch.Tensor
    ) -> None:
        """Capture seeding: the idle refresh over the same buffers replay
        refreshes. Override only for a kernel-imposed capture asymmetry."""
        self.refresh_decode_metadata(bs, 0, seq_lens, page_table, for_graph_replay=True)

    def advance_draft_forward_metadata(self, seq_lens: torch.Tensor) -> None:
        """Publish a drafter's in-graph seq_lens edits into this leaf's own
        cache-seqlens buffer (one token per request per step)."""
        buf = self.decode_seq_lens_buffer
        bs = seq_lens.shape[0]
        buf[:bs].copy_(seq_lens[:bs])

    def fill_block_decode_seq_lens(self, bs: int, block_seq_lens: torch.Tensor) -> None:
        """DFLASH: broadcast each request's block-end length to its
        spec_num_tokens decode rows (uniform, non-causal). Leaves whose
        kernel takes one row per request override with the bs-row form."""
        spec = self.spec_num_tokens
        self.decode_seq_lens_buffer[: bs * spec].view(bs, spec).copy_(
            block_seq_lens[:bs].clamp(spec, self.max_context_len).unsqueeze(1)
        )

    @property
    @abstractmethod
    def decode_seq_lens_buffer(self) -> torch.Tensor:
        """The persistent cache-seqlens buffer decode metadata views."""

    @contextmanager
    def override_num_extends(self, num_extends: int):
        """Temporarily override the decode-row slice discriminator (MLA
        family: drafter step 0 slices ``[num_extends:]``, step 1+ ``[0:]``).
        Default no-op for leaves with separate prefill/decode slots."""
        yield

    def support_kv_cache_prewrite(
        self, forward_mode: ForwardMode | None = None
    ) -> bool:
        return False

    def set_request_slots(self, req_pool_indices: torch.Tensor) -> None:
        """Publish this forward's ``[bs]`` request-pool slots (batch order).

        The one side channel beyond ``page_table`` / ``seq_lens``: a leaf that
        owns per-request side state indexed by pool slot (DSA's KPool tails)
        reads it; paged KV leaves need nothing but the table. The router
        calls this after every metadata build (extend init, decode refresh,
        capture seeding), before the model runs. Default: no-op.
        """
        del req_pool_indices

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    @abstractmethod
    def forward_decode(
        self,
        q: torch.Tensor,
        k: torch.Tensor | None,
        v: torch.Tensor | None,
        layer: PagedAttention,
        out_cache_loc: torch.Tensor,
        token_to_kv_pool: CachePool,
        bs: int,
        save_kv_cache: bool = True,
        **kwargs,
    ) -> torch.Tensor:
        """Decode attention over the current ``forward_decode_metadata``."""

    @abstractmethod
    def forward_extend(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        layer: PagedAttention,
        out_cache_loc: torch.Tensor,
        token_to_kv_pool: CachePool,
        bs: int,
        save_kv_cache: bool = True,
        **kwargs,
    ) -> torch.Tensor:
        """Extend attention over the current extend metadata."""

    def forward_extend_chunked(self, *args, **kwargs):
        """DeepSeek's chunked prefix replay (MLA family); others never call it."""
        raise NotImplementedError(
            f"{type(self).__name__} has no chunked prefix-replay prefill"
        )
