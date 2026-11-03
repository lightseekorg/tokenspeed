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

"""LongCat-Flash N-gram runtime plugin.

Registers the ``LongcatFlashNgramForCausalLM`` architecture (the checkpoint's
``modeling_longcat_ngram``) on the in-tree ``LongcatFlashForCausalLM`` model:
the decoder body is the in-tree LongCat implementation and the checkpoint's
``model.ngram_embeddings.*`` tables (12 embedders + 12 post projections) are
served by :class:`NgramEmbedding`, whose forward mirrors the checkpoint
reference implementation (base word embedding + per-neighbor polynomial-hash
embedder/projection contributions, normalized by ``1 + k * (n - 1)``).

**Host-resident n-gram tables (W18 scheme 2).** The embedder tables are
huge — 12 tables of ``~10.24M x 256`` bf16 each, ~63 GB in total for the
LongCat-Flash-Lite checkpoint — so they live on the host (CPU RAM) and are
gathered on demand: every forward only moves the current batch's hashed ids
to the CPU, looks up the rows there, and moves the (small) per-slot rows
back to the NPU.  The device keeps only the post projections (~18 MB).
The tables are created with an explicit ``device='cpu'`` (the runtime builds
the model inside a ``torch.device(target)`` context, which would otherwise
allocate the 63 GB on the accelerator) and a ``_apply`` override keeps them
on the CPU even when the rest of the module is moved (``.to('npu')`` and
friends).  The ``nn.ModuleList`` is preserved so the checkpoint's 24
``model.ngram_embeddings.*`` keys keep resolving through
``named_parameters()`` (embedder weights land in the CPU tables, post
projection weights land in the NPU linears).

**Device staging for ACL Graph decode (方案C L1).** The per-step
host<->device transfers cannot run inside a captured decode graph (910B3
rejects host memcpy under stream capture), so when the decode ACL Graph is
enabled the executor preloads the current batch's table rows H2D into a
fixed device ``staging`` buffer *outside* the graph (``preload_ngram_staging``
called from ``ModelExecutor.settle()``) and the in-graph forward reads only
the staging.  The preload reuses the same ``_compute_new_ids`` hash the
in-graph partial computes, so the staging rows align with the in-graph reads
bit-exactly (verified maxdiff=0 over ACL Graph replays).  ``use_staging``
selects the lookup path: False keeps the original host lookup (eager
baseline), True reads the preloaded staging (decode ACL Graph).  Staging
allocation is decoupled from capture — ``_ensure_staging`` allocates it on
module placement and in ``preload_staging``, before any capture — so a
captured graph always records reads of an already-fixed staging address and
the eager baseline never silently falls back to the host lookup inside a
graph (the fallback is what 910B3 rejects as 107030).

The module is an in-tree runtime plugin: importing it runs ``register()``,
which calls :func:`tokenspeed.runtime.plugins.registry.register_model`. The
model registry's ``import_model_classes`` imports the in-tree plugin modules
of this package, so every process that resolves an architecture sees the
registration without needing a ``tokenspeed.plugins`` distribution entry
point. The checkpoint's own ``configuration_longcat_ngram`` remote code keeps
parsing ``config.json`` (it carries ``model_type=None`` and the Ngram arch),
so no in-tree config class is registered here.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from tokenspeed.runtime.distributed.comm_ops import all_reduce
from tokenspeed.runtime.models.longcat_flash import (
    LongcatFlashForCausalLM,
    _RuntimeLongcatModel,
)
from tokenspeed.runtime.plugins.registry import register_model


def _ensure_ngram_config(config) -> None:
    """Fill defaults for the n-gram embedding hyper-parameters."""
    if not hasattr(config, "ngram_vocab_size_ratio"):
        config.ngram_vocab_size_ratio = 78
    if not hasattr(config, "emb_split_num"):
        config.emb_split_num = 4
    if not hasattr(config, "emb_neighbor_num"):
        config.emb_neighbor_num = 4
    if not hasattr(config, "eos_token_id"):
        config.eos_token_id = 2
    if not hasattr(config, "pad_token_id"):
        config.pad_token_id = None
    # Caller-owned predecessor-window width (n-1): the shared ngram-history
    # machinery (``prepare_ngram_inputs`` / ``fill_ngram_inputs``) resolves
    # per-token predecessor windows for this model, so the executor hands the
    # embedding its predecessor window instead of request-level history rows.
    config.ngram_context_len = config.emb_neighbor_num - 1
    return config


class _HostEmbedding(nn.Embedding):
    """Host-resident n-gram table with a cheap deterministic initialization.

    The standard ``nn.Embedding`` reset fills the whole table with
    ``normal_`` — for the LongCat-Flash-Lite tables that is ~63 GB of random
    numbers, on the order of twenty minutes of pure CPU fill that is thrown
    away the moment ``load_weights`` streams the real checkpoint values in
    (the 24 ``model.ngram_embeddings.*`` keys are always loaded before any
    forward).  This subclass keeps the exact ``nn.Embedding`` semantics
    (``weight`` parameter, ``padding_idx``, state-dict keys) but replaces
    the reset with a fast row-index fill: deterministic, finite, and each
    row differs, so even an unloaded table is a well-defined embedding.
    """

    def reset_parameters(self) -> None:
        with torch.no_grad():
            row = torch.arange(
                self.num_embeddings,
                dtype=torch.float32,
                device="cpu",
            ) % 1000
            self.weight.data.copy_(row.unsqueeze(1))


class NgramEmbedding(nn.Module):
    """N-gram enriched word embeddings (LongCat-Flash checkpoint layout).

    Mirrors the checkpoint reference ``NgramEmbedding``: the base word
    embedding, plus for every neighbor distance ``i`` in ``2..n`` and split
    ``j`` in ``0..k-1`` a polynomial rolling hash over the n-gram window,
    looked up in a per-slot embedding table and projected back to the hidden
    size, then normalized by ``1 + k * (n - 1)``. Uses only plain ``torch``
    ops so it runs unchanged on NPU.

    The per-slot embedder tables are host-resident (``_host_embedders``,
    ~63 GB total for the LongCat-Flash-Lite checkpoint): they are created
    on the CPU and a ``_apply`` override keeps them there under any module
    device move, while the forward gathers only the current batch's rows on
    demand.  The post projections (small, ~18 MB) stay on the module's
    device.

    Args:
        config: The checkpoint config; needs ``ngram_vocab_size_ratio``,
            ``emb_split_num``, ``emb_neighbor_num``, ``vocab_size``,
            ``hidden_size``, ``eos_token_id``, ``pad_token_id``.
        base_embeddings: The word-embedding module (in the runtime, the
            in-tree ``embed_tokens``).
    """

    def __init__(
        self,
        config,
        base_embeddings,
        *,
        tp_rank: int = 0,
        tp_size: int = 1,
        tp_group=None,
        max_num_seqs: int = 160,
    ):
        super().__init__()
        _ensure_ngram_config(config)
        self.config = config
        self.word_embeddings = base_embeddings

        self.m = config.ngram_vocab_size_ratio * config.vocab_size
        self.k = config.emb_split_num
        self.n = config.emb_neighbor_num

        # TP feature-column sharding (official LongCat OE layout): with
        # ``tp_size > 1`` every rank hosts only its feature slice of the
        # n-gram tables (~63 GB / tp_size per rank instead of a full copy
        # each) and the forward all-reduces the local partial contributions.
        self.tp_rank = tp_rank
        self.tp_size = tp_size
        self.tp_group = tp_group

        self._init_ngram_embeddings()
        self._vocab_mods_cache = None

        # Device staging for 方案C L1 (ACL Graph decode): the ~63 GB host
        # tables cannot be read from inside a captured graph (host<->device
        # memcpy is rejected during capture on 910B3), so the executor
        # preloads the current batch's rows H2D into this fixed staging
        # buffer *outside* the graph and the in-graph lookup reads only the
        # staging.  Staging allocation is decoupled from capture:
        # ``_ensure_staging`` allocates it once the module is placed on its
        # device (``_apply``) and ``preload_staging`` keeps it alive, so a
        # captured graph always records reads of an already-fixed staging
        # address.  ``use_staging`` selects the lookup path: False keeps the
        # original host-table lookup (eager baseline), True reads the
        # preloaded device staging (decode ACL Graph).  The eager baseline
        # must never silently fall back to the host lookup inside a captured
        # graph (that fallback is what 910B3 rejects as 107030), so the
        # staging branch raises when staging is missing.
        self._max_num_seqs = max_num_seqs
        self.use_staging: bool = False
        self.register_buffer("staging", None, persistent=False)

    def _ensure_staging(self) -> None:
        """Allocate the fixed device ``staging`` buffer (idempotent).

        The buffer is created on the module's device with the host tables'
        dtype (bf16 under the deployment dtype) so the H2D preload copy is
        bit-exact.  Allocation is decoupled from graph capture: it happens on
        module placement (``_apply``) and again in ``preload_staging``, so by
        the time any capture runs the staging address is fixed and the graph
        records reads of an already-allocated buffer.
        """
        if self.staging is not None:
            return
        device = self.post_projs[0].weight.device
        num_embedders = self.k * (self.n - 1)
        self.staging = torch.empty(
            (self._max_num_seqs, num_embedders, self._feature_width),
            dtype=self._host_embedders[0].dtype,
            device=device,
        ).zero_()

    def _init_ngram_embeddings(self) -> None:
        """Initialize the per-slot embedder/projection pairs.

        The embedder tables are host-resident (see the module docstring):
        they are created with an explicit ``device='cpu'`` so the runtime's
        ``torch.device(target)`` construction context cannot allocate the
        ~63 GB tables on the accelerator, and the ``_apply`` override below
        keeps them on the CPU for any later device move.  The
        ``nn.ModuleList`` registration is kept (not replaced by a plain
        list) because the checkpoint loader resolves the 24
        ``model.ngram_embeddings.embedders.*.weight`` /
        ``...post_projs.*.weight`` keys through ``named_parameters()``.

        Under TP feature-column sharding the tables hold only
        ``emb_dim // tp_size`` columns per rank and the projections map that
        slice onto the full hidden size; checkpoint loaders slice the full
        weights down to each rank's feature range.
        """
        num_embedders = self.k * (self.n - 1)
        emb_dim = self.config.hidden_size // num_embedders
        if self.tp_size > 1 and emb_dim % self.tp_size:
            raise ValueError(
                f"TP-sharded n-gram tables need emb_dim={emb_dim} divisible "
                f"by tp_size={self.tp_size}"
            )
        local_width = emb_dim // self.tp_size
        feature_begin = local_width * self.tp_rank
        feature_end = feature_begin + local_width
        self._feature_begin = feature_begin
        self._feature_width = local_width

        def _column_loader(param: torch.Tensor, loaded_weight: torch.Tensor) -> None:
            """Slice the checkpoint's full table/projection to this rank's
            feature columns (a row-major ``[rows, emb_dim]`` tensor)."""
            if loaded_weight.size(-1) != emb_dim:
                raise ValueError(
                    f"loaded n-gram weight has {loaded_weight.size(-1)} "
                    f"columns, expected {emb_dim}"
                )
            param.data.copy_(loaded_weight[:, feature_begin:feature_end])

        embedders = []
        post_projs = []
        for i in range(num_embedders):
            vocab_size = int(self.m + i * 2 + 1)
            emb = _HostEmbedding(
                vocab_size,
                local_width,
                padding_idx=self.config.pad_token_id,
                device="cpu",
            )
            if self.tp_size > 1:
                emb.weight.weight_loader = _column_loader
            proj = nn.Linear(local_width, self.config.hidden_size, bias=False)
            if self.tp_size > 1:
                proj.weight.weight_loader = _column_loader
            embedders.append(emb)
            post_projs.append(proj)

        self.embedders = nn.ModuleList(embedders)
        self.post_projs = nn.ModuleList(post_projs)

    @property
    def _host_embedders(self) -> list[torch.Tensor]:
        """CPU-resident n-gram table weights (one ``[vocab_i, emb_dim]`` per slot).

        The weights are the same tensors the ``nn.Embedding`` modules hold
        (shared storage, so ``load_weights`` writes are visible here); they
        are simply never moved off the CPU.
        """
        return [emb.weight for emb in self.embedders]

    def _apply(self, fn, recurse=True):
        """Module-wide device/dtype moves must never take the host tables off CPU.

        The n-gram embedder tables are deliberately host-resident (~63 GB);
        a ``.to('npu')`` (or any other ``_apply``) on the whole module would
        otherwise move them onto the accelerator and OOM the device.  The
        embedders are temporarily unregistered so ``super()._apply`` recurses
        past them, then restored (the registration is what keeps the 24-key
        ``named_parameters()`` loading contract intact).  Everything else —
        the base word embedding, the post projections — follows the module's
        device as usual.
        """
        embedders = self._modules.pop("embedders", None)
        try:
            result = super()._apply(fn, recurse=recurse)
        finally:
            if embedders is not None:
                self._modules["embedders"] = embedders
        # 方案C L1 修复: staging 分配与图捕获解耦 —— 模块就位（.to('npu')
        # 等）后立即在模块 device 上分配固定 staging，任何捕获之前就绪。
        # 幂等：已分配时是 no-op。host 表保持 float32 不受 fn 影响，staging
        # 以 host 表 dtype 分配，H2D 预载逐位一致。
        self._ensure_staging()
        return result

    def _window_from_context(
        self, context: torch.Tensor, committed: int
    ) -> torch.Tensor:
        """Build the per-token predecessor window from full context rows.

        ``context`` ``[.., committed + seq_len]`` holds each row's committed
        history prefix followed by its input ids; ``committed`` is the prefix
        width (0 when there is none). The returned window
        ``[.., seq_len, n - 1]`` holds raw predecessor ids (``-1`` where the
        predecessor lies before the context start), newest first: column
        ``d - 1`` is the token ``d`` positions back.
        """
        seq_len = context.size(-1) - committed
        window = torch.full(
            (*context.shape[:-1], seq_len, self.n - 1),
            -1,
            dtype=context.dtype,
            device=context.device,
        )
        for d in range(1, self.n):
            # Predecessor of position ``p`` is ``context[p + committed - d]``;
            # shift the row right by ``d`` (left edge ``-1``) and read the
            # input positions.
            shifted = torch.full_like(context, -1)
            if d < context.size(-1):
                shifted[..., d:] = context[..., :-d]
            window[..., d - 1] = shifted[..., committed : committed + seq_len]
        return window

    def _precompute_vocab_mods(self):
        """Polynomial rolling-hash moduli per (neighbor, split) pair, cached."""
        if self._vocab_mods_cache is not None:
            return self._vocab_mods_cache

        vocab_mods = {}
        vocab_size = self.config.vocab_size
        for i in range(2, self.n + 1):
            for j in range(self.k):
                index = (i - 2) * self.k + j
                emb_vocab_dim = int(self.m + index * 2 + 1)
                mods = []
                power_mod = 1
                for _ in range(i - 1):
                    power_mod = (power_mod * vocab_size) % emb_vocab_dim
                    mods.append(power_mod)
                vocab_mods[(i, j)] = mods

        self._vocab_mods_cache = vocab_mods
        return vocab_mods

    def _compute_new_ids(
        self,
        input_ids: torch.Tensor,
        predecessor_window: torch.Tensor,
        index: int,
    ) -> torch.Tensor:
        """Polynomial rolling-hash ids for embedder ``index`` (shared hash).

        The single source of the per-slot hash used by both the in-graph
        ``_ngram_local_partial`` and the graph-external ``preload_staging``:
        identical ``input_ids`` / ``predecessor_window`` inputs must produce
        bit-identical ``new_ids`` so the staging rows preloaded by hash
        align with the in-graph reads.  ``index`` enumerates the
        ``(neighbor, split)`` pairs row-major (``index = (i - 2) * k + j``).
        """
        i = 2 + index // self.k
        j = index % self.k
        emb_vocab_dim = int(self.m + index * 2 + 1)

        # EOS reset, vectorized over the window columns (newest first): the
        # predecessor at distance ``d`` contributes zero when any of the ``d``
        # tokens immediately before the current token is EOS (or is dead).
        eos = self.config.eos_token_id
        crossed = (predecessor_window == eos).cumsum(-1) > 0
        effective = predecessor_window.masked_fill(
            (predecessor_window < 0) | crossed, 0
        )

        mods = self._precompute_vocab_mods()[(i, j)]
        ngram_ids = input_ids.clone()
        for k in range(2, i + 1):
            ngram_ids = ngram_ids + effective[..., k - 2] * mods[k - 2]
        return ngram_ids % emb_vocab_dim

    def preload_staging(
        self,
        input_ids: torch.Tensor,
        predecessor_window: torch.Tensor,
    ) -> None:
        """Graph-external preload of this batch's table rows into ``staging``.

        Computed *outside* the captured decode graph (the executor calls it
        in ``settle()`` right after ``load_round_inputs`` has filled the
        packed ids and the per-token predecessor window).  For every
        embedder the polynomial hash ids are computed on the device (same
        ``_compute_new_ids`` the in-graph path uses, so the staging rows are
        aligned by construction), the corresponding host-table rows are
        gathered on the CPU and copied H2D into ``staging[t, index, :]``.

        Args:
            input_ids: Packed ``[T]`` (or ``[.., seq_len]``) token ids.
            predecessor_window: ``[T, n-1]`` (or broadcastable) raw
                predecessor ids, as ``_ngram_local_partial`` reads.
        """
        device = input_ids.device
        num_tokens = input_ids.numel()
        if num_tokens > self._max_num_seqs:
            raise ValueError(
                f"NgramEmbedding staging holds {self._max_num_seqs} rows; "
                f"the batch needs {num_tokens}"
            )
        num_embedders = self.k * (self.n - 1)
        # Allocate (idempotent) before any copy: a captured graph pins the
        # staging data pointer, so it must never be re-allocated after
        # capture. The dtype tracks the host tables (bf16 under the
        # deployment dtype), which keeps the H2D copy bit-exact for the
        # maxdiff=0 contract. Zero-fill so rows beyond the current batch are
        # finite (benign padding) — capture warmup reads the staging before
        # any live preload has filled it.
        self._ensure_staging()
        with torch.no_grad():
            for index in range(num_embedders):
                new_ids = self._compute_new_ids(
                    input_ids, predecessor_window, index
                )
                host_weight = self._host_embedders[index]
                rows = F.embedding(
                    new_ids.to("cpu"),
                    host_weight,
                    padding_idx=self.config.pad_token_id,
                ).reshape(-1, self._feature_width)
                self.staging[:num_tokens, index, :].copy_(rows.to(device))
        if device.type == "npu":
            # Make sure the preload is visible to the graph replay that
            # follows the settle() call on the same stream.
            torch.npu.synchronize()

    def forward_from_window(
        self,
        input_ids: torch.Tensor,
        predecessor_window: torch.Tensor,
        *,
        query_shard=None,
    ) -> torch.Tensor:
        """Stateless forward from a per-token predecessor window.

        Args:
            input_ids: ``[.., seq_len]`` current token ids.
            predecessor_window: ``[.., seq_len, n-1]`` raw predecessor token
                ids, column ``d-1`` holding the token ``d`` positions back
                (newest first), ``-1`` where no predecessor exists. The window
                already folds each request's committed history in (shared
                ngram-history machinery, ``prepare_ngram_inputs``).
            query_shard: Forward query-shard plan forwarded to the base word
                embedding (query context parallelism).

        Returns:
            ``[.., seq_len, hidden_size]`` enriched embeddings.
        """
        seq_len = input_ids.size(-1)
        if self.tp_size > 1 and query_shard is not None:
            raise NotImplementedError(
                "TP-sharded LongCat n-gram tables require the complete packed "
                "input batch (query context parallel must be disabled)"
            )

        # Base word embeddings (TP/query-shard aware through the embedding
        # module); the clone keeps the checkpoint's in-place-friendly layout.
        if query_shard is not None:
            x = self.word_embeddings(input_ids, query_shard=query_shard).clone()
        else:
            x = self.word_embeddings(input_ids).clone()

        local_partial = self._ngram_local_partial(input_ids, predecessor_window)

        # TP feature-column sharding: the local table slice contributes only
        # this rank's feature columns; all-reduce sums the full contribution.
        if self.tp_size > 1:
            assert self.tp_group is not None
            local_partial = all_reduce(local_partial, self.tp_group)
        x = x + local_partial
        x = x / (1 + self.k * (self.n - 1))
        return x

    def _ngram_local_partial(
        self,
        input_ids: torch.Tensor,
        predecessor_window: torch.Tensor,
    ) -> torch.Tensor:
        """Sum of the n-gram table contribution branches (pre all-reduce).

        Computes the polynomial hash ids from the per-token predecessor
        window (via the shared ``_compute_new_ids``) and accumulates the 12
        table lookups projected onto the hidden size.  Two lookup paths:

        * ``use_staging`` is False (eager / no decode graph): the original
          host-resident lookup — ids D2H, ``F.embedding`` on the CPU table,
          rows H2D (W18 scheme 2).  Preserved as the baseline path.
        * ``use_staging`` is True (decode ACL Graph, 方案C L1): the graph
          -external ``preload_staging`` has already H2D-copied this batch's
          rows into the device buffer; the in-graph path reads them directly
          (no host/device transfer inside the captured graph).  Staging must
          already be allocated at that point — the branch raises instead of
          silently falling back to the host lookup, because that silent
          fallback is exactly the host<->device memcpy 910B3 rejects under
          capture (107030).

        Under TP feature-column sharding this is the local partial; the
        caller all-reduces it.
        """
        local_partial = None
        for index in range(self.k * (self.n - 1)):
            new_ids = self._compute_new_ids(input_ids, predecessor_window, index)
            if not self.use_staging:
                # Host-resident lookup (W18 scheme 2, eager baseline).
                host_weight = self._host_embedders[index]
                new_ids_cpu = new_ids.to("cpu")
                x_ngram = F.embedding(
                    new_ids_cpu,
                    host_weight,
                    padding_idx=self.config.pad_token_id,
                ).to(input_ids.device)
            else:
                # 方案C L1: in-graph read-only lookup of the preloaded
                # device staging (token rows were aligned by the shared
                # hash at preload time).
                if self.staging is None:
                    raise RuntimeError(
                        "NgramEmbedding staging enabled (use_staging=True) but "
                        "the staging buffer is not allocated; the executor must "
                        "allocate it before any capture (107030 regression guard)"
                    )
                x_ngram = self.staging[: input_ids.numel(), index, :]
                if input_ids.dim() > 1:
                    x_ngram = x_ngram.reshape(*input_ids.shape, -1)
            x_proj = self.post_projs[index](x_ngram)
            local_partial = (
                x_proj if local_partial is None else local_partial + x_proj
            )
        return local_partial

    def forward(
        self,
        input_ids: torch.Tensor,
        ngram_context: torch.Tensor | None = None,
        *,
        query_shard=None,
    ) -> torch.Tensor:
        """Stateless forward: base embedding + n-gram contributions.

        Args:
            input_ids: ``[batch, seq_len]`` current token ids.
            ngram_context: ``[batch, context_len]`` committed predecessor
                tokens (the last ``n-1`` of each request's history); None when
                every predecessor is inside ``input_ids``.
            query_shard: Forward query-shard plan forwarded to the base word
                embedding (query context parallelism).

        Returns:
            ``[batch, seq_len, hidden_size]`` enriched embeddings.
        """
        if ngram_context is not None:
            committed = self.n - 1
            context = torch.cat(
                [ngram_context[..., -committed:], input_ids], dim=-1
            )
        else:
            committed = 0
            context = input_ids
        window = self._window_from_context(context, committed)
        return self.forward_from_window(
            input_ids, window, query_shard=query_shard
        )


class _RuntimeLongcatNgramModel(_RuntimeLongcatModel):
    """In-tree LongCat runtime model with the checkpoint's N-gram embeddings.

    The decoder stack, the MoE layers and the MLA attention all come from the
    in-tree ``_RuntimeLongcatModel``; this subclass adds the
    ``model.ngram_embeddings.*`` tables and injects the N-gram embedding in
    place of the plain ``embed_tokens`` lookup in forward.
    """

    def __init__(
        self,
        config,
        mapping,
        quant_config=None,
        prefix="",
    ) -> None:
        super().__init__(config, mapping, quant_config=quant_config, prefix=prefix)
        self.ngram_embeddings = NgramEmbedding(
            config,
            self.embed_tokens,
            tp_rank=mapping.attn.tp_rank,
            tp_size=mapping.attn.tp_size,
            tp_group=mapping.attn.tp_group,
            # 方案C L1 staging 行数：默认 160（NPU 部署 --max-num-seqs 默认），
            # 超限在 preload_staging 内报错，不静默。
            max_num_seqs=getattr(config, "max_num_seqs", 160),
        )

    def preload_ngram_staging(
        self,
        input_ids: torch.Tensor,
        predecessor_window: torch.Tensor,
    ) -> None:
        """Graph-external preload hook (方案C L1), called by the executor.

        The ``ModelExecutor.settle()`` runs this *outside* the decode ACL
        Graph, after ``load_round_inputs`` has filled the packed ids and the
        per-token predecessor window; the captured graph then reads only the
        preloaded device staging (never the host tables).
        """
        self.ngram_embeddings.preload_staging(input_ids, predecessor_window)

    def _history_pad(self, input_ids: torch.Tensor, view) -> torch.Tensor:
        """Per-row committed history, left-padded to width ``n - 1``.

        Returns a ``[bs, n - 1]`` tensor whose row ``r`` holds the last
        ``min(committed, n - 1)`` committed tokens of request ``r`` right-
        justified (missing predecessors are zero). A zero predecessor
        contributes zero to the polynomial hash, identical to the reference
        implementation's "no predecessor" behavior. Vectorized: one ragged
        gather, no per-row loop.
        """
        n_1 = self.ngram_embeddings.n - 1
        bs = view.req_pool_indices.shape[0]
        if bs == 0:
            return torch.zeros(
                (0, n_1), dtype=input_ids.dtype, device=input_ids.device
            )
        history = view.history_token_ids  # [pool, max_committed]
        committed = view.committed_lengths
        slots = view.req_pool_indices
        hl = committed[slots]  # [bs]
        take = hl.clamp(max=n_1)  # [bs]
        # Gather the last ``take`` committed tokens right-justified.
        idx = (
            torch.arange(n_1, device=input_ids.device, dtype=torch.int64)[None, :]
            - take[:, None]
            + hl[:, None]
        )
        valid = (idx >= 0) & (idx < hl[:, None])
        flat = slots.to(input_ids.device).unsqueeze(-1) * history.size(-1) + idx.clamp_min(
            0
        )
        gathered = history.reshape(-1)[flat]  # [bs, n-1]
        hist = gathered.masked_fill(~valid, 0).to(
            device=input_ids.device, dtype=input_ids.dtype
        )
        return hist

    def _ngram_embed_packed(self, input_ids: torch.Tensor, view, ctx) -> torch.Tensor:
        """Embed a 1-D packed serving batch (``[T]`` ids, per-request rows).

        Assembles the packed ids into a padded ``[bs, max_seq]`` batch with
        the per-request committed history, runs the N-gram embedding and
        flattens the rows back to ``[T, hidden_size]``.
        """
        bs = view.req_pool_indices.shape[0]
        starts = view.input_start_offsets  # [bs + 1]
        dev = input_ids.device
        dtype = input_ids.dtype
        lengths = [int(starts[r + 1]) - int(starts[r]) for r in range(bs)]
        max_seq = max(lengths) if lengths else 0
        rows = torch.zeros((bs, max_seq), dtype=dtype, device=dev)
        for r in range(bs):
            s = int(starts[r])
            rows[r, : lengths[r]] = input_ids[s : s + lengths[r]]

        hist = self._history_pad(rows, view)  # [bs, n - 1]
        out = self.ngram_embeddings(
            rows, ngram_context=hist, query_shard=ctx.query_shard
        )
        if not lengths:
            return torch.empty(
                (0, out.size(-1)), dtype=out.dtype, device=out.device
            )
        out_rows = [out[r, : lengths[r]] for r in range(bs)]
        return torch.cat(out_rows, dim=0)

    def _ngram_embed(self, input_ids: torch.Tensor, ctx, kwargs) -> torch.Tensor:
        """Compute the N-gram input embeddings for this forward."""
        view = kwargs.get("request_token_history")
        window = kwargs.get("ngram_previous_tokens")
        if input_ids.dim() == 2:
            # Direct ``[bs, seq]`` ids (verification / decode views): attach
            # the per-row committed history when the executor provides it.
            ngram_context = None
            if view is not None:
                ngram_context = self._history_pad(input_ids, view)
            return self.ngram_embeddings(
                input_ids, ngram_context=ngram_context, query_shard=ctx.query_shard
            )
        # 1-D packed serving layout.
        if window is not None:
            # The shared ngram-history machinery (``prepare_ngram_inputs``)
            # already resolved per-token predecessor windows for the packed
            # batch, committed history folded in.
            return self.ngram_embeddings.forward_from_window(
                input_ids, window, query_shard=ctx.query_shard
            )
        if view is not None:
            return self._ngram_embed_packed(input_ids, view, ctx)
        # No history view (defensive; the profile enables one in serving):
        # treat the packed ids as a single row.
        return self.ngram_embeddings(
            input_ids.unsqueeze(0), query_shard=ctx.query_shard
        ).squeeze(0)

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        ctx,
        input_embeds: torch.Tensor | None = None,
        **kwargs,
    ) -> tuple[torch.Tensor, list[torch.Tensor] | None]:
        if input_embeds is not None:
            hidden_states = input_embeds
        else:
            hidden_states = self._ngram_embed(input_ids, ctx, kwargs)

        residual = None
        aux_hidden_states = [] if self.layers_to_capture else None
        layer = None
        for layer_id, layer in enumerate(self.layers):
            capture = layer_id in self.layers_to_capture
            hidden_states, residual = layer(
                positions,
                hidden_states,
                ctx,
                residual,
                aux_hidden_states=aux_hidden_states if capture else None,
            )

        if not ctx.forward_mode.is_idle() and layer is not None:
            hidden_states, _ = layer.final_norm_comm.final_norm(
                hidden_states,
                residual,
                ctx,
                self.norm,
            )
        return hidden_states, aux_hidden_states


class LongcatFlashNgramForCausalLM(LongcatFlashForCausalLM):
    """LongCat-Flash causal LM with N-gram enhanced embeddings (plugin arch).

    Reuses the in-tree LongCat body (MLA attention, ScMoE decoder stack,
    weight-loading and MLA LoRA-scale fold) and swaps the embedding stage for
    the checkpoint's N-gram embedding module.
    """

    model_cls = _RuntimeLongcatNgramModel

    def prepare_model_kwargs(self, ctx, input_ids, kwargs) -> dict:
        """Pass the n-gram serving kwargs through the CausalLM wrapper.

        The base ``CausalLM`` wrapper forwards only ``input_embeds`` /
        ``inputs_embeds`` / ``pp_inbound``; without this override the n-gram
        embedding would never see its per-token predecessor window or
        request-token history and would fall back to a single-row in-input
        window.  Besides degrading the serving numerics, that fallback breaks
        the 方案C L1 staging alignment: the executor preloads rows by the
        packed window while the in-graph forward would hash a different
        (in-input-only) window.  Forwarding the n-gram kwargs keeps the
        in-graph path and the preload on the same ``(input_ids, window)``
        pair.
        """
        model_kwargs = super().prepare_model_kwargs(ctx, input_ids, kwargs)
        for key in (
            "ngram_previous_tokens",
            "request_token_history",
            "engram_hash_ids",
            "engram_token_mask",
        ):
            value = kwargs.get(key)
            if value is not None:
                model_kwargs[key] = value
        return model_kwargs

    def preload_ngram_staging(
        self,
        input_ids: torch.Tensor,
        predecessor_window: torch.Tensor,
    ) -> None:
        """Graph-external preload hook (方案C L1), exposed on the wrapper.

        ``ModelExecutor`` calls hooks on ``model_runner.model``, which is
        this CausalLM wrapper; the inner runtime model owns the n-gram
        embeddings, so delegate (the runtime model's own hook reaches
        ``NgramEmbedding.preload_staging``).
        """
        self.model.preload_ngram_staging(input_ids, predecessor_window)

    def set_ngram_staging(self, enabled: bool) -> None:
        """Toggle the device-staging lookup path for the decode ACL Graph."""
        embeddings = self.model.ngram_embeddings
        embeddings.use_staging = bool(enabled)
        if enabled:
            # Allocate the fixed staging address before any capture records
            # reads of it (idempotent).
            embeddings._ensure_staging()

    @property
    def ngram_staging_enabled(self) -> bool:
        """Whether the n-gram device-staging path is active."""
        return self.model.ngram_embeddings.use_staging

    def get_ngram_hash_parameters(self) -> None:
        """The n-gram embedding hashes with its own polynomial; no Engram params.

        Returning ``None`` keeps the executor's caller-owned ngram buffers in
        window-only mode (raw predecessor window, no Engram hash buffers).
        """
        return None

    @classmethod
    def model_profile(cls, hf_config):
        """Declare the model's family facts to ``ModelConfig``.

        The N-gram arch keeps the in-tree LongCat MLA geometry and the
        double-attention layer count, and reads each request's committed
        token history (``ForwardContext`` / ``request_token_history``) for
        its predecessor window.
        """
        from tokenspeed.runtime.configs.model_config import configure_mla_attention
        from tokenspeed.runtime.configs.model_profile import ModelProfile

        return ModelProfile(
            configure_attention=configure_mla_attention,
            cache_family="mla",
            linear_attention=None,
            default_attention_backend=None,
            default_prefix_granularity=None,
            request_token_history=True,
            is_multimodal=False,
            tokenizer_kwargs={},
            attention_instances_per_layer=2,
            numerics_envelopes=frozenset({"auto"}),
        )


_MODEL_REGISTERED = False


def register() -> None:
    """Entry-point contract: register the N-gram architecture (idempotent)."""
    global _MODEL_REGISTERED
    if _MODEL_REGISTERED:
        return
    _MODEL_REGISTERED = True
    register_model(
        LongcatFlashNgramForCausalLM,
        architectures=("LongcatFlashNgramForCausalLM",),
    )


# In-tree plugin contract: importing this module registers the architecture
# (the model registry's ``import_model_classes`` imports this package's
# modules; an installed distribution may additionally declare a
# ``tokenspeed.plugins`` entry point targeting this same ``register``).
register()
