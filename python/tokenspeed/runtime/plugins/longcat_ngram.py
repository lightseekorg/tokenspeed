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

    def __init__(self, config, base_embeddings):
        super().__init__()
        _ensure_ngram_config(config)
        self.config = config
        self.word_embeddings = base_embeddings

        self.m = config.ngram_vocab_size_ratio * config.vocab_size
        self.k = config.emb_split_num
        self.n = config.emb_neighbor_num

        self._init_ngram_embeddings()
        self._vocab_mods_cache = None

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
        """
        num_embedders = self.k * (self.n - 1)
        emb_dim = self.config.hidden_size // num_embedders

        embedders = []
        post_projs = []
        for i in range(num_embedders):
            vocab_size = int(self.m + i * 2 + 1)
            emb = _HostEmbedding(
                vocab_size,
                emb_dim,
                padding_idx=self.config.pad_token_id,
                device="cpu",
            )
            proj = nn.Linear(emb_dim, self.config.hidden_size, bias=False)
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
            return super()._apply(fn, recurse=recurse)
        finally:
            if embedders is not None:
                self._modules["embedders"] = embedders

    def _shift_right_ignore_eos(self, tensor: torch.Tensor, n: int) -> torch.Tensor:
        """Shift right by ``n``, resetting at EOS boundaries (vectorized).

        Equivalent to the checkpoint reference loop: within every segment
        bounded by EOS tokens, position ``j`` copies ``tensor[j - n]`` when
        ``j - n`` lies in the same segment, and stays zero otherwise.
        """
        eos = (tensor == self.config.eos_token_id).to(tensor.dtype)
        # Segment id of position j = number of EOS strictly before j.
        seg = eos.cumsum(dim=-1) - eos
        result = torch.zeros_like(tensor)
        seq_len = tensor.size(-1)
        if n < seq_len:
            result[:, n:] = tensor[:, :-n]
            same = seg[:, n:] == seg[:, :-n]
            result[:, n:] = torch.where(
                same, result[:, n:], torch.zeros_like(result[:, n:])
            )
        return result

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

    def _get_ngram_ids(
        self,
        input_ids: torch.Tensor,
        shifted_ids: dict[int, torch.Tensor],
        vocab_mods: list[int],
        ngram: int,
    ) -> torch.Tensor:
        """Compute n-gram hash IDs using the polynomial rolling hash."""
        ngram_ids = input_ids.clone()
        for k in range(2, ngram + 1):
            ngram_ids = ngram_ids + shifted_ids[k] * vocab_mods[k - 2]
        return ngram_ids

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
        seq_len = input_ids.size(-1)

        if ngram_context is not None:
            context = torch.cat(
                [ngram_context[..., -(self.n - 1) :], input_ids], dim=-1
            )
        else:
            context = input_ids

        # Base word embeddings (TP/query-shard aware through the embedding
        # module); the clone keeps the checkpoint's in-place-friendly layout.
        if query_shard is not None:
            x = self.word_embeddings(input_ids, query_shard=query_shard).clone()
        else:
            x = self.word_embeddings(input_ids).clone()

        vocab_mods = self._precompute_vocab_mods()

        shifted_ids = {}
        for i in range(2, self.n + 1):
            shifted_ids[i] = self._shift_right_ignore_eos(context, i - 1)

        for i in range(2, self.n + 1):
            for j in range(self.k):
                index = (i - 2) * self.k + j
                emb_vocab_dim = int(self.m + index * 2 + 1)

                ngram_ids = self._get_ngram_ids(
                    context, shifted_ids, vocab_mods[(i, j)], ngram=i
                )
                new_ids = (ngram_ids % emb_vocab_dim)[..., -seq_len:]

                # Host-resident lookup (W18 scheme 2): the ids were hashed on
                # the NPU; move only this batch's rows to the CPU, gather the
                # embedder rows there, and bring the small result back to the
                # device for the (NPU-resident) post projection.  Numerically
                # identical to an on-device ``embedder(new_ids)`` — same ids,
                # same weights — and leaves the ~63 GB tables off the device.
                host_weight = self._host_embedders[index]
                new_ids_cpu = new_ids.to("cpu")
                x_ngram = F.embedding(
                    new_ids_cpu,
                    host_weight,
                    padding_idx=self.config.pad_token_id,
                ).to(x.device)
                x_proj = self.post_projs[index](x_ngram)
                x = x + x_proj.to(x.device)

        x = x / (1 + self.k * (self.n - 1))
        return x


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
        self.ngram_embeddings = NgramEmbedding(config, self.embed_tokens)

    def _history_pad(self, input_ids: torch.Tensor, view) -> torch.Tensor:
        """Per-row committed history, left-padded to width ``n - 1``.

        Returns a ``[bs, n - 1]`` tensor whose row ``r`` holds the last
        ``min(committed, n - 1)`` committed tokens of request ``r`` right-
        justified (missing predecessors are zero). A zero predecessor
        contributes zero to the polynomial hash, identical to the reference
        implementation's "no predecessor" behavior.
        """
        n_1 = self.ngram_embeddings.n - 1
        bs = view.req_pool_indices.shape[0]
        hist = torch.zeros(
            (bs, n_1), dtype=input_ids.dtype, device=input_ids.device
        )
        history = view.history_token_ids
        committed = view.committed_lengths
        for r in range(bs):
            slot = int(view.req_pool_indices[r])
            hl = int(committed[slot])
            take = min(hl, n_1)
            if take > 0:
                hist[r, n_1 - take :] = history[slot, hl - take : hl].to(
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
