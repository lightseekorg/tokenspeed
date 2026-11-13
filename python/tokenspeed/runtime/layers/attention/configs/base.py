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

"""Composite attention configuration.

``AttnConfig`` carries the model-wide serving facts (device, cache precision,
paging, capacity, speculative shape) and a tuple of attention components.
Each component is an ``AttnComponentSpec`` subclass owning one mechanism's
kernel choice, head geometry, per-layer vectors, and parallel width;
``SoftmaxAttnConfig`` is the softmax-family base (MHA/MLA/DSA/MSA), the
counterpart of ``LinearAttnConfig``'s recurrent state.

The softmax component is unique per model (validated at construction) and
addressed by type: ``config.component(SoftmaxAttnConfig)``. Backends receive
their component explicitly: ``Backend(config, spec)``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import ClassVar, TypeVar

import torch

from tokenspeed.runtime.configs.model_config import ModelConfig
from tokenspeed.runtime.layers.attention.kv_cache.recipes.spec import FULL_ATTENTION
from tokenspeed.runtime.utils.server_args import ServerArgs
from tokenspeed.runtime.utils.spec_block_geometry import resolve_draft_query_width

ComponentT = TypeVar("ComponentT", bound="AttnComponentSpec")


def resolve_speculative_num_tokens(
    server_args: ServerArgs, is_draft: bool, *, sample_from_anchor: bool
) -> int:
    """Return the query width seen by this attention backend.

    Target verification consumes the full candidate window. Draft queries
    follow the checkpoint's normalized anchor-sampling policy.
    """
    width = int(server_args.speculative_num_draft_tokens)
    if is_block_drafter(server_args.speculative_algorithm, is_draft):
        return resolve_draft_query_width(width, sample_from_anchor)
    return width


def is_block_drafter(speculative_algorithm: str | None, is_draft: bool) -> bool:
    """A draft that proposes a whole block in one forward and writes its KV at
    the target's cache locations (DFLASH / DSPARK)."""
    return bool(is_draft and speculative_algorithm in ("DFLASH", "DSPARK"))


def block_draft_is_causal(hf_config) -> bool:
    """Whether a block draft's queries attend causally within their block.

    Only an explicit declaration makes a block causal: ``is_causal``, else
    ``dflash_config.causal`` (Kimi-K3.1's DSpark draft). A draft declaring
    neither (Kimi-K3 DSpark) attends the whole block; ``sliding_attention``
    layers only bound how far back a query reads.
    """
    is_causal = getattr(hf_config, "is_causal", None)
    if is_causal is not None:
        return bool(is_causal)
    dflash_config = getattr(hf_config, "dflash_config", None) or {}
    return bool(dflash_config.get("causal", False))


def resolve_cache_layer_types(
    hf_config,
    *,
    num_layers: int,
    is_draft: bool,
    draft_block_decode: bool,
) -> tuple[str, ...]:
    """Per-layer cache-group labels of one model side, read once for all
    softmax families.

    ``cache_layer_types`` wins over ``layer_types``: it can carry labels
    outside transformers' ``ALLOWED_LAYER_TYPES`` (Inkling's sliding
    sub-groups). Target-stack labels that do not fit a draft's depth are
    dropped so the recipe resolves the draft to full history. A block drafter
    writes at the target's cache locations, so its storage IS the target's
    full-history group whatever compute mask its layers apply: the label
    vector says so explicitly instead of losing the labels.
    """
    layer_types = tuple(
        getattr(hf_config, "cache_layer_types", None)
        or getattr(hf_config, "layer_types", None)
        or ()
    )
    if draft_block_decode:
        return (FULL_ATTENTION,) * num_layers
    if is_draft and layer_types and len(layer_types) != num_layers:
        return ()
    return layer_types


def resolve_dtype(kv_cache_dtype_str: str) -> torch.dtype:
    if kv_cache_dtype_str == "auto" or kv_cache_dtype_str == "bfloat16":
        return torch.bfloat16
    elif kv_cache_dtype_str in ("fp8", "fp8_e4m3"):
        return torch.float8_e4m3fn
    elif kv_cache_dtype_str == "mxfp8":
        # fp8-e4m3 data; the UE8M0 scale sidecar lives in the pool's scale buffers (kv_cache_mxfp8)
        return torch.float8_e4m3fn
    else:
        raise ValueError(f"Unsupported kv_cache_dtype: {kv_cache_dtype_str!r}")


@dataclass(kw_only=True)
class AttnComponentSpec:
    """One attention mechanism's boot-constant facts.

    Subclasses are the identity; a mixed full+sliding softmax model stays ONE
    component carrying its per-layer vectors.
    """

    # The compute backend requested for this component; None lets the backend
    # registry pick the component's default.
    backend_name: str | None = None


@dataclass(kw_only=True)
class SoftmaxAttnConfig(AttnComponentSpec):
    """Base of the softmax-attention families (MHA / MLA / DSA / MSA)."""

    is_dsa: ClassVar[bool] = False
    # Whether this family stores a latent KV page (MLA and its DSA variant) as
    # opposed to K/V head rows (MHA, MSA). A draft of the latter kind plans an
    # attention family of its own beside an MLA target.
    is_latent_kv: ClassVar[bool] = False

    num_attention_heads: int
    num_kv_heads: int
    head_dim: int
    attn_tp_size: int
    # Per-layer cache-group labels (the storage vocabulary of
    # recipes/spec.py, NOT the checkpoint's compute labels), consumed only by
    # the cache recipes; empty -> single full-history group. A layer's
    # compute mask lives on its PagedAttention (sliding_window_size).
    cache_layer_types: tuple[str, ...] = ()
    # Sliding window for per-layer cache recipes. Recipes that declare whole
    # groups, such as DeepSeek V4.1, put their windows on those groups.
    sliding_window_tokens: int | tuple[int | None, ...] | None = None

    @classmethod
    def generate(
        cls, server_args: ServerArgs, model_config: ModelConfig, is_draft: bool = False
    ) -> AttnConfig:
        raise NotImplementedError("Not Implemented!")

    def cache_cell_size(self, config: AttnConfig) -> int:
        """Per-token cache bytes of this component's dominant field.

        A per-TOKEN concept, hence softmax-family only: linear-attention
        state is fixed-size per request, sized via its shape properties.
        Takes the ``AttnConfig`` for the model-wide facts (cache dtype,
        quant method) that deliberately do not live on the spec.
        """
        raise NotImplementedError("Not Implemented!")


@dataclass(kw_only=True)
class AttnConfig:
    """The model's attention configuration: model-wide facts + components.

    Exactly one component is the softmax family (validated below); extra
    components (linear attention) ride alongside as peers. Built exclusively
    by ``registry._create_attn_config`` — one construction seam.
    """

    device: str
    dtype: torch.dtype
    kv_cache_dtype: torch.dtype
    kv_cache_quant_method: str
    # MXFP8 KV: UE8M0 vec-32 scale sidecar in pool buffers; kv_cache_dtype alone can't express it
    kv_cache_mxfp8: bool = False
    # Tokens covered by one attention-kernel page. None picks the backend's
    # per-kernel default (flexible kernels use their group's block
    # granularity; fixed-page kernels use their own constant). For the
    # ordinary families the registry pins a fixed-page leaf's constant here
    # before planning, so the recipe sizes its block span to hold whole pages.
    kernel_page_size: int | None = None
    # Physical per-request KV extent: the model's logical context_len plus
    # ServerArgs.spec_context_pad (spec verify overshoot for a finished request
    # lingering one overlap step). Backends size page tables and clamp seq_lens
    # against this; user-facing limits stay on the logical context_len.
    context_len: int
    max_bs: int
    # Real resident requests. Null row 0 and graph padding are additional rows.
    max_resident_requests: int
    # True iff server_args.disaggregation_mode != "null"; cache recipes use
    # it to stamp transfer policies onto the cache group specs.
    pd_disaggregation_enabled: bool = False
    speculative_num_steps: int = 0
    speculative_num_draft_tokens: int = 1
    is_draft: bool = False
    # DFLASH drafts a whole block in one decode forward (q_len = spec_num_tokens
    # per request) instead of Eagle/MTP's per-step single-token decode. Backends
    # use this to expand decode metadata to spec_num_tokens rows per request.
    draft_block_decode: bool = False
    # Causal within a block draft; otherwise every query sees the whole block.
    draft_block_causal: bool = False
    # KV parallelism (page-cyclic KV page sharding over a consecutive subgroup
    # of attention TP): one topology for target and continuation/MTP views;
    # KVP does not add ranks. Decode attends the owned pages and merges the
    # partials by LSE (DCP).
    kvp_size: int = 1
    kvp_rank: int = 0
    kvp_group: tuple[int, ...] = (0,)
    # Query context parallelism: each rank computes early and late segments
    # of every request; the group is the attention TP group.
    qcp_size: int = 1
    qcp_rank: int = 0
    qcp_group: tuple[int, ...] = (0,)
    components: tuple[AttnComponentSpec, ...]

    def __post_init__(self):
        from tokenspeed.runtime.execution.request_slots import RequestSlotLayout

        if type(self.max_bs) is not int or self.max_bs <= 0:
            raise ValueError("max_bs must be a positive integer")
        RequestSlotLayout(self.max_resident_requests)
        if self.max_resident_requests < self.max_bs:
            raise ValueError("max_resident_requests must be at least max_bs")
        softmax_components = [
            c for c in self.components if isinstance(c, SoftmaxAttnConfig)
        ]
        if len(softmax_components) != 1:
            raise ValueError(
                "AttnConfig requires exactly one softmax-family component, got "
                f"{[type(c).__name__ for c in self.components] or 'none'}"
            )
        if self.kv_cache_dtype == torch.int8:
            raise ValueError("INT8 KV cache is not supported by the runtime")
        if self.qcp_size > 1:
            softmax = softmax_components[0]
            # DSA gathers paged history; V4.1 publishes its packed KV from
            # gathered projections and keeps full logical metadata.
            is_v41 = softmax.backend_name == "deepseek_v41"
            if not (
                is_v41 or (softmax.is_dsa and softmax.backend_name in (None, "dsa"))
            ):
                raise ValueError(
                    "query context parallelism requires GPU DSA or V4.1 attention; "
                    f"got {softmax.backend_name!r}"
                )
            if not is_v41 and self.qcp_size != self.kvp_size:
                raise ValueError(
                    "DSA query context parallelism requires KVP equal to QCP"
                )
            if torch.device(self.device).type != "cuda":
                raise ValueError("GPU query context parallelism requires CUDA")
            # The sharded KV write gathers the rotated latent and stores it
            # with latent_store, which writes native (bf16) rows only.
            if not is_v41 and (
                self.kv_cache_dtype is not torch.bfloat16
                or self.kv_cache_mxfp8
                or self.kv_cache_quant_method != "none"
            ):
                raise ValueError(
                    "query context parallelism requires a bf16 KV cache; got "
                    f"{self.kv_cache_dtype} (mxfp8={self.kv_cache_mxfp8}, quant "
                    f"method {self.kv_cache_quant_method!r})"
                )
        if self.kvp_size > 1:
            softmax = softmax_components[0]
            if self.is_draft and not softmax.is_latent_kv:
                # A block drafter of its own attention family (DFlash: a GQA
                # stack storing K/V head rows) shares the target's full-history
                # group (check_block_drafter_storage), so page-sharding that
                # group puts head-partitioned K/V inside a sharded group, which
                # PD refuses to transfer (ts-pd layout/route.rs). Refused here,
                # at startup, not per request once traffic arrives.
                raise ValueError(
                    "--kv-parallel-size > 1 page-shards the cache group a "
                    "block drafter shares with the target, but this draft "
                    "stores K/V head rows, which PD cannot transfer from a "
                    "page-sharded group; launch with --kv-parallel-size 1 "
                    "(and --query-context-parallel-size 1), or use a draft "
                    "that plans the target's own attention"
                )
            if softmax.backend_name == "flashmla":
                if torch.device(self.device).type != "cuda":
                    raise ValueError("FlashMLA DCP decode requires CUDA")
            elif softmax.backend_name == "tokenspeed_mla":
                pass
            elif softmax.is_dsa and softmax.backend_name in (None, "dsa"):
                device_type = torch.device(self.device).type
                if device_type not in ("cuda", "npu"):
                    raise ValueError("DSA DCP decode requires CUDA or Ascend NPU")
                if device_type == "cuda" and self.draft_block_decode:
                    raise ValueError(
                        "GPU DSA DCP decode does not yet support non-causal "
                        "block drafts"
                    )
            elif softmax.backend_name == "hybrid_linear_attn":
                # The registry resolves the user's full-attention leaf after
                # composing the hybrid components, and validates it before
                # cache allocation. The composite name is not a capability.
                pass
            elif softmax.backend_name == "deepseek_v41":
                if self.is_draft:
                    raise ValueError("V4.1 KVP requires the target attention backend")
            else:
                raise ValueError(
                    "--kv-parallel-size > 1 (DCP decode) currently requires "
                    "DSA, FlashMLA or CuTe MLA attention"
                )

    def component(self, cls: type[ComponentT]) -> ComponentT | None:
        """The first component that is a ``cls``, or None.

        The one generic accessor — ``config.component(LinearAttnConfig)``;
        new component classes need no change here.
        """
        for component in self.components:
            if isinstance(component, cls):
                return component
        return None

    def cache_cell_size(self) -> int:
        return self.component(SoftmaxAttnConfig).cache_cell_size(self)


def model_wide_kwargs(
    server_args: ServerArgs,
    model_config: ModelConfig,
    is_draft: bool,
    *,
    kv_cache_dtype: torch.dtype,
    kv_cache_mxfp8: bool = False,
    draft_block_decode: bool = False,
    draft_block_causal: bool = False,
    speculative_num_draft_tokens: int | None = None,
    shards_query_context: bool = False,
) -> dict:
    """The AttnConfig fields shared by every family's generate().

    The softmax family passes in the facts its policies decide (cache dtype,
    block-decode mode, and — for families that bypass the DSpark width
    convention — the raw speculative width).

    ``shards_query_context`` says whether this family can compute a query
    context parallel shard (GPU DSA through the MLA config, and V4.1). It only
    affects a draft config; ``__post_init__`` refuses a target that cannot shard.
    """
    max_bs = server_args.max_num_seqs // (
        server_args.data_parallel_size or server_args.mapping.attn.dp_size
    )
    kwargs = dict(
        device=server_args.device,
        dtype=model_config.dtype,
        kv_cache_dtype=kv_cache_dtype,
        kv_cache_mxfp8=kv_cache_mxfp8,
        kv_cache_quant_method=server_args.kv_cache_quant_method,
        context_len=model_config.context_len + server_args.spec_context_pad,
        max_bs=max_bs,
        max_resident_requests=max_bs + server_args.disaggregation_decode_extra_slots,
        pd_disaggregation_enabled=server_args.disaggregation_mode != "null",
        is_draft=is_draft,
        draft_block_decode=draft_block_decode,
        draft_block_causal=draft_block_causal,
    )
    attn_mapping = server_args.mapping.attn
    # QCP shards the target's extend rows. A draft takes that shard only when it
    # plans the target's own sharded attention (an MTP depth). A draft with its
    # own ordinary attention is handed the rows the target computed and runs
    # over them whole (the DFlash drafter builds its logits metadata with
    # ``query_shard=None``), so it is configured unsharded.
    draft_unsharded = is_draft and not shards_query_context
    kwargs.update(
        kvp_size=attn_mapping.kvp_size,
        kvp_rank=attn_mapping.kvp_rank,
        kvp_group=attn_mapping.kvp_group,
        qcp_size=1 if draft_unsharded else attn_mapping.qcp_size,
        qcp_rank=0 if draft_unsharded else attn_mapping.qcp_rank,
        qcp_group=(0,) if draft_unsharded else attn_mapping.qcp_group,
    )
    if server_args.speculative_algorithm is not None:
        kwargs.update(
            speculative_num_steps=server_args.speculative_num_steps,
            speculative_num_draft_tokens=(
                speculative_num_draft_tokens
                if speculative_num_draft_tokens is not None
                else resolve_speculative_num_tokens(
                    server_args,
                    is_draft,
                    sample_from_anchor=(
                        model_config.draft_sample_from_anchor
                        if is_block_drafter(server_args.speculative_algorithm, is_draft)
                        else False
                    ),
                )
            ),
        )
    return kwargs
