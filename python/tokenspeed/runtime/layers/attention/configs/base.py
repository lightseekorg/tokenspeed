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
from typing import TypeVar

import torch

from tokenspeed.runtime.configs.model_config import ModelConfig
from tokenspeed.runtime.utils.server_args import ServerArgs

ComponentT = TypeVar("ComponentT", bound="AttnComponentSpec")


def resolve_speculative_num_tokens(
    server_args: ServerArgs, is_draft: bool = False
) -> int:
    """Return the query width seen by this attention backend.

    Target verification consumes the full candidate window. DSpark's draft
    model samples from its anchor row, so seven draft queries produce the seven
    proposals in an eight-token verify window.
    """
    width = int(server_args.speculative_num_draft_tokens)
    if is_draft and server_args.speculative_algorithm == "DSPARK":
        return width - 1
    return width


def resolve_dtype(kv_cache_dtype_str: str) -> torch.dtype:
    if kv_cache_dtype_str == "auto":
        return torch.bfloat16
    elif kv_cache_dtype_str == "bfloat16":
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

    num_attention_heads: int
    num_kv_heads: int
    head_dim: int
    attn_tp_size: int
    # Per-layer attention-type labels, forwarded to the KV pool for
    # cache_group_specs publication (empty -> single full-history group).
    layer_types: tuple[str, ...] = ()
    # Retention window; families narrow the type (per-layer tuple on MHA,
    # DeepSeek V4's int on MLA, None on MSA).
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
    # Scheduler prefix granularity (CLI --block-size): the identity axis.
    prefix_granularity: int
    # Tokens covered by one attention-kernel page. None picks the backend's
    # per-kernel default (flexible kernels use the prefix granularity;
    # fixed-page kernels use their own constant).
    kernel_page_size: int | None = None
    # Physical per-request KV extent: the model's logical context_len plus
    # ServerArgs.spec_context_pad (spec verify overshoot for a finished request
    # lingering one overlap step). Backends size page tables and clamp seq_lens
    # against this; user-facing limits stay on the logical context_len.
    context_len: int
    max_bs: int
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
    components: tuple[AttnComponentSpec, ...]

    def __post_init__(self):
        softmax_components = [
            c for c in self.components if isinstance(c, SoftmaxAttnConfig)
        ]
        if len(softmax_components) != 1:
            raise ValueError(
                "AttnConfig requires exactly one softmax-family component, got "
                f"{[type(c).__name__ for c in self.components] or 'none'}"
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
    speculative_num_draft_tokens: int | None = None,
) -> dict:
    """The AttnConfig fields shared by every family's generate().

    The softmax family passes in the facts its policies decide (cache dtype,
    block-decode mode, and — for families that bypass the DSpark width
    convention — the raw speculative width).
    """
    kwargs = dict(
        device=server_args.device,
        dtype=model_config.dtype,
        kv_cache_dtype=kv_cache_dtype,
        kv_cache_mxfp8=kv_cache_mxfp8,
        kv_cache_quant_method=server_args.kv_cache_quant_method,
        prefix_granularity=server_args.prefix_granularity,
        context_len=model_config.context_len + server_args.spec_context_pad,
        max_bs=server_args.max_num_seqs
        // (server_args.data_parallel_size or server_args.mapping.attn.dp_size),
        pd_disaggregation_enabled=server_args.disaggregation_mode != "null",
        is_draft=is_draft,
        draft_block_decode=draft_block_decode,
    )
    if server_args.speculative_algorithm is not None:
        kwargs.update(
            speculative_num_steps=server_args.speculative_num_steps,
            speculative_num_draft_tokens=(
                speculative_num_draft_tokens
                if speculative_num_draft_tokens is not None
                else resolve_speculative_num_tokens(server_args, is_draft)
            ),
        )
    return kwargs
