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

from dataclasses import dataclass

import torch

from tokenspeed.runtime.configs.model_config import ModelConfig
from tokenspeed.runtime.utils.server_args import ServerArgs


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
class BaseAttnConfig:
    device: str
    backend_name: str
    num_attention_heads: int
    num_kv_heads: int
    head_dim: int
    attn_tp_size: int
    dtype: torch.dtype
    kv_cache_dtype: torch.dtype
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
    max_graph_bs: int
    kv_cache_quant_method: str
    # MXFP8 KV: UE8M0 vec-32 scale sidecar in pool buffers; kv_cache_dtype alone can't express it
    kv_cache_mxfp8: bool = False
    speculative_num_steps: int = 0
    speculative_num_draft_tokens: int = 1
    is_draft: bool = False
    # DFLASH drafts a whole block in one decode forward (q_len = spec_num_tokens
    # per request) instead of Eagle/MTP's per-step single-token decode. Backends
    # use this to expand decode metadata to spec_num_tokens rows per request.
    draft_block_decode: bool = False
    # Decode context parallelism is nested inside attention TP. Non-MLA and
    # replicated draft configurations retain the identity defaults.
    dcp_size: int = 1
    dcp_rank: int = 0
    dcp_group: tuple[int, ...] = (0,)

    @classmethod
    def generate(
        cls, server_args: ServerArgs, model_config: ModelConfig, is_draft: bool = False
    ):
        raise NotImplementedError("Not Implemented!")

    def cache_cell_size(self) -> int:
        raise NotImplementedError("Not Implemented!")
