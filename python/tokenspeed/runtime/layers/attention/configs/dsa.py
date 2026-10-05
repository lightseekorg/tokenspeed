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
from typing import ClassVar

import torch
from tokenspeed_kernel.platform import current_platform

from tokenspeed.runtime.configs.model_config import ModelConfig
from tokenspeed.runtime.layers.attention.configs.base import AttnConfig
from tokenspeed.runtime.layers.attention.configs.mla import MLAConfig
from tokenspeed.runtime.layers.attention.kernel_page_sizes import (
    DSA_SPARSE_PAGE_SIZE,
)
from tokenspeed.runtime.utils.server_args import ServerArgs

_INDEX_K_FP8_GROUP_SIZE = 128
_INDEX_K_SCALE_BYTES = torch._utils._element_size(torch.float32)

# Storage of the indexer's key rows (``tokenspeed_kernel`` DSA README, "Index-K
# plane formats"): ``fp8_scaled`` packs FP8 keys with one fp32 scale per 128
# elements into uint8 rows and is what the in-tree scoring leaves read;
# ``bf16`` keeps the keys as the indexer produced them, no scale plane. The
# model config selects one (``index_k_format``); the pool writes exactly that
# layout and the top-k facades read it off the plane's dtype, so a plane is
# never converted on the way in or out.
INDEX_K_FORMATS = ("fp8_scaled", "bf16")
_INDEX_K_PLANE_DTYPES = {"fp8_scaled": torch.uint8, "bf16": torch.bfloat16}


def index_k_plane_dtype(index_k_format: str) -> torch.dtype:
    """The storage dtype of an index-K plane in ``index_k_format``."""
    if index_k_format not in _INDEX_K_PLANE_DTYPES:
        raise ValueError(
            f"index_k_format must be one of {list(INDEX_K_FORMATS)}, got "
            f"{index_k_format!r}"
        )
    return _INDEX_K_PLANE_DTYPES[index_k_format]


def dsa_index_k_row_bytes(index_head_dim: int) -> int:
    """Bytes of one ``fp8_scaled`` index-K row (FP8 keys plus fp32 scales)."""
    if index_head_dim <= 0 or index_head_dim % _INDEX_K_FP8_GROUP_SIZE != 0:
        raise ValueError(
            f"DSA index_head_dim must be a positive multiple of {_INDEX_K_FP8_GROUP_SIZE}, got {index_head_dim}"
        )
    return (
        index_head_dim
        + index_head_dim // _INDEX_K_FP8_GROUP_SIZE * _INDEX_K_SCALE_BYTES
    )


def index_k_row_bytes(index_head_dim: int, index_k_format: str) -> int:
    """Bytes of one index-K row in ``index_k_format``."""
    if index_k_format == "fp8_scaled":
        return dsa_index_k_row_bytes(index_head_dim)
    return index_head_dim * torch._utils._element_size(
        index_k_plane_dtype(index_k_format)
    )


def dsa_history_gather_workspace_rows(max_model_len: int, *, page_size: int) -> int:
    """Rows of the query-context-parallel history gather workspace.

    One request's whole history (``max_model_len`` rows), rounded up to whole
    kernel pages: the sharded extend arm hands ``dsa_prefill`` each group's
    gathered rows as a flat ``[slots, dim]`` cache, and the paged solutions
    view such a buffer as ``[slots / page_size, page_size, dim]``, so the view
    every solution takes must be a whole number of pages. The rows past a
    group's history are never selected.

    Args:
        max_model_len: Longest history a request can have.
        page_size: The DSA leaf's kernel page size.
    """
    if max_model_len <= 0:
        raise ValueError("history workspace needs a positive max_model_len")
    if page_size <= 0:
        raise ValueError("history workspace needs a positive kernel page size")
    return -(-int(max_model_len) // int(page_size)) * int(page_size)


def dsa_history_gather_page_size(config: AttnConfig) -> int:
    """The kernel page size the GPU DSA leaf runs at under ``config``: the
    explicit override, else the sparse kernels' fixed page
    (``DSABackend.resolve_kernel_page_size`` makes the same choice)."""
    if config.kernel_page_size is not None:
        return int(config.kernel_page_size)
    return DSA_SPARSE_PAGE_SIZE


def dsa_history_gather_workspace_bytes(
    config: AttnConfig, *, max_model_len: int
) -> int:
    """Bytes of the query-context-parallel history gather workspace.

    One whole history (``max_model_len`` rows, padded to kernel pages by
    :func:`dsa_history_gather_workspace_rows`) of latent rows in the KV cache
    dtype plus index-K rows packed in the plane's own format
    (:func:`index_k_row_bytes` of ``spec.index_k_format``: FP8 keys with
    their scales, or bf16 keys): the sharded extend arm gathers each request
    group's history into it, so a request's history may never exceed the
    model length. ``DSABackend.preallocate_history_gather_workspace``
    allocates exactly these bytes.
    """
    spec = config.component(DSAConfig)
    if spec is None:
        raise ValueError("the history gather workspace is a DSA quantity")
    rows = dsa_history_gather_workspace_rows(
        max_model_len, page_size=dsa_history_gather_page_size(config)
    )
    kv_bytes = (
        spec.kv_cache_dim * torch.tensor([], dtype=config.kv_cache_dtype).element_size()
    )
    return rows * (
        kv_bytes + index_k_row_bytes(spec.index_head_dim, spec.index_k_format)
    )


@dataclass(kw_only=True)
class DSAConfig(MLAConfig):
    is_dsa: ClassVar[bool] = True
    index_topk: int
    index_head_dim: int
    index_n_heads: int
    # Storage of the index-key plane, one of INDEX_K_FORMATS.
    index_k_format: str
    index_kpool: int | None = None

    def __post_init__(self) -> None:
        # None is a DSA configure-attention hook that named no plane.
        index_k_plane_dtype(self.index_k_format)

    @classmethod
    def _spec_kwargs(
        cls, server_args: ServerArgs, model_config: ModelConfig, is_draft: bool
    ) -> dict:
        return dict(
            **super()._spec_kwargs(server_args, model_config, is_draft),
            index_topk=model_config.index_topk,
            index_head_dim=model_config.index_head_dim,
            index_n_heads=model_config.index_n_heads,
            # Named by the model's configure-attention hook: the in-tree hook
            # keeps the FP8-with-scale rows every in-tree scoring leaf reads; a
            # plugin that scores the checkpoint's bf16 keys names "bf16".
            index_k_format=model_config.index_k_format,
            index_kpool=getattr(model_config, "index_kpool", None),
        )

    @classmethod
    def generate(
        cls,
        server_args: ServerArgs,
        model_config: ModelConfig,
        is_draft: bool = False,
    ) -> AttnConfig:
        config = super().generate(server_args, model_config, is_draft)
        if config.kv_cache_dtype in (torch.float8_e4m3fn, torch.float8_e5m2):
            platform = current_platform()
            if not (platform.is_blackwell_plus or platform.is_cdna4_plus):
                raise ValueError(
                    "GLM DSA FP8 KV cache currently requires NVIDIA Blackwell "
                    "or AMD CDNA4 sparse attention support; use --kv-cache-dtype "
                    "auto or bfloat16 on this platform, got "
                    f"{server_args.kv_cache_dtype}."
                )
        return config

    def cache_cell_size(self, config: AttnConfig) -> int:
        index_k_cell_size = index_k_row_bytes(self.index_head_dim, self.index_k_format)
        return super().cache_cell_size(config) + index_k_cell_size
