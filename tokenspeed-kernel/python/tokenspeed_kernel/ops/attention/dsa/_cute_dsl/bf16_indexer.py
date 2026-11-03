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

"""BF16 MQA scoring entry and shared paged/workspace score semantics.

The normal and swapAB implementations live in separate sibling modules.
Kernel geometry is static; batch sizes, history lengths and splits are runtime.
"""

from typing import Final, Literal

import cutlass
import torch
from cutlass import Float32, cute

# Host-side scheduling constants. These are not per-request JIT parameters.
_PAGE_SIZE: Final[int] = 64
_PAIR_QUERY_GROUP: Final[int] = 2
_QUAD_QUERY_GROUP: Final[int] = 4
_MEDIUM_HISTORY: Final[int] = 16_384
_LONG_HISTORY: Final[int] = 65_536
_PAIR_QUERY_ROWS: Final[int] = 32
_QUAD_QUERY_ROWS: Final[int] = 64
_LARGE_QUERY_ROWS: Final[int] = 128
_GROUPED_SPLITS: Final[int] = 512
_SHORT_HISTORY_MIN_SPLITS: Final[int] = 32
_MEDIUM_HISTORY_MIN_SPLITS: Final[int] = 64
_GROUPED_CTA_TARGET: Final[int] = 8_192
_SMALL_DECODE_SCORE_COUNT: Final[int] = 196_608
_DECODE_CTA_TARGET: Final[int] = 512
_LONG_DECODE_SPLITS: Final[int] = 1_024
_DECODE_PAGE_SPLIT_DIVISOR: Final[int] = 2
_LONG_UNSHARED_VERIFY_SPLITS: Final[int] = 256
_VERIFY_MIN_SPLITS: Final[int] = 16

_HEAD_GRANULARITY: Final[int] = 8
_MAX_WGMMA_HEADS: Final[int] = 256
_HEAD_DIM_ALIGNMENT: Final[int] = 64
_MAX_HEAD_DIM: Final[int] = 256
# Bound Q reuse to the existing shared-memory/register budget.
_MAX_SHARED_HEADS: Final[int] = 64


class BF16IndexScoreBase:
    ROWS = _PAGE_SIZE
    PAGE = _PAGE_SIZE

    def __init__(self, heads, head_dim, stages, initial, recent, paged, query_group):
        self.query_heads = heads
        self.head_dim = head_dim
        self.query_group = query_group
        self.heads = heads * query_group
        self.stages = stages
        self.initial = initial
        self.recent = recent
        self.paged = paged

    @cute.jit
    def store(
        self,
        out,
        token,
        column,
        logical,
        position,
        visible,
        valid_rows,
        value,
        source,
        key_rows,
        valid,
    ):
        result = Float32(float("-inf"))
        in_bounds = (
            position >= 0
            and column < valid_rows
            and logical < visible
            and source < key_rows
            and source >= 0
        )
        if cutlass.const_expr(not self.paged):
            if in_bounds:
                in_bounds = valid[source] != 0
        if in_bounds:
            if value > Float32(float("-inf")) and value < Float32(float("inf")):
                result = value
            if logical < self.initial or logical >= visible - self.recent:
                result = Float32(float("inf"))
        if column < out.shape[1]:
            out[token, column] = result


def score_config(
    rows: int, width: int, query_group: int, heads: int
) -> tuple[int, int]:
    """Return a bounded query-sharing bucket and a runtime key split count.

    ``rows`` and ``width`` vary per batch; they must not become Constexpr
    arguments. Only the returned query-sharing bucket specializes the kernel.
    """
    group = 1
    if rows % query_group == 0 and query_group % _PAIR_QUERY_GROUP == 0:
        if (
            width >= _LONG_HISTORY
            or (width >= _MEDIUM_HISTORY and rows >= _PAIR_QUERY_ROWS)
            or rows >= _LARGE_QUERY_ROWS
        ):
            group = _PAIR_QUERY_GROUP
        if (
            query_group % _QUAD_QUERY_GROUP == 0
            and width >= _LONG_HISTORY
            and rows >= _QUAD_QUERY_ROWS
        ):
            group = _QUAD_QUERY_GROUP
    # Share whole queries only; a query's head reduction must stay in one CTA.
    while group > 1 and heads * group > _MAX_SHARED_HEADS:
        group //= _PAIR_QUERY_GROUP
    pages = (width + _PAGE_SIZE - 1) // _PAGE_SIZE
    if group == _QUAD_QUERY_GROUP:
        split = _GROUPED_SPLITS
    elif group == _PAIR_QUERY_GROUP:
        if width >= _LONG_HISTORY:
            split = _GROUPED_SPLITS
        else:
            min_split = (
                _SHORT_HISTORY_MIN_SPLITS
                if width < _MEDIUM_HISTORY
                else _MEDIUM_HISTORY_MIN_SPLITS
            )
            split = max(min_split, _GROUPED_CTA_TARGET // rows)
    elif query_group == 1:
        if rows * width <= _SMALL_DECODE_SCORE_COUNT:
            split = max(1, _DECODE_CTA_TARGET // rows)
        else:
            split = (
                _LONG_DECODE_SPLITS
                if width >= _LONG_HISTORY
                else max(
                    _MEDIUM_HISTORY_MIN_SPLITS, pages // _DECODE_PAGE_SPLIT_DIVISOR
                )
            )
    else:
        split = (
            _LONG_UNSHARED_VERIFY_SPLITS
            if width >= _LONG_HISTORY
            else max(_VERIFY_MIN_SPLITS, _DECODE_CTA_TARGET // rows)
        )
    return group, min(pages, split)


_NORMAL_TARGET_M: Final[int] = 64
_NORMAL_QUERY_GROUPS: Final[tuple[int, ...]] = (8, 4, 2, 1)
_NORMAL_CTA_TARGET: Final[int] = 4096
_PIPELINE_STAGES: Final[int] = 1
_SMALL_QUERY_M: Final[int] = 64


def select_score_config(
    rows: int,
    width: int,
    query_group: int,
    heads: int,
    *,
    paged: bool,
    strategy: Literal["auto", "normal", "swap_ab"],
) -> tuple[str, int, int]:
    """Select orientation and bounded query sharing; splits stay runtime.

    Logical M counts heads from queries of one request, never unrelated
    requests. Explicit strategies let callers compare both implementations.
    """
    if strategy not in ("auto", "normal", "swap_ab"):
        raise ValueError("strategy must be auto, normal or swap_ab")
    normal_group = 1
    if rows % query_group == 0:
        for candidate in _NORMAL_QUERY_GROUPS:
            if query_group % candidate == 0 and heads * candidate <= _NORMAL_TARGET_M:
                normal_group = candidate
                break
    if strategy == "auto":
        strategy = (
            "normal"
            if not paged or normal_group * heads >= _SMALL_QUERY_M
            else "swap_ab"
        )
    if strategy == "swap_ab":
        group, split = score_config(rows, width, query_group, heads)
    else:
        group = normal_group
        pages = (width + _PAGE_SIZE - 1) // _PAGE_SIZE
        # Keep one warp-group's M tile full before adding more query reuse.
        # Large row counts already provide parallelism; bound the split floor
        # using the same history buckets as the established swapAB schedule.
        minimum = (
            _SHORT_HISTORY_MIN_SPLITS
            if width < _MEDIUM_HISTORY
            else _MEDIUM_HISTORY_MIN_SPLITS
        )
        if width >= _LONG_HISTORY:
            minimum = _GROUPED_SPLITS
        split = min(pages, max(minimum, _NORMAL_CTA_TARGET // (rows // group)))
    return strategy, group, split


def index_scores(
    q,
    weights,
    keys,
    pages,
    positions,
    lengths,
    causal,
    starts,
    valid,
    *,
    scale,
    initial,
    recent,
    paged,
    page_size: int,
    width,
    clear_padding,
    query_group,
    strategy: Literal["auto", "normal", "swap_ab"],
):
    """Score BF16 paged or workspace keys using normal or swapAB WGMMA.

    Q is BF16 [queries, heads, head_dim]; keys are BF16 [slots, head_dim].
    Weights have one FP32 or BF16 value per query/head and are promoted to
    FP32. Pages and positions map compact page columns to physical IDs and
    logical page IDs.
    Lengths count compact key rows; causal contains global visible lengths.
    For paged keys, starts/valid are unused tensor placeholders. For workspace
    keys, starts gives each query's first key row and valid masks workspace
    rows. Grouped queries must share starts and their ordered key prefix.

    ``query_group`` is the number of consecutive queries per request, not the
    batch size. Each group must share physical pages and have nondecreasing
    causal lengths. An incomplete group runs without sharing. The output is
    FP32 [queries, round_up(width, 4)]; clear_padding chooses whether unscored
    columns are filled with -inf. Initial/recent windows use +inf scores.

    Heads and head_dim are derived from Q. Heads must be a multiple of 8 in
    [8, 256]; head_dim must be a multiple of 64 in [64, 256]. These are WGMMA
    layout/resource constraints, not model presets. Pages contain 64 tokens.
    This direct Hopper entry is not registered in runtime dispatch and makes
    no rl-bitwise claim. ``page_size`` must be 64.
    ``strategy`` explicitly selects normal Q @ K.T,
    swap_ab K @ Q.T, or automatic dispatch. Both use WGMMA.
    """
    if strategy not in ("auto", "normal", "swap_ab"):
        raise ValueError("strategy must be auto, normal or swap_ab")
    if page_size != _PAGE_SIZE:
        raise ValueError(f"page_size must be {_PAGE_SIZE}")
    if q.ndim != 3 or keys.ndim != 2:
        raise ValueError(
            "q must have shape [queries, heads, dim] and keys [slots, dim]"
        )
    rows, heads, head_dim = q.shape
    if not _HEAD_GRANULARITY <= heads <= _MAX_WGMMA_HEADS or heads % _HEAD_GRANULARITY:
        raise ValueError("heads must be a multiple of 8 in [8, 256]")
    if (
        not _HEAD_DIM_ALIGNMENT <= head_dim <= _MAX_HEAD_DIM
        or head_dim % _HEAD_DIM_ALIGNMENT
    ):
        raise ValueError("head_dim must be a multiple of 64 in [64, 256]")
    if keys.shape[1] != head_dim or weights.shape != (rows, heads):
        raise ValueError("keys and weights must match q's head geometry")
    if q.dtype != torch.bfloat16 or keys.dtype != torch.bfloat16:
        raise ValueError("q and keys must have dtype bfloat16")
    if weights.dtype not in (torch.float32, torch.bfloat16):
        raise ValueError("weights must have dtype float32 or bfloat16")
    if not q.is_cuda or any(
        tensor.device != q.device
        for tensor in (weights, keys, pages, positions, lengths, causal, starts, valid)
    ):
        raise ValueError("all inputs must be on the same CUDA device")
    if keys.stride(1) != 1 or keys.stride(0) % 8 or keys.data_ptr() % 16:
        raise ValueError("keys require unit column stride and 16-byte-aligned rows")
    if (
        pages.ndim != 2
        or pages.shape[0] != rows
        or positions.shape != pages.shape
        or lengths.shape != (rows,)
        or causal.shape != (rows,)
    ):
        raise ValueError("page metadata and lengths must have one row per query")
    if any(
        t.dtype not in (torch.int32, torch.int64)
        for t in (pages, positions, lengths, causal)
    ):
        raise ValueError("page metadata and lengths must be int32 or int64")
    if not paged:
        if starts.shape != (rows,) or valid.shape != (keys.shape[0],):
            raise ValueError(
                "workspace starts and valid must match queries and key rows"
            )
        if starts.dtype not in (torch.int32, torch.int64):
            raise ValueError("workspace starts must be int32 or int64")
        if valid.dtype not in (torch.bool, torch.uint8, torch.int32, torch.int64):
            raise ValueError("workspace valid must be a boolean or integer mask")
    if query_group < 1:
        raise ValueError("query_group must be a positive request width")
    if width < 0 or initial < 0 or recent < 0:
        raise ValueError("width and forced windows must be nonnegative")
    out = torch.empty(
        (q.shape[0], (width + 3) // 4 * 4), device=q.device, dtype=torch.float32
    )
    if clear_padding or not keys.shape[0]:
        out.fill_(-float("inf"))
    if not q.shape[0] or not width or not keys.shape[0]:
        return out
    selected, group, split = select_score_config(
        rows, width, query_group, heads, paged=paged, strategy=strategy
    )
    if width % 4:
        lengths = lengths.clamp(max=width)
    if selected == "normal":
        from tokenspeed_kernel.ops.attention.dsa._cute_dsl.bf16_indexer_normal import (
            compile_normal_score,
        )

        compiled = compile_normal_score(
            heads, head_dim, _PIPELINE_STAGES, initial, recent, paged, group
        )
    else:
        from tokenspeed_kernel.ops.attention.dsa._cute_dsl.bf16_indexer_swapAB import (
            compile_swap_ab_score,
        )

        compiled = compile_swap_ab_score(
            heads, head_dim, _PIPELINE_STAGES, initial, recent, paged, group
        )
    q = q.contiguous()
    if q.data_ptr() % 16:
        q = q.clone()
    compiled(
        q,
        weights.float().contiguous(),
        keys,
        pages.int().contiguous(),
        positions.int().contiguous(),
        lengths.reshape(-1).int().contiguous(),
        causal.int().contiguous(),
        starts.int().contiguous(),
        valid.int().contiguous(),
        out,
        scale,
        split,
    )
    return out
