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

"""Benchmark generators for DeepSeek V4.1 compressed sparse attention.

The generators mirror the V4.1 attention backend: packed page-planar SWA,
global and index caches, a 128-token sliding window, up to 512 compressed rows
selected per query, and the indexer that selects them. Selected rows are spread
evenly over each query's visible history, which matches the selection width and
gather footprint of a real top-k without depending on model weights.
"""

from __future__ import annotations

import math
from collections.abc import Collection
from dataclasses import dataclass
from typing import Any

import torch
from tokenspeed_kernel.benchmark.graph import PreparedInvocation
from tokenspeed_kernel.benchmark.harness import (
    BenchmarkCaseError,
    BenchmarkRequest,
    BenchmarkStatus,
    PreparedBenchmark,
)
from tokenspeed_kernel.platform import PlatformInfo
from tokenspeed_kernel.registry import KernelRegistry, KernelSpec, load_builtin_kernels
from tokenspeed_kernel.selection import NoKernelFoundError, select_kernel
from tokenspeed_kernel.signature import dense_tensor_format, format_signature

__all__ = [
    "prepare_dsv41_index_topk",
    "prepare_dsv41_selected_attention",
    "prepare_dsv4_prefill",
]


_IMPLEMENTED_MODEL_PROFILES = frozenset({"dsv41_flash_tp4"})
_IMPLEMENTED_COMPRESS_RATIOS = frozenset({0, 1, 2})
_IMPLEMENTED_SELECTIONS = frozenset({"full", "full_candidates", "reindex"})
_HEAD_DIM = 512
_INDEX_HEAD_DIM = 128
_PAGE_ROWS = 64
_ROW_BYTES = {"swa": 528, "global": 288, "index": 68}
_ROW_DIMS = {"swa": _HEAD_DIM, "global": _HEAD_DIM, "index": _INDEX_HEAD_DIM}
# Rows quantized per cache_scatter launch while filling a cache.
_FILL_CHUNK_ROWS = 1 << 16
# The backend's decode indexer tile and score tile.
_DECODE_QUERY_TILE = 256
_SCORE_CHUNK_SIZE = 256


@dataclass(frozen=True)
class _DSV41Config:
    model_profile: str
    local_heads: int
    swa_window: int
    index_topk: int
    index_heads: int
    candidate_block_size: int
    candidate_topk_blocks: int

    @property
    def softmax_scale(self) -> float:
        return _HEAD_DIM**-0.5

    @property
    def index_weight_scale(self) -> float:
        return _INDEX_HEAD_DIM**-0.5 * self.index_heads**-0.5


def _invalid(message: str) -> BenchmarkCaseError:
    return BenchmarkCaseError(BenchmarkStatus.INVALID_CASE, message)


def _implemented_value(name: str, value: object, implemented: Collection[object]):
    if value not in implemented:
        accepted = ", ".join(str(item) for item in sorted(implemented))
        raise _invalid(f"Implemented DSV4.1 {name} values: {accepted}")
    return value


def _positive(parameters: dict[str, Any], name: str) -> int:
    value = parameters[name]
    if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
        raise _invalid(f"DSV4.1 {name} must be a positive integer")
    return value


def _nonnegative(parameters: dict[str, Any], name: str) -> int:
    value = parameters[name]
    if not isinstance(value, int) or isinstance(value, bool) or value < 0:
        raise _invalid(f"DSV4.1 {name} must be a nonnegative integer")
    return value


def _resolve_config(request: BenchmarkRequest) -> _DSV41Config:
    parameters = request.parameters
    if parameters.get("validation") is not None:
        raise _invalid("DSV4.1 benchmark correctness validation is not implemented")
    config = _DSV41Config(
        model_profile=_implemented_value(
            "model_profile", parameters["model_profile"], _IMPLEMENTED_MODEL_PROFILES
        ),
        local_heads=_positive(parameters, "local_heads"),
        swa_window=_positive(parameters, "swa_window"),
        index_topk=_positive(parameters, "index_topk"),
        index_heads=_positive(parameters, "index_heads"),
        candidate_block_size=_positive(parameters, "candidate_block_size"),
        candidate_topk_blocks=_positive(parameters, "candidate_topk_blocks"),
    )
    if config.candidate_block_size != 8:
        raise _invalid("DSV4.1 candidate_block_size must be 8")
    return config


def _common_parameters(config: _DSV41Config) -> dict[str, object]:
    return {
        "model_profile": config.model_profile,
        "local_heads": config.local_heads,
        "head_dim": _HEAD_DIM,
        "swa_window": config.swa_window,
        "index_topk": config.index_topk,
        "index_heads": config.index_heads,
        "index_head_dim": _INDEX_HEAD_DIM,
        "candidate_block_size": config.candidate_block_size,
        "candidate_topk_blocks": config.candidate_topk_blocks,
    }


def _select(
    request: BenchmarkRequest,
    platform: PlatformInfo,
    mode: str,
    signature_roles: dict[str, torch.dtype],
    traits: dict[str, object],
) -> tuple[KernelSpec, Any]:
    load_builtin_kernels()
    signature = format_signature(
        **{role: dense_tensor_format(dtype) for role, dtype in signature_roles.items()}
    )
    try:
        selected = select_kernel(
            "attention",
            mode,
            signature,
            platform=platform,
            traits=traits,
            solution=request.solution,
            override=request.registration,
        )
    except NoKernelFoundError as error:
        raise BenchmarkCaseError(BenchmarkStatus.NOT_APPLICABLE, str(error)) from error
    spec = KernelRegistry.get().get_by_name(selected.name)
    if spec is None:
        raise BenchmarkCaseError(
            BenchmarkStatus.REGISTRATION_MISSING,
            f"Selected registration {selected.name!r} is not available",
        )
    return spec, selected


def _generator(seed: int) -> torch.Generator:
    return torch.Generator(device="cuda").manual_seed(seed)


def _randn(
    shape: tuple[int, ...],
    *,
    generator: torch.Generator,
    dtype: torch.dtype = torch.bfloat16,
) -> torch.Tensor:
    return torch.randn(shape, device="cuda", dtype=dtype, generator=generator)


def _filled_cache(
    cache_format: str,
    rows: int,
    *,
    generator: torch.Generator,
) -> torch.Tensor:
    """Return a packed ``[pages, 64, row_bytes]`` field with ``rows`` valid rows.

    Rows are quantized from random BF16 vectors with the cache codec, so every
    stored byte is a value the attention kernels can meet in serving.
    """
    from tokenspeed_kernel.ops.attention import dsv41

    pages = max(1, math.ceil(rows / _PAGE_ROWS))
    cache = torch.zeros(
        (pages, _PAGE_ROWS, _ROW_BYTES[cache_format]),
        dtype=torch.uint8,
        device="cuda",
    )
    for start in range(0, rows, _FILL_CHUNK_ROWS):
        stop = min(start + _FILL_CHUNK_ROWS, rows)
        values = _randn((stop - start, _ROW_DIMS[cache_format]), generator=generator)
        slots = torch.arange(start, stop, dtype=torch.int64, device="cuda")
        dsv41.cache_scatter(values, cache, slots, cache_format)
    return cache


def spread_rows(visible: torch.Tensor, width: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Select ``min(width, visible)`` distinct rows spread over each history.

    Args:
        visible: Integer ``[T]`` visible row counts.
        width: Selection capacity.

    Returns:
        Position-sorted int32 ``[T, width]`` logical rows padded with -1, and
        int32 ``[T]`` selected counts.
    """
    visible = visible.to(torch.int64).clamp_min(0)
    lens = visible.clamp_max(width)
    column = torch.arange(width, dtype=torch.int64, device=visible.device)
    # floor(j * visible / lens) is strictly increasing in j when visible >= lens.
    rows = column[None, :] * visible[:, None] // lens.clamp_min(1)[:, None]
    rows = rows.masked_fill(column[None, :] >= lens[:, None], -1)
    return rows.to(torch.int32), lens.to(torch.int32)


def _query_positions(
    batch: int, q_len_per_req: int, context_length: int
) -> tuple[torch.Tensor, torch.Tensor]:
    if context_length < q_len_per_req:
        raise _invalid("DSV4.1 context_length must cover the query tokens")
    offsets = torch.arange(q_len_per_req, dtype=torch.int64, device="cuda")
    positions = (context_length - q_len_per_req + offsets).repeat(batch)
    requests = torch.arange(batch, dtype=torch.int64, device="cuda")
    return positions, requests.repeat_interleave(q_len_per_req)


def prepare_dsv41_selected_attention(
    request: BenchmarkRequest,
    platform: PlatformInfo,
) -> PreparedBenchmark:
    """Prepare one paged V4.1 decode/verify attention call.

    Each query attends to its 128-token SWA window and, for compressed layers
    (``compress_ratio`` 1 or 2), up to ``index_topk`` global rows. Every
    request keeps its window in a page-aligned ring of the SWA field and its
    compressed history in a separate global field, as the LCM groups do.
    """

    config = _resolve_config(request)
    parameters = request.parameters
    batch = _positive(parameters, "batch")
    q_len_per_req = _positive(parameters, "q_len_per_req")
    context_length = _positive(parameters, "context_length")
    ratio = _implemented_value(
        "compress_ratio", parameters["compress_ratio"], _IMPLEMENTED_COMPRESS_RATIOS
    )
    spec, kernel = _select(
        request,
        platform,
        "dsv41_selected_attention",
        {"x": torch.bfloat16},
        {"flashmla_eligible": False},
    )

    generator = _generator(request.seed)
    positions, requests = _query_positions(batch, q_len_per_req, context_length)
    tokens = positions.numel()
    q = _randn((tokens, config.local_heads, _HEAD_DIM), generator=generator)
    attn_sink = _randn((config.local_heads,), generator=generator, dtype=torch.float32)

    # The ring holds the window plus the verified tokens without aliasing.
    ring_rows = (
        math.ceil((config.swa_window + q_len_per_req) / _PAGE_ROWS) + 1
    ) * _PAGE_ROWS
    swa_cache = _filled_cache("swa", batch * ring_rows, generator=generator)
    # Valid window slots form a prefix, oldest first.
    swa_lens = (positions + 1).clamp_max(config.swa_window)
    column = torch.arange(config.swa_window, dtype=torch.int64, device="cuda")
    window_positions = (positions - swa_lens + 1)[:, None] + column[None, :]
    swa_slots = (
        (requests[:, None] * ring_rows + window_positions.remainder(ring_rows))
        .masked_fill(column[None, :] >= swa_lens[:, None], -1)
        .to(torch.int32)
    )
    swa_lens = swa_lens.to(torch.int32)

    global_cache = global_slots = global_lens = None
    global_rows = 0
    if ratio:
        global_rows = math.ceil(context_length // ratio / _PAGE_ROWS) * _PAGE_ROWS
        global_cache = _filled_cache(
            "global", batch * max(global_rows, _PAGE_ROWS), generator=generator
        )
        rows, global_lens = spread_rows((positions + 1) // ratio, config.index_topk)
        global_slots = torch.where(
            rows >= 0, rows + (requests * global_rows)[:, None].to(torch.int32), -1
        ).to(torch.int32)

    out = torch.empty_like(q)

    def invoke() -> object:
        return kernel(
            q,
            swa_cache,
            swa_slots,
            swa_lens,
            global_cache,
            global_slots,
            global_lens,
            attn_sink,
            config.softmax_scale,
            out,
            _DECODE_QUERY_TILE,
            None,
            None,
            None,
        )

    return PreparedBenchmark(
        registration=spec,
        invocation=PreparedInvocation(invoke=invoke),
        parameters={
            **_common_parameters(config),
            "batch": batch,
            "q_len_per_req": q_len_per_req,
            "context_length": context_length,
            "compress_ratio": ratio,
            "swa_ring_rows_per_request": ring_rows,
            "global_rows_per_request": global_rows,
            "selected_width": config.swa_window + (config.index_topk if ratio else 0),
        },
        validation=None,
    )


def prefill_swa_indices(
    positions: torch.Tensor, prefix_begin: int, swa_window: int
) -> torch.Tensor:
    """Workspace rows of each query's SWA window, oldest first, -1 when absent.

    The workspace starts with the retained prefix rows from ``prefix_begin``
    followed by the current chunk, so a position's row is its offset from
    ``prefix_begin``.
    """
    wanted = positions[:, None] - torch.arange(
        swa_window - 1, -1, -1, dtype=positions.dtype, device=positions.device
    )
    return (
        (wanted - prefix_begin).masked_fill(wanted < prefix_begin, -1).to(torch.int32)
    )


def prepare_dsv4_prefill(
    request: BenchmarkRequest,
    platform: PlatformInfo,
) -> PreparedBenchmark:
    """Prepare one V4.1 prefill selected-attention call over a dense workspace.

    One request contributes ``tokens`` queries after ``prefix_tokens`` cached
    tokens. The BF16 workspace holds the retained SWA prefix, the current
    chunk and, for compressed layers, the whole compressed history; indices
    concatenate the 128-row window with up to ``index_topk`` selected history
    rows, as the backend builds them.
    """

    config = _resolve_config(request)
    parameters = request.parameters
    prefix_tokens = _nonnegative(parameters, "prefix_tokens")
    tokens = _positive(parameters, "tokens")
    ratio = _implemented_value(
        "compress_ratio", parameters["compress_ratio"], _IMPLEMENTED_COMPRESS_RATIOS
    )
    width = config.swa_window + (config.index_topk if ratio else 0)
    spec, _ = _select(
        request,
        platform,
        "dsv4_prefill",
        {"q": torch.bfloat16, "kv": torch.bfloat16},
        {
            "num_q_heads": config.local_heads,
            "head_dim": _HEAD_DIM,
            "selected_width": width,
            "cache_layout": "dense_workspace",
            "metadata_dtypes": frozenset({torch.int32}),
            "sinks": True,
        },
    )

    generator = _generator(request.seed)
    prefix_begin = max(0, prefix_tokens - (config.swa_window - 1))
    swa_rows = prefix_tokens - prefix_begin + tokens
    last_position = prefix_tokens + tokens - 1
    history_rows = (last_position + 1) // ratio if ratio else 0
    q = _randn((tokens, config.local_heads, _HEAD_DIM), generator=generator)
    kv = _randn((swa_rows + history_rows, 1, _HEAD_DIM), generator=generator)
    attn_sink = _randn((config.local_heads,), generator=generator, dtype=torch.float32)
    positions = torch.arange(
        prefix_tokens, prefix_tokens + tokens, dtype=torch.int64, device="cuda"
    )
    indices = prefill_swa_indices(positions, prefix_begin, config.swa_window)
    if ratio:
        selected, _ = spread_rows((positions + 1) // ratio, config.index_topk)
        indices = torch.cat(
            (indices, torch.where(selected >= 0, selected + swa_rows, -1)), dim=-1
        )
    indices = indices.to(torch.int32).contiguous()
    lens = torch.full((tokens,), width, dtype=torch.int32, device="cuda")
    out = torch.empty_like(q)

    from tokenspeed_kernel.ops.attention.dsv4 import dsv4_prefill

    def invoke() -> object:
        return dsv4_prefill(
            q=q,
            kv=kv,
            indices=indices,
            lens=lens,
            attn_sink=attn_sink,
            softmax_scale=config.softmax_scale,
            out=out,
            override=spec.name,
            solution=None,
        )

    return PreparedBenchmark(
        registration=spec,
        invocation=PreparedInvocation(invoke=invoke),
        parameters={
            **_common_parameters(config),
            "prefix_tokens": prefix_tokens,
            "tokens": tokens,
            "compress_ratio": ratio,
            "selected_width": width,
            "workspace_rows": swa_rows + history_rows,
        },
        validation=None,
    )


def prepare_dsv41_index_topk(
    request: BenchmarkRequest,
    platform: PlatformInfo,
) -> PreparedBenchmark:
    """Prepare one V4.1 indexer call: score index rows and select the top rows.

    ``selection`` picks the indexer layer kind: ``full`` scans the visible
    history, ``full_candidates`` also emits source candidate blocks (the
    candidate-source layer), and ``reindex`` scores only the source's
    ``candidate_topk_blocks`` 8-row blocks. Every query reads its request's
    page table of ``table_pages`` columns; decode tables are sized by the
    serving context capacity rather than by the live history, and the scored
    width follows the table.
    """

    config = _resolve_config(request)
    parameters = request.parameters
    batch = _positive(parameters, "batch")
    q_len_per_req = _positive(parameters, "q_len_per_req")
    context_length = _positive(parameters, "context_length")
    ratio = _implemented_value(
        "compress_ratio", parameters["compress_ratio"], _IMPLEMENTED_COMPRESS_RATIOS
    )
    if ratio == 0:
        raise _invalid("DSV4.1 indexer layers compress their history")
    selection = _implemented_value(
        "selection", parameters["selection"], _IMPLEMENTED_SELECTIONS
    )
    table_pages = _positive(parameters, "table_pages")
    query_chunk_size = _positive(parameters, "query_chunk_size")
    history_pages = math.ceil(context_length // ratio / _PAGE_ROWS)
    if table_pages < history_pages:
        raise _invalid("DSV4.1 table_pages must cover the compressed history")
    spec, kernel = _select(
        request,
        platform,
        "dsv41_index_topk",
        {"x": torch.bfloat16},
        {
            "index_heads": config.index_heads,
            "index_k_format": "mxfp4",
            "index_shards": 1,
            "native_indexer": False,
        },
    )

    generator = _generator(request.seed)
    positions, requests = _query_positions(batch, q_len_per_req, context_length)
    tokens = positions.numel()
    index_q = _randn((tokens, config.index_heads, _INDEX_HEAD_DIM), generator=generator)
    weights = _randn((tokens, config.index_heads), generator=generator).mul_(
        config.index_weight_scale
    )
    # Page zero is the null page; each request owns history_pages live pages.
    index_cache = _filled_cache(
        "index", (1 + batch * history_pages) * _PAGE_ROWS, generator=generator
    )
    column = torch.arange(table_pages, dtype=torch.int32, device="cuda")
    request_tables = torch.where(
        column[None, :] < history_pages,
        1
        + torch.arange(batch, dtype=torch.int32, device="cuda")[:, None] * history_pages
        + column[None, :],
        -1,
    )
    page_table = request_tables[requests].contiguous()
    visible = ((positions + 1) // ratio).to(torch.int32)

    candidate_blocks = None
    candidate_topk = 0
    if selection == "reindex":
        candidate_blocks, _ = spread_rows(
            (visible + config.candidate_block_size - 1) // config.candidate_block_size,
            config.candidate_topk_blocks,
        )
    elif selection == "full_candidates":
        candidate_topk = min(
            config.candidate_topk_blocks,
            table_pages * _PAGE_ROWS // config.candidate_block_size,
        )
    out = (
        torch.empty((tokens, config.index_topk), dtype=torch.int32, device="cuda"),
        torch.empty((tokens,), dtype=torch.int32, device="cuda"),
        torch.empty((tokens, candidate_topk), dtype=torch.int32, device="cuda"),
        torch.empty((tokens,), dtype=torch.int32, device="cuda"),
    )

    def invoke() -> object:
        return kernel(
            index_q,
            weights,
            index_cache,
            page_table,
            visible,
            candidate_blocks,
            config.index_topk,
            candidate_topk,
            config.candidate_block_size,
            query_chunk_size,
            _SCORE_CHUNK_SIZE,
            None,
            out,
        )

    return PreparedBenchmark(
        registration=spec,
        invocation=PreparedInvocation(invoke=invoke),
        parameters={
            **_common_parameters(config),
            "batch": batch,
            "q_len_per_req": q_len_per_req,
            "context_length": context_length,
            "compress_ratio": ratio,
            "selection": selection,
            "table_pages": table_pages,
            "history_pages": history_pages,
            "query_chunk_size": query_chunk_size,
            "score_chunk_size": _SCORE_CHUNK_SIZE,
            "candidate_topk": candidate_topk,
        },
        validation=None,
    )
