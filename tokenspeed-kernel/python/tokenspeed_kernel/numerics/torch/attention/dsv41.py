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

"""PyTorch references for ``tokenspeed_kernel.ops.attention.dsv41``
(DeepSeek-V4.1): cache row quantization, the compressor metadata, index-key
norm and rotation, and the indexer's scores and selection inputs."""

from __future__ import annotations

import torch
from tokenspeed_kernel.numerics.torch.quantization import dequantize_mxfp4
from tokenspeed_kernel.registry import Priority, register_kernel
from tokenspeed_kernel.signature import format_signatures

# (row width, values per scale) of each V4.1 cache row format.
DSV41_CACHE_FORMATS = {"swa": (512, 32), "global": (512, 16), "index": (128, 32)}
_E2M1_LEVELS = (0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0)


@register_kernel(
    "attention",
    "dsv41_cache_quantize",
    name="torch_dsv41_cache_quantize",
    solution="torch",
    signatures=format_signatures("x", "dense", {torch.bfloat16, torch.float32}),
    priority=Priority.REFERENCE,
)
def torch_dsv41_cache_quantize(
    x: torch.Tensor, cache_format: str
) -> tuple[torch.Tensor, torch.Tensor]:
    """Quantize cache rows to a V4.1 format by nearest value, on the host.

    ``swa`` stores FP8 E4M3 with power-of-two scales of ``amax / 448``
    (``amax >= 1e-4``) per 32 values; ``index`` stores E2M1 with power-of-two
    scales of ``amax / 6`` per 32; ``global`` stores E2M1 with FP8 E4M3
    scales of ``amax / 6`` per 16. E2M1 ties round to the even code.
    Semantics follow DeepSeek's reference ``inference/kernel.py``.

    Returns:
        ``(packed, decoded)`` on the CPU: ``[rows, payload + scales]`` uint8
        rows (payload, then one scale byte per group) and the FP32
        dequantized ``[rows, dim]`` values.
    """
    dim, group = DSV41_CACHE_FORMATS[cache_format]
    x = x.detach().cpu().float().reshape(-1, dim // group, group)
    amax = x.abs().amax(dim=-1)
    if cache_format == "global":
        scale = (amax.clamp_min(6 * 2.0**-9) / 6).to(torch.float8_e4m3fn)
        scale_bytes = scale.view(torch.uint8)
        scale = scale.float()
    else:
        swa = cache_format == "swa"
        scaled = amax.clamp_min(1e-4 if swa else 6 * 2.0**-126)
        scaled = scaled * (1 / (448 if swa else 6))
        bits = scaled.view(torch.int32)
        exponent = ((bits >> 23) & 255) - 127 + ((bits & 0x7FFFFF) != 0).int()
        scale = torch.exp2(exponent.float())
        scale_bytes = (exponent + 127).to(torch.uint8)
    normalized = x / scale.unsqueeze(-1)
    if cache_format == "swa":
        quant = normalized.clamp(-448, 448).to(torch.float8_e4m3fn)
        values = quant.view(torch.uint8).flatten(1)
        decoded = quant.float()
    else:
        levels = torch.tensor(_E2M1_LEVELS)
        # Searching even codes first implements ties-to-even.
        order = torch.tensor([0, 2, 4, 6, 1, 3, 5, 7])
        distance = (normalized.abs().unsqueeze(-1) - levels[order]).abs()
        code = order[distance.argmin(dim=-1)]
        negative = torch.signbit(normalized)
        decoded = levels[code] * torch.where(negative, -1.0, 1.0)
        code = (code | (negative.int() << 3)).to(torch.uint8).flatten(1)
        values = code[:, ::2] | (code[:, 1::2] << 4)
    packed = torch.cat((values, scale_bytes.flatten(1)), dim=1)
    return packed, (decoded * scale.unsqueeze(-1)).reshape(-1, dim)


@register_kernel(
    "attention",
    "dsv41_compressor_pairs",
    name="torch_dsv41_compressor_pairs",
    solution="torch",
    signatures=format_signatures("positions", "dense", {torch.int32, torch.int64}),
    priority=Priority.REFERENCE,
)
def torch_dsv41_compressor_pairs(
    positions: torch.Tensor, requests: torch.Tensor, table: torch.Tensor, pages: int
) -> tuple[torch.Tensor, ...]:
    """The two-row compressor's pairing metadata, on the host.

    A token at an odd position closes the pair ``(position - 1, position)``;
    its source row of the pair's first position is the earlier token of the
    same request at that position. Cache slots map position ``p`` of request
    ``r`` to ``table[r, p // 2] * 2 + p % 2`` for pages in ``(0, pages)``.

    Returns:
        ``(active, pair_positions, pair_requests, previous, read_slots,
        write_slots)`` per token; inactive pairs and unmapped slots are -1.
    """
    positions, requests, table = positions.cpu(), requests.cpu(), table.cpu()
    sources = {(int(r), int(p)): i for i, (p, r) in enumerate(zip(positions, requests))}

    def slot(p: int, r: int) -> int:
        if p < 0 or not 0 <= r < table.shape[0] or p // 2 >= table.shape[1]:
            return -1
        page = int(table[r, p // 2])
        return page * 2 + p % 2 if 0 < page < pages else -1

    rows = []
    for p, r in zip(positions.tolist(), requests.tolist(), strict=True):
        active = p >= 0 and p % 2 == 1
        pair, request = (p - 1, r) if active else (-1, -1)
        previous = sources.get((request, pair), -1) if active and r >= 0 else -1
        rows.append((active, pair, request, previous, slot(pair, request), slot(p, r)))
    dtypes = (
        torch.bool,
        positions.dtype,
        requests.dtype,
        torch.int64,
        torch.int64,
        torch.int64,
    )
    return tuple(
        torch.tensor([row[i] for row in rows], dtype=dtype)
        for i, dtype in enumerate(dtypes)
    )


def dsv41_key_norm_rope(
    values: torch.Tensor,
    weight: torch.Tensor,
    positions: torch.Tensor,
    cos_sin_cache: torch.Tensor,
    eps: float,
    *,
    round_normalized: bool,
) -> torch.Tensor:
    """RMSNorm, then rotate the trailing ``rotary_dim`` features in adjacent
    pairs; ``round_normalized`` rounds the norm to the input dtype first.
    Negative positions read row 0."""
    x = values.float()
    normalized = x * torch.rsqrt(x.square().mean(-1, keepdim=True) + eps)
    normalized = normalized * weight.float()
    if round_normalized:
        normalized = normalized.to(values.dtype).float()
    rotary_dim = cos_sin_cache.shape[1]
    pairs = normalized[:, -rotary_dim:].unflatten(-1, (-1, 2))
    cosine, sine = cos_sin_cache[positions.clamp_min(0)].chunk(2, dim=-1)
    even = pairs[..., 0] * cosine - pairs[..., 1] * sine
    odd = pairs[..., 1] * cosine + pairs[..., 0] * sine
    rotated = torch.stack((even, odd), dim=-1).flatten(-2)
    return torch.cat((normalized[:, :-rotary_dim], rotated), dim=-1).to(values.dtype)


@register_kernel(
    "attention",
    "dsv41_index_key_norm_rope",
    name="torch_dsv41_index_key_norm_rope",
    solution="torch",
    signatures=format_signatures("x", "dense", {torch.bfloat16}),
    priority=Priority.REFERENCE,
)
def torch_dsv41_index_key_norm_rope(
    values: torch.Tensor,
    weight: torch.Tensor,
    positions: torch.Tensor,
    cos_sin_cache: torch.Tensor,
    eps: float,
) -> torch.Tensor:
    """Index-key RMSNorm (rounded to the input dtype) and trailing-pair
    rotation; see :func:`dsv41_key_norm_rope`."""
    return dsv41_key_norm_rope(
        values, weight, positions, cos_sin_cache, eps, round_normalized=True
    )


@register_kernel(
    "attention",
    "dsv41_index_scores",
    name="torch_dsv41_index_scores",
    solution="torch",
    signatures=format_signatures("q", "dense", {torch.bfloat16}),
    priority=Priority.REFERENCE,
)
def torch_dsv41_index_scores(
    q: torch.Tensor, weights: torch.Tensor, keys: torch.Tensor, shards: int
) -> torch.Tensor:
    """Indexer scores of index-quantized queries and keys, on the host.

    Per-head ``relu(q . k)`` rounds to BF16 and is weighted; heads sum within
    each of ``shards`` equal head shards, the shard sums add in FP32 in shard
    order, and the total rounds to ``weights``'s dtype.

    Args:
        q: ``[tokens, heads, 128]``.
        weights: ``[tokens, heads]``.
        keys: ``[keys, 128]``.

    Returns:
        FP32 ``[tokens, keys]``.
    """
    _, q_ref = torch_dsv41_cache_quantize(q, "index")
    _, k_ref = torch_dsv41_cache_quantize(keys, "index")
    dots = torch.einsum("thd,nd->thn", q_ref.reshape(q.shape), k_ref).bfloat16()
    products = dots.relu() * weights.cpu().unsqueeze(-1)
    scores = torch.zeros((q.shape[0], keys.shape[0]), dtype=torch.float32)
    for partial in products.chunk(shards, dim=1):
        scores += partial.sum(dim=1).float()
    return scores.to(weights.dtype).float()


@register_kernel(
    "attention",
    "dsv41_gather_candidates",
    name="torch_dsv41_gather_candidates",
    solution="torch",
    signatures=format_signatures("logits", "dense", {torch.float32, torch.bfloat16}),
    priority=Priority.REFERENCE,
)
def torch_dsv41_gather_candidates(
    logits: torch.Tensor,
    lengths: torch.Tensor,
    candidates: torch.Tensor,
    block_size: int,
    alignment: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Gather each row's candidate blocks of logits into a packed row.

    A row's candidates inside its visible ``lengths`` are deduplicated and
    sorted; block ``i`` lands at columns ``[i * block_size, ...)`` truncated at
    the visible end. Rows pad to a multiple of ``alignment`` with ``-inf``.

    Returns:
        On the host: ``(scores [rows, stride] FP32, ends [rows] int32,
        normalized [rows, candidates] int32)``: the packed logits, each
        row's packed length and its sorted candidates (``-1`` padded).
    """
    logits, lengths, candidates = (x.cpu() for x in (logits, lengths, candidates))
    count = candidates.shape[1]
    stride = (count * block_size + alignment - 1) // alignment * alignment
    scores = torch.full((logits.shape[0], stride), -torch.inf)
    normalized = torch.full(candidates.shape, -1, dtype=torch.int32)
    ends = torch.zeros(logits.shape[0], dtype=torch.int32)
    for row in range(logits.shape[0]):
        end = max(0, min(int(lengths[row]), logits.shape[1]))
        ids = sorted(
            int(c)
            for c in candidates[row]
            if 0 <= c < (end + block_size - 1) // block_size
        )
        for i, c in enumerate(ids):
            if i and c == ids[i - 1]:
                continue
            normalized[row, i] = c
            n = min(block_size, end - c * block_size)
            scores[row, i * block_size : i * block_size + n] = logits[
                row, c * block_size : c * block_size + n
            ]
            ends[row] = i * block_size + n
    return scores, ends, normalized


@register_kernel(
    "attention",
    "dsv41_index_cache_gather",
    name="torch_dsv41_index_cache_gather",
    solution="torch",
    signatures=format_signatures("cache", "dense", {torch.uint8}),
    priority=Priority.REFERENCE,
)
def torch_dsv41_index_cache_gather(
    cache: torch.Tensor, table: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """Gather the index cache rows a page table maps, 64 rows per page.

    Each page holds 64 rows of 64 payload bytes, then 64 rows of four UE8M0
    scale bytes. Unmapped pages read zero payload and scale bytes 127.

    Returns:
        ``(payload [rows, 64] uint8, scales [rows] int32)``: the four scale
        bytes of a row as one little-endian word.
    """
    logical = torch.arange(table.numel() * 64, device=cache.device, dtype=torch.int64)
    pages = table[logical // 64]
    valid = (pages >= 0) & (pages < cache.shape[0])
    page = pages.to(torch.int64).clamp(0, cache.shape[0] - 1)[:, None]
    row = (logical % 64)[:, None]
    flat = cache.as_strided((cache.shape[0], 64 * 68), (cache.stride(0), 1))
    data = flat[page, row * 64 + torch.arange(64, device=cache.device)]
    data = data.masked_fill(~valid[:, None], 0)
    sf = flat[page, 64 * 64 + row * 4 + torch.arange(4, device=cache.device)]
    sf = sf.masked_fill(~valid[:, None], 127).to(torch.int64)
    word = (sf << torch.tensor([0, 8, 16, 24], device=cache.device)).sum(-1)
    return data, ((word + 2**31) % 2**32 - 2**31).to(torch.int32)


@register_kernel(
    "attention",
    "dsv41_block_maxima",
    name="torch_dsv41_block_maxima",
    solution="torch",
    signatures=format_signatures("logits", "dense", {torch.float32, torch.bfloat16}),
    priority=Priority.REFERENCE,
)
def torch_dsv41_block_maxima(
    logits: torch.Tensor, visible: torch.Tensor
) -> torch.Tensor:
    """Maximum logit of each 8-column block; a row's newest visible block, if
    it has a finite maximum, becomes ``+inf`` so it is always selected.

    Returns:
        ``[rows, ceil(width / 8)]`` in ``logits``'s dtype.
    """
    rows, width = logits.shape
    if width % 8:
        logits = torch.nn.functional.pad(logits, (0, (-width) % 8), value=-float("inf"))
    result = logits.reshape(rows, (width + 7) // 8, 8).amax(-1)
    newest = ((visible.to(torch.int64) - 1) // 8).clamp(0, result.shape[1] - 1)
    row = torch.arange(rows, device=logits.device)
    current = result[row, newest]
    result[row, newest] = torch.where(
        current > -float("inf"), torch.full_like(current, float("inf")), current
    )
    return result


@register_kernel(
    "attention",
    "dsv41_sparse_scores",
    name="torch_dsv41_sparse_scores",
    solution="torch",
    signatures=format_signatures("cache", "dense", {torch.uint8}),
    priority=Priority.REFERENCE,
)
def torch_dsv41_sparse_scores(
    queries: tuple[torch.Tensor, torch.Tensor],
    weights: torch.Tensor,
    cache: torch.Tensor,
    table: torch.Tensor,
    lengths: torch.Tensor,
    candidates: torch.Tensor,
    *,
    alignment: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Indexer scores of each token's candidate 8-row blocks, with DeepGEMM's
    BF16 head arithmetic.

    Candidate blocks inside the visible length with a mapped page are
    deduplicated and sorted. Their E2M1/UE8M0 keys (64 rows of 64 payload
    bytes, then 4 scale bytes per row, per page) and the queries are decoded
    explicitly; ``relu(q . k)`` rounds to BF16, heads accumulate four at a
    time in BF16, and the four partials add pairwise in BF16.

    Args:
        queries: ``(payload [tokens, 32, 64] uint8, scales [tokens, 32]
            int32)`` packed queries (scale bytes little-endian in a word).
        weights: ``[tokens, 32]`` head weights.
        cache: ``[pages, 64 * 68]`` uint8 index cache.
        table: ``[tokens, pages]`` page table.
        lengths: ``[tokens]`` visible positions.
        candidates: ``[tokens, count]`` block ids.
        alignment: Column alignment of the packed scores.

    Returns:
        ``(scores [tokens, width] BF16, ends [tokens] int32, normalized
        [tokens, width / 8] int32, full [tokens, pages * 64] BF16)``: packed
        scores, valid columns, sorted blocks, and scores by position.
    """
    device = cache.device
    tokens, count = candidates.shape
    width = (count * 8 + alignment - 1) // alignment * alignment
    scores = torch.full(
        (tokens, width), -torch.inf, device=device, dtype=torch.bfloat16
    )
    full = torch.full(
        (tokens, table.shape[1] * 64), -torch.inf, device=device, dtype=torch.bfloat16
    )
    normalized = torch.full((tokens, width // 8), -1, device=device, dtype=torch.int32)
    ends = torch.zeros(tokens, device=device, dtype=torch.int32)
    exponents = (
        queries[1].long()[..., None] >> (torch.arange(4, device=device) * 8)
    ) & 255
    q = dequantize_mxfp4(queries[0], exponents.to(torch.uint8))
    tables, visible, blocks = (
        table.cpu().tolist(),
        lengths.cpu().tolist(),
        candidates.cpu().tolist(),
    )
    for row in range(tokens):
        end = min(max(visible[row], 0), full.shape[1])
        chosen = sorted(
            {
                block
                for block in blocks[row]
                if 0 <= block * 8 < end
                and 0 <= tables[row][block // 8] < cache.shape[0]
            }
        )
        if not chosen:
            continue
        chosen = torch.tensor(chosen, device=device)
        logical = (chosen[:, None] * 8 + torch.arange(8, device=device)).flatten()
        pages = table[row, logical // 64].long()
        offsets = logical % 64
        payload = cache[
            pages[:, None], offsets[:, None] * 64 + torch.arange(64, device=device)
        ]
        scale_bytes = cache[
            pages[:, None], 4096 + offsets[:, None] * 4 + torch.arange(4, device=device)
        ]
        keys = dequantize_mxfp4(payload, scale_bytes)
        dots = (q[row] @ keys.T).relu().bfloat16().float()
        partial = torch.zeros((4, logical.numel()), device=device)
        for head in range(0, 32, 4):
            partial = (
                (
                    dots[head : head + 4]
                    * weights[row, head : head + 4].float()[:, None]
                    + partial
                )
                .bfloat16()
                .float()
            )
        pairs = (partial[:2] + partial[2:]).bfloat16().float()
        values = (pairs[0] + pairs[1]).bfloat16()
        valid = logical < end
        scores[row, : logical.numel()] = values.masked_fill(~valid, -torch.inf)
        full[row, logical[valid]] = values[valid]
        normalized[row, : chosen.numel()] = chosen.int()
        ends[row] = valid.sum()
    return scores, ends, normalized, full
