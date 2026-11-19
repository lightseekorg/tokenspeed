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

"""DSA index-K planes for tests, in either ``index_k_format``.

One pool stub over a single layer's plane, and planes written through the
production write path (``DSATokenToKVPool.set_index_k_buffer``) so a test of
the read side (``gather_index_k_rows``, the history gather) round-trips the
layout the pool actually stores rather than a hand-rolled copy of it. The
``fp8_scaled`` write is a Triton kernel: those planes are built on CUDA and
moved back to the host, and a host without CUDA skips them.

Also tie-aware checks of top-k selections against the scores a kernel ranked.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch
from tokenspeed_kernel.ops.kvcache.triton import quantize_index_k_rows

from tokenspeed.runtime.layers.attention.configs.dsa import (
    index_k_plane_dtype,
    index_k_row_bytes,
)
from tokenspeed.runtime.layers.attention.kv_cache.dsa import DSATokenToKVPool

# The quantization group of an fp8_scaled plane (kv_cache/dsa.py).
FP8_GROUP_SIZE = 128


def index_k_pool(
    plane: torch.Tensor, *, head_dim: int, page_size: int
) -> DSATokenToKVPool:
    """A ``DSATokenToKVPool`` over one layer's index-K plane and nothing else:
    ``set_index_k_buffer`` and ``gather_index_k_rows`` run as in production,
    with the plane's page size as the arena's."""
    pool = object.__new__(DSATokenToKVPool)
    pool.index_head_dim = head_dim
    pool.model_dtype = torch.bfloat16
    pool.layerwise_load_tracker = None
    pool.arena = SimpleNamespace(kv_page_size=page_size)
    pool.index_k_buffer = [plane]
    return pool


def empty_index_k_plane(
    index_k_format: str, *, head_dim: int, slots: int, device
) -> torch.Tensor:
    """A zeroed plane of ``slots`` rows in the pool's layout for the format:
    block-split uint8 rows for ``fp8_scaled``, ``[slots, head_dim]`` keys for
    ``bf16``."""
    dtype = index_k_plane_dtype(index_k_format)
    if index_k_format == "fp8_scaled":
        return torch.zeros(
            (slots, index_k_row_bytes(head_dim, index_k_format)),
            dtype=dtype,
            device=device,
        )
    return torch.zeros((slots, head_dim), dtype=dtype, device=device)


def write_index_k_plane(
    index_k_format: str,
    *,
    head_dim: int,
    page_size: int,
    slots: int,
    loc: torch.Tensor,
    keys: torch.Tensor,
) -> torch.Tensor:
    """A host plane of ``slots`` rows holding ``keys[i]`` at slot ``loc[i]``,
    written through ``DSATokenToKVPool.set_index_k_buffer``; the other slots
    stay zero."""
    device = _write_device(index_k_format)
    plane = empty_index_k_plane(
        index_k_format, head_dim=head_dim, slots=slots, device=device
    )
    pool = index_k_pool(plane, head_dim=head_dim, page_size=page_size)
    pool.set_index_k_buffer(
        0, loc.to(device), keys.to(device=device, dtype=torch.bfloat16), write_mask=None
    )
    return plane.cpu()


def expected_index_k_rows(
    index_k_format: str, keys: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor | None]:
    """What the pool stores for ``keys`` and the history gather hands back:
    the bf16 keys and no scale, or their FP8 bytes (uint8) and scale words as
    the pool's own quantizer produces them."""
    keys = keys.to(torch.bfloat16)
    if index_k_format == "bf16":
        return keys, None
    fp8, scale = quantize_index_k_rows(
        keys.to(_write_device(index_k_format)), group_size=FP8_GROUP_SIZE
    )
    return fp8.view(torch.uint8).cpu(), scale.cpu()


def _write_device(index_k_format: str) -> str:
    if index_k_format == "bf16":
        return "cpu"
    if not torch.cuda.is_available():
        pytest.skip("the fp8_scaled index-K write path is a Triton kernel")
    return "cuda"


def index_candidate_scores(
    q: torch.Tensor,
    weights: torch.Tensor,
    plane: torch.Tensor,
    table: torch.Tensor,
    requests: torch.Tensor,
    causal: torch.Tensor,
    *,
    page_size: int,
    softmax_scale: float,
    q_scales: torch.Tensor | None,
) -> torch.Tensor:
    """The FP32 index scores of query rows, by logical position.

    Every visible token is a candidate of ``dsa_index_candidates``, so each
    carries the score a top-k over the same plane ranks, bit for bit. Pages
    of ``table`` must not be the reserved null page 0, which it never scores.

    Returns:
        ``[rows, longest causal length]`` scores, -inf past each row's
        causal length.
    """
    from tokenspeed_kernel.ops.attention.dsa import dsa_index_candidates

    width = int(causal.max().clamp_min(1))
    positions, values = dsa_index_candidates(
        q,
        weights,
        plane,
        table,
        requests,
        causal,
        page_size=page_size,
        # A power of two no smaller than any row's visible tokens.
        topk=1 << (width - 1).bit_length(),
        softmax_scale=softmax_scale,
        initial_tokens=0,
        local_tokens=0,
        solution=None,
        q_scales=q_scales,
    )
    scores = values.new_full((q.shape[0], width + 1), -float("inf"))
    # Invalid candidates (-1, -inf) land in the spare last column.
    scores.scatter_(1, torch.where(positions >= 0, positions, width).long(), values)
    return scores[:, :width]


def assert_top_k_up_to_ties(
    row: int,
    selections: tuple[torch.Tensor, torch.Tensor],
    length: int,
    scores: torch.Tensor,
    topk: int,
) -> None:
    """Both ``selections`` (one row's positions, padding -1) are exact top-k
    selections of the row's ``length`` visible tokens under ``scores`` and
    differ only in tokens that score the k-th score T bit for bit."""
    visible = scores[:length]
    k = min(length, topk)
    assert k > 0, f"row {row}: no visible tokens, yet the selections differ"
    # The k-th score, as a one-element tensor to compare bit patterns.
    threshold = visible.topk(k).values[-1:]
    above = int((visible > threshold).sum())
    chosen = []
    for selection in selections:
        positions = selection[selection >= 0].long()
        count = positions.numel()
        assert count == k, f"row {row}: {count} tokens selected, expected {k}"
        assert positions.unique().numel() == k, f"row {row}: a token selected twice"
        past = positions[positions >= length].tolist()
        assert not past, f"row {row}: tokens {past[:8]} past causal length {length}"
        picked = visible[positions]
        below = positions[picked < threshold].tolist()
        assert not below, (
            f"row {row}: tokens {below[:8]} score below the k-th score "
            f"{float(threshold).hex()}"
        )
        missed = above - int((picked > threshold).sum())
        assert not missed, (
            f"row {row}: {missed} tokens above the k-th score "
            f"{float(threshold).hex()} not selected"
        )
        chosen.append(set(positions.tolist()))
    differ = sorted(chosen[0] ^ chosen[1])
    tied = visible[differ].view(torch.int32) == threshold.view(torch.int32)
    assert bool(tied.all()), (
        f"row {row}: tokens {differ[:8]} differ but do not all score the k-th "
        f"score {float(threshold).hex()} exactly"
    )
