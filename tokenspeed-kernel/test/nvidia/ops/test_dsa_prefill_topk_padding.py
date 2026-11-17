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

"""The DeepGEMM prefill top-k never ranks a row's allocation padding.

DeepGEMM's ``fp8_fp4_mqa_logits`` family allocates ``[rows, aligned_cols]``
and writes only the leading scores of each row; the leaf widens the returned
slice back to the padded rows so the TRT-LLM top-k takes it without a copy.
The padding columns keep whatever the caching allocator last held, finite
values that ``nan_to_num_`` does not scrub, and the TRT-LLM top-k reads
whole padded rows for some launch shapes. These tests fill the caching
allocator with a huge finite score first, so a selection that so much as
reads one padding column stands out: the value outranks every real
candidate.
"""

from __future__ import annotations

import pytest
import torch
from tokenspeed_kernel.platform import current_platform

platform = current_platform()
if not (torch.cuda.is_available() and platform.is_nvidia and platform.is_hopper_plus):
    pytest.skip("the DeepGEMM DSA top-k needs a Hopper+ GPU", allow_module_level=True)

from tokenspeed_kernel._optional import has_optional_package  # noqa: E402

if not has_optional_package("deep_gemm"):
    pytest.skip(
        "the default DSA prefill top-k needs deep_gemm",
        allow_module_level=True,
    )

from tokenspeed_kernel.ops.attention.dsa import dsa_prefill_topk  # noqa: E402
from tokenspeed_kernel.ops.attention.dsa.flashinfer import (  # noqa: E402
    has_deterministic_decode_topk,
)

_HEADS = 32
_DIM = 128
_PAGE = 64
_SLOTS = 4096
_ROWS = 300
_TOPK = 2048
_INITIAL = 16
_LOCAL = 128
# Finite, so nan_to_num_ keeps it, and far above any score a relu dot over
# FP8 queries and keys can reach.
_POISON = 3.0e38


def _lengths() -> list[int]:
    # Varied candidate counts, all above topk so every selected slot must
    # come from a real candidate, and a maximum no power-of-two alignment
    # divides, so the logits allocation always has padding columns.
    lengths = (torch.arange(_ROWS) * 37 % 1952 + 2049).tolist()
    lengths[0] = 4000
    return lengths


def _index_k_cache() -> torch.Tensor:
    """Random finite FP8 keys with unit scales, in the packed row format."""
    keys = torch.randn(_SLOTS, _DIM, device="cuda").to(torch.float8_e4m3fn)
    scales = torch.ones(_SLOTS, dtype=torch.float32, device="cuda")
    return torch.cat(
        [keys.view(torch.uint8), scales.view(torch.uint8).reshape(_SLOTS, 4)], dim=1
    ).contiguous()


def _prefill(lengths: list[int], *, batch_invariant: bool) -> torch.Tensor:
    """Select over a poisoned caching allocator; return CPU selections.

    The inputs are allocated on a drained allocator first; one large freed
    block of ``_POISON`` then services every allocation the call makes, so
    the logits allocation's unwritten padding columns all read ``_POISON``.
    """
    torch.manual_seed(0)
    torch.cuda.empty_cache()
    tokens = len(lengths)
    q = torch.randn(tokens, _HEADS, _DIM, device="cuda", dtype=torch.bfloat16)
    weights = torch.randn(tokens, _HEADS, device="cuda")
    ends = torch.tensor(lengths, dtype=torch.int32, device="cuda")
    cache = _index_k_cache()
    poison = torch.full((32 * 1024 * 1024,), _POISON, device="cuda")
    del poison
    rows, _ = dsa_prefill_topk(
        q,
        weights,
        torch.arange(_SLOTS, dtype=torch.int64, device="cuda"),
        torch.zeros(tokens, dtype=torch.int32, device="cuda"),
        ends,
        topk=_TOPK,
        softmax_scale=1.0,
        batch_invariant=batch_invariant,
        index_k_cache=cache,
        page_size=_PAGE,
        max_logits_bytes=1 << 30,
        candidate_lens_cpu=torch.tensor(lengths, dtype=torch.int64),
        initial_tokens=_INITIAL,
        local_tokens=_LOCAL,
        slot_order="selection",
        solution="deep_gemm",
    )
    return rows.cpu()


def test_prefill_topk_never_selects_columns_past_a_rows_length() -> None:
    lengths = _lengths()
    selected = _prefill(lengths, batch_invariant=False)
    assert selected.shape == (len(lengths), _TOPK)
    for row, length in enumerate(lengths):
        assert selected[row].min() >= 0, f"row {row} has an unwritten slot"
        assert selected[row].max() < length, (
            f"row {row} selected a padding column at or past its length "
            f"{length}: {selected[row].max()}"
        )


def test_prefill_topk_matches_the_row_invariant_selection() -> None:
    if not has_deterministic_decode_topk():
        pytest.skip("needs the lowest-index tie-break top-k")
    lengths = _lengths()
    selected = _prefill(lengths, batch_invariant=False)
    invariant = _prefill(lengths, batch_invariant=True)
    for row, length in enumerate(lengths):
        assert set(selected[row].tolist()) == set(invariant[row].tolist()), (
            f"row {row} (length {length}) selects a different set with the "
            "default top-k than with the row-local one"
        )
