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

import importlib
import sys

import pytest
import torch


def _is_gfx1250() -> bool:
    if not torch.cuda.is_available():
        return False
    arch = getattr(torch.cuda.get_device_properties(0), "gcnArchName", "")
    return "gfx1250" in arch


if not _is_gfx1250():
    pytest.skip(
        "AMD GFX1250 is required for paged GQA decode tests", allow_module_level=True
    )


def _ensure_tokenspeed_triton_importable() -> None:
    try:
        import tokenspeed_triton  # noqa: F401

        return
    except ModuleNotFoundError:
        pass

    triton = pytest.importorskip("triton")
    sys.modules["tokenspeed_triton"] = triton
    for submodule in (
        "language",
        "language.core",
        "experimental",
        "experimental.gluon",
        "experimental.gluon.language",
        "experimental.gluon.language.amd",
        "experimental.gluon.language.amd.cdna4",
        "experimental.gluon.language.amd.cdna4.async_copy",
        "experimental.gluon.language.amd.gfx1250",
        "experimental.gluon.language.amd.gfx1250.tdm",
    ):
        sys.modules[f"tokenspeed_triton.{submodule}"] = importlib.import_module(
            f"triton.{submodule}"
        )


_ensure_tokenspeed_triton_importable()

from tokenspeed_kernel_amd.ops.attention.gluon.mha_decode_gfx1250 import (  # noqa: E402
    gluon_paged_gqa_decode_gfx1250,
)


def test_paged_gqa_decode_matches_sdpa() -> None:
    cache_seqlens_list = [64, 130, 18, 192]
    batch = len(cache_seqlens_list)
    max_seqlen_k = 256
    num_q_heads = 8
    num_kv_heads = 2
    head_dim = 64
    page_size = 64
    max_pages_per_sequence = max_seqlen_k // page_size

    torch.manual_seed(0)
    q = torch.randn(batch, num_q_heads, head_dim, dtype=torch.bfloat16)
    cache_seqlens = torch.tensor(cache_seqlens_list, dtype=torch.int32)
    pages_per_sequence = [
        (seqlen + page_size - 1) // page_size for seqlen in cache_seqlens_list
    ]
    total_pages = sum(pages_per_sequence)
    k_cache = torch.zeros(
        total_pages, page_size, num_kv_heads, head_dim, dtype=torch.bfloat16
    )
    v_cache = torch.zeros_like(k_cache)
    page_table = torch.zeros(batch, max_pages_per_sequence, dtype=torch.int32)

    expected = []
    next_page = 0
    group_size = num_q_heads // num_kv_heads
    for batch_idx, (seqlen_k, num_pages) in enumerate(
        zip(cache_seqlens_list, pages_per_sequence)
    ):
        k = torch.randn(num_kv_heads, seqlen_k, head_dim, dtype=torch.bfloat16)
        v = torch.randn_like(k)
        physical_pages = torch.arange(
            next_page, next_page + num_pages, dtype=torch.int32
        )
        page_table[batch_idx, :num_pages] = physical_pages
        for page_idx, physical_page in enumerate(physical_pages.tolist()):
            start = page_idx * page_size
            end = min(start + page_size, seqlen_k)
            tokens = end - start
            k_cache[physical_page, :tokens] = k[:, start:end].permute(1, 0, 2)
            v_cache[physical_page, :tokens] = v[:, start:end].permute(1, 0, 2)
        next_page += num_pages

        k_ref = k.repeat_interleave(group_size, dim=0)
        v_ref = v.repeat_interleave(group_size, dim=0)
        expected.append(
            torch.nn.functional.scaled_dot_product_attention(
                q[batch_idx : batch_idx + 1].unsqueeze(2),
                k_ref.unsqueeze(0),
                v_ref.unsqueeze(0),
            ).squeeze(2)
        )
    expected = torch.cat(expected, dim=0)

    actual = gluon_paged_gqa_decode_gfx1250(
        q.cuda(),
        k_cache.cuda(),
        v_cache.cuda(),
        page_table.cuda(),
        cache_seqlens.cuda(),
        max_seqlen_k,
    )

    torch.testing.assert_close(actual.cpu(), expected, rtol=1e-2, atol=1e-2)
