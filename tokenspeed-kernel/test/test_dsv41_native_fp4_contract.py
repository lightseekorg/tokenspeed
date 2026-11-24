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

"""CPU/mock contracts for the opt-in SM120 dense-only native Indexer."""

import sys
import types

import pytest
import torch
from tokenspeed_kernel.ops.attention.dsv41 import deep_gemm as impl


def _case(tokens=2, *, broadcast=True):
    q = torch.zeros(tokens, 32, 128, dtype=torch.bfloat16)
    weights = torch.zeros(tokens, 32, dtype=torch.float32)
    cache = torch.zeros(2, 64, 68, dtype=torch.uint8)
    table = torch.arange(2, dtype=torch.int32)[None, :]
    table = table.expand(tokens, -1)
    if not broadcast:
        table = table.clone()
    visible = torch.full((tokens,), 128, dtype=torch.int32)
    return q, weights, cache, table, visible


def _eligible(
    args,
    candidate_blocks=None,
    candidate_topk=0,
    candidate_block_size=8,
    topk=512,
    process_group=None,
):
    q, _, _, table, _ = args
    return impl._sm120_dense_full_eligible(
        q,
        table,
        candidate_blocks,
        candidate_topk,
        candidate_block_size,
        topk,
        process_group,
    )


def test_dense_full_requires_explicit_opt_in(monkeypatch):
    # Simulate only the absent optional vendor package's discovery on CPU.
    monkeypatch.setattr(
        impl, "has_optional_package", lambda name: name == "deep_gemm_sm120"
    )
    args = _case()
    monkeypatch.setattr(impl, "_SM120_NATIVE", False)
    assert not _eligible(args)
    assert not impl.is_sm120_native_indexer_available()
    monkeypatch.setattr(impl, "_SM120_NATIVE", True)
    assert _eligible(args)
    assert impl.is_sm120_native_indexer_available()


@pytest.mark.parametrize(
    "kind",
    [
        "candidate_blocks",
        "candidate_topk",
        "nonbroadcast",
        "single_query",
        "process_group",
        "candidate_block_size",
        "topk",
    ],
)
def test_dense_full_rejects_every_other_contract(monkeypatch, kind):
    monkeypatch.setattr(impl, "_SM120_NATIVE", True)
    args = _case(
        tokens=1 if kind == "single_query" else 2, broadcast=kind != "nonbroadcast"
    )
    kwargs = {
        "candidate_blocks": (
            torch.zeros(args[0].shape[0], 1, dtype=torch.int32)
            if kind == "candidate_blocks"
            else None
        ),
        "candidate_topk": 17 if kind == "candidate_topk" else 0,
        "candidate_block_size": 4 if kind == "candidate_block_size" else 8,
        "topk": 256 if kind == "topk" else 512,
        "process_group": object() if kind == "process_group" else None,
    }
    assert not _eligible(args, **kwargs)


@pytest.mark.parametrize(
    "kind",
    [
        "candidate_blocks",
        "candidate_topk",
        "nonbroadcast",
        "single_query",
        "process_group",
        "empty_query",
    ],
)
def test_non_dense_calls_existing_bf16_fallback(monkeypatch, kind):
    monkeypatch.setattr(impl, "_SM120_NATIVE", True)
    args = _case(
        tokens=1 if kind == "single_query" else (0 if kind == "empty_query" else 2),
        broadcast=kind != "nonbroadcast",
    )
    q, weights, cache, table, visible = args
    candidates = (
        torch.zeros(q.shape[0], 1, dtype=torch.int32)
        if kind == "candidate_blocks"
        else None
    )
    candidate_topk = 17 if kind == "candidate_topk" else 0
    process_group = object() if kind == "process_group" else None
    out = object()
    metadata = object()
    calls = []
    fallback = types.ModuleType(
        "tokenspeed_kernel.ops.attention.dsv41.flashinfer_index_sm120"
    )

    def record(*values):
        calls.append(values)
        return "bf16-fallback"

    fallback.flashinfer_sm120_dsv41_index_topk = record
    monkeypatch.setitem(sys.modules, fallback.__name__, fallback)
    result = impl.index_topk(
        q,
        weights,
        cache,
        table,
        visible,
        candidates,
        512,
        candidate_topk,
        8,
        256,
        256,
        process_group,
        out,
        metadata,
        None,
    )
    assert result == "bf16-fallback"
    assert len(calls) == 1
    values = calls[0]
    assert values[0] is q and values[1] is weights and values[2] is cache
    assert values[3] is table and values[4] is visible and values[5] is candidates
    assert values[6:10] == (512, candidate_topk, 8, 256)
    assert values[10] == 256 and values[11] is process_group
    assert values[12] is out and values[13] is metadata
