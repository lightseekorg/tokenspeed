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

"""LongCat n-gram ``HistoryEmbeddingModel`` prefill-graph seam (阶段 II).

The runtime disables the prefill graph for a model whose profile declares
``request_token_history`` unless its inner model implements the official
``HistoryEmbeddingModel`` protocol (``compute_input_embeddings``).  These
tests lock the seam:

  * protocol hit: ``isinstance(inner, HistoryEmbeddingModel)`` is True and
    ``PrefillGraph.input_embedding`` returns the bound method, so the
    ``requires_request_token_history`` disable guard is lifted;
  * dual-source consistency: the view-derived embedding
    (``compute_input_embeddings``, prefill graph path) is bit-identical
    (maxdiff=0) to the window-derived embedding (``forward_from_window``,
    eager/decode path) for the same committed history;
  * append contract: active rows append past their committed frontier,
    padding rows do not, ``committed_lengths`` stays read-only, and writes
    clamp to the history capacity;
  * chunked prefill continuity: across prefill chunks the view grows and a
    later chunk reads the earlier chunks' appended tokens.

The inner model is exercised through ``_RuntimeLongcatNgramModel`` instances
built with ``__new__`` (the protocol methods only touch ``ngram_embeddings``
and ``embed_tokens``), keeping the tests free of the full MLA/MoE decoder
stack, weights and process groups.
"""

import os
import sys

# CI Registration (parsed via AST, runtime no-op)
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from ci_system.ci_register import register_cuda_ci  # noqa: E402

register_cuda_ci(est_time=30, suite="runtime-1gpu")

from types import SimpleNamespace  # noqa: E402

import pytest  # noqa: E402
import torch  # noqa: E402
import torch.nn as nn  # noqa: E402

from tokenspeed.runtime.execution.history_embedding import (  # noqa: E402
    HistoryEmbeddingModel,
)
from tokenspeed.runtime.execution.prefill_graph import input_embedding  # noqa: E402
from tokenspeed.runtime.execution.request_token_history import (  # noqa: E402
    RequestTokenHistoryView,
)
from tokenspeed.runtime.models.registry import ModelRegistry  # noqa: E402
from tokenspeed.runtime.plugins.longcat_ngram import (  # noqa: E402
    LongcatFlashNgramForCausalLM,
    NgramEmbedding,
    _RuntimeLongcatNgramModel,
)

HIDDEN = 48
K, N, VOCAB, RATIO = 4, 4, 256, 2
POOL, CAPACITY = 4, 64


class Cfg:
    ngram_vocab_size_ratio = RATIO
    emb_split_num = K
    emb_neighbor_num = N
    vocab_size = VOCAB
    hidden_size = HIDDEN
    eos_token_id = 2
    pad_token_id = 0
    ngram_context_len = N - 1


def _make_embedding(device):
    torch.manual_seed(0)
    cfg = Cfg()
    word = nn.Embedding(VOCAB, HIDDEN)
    torch.nn.init.normal_(word.weight, std=0.02)
    emb = NgramEmbedding(cfg, word, tp_rank=0, tp_size=1).to(device)
    with torch.no_grad():
        for table in emb._host_embedders:
            table.normal_(std=0.05)
        for proj in emb.post_projs:
            proj.weight.normal_(std=0.02)
    emb.post_projs.to(device)
    emb.eval()
    emb.requires_grad_(False)
    return emb


def _make_inner(device):
    """A real ``_RuntimeLongcatNgramModel`` carrying only the n-gram module.

    ``__new__`` + ``nn.Module.__init__`` skips the decoder-stack constructor;
    the protocol methods only reach ``ngram_embeddings`` (and ``embed_tokens``
    for ``input_embedding``).
    """
    inner = _RuntimeLongcatNgramModel.__new__(_RuntimeLongcatNgramModel)
    nn.Module.__init__(inner)
    inner.ngram_embeddings = _make_embedding(device)
    return inner


def _view(
    *,
    device,
    history=None,
    committed=None,
    req_pool_indices=None,
    input_start_offsets=None,
    active_request_mask=None,
):
    history = (
        torch.zeros(POOL + 1, CAPACITY, dtype=torch.int32, device=device)
        if history is None
        else history
    )
    committed = (
        torch.zeros(POOL + 1, dtype=torch.int32, device=device)
        if committed is None
        else committed
    )
    return RequestTokenHistoryView(
        history_token_ids=history,
        committed_lengths=committed,
        req_pool_indices=req_pool_indices,
        input_start_offsets=input_start_offsets,
        active_request_mask=active_request_mask,
    )


def _reference_window(ids, offsets, slots, committed, history):
    """The serving predecessor window (``prepare_ngram_inputs`` semantics).

    For request row ``r`` covering tokens ``[start, end)`` with committed
    width ``c`` and slot ``s``, the distance-``d`` predecessor of the token
    at local position ``j`` is ``ids[start + j - d]`` when ``j >= d``, else
    ``history[s, c - d + j]``; missing history (before the row start) is a
    dead ``-1`` predecessor.  Newest first, exactly as the decode window path
    reads it.
    """
    bs = slots.numel()
    starts = offsets.tolist()
    window = torch.full((ids.numel(), N - 1), -1, dtype=torch.int64, device=ids.device)
    for r in range(bs):
        start, end = starts[r], starts[r + 1]
        slot = int(slots[r])
        committed_c = int(committed[slot])
        for j in range(end - start):
            for d in range(1, N):
                if j >= d:
                    window[start + j, d - 1] = int(ids[start + j - d])
                else:
                    col = committed_c - d + j
                    window[start + j, d - 1] = (
                        int(history[slot, col]) if col >= 0 else -1
                    )
    return window


@pytest.mark.skipif(not torch.npu.is_available(), reason="requires NPU")
class TestHistoryEmbeddingSeam:
    def test_protocol_hit_lifts_prefill_graph_guard(self):
        """The inner model is a HistoryEmbeddingModel, so the prefill-graph
        ``requires_request_token_history`` guard no longer disables the graph
        (``input_embedding`` returns the bound method instead of None)."""
        torch.npu.set_device(0)
        inner = _make_inner("npu:0")
        inner.embed_tokens = object()  # what PrefillGraph.input_embedding probes
        assert isinstance(inner, HistoryEmbeddingModel)

        model = SimpleNamespace(model=inner)
        compute = input_embedding(model, requires_request_token_history=True)
        assert compute is not None
        assert compute.__self__ is inner

        model_cls, arch = ModelRegistry.resolve_model_cls(
            ["LongcatFlashNgramForCausalLM"]
        )
        assert model_cls is LongcatFlashNgramForCausalLM
        assert arch == "LongcatFlashNgramForCausalLM"
        assert hasattr(model_cls.model_cls, "compute_input_embeddings")
        assert isinstance(
            model_cls.model_cls.__new__(model_cls.model_cls), HistoryEmbeddingModel
        )

    def test_compute_input_embeddings_matches_window_reference(self):
        """View-derived embedding == window-derived embedding for the same
        committed history (dual-source consistency, maxdiff=0)."""
        torch.npu.set_device(0)
        dev = "npu:0"
        inner = _make_inner(dev)
        emb = inner.ngram_embeddings

        history = torch.zeros(POOL + 1, CAPACITY, dtype=torch.int32, device=dev)
        committed = torch.zeros(POOL + 1, dtype=torch.int32, device=dev)
        # Two live requests with committed prefixes, one padding row.
        history[0, :2] = torch.tensor([7, 8], device=dev)
        history[2, :1] = torch.tensor([3], device=dev)
        committed[0] = 2
        committed[2] = 1
        slots = torch.tensor([0, 2, 3], dtype=torch.int64, device=dev)
        offsets = torch.tensor([0, 3, 7, 9], dtype=torch.int32, device=dev)
        active = torch.tensor([True, True, False], dtype=torch.bool, device=dev)
        view = _view(
            device=dev,
            history=history,
            committed=committed,
            req_pool_indices=slots,
            input_start_offsets=offsets,
            active_request_mask=active,
        )
        ids = torch.tensor([10, 11, 12, 20, 21, 22, 23, 99, 100], device=dev)

        ctx = SimpleNamespace(query_shard=None)
        got = inner.compute_input_embeddings(ids, ctx, request_token_history=view)
        assert list(got.shape) == [ids.numel(), HIDDEN]
        assert torch.isfinite(got).all()

        window = _reference_window(ids, offsets, slots, committed, history)
        ref = emb.forward_from_window(ids, window)
        assert list(ref.shape) == [ids.numel(), HIDDEN]
        md = (got.float() - ref.float()).abs().max().item()
        assert md == 0.0, f"view vs window embedding maxdiff={md}"

    def test_append_respects_active_mask_and_frontier(self):
        """Only active rows append, at the committed frontier; the executor
        still owns the frontier (committed_lengths unchanged)."""
        torch.npu.set_device(0)
        dev = "npu:0"
        inner = _make_inner(dev)
        history = torch.zeros(POOL + 1, CAPACITY, dtype=torch.int32, device=dev)
        committed = torch.zeros(POOL + 1, dtype=torch.int32, device=dev)
        history[0, :2] = torch.tensor([7, 8], device=dev)
        history[2, :1] = torch.tensor([3], device=dev)
        committed[0] = 2
        committed[2] = 1
        slots = torch.tensor([0, 2, 3], dtype=torch.int64, device=dev)
        offsets = torch.tensor([0, 3, 7, 9], dtype=torch.int32, device=dev)
        active = torch.tensor([True, True, False], dtype=torch.bool, device=dev)
        view = _view(
            device=dev,
            history=history,
            committed=committed,
            req_pool_indices=slots,
            input_start_offsets=offsets,
            active_request_mask=active,
        )
        ids = torch.tensor([10, 11, 12, 20, 21, 22, 23, 99, 100], device=dev)
        committed_before = committed.clone()

        inner._append_request_token_history(ids, view)

        # Active row 0 (slot 0, committed 2) appends ids[0:3] at [2:5].
        assert history[0, :5].tolist() == [7, 8, 10, 11, 12]
        # Active row 1 (slot 2, committed 1) appends ids[3:7] at [1:5].
        assert history[2, :5].tolist() == [3, 20, 21, 22, 23]
        # Padding row 2 (slot 3) must not append.
        assert history[3, :4].eq(0).all()
        # The frontier is executor-owned: the model must not advance it.
        assert torch.equal(committed, committed_before)

    def test_append_clamps_to_capacity(self):
        """Overflow past the history capacity is dropped, not wrapped."""
        torch.npu.set_device(0)
        dev = "npu:0"
        inner = _make_inner(dev)
        history = torch.zeros(POOL + 1, CAPACITY, dtype=torch.int32, device=dev)
        committed = torch.zeros(POOL + 1, dtype=torch.int32, device=dev)
        committed[0] = CAPACITY - 2  # only two free columns
        slots = torch.tensor([0], dtype=torch.int64, device=dev)
        offsets = torch.tensor([0, 5], dtype=torch.int32, device=dev)
        active = torch.tensor([True], dtype=torch.bool, device=dev)
        view = _view(
            device=dev,
            history=history,
            committed=committed,
            req_pool_indices=slots,
            input_start_offsets=offsets,
            active_request_mask=active,
        )
        ids = torch.tensor([10, 11, 12, 13, 14], device=dev)

        inner._append_request_token_history(ids, view)

        # Only the first two tokens fit; the tail is dropped, never wrapped
        # into the previous row or the padding row.
        assert history[0, CAPACITY - 2 :].tolist() == [10, 11]
        assert history[0, : CAPACITY - 2].eq(0).all()
        assert history[POOL, :].eq(0).all()

    def test_chunked_prefill_history_continuity(self):
        """Across prefill chunks the view grows; a later chunk's embedding
        reads the earlier chunks' appended tokens (maxdiff=0 vs window)."""
        torch.npu.set_device(0)
        dev = "npu:0"
        inner = _make_inner(dev)
        emb = inner.ngram_embeddings
        history = torch.zeros(POOL + 1, CAPACITY, dtype=torch.int32, device=dev)
        committed = torch.zeros(POOL + 1, dtype=torch.int32, device=dev)
        history[0, :2] = torch.tensor([7, 8], device=dev)
        committed[0] = 2
        slots = torch.tensor([0], dtype=torch.int64, device=dev)
        active = torch.tensor([True], dtype=torch.bool, device=dev)
        ctx = SimpleNamespace(query_shard=None)

        chunk1 = torch.tensor([10, 11, 12, 13, 14], device=dev)
        offsets1 = torch.tensor([0, 5], dtype=torch.int32, device=dev)
        view1 = _view(
            device=dev,
            history=history,
            committed=committed,
            req_pool_indices=slots,
            input_start_offsets=offsets1,
            active_request_mask=active,
        )
        got1 = inner.compute_input_embeddings(chunk1, ctx, request_token_history=view1)
        ref1 = emb.forward_from_window(
            chunk1, _reference_window(chunk1, offsets1, slots, committed, history)
        )
        assert (got1.float() - ref1.float()).abs().max().item() == 0.0

        # The executor advances the frontier after the forward.
        committed[0] += 5
        chunk2 = torch.tensor([20, 21, 22, 23], device=dev)
        offsets2 = torch.tensor([0, 4], dtype=torch.int32, device=dev)
        view2 = _view(
            device=dev,
            history=history,
            committed=committed,
            req_pool_indices=slots,
            input_start_offsets=offsets2,
            active_request_mask=active,
        )
        got2 = inner.compute_input_embeddings(chunk2, ctx, request_token_history=view2)
        ref2 = emb.forward_from_window(
            chunk2, _reference_window(chunk2, offsets2, slots, committed, history)
        )
        assert (got2.float() - ref2.float()).abs().max().item() == 0.0
        # Chunk 2's history came from chunk 1's appended tokens.
        assert history[0, 2:7].tolist() == [10, 11, 12, 13, 14]


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, *sys.argv[1:]]))
