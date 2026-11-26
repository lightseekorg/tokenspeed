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

"""LongCat n-gram staging decode-graph capture (方案C L1, L1-2).

Simulates a small decode step: a packed batch ``input_ids [T]`` plus its
per-token predecessor window are preloaded H2D into the device staging
*outside* the graph, and an ``torch.npu.NPUGraph`` capture records the
``forward_from_window`` path (including the in-graph ``_ngram_local_partial``
staging lookup).  Two replay rounds with refreshed input/staging must be
bit-identical to the eager host-lookup reference (maxdiff=0).

The second test covers the engine probe path (方案C L1 fix regression): the
boot's probe capture runs a dummy decode through ``NgramEmbedding.forward``
(the ``[bs, seq]`` 2-D entry the wrapper reaches when it drops the n-gram
kwargs) — with ``use_staging=True`` and the staging allocated the captured
forward must read the staging and replay 2 rounds maxdiff=0, never touching
the host tables inside the graph.
"""

import os
import sys

# CI Registration (parsed via AST, runtime no-op)
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from ci_system.ci_register import register_cuda_ci

register_cuda_ci(est_time=30, suite="runtime-1gpu")

from types import SimpleNamespace

import pytest
import torch

from tokenspeed.runtime.plugins.longcat_ngram import NgramEmbedding


def _cfg(hidden=3072, vocab=512):
    return SimpleNamespace(
        ngram_vocab_size_ratio=78,
        emb_split_num=4,
        emb_neighbor_num=4,
        vocab_size=vocab,
        hidden_size=hidden,
        eos_token_id=2,
        pad_token_id=None,
    )


def _fresh_window(ids, n, eos, vocab, device):
    """Predecessor window mixing -1 / EOS / normal ids (exercises the hash
    EOS-reset and dead-predecessor branches on every round)."""
    window = torch.randint(0, vocab, (ids.numel(), n - 1), device=device)
    window[0, 0] = -1
    window[ids.numel() // 2, 1] = eos
    return window


def _build_graph_ref_pair(cfg, dev):
    """Graph-side module (staging path) and host-side reference (host path),
    identical weights, everything but the CPU host tables on ``dev``."""
    torch.manual_seed(0)
    with torch.device(dev):
        base_g = torch.nn.Embedding(cfg.vocab_size, cfg.hidden_size)
        base_r = torch.nn.Embedding(cfg.vocab_size, cfg.hidden_size)
        m_graph = NgramEmbedding(cfg, base_g).to(dev)
        m_ref = NgramEmbedding(cfg, base_r).to(dev)
    with torch.no_grad():
        m_ref.word_embeddings.weight.data.copy_(m_graph.word_embeddings.weight.data)
        for i in range(cfg.emb_split_num * (cfg.emb_neighbor_num - 1)):
            m_ref.embedders[i].weight.data.copy_(m_graph.embedders[i].weight.data)
            m_ref.post_projs[i].weight.data.copy_(m_graph.post_projs[i].weight.data)
    m_graph.eval()
    m_ref.eval()
    m_graph.requires_grad_(False)
    m_ref.requires_grad_(False)
    return m_graph, m_ref


@pytest.mark.skipif(not torch.npu.is_available(), reason="requires NPU")
def test_decode_graph_staging_replays_match_host_reference():
    dev = "npu:0"
    cfg = _cfg()
    hidden = cfg.hidden_size
    T = 8  # decode batch (bs)

    m_graph, m_ref = _build_graph_ref_pair(cfg, dev)

    # Address-stable device buffers the captured graph reads.
    ids_buf = torch.randint(0, cfg.vocab_size, (T,), device=dev)
    win_buf = _fresh_window(
        ids_buf, cfg.emb_neighbor_num, cfg.eos_token_id, cfg.vocab_size, dev
    )
    out_buf = torch.empty((T, hidden), dtype=torch.float32, device=dev)

    def graph_body():
        out_buf.copy_(m_graph.forward_from_window(ids_buf, win_buf))

    # The staging path is on: the in-graph forward reads the preloaded rows.
    m_graph.use_staging = True

    # Warm-up run (eager) so the captured kernels are warm, then capture.
    with torch.npu.stream(torch.npu.current_stream()):
        m_graph.preload_staging(ids_buf, win_buf)
        graph_body()
        torch.npu.synchronize()

    g = torch.npu.NPUGraph()
    with torch.npu.stream(torch.npu.current_stream()):
        with torch.npu.graph(g):
            graph_body()
        torch.npu.synchronize()

    for round_ in range(2):
        ids = torch.randint(0, cfg.vocab_size, (T,), device=dev)
        window = _fresh_window(
            ids, cfg.emb_neighbor_num, cfg.eos_token_id, cfg.vocab_size, dev
        )
        ids_buf.copy_(ids)
        win_buf.copy_(window)
        # Graph-external refresh: preload the new batch's rows H2D.
        m_graph.preload_staging(ids_buf, win_buf)
        with torch.no_grad():
            ref = m_ref.forward_from_window(ids_buf, win_buf).float()
        g.replay()
        torch.npu.synchronize()
        got = out_buf.float()
        assert torch.equal(
            got, ref
        ), f"round {round_}: graph replay vs eager host reference mismatch"
        md = (got - ref).abs().max().item()
        assert md == 0.0, f"round {round_}: maxdiff={md}"


@pytest.mark.skipif(not torch.npu.is_available(), reason="requires NPU")
def test_probe_forward_2d_staging_replays_match_host_reference():
    """Engine probe path: a dummy decode through ``NgramEmbedding.forward``
    (the ``[bs, seq]`` 2-D entry) with staging allocated + use_staging=True
    must replay 2 rounds maxdiff=0 vs the host reference (no host memcpy
    inside the captured graph — the 107030 regression)."""
    dev = "npu:0"
    cfg = _cfg()
    hidden = cfg.hidden_size
    bs, seq = 1, 8

    m_graph, m_ref = _build_graph_ref_pair(cfg, dev)
    m_graph.use_staging = True

    ids_buf = torch.randint(0, cfg.vocab_size, (bs, seq), device=dev)
    out_buf = torch.empty((bs, seq, hidden), dtype=torch.float32, device=dev)

    def graph_body():
        out_buf.copy_(m_graph.forward(ids_buf))

    # Preload the same (input, in-input window) pair the in-graph forward
    # hashes — this is what the engine's capture-time preload guarantees.
    window = m_graph._window_from_context(ids_buf, 0)
    m_graph.preload_staging(ids_buf, window)

    with torch.npu.stream(torch.npu.current_stream()):
        graph_body()
        torch.npu.synchronize()

    g = torch.npu.NPUGraph()
    with torch.npu.stream(torch.npu.current_stream()):
        with torch.npu.graph(g):
            graph_body()
        torch.npu.synchronize()

    for round_ in range(2):
        ids = torch.randint(0, cfg.vocab_size, (bs, seq), device=dev)
        ids_buf.copy_(ids)
        window = m_graph._window_from_context(ids_buf, 0)
        m_graph.preload_staging(ids_buf, window)
        with torch.no_grad():
            ref = m_ref.forward(ids_buf).float()
        g.replay()
        torch.npu.synchronize()
        got = out_buf.float()
        assert torch.equal(
            got, ref
        ), f"round {round_}: probe 2-D graph replay vs eager host reference mismatch"
        md = (got - ref).abs().max().item()
        assert md == 0.0, f"round {round_}: maxdiff={md}"


if __name__ == "__main__":
    pytest.main([__file__, "-q"])
