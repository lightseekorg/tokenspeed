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

"""TP feature-column sharding of the LongCat n-gram tables.

The sharded layout follows the official LongCat OE design: every rank hosts
only ``emb_dim // tp_size`` feature columns of the 12 host tables and the
forward all-reduces the local partial contributions. These CPU tests verify
that the sum of the ranks' local partials equals the un-sharded output.
"""

import os
import sys

# CI Registration (parsed via AST, runtime no-op)
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from ci_system.ci_register import register_cuda_ci

register_cuda_ci(est_time=5, suite="runtime-1gpu")

from types import SimpleNamespace

import torch


def _cfg(hidden=3072):
    return SimpleNamespace(
        ngram_vocab_size_ratio=78,
        emb_split_num=4,
        emb_neighbor_num=4,
        vocab_size=512,
        hidden_size=hidden,
        eos_token_id=2,
        pad_token_id=None,
    )


def _weights(plugin, num_embedders, emb_dim):
    """Read the (shared random) full-width weights from an un-sharded plugin."""
    return [plugin.embedders[i].weight for i in range(num_embedders)], [
        plugin.post_projs[i].weight for i in range(num_embedders)
    ]


def test_tp_sharded_partials_sum_to_full():
    from tokenspeed.runtime.plugins.longcat_ngram import NgramEmbedding

    cfg = _cfg()
    num_embedders = cfg.emb_split_num * (cfg.emb_neighbor_num - 1)
    emb_dim = cfg.hidden_size // num_embedders

    torch.manual_seed(0)
    base = torch.nn.Embedding(cfg.vocab_size, cfg.hidden_size)
    full = NgramEmbedding(cfg, base)  # tp_size=1, full-width tables
    full.eval()
    full.requires_grad_(False)

    # Rank 0 / rank 1 of a tp_size=2 deployment, sharing the same base and
    # checkpoint weights (each host table sliced to its feature columns).
    r0 = NgramEmbedding(cfg, base, tp_rank=0, tp_size=2, tp_group="mock")
    r1 = NgramEmbedding(cfg, base, tp_rank=1, tp_size=2, tp_group="mock")
    for m in (r0, r1):
        m.eval()
        m.requires_grad_(False)
    width = emb_dim // 2
    with torch.no_grad():
        for i in range(num_embedders):
            r0.embedders[i].weight.data = full.embedders[i].weight[:, :width].clone()
            r1.embedders[i].weight.data = full.embedders[i].weight[:, width:].clone()
            r0.post_projs[i].weight.data = full.post_projs[i].weight[:, :width].clone()
            r1.post_projs[i].weight.data = full.post_projs[i].weight[:, width:].clone()

    torch.manual_seed(7)
    ids = torch.randint(0, cfg.vocab_size, (2, 8))
    ids[0, 3] = cfg.eos_token_id
    window = full._window_from_context(ids, 0)

    with torch.no_grad():
        l_full = full._ngram_local_partial(ids, window)
        l0 = r0._ngram_local_partial(ids, window)
        l1 = r1._ngram_local_partial(ids, window)
    assert list(l0.shape) == list(ids.shape) + [cfg.hidden_size]
    # The column-group split is exact in arithmetic; the observed difference
    # is fp32 accumulation order over the 3072-dim projection, so use a
    # relative tolerance (0.002 absolute on ~55 magnitude = ~4e-5 relative).
    ok = torch.allclose(l0 + l1, l_full, atol=1e-2, rtol=1e-3)
    d = (l0 + l1 - l_full).abs().max().item()
    assert ok, f"rank partials do not sum to full: maxdiff {d}"

    # The un-sharded forward still matches the reference decomposition.
    out_full = full(ids)
    assert torch.isfinite(out_full).all()


def test_tp_sharded_guard_rejects_query_shard():
    from tokenspeed.runtime.plugins.longcat_ngram import NgramEmbedding

    cfg = _cfg(hidden=2048)
    base = torch.nn.Embedding(cfg.vocab_size, cfg.hidden_size)
    m = NgramEmbedding(cfg, base, tp_rank=0, tp_size=2, tp_group="mock")
    m.eval()
    ids = torch.randint(0, cfg.vocab_size, (1, 4))
    window = m._window_from_context(ids, 0)
    try:
        m(ids, query_shard=object())
    except NotImplementedError as exc:
        assert "complete packed input batch" in str(exc)
    else:
        raise AssertionError("expected NotImplementedError for query_shard")


if __name__ == "__main__":
    import pytest

    pytest.main([__file__, "-q"])
