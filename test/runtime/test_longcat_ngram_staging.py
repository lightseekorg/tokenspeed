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

"""LongCat n-gram device staging preload (方案C L1).

The ACL-Graph decode path preloads the current batch's host-table rows H2D
into a fixed device ``staging`` buffer *outside* the captured graph, and the
in-graph forward reads only the staging.  These tests verify the L1-1
contract: (1) the preloaded staging content is bit-identical to the original
host lookup ``F.embedding(new_ids_cpu, host_weight)``; (2) the staging-path
``_ngram_local_partial`` output is bit-identical to the host-lookup path;
(3) the eager baseline (``use_staging=False``) still uses the original host
path; (4) batch overflow raises instead of silently truncating.

Staging allocation is decoupled from capture (方案C L1 fix): ``_ensure_staging``
allocates the buffer on module placement, so ``staging`` may be non-None even
on the eager baseline — the ``use_staging`` flag, not ``staging is None``,
selects the lookup path.
"""

import os
import sys

# CI Registration (parsed via AST, runtime no-op)
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from ci_system.ci_register import register_cuda_ci

register_cuda_ci(est_time=20, suite="runtime-1gpu")

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


def _build_pair(cfg, device, seed=0):
    """Two NgramEmbedding instances with identical weights on ``device``.

    Returns ``(host_mod, staging_mod)`` where ``host_mod.use_staging`` stays
    False (original host-lookup path) and ``staging_mod`` gets preloaded with
    ``use_staging=True``.
    """
    torch.manual_seed(seed)
    base = torch.nn.Embedding(cfg.vocab_size, cfg.hidden_size, device=device)
    host_mod = NgramEmbedding(cfg, base).to(device)
    staging_mod = NgramEmbedding(
        cfg, torch.nn.Embedding(cfg.vocab_size, cfg.hidden_size, device=device)
    ).to(device)
    # Share identical weights (embedder tables live on CPU either way).
    with torch.no_grad():
        staging_mod.word_embeddings.weight.data.copy_(
            host_mod.word_embeddings.weight.data
        )
        for i in range(cfg.emb_split_num * (cfg.emb_neighbor_num - 1)):
            staging_mod.embedders[i].weight.data.copy_(
                host_mod.embedders[i].weight.data
            )
            staging_mod.post_projs[i].weight.data.copy_(
                host_mod.post_projs[i].weight.data
            )
    host_mod.eval()
    staging_mod.eval()
    host_mod.requires_grad_(False)
    staging_mod.requires_grad_(False)
    return host_mod, staging_mod


def _rand_window(ids, n, eos, device):
    """Random predecessor window with a few -1 / EOS entries to exercise the
    EOS reset and dead-predecessor paths of the polynomial hash."""
    window = torch.randint(0, 512, (ids.numel(), n - 1), device=device)
    window[0, 0] = -1
    window[ids.numel() // 2, 1] = eos
    return window


def test_staging_preload_matches_host_lookup():
    """preload_staging rows == F.embedding(new_ids_cpu, host_weight) bit-exact."""
    cfg = _cfg()
    host_mod, staging_mod = _build_pair(cfg, "cpu")
    ids = torch.randint(0, cfg.vocab_size, (16,))
    window = _rand_window(ids, cfg.emb_neighbor_num, cfg.eos_token_id, "cpu")

    staging_mod.preload_staging(ids, window)
    assert staging_mod.staging is not None
    assert list(staging_mod.staging.shape) == [
        160,
        cfg.emb_split_num * (cfg.emb_neighbor_num - 1),
        256,
    ]

    num_embedders = cfg.emb_split_num * (cfg.emb_neighbor_num - 1)
    for index in range(num_embedders):
        new_ids = staging_mod._compute_new_ids(ids, window, index)
        host_rows = torch.nn.functional.embedding(
            new_ids, host_mod._host_embedders[index]
        ).reshape(-1, 256)
        staged = staging_mod.staging[: ids.numel(), index, :]
        assert torch.equal(staged, host_rows), f"embedder {index}: staging row mismatch"
    assert True  # 12/12 tables bit-exact


def test_staging_partial_matches_host_partial_bit_exact():
    """Staging-path _ngram_local_partial output == host-lookup path (maxdiff=0)."""
    cfg = _cfg()
    host_mod, staging_mod = _build_pair(cfg, "cpu")
    ids = torch.randint(0, cfg.vocab_size, (2, 8))
    window = host_mod._window_from_context(ids, 0)

    with torch.no_grad():
        host_out = host_mod._ngram_local_partial(ids, window)
        staging_mod.use_staging = True
        staging_mod.preload_staging(ids, window)
        staging_out = staging_mod._ngram_local_partial(ids, window)
    assert list(staging_out.shape) == list(ids.shape) + [cfg.hidden_size]
    assert torch.equal(
        staging_out, host_out
    ), "staging path and host path must be bit-identical (maxdiff=0)"
    md = (staging_out.float() - host_out.float()).abs().max().item()
    assert md == 0.0, f"maxdiff={md}"


def test_eager_baseline_keeps_host_lookup():
    """With use_staging=False the original host-lookup path still runs (eager)."""
    cfg = _cfg()
    host_mod, staging_mod = _build_pair(cfg, "cpu")
    ids = torch.randint(0, cfg.vocab_size, (2, 8))
    window = host_mod._window_from_context(ids, 0)
    assert staging_mod.use_staging is False
    with torch.no_grad():
        out = staging_mod(ids)  # forward without staging -> host path
        ref = host_mod(ids)
    assert out.shape == (2, 8, cfg.hidden_size)
    assert torch.isfinite(out).all()
    # The eager baseline must stay bit-identical to the host reference even
    # though the staging buffer itself may already be allocated (allocation
    # is decoupled from the lookup-path switch).
    assert torch.equal(out, ref), "eager forward must match host reference"
    assert staging_mod.use_staging is False


def test_use_staging_requires_allocated_staging():
    """use_staging=True with no staging allocated must raise, not fall back."""
    cfg = _cfg()
    host_mod, staging_mod = _build_pair(cfg, "cpu")
    ids = torch.randint(0, cfg.vocab_size, (2, 8))
    window = host_mod._window_from_context(ids, 0)
    # Simulate a capture where the executor forgot to pre-allocate: force the
    # buffer away and keep the switch on — the staging branch must raise.
    staging_mod.staging = None
    staging_mod.use_staging = True
    with pytest.raises(RuntimeError):
        staging_mod._ngram_local_partial(ids, window)


def test_ensure_staging_idempotent_and_device_placed():
    """_ensure_staging allocates once, on the module's device."""
    cfg = _cfg()
    _, mod = _build_pair(cfg, "cpu")
    mod.staging = None
    mod._ensure_staging()
    first = mod.staging
    assert first is not None
    assert first.device.type == "cpu"
    mod._ensure_staging()
    assert mod.staging is first  # idempotent, no re-allocation


def test_preload_overflow_raises():
    """T > max_num_seqs must raise, not silently truncate."""
    cfg = _cfg()
    _, staging_mod = _build_pair(cfg, "cpu", seed=3)
    staging_mod._max_num_seqs = 4
    ids = torch.randint(0, cfg.vocab_size, (8,))
    window = _rand_window(ids, cfg.emb_neighbor_num, cfg.eos_token_id, "cpu")
    with pytest.raises(ValueError):
        staging_mod.preload_staging(ids, window)


if __name__ == "__main__":
    pytest.main([__file__, "-q"])
