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

"""Synthetic AL commits the requested draft prefix, including under graph replay."""

from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import patch

import pytest
import torch

from tokenspeed.runtime.sampling.backends.base import (
    SamplingBackend,
    SamplingBackendConfig,
)

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from ci_system.ci_register import register_cuda_ci

register_cuda_ci(est_time=120, suite="runtime-1gpu")


class _BufferBackend(SamplingBackend):
    _SUPPORTS_SYNTHETIC_ACCEPTANCE = True

    def sample(self, logits_output, sampling_info):
        raise NotImplementedError

    def verify(self, logits_output, sampling_info, candidates):
        raise NotImplementedError


def _config(al, bs, width, device):
    return SamplingBackendConfig(
        enable_block_rejection=False,
        enable_speculative_sampling=False,
        sampling_stream="batch",
        logprob_order="torch",
        synthetic_acceptance_length=al,
        max_bs=bs,
        max_draft_tokens_per_req=width,
        max_req_pool_size=bs + 1,
        vocab_size=512,
        device=device,
        random_seed=48,
    )


def test_config_requires_explicit_synthetic_acceptance():
    with pytest.raises(TypeError, match="synthetic_acceptance_length"):
        SamplingBackendConfig()
    for length in (None, 2.6):
        config = SamplingBackendConfig(
            enable_block_rejection=False,
            synthetic_acceptance_length=length,
            enable_speculative_sampling=False,
            sampling_stream="batch",
            logprob_order="torch",
        )
        assert config.synthetic_acceptance_length == length


@pytest.mark.parametrize("al", [float("nan"), float("inf"), 0.99, 4.01])
def test_reject_invalid_final_width(al):
    with pytest.raises(ValueError, match=rf"{al}.*verify width \[1, 4\]"):
        _BufferBackend(_config(al, 4, 4, "cpu"))


def test_unsupported_backend_rejects_synthetic_acceptance():
    class UnsupportedBackend(_BufferBackend):
        _SUPPORTS_SYNTHETIC_ACCEPTANCE = False

    UnsupportedBackend(_config(None, 4, 4, "cpu"))
    with pytest.raises(ValueError, match="requires the greedy sampling backend"):
        UnsupportedBackend(_config(2.6, 4, 4, "cpu"))


@pytest.mark.parametrize(
    "device",
    [
        "cpu",
        pytest.param(
            "cuda",
            marks=pytest.mark.skipif(
                not torch.cuda.is_available(), reason="requires CUDA"
            ),
        ),
    ],
)
def test_fractional_lengths_and_rng_isolation(device):
    backend = _BufferBackend(_config(3.77, 20000, 5, device))
    repeat = _BufferBackend(_config(3.77, 20000, 5, device))
    sync_mode = torch.cuda.get_sync_debug_mode() if device == "cuda" else None
    try:
        if sync_mode is not None:
            torch.cuda.set_sync_debug_mode("error")
        backend.prepare_step([], [], [], 5)
        torch.manual_seed(991)
        torch.rand(100, device=device)
        repeat.prepare_step([], [], [], 5)
    finally:
        if sync_mode is not None:
            torch.cuda.set_sync_debug_mode(sync_mode)
    torch.testing.assert_close(backend._synthetic_lengths, repeat._synthetic_lengths)
    assert set(backend._synthetic_lengths.tolist()) == {3, 4}
    assert abs(backend._synthetic_lengths.float().mean().item() - 3.77) < 0.015
    first = backend._synthetic_lengths.clone()
    with patch("torch.rand", side_effect=AssertionError("must reuse coin storage")):
        backend.prepare_step([], [], [], 5)
    assert not torch.equal(first, backend._synthetic_lengths)


@pytest.mark.parametrize("width", [1, 4])
def test_prefix_bonus_indices_and_mixed_row_offset(width):
    backend = _BufferBackend(_config(float(width), 5, width, "cpu"))
    candidates = torch.arange(3 * width, dtype=torch.int32).reshape(3, width) + 10
    lengths = torch.tensor([1, max(1, width - 1), width], dtype=torch.int32)
    backend._synthetic_lengths[1:4].copy_(lengths)
    actual_lengths = backend.synthetic_lengths(candidates, 1)
    tokens = torch.empty(3 * width, dtype=torch.int32)
    indices = torch.empty((3, width), dtype=torch.int32)
    accepted = torch.empty(3, dtype=torch.int32)
    bonus = torch.tensor([91, 92, 93])
    backend.write_synthetic_outputs(
        candidates, bonus, actual_lengths, tokens, indices, accepted
    )
    for row, length in enumerate(lengths.tolist()):
        assert tokens.reshape(3, width)[row, :length].tolist() == candidates[
            row, 1:length
        ].tolist() + [bonus[row].item()]
        assert indices[row].tolist() == list(
            range(row * width, row * width + length)
        ) + [-1] * (width - length)
        assert not tokens.reshape(3, width)[row, length:].any()
    torch.testing.assert_close(accepted + 1, lengths)


def test_synthetic_targets_follow_the_target_row_at_the_cutoff():
    torch.manual_seed(5)
    reps, width, vocab = 40000, 3, 6
    probs = torch.zeros(3, width, vocab)
    # Every other position puts all mass on token 5, so reading the wrong
    # position shows up as a 5.
    probs[:, :, 5] = 1.0
    probs[0, 0] = torch.tensor([0.5, 0.0, 0.3, 0.2, 0.0, 0.0])
    probs[1, 1] = torch.tensor([0.0, 0.1, 0.0, 0.0, 0.9, 0.0])
    probs[2, 2] = torch.tensor([0.0, 0.0, 0.0, 1.0, 0.0, 0.0])  # greedy one-hot
    lengths = torch.tensor([1, 2, 3], dtype=torch.int32)
    tokens = SamplingBackend.sample_synthetic_targets(
        probs.repeat(reps, 1, 1), lengths.repeat(reps), torch.rand(3 * reps)
    ).view(reps, 3)
    for row in range(3):
        expected = probs[row, lengths[row] - 1]
        observed = torch.bincount(tokens[:, row], minlength=vocab).float() / reps
        assert observed[expected == 0].sum().item() == 0.0
        torch.testing.assert_close(observed, expected, atol=0.01, rtol=0.0)


def test_synthetic_target_coin_extremes_stay_on_the_support():
    probs = torch.tensor([[[0.0, 0.25, 0.0, 0.75, 0.0, 0.0]]]).repeat(2, 1, 1)
    eps = torch.finfo(torch.float32).eps
    tokens = SamplingBackend.sample_synthetic_targets(
        probs, torch.tensor([1, 1]), torch.tensor([eps, 1.0 - eps])
    )
    assert tokens.tolist() == [1, 3]


def test_disabled_preparation_leaves_synthetic_buffers_unallocated():
    backend = _BufferBackend(_config(None, 5, 4, "cpu"))
    backend.prepare_step([], [], [], 4)
    backend.prepare_capture(5, 4)
    assert backend._synthetic_lengths is None
    assert backend._synthetic_coins is None


@pytest.mark.parametrize("al, allowed_lengths", [(1.0, {1}), (2.6, {2, 3}), (4.0, {4})])
@pytest.mark.parametrize("capture", [False, True])
def test_bootstrap_limit_is_refreshed_for_next_step(al, allowed_lengths, capture):
    backend = _BufferBackend(_config(al, 5, 4, "cpu"))

    def prepare():
        if capture:
            backend.prepare_capture(5, 4)
        else:
            backend.prepare_step([], [], [], 4)

    prepare()
    backend.limit_synthetic_acceptance(torch.tensor([True, False, True]), 1)
    assert backend._synthetic_lengths[1].item() == 1
    assert backend._synthetic_lengths[3].item() == 1
    assert backend._synthetic_lengths[2].item() in allowed_lengths
    prepare()
    assert set(backend._synthetic_lengths.tolist()) <= allowed_lengths


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
@pytest.mark.parametrize("al", [None, 1.0, 2.6, 4.0])
@pytest.mark.parametrize("natural_length", [1, 4])
def test_greedy_verify_and_cuda_graph_replay(al, natural_length):
    from tokenspeed.runtime.layers.logits_processor import LogitsProcessorOutput
    from tokenspeed.runtime.sampling.backends.greedy import (
        GreedySamplingBackend,
        _verify_chain_greedy,
    )
    from tokenspeed.runtime.sampling.sampling_batch_info import SamplingBatchInfo
    from tokenspeed.runtime.sampling.sampling_params import SamplingParams

    backend = GreedySamplingBackend(_config(al, 5, 4, "cuda"))
    params = []
    rids = [f"synthetic-{i}" for i in range(5)]
    for rid in rids:
        sp = SamplingParams(temperature=0.0, top_k=-1, top_p=1.0)
        sp.resolve_seed(rid)
        sp.normalize(None)
        params.append(sp)

    def prepare():
        backend.prepare_step(rids, list(range(5)), params, 4)
        if al is not None:
            backend.limit_synthetic_acceptance(
                torch.tensor([True, False, False, False], device="cuda"), 1
            )

    # Exercise the decode suffix of a mixed batch: row zero belongs to prefill.
    info = SamplingBatchInfo(
        req_pool_indices=torch.arange(1, 5, dtype=torch.int64, device="cuda"),
        vocab_size=512,
        device="cuda",
        batch_row_offset=1,
    )
    candidates = torch.arange(16, dtype=torch.int32, device="cuda").reshape(4, 4) + 10
    logits = torch.full((16, 512), -100.0, device="cuda")
    target_ids = torch.arange(16, device="cuda") + 80
    if natural_length == 4:
        candidates[:, 1:].copy_(target_ids.reshape(4, 4)[:, :3])
    logits.scatter_(1, target_ids[:, None], 100.0)

    def verify():
        return backend.verify(
            LogitsProcessorOutput(next_token_logits=logits.clone()),
            info,
            candidates,
            tree=None,
        )

    def check(tokens, lengths):
        expected = (
            torch.full((4,), natural_length, dtype=torch.int32, device="cuda")
            if al is None
            else backend._synthetic_lengths[1:5]
        )
        torch.testing.assert_close(lengths, expected)
        for row, length in enumerate(lengths.tolist()):
            assert tokens.reshape(4, 4)[row, :length].tolist() == candidates[
                row, 1:length
            ].tolist() + [target_ids.reshape(4, 4)[row, length - 1].item()]
            assert backend._accept_index_buf.view(5, 4)[row].tolist() == list(
                range(row * 4, row * 4 + length)
            ) + [-1] * (4 - length)

    prepare()
    with patch(
        "tokenspeed.runtime.sampling.backends.greedy._verify_chain_greedy",
        wraps=_verify_chain_greedy,
    ) as normal_verify:
        write_outputs = backend.write_synthetic_outputs

        def override(*args, **kwargs):
            # The full-width kernel must run before its outputs are replaced.
            normal_verify.assert_called_once()
            return write_outputs(*args, **kwargs)

        with patch.object(backend, "write_synthetic_outputs", side_effect=override):
            check(*verify())
        normal_verify.assert_called_once()
        assert normal_verify.call_args.kwargs["candidates"].shape == candidates.shape

    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        prepare()
        verify()
    torch.cuda.current_stream().wait_stream(stream)
    graph = torch.cuda.CUDAGraph()
    prepare()
    with torch.cuda.graph(graph):
        output, lengths = verify()
    observed = set()
    for _ in range(12):
        prepare()
        graph.replay()
        check(output, lengths)
        observed.update(lengths.tolist())
    if al == 2.6:
        assert observed == {1, 2, 3}


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
@pytest.mark.parametrize("al", [1.0, 2.6, 4.0])
def test_block_rejection_synthetic_verify_and_cuda_graph_replay(al):
    """Probabilistic-draft synthetic mode on the real flashinfer verify path."""
    from tokenspeed.runtime.layers.logits_processor import LogitsProcessorOutput
    from tokenspeed.runtime.sampling.backends import flashinfer as fi
    from tokenspeed.runtime.sampling.sampling_batch_info import SamplingBatchInfo
    from tokenspeed.runtime.sampling.sampling_params import SamplingParams

    pool, width, vocab = 6, 4, 512
    config = SamplingBackendConfig(
        enable_block_rejection=True,
        enable_speculative_sampling=True,
        sampling_stream="batch",
        logprob_order="torch",
        synthetic_acceptance_length=al,
        max_bs=5,
        max_draft_tokens_per_req=width,
        max_req_pool_size=pool,
        vocab_size=vocab,
        device="cuda",
        random_seed=48,
    )
    backend = fi.FlashInferSamplingBackend(config)
    rids = [f"block-synthetic-{i}" for i in range(5)]
    params = []
    for i, rid in enumerate(rids):
        # Mix sampled and greedy requests in one captured batch.
        sp = SamplingParams(temperature=0.0 if i % 2 else 1.0, top_k=-1, top_p=0.95)
        sp.resolve_seed(rid)
        sp.normalize(None)
        params.append(sp)

    def prepare():
        backend.prepare_step(rids, list(range(1, 6)), params, width)
        backend.limit_synthetic_acceptance(
            torch.tensor([True, False, False, False], device="cuda"), 1
        )

    info = SamplingBatchInfo(
        req_pool_indices=torch.arange(2, 6, dtype=torch.int64, device="cuda"),
        draft_probs=torch.softmax(
            torch.randn(pool + 1, width, vocab, device="cuda"), -1
        ),
        vocab_size=vocab,
        device="cuda",
        batch_row_offset=1,
    )
    torch.manual_seed(3)
    logits = torch.randn(16, vocab, device="cuda") * 4
    support = torch.zeros(16, vocab, dtype=torch.bool, device="cuda")
    support[:, : vocab // 2] = True
    logits.masked_fill_(~support, float("-inf"))
    candidates = torch.randint(0, vocab, (4, width), dtype=torch.int32, device="cuda")

    def verify():
        return backend.verify(
            LogitsProcessorOutput(next_token_logits=logits.clone()),
            info,
            candidates,
            tree=None,
        )

    def check(tokens, lengths):
        torch.testing.assert_close(lengths, backend._synthetic_lengths[1:5])
        assert lengths[0].item() == 1
        for row, length in enumerate(lengths.tolist()):
            row_tokens = tokens.reshape(4, width)[row]
            assert (
                row_tokens[: length - 1].tolist() == candidates[row, 1:length].tolist()
            )
            assert support[row * width + length - 1, row_tokens[length - 1]].item()

    prepare()
    with patch.object(
        fi, "block_speculative_sampling", side_effect=AssertionError("not run")
    ):
        check(*verify())

        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            prepare()
            verify()
        torch.cuda.current_stream().wait_stream(stream)
        graph = torch.cuda.CUDAGraph()
        prepare()
        with torch.cuda.graph(graph):
            output, lengths = verify()
    observed = set()
    for _ in range(12):
        prepare()
        graph.replay()
        check(output, lengths)
        observed.update(lengths[1:].tolist())
    if al == 2.6:
        assert observed == {2, 3}


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
