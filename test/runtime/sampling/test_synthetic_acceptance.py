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
    def sample(self, logits_output, sampling_info):
        raise NotImplementedError

    def verify(self, logits_output, sampling_info, candidates):
        raise NotImplementedError


def _config(al, bs, width, device):
    return SamplingBackendConfig(
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
        config = SamplingBackendConfig(synthetic_acceptance_length=length)
        assert config.synthetic_acceptance_length == length


@pytest.mark.parametrize("al", [float("nan"), float("inf"), 0.99, 4.01])
def test_reject_invalid_final_width(al):
    with pytest.raises(ValueError, match="verify width"):
        _BufferBackend(_config(al, 4, 4, "cpu"))


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


@pytest.mark.parametrize("al", [2.6, 4.0])
def test_bootstrap_limit_is_refreshed_for_next_step(al):
    backend = _BufferBackend(_config(al, 5, 4, "cpu"))
    backend.prepare_step([], [], [], 4)
    backend.limit_synthetic_acceptance(torch.tensor([True, False, True]), 1)
    assert backend._synthetic_lengths[1].item() == 1
    assert backend._synthetic_lengths[3].item() == 1
    assert backend._synthetic_lengths[2].item() >= 2
    backend.prepare_step([], [], [], 4)
    assert (backend._synthetic_lengths >= 2).all()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
@pytest.mark.parametrize(
    "name", ["greedy", "triton", "triton_full", "flashinfer", "flashinfer_full"]
)
@pytest.mark.parametrize("al", [None, 1.0, 2.6, 4.0])
@pytest.mark.parametrize("natural_length", [1, 4])
def test_backend_verify_and_cuda_graph_replay(name, al, natural_length):
    from tokenspeed.runtime.layers.logits_processor import LogitsProcessorOutput
    from tokenspeed.runtime.sampling.backends.flashinfer import (
        FlashInferSamplingBackend,
        chain_speculative_sampling_target_only,
    )
    from tokenspeed.runtime.sampling.backends.flashinfer_full import (
        FlashInferFullSamplingBackend,
    )
    from tokenspeed.runtime.sampling.backends.greedy import (
        GreedySamplingBackend,
        _verify_chain_greedy,
    )
    from tokenspeed.runtime.sampling.backends.triton import (
        TritonSamplingBackend,
        verify_chain_target_sampled,
    )
    from tokenspeed.runtime.sampling.backends.triton_full import (
        TritonFullSamplingBackend,
    )
    from tokenspeed.runtime.sampling.sampling_batch_info import SamplingBatchInfo
    from tokenspeed.runtime.sampling.sampling_params import SamplingParams

    classes = {
        "greedy": GreedySamplingBackend,
        "triton": TritonSamplingBackend,
        "triton_full": TritonFullSamplingBackend,
        "flashinfer": FlashInferSamplingBackend,
        "flashinfer_full": FlashInferFullSamplingBackend,
    }
    backend = classes[name](_config(al, 5, 4, "cuda"))
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
            LogitsProcessorOutput(next_token_logits=logits.clone()), info, candidates
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

    if name == "greedy":
        kernel_name, verify_kernel = "_verify_chain_greedy", _verify_chain_greedy
    elif name in ("triton", "triton_full"):
        kernel_name, verify_kernel = (
            "verify_chain_target_sampled",
            verify_chain_target_sampled,
        )
    else:
        kernel_name, verify_kernel = (
            "chain_speculative_sampling_target_only",
            chain_speculative_sampling_target_only,
        )

    prepare()
    with patch(
        f"{classes[name].__module__}.{kernel_name}", wraps=verify_kernel
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


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
