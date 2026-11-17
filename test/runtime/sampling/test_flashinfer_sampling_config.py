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

"""Construct the FlashInfer backend with supported and unsupported verify modes."""

from __future__ import annotations

import os
import sys

import pytest
import torch

sys.path.insert(
    0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
)
from ci_system.ci_register import register_cuda_ci

register_cuda_ci(est_time=5, suite="runtime-1gpu", disabled_on_runners=["amd-*"])

from tokenspeed.runtime.layers.logits_processor import (
    LogitsProcessorOutput,
)
from tokenspeed.runtime.sampling.backends import flashinfer as fi
from tokenspeed.runtime.sampling.backends.base import (
    SamplingBackendConfig,
)
from tokenspeed.runtime.sampling.sampling_batch_info import (
    SamplingBatchInfo,
)
from tokenspeed.runtime.utils.env import global_server_args_dict


def _config(
    *,
    sampled: bool,
    stream: str,
    synthetic_length: float | None,
) -> SamplingBackendConfig:
    return SamplingBackendConfig(
        synthetic_acceptance_length=synthetic_length,
        enable_speculative_sampling=sampled,
        sampling_stream=stream,
        logprob_order="torch",
        max_bs=2,
        max_draft_tokens_per_req=4,
        max_req_pool_size=2,
        vocab_size=8,
        device="cpu",
    )


@pytest.fixture(autouse=True)
def _cpu_constructor(monkeypatch):
    # Construction allocates ordinary CPU tensors; only the native renorm
    # stream preparation needs CUDA, and these tests never launch its kernels.
    monkeypatch.setattr(fi, "fused_topk_topp_prepare", lambda device: None)


@pytest.mark.parametrize(
    "stream, method, synthetic_length",
    [
        ("batch", "standard", None),
        ("per-request", "block", None),
        ("per-request", "synthetic", None),
        ("per-request", "standard", 2.0),
    ],
)
def test_sampled_adaptive_dspark_rejects_unsupported_modes(
    monkeypatch, stream, method, synthetic_length
):
    monkeypatch.setitem(global_server_args_dict, "enable_dsv41_adaptive_dspark", True)
    monkeypatch.setitem(
        global_server_args_dict, "speculative_rejection_sample_method", method
    )
    with pytest.raises(ValueError, match="sampled adaptive DSpark requires"):
        fi.FlashInferSamplingBackend(
            _config(sampled=True, stream=stream, synthetic_length=synthetic_length)
        )


@pytest.mark.parametrize(
    "adaptive, sampled, stream, method, synthetic_length",
    [
        (True, True, "per-request", "standard", None),
        (False, True, "batch", "block", None),
        (False, True, "batch", "synthetic", 2.0),
        (True, False, "batch", "block", None),
    ],
)
def test_flashinfer_preserves_supported_construction_modes(
    monkeypatch, adaptive, sampled, stream, method, synthetic_length
):
    monkeypatch.setitem(
        global_server_args_dict, "enable_dsv41_adaptive_dspark", adaptive
    )
    monkeypatch.setitem(
        global_server_args_dict, "speculative_rejection_sample_method", method
    )
    fi.FlashInferSamplingBackend(
        _config(sampled=sampled, stream=stream, synthetic_length=synthetic_length)
    )


def test_flashinfer_requires_adaptive_dspark_configuration(monkeypatch):
    monkeypatch.delitem(global_server_args_dict, "enable_dsv41_adaptive_dspark")
    with pytest.raises(KeyError, match="enable_dsv41_adaptive_dspark"):
        fi.FlashInferSamplingBackend(
            _config(sampled=True, stream="per-request", synthetic_length=None)
        )


def test_flashinfer_rejects_packed_offsets_outside_sampled_adaptive_dspark(monkeypatch):
    monkeypatch.setitem(global_server_args_dict, "enable_dsv41_adaptive_dspark", False)
    monkeypatch.setitem(
        global_server_args_dict, "speculative_rejection_sample_method", "standard"
    )
    backend = fi.FlashInferSamplingBackend(
        _config(sampled=True, stream="per-request", synthetic_length=None)
    )
    with pytest.raises(ValueError, match="requires sampled adaptive DSpark"):
        backend.verify(
            LogitsProcessorOutput(next_token_logits=torch.zeros(2, 8)),
            SamplingBatchInfo(
                req_pool_indices=torch.tensor([1]),
                valid_cache_lengths=torch.zeros(3, dtype=torch.int32),
                vocab_size=8,
                device="cpu",
            ),
            torch.zeros((1, 4), dtype=torch.int32),
            tree=None,
            query_start_loc=torch.tensor([0, 2], dtype=torch.int32),
        )


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, *sys.argv[1:]]))
