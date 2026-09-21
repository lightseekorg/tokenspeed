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

"""The verify-output triple lives in one buffer and syncs with one broadcast."""

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

from tokenspeed.runtime.sampling.backends import base
from tokenspeed.runtime.sampling.backends.base import SamplingBackendConfig
from tokenspeed.runtime.sampling.backends.greedy import GreedySamplingBackend

MAX_BS, MAX_N = 4, 6


def _backend(device: str) -> GreedySamplingBackend:
    return GreedySamplingBackend(
        SamplingBackendConfig(
            max_bs=MAX_BS,
            max_draft_tokens_per_req=MAX_N,
            max_req_pool_size=8,
            vocab_size=64,
            device=torch.device(device),
        )
    )


def _pack_regions(backend) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    pack = backend._output_pack_buf
    predict_max = backend._predict_max
    return (
        pack[:predict_max],
        pack[predict_max : predict_max + MAX_BS],
        pack[predict_max + MAX_BS :],
    )


def test_verify_outputs_are_carved_from_one_packed_buffer() -> None:
    backend = _backend("cpu")
    assert backend._predict_max == MAX_BS * MAX_N
    assert backend._output_pack_buf.shape == (2 * MAX_BS * MAX_N + MAX_BS,)
    predict, lengths, index = _pack_regions(backend)
    for view, region in (
        (backend._predict_buf, predict),
        (backend._accept_length_buf, lengths),
        (backend._accept_index_buf, index),
    ):
        assert view.data_ptr() == region.data_ptr() and view.shape == region.shape
    # Re-carving for a padded batch keeps the layout and the aliases.
    backend._allocate_verify_outputs(2 * MAX_BS, MAX_N)
    assert backend._output_pack_buf.shape == (4 * MAX_BS * MAX_N + 2 * MAX_BS,)
    assert backend._accept_index_buf.shape == (2 * MAX_BS * MAX_N,)
    assert (
        backend._accept_length_buf.data_ptr()
        == backend._output_pack_buf[2 * MAX_BS * MAX_N :].data_ptr()
    )


def test_broadcast_verify_outputs_sends_the_whole_pack_once(monkeypatch) -> None:
    backend = _backend("cpu")
    calls = []
    monkeypatch.setattr(
        base.dist,
        "broadcast",
        lambda tensor, src, group: calls.append(
            (tensor.data_ptr(), tensor.numel(), src, group)
        ),
    )
    backend.broadcast_verify_outputs()
    assert calls == []
    backend._tp_pg = group = SimpleNamespace()
    backend._tp_src_global_rank = 3
    backend.broadcast_verify_outputs()
    pack = backend._output_pack_buf
    assert calls == [(pack.data_ptr(), pack.numel(), 3, group)]


@pytest.mark.skipif(not torch.cuda.is_available(), reason="verify kernels need CUDA")
def test_verify_lands_all_three_outputs_in_the_pack(monkeypatch) -> None:
    backend = _backend("cuda")
    sent = []
    monkeypatch.setattr(
        base.dist, "broadcast", lambda tensor, src, group: sent.append(tensor)
    )
    backend._tp_pg = SimpleNamespace()
    backend._tp_src_global_rank = 0
    bs, n = 2, 3
    logits = torch.full((bs * n, 64), -1.0, device="cuda")
    logits[torch.arange(bs * n), torch.arange(bs * n) + 10] = 5.0
    # Row 0's drafts match the target at every step; row 1's second draft does not.
    candidates = torch.tensor([[7, 10, 11], [8, 13, 99]], device="cuda")
    predict, lengths = backend.verify(
        SimpleNamespace(next_token_logits=logits),
        SimpleNamespace(vocab_mask=None, req_pool_indices=None),
        candidates,
    )
    _, _, index = _pack_regions(backend)
    assert predict.data_ptr() == backend._output_pack_buf.data_ptr()
    assert lengths.data_ptr() == backend._accept_length_buf.data_ptr()
    assert predict.tolist() == [10, 11, 12, 13, 14, 0]
    assert lengths.tolist() == [3, 2]
    # accept_index is an offset table into the flat predict rows.
    assert index[: bs * n].view(bs, n).tolist() == [[0, 1, 2], [3, 4, -1]]
    assert len(sent) == 1 and sent[0].data_ptr() == backend._output_pack_buf.data_ptr()
