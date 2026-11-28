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

"""Verify outputs and request_ends pack cleanly for TP sync and single pinned D2H."""

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
            synthetic_acceptance_length=None,
            enable_speculative_sampling=False,
            sampling_stream="batch",
            logprob_order="torch",
            max_bs=MAX_BS,
            max_draft_tokens_per_req=MAX_N,
            max_req_pool_size=8,
            vocab_size=64,
            device=torch.device(device),
        )
    )


def _pack_regions(
    backend,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    pack = backend._output_pack_buf
    predict_max = backend._predict_max
    rows = backend._pack_rows
    return (
        pack[:predict_max],
        pack[predict_max : predict_max + rows],
        pack[predict_max + rows : predict_max + 2 * rows],
        pack[predict_max + 2 * rows :],
    )


def test_verify_outputs_are_carved_from_one_packed_buffer() -> None:
    backend = _backend("cpu")
    assert backend._predict_max == MAX_BS * MAX_N
    assert backend._output_pack_buf.shape == (2 * MAX_BS * MAX_N + 2 * MAX_BS,)
    predict, lengths, request_ends, index = _pack_regions(backend)
    for view, region in (
        (backend._predict_buf, predict),
        (backend._accept_length_buf, lengths),
        (backend.request_ends_buf, request_ends),
        (backend._accept_index_buf, index),
    ):
        assert view.data_ptr() == region.data_ptr() and view.shape == region.shape
    # Re-carving for a padded batch keeps the layout and the aliases.
    backend._allocate_verify_outputs(2 * MAX_BS, MAX_N)
    assert backend._output_pack_buf.shape == (4 * MAX_BS * MAX_N + 4 * MAX_BS,)
    assert backend._accept_index_buf.shape == (2 * MAX_BS * MAX_N,)
    assert (
        backend._accept_length_buf.data_ptr()
        == backend._output_pack_buf[2 * MAX_BS * MAX_N :].data_ptr()
    )
    assert (
        backend.request_ends_buf.data_ptr()
        == backend._output_pack_buf[2 * MAX_BS * MAX_N + 2 * MAX_BS :].data_ptr()
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
        tree=None,
    )
    _, _, _, index = _pack_regions(backend)
    assert predict.data_ptr() == backend._output_pack_buf.data_ptr()
    assert lengths.data_ptr() == backend._accept_length_buf.data_ptr()
    assert predict.tolist() == [10, 11, 12, 13, 14, 0]
    assert lengths.tolist() == [3, 2]
    # accept_index is an offset table into the flat predict rows.
    assert index[: bs * n].view(bs, n).tolist() == [[0, 1, 2], [3, 4, -1]]
    assert len(sent) == 1 and sent[0].data_ptr() == backend._output_pack_buf.data_ptr()


def test_get_packed_outputs_and_ends_d2h_reuses_host_ring() -> None:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    backend = _backend(device)
    tokens = backend._predict_buf[:4]
    lengths = backend._accept_length_buf[:2]
    ends = backend.request_ends_buf[:2]

    tokens.copy_(torch.tensor([11, 12, 13, 14], dtype=torch.int32, device=device))
    lengths.copy_(torch.tensor([2, 1], dtype=torch.int32, device=device))
    ends.copy_(torch.tensor([1, 0], dtype=torch.int32, device=device))

    held = backend.get_packed_outputs_and_ends_d2h(tokens, lengths, ends)
    assert held is not None
    if device == "cuda":
        torch.cuda.synchronize()
    assert held[0].tolist() == [11, 12, 13, 14]
    assert held[1].tolist() == [2, 1]
    assert held[2].tolist() == [1, 0]

    # Overwriting device buffers and copying again must not corrupt held outputs.
    tokens.fill_(99)
    lengths.fill_(3)
    ends.fill_(2)
    second = backend.copy_outputs_d2h(tokens, lengths, ends)
    if device == "cuda":
        torch.cuda.synchronize()
    assert held[0].tolist() == [11, 12, 13, 14]
    assert second[0].tolist() == [99, 99, 99, 99]
    assert second[2].tolist() == [2, 2]
    del held, second

    # After outputs are released, repeated steps reuse ring slots with zero new allocs.
    base_allocs = backend._d2h_host_ring.alloc_count
    for step in range(12):
        tokens.fill_(step)
        lengths.fill_(1)
        ends.fill_(step & 1)
        out = backend.copy_outputs_d2h(tokens, lengths, ends)
        if device == "cuda":
            torch.cuda.synchronize()
        assert out[0].tolist() == [step] * 4
        assert out[2].tolist() == [step & 1, step & 1]
        del out
    assert backend._d2h_host_ring.alloc_count <= base_allocs + 1

    # Unpacked or non-contiguous / mismatched-dtype views decline the pack fast path cleanly.
    assert backend.get_packed_output_d2h(backend._predict_buf[:4:2], lengths) is None
    assert (
        backend.get_packed_outputs_and_ends_d2h(
            backend._predict_buf[:4:2], lengths, ends
        )
        is None
    )
    standalone_ends = torch.tensor([7, 8], dtype=torch.int32, device=device)
    assert (
        backend.get_packed_outputs_and_ends_d2h(tokens, lengths, standalone_ends)
        is None
    )
    fb_t, fb_l, fb_e = backend.copy_outputs_d2h(tokens, lengths, standalone_ends)
    if device == "cuda":
        torch.cuda.synchronize()
    assert fb_e.tolist() == [7, 8]
    assert fb_t.device.type == "cpu" and fb_l.device.type == "cpu"


@pytest.mark.skipif(not torch.cuda.is_available(), reason="Sonic requires CUDA")
@pytest.mark.parametrize("max_n", [1, 4])
def test_sonic_backend_packs_outputs_and_request_ends(max_n: int) -> None:
    from tokenspeed.runtime.sampling.backends.sonic import SonicSamplingBackend

    backend = SonicSamplingBackend(
        SamplingBackendConfig(
            synthetic_acceptance_length=None,
            enable_speculative_sampling=False,
            sampling_stream="batch",
            logprob_order="torch",
            max_bs=4,
            max_draft_tokens_per_req=max_n,
            max_req_pool_size=8,
            vocab_size=512,
            device=torch.device("cuda"),
        )
    )
    assert backend.out_tok.data_ptr() == backend._output_pack_buf.data_ptr()
    assert backend.lengths_buf.data_ptr() == backend._accept_length_buf.data_ptr()
    assert (
        backend.request_ends_buf.data_ptr()
        == backend._accept_length_buf.data_ptr()
        + backend._pack_rows * backend.request_ends_buf.element_size()
    )
    if max_n > 1:
        assert backend.v_drafted.data_ptr() == backend._output_pack_buf.data_ptr()
        assert backend.accept_buf.data_ptr() == backend._accept_length_buf.data_ptr()
        predict = backend.v_drafted[:2].view(-1)
        lengths = backend.accept_buf[:2]
    else:
        predict = backend.out_tok[:2].view(-1)
        lengths = backend.lengths_buf[:2]
    ends = backend.request_ends_buf[:2]
    predict.fill_(42)
    lengths.fill_(3 if max_n > 1 else 1)
    ends.copy_(torch.tensor([0, 1], dtype=torch.int32, device="cuda"))

    packed = backend.get_packed_outputs_and_ends_d2h(predict, lengths, ends)
    assert packed is not None
    torch.cuda.synchronize()
    cpu_t, cpu_l, cpu_e = packed
    assert cpu_t.is_pinned() and cpu_l.is_pinned() and cpu_e.is_pinned()
    assert cpu_t.tolist() == [42] * (2 * max_n)
    assert cpu_l.tolist() == ([3, 3] if max_n > 1 else [1, 1])
    assert cpu_e.tolist() == [0, 1]


def _check_runtime_states_pack_release() -> None:
    from tokenspeed.runtime.utils.torch_memory_saver_adapter import (
        TorchMemorySaverAdapter,
    )

    def _ordinary_bytes() -> int:
        return sum(
            segment["total_size"]
            for segment in torch.cuda.memory_snapshot()
            if segment["device"] == torch.cuda.current_device()
            and tuple(segment["segment_pool_id"]) == (0, 0)
        )

    saver = TorchMemorySaverAdapter.create(enable=True, device="cuda")
    with saver.named_region("runtime_states"):
        backend = _backend("cuda")
    pack_ptr = backend._output_pack_buf.data_ptr()
    ends_ptr = backend.request_ends_buf.data_ptr()
    backend._predict_buf[:4].fill_(17)
    backend._accept_length_buf[:2].fill_(2)
    backend.request_ends_buf[:2].copy_(
        torch.tensor([1, 0], dtype=torch.int32, device="cuda")
    )

    for _ in range(2):
        torch.cuda.synchronize()
        torch.cuda.empty_cache()
        saver.pause()
        torch.cuda.synchronize()
        torch.cuda.empty_cache()
        assert _ordinary_bytes() == 0
        saver.resume()
        assert backend._output_pack_buf.data_ptr() == pack_ptr
        assert backend.request_ends_buf.data_ptr() == ends_ptr
        packed = backend.get_packed_outputs_and_ends_d2h(
            backend._predict_buf[:4],
            backend._accept_length_buf[:2],
            backend.request_ends_buf[:2],
        )
        assert packed is not None
        torch.cuda.synchronize()
        assert packed[0].tolist() == [17] * 4
        assert packed[1].tolist() == [2, 2]
        assert packed[2].tolist() == [1, 0]
        del packed

    torch.cuda.synchronize()
    torch.cuda.empty_cache()
    saver.pause()
    assert _ordinary_bytes() == 0


def test_runtime_states_pack_and_host_ring_zero_gpu_occupancy() -> None:
    import subprocess
    import sys
    from pathlib import Path

    if not torch.cuda.is_available():
        pytest.skip("requires CUDA")
    memory_saver = pytest.importorskip("torch_memory_saver")
    with memory_saver.configure_subprocess():
        subprocess.run(
            [
                sys.executable,
                "-c",
                (
                    "from test.runtime.sampling.test_verify_output_pack import "
                    "_check_runtime_states_pack_release; "
                    "_check_runtime_states_pack_release()"
                ),
            ],
            cwd=Path(__file__).resolve().parents[3],
            check=True,
            timeout=120,
        )
