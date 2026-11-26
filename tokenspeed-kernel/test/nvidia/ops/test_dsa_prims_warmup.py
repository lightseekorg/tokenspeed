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

"""CPU tests for Prims-TS DSA power-of-two bucket warmup."""

from __future__ import annotations

import pytest
import torch
from tokenspeed_kernel.ops.tuning import kernel_warmup
from tokenspeed_kernel.thirdparty.flashinfer import dsa_prims


class _FakeStream:
    def __init__(self, cuda_stream: int) -> None:
        self.cuda_stream = cuda_stream


class _FakeWrapper:
    def __init__(
        self,
        planned: list[tuple[int, int]],
        runs: list[tuple[int, int, int]],
    ) -> None:
        self._planned = planned
        self._runs = runs
        self.batch_capacity = 0
        self.max_q_len = 0
        self.return_lse = False

    def plan(
        self,
        device,
        batch_capacity: int,
        heads: int,
        *,
        max_topk: int,
        max_extra_topk: int,
        max_q_len: int,
        return_lse: bool,
        **kwargs,
    ) -> None:
        del device, heads, max_topk, max_extra_topk, kwargs
        self.batch_capacity = batch_capacity
        self.max_q_len = max_q_len
        self.return_lse = return_lse
        self._planned.append((batch_capacity, max_q_len))

    def run(
        self,
        q: torch.Tensor,
        kv_cache: torch.Tensor,
        topk_slots: torch.Tensor,
        lengths: torch.Tensor,
        *,
        qo_indptr: torch.Tensor,
        out: torch.Tensor,
        lse: torch.Tensor | None,
        **kwargs,
    ) -> None:
        del kv_cache, kwargs
        rows = q.shape[0]
        assert q.is_contiguous()
        assert topk_slots.is_contiguous() and topk_slots.shape[0] == rows
        assert lengths.is_contiguous() and lengths.shape == (rows,)
        assert out.is_contiguous() and out.shape == (rows, q.shape[1], 512)
        assert int(qo_indptr[-1].item()) == rows
        assert rows <= self.batch_capacity * self.max_q_len
        # Stamp the row count so tests verify the final run overwrites warmup writes.
        out.fill_(float(rows))
        if self.return_lse:
            assert lse is not None and lse.is_contiguous()
            assert lse.shape == (rows, q.shape[1])
            lse.fill_(float(rows))
        self._runs.append((self.batch_capacity, self.max_q_len, rows))


@pytest.fixture()
def fake_prims(monkeypatch: pytest.MonkeyPatch):
    dsa_prims._plans.clear()
    dsa_prims._warmed_plans.clear()
    dsa_prims._captured_plans.clear()
    dsa_prims._warmed_buckets.clear()
    planned: list[tuple[int, int]] = []
    runs: list[tuple[int, int, int]] = []
    capturing = {"active": False}
    stream = _FakeStream(101)

    monkeypatch.setattr(dsa_prims, "_make_wrapper", lambda: _FakeWrapper(planned, runs))
    monkeypatch.setattr(torch.cuda, "current_stream", lambda device=None: stream)
    monkeypatch.setattr(
        torch.cuda, "is_current_stream_capturing", lambda: capturing["active"]
    )
    yield planned, runs, capturing, stream
    dsa_prims._plans.clear()
    dsa_prims._warmed_plans.clear()
    dsa_prims._captured_plans.clear()
    dsa_prims._warmed_buckets.clear()


def _inputs(rows: int, *, heads: int = 4, topk: int = 512):
    q = torch.zeros((rows, heads, 576), dtype=torch.float8_e4m3fn)
    kv = torch.zeros((64, 576), dtype=torch.float8_e4m3fn)
    slots = torch.full((rows, topk), -1, dtype=torch.int32)
    lengths = torch.zeros((rows,), dtype=torch.int32)
    return q, kv, slots, lengths


def test_prefill_warmup_plans_and_runs_all_smaller_power_of_two_buckets(fake_prims):
    planned, runs, _, _ = fake_prims
    q, kv, slots, lengths = _inputs(17)
    out = torch.empty((17, 4, 512), dtype=torch.bfloat16)

    with kernel_warmup(enabled=True):
        got_out, got_lse = dsa_prims.sparse_attention(
            q,
            kv,
            slots,
            lengths,
            mode="prefill",
            query_width=1,
            softmax_scale=1.0,
            return_lse=True,
            out=out,
        )

    assert got_out is out
    assert torch.all(got_out == 17.0)
    assert torch.all(got_lse == 17.0)
    assert planned == [(1, 1), (1, 2), (1, 4), (1, 8), (1, 16), (1, 32)]
    assert runs == [
        (1, 1, 1),
        (1, 2, 2),
        (1, 4, 4),
        (1, 8, 8),
        (1, 16, 16),
        (1, 32, 17),
    ]

    # Repeated layers in the same warmup pass skip the warmup ladder.
    runs.clear()
    with kernel_warmup(enabled=True):
        dsa_prims.sparse_attention(
            q,
            kv,
            slots,
            lengths,
            mode="prefill",
            query_width=1,
            softmax_scale=1.0,
            return_lse=True,
            out=None,
        )
    assert len(planned) == 6
    assert runs == [(1, 32, 17)]

    # Serving calls across every smaller row count reuse the warmed plans.
    planned.clear()
    for rows in (1, 2, 3, 5, 8, 9, 16, 17, 32):
        sq, skv, sslots, slens = _inputs(rows)
        res, lse = dsa_prims.sparse_attention(
            sq,
            skv,
            sslots,
            slens,
            mode="prefill",
            query_width=1,
            softmax_scale=1.0,
            return_lse=True,
            out=None,
        )
        assert torch.all(res == float(rows))
        assert torch.all(lse == float(rows))
    assert planned == []


def test_prefill_warmup_extends_incrementally_for_larger_peak(fake_prims):
    planned, runs, _, _ = fake_prims
    q8, kv, slots8, _ = _inputs(8)
    with kernel_warmup(enabled=True):
        dsa_prims.sparse_attention(
            q8,
            kv,
            slots8,
            None,
            mode="prefill",
            query_width=1,
            softmax_scale=1.0,
            return_lse=False,
            out=None,
        )
    assert planned == [(1, 1), (1, 2), (1, 4), (1, 8)]

    planned.clear()
    runs.clear()
    q64, kv, slots64, _ = _inputs(64)
    with kernel_warmup(enabled=True):
        dsa_prims.sparse_attention(
            q64,
            kv,
            slots64,
            None,
            mode="prefill",
            query_width=1,
            softmax_scale=1.0,
            return_lse=False,
            out=None,
        )
    assert planned == [(1, 16), (1, 32), (1, 64)]
    assert runs == [(1, 16, 16), (1, 32, 32), (1, 64, 64)]


def test_decode_warmup_covers_smaller_batch_capacities_and_4d_output(fake_prims):
    planned, runs, _, _ = fake_prims
    width = 4
    batch = 5
    rows = batch * width
    q, kv, slots, lengths = _inputs(rows)
    out_4d = torch.empty((batch, width, 4, 512), dtype=torch.bfloat16)

    with kernel_warmup(enabled=True):
        got = dsa_prims.sparse_attention(
            q,
            kv,
            slots,
            lengths,
            mode="decode",
            query_width=width,
            softmax_scale=1.0,
            return_lse=False,
            out=out_4d,
        )

    assert got is out_4d
    assert torch.all(got == float(rows))
    assert planned == [(1, width), (2, width), (4, width), (8, width)]
    assert runs == [
        (1, width, 1 * width),
        (2, width, 2 * width),
        (4, width, 4 * width),
        (8, width, rows),
    ]


def test_serving_outside_warmup_plans_only_requested_bucket(fake_prims):
    planned, runs, _, _ = fake_prims
    q, kv, slots, lengths = _inputs(17)
    dsa_prims.sparse_attention(
        q,
        kv,
        slots,
        lengths,
        mode="prefill",
        query_width=1,
        softmax_scale=1.0,
        return_lse=False,
        out=None,
    )
    assert planned == [(1, 32)]
    assert runs == [(1, 32, 17)]


def test_capture_skips_warmup_and_rejects_unwarmed_bucket(fake_prims):
    planned, _, capturing, _ = fake_prims
    capturing["active"] = True
    q, kv, slots, lengths = _inputs(8)
    with kernel_warmup(enabled=True), pytest.raises(RuntimeError, match="warm up"):
        dsa_prims.sparse_attention(
            q,
            kv,
            slots,
            lengths,
            mode="prefill",
            query_width=1,
            softmax_scale=1.0,
            return_lse=False,
            out=None,
        )
    assert planned == []


def test_warmed_plans_survive_multi_config_multi_stream_and_eager_lru_eviction(
    fake_prims,
):
    """Warmed ladders across 2 configs x 2 streams exceed _MAX_EAGER_PLANS and
    must never be evicted by one another or by subsequent eager serving plans."""
    planned, _, _, stream = fake_prims
    configs = ((12, 512), (24, 1024))
    streams = (101, 202)
    peak_prefill = 128  # 8 buckets: 1..128 -> 32 prefill plans across 2x2
    decode_batch = 32  # 6 buckets: 1..32 -> 24 decode plans across 2x2
    # Add a second decode width so total warmed plans = 32 + 24 + 24 = 80 > 64.
    decode_widths = (1, 4)

    with kernel_warmup(enabled=True):
        for stream_id in streams:
            stream.cuda_stream = stream_id
            for heads, topk in configs:
                q, kv, slots, lengths = _inputs(peak_prefill, heads=heads, topk=topk)
                dsa_prims.sparse_attention(
                    q,
                    kv,
                    slots,
                    lengths,
                    mode="prefill",
                    query_width=1,
                    softmax_scale=1.0,
                    return_lse=False,
                    out=None,
                )
                for width in decode_widths:
                    dq, dkv, dslots, dlens = _inputs(
                        decode_batch * width, heads=heads, topk=topk
                    )
                    dsa_prims.sparse_attention(
                        dq,
                        dkv,
                        dslots,
                        dlens,
                        mode="decode",
                        query_width=width,
                        softmax_scale=1.0,
                        return_lse=False,
                        out=None,
                    )

    # 2 streams x 2 configs x (8 prefill + 5 new decode w=1 + 5 new decode w=4) = 72
    # ((batch_capacity=1, query_width in {1, 4}) is shared between prefill and decode).
    assert len(dsa_prims._warmed_plans) == 72
    assert len(dsa_prims._warmed_plans) > dsa_prims._MAX_EAGER_PLANS
    assert len(dsa_prims._plans) == 0

    # Flood the eager LRU past _MAX_EAGER_PLANS with unwarmed streams/configs.
    for extra_stream in range(300, 370):
        stream.cuda_stream = extra_stream
        q, kv, slots, lengths = _inputs(4, heads=4, topk=512)
        dsa_prims.sparse_attention(
            q,
            kv,
            slots,
            lengths,
            mode="prefill",
            query_width=1,
            softmax_scale=1.0,
            return_lse=False,
            out=None,
        )
    assert len(dsa_prims._plans) == dsa_prims._MAX_EAGER_PLANS

    # Every warmed key across both configs and both streams still hits without re-planning.
    planned.clear()
    for stream_id in streams:
        stream.cuda_stream = stream_id
        for heads, topk in configs:
            for rows in (1, 2, 3, 7, 16, 63, 128):
                q, kv, slots, lengths = _inputs(rows, heads=heads, topk=topk)
                dsa_prims.sparse_attention(
                    q,
                    kv,
                    slots,
                    lengths,
                    mode="prefill",
                    query_width=1,
                    softmax_scale=1.0,
                    return_lse=False,
                    out=None,
                )
            for width in decode_widths:
                for batch in (1, 2, 5, 16, 32):
                    dq, dkv, dslots, dlens = _inputs(
                        batch * width, heads=heads, topk=topk
                    )
                    dsa_prims.sparse_attention(
                        dq,
                        dkv,
                        dslots,
                        dlens,
                        mode="decode",
                        query_width=width,
                        softmax_scale=1.0,
                        return_lse=False,
                        out=None,
                    )
    assert planned == []


def test_capture_transfers_warmed_plan_ownership_to_graph(fake_prims, monkeypatch):
    planned, runs, capturing, _stream = fake_prims
    retained: list[object] = []
    monkeypatch.setattr(dsa_prims, "retain_for_capture", retained.append)

    q, kv, slots, lengths = _inputs(8)
    with kernel_warmup(enabled=True):
        dsa_prims.sparse_attention(
            q,
            kv,
            slots,
            lengths,
            mode="decode",
            query_width=1,
            softmax_scale=1.0,
            return_lse=False,
            out=None,
        )
    assert len(dsa_prims._warmed_plans) == 4  # buckets 1, 2, 4, 8

    capturing["active"] = True
    planned.clear()
    runs.clear()
    dsa_prims.sparse_attention(
        q,
        kv,
        slots,
        lengths,
        mode="decode",
        query_width=1,
        softmax_scale=1.0,
        return_lse=False,
        out=None,
    )
    assert planned == []
    assert len(runs) == 1
    assert len(retained) == 1
    assert len(dsa_prims._captured_plans) == 1
    assert len(dsa_prims._warmed_plans) == 3
    # Dropping the graph owner releases the captured plan via WeakValueDictionary.
    retained.clear()
    assert len(dsa_prims._captured_plans) == 0
