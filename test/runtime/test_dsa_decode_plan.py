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

"""DSA decode plan buffers: refreshed in place across decode and mixed rounds."""

import os
import sys
from types import SimpleNamespace

# CI Registration (parsed via AST, runtime no-op)
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from ci_system.ci_register import register_cuda_ci  # noqa: E402

register_cuda_ci(est_time=10, suite="runtime-1gpu")

import torch

from tokenspeed.runtime.execution.forward_batch_info import ForwardMode
from tokenspeed.runtime.layers.attention.backends.paged import dsa


def _backend(monkeypatch, planned, *, spec_num_tokens):
    def plan(*, seq_lens_2d, page_size, out=None):
        planned.append(seq_lens_2d.flatten().tolist())
        if out is None:
            return torch.zeros(2, dtype=torch.int32)
        out.fill_(len(planned))
        return out

    monkeypatch.setattr(dsa, "dsa_plan", plan)
    backend = object.__new__(dsa.DSABackend)
    backend.kernel_page_size = 64
    backend.spec_num_tokens = spec_num_tokens
    backend.is_draft = False
    backend._dense_backend = SimpleNamespace(
        forward_decode_metadata=SimpleNamespace(_dsa_seq_lens_2d=None, _dsa_plan=None),
        refresh_decode_metadata=lambda *args, **kwargs: None,
        init_forward_metadata=lambda *args, **kwargs: None,
        chunked_prefill_metadata=None,
    )
    return backend


def _seq_lens(*lens):
    return torch.tensor(lens, dtype=torch.int32)


def _mixed_round(backend, num_extends, seq_lens):
    extend_lens = seq_lens[:num_extends]
    backend.init_forward_metadata(
        len(seq_lens),
        num_extends,
        seq_lens,
        torch.zeros(len(seq_lens), 8, dtype=torch.int32),
        ForwardMode.MIXED,
        extend_seq_lens=extend_lens,
        extend_seq_lens_cpu=extend_lens,
        extend_prefix_lens=torch.zeros_like(extend_lens),
        extend_prefix_lens_cpu=torch.zeros_like(extend_lens),
        extend_with_prefix=False,
        query_shard=None,
        page_table_cpu=None,
    )


def test_mixed_round_writes_the_captured_decode_plan_in_place(monkeypatch):
    # The decode graph of a bs bakes that bs's plan buffers, and a mixed
    # round of the same bs publishes its plan on the same leaf view. Rebinding
    # the buffers there would leave the graph replaying a stale plan against
    # live lengths.
    planned = []
    backend = _backend(monkeypatch, planned, spec_num_tokens=1)
    view = backend.forward_decode_metadata
    page_table = torch.zeros(3, 8, dtype=torch.int32)
    backend.refresh_decode_metadata(3, 3, _seq_lens(900, 950, 1000), page_table)
    lens, plan = view._dsa_seq_lens_2d, view._dsa_plan

    _mixed_round(backend, 2, _seq_lens(100, 120, 130))
    assert view._dsa_seq_lens_2d is lens and view._dsa_plan is plan
    seq_lens = _seq_lens(101, 121, 131)
    backend.refresh_decode_metadata(3, 3, seq_lens, page_table)
    assert view._dsa_seq_lens_2d is lens and view._dsa_plan is plan
    assert planned == [[900, 950, 1000], [130], [101, 121, 131]]
    assert plan.tolist() == [3, 3]

    seq_lens.fill_(7)
    assert lens.flatten().tolist() == [101, 121, 131]


def test_mixed_round_plans_only_its_decode_rows(monkeypatch):
    planned = []
    backend = _backend(monkeypatch, planned, spec_num_tokens=2)
    _mixed_round(backend, 1, _seq_lens(40, 300, 500))
    view = backend.forward_decode_metadata
    assert view._dsa_seq_lens_2d.flatten().tolist() == [40, 40, 300, 300, 500, 500]
    assert planned == [[300, 300, 500, 500]]

    _mixed_round(backend, 3, _seq_lens(41, 301, 501))
    assert planned[-1] == [41, 41, 301, 301, 501, 501]


def test_draft_mixed_refresh_plans_only_its_decode_rows(monkeypatch):
    # A mixed round's draft first step gets its decode metadata from the
    # refresh, which still carries the round's extend requests; its decode
    # top-k scores only the rows past them, so the plan must start there too.
    planned = []
    backend = _backend(monkeypatch, planned, spec_num_tokens=2)
    backend.is_draft = True
    seq_lens = _seq_lens(40, 300, 500)
    _mixed_round(backend, 1, seq_lens)
    assert planned == []

    page_table = torch.zeros(3, 8, dtype=torch.int32)
    backend.refresh_decode_metadata(3, 3, seq_lens, page_table, num_extends=1)
    view = backend.forward_decode_metadata
    assert view._dsa_seq_lens_2d.flatten().tolist() == [40, 40, 300, 300, 500, 500]
    assert planned == [[300, 300, 500, 500]]

    backend.refresh_decode_metadata(3, 3, _seq_lens(41, 301, 501), page_table)
    assert planned[-1] == [41, 41, 301, 301, 501, 501]
