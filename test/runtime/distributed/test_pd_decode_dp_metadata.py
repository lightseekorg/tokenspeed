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

from tokenspeed.runtime.engine.event_loop import _forward_op_executes_model_forward


class FakeForwardOp:
    def __init__(self, *, input_lengths, request_ids=None, num_extends=0):
        self.input_lengths = input_lengths
        self.request_ids = request_ids or [f"req-{i}" for i in range(len(input_lengths))]
        self._num_extends = num_extends

    def num_extends(self):
        return self._num_extends


def test_pd_decode_extend_only_does_not_require_idle_forward():
    # Decode-side PD EXTEND starts KV receive only; no model collectives run on
    # the active DP rank yet, so idle DP ranks must not enter dummy forward.
    op = FakeForwardOp(input_lengths=[17], num_extends=1)

    assert not _forward_op_executes_model_forward(op, is_disagg_decode=True)


def test_pd_decode_decode_step_requires_idle_forward():
    op = FakeForwardOp(input_lengths=[1], num_extends=0)

    assert _forward_op_executes_model_forward(op, is_disagg_decode=True)


def test_non_pd_extend_still_executes_model_forward():
    op = FakeForwardOp(input_lengths=[17], num_extends=1)

    assert _forward_op_executes_model_forward(op, is_disagg_decode=False)


def test_zero_token_forward_op_is_not_model_work():
    op = FakeForwardOp(input_lengths=[0], num_extends=1)

    assert not _forward_op_executes_model_forward(op, is_disagg_decode=False)
