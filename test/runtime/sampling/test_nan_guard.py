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

"""CPU unit tests for the executor NaN guard (execution/nan_guard.py)."""

from __future__ import annotations

from types import SimpleNamespace

import torch

from tokenspeed.runtime.execution.nan_guard import NanGuard

NAN = float("nan")


def _ctx(bs: int, num_extends: int = 0):
    return SimpleNamespace(bs=bs, num_extends=num_extends)


def _logits_output(logits: torch.Tensor, layout_plan=None):
    return SimpleNamespace(next_token_logits=logits, logits_layout_plan=layout_plan)


def test_audit_logits_flags_per_extend_row():
    guard = NanGuard(max_bs=4, device="cpu")
    logits = torch.zeros((3, 8))
    logits[1, 5] = NAN

    guard.audit_logits(_logits_output(logits), _ctx(bs=3, num_extends=3))

    assert guard.flags.tolist() == [0, 1, 0, 0]
    assert torch.isfinite(logits).all()  # sanitized in place


def test_audit_logits_reduces_verify_rows_per_decode_slot():
    """Spec-verify layout: ne extend rows then nd*n verify rows reduce to
    one flag per request slot."""
    guard = NanGuard(max_bs=4, device="cpu")
    # 1 extend row + 2 decode slots x 3 verify rows.
    logits = torch.zeros((7, 8))
    logits[5, 0] = NAN  # decode slot 1 (rows 4-6), middle row

    guard.audit_logits(_logits_output(logits), _ctx(bs=3, num_extends=1))

    assert guard.flags.tolist() == [0, 0, 1, 0]


def test_audit_logits_accumulates_and_reset_clears():
    """Flags OR across cycles (multi-cycle decode graphs); reset zeroes."""
    guard = NanGuard(max_bs=2, device="cpu")
    bad = torch.full((2, 4), 0.0)
    bad[0, 0] = NAN

    guard.audit_logits(_logits_output(bad.clone()), _ctx(bs=2, num_extends=2))
    # A clean later cycle must not clear the flag.
    guard.audit_logits(_logits_output(torch.zeros((2, 4))), _ctx(bs=2, num_extends=2))
    assert guard.flags.tolist() == [1, 0]

    guard.reset(2)
    assert guard.flags.tolist() == [0, 0]


def test_audit_logits_skips_attribution_for_dp_sharded_layout():
    """With a logits_layout_plan (Batch-DP verify), rows don't map onto this
    rank's batch: no flags, but sanitize still applies."""
    guard = NanGuard(max_bs=2, device="cpu")
    logits = torch.zeros((2, 4))
    logits[0, 0] = NAN

    guard.audit_logits(
        _logits_output(logits, layout_plan=object()), _ctx(bs=2, num_extends=2)
    )

    assert guard.flags.tolist() == [0, 0]
    assert torch.isfinite(logits).all()


def test_audit_logits_ignores_inf_but_sanitizes_it():
    """Only NaN flags a request — +-inf logits are legitimate (vocab masks,
    logit bias) — yet sanitize maps them to large finite values."""
    guard = NanGuard(max_bs=2, device="cpu")
    logits = torch.zeros((2, 4))
    logits[0, 1] = float("inf")
    logits[1, 2] = float("-inf")

    guard.audit_logits(_logits_output(logits), _ctx(bs=2, num_extends=2))

    assert guard.flags.tolist() == [0, 0]
    expected = float(torch.tensor(1e30, dtype=torch.float32))
    assert logits[0, 1].item() == expected
    assert logits[1, 2].item() == -expected


def test_merge_oov_flags_decode_slots():
    guard = NanGuard(max_bs=3, device="cpu")
    # 1 extend token + 2 decode slots x 2 predictions.
    tokens = torch.tensor([5, 7, 9, -1, 11])

    guard.merge_oov(tokens, _ctx(bs=3, num_extends=1), vocab_size=100)

    assert guard.flags.tolist() == [0, 0, 1]


def test_merge_oov_flags_above_vocab():
    guard = NanGuard(max_bs=2, device="cpu")
    tokens = torch.tensor([100, 99])

    guard.merge_oov(tokens, _ctx(bs=2, num_extends=2), vocab_size=100)

    assert guard.flags.tolist() == [1, 0]


def test_disabled_guard_is_inert():
    guard = NanGuard.create(False, 4, "cpu")
    logits = torch.zeros((1, 4))
    logits[0, 0] = NAN

    guard.reset(1)
    guard.audit_logits(_logits_output(logits), _ctx(bs=1, num_extends=1))
    guard.merge_oov(torch.tensor([-1]), _ctx(bs=1, num_extends=1), vocab_size=10)

    assert guard.flags_cpu is None
    # Disabled guard must not touch the logits either.
    assert torch.isnan(logits[0, 0])


def test_flags_cpu_returns_this_steps_flags():
    guard = NanGuard.create(True, 4, "cpu")
    guard.reset(2)
    guard.flags[1] = 1

    flags = guard.flags_cpu

    assert flags.device.type == "cpu"
    assert flags.tolist() == [0, 1]
