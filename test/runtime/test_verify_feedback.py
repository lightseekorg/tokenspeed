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

import pytest

from tokenspeed.runtime.execution.drafter.verify_feedback import VerifyFeedback


def test_exact_generation_and_reordered_requests():
    history = VerifyFeedback(draft_width=2)
    history.prepare(["a", "b"], [1, 2], num_extends=0)
    first = history.snapshot()
    history.publish(first, [[1.0, 2.0], [3.0, 4.0]])
    assert history.scores() is None
    history.prepare(["b", "a"], [2, 1], num_extends=0)
    assert history.scores() is None
    history.prepare(["b", "a"], [2, 1], num_extends=0)
    assert history.scores() == ((3.0, 4.0), (1.0, 2.0))
    history.prepare(["b", "a"], [2, 1], num_extends=0)
    assert history.scores() is None


def test_late_copy_cannot_cross_slot_reuse_or_extend_restart():
    history = VerifyFeedback(draft_width=1)
    history.prepare(["a"], [1], num_extends=0)
    old = history.snapshot()
    history.prepare(["b"], [1], num_extends=0)
    history.publish(old, [[9.0]])
    for _ in range(2):
        history.prepare(["b"], [1], num_extends=0)
    assert history.scores() is None
    old = history.snapshot()
    history.prepare(["b"], [1], num_extends=1)
    history.publish(old, [[9.0]])
    assert history.slots[1].completed == {}


def test_duplicate_nonfinite_and_invalidated_feedback():
    history = VerifyFeedback(draft_width=1)
    history.prepare(["a"], [0], num_extends=0)
    snapshot = history.snapshot()
    history.publish(snapshot, [[float("inf")]])
    history.publish(snapshot, [[1.0]])
    history.publish(snapshot, [[2.0]])
    for _ in range(2):
        history.prepare(["a"], [0], num_extends=0)
    assert history.scores() == ((1.0,),)
    history.invalidate()
    history.publish(snapshot, [[1.0]])
    assert history.scores() is None


def test_invalid_batch_does_not_change_state():
    history = VerifyFeedback(draft_width=1)
    with pytest.raises(ValueError):
        history.prepare(["a", "b"], [0, 0], num_extends=0)
    assert history.snapshot() == ()
