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

"""The Iris heap's headroom covers what its users reserve before it exists."""

import pytest

# The Iris heap ships only with AMD's communication stack.
pytest.importorskip("iris")

from tokenspeed_kernel.ops.communication import _iris  # noqa: E402


@pytest.fixture(autouse=True)
def no_reservations(monkeypatch):
    monkeypatch.setattr(_iris, "_reserved", {})
    monkeypatch.setattr(_iris, "_context", None)


def test_small_reservations_keep_the_floor() -> None:
    _iris.reserve_heap("a", 1 << 20)
    assert _iris.heap_headroom_bytes() == _iris.HEAP_HEADROOM_BYTES


def test_the_headroom_covers_every_owner_plus_the_margin() -> None:
    _iris.reserve_heap("a", 40 << 20)
    _iris.reserve_heap("b", 22 << 20)
    assert _iris.heap_headroom_bytes() == (62 << 20) + _iris.HEAP_MARGIN_BYTES


def test_an_owner_reserving_again_replaces_its_reservation() -> None:
    _iris.reserve_heap("a", 60 << 20)
    _iris.reserve_heap("a", 50 << 20)
    assert _iris.heap_headroom_bytes() == (50 << 20) + _iris.HEAP_MARGIN_BYTES


def test_a_live_heap_takes_no_reservation(monkeypatch) -> None:
    # Stands in for a created heap (creating one needs the GPUs' group).
    monkeypatch.setattr(_iris, "_context", object())
    with pytest.raises(RuntimeError, match="late_owner reserved 1048576 bytes"):
        _iris.reserve_heap("late_owner", 1 << 20)
