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

"""The Iris heap is created with what its users reserved before it existed
on top of the size its creator planned."""

import pytest
import torch

# The Iris heap ships only with AMD's communication stack.
pytest.importorskip("iris")

from tokenspeed_kernel.ops.communication import _iris  # noqa: E402

MiB = 1 << 20


@pytest.fixture(autouse=True)
def created_sizes(monkeypatch) -> list[int]:
    """A world of one rank whose heap records the size it is created at
    (creating a real heap needs a GPU)."""
    sizes: list[int] = []

    def create(heap_size, device, **options):
        sizes.append(heap_size)
        return object()

    monkeypatch.setattr(_iris, "_reserved", {})
    monkeypatch.setattr(_iris, "_context", None)
    monkeypatch.setattr(_iris, "_heap_size", None)
    monkeypatch.setattr(_iris, "_spans_hosts", False)
    monkeypatch.setattr(_iris, "_unmapped_reason", None)
    monkeypatch.setattr(_iris.iris, "iris", create)
    monkeypatch.setattr(torch.cuda, "current_device", lambda: 0)
    return sizes


def test_without_reservations_the_heap_is_the_planned_size(created_sizes) -> None:
    _iris.get_or_create_context(heap_size=300 * MiB)
    assert created_sizes == [300 * MiB]


def test_reservations_add_to_the_planned_size(created_sizes) -> None:
    _iris.reserve_heap("a", 40 * MiB)
    _iris.reserve_heap("b", 22 * MiB)
    _iris.get_or_create_context(heap_size=300 * MiB)
    _iris.get_or_create_context(heap_size=500 * MiB)
    assert created_sizes == [362 * MiB]


def test_an_owner_reserving_again_replaces_its_reservation(created_sizes) -> None:
    _iris.reserve_heap("a", 60 * MiB)
    _iris.reserve_heap("a", 50 * MiB)
    _iris.get_or_create_context(heap_size=300 * MiB)
    assert created_sizes == [350 * MiB]


def test_a_configured_heap_is_the_total(created_sizes) -> None:
    _iris.configure_iris(heap_size=1024 * MiB, allocator_type=None)
    _iris.reserve_heap("a", 40 * MiB)
    _iris.get_or_create_context(heap_size=300 * MiB)
    assert created_sizes == [1024 * MiB]


def test_a_live_heap_takes_no_reservation() -> None:
    _iris.get_or_create_context(heap_size=300 * MiB)
    with pytest.raises(RuntimeError, match="late_owner reserved 1048576 bytes"):
        _iris.reserve_heap("late_owner", 1 * MiB)
