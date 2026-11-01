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

"""CPU checks for managed versus resident native allocation scopes."""

from contextlib import contextmanager

import pytest
import torch

from tokenspeed.runtime.utils import torch_memory_saver_adapter as adapter


class CpuMemorySaver:
    """Stand in for the unavailable CUDA allocator, retaining region semantics."""

    def __init__(self):
        self.active = False
        self.tag = None
        self.allocations = []

    @contextmanager
    def region(self, *, tag, enable_cpu_backup):
        del enable_cpu_backup
        assert not self.active
        self.active = True
        self.tag = tag
        try:
            yield
        finally:
            assert self.active
            self.active = False
            self.tag = None

    @contextmanager
    def disable(self):
        # The real CUDA allocator rejects disable() outside an active region.
        assert self.active
        self.active = False
        try:
            yield
        finally:
            self.active = True

    def allocate(self, value):
        tensor = torch.tensor([value], dtype=torch.int32)
        self.allocations.append((self.tag if self.active else None, tensor))
        return tensor

    def pause(self, *, tag):
        for allocation_tag, tensor in self.allocations:
            if allocation_tag == tag:
                tensor.zero_()


@pytest.fixture
def memory(monkeypatch):
    saver = CpuMemorySaver()
    monkeypatch.setattr(adapter, "_primary_memory_saver", saver, raising=False)
    return saver


def test_native_storage_initializes_after_weight_region_and_survives_discard(memory):
    saver = adapter.TorchMemorySaverAdapter.create(enable=True)
    communication = []
    with saver.region(tag="weights", enable_cpu_backup=False):
        weight = memory.allocate(5)
        adapter.initialize_communication(
            lambda: communication.append(memory.allocate(7))
        )
        assert communication == []
        derived_weight = memory.allocate(11)
    saver.pause(tag="weights")
    assert weight.item() == derived_weight.item() == 0
    assert communication[0].item() == 7


def test_multiple_preparations_finish_before_region_returns(memory):
    saver = adapter.TorchMemorySaverAdapter.create(enable=True)
    outputs = []
    with saver.region(tag="weights", enable_cpu_backup=False):
        for value in (3, 5, 7):
            adapter.initialize_communication(
                lambda value=value: outputs.append(memory.allocate(value))
            )
        assert outputs == []
    saver.pause(tag="weights")
    assert [tensor.item() for tensor in outputs] == [3, 5, 7]


def test_failed_weight_load_does_not_initialize_communication(memory):
    saver = adapter.TorchMemorySaverAdapter.create(enable=True)
    outputs = []
    with pytest.raises(RuntimeError, match="load failed"):
        with saver.region(tag="weights", enable_cpu_backup=False):
            adapter.initialize_communication(
                lambda: outputs.append(memory.allocate(13))
            )
            raise RuntimeError("load failed")
    assert outputs == []
    adapter.initialize_communication(lambda: outputs.append(memory.allocate(17)))
    assert outputs[0].item() == 17


def test_preparation_failure_fails_loading_after_managed_scope_exits(memory):
    saver = adapter.TorchMemorySaverAdapter.create(enable=True)

    def fail():
        assert not memory.active
        raise RuntimeError("native preparation failed")

    with pytest.raises(RuntimeError, match="native preparation failed"):
        with saver.region(tag="weights", enable_cpu_backup=False):
            adapter.initialize_communication(fail)
    assert not memory.active


def test_disabled_adapter_initializes_communication_immediately(memory):
    saver = adapter.TorchMemorySaverAdapter.create(enable=False)
    outputs = []
    with saver.region(tag="weights", enable_cpu_backup=False):
        adapter.initialize_communication(lambda: outputs.append(memory.allocate(23)))
        assert outputs[0].item() == 23
    memory.pause(tag="weights")
    assert outputs[0].item() == 23
