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

from __future__ import annotations

from collections.abc import Iterable

from tokenspeed.runtime.utils import get_device_module

device_module = get_device_module()

_NUM_LOAD_EVENT_SETS = 3


class _LayerwiseLoadEvents:
    def __init__(self, num_layers: int):
        self.layer_done_events = [device_module.Event() for _ in range(num_layers)]
        self.start_event = device_module.Event()
        self.layer_ready_init_event = device_module.Event()
        self.layer_ready_flags = None
        self.wait_layer_ready = None

    def wait_for_layer(self, layer_index: int) -> None:
        if self.layer_ready_flags is not None:
            waiter = self.wait_layer_ready
            if waiter is None:
                raise RuntimeError("layer-ready flags bound without a waiter")
            # Flag reuse leaves stale 1s until load_stream's zero_() lands.
            # Wait for that reset before spinning, or a reused workspace can
            # look finished and attention reads unrestored Device KV.
            device_module.current_stream().wait_event(self.layer_ready_init_event)
            waiter(self.layer_ready_flags, layer_index)
            return
        device_module.current_stream().wait_event(self.layer_done_events[layer_index])

    @property
    def last_layer_done_event(self):
        return self.layer_done_events[-1]


class LayerwiseLoadTracker:
    def __init__(self, num_layers: int):
        # Keep separate events for the H2D load being produced, the load used
        # by the current forward, and the next one-step-overlap load.
        self.event_sets = [
            _LayerwiseLoadEvents(num_layers) for _ in range(_NUM_LOAD_EVENT_SETS)
        ]
        self.current_load_index = -1
        self.consumer_indices: tuple[int, ...] = ()

    def begin_load(self) -> int:
        next_index = (self.current_load_index + 1) % len(self.event_sets)
        last_layer_done = self.event_sets[next_index].last_layer_done_event
        if not last_layer_done.query():
            last_layer_done.synchronize()
        self.current_load_index = next_index
        return next_index

    def set_consumers(self, indices: int | Iterable[int]) -> None:
        if isinstance(indices, int):
            self.consumer_indices = () if indices < 0 else (indices,)
            return
        deduped = []
        for index in indices:
            if index >= 0 and index not in deduped:
                deduped.append(index)
        self.consumer_indices = tuple(deduped)

    def wait_for_layer(self, layer_index: int) -> None:
        if not self.consumer_indices:
            return
        for consumer_index in self.consumer_indices:
            self.event_sets[consumer_index].wait_for_layer(layer_index)

    def reset(self) -> None:
        self.current_load_index = -1
        self.consumer_indices = ()
        for events in self.event_sets:
            events.layer_ready_flags = None
            events.wait_layer_ready = None
