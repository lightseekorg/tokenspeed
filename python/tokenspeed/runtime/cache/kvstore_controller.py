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

import logging
from typing import Iterable

from tokenspeed.runtime.utils import get_colorful_logger, get_device_module

logger = get_colorful_logger(__name__)

device_module = get_device_module()


class LayerLoadingEvent:
    def __init__(self, num_layers: int):
        self._num_layers = num_layers
        self.load_events = [device_module.Event() for _ in range(num_layers)]
        self.start_event = device_module.Event()  # start event on controller stream

    def complete(self, layer_index: int):
        assert 0 <= layer_index < self._num_layers
        self.load_events[layer_index].record()

    def wait(self, layer_index: int):
        device_module.current_stream().wait_event(self.load_events[layer_index])

    @property
    def finish_event(self):
        return self.load_events[-1]


class LayerDoneCounter:
    def __init__(self, num_layers: int):
        self.num_layers = num_layers
        # extra producer and consumer counters for overlap mode
        self.num_counters = 3
        self.events = [LayerLoadingEvent(num_layers) for _ in range(self.num_counters)]
        self.producer_index = -1
        self.consumer_indices: tuple[int, ...] = ()
        self._debug_wait_records: list[tuple[int, int, object, object]] = []
        self._debug_sample_layers = {0, max(0, num_layers // 2), max(0, num_layers - 1)}
        self._debug_record_limit = 96

    def update_producer(self):
        next_index = (self.producer_index + 1) % self.num_counters
        if not self.events[next_index].finish_event.query():
            self.events[next_index].finish_event.synchronize()
        self.producer_index = next_index
        return self.producer_index

    def set_consumer(self, indices: int | Iterable[int]):
        if isinstance(indices, int):
            self.consumer_indices = () if indices < 0 else (indices,)
            return
        deduped = []
        for index in indices:
            if index >= 0 and index not in deduped:
                deduped.append(index)
        self.consumer_indices = tuple(deduped)

    def wait_until(self, threshold: int):
        if not self.consumer_indices:
            return
        record_wait = (
            logger.isEnabledFor(logging.DEBUG)
            and threshold in self._debug_sample_layers
            and len(self._debug_wait_records) < self._debug_record_limit
        )
        for consumer_index in self.consumer_indices:
            if record_wait:
                start_event = device_module.Event(enable_timing=True)
                end_event = device_module.Event(enable_timing=True)
                start_event.record()
                self.events[consumer_index].wait(threshold)
                end_event.record()
                self._debug_wait_records.append(
                    (consumer_index, threshold, start_event, end_event)
                )
            else:
                self.events[consumer_index].wait(threshold)

    def consume_debug_wait_records(self, sync: bool = False):
        records = []
        pending = 0
        for (
            consumer_index,
            layer_index,
            start_event,
            end_event,
        ) in self._debug_wait_records:
            if sync:
                end_event.synchronize()
            if not end_event.query():
                pending += 1
                continue
            try:
                elapsed_ms = start_event.elapsed_time(end_event)
            except (AttributeError, RuntimeError, ValueError):
                pending += 1
                continue
            records.append(
                {
                    "producer": consumer_index,
                    "layer": layer_index,
                    "wait_ms": round(float(elapsed_ms), 3),
                }
            )
        self._debug_wait_records.clear()
        return records, pending

    def reset(self):
        self.producer_index = -1
        self.consumer_indices = ()
        self._debug_wait_records.clear()
