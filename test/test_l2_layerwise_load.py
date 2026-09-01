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

import importlib
import sys
from types import SimpleNamespace


class _Event:
    def query(self):
        return True

    def synchronize(self):
        raise AssertionError("a completed load event set must not synchronize")


class _Stream:
    def __init__(self):
        self.waited_events = []

    def wait_event(self, event):
        self.waited_events.append(event)


def _load_tracker_module(monkeypatch):
    module_name = "tokenspeed.runtime.cache.l2.layerwise_load"
    package = importlib.import_module("tokenspeed.runtime.cache.l2")
    monkeypatch.delitem(sys.modules, module_name, raising=False)
    monkeypatch.delattr(package, "layerwise_load", raising=False)
    stream = _Stream()
    device_module = SimpleNamespace(Event=_Event, current_stream=lambda: stream)
    monkeypatch.setitem(
        sys.modules,
        "tokenspeed.runtime.utils",
        SimpleNamespace(
            get_device_module=lambda: device_module,
        ),
    )
    return importlib.import_module(module_name), stream


def test_layerwise_load_waits_for_the_selected_event_set(monkeypatch):
    module, stream = _load_tracker_module(monkeypatch)

    tracker = module.LayerwiseLoadTracker(num_layers=2)
    load_index = tracker.begin_load()
    tracker.set_consumers(load_index)

    tracker.wait_for_layer(1)

    assert stream.waited_events == [tracker.event_sets[load_index].layer_done_events[1]]


def test_cross_layer_wait_uses_the_restore_finish_event(monkeypatch):
    module, stream = _load_tracker_module(monkeypatch)

    tracker = module.LayerwiseLoadTracker(num_layers=2)
    load_index = tracker.begin_load()
    tracker.set_consumers(load_index)
    finish = tracker.event_sets[load_index].last_layer_done_event
    finish.query = lambda: False

    tracker.wait_for_all_layers()

    assert stream.waited_events == [finish]

    finish.query = lambda: True
    tracker.wait_for_all_layers()

    assert tracker.consumer_indices == ()
    assert stream.waited_events == [finish]
