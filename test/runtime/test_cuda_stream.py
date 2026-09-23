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

"""Stream-fork staging orders event generations without host synchronization."""

import os
import sys
from contextlib import contextmanager

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from ci_system.ci_register import register_cuda_ci

register_cuda_ci(est_time=5, suite="runtime-1gpu")

from tokenspeed.runtime.utils import cuda_stream


@pytest.mark.parametrize("enable", [False, True])
@pytest.mark.parametrize("overlap", [False, True])
def test_staged_branch_event_generations(monkeypatch, enable, overlap):
    calls = []
    events = []

    class Event:
        def __init__(self):
            self.name = len(events)
            self.generation = 0
            events.append(self)

        def record(self, stream):
            self.generation += 1
            calls.append(("record", self.name, self.generation, stream))

        def wait(self, stream):
            calls.append(("wait", self.name, self.generation, stream))

    @contextmanager
    def stream_context(stream):
        assert stream == "aux"
        yield

    monkeypatch.setattr(cuda_stream.torch.cuda, "Event", Event)
    monkeypatch.setattr(cuda_stream.torch.cuda, "current_stream", lambda: "main")
    monkeypatch.setattr(cuda_stream.torch.cuda, "stream", stream_context)
    fork = cuda_stream.StreamFork("aux")
    for _ in range(2):
        with fork.scope(enable=enable, overlap=overlap):
            with fork.branch():
                fork.record_checkpoint()
            fork.join_checkpoint()
            fork.join()
            with fork.branch_after_main():
                pass
            fork.join()
    assert not fork._active
    if not enable:
        assert calls == []
        return
    expected = []
    for checkpoint_generation, generation in enumerate((1, 3), start=1):
        expected.extend(
            [
                ("record", 0, generation, "main"),
                ("wait", 0, generation, "aux"),
                ("record", 2, checkpoint_generation, "aux"),
                ("record", 1, generation, "aux"),
            ]
        )
        if not overlap:
            expected.append(("wait", 1, generation, "main"))
        expected.extend(
            [
                ("wait", 2, checkpoint_generation, "main"),
                ("wait", 1, generation, "main"),
                ("record", 0, generation + 1, "main"),
                ("wait", 0, generation + 1, "aux"),
                ("record", 1, generation + 1, "aux"),
            ]
        )
        if not overlap:
            expected.append(("wait", 1, generation + 1, "main"))
        expected.extend([("wait", 1, generation + 1, "main")] * 2)
    assert calls == expected


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
