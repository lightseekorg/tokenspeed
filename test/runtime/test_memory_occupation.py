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

"""CPU-only tests for GPU memory release coordination."""

from tokenspeed.runtime.engine.io_struct import (
    ReleaseMemoryOccupationReqInput,
    ReleaseMemoryOccupationReqOutput,
)
from tokenspeed.runtime.engine.memory_occupation import MemoryOccupationController
from tokenspeed.runtime.engine.pause import PauseController, PauseState


class _Sender:
    def __init__(self) -> None:
        self.items = []

    def send_pyobj(self, item) -> None:
        self.items.append(item)


class _Scheduler:
    def waiting_size(self) -> int:
        return 0

    def decoding_size(self) -> int:
        return 0

    def prefilling_size(self) -> int:
        return 0


class _MemoryAdapter:
    def __init__(self) -> None:
        self.paused_tags = []

    def pause(self, tag: str) -> None:
        self.paused_tags.append(tag)


def test_kv_release_waits_until_cache_can_be_cleared():
    sender = _Sender()
    pause = PauseController(sender)
    adapter = _MemoryAdapter()
    clear_results = iter((False, True))
    controller = MemoryOccupationController(
        send_func=sender,
        pause_controller=pause,
        adapter=adapter,
        enabled=True,
        reset_caches_fn=lambda: next(clear_results),
        kv_repair_fn=lambda: None,
    )

    controller.handle_release(ReleaseMemoryOccupationReqInput(tags=["kv_cache"]))
    pause.maybe_finish_drain(_Scheduler())

    assert pause.is_drain_pending
    assert adapter.paused_tags == []
    assert sender.items == []

    pause.maybe_finish_drain(_Scheduler())

    assert not pause.is_drain_pending
    assert pause.state == PauseState.PAUSED_ALL
    assert adapter.paused_tags == ["kv_cache"]
    assert len(sender.items) == 1
    assert isinstance(sender.items[0], ReleaseMemoryOccupationReqOutput)
    assert sender.items[0].success
