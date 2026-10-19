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

import os
import sys

import pytest

# CI registration (AST-parsed, runtime no-op).
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from ci_system.ci_register import register_cuda_ci  # noqa: E402

register_cuda_ci(est_time=5, suite="runtime-1gpu")

from tokenspeed.runtime.engine.io_struct import (  # noqa: E402
    PauseSchedulerReqInput,
    ReleaseMemoryOccupationReqInput,
    ReleaseMemoryOccupationReqOutput,
    ResumeMemoryOccupationReqInput,
    ResumeMemoryOccupationReqOutput,
)
from tokenspeed.runtime.engine.memory_occupation import (  # noqa: E402
    VALID_TAGS,
    MemoryOccupationController,
    MemoryRegion,
)
from tokenspeed.runtime.engine.pause import PauseController, PauseState  # noqa: E402


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


class _Regions:
    """Records the release/resume of every tag, in call order."""

    def __init__(self) -> None:
        self.calls: list[tuple[str, str]] = []

    def build(self) -> dict[str, MemoryRegion]:
        return {
            tag: MemoryRegion(
                release=lambda tag=tag: self.calls.append(("release", tag)),
                resume=lambda tag=tag: self.calls.append(("resume", tag)),
            )
            for tag in VALID_TAGS
        }


def _controller(
    sender,
    pause,
    regions: _Regions,
    *,
    reset_caches_fn=lambda: True,
    transfers_idle_fn=lambda: True,
):
    return MemoryOccupationController(
        send_func=sender,
        pause_controller=pause,
        regions=regions.build(),
        enabled=True,
        reset_caches_fn=reset_caches_fn,
        transfers_idle_fn=transfers_idle_fn,
        release_refusal_fn=lambda: None,
        weights_busy_fn=lambda: False,
    )


@pytest.mark.parametrize("resume_tags", [None, []])
def test_kv_release_waits_until_cache_can_be_cleared(resume_tags):
    sender = _Sender()
    pause = PauseController(sender)
    regions = _Regions()
    clear_results = iter((False, True))
    controller = _controller(
        sender, pause, regions, reset_caches_fn=lambda: next(clear_results)
    )

    controller.handle_release(ReleaseMemoryOccupationReqInput(tags=["kv_cache"]))
    pause.maybe_finish_drain(_Scheduler())

    assert pause.is_drain_pending
    assert regions.calls == []
    assert sender.items == []

    # An empty memory resume cannot release another operation's drain slot.
    controller.handle_resume(ResumeMemoryOccupationReqInput(tags=resume_tags))
    assert pause.is_drain_pending and pause.admit_blocked
    assert regions.calls == []
    assert isinstance(sender.items[0], ResumeMemoryOccupationReqOutput)
    assert sender.items[0].success
    sender.items.clear()

    pause.maybe_finish_drain(_Scheduler())

    assert not pause.is_drain_pending
    assert pause.state == PauseState.PAUSED_ALL
    assert regions.calls == [("release", "kv_cache")]
    assert len(sender.items) == 1
    assert isinstance(sender.items[0], ReleaseMemoryOccupationReqOutput)
    assert sender.items[0].success


def test_release_waits_for_pd_transfers_and_refuses_new_pd_work_meanwhile():
    sender = _Sender()
    pause = PauseController(sender)
    regions = _Regions()
    transfers_idle = iter((False, False, True))
    controller = _controller(
        sender, pause, regions, transfers_idle_fn=lambda: next(transfers_idle)
    )
    assert not controller.is_releasing_or_released

    controller.handle_release(ReleaseMemoryOccupationReqInput(tags=["weights"]))
    # Drained, but a transfer still touches the cache: nothing is freed yet,
    # and admission already sees the release coming.
    pause.maybe_finish_drain(_Scheduler())
    pause.maybe_finish_drain(_Scheduler())
    assert regions.calls == [] and sender.items == []
    assert controller.is_releasing_or_released

    pause.maybe_finish_drain(_Scheduler())
    assert regions.calls == [("release", "weights")]
    assert sender.items[-1].success
    assert controller.is_releasing_or_released

    controller.handle_resume(ResumeMemoryOccupationReqInput(tags=None))
    assert regions.calls[-1] == ("resume", "weights")
    assert not controller.is_releasing_or_released
    assert pause.state == PauseState.UNPAUSED


def test_full_release_frees_kv_first_and_nccl_last_and_resume_reverses():
    sender = _Sender()
    pause = PauseController(sender)
    regions = _Regions()
    controller = _controller(sender, pause, regions)

    controller.handle_release(ReleaseMemoryOccupationReqInput(tags=None))
    pause.maybe_finish_drain(_Scheduler())
    assert regions.calls == [
        ("release", "kv_cache"),
        ("release", "weights"),
        ("release", "cuda_graph"),
        ("release", "nccl"),
    ]
    regions.calls.clear()

    # The RL order: NCCL, graphs and weights back before new weights are
    # pushed; the KV cache last. Tags given out of order still resume in the
    # fixed order.
    controller.handle_resume(
        ResumeMemoryOccupationReqInput(tags=["weights", "nccl", "cuda_graph"])
    )
    assert regions.calls == [
        ("resume", "nccl"),
        ("resume", "cuda_graph"),
        ("resume", "weights"),
    ]
    assert pause.state == PauseState.PAUSED_ALL
    controller.handle_resume(ResumeMemoryOccupationReqInput(tags=None))
    assert regions.calls[-1] == ("resume", "kv_cache")
    assert pause.state == PauseState.UNPAUSED


def test_weight_load_is_refused_while_weights_or_nccl_are_released():
    sender = _Sender()
    pause = PauseController(sender)
    regions = _Regions()
    controller = _controller(sender, pause, regions)
    assert controller.refuse_weight_load() is None

    controller.handle_release(ReleaseMemoryOccupationReqInput(tags=None))
    pause.maybe_finish_drain(_Scheduler())
    assert "weights" in controller.refuse_weight_load()

    controller.handle_resume(ResumeMemoryOccupationReqInput(tags=["weights"]))
    # The weights are back, but a load may issue collectives on a suspended
    # communicator.
    assert "nccl" in controller.refuse_weight_load()

    controller.handle_resume(ResumeMemoryOccupationReqInput(tags=["nccl"]))
    # The KV cache and the graphs need not be resident to load weights.
    assert controller.refuse_weight_load() is None


def test_unknown_tag_is_rejected_without_draining():
    sender = _Sender()
    pause = PauseController(sender)
    regions = _Regions()
    controller = _controller(sender, pause, regions)

    controller.handle_release(ReleaseMemoryOccupationReqInput(tags=["optimizer"]))

    assert not sender.items[-1].success
    assert "invalid tags" in sender.items[-1].message
    assert not pause.is_drain_pending
    assert not controller.is_releasing_or_released


def test_regions_must_name_known_tags():
    with pytest.raises(ValueError, match="unknown tags"):
        MemoryOccupationController(
            send_func=_Sender(),
            pause_controller=PauseController(_Sender()),
            regions={
                "optimizer": MemoryRegion(release=lambda: None, resume=lambda: None)
            },
            enabled=True,
            reset_caches_fn=lambda: True,
            transfers_idle_fn=lambda: True,
            release_refusal_fn=lambda: None,
            weights_busy_fn=lambda: False,
        )


def test_an_engine_without_nccl_suspend_releases_the_rest_and_refuses_nccl():
    sender = _Sender()
    pause = PauseController(sender)
    regions = _Regions()
    built = regions.build()
    del built["nccl"]
    controller = MemoryOccupationController(
        send_func=sender,
        pause_controller=pause,
        regions=built,
        enabled=True,
        reset_caches_fn=lambda: True,
        transfers_idle_fn=lambda: True,
        release_refusal_fn=lambda: None,
        weights_busy_fn=lambda: False,
    )

    controller.handle_release(ReleaseMemoryOccupationReqInput(tags=["nccl"]))
    assert not sender.items[-1].success
    assert "not releasable on this engine" in sender.items[-1].message
    assert not pause.is_drain_pending

    controller.handle_release(ReleaseMemoryOccupationReqInput(tags=None))
    pause.maybe_finish_drain(_Scheduler())
    assert sender.items[-1].success
    assert regions.calls == [
        ("release", "kv_cache"),
        ("release", "weights"),
        ("release", "cuda_graph"),
    ]
    # Nothing to suspend, so nothing blocks a weight load once the weights
    # are back.
    controller.handle_resume(ResumeMemoryOccupationReqInput(tags=["weights"]))
    assert controller.refuse_weight_load() is None


def test_pause_is_refused_until_every_released_region_is_resumed():
    sender = _Sender()
    pause = PauseController(sender)
    controller = _controller(sender, pause, _Regions())
    controller.handle_release(
        ReleaseMemoryOccupationReqInput(tags=["kv_cache", "weights"])
    )
    pause.maybe_finish_drain(_Scheduler())
    assert sender.items[-1].success

    # A wait/abort pause would let forwards run before the memory is back, and
    # the wake would undo a keep pause. The RL order wakes kv_cache last, so a
    # partial wake must still refuse.
    for resumed in (["weights"], None):
        for mode in ("wait", "abort", "keep"):
            pause.handle_pause(PauseSchedulerReqInput(mode=mode))
            assert not sender.items[-1].success
            assert "resume_memory_occupation" in sender.items[-1].message
            assert pause.state == PauseState.PAUSED_ALL
            assert not pause.is_drain_pending
        controller.handle_resume(ResumeMemoryOccupationReqInput(tags=resumed))
        assert sender.items[-1].success

    pause.handle_pause(PauseSchedulerReqInput(mode="keep"))
    assert sender.items[-1].success
    assert pause.state == PauseState.PAUSED_ALL
