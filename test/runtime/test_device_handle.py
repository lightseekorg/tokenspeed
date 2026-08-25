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

"""The cross-plane contract, at the places that can break it.

Two halves, both stated in ``execution/forward_thread.py``. What the control
plane hands over travels in a closure and is frozen once submitted; and the
device side is not reachable from the control plane at all, because the loop
holds a ``DeviceHandle`` rather than the executor behind it.

The first half is easy to lose in three specific spots, so each gets a test:
the PD completion path (device writes issued straight from an event handler),
the multimodal gather (a live request struct captured by reference), and the
SHM release (a resource freed while a queued forward may still read it).

The second half is a property of the code's shape rather than of any one call
site, so it is asserted over the shape: the loop must not bind a device object
even as a local, must not keep the startup wiring, must find no startup hook on
the running handle, and its collaborators must be handed that handle rather
than walk to it.
"""

from __future__ import annotations

import ast
from concurrent.futures import Future
from types import SimpleNamespace

import pytest
import torch
from tokenspeed_scheduler import PD

from tokenspeed.runtime.execution.device import DeviceHandle, DeviceWiring
from tokenspeed.runtime.multimodal.inputs import (
    Modality,
    MultimodalDataItem,
    MultimodalInputs,
    multimodal_context_for_forward,
)
from tokenspeed.runtime.pd.decode_executor import DisaggDecodeExecutor
from tokenspeed.runtime.pd.transfer_hooks import PdTransferHooks


class _ForwardThread:
    """Records what crossed to the data plane, and in what order.

    Mirrors the real contract: ``submit`` returns a future resolved with the
    callable's result or exception (never raising at the call site), ``run``
    blocks and re-raises.
    """

    def __init__(self, trace) -> None:
        self._trace = trace

    def submit(self, fn):
        self._trace.append("submit")
        future: Future = Future()
        try:
            future.set_result(fn())
        except BaseException as exc:  # noqa: BLE001 — mirrored to the future
            future.set_exception(exc)
        return future

    def run(self, fn):
        self._trace.append("run")
        return fn()


class _DecodeExecutor(DisaggDecodeExecutor):
    """A decode-role kv_transfer the hooks will recognize by type.

    Subclasses rather than fakes because the hooks dispatch on
    ``isinstance``; the real ``__init__`` (sockets, KV manager) is bypassed
    on purpose — only the three pop/generate methods are exercised.
    """

    def __init__(self, events, slot=None, candidates=None) -> None:
        self._events = events
        self._slot = slot
        self._candidates = candidates

    def generate_events(self):
        return self._events

    def pop_remote_cache_slot(self, req_id):
        return self._slot

    def pop_remote_spec_candidate_ids(self, req_id):
        return self._candidates


def _handle(trace):
    return DeviceHandle(
        SimpleNamespace(
            forward_thread=_ForwardThread(trace),
            write_remote_spec_candidate_ids=lambda idx, ids: trace.append(
                ("candidates", idx, list(ids))
            ),
            mark_remote_cache_ready=lambda slot: trace.append(("ready", slot)),
        )
    )


def _loop(trace, kv_transfer, state):
    output_processor = SimpleNamespace(
        rid_to_state={"r0": state} if state is not None else {},
        on_remote_prefill_done=lambda rid, tok: trace.append(("bootstrap", tok)),
        finish_remote_prefill_only_request=lambda rid: [],
    )
    return SimpleNamespace(
        kv_transfer=kv_transfer,
        output_processor=output_processor,
        _pd_cache_enabled=False,
    )


def _decoding_state():
    return SimpleNamespace(to_abort=False, finished=False)


# ----------------------------------------------------------------------
# PD completion: device writes belong to the data plane, in one submission.
# ----------------------------------------------------------------------


def test_remote_prefill_completion_lands_both_writes_on_the_device():
    trace: list = []
    event = PD.RemotePrefillDoneEvent("r0", 42)
    kv_transfer = _DecodeExecutor([event], slot=7, candidates=(3, [11, 12]))
    state = _decoding_state()

    hooks = PdTransferHooks(_loop(trace, kv_transfer, state), _handle(trace))
    hooks.poll_transfer_events()

    # One crossing, not two, and the candidates precede the readiness arm:
    # hydration reads the row the candidates were just written into.
    assert trace == [
        ("bootstrap", 42),
        "run",
        ("candidates", 3, [11, 12]),
        ("ready", 7),
    ]


def test_completion_without_device_work_submits_nothing():
    trace: list = []
    event = PD.RemotePrefillDoneEvent("r0", 42)
    kv_transfer = _DecodeExecutor([event], slot=None, candidates=None)
    state = _decoding_state()

    hooks = PdTransferHooks(_loop(trace, kv_transfer, state), _handle(trace))
    hooks.poll_transfer_events()

    assert trace == [("bootstrap", 42)]


def test_an_aborted_request_still_lands_its_candidates_but_is_not_armed():
    trace: list = []
    event = PD.RemotePrefillDoneEvent("r0", 42)
    kv_transfer = _DecodeExecutor([event], slot=7, candidates=(3, [11, 12]))
    aborted = SimpleNamespace(to_abort=True, finished=False)

    hooks = PdTransferHooks(_loop(trace, kv_transfer, aborted), _handle(trace))
    hooks.poll_transfer_events()

    assert trace == ["run", ("candidates", 3, [11, 12])]


# ----------------------------------------------------------------------
# L2 cache plans: transfers launch on the data plane, behind the zeroing.
# ----------------------------------------------------------------------


def test_cache_plan_submission_rides_the_fifo_behind_page_zeroing():
    """The FIFO order zero-then-load is what keeps a reused page from being
    zeroed after the load already overwrote it; submitting the plan from the
    control plane would race that."""
    trace: list = []
    l2 = SimpleNamespace(
        submit_plan=lambda plan: trace.append(("cache_plan", plan)),
        poll_results=lambda: ["done"],
    )
    handle = DeviceHandle(
        SimpleNamespace(
            forward_thread=_ForwardThread(trace),
            zero_cache_pages=lambda pages: trace.append(("zero", tuple(pages))),
        ),
        l2_cache_executor=l2,
    )

    handle.submit_page_zeroing([3, 4])
    handle.submit_cache_plan("PLAN")

    assert trace == ["submit", ("zero", (3, 4)), "submit", ("cache_plan", "PLAN")]
    # Polling never touches the FIFO — the round head must not wait on it.
    assert handle.poll_cache_results() == ["done"]
    assert trace[-1] != "submit"


def test_a_failed_cache_plan_submission_surfaces_at_the_next_poll():
    """A submission that raised produces no completion acks; swallowing it
    would leave its ops counted in flight forever."""
    trace: list = []

    def exploding_submit_plan(plan):
        raise ValueError("bad cache op")

    handle = DeviceHandle(
        SimpleNamespace(forward_thread=_ForwardThread(trace)),
        l2_cache_executor=SimpleNamespace(
            submit_plan=exploding_submit_plan,
            poll_results=lambda: [],
        ),
    )

    # Submission itself never raises (fire-and-forget)...
    handle.submit_cache_plan("PLAN")
    # ...the failure re-raises at the round head, data-plane cause chained.
    with pytest.raises(RuntimeError, match="cache-plan submission failed") as info:
        handle.poll_cache_results()
    assert isinstance(info.value.__cause__, ValueError)

    # A clean submission is settled and dropped; polling keeps working.
    handle._l2.submit_plan = lambda plan: trace.append(("plan", plan))
    handle.submit_cache_plan("PLAN2")
    assert handle.poll_cache_results() == []
    assert not handle._l2_submissions


def test_cache_ops_without_kvstore_refuse_loudly():
    handle = _handle([])
    with pytest.raises(RuntimeError, match="enable-kvstore"):
        handle.submit_cache_plan("PLAN")
    with pytest.raises(RuntimeError, match="enable-kvstore"):
        handle.poll_cache_results()


# ----------------------------------------------------------------------
# EPD admission: encoder facts resolve past the gate, never before.
# ----------------------------------------------------------------------


def test_text_only_pd_nodes_never_read_the_encoder_facts():
    """The facts callable must not fire unless the node is an EPD prefill.

    Reading the vision tower's dtype raises on a text-only model, and every
    text-only PD node passes through this factory — so the facts are handed
    over as a bound method and resolved only past the manager gate. Passing
    the VALUE here once crashed every text-only PD deployment at startup.
    """
    from tokenspeed.runtime.epd.prefill_admission import make_epd_prefill_admission

    def facts():  # pragma: no cover — reaching this is the failure
        raise AssertionError("encoder facts read on a non-EPD node")

    admission = make_epd_prefill_admission(
        SimpleNamespace(disaggregation_mode="decode"),
        0,
        model_config=SimpleNamespace(is_multimodal_active=False),
        encoder_model_facts=facts,
        mapping=None,
        attn_tp_rank=0,
        attn_tp_size=1,
        attn_tp_cpu_group=None,
        pg_manager=None,
    )

    assert admission is None


# ----------------------------------------------------------------------
# Multimodal gather: the forward gets a snapshot, not the live struct.
# ----------------------------------------------------------------------


def _mm_inputs(positions=None, delta=None):
    return MultimodalInputs(
        mm_items=[MultimodalDataItem(modality=Modality.IMAGE, hash=1, pad_value=1)],
        mrope_positions=positions,
        mrope_position_delta=delta,
    )


def _forward_op(num_extends=1):
    return SimpleNamespace(
        request_ids=["r0"],
        num_extends=lambda: num_extends,
        extend_prefix_lens=[0],
        input_lengths=[4],
    )


def test_gathered_context_does_not_see_later_control_plane_edits():
    positions = torch.arange(12, dtype=torch.int64).reshape(3, 4)
    mm = _mm_inputs(positions=positions)
    state = SimpleNamespace(
        multimodal_inputs=mm,
        maybe_extend_multimodal_mrope_positions=lambda: None,
    )

    ctx = multimodal_context_for_forward(_forward_op(), {"r0": state})

    # The next round's gather extends the live struct's table; a forward
    # already dispatched with the previous context must not observe it.
    mm.mrope_positions = torch.zeros(3, 8, dtype=torch.int64)
    assert ctx.mm_inputs[0] is not mm
    assert torch.equal(ctx.mm_inputs[0].mrope_positions, positions)


def test_gather_resolves_the_decode_delta_on_the_live_struct():
    mm = _mm_inputs(delta=torch.tensor([[5]], dtype=torch.int64))
    state = SimpleNamespace(
        multimodal_inputs=mm,
        maybe_extend_multimodal_mrope_positions=lambda: None,
    )

    ctx = multimodal_context_for_forward(_forward_op(num_extends=0), {"r0": state})

    # Resolved on the control plane, so the forward only reads — and resolved
    # on the LIVE struct, so the next round's snapshot inherits it instead of
    # paying the item() again.
    assert mm.mrope_position_delta_scalar == 5
    assert ctx.mm_inputs[0].mrope_position_delta_scalar == 5


# ----------------------------------------------------------------------
# SHM release: freed behind any forward that captured the features.
# ----------------------------------------------------------------------


def test_shm_release_is_queued_behind_the_forwards_that_may_read_it():
    from tokenspeed.runtime.engine.generation_output_processor import OutputProcesser

    trace: list = []
    released: list = []
    processor = OutputProcesser(
        send_to_tokenizer=lambda *a, **k: None,
        metrics=SimpleNamespace(),
        defer_to_device=_handle(trace).submit_release,
    )
    state = SimpleNamespace(
        has_pending_multimodal_features=lambda: True,
        release_pending_multimodal_features=lambda: released.append("released"),
    )

    processor._release_multimodal_features(state)

    assert trace == ["submit"]
    assert released == ["released"]


def test_a_request_with_no_pending_features_releases_inline():
    from tokenspeed.runtime.engine.generation_output_processor import OutputProcesser

    trace: list = []
    released: list = []
    processor = OutputProcesser(
        send_to_tokenizer=lambda *a, **k: None,
        metrics=SimpleNamespace(),
        defer_to_device=_handle(trace).submit_release,
    )
    state = SimpleNamespace(
        has_pending_multimodal_features=lambda: False,
        release_pending_multimodal_features=lambda: released.append("released"),
    )

    processor._release_multimodal_features(state)

    assert trace == []
    assert released == ["released"]


# ----------------------------------------------------------------------
# The opt-in dispatch guard: device work on the control thread raises.
# ----------------------------------------------------------------------


def test_the_guard_rejects_cuda_factories_before_they_run():
    """Caught at dispatch, so this holds even on a CUDA-less machine."""
    from tokenspeed.runtime.execution.device import _NoDeviceWork

    with _NoDeviceWork():
        cpu = torch.zeros(3)
        cpu += 1  # CPU work passes
        with pytest.raises(RuntimeError, match="Principle 1"):
            torch.empty(2, device="cuda")


def test_the_guard_scans_tensor_lists():
    from tokenspeed.runtime.execution.device import _NoDeviceWork

    with _NoDeviceWork():
        # aten::cat takes a List[Tensor]; the scan must descend into it.
        out = torch.cat([torch.ones(2), torch.ones(2)])
        assert out.numel() == 4


def test_the_guard_is_on_by_default(monkeypatch):
    import contextlib

    from tokenspeed.runtime.execution.device import (
        _NoDeviceWork,
        maybe_control_plane_guard,
    )

    monkeypatch.delenv("TOKENSPEED_GUARD_CONTROL_PLANE", raising=False)
    assert isinstance(maybe_control_plane_guard(), _NoDeviceWork)
    monkeypatch.setenv("TOKENSPEED_GUARD_CONTROL_PLANE", "0")
    assert isinstance(maybe_control_plane_guard(), contextlib.nullcontext)


def test_metadata_only_ops_are_recognized_by_schema():
    """Views alias without writing; kernels do not. The rule must separate
    them without an op-by-op allowlist."""
    from tokenspeed.runtime.execution.device import _only_aliases_inputs

    assert _only_aliases_inputs(torch.ops.aten.view.default)
    assert _only_aliases_inputs(torch.ops.aten.slice.Tensor)
    # Data-producing and in-place ops both stay banned.
    assert not _only_aliases_inputs(torch.ops.aten.add.Tensor)
    assert not _only_aliases_inputs(torch.ops.aten.clone.default)
    assert not _only_aliases_inputs(torch.ops.aten.copy_.default)


def test_epd_receive_allocation_crosses_through_the_runner(monkeypatch):
    """The job's device steps run wherever the runner says — the engine
    passes ``DeviceHandle.run_embedding_work``, so they land on the forward
    thread; ``None`` (the blocking test wrapper) runs them inline."""
    from tokenspeed.runtime.epd import prefill_admission

    # Force the legacy path: with no pool, the receive buffer must be
    # allocated through the runner.
    monkeypatch.setattr(prefill_admission, "_get_pool", lambda engine, device: None)

    ran: list = []

    def runner(work):
        ran.append(work)
        return work()

    item = SimpleNamespace(
        encode_handshake={
            "bootstrap_host": "h",
            "bootstrap_port": 1,
            "bootstrap_room": 2,
        },
        encoded=None,
        offsets=[(0, 3)],  # 4 encoded tokens
    )

    class _Receiver:
        def __init__(self, manager, addr, room) -> None:
            pass

        def poll(self):  # pragma: no cover — not driven here
            return "Bootstrapped"

    job = prefill_admission.EmbeddingReceiveJob(
        [item],
        SimpleNamespace(engine=SimpleNamespace(register=lambda *a: None)),
        hidden=8,
        num_deepstack=0,
        dtype=torch.float32,
        device="cpu",
        receiver_factory=_Receiver,
        run_device_work=runner,
    )

    assert len(ran) == 1  # exactly the recv-buffer allocation
    assert job._items[0].recv_main.shape == (4, 8)


# ----------------------------------------------------------------------
# Visibility: the loop cannot name a device object, so it cannot keep one.
# ----------------------------------------------------------------------

_RAW_DEVICE_OBJECTS = frozenset(
    {
        "executor",
        "model_executor",
        "attn_backend",
        "draft_attn_backend",
        "token_to_kv_pool",
        "draft_token_to_kv_pool",
        "model_runner",
        "target",
        "draft",
    }
)


def event_loop_module():
    from tokenspeed.runtime.engine import event_loop as module

    return module


def _event_loop_init():
    import inspect

    tree = ast.parse(inspect.getsource(event_loop_module()))
    cls = next(
        node
        for node in tree.body
        if isinstance(node, ast.ClassDef) and node.name == "EventLoop"
    )
    return next(
        node
        for node in cls.body
        if isinstance(node, ast.FunctionDef) and node.name == "__init__"
    )


def test_the_event_loop_never_binds_a_device_object():
    """Locals count too: a name it can write is a name it can misuse later.

    ``build_device_side`` owns the model runners, attention backends and KV
    pools; nothing they produce comes back out except plain facts and the
    handle. This is the property the whole design rests on, so it is asserted
    rather than left to review.
    """
    bound = {
        node.id
        for node in ast.walk(_event_loop_init())
        if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Store)
    }
    assert not (bound & _RAW_DEVICE_OBJECTS)


def test_the_event_loop_stores_only_the_running_handle():
    """The wiring and the build result must not outlive the constructor.

    ``DeviceWiring`` can reach a KV pool and an attention backend. Keeping
    one would put every startup hook back within reach of the running loop,
    which is the whole reason the wiring is a separate object.
    """
    stored = {
        node.attr
        for node in ast.walk(_event_loop_init())
        if isinstance(node, ast.Attribute)
        and isinstance(node.ctx, ast.Store)
        and isinstance(node.value, ast.Name)
        and node.value.id == "self"
    }
    assert "_device" in stored
    assert not (stored & _RAW_DEVICE_OBJECTS)
    assert not (stored & {"device", "wiring", "specs"})


def test_the_running_handle_carries_no_startup_hook():
    """Startup capability grows on DeviceWiring, never on DeviceHandle.

    The wiring list is the one that keeps growing — the next cache tier or
    transport wants a hook that needs a pool. This asserts that growth lands
    where the running loop cannot reach it.
    """
    running = {name for name in vars(DeviceHandle) if not name.startswith("_")}
    startup = {name for name in vars(DeviceWiring) if not name.startswith("_")}
    assert startup
    assert not (running & startup)
    assert not (running & _RAW_DEVICE_OBJECTS)


def test_the_handle_hands_back_no_device_object():
    """Every public name on the handle is an operation, not an object."""
    trace: list = []
    handle = _handle(trace)
    public = {name for name in dir(handle) if not name.startswith("_")}
    assert not (public & _RAW_DEVICE_OBJECTS)
    assert not any(getattr(handle, name, None) is handle._executor for name in public)


def test_only_the_builder_constructs_the_device_side():
    """The name denylist above is a proxy; this is the property itself.

    A device object can only reach the loop if someone constructs one there,
    so pin the constructors: the three factories that produce model runners,
    attention backends, KV pools and the executor are called from
    ``execution/device.py`` alone. (``epd/encode_loop.py`` is a different
    worker — it builds a vision tower and never a ModelExecutor — so it is
    not in scope for the scheduler loop's invariant.)
    """
    import pathlib

    factories = (
        "create_model_runner(",
        "create_attn_components(",
        "create_model_executor(",
    )
    allowed = {"execution/device.py", "execution/factory.py", "epd/encode_loop.py"}
    root = pathlib.Path(__file__).resolve().parents[2] / "python" / "tokenspeed"
    offenders = []
    for path in root.rglob("*.py"):
        rel = path.relative_to(root).as_posix()
        if any(rel.endswith(suffix) for suffix in allowed):
            continue
        text = path.read_text()
        for factory in factories:
            if factory in text and f"def {factory}" not in text:
                offenders.append(f"{rel}: {factory}")
    assert not offenders, offenders


def test_collaborators_hold_the_handle_instead_of_walking_to_it():
    """No ``loop.<x>.<y>`` path to the GPU: each hook is handed its own.

    A traversal is the seam a later change widens — first ``loop.device``,
    then whatever the handle happens to expose. Injection keeps the
    dependency in the constructor where it is visible.
    """
    import inspect

    from tokenspeed.runtime.engine import pause
    from tokenspeed.runtime.pd import transfer_hooks

    for module in (pause, transfer_hooks, event_loop_module()):
        source = inspect.getsource(module)
        assert "loop.device" not in source
        assert "loop._device" not in source
