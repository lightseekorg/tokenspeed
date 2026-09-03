"""Regression tests for logits processing helpers."""

from __future__ import annotations

import os
import sys
from types import SimpleNamespace

# CI Registration (parsed via AST, runtime no-op)
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from ci_system.ci_register import register_cuda_ci  # noqa: E402

register_cuda_ci(est_time=90, suite="runtime-1gpu")

import pytest  # noqa: E402
import torch  # noqa: E402

import tokenspeed.runtime.layers.logits_processor as logits_processor_module  # noqa: E402
from tokenspeed.runtime.execution.forward_batch_info import ForwardMode  # noqa: E402
from tokenspeed.runtime.layers.logits_processor import (  # noqa: E402
    LogitsMetadata,
    LogitsProcessor,
    fused_softcap,
)
from tokenspeed.runtime.utils.env import global_server_args_dict  # noqa: E402


def test_logits_processor_only_uses_fused_lm_head_for_kimi(monkeypatch):
    hidden_states = torch.tensor([[1.0, 2.0]], dtype=torch.float32)
    lm_head = SimpleNamespace(weight=torch.eye(2, dtype=torch.float32))
    metadata = LogitsMetadata(forward_mode=ForwardMode.DECODE)
    calls = {"fused": 0}

    def fake_lm_head_matmul(hidden, weight):
        calls["fused"] += 1
        return torch.matmul(hidden.to(weight.dtype), weight.T)

    monkeypatch.setattr(logits_processor_module, "_lm_head_matmul", fake_lm_head_matmul)

    non_kimi = LogitsProcessor(config=SimpleNamespace(model_type="test", vocab_size=2))
    non_kimi(
        input_ids=None,
        hidden_states=hidden_states,
        lm_head=lm_head,
        logits_metadata=metadata,
    )
    assert calls["fused"] == 0

    kimi = LogitsProcessor(config=SimpleNamespace(model_type="kimi_k2", vocab_size=2))
    kimi(
        input_ids=None,
        hidden_states=hidden_states,
        lm_head=lm_head,
        logits_metadata=metadata,
    )
    assert calls["fused"] == 1


def test_tp_logits_all_gather_handles_zero_rows(monkeypatch):
    processor = LogitsProcessor(
        config=SimpleNamespace(model_type="test", vocab_size=6),
        tp_rank=0,
        tp_size=2,
        tp_group=(0, 1),
    )
    hidden_states = torch.empty((0, 2), dtype=torch.float32)
    lm_head = SimpleNamespace(weight=torch.ones((3, 2), dtype=torch.float32))
    metadata = LogitsMetadata(forward_mode=ForwardMode.DECODE)
    calls = {"all_gather": 0}

    def fake_all_gather_into_tensor(output, input_, group):
        calls["all_gather"] += 1
        assert group == (0, 1)
        assert tuple(output.shape) == (0, 3)
        assert tuple(input_.shape) == (0, 3)

    monkeypatch.setattr(
        logits_processor_module,
        "all_gather_into_tensor",
        fake_all_gather_into_tensor,
    )

    output = processor(
        input_ids=None,
        hidden_states=hidden_states,
        lm_head=lm_head,
        logits_metadata=metadata,
    )

    assert calls["all_gather"] == 1
    assert tuple(output.next_token_logits.shape) == (0, 6)


@pytest.mark.parametrize(
    "initializer_name",
    ["_init_all_gather_state", "_init_dist_argmax_state"],
)
def test_force_deterministic_rsag_disables_logits_symm_mem(
    monkeypatch, initializer_name
):
    monkeypatch.setitem(global_server_args_dict, "force_deterministic_rsag", True)
    monkeypatch.setattr(
        logits_processor_module,
        "create_state",
        lambda *args, **kwargs: pytest.fail(
            "symmetric-memory state must not initialize"
        ),
    )
    monkeypatch.setattr(
        logits_processor_module,
        "try_create_dist_argmax_state",
        lambda *args, **kwargs: pytest.fail(
            "symmetric-memory state must not initialize"
        ),
    )
    processor = LogitsProcessor(
        config=SimpleNamespace(model_type="test", vocab_size=8),
        tp_rank=0,
        tp_size=2,
        tp_group=(0, 1),
    )

    assert getattr(processor, initializer_name)(SimpleNamespace()) is None


def _set_fabric(monkeypatch, supported: bool) -> None:
    import tokenspeed_kernel.ops.communication.fabric as fabric

    # The topology is what makes these groups host-spread; without it the tests
    # would name a property their own setup never established.
    monkeypatch.setitem(
        global_server_args_dict, "mapping", SimpleNamespace(nprocs_per_node=4)
    )
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 4)
    monkeypatch.setattr(torch.cuda, "current_device", lambda: 0)
    monkeypatch.setattr(
        fabric, "fabric_allocation_supported", lambda device_index: supported
    )
    # The gate votes; stand in for the collective so the local answer survives.
    logits_processor_module.LogitsProcessor._LOGITS_MC_REACHABLE.clear()
    monkeypatch.setattr(
        logits_processor_module.pg_manager, "get_process_group", lambda *a: "pg"
    )
    monkeypatch.setattr(
        torch.distributed, "all_reduce", lambda tensor, op=None, group=None: None
    )


def test_tp_logits_custom_collectives_skip_host_spread_group_without_fabric(
    monkeypatch,
):
    _set_fabric(monkeypatch, False)
    processor = LogitsProcessor(
        config=SimpleNamespace(model_type="test", vocab_size=64),
        tp_rank=0,
        tp_size=8,
        tp_group=tuple(range(8)),
    )
    lm_head = SimpleNamespace(weight=torch.ones((8, 2), dtype=torch.float32))

    monkeypatch.setattr(
        logits_processor_module,
        "create_state",
        lambda **kwargs: pytest.fail("a group without fabric must not gather"),
    )
    # Distributed argmax is no longer topology-gated: cross-node groups probe
    # for NVLS instead. This shard is below the kernel's vocab floor, so the
    # state is rejected before any collective work.
    monkeypatch.setattr(
        logits_processor_module,
        "try_create_dist_argmax_state",
        lambda **kwargs: pytest.fail(
            "a shard below the vocab floor must not reach the constructor"
        ),
    )

    assert processor._init_all_gather_state(lm_head) is None
    assert processor._init_dist_argmax_state(lm_head) is None


def test_a_strided_tp_group_smaller_than_one_host_is_still_probed(monkeypatch):
    """Two ranks on two hosts is fewer ranks than one host holds.

    Sizing the group against the local device count would admit it with no
    probe, and a group the fabric cannot map hangs in the rendezvous.
    """
    _set_fabric(monkeypatch, False)
    processor = LogitsProcessor(
        config=SimpleNamespace(model_type="test", vocab_size=64),
        tp_rank=0,
        tp_size=2,
        tp_group=(0, 4),
    )
    assert not processor._tp_group_multicast_reachable()


def test_a_peer_without_fabric_takes_the_whole_group_off_the_gather(monkeypatch):
    """The probe allocates on this device alone, so a lone no must carry.

    One node with no IMEX channels answers no while its peers answer yes; the
    yes-ranks would then block in a rendezvous the no-ranks never enter.
    """
    _set_fabric(monkeypatch, True)
    monkeypatch.setattr(
        torch.distributed,
        "all_reduce",
        lambda tensor, op=None, group=None: tensor.zero_(),
    )
    processor = LogitsProcessor(
        config=SimpleNamespace(model_type="test", vocab_size=64),
        tp_rank=0,
        tp_size=8,
        tp_group=tuple(range(8)),
    )
    assert not processor._tp_group_multicast_reachable()


def test_tp_logits_custom_collectives_serve_host_spread_group_with_fabric(monkeypatch):
    """An NVLink domain can span hosts, so fabric decides, not the host count."""
    _set_fabric(monkeypatch, True)
    processor = LogitsProcessor(
        config=SimpleNamespace(model_type="test", vocab_size=64),
        tp_rank=0,
        tp_size=8,
        tp_group=tuple(range(8)),
    )
    lm_head = SimpleNamespace(weight=torch.ones((8, 2), dtype=torch.float32))
    created = {}

    def _create_state(**kwargs):
        created.update(kwargs)
        return "ag-state"

    monkeypatch.setattr(logits_processor_module, "create_state", _create_state)
    monkeypatch.setattr(
        logits_processor_module.pg_manager,
        "get_process_group",
        lambda backend, group: "pg",
    )

    try:
        assert processor._init_all_gather_state(lm_head) == "ag-state"
        assert created["hidden_size"] == 64
    finally:
        # The cache is class-level; drop the stub so it cannot leak.
        LogitsProcessor._LOGITS_AG_STATES.pop((tuple(range(8)), 64), None)


def test_dist_argmax_probe_failure_falls_back_and_latches(monkeypatch):
    """A failed probe falls back, latches its verdict, and skips capture."""
    monkeypatch.setitem(
        global_server_args_dict,
        "mapping",
        SimpleNamespace(nprocs_per_node=4),
    )
    monkeypatch.setattr(logits_processor_module, "dist_argmax_available", lambda: True)
    monkeypatch.setattr(
        logits_processor_module,
        "current_platform",
        lambda: SimpleNamespace(is_nvidia=True),
    )
    capturing = {"on": True}
    monkeypatch.setattr(
        logits_processor_module.torch.cuda,
        "is_current_stream_capturing",
        lambda: capturing["on"],
    )
    monkeypatch.setattr(
        logits_processor_module.pg_manager,
        "get_process_group",
        lambda *a, **k: object(),
    )
    monkeypatch.setattr(
        logits_processor_module.torch.distributed,
        "all_reduce",
        lambda tensor, **k: None,
    )
    calls = []

    def failing_create(**kwargs):
        calls.append(1)
        return None  # the group has no NVLS multicast

    monkeypatch.setattr(
        logits_processor_module, "try_create_dist_argmax_state", failing_create
    )
    processor = LogitsProcessor(
        config=SimpleNamespace(model_type="test", vocab_size=8192),
        tp_rank=0,
        tp_size=2,
        tp_group=(0, 4),
    )
    lm_head = SimpleNamespace(weight=torch.ones((4096, 2), dtype=torch.float32))

    # Inside capture: no collective work, and nothing may latch.
    assert processor._init_dist_argmax_state(lm_head) is None
    assert len(calls) == 0

    capturing["on"] = False
    assert processor._init_dist_argmax_state(lm_head) is None
    # The verdict is cached: the constructor must not be retried.
    assert processor._init_dist_argmax_state(lm_head) is None
    assert len(calls) == 1


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_fused_softcap_handles_large_logits_without_nan():
    cap = 30.0
    logits = torch.tensor(
        [[5000.0, 2000.0, 1500.0, 100.0, 0.0, -100.0, -1500.0, -5000.0]],
        device="cuda",
        dtype=torch.float32,
    )
    expected = cap * torch.tanh(logits / cap)

    out = fused_softcap(logits.clone(), cap)
    torch.cuda.synchronize()

    assert torch.isfinite(out).all()
    torch.testing.assert_close(out, expected, rtol=1e-5, atol=2e-5)


def test_argmax_routes_sharded_to_kernel(monkeypatch):
    """Sharded logits (tp shards reconstruct vocab) hit the fused kernel."""
    proc = LogitsProcessor(
        config=SimpleNamespace(model_type="test", vocab_size=8),
        tp_rank=0,
        tp_size=2,
        tp_group=(0, 1),
    )
    proc._dist_argmax_state = object()  # non-None, non-sentinel => active

    recorded = {}

    def fake_dist(state, logits):
        recorded["called"] = True
        return None, logits.argmax(dim=-1)

    monkeypatch.setattr(logits_processor_module, "distributed_argmax", fake_dist)

    shard = torch.randn(4, 4, dtype=torch.float32)  # 4 * tp_size(2) == vocab_size(8)
    ids = proc._argmax(shard)
    assert recorded.get("called")
    assert torch.equal(ids, shard.argmax(dim=-1))


def test_argmax_falls_back_without_state(monkeypatch):
    """No fused state (e.g. EAGLE3 draft vocab != target, or gate failed):
    _argmax falls back to a plain argmax instead of routing to the kernel."""
    proc = LogitsProcessor(
        config=SimpleNamespace(model_type="test", vocab_size=100),
        tp_rank=0,
        tp_size=2,
        tp_group=(0, 1),
    )
    proc._dist_argmax_state = None  # gate failed (draft vocab != target vocab)

    monkeypatch.setattr(
        logits_processor_module,
        "distributed_argmax",
        lambda *a, **k: pytest.fail("kernel must not run without a fused state"),
    )

    # Gathered draft logits are narrower than the target config.vocab_size.
    draft = torch.randn(4, 32, dtype=torch.float32)
    ids = proc._argmax(draft)
    assert torch.equal(ids, draft.argmax(dim=-1))


def test_get_logits_skips_gather_when_dist_argmax_active(monkeypatch):
    """do_argmax + active state keeps logits sharded (no all-gather)."""
    proc = LogitsProcessor(
        config=SimpleNamespace(
            model_type="test", vocab_size=8, final_logit_softcapping=None
        ),
        tp_rank=0,
        tp_size=2,
        tp_group=(0, 1),
        do_argmax=True,
    )
    monkeypatch.setattr(proc, "_init_dist_argmax_state", lambda lm_head: object())
    monkeypatch.setattr(
        logits_processor_module,
        "all_gather_inner",
        lambda *a, **k: pytest.fail("gather must be skipped on the fused path"),
    )

    hidden = torch.randn(4, 2, dtype=torch.float32)
    lm_head = SimpleNamespace(weight=torch.randn(4, 2, dtype=torch.float32))  # 4*2 == 8
    md = LogitsMetadata(forward_mode=ForwardMode.DECODE)
    out = proc._get_logits(hidden, lm_head, md)
    assert out.shape == (4, 4)  # local shard width retained, not gathered to 8


def test_capture_takes_the_plain_gather_and_leaves_the_gate_for_later(monkeypatch):
    """The uninitialised sentinel must not be mistaken for a built state.

    The gate reduces across the group, so it is skipped inside a capture. The
    sentinel is an ``object()`` and so passes ``is not None``: leaving it in
    place would hand it to ``all_gather_inner`` as if it were a state. It must
    also survive, or an eager call afterwards would never build the real one.
    """
    proc = LogitsProcessor(
        config=SimpleNamespace(
            model_type="test", vocab_size=8, final_logit_softcapping=None
        ),
        tp_rank=0,
        tp_size=2,
        tp_group=(0, 1),
    )
    monkeypatch.setattr(
        proc,
        "_init_all_gather_state",
        lambda lm_head: pytest.fail("the gate must not run inside a capture"),
    )
    monkeypatch.setattr(
        logits_processor_module,
        "all_gather_inner",
        lambda *a, **k: pytest.fail("the sentinel must never reach the gather"),
    )
    monkeypatch.setattr(torch.cuda, "is_current_stream_capturing", lambda: True)
    monkeypatch.setattr(
        logits_processor_module,
        "all_gather_into_tensor",
        lambda out, inp, group: None,
    )

    hidden = torch.randn(4, 2, dtype=torch.float32)
    lm_head = SimpleNamespace(weight=torch.randn(4, 2, dtype=torch.float32))
    md = LogitsMetadata(forward_mode=ForwardMode.DECODE)
    out = proc._get_logits(hidden, lm_head, md)

    assert out.shape == (4, 8)
    assert proc._all_gather_state is LogitsProcessor._LOGITS_AG_STATE_UNINITIALIZED


def test_get_logits_softcap_disables_fused_argmax(monkeypatch):
    """final_logit_softcapping must disable the fused early-return so the
    softcap is applied to full-vocab logits (then a plain argmax runs)."""
    proc = LogitsProcessor(
        config=SimpleNamespace(
            model_type="test", vocab_size=8, final_logit_softcapping=30.0
        ),
        tp_rank=0,
        tp_size=2,
        tp_group=(0, 1),
        do_argmax=True,
    )
    # Fused state is otherwise eligible; softcap must still force the gather.
    monkeypatch.setattr(proc, "_init_dist_argmax_state", lambda lm_head: object())
    monkeypatch.setattr(proc, "_init_all_gather_state", lambda lm_head: object())
    called = {}

    def fake_ag(state, logits, **kw):
        called["ag"] = True
        return logits.repeat(1, proc.tp_size)  # [bs, vocab/tp] -> [bs, vocab]

    monkeypatch.setattr(logits_processor_module, "all_gather_inner", fake_ag)
    monkeypatch.setattr(
        logits_processor_module, "fused_softcap_generic", lambda *a, **k: None
    )

    hidden = torch.randn(4, 2, dtype=torch.float32)
    lm_head = SimpleNamespace(weight=torch.randn(4, 2, dtype=torch.float32))  # 4*2 == 8
    md = LogitsMetadata(forward_mode=ForwardMode.DECODE)
    out = proc._get_logits(hidden, lm_head, md)
    assert called.get("ag")  # gathered (softcap on full vocab), not early-returned
    assert out.shape == (4, 8)
