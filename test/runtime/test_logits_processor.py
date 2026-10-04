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

import tokenspeed.runtime.distributed.comm_manager as comm_manager_module  # noqa: E402
import tokenspeed.runtime.layers.logits_processor as logits_processor_module  # noqa: E402
from tokenspeed.runtime.execution.context import InputLogprobRows  # noqa: E402
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
    metadata = LogitsMetadata(forward_mode=ForwardMode.DECODE, query_shard=None)
    calls = {"fused": 0}

    def fake_lm_head_matmul(hidden, weight):
        calls["fused"] += 1
        return torch.matmul(hidden.to(weight.dtype), weight.T)

    monkeypatch.setattr(logits_processor_module, "_lm_head_matmul", fake_lm_head_matmul)

    non_kimi = LogitsProcessor(
        config=SimpleNamespace(model_type="test", vocab_size=2), dp_lm_head_tp=False
    )
    non_kimi(
        input_ids=None,
        hidden_states=hidden_states,
        lm_head=lm_head,
        logits_metadata=metadata,
    )
    assert calls["fused"] == 0

    kimi = LogitsProcessor(
        config=SimpleNamespace(model_type="kimi_k2", vocab_size=2), dp_lm_head_tp=False
    )
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
        dp_lm_head_tp=False,
    )
    hidden_states = torch.empty((0, 2), dtype=torch.float32)
    lm_head = SimpleNamespace(weight=torch.ones((3, 2), dtype=torch.float32))
    metadata = LogitsMetadata(forward_mode=ForwardMode.DECODE, query_shard=None)
    calls = {"all_gather": 0}

    def fake_all_gather_single(output, input_, group):
        calls["all_gather"] += 1
        assert group == (0, 1)
        assert tuple(output.shape) == (0, 3)
        assert tuple(input_.shape) == (0, 3)

    monkeypatch.setattr(
        logits_processor_module,
        "all_gather_single",
        fake_all_gather_single,
    )

    output = processor(
        input_ids=None,
        hidden_states=hidden_states,
        lm_head=lm_head,
        logits_metadata=metadata,
    )

    assert calls["all_gather"] == 1
    assert tuple(output.next_token_logits.shape) == (0, 6)


@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
@pytest.mark.parametrize("cached", [False, True])
def test_tp_logits_gather_preserves_dtype(monkeypatch, dtype, cached):
    processor = LogitsProcessor(
        config=SimpleNamespace(model_type="test", vocab_size=16),
        skip_all_gather=False,
        do_argmax=False,
        logit_scale=None,
        tp_rank=0,
        tp_size=2,
        tp_group=(0, 1),
        dp_lm_head_tp=False,
    )
    state = object()
    if cached:
        processor._all_gather_state = state
    calls = []

    def initialize(lm_head):
        assert dtype == torch.bfloat16
        calls.append("init")
        return state

    def multicast(received_state, logits, *, tp_hidden_dim, skip_entry_sync, safe):
        assert received_state is state
        assert logits.dtype == torch.bfloat16
        assert tp_hidden_dim == 16 and skip_entry_sync and not safe
        calls.append("multicast")
        return torch.cat((logits, logits), dim=-1)

    def collective(output, logits, group):
        assert dtype != torch.bfloat16
        assert output.dtype == logits.dtype == dtype
        assert group == (0, 1)
        calls.append("collective")
        output.copy_(torch.cat((logits, logits), dim=0))

    monkeypatch.setattr(processor, "_init_all_gather_state", initialize)
    monkeypatch.setattr(torch.cuda, "is_current_stream_capturing", lambda: False)
    monkeypatch.setattr(logits_processor_module, "all_gather_inner", multicast)
    monkeypatch.setattr(logits_processor_module, "all_gather_single", collective)
    hidden = torch.tensor([[1.0, 1 / 512]], dtype=dtype)
    weight = torch.zeros((8, 2), dtype=dtype)
    weight[0, 0] = 1
    weight[1] = 1
    local = hidden @ weight.T
    output = processor._get_logits(
        hidden,
        SimpleNamespace(weight=weight),
        logits_metadata=None,
        embedding_bias=None,
        plan=None,
        require_full_vocab=False,
    )
    assert output.dtype == dtype
    torch.testing.assert_close(
        output, torch.cat((local, local), dim=-1), rtol=0, atol=0
    )
    if dtype == torch.bfloat16:
        assert calls == (["multicast"] if cached else ["init", "multicast"])
    else:
        assert calls == ["collective"]
        assert output.argmax(-1).item() == 1


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
        dp_lm_head_tp=False,
    )

    assert getattr(processor, initializer_name)(SimpleNamespace()) is None


def _set_fabric(monkeypatch, supported: bool) -> None:
    import tokenspeed_kernel.ops.communication.fabric as fabric

    # These tests model NVIDIA multicast regardless of the runner's platform.
    monkeypatch.setattr(
        logits_processor_module,
        "current_platform",
        lambda: SimpleNamespace(is_nvidia=True),
    )
    # The topology is what makes these groups host-spread; without it the tests
    # would name a property their own setup never established.
    monkeypatch.setitem(
        global_server_args_dict, "mapping", SimpleNamespace(nprocs_per_node=4)
    )
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 4)
    monkeypatch.setattr(fabric, "group_has_fabric", lambda ranks: supported)


def test_tp_logits_custom_collectives_skip_host_spread_group_without_fabric(
    monkeypatch,
):
    _set_fabric(monkeypatch, False)
    processor = LogitsProcessor(
        config=SimpleNamespace(model_type="test", vocab_size=64),
        tp_rank=0,
        tp_size=8,
        tp_group=tuple(range(8)),
        dp_lm_head_tp=False,
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
        dp_lm_head_tp=False,
    )
    assert not processor._tp_group_multicast_reachable()


def test_a_peer_without_fabric_takes_the_whole_group_off_the_gather(monkeypatch):
    """The probe allocates on this device alone, so a lone no must carry.

    One node with no IMEX channels answers no while its peers answer yes; the
    yes-ranks would then block in a rendezvous the no-ranks never enter.
    """
    _set_fabric(monkeypatch, True)
    import tokenspeed_kernel.ops.communication.fabric as fabric

    monkeypatch.setattr(
        fabric,
        "group_has_fabric",
        lambda ranks: False,
    )
    processor = LogitsProcessor(
        config=SimpleNamespace(model_type="test", vocab_size=64),
        tp_rank=0,
        tp_size=8,
        tp_group=tuple(range(8)),
        dp_lm_head_tp=False,
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
        dp_lm_head_tp=False,
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
        dp_lm_head_tp=False,
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


def test_dist_argmax_state_cache_separates_logits_dtypes(monkeypatch):
    """An FP32 corrected-logits user must not reuse a BF16 sampler state."""
    monkeypatch.setattr(logits_processor_module, "dist_argmax_available", lambda: True)
    monkeypatch.setattr(
        logits_processor_module,
        "current_platform",
        lambda: SimpleNamespace(is_nvidia=True),
    )
    monkeypatch.setattr(
        logits_processor_module.torch.cuda,
        "is_current_stream_capturing",
        lambda: False,
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
    created = []

    def fake_create(**kwargs):
        created.append(kwargs["dtype"])
        return SimpleNamespace(dtype=kwargs["dtype"])

    monkeypatch.setattr(
        logits_processor_module, "try_create_dist_argmax_state", fake_create
    )
    monkeypatch.setattr(LogitsProcessor, "_LOGITS_DIST_ARGMAX_STATES", {})
    processor = LogitsProcessor(
        config=SimpleNamespace(model_type="test", vocab_size=8192),
        tp_rank=0,
        tp_size=2,
        tp_group=(0, 1),
        dp_lm_head_tp=False,
    )
    lm_head = SimpleNamespace(weight=torch.ones((4096, 2), dtype=torch.bfloat16))

    bf16_state = processor.acquire_dist_argmax_state(
        lm_head, max_M=8, skip_ping_pong=False, dtype=torch.bfloat16
    )
    fp32_state = processor.acquire_dist_argmax_state(
        lm_head, max_M=8, skip_ping_pong=False, dtype=torch.float32
    )
    assert bf16_state.dtype == torch.bfloat16
    assert fp32_state.dtype == torch.float32
    assert created == [torch.bfloat16, torch.float32]

    # Both verdicts are independently cached.
    assert (
        processor.acquire_dist_argmax_state(
            lm_head, max_M=8, skip_ping_pong=False, dtype=torch.float32
        )
        is fp32_state
    )
    assert len(created) == 2


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
        dp_lm_head_tp=False,
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
        dp_lm_head_tp=False,
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
        dp_lm_head_tp=False,
    )
    monkeypatch.setattr(proc, "_init_dist_argmax_state", lambda lm_head: object())
    monkeypatch.setattr(
        logits_processor_module,
        "all_gather_inner",
        lambda *a, **k: pytest.fail("gather must be skipped on the fused path"),
    )

    hidden = torch.randn(4, 2, dtype=torch.float32)
    lm_head = SimpleNamespace(weight=torch.randn(4, 2, dtype=torch.float32))  # 4*2 == 8
    md = LogitsMetadata(forward_mode=ForwardMode.DECODE, query_shard=None)
    out = proc._get_logits(hidden, lm_head, md, require_full_vocab=False)
    assert out.shape == (4, 4)  # local shard width retained, not gathered to 8


def test_require_full_vocab_logits_turns_the_fused_draft_argmax_off(monkeypatch):
    """A consumer that samples from the draft distribution asks for the
    full-vocab gather explicitly; the draft model keeps constructing with
    do_argmax=True and no global flag is consulted."""
    proc = LogitsProcessor(
        config=SimpleNamespace(
            model_type="test", vocab_size=8, final_logit_softcapping=None
        ),
        tp_rank=0,
        tp_size=2,
        tp_group=(0, 1),
        do_argmax=True,
        dp_lm_head_tp=False,
    )
    assert proc.do_argmax
    proc.require_full_vocab_logits()
    assert not proc.do_argmax

    monkeypatch.setattr(
        proc,
        "_init_dist_argmax_state",
        lambda lm_head: pytest.fail("the fused argmax gate must stay off"),
    )
    monkeypatch.setattr(proc, "_init_all_gather_state", lambda lm_head: None)
    monkeypatch.setattr(
        logits_processor_module, "all_gather_single", lambda out, inp, group: None
    )
    hidden = torch.randn(4, 2, dtype=torch.float32)
    lm_head = SimpleNamespace(weight=torch.randn(4, 2, dtype=torch.float32))
    md = LogitsMetadata(forward_mode=ForwardMode.DECODE, query_shard=None)
    out = proc._get_logits(hidden, lm_head, md, require_full_vocab=True)
    assert out.shape == (4, 8)  # gathered to the full vocab


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
        dp_lm_head_tp=False,
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
        "all_gather_single",
        lambda out, inp, group: None,
    )

    hidden = torch.randn(4, 2, dtype=torch.float32)
    lm_head = SimpleNamespace(weight=torch.randn(4, 2, dtype=torch.float32))
    md = LogitsMetadata(forward_mode=ForwardMode.DECODE, query_shard=None)
    out = proc._get_logits(hidden, lm_head, md, require_full_vocab=False)

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
        dp_lm_head_tp=False,
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

    hidden = torch.randn(4, 2, dtype=torch.bfloat16)
    lm_head = SimpleNamespace(
        weight=torch.randn(4, 2, dtype=torch.bfloat16)
    )  # 4*2 == 8
    md = LogitsMetadata(forward_mode=ForwardMode.DECODE, query_shard=None)
    out = proc._get_logits(hidden, lm_head, md, require_full_vocab=False)
    assert called.get("ag")  # gathered (softcap on full vocab), not early-returned
    assert out.shape == (4, 8)


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))


def _logprob_device() -> str:
    return "cuda" if torch.cuda.is_available() else "cpu"


def _input_logprob_rows(
    rows, targets, *, num_input_rows, chunk_tokens, device, rows_per_rank=None
):
    return InputLogprobRows(
        rows=torch.tensor(rows, dtype=torch.int64, device=device),
        targets=torch.tensor(targets, dtype=torch.int64, device=device),
        slots=torch.zeros(
            len(rows) if rows_per_rank is None else sum(rows_per_rank),
            dtype=torch.int64,
            device=device,
        ),
        num_input_rows=num_input_rows,
        chunk_tokens=chunk_tokens,
        rows_per_rank=rows_per_rank,
    )


def test_gather_token_logprobs_widens_inside_the_kernel_bitwise():
    """The ``dtype=float32`` log-softmax (no fp32 copy of the logits) is the
    same arithmetic as ``log_softmax(logits.float())``, bit for bit."""
    from tokenspeed.runtime.sampling.utils import gather_token_logprobs_torch

    torch.manual_seed(3)
    for dtype in (torch.bfloat16, torch.float16, torch.float32):
        logits = (torch.randn(7, 1000) * 8).to(dtype)
        tokens = torch.randint(0, 1000, (7,))
        got = gather_token_logprobs_torch(logits, tokens)
        reference = (
            torch.log_softmax(logits.float(), dim=-1)
            .gather(-1, tokens.unsqueeze(-1))
            .squeeze(-1)
        )
        assert got.dtype == torch.float32
        assert torch.equal(got, reference), dtype


@pytest.mark.parametrize("chunk_tokens", [1, 2, 3, 64])
def test_input_logprobs_match_the_output_logprob_arithmetic(chunk_tokens):
    """Prompt logprobs are the sampler's fp32 ``log_softmax(...).gather`` on
    the same ``_get_logits`` route, independent of the position chunk size,
    and leave the sampled logits of the last row per request untouched."""
    from tokenspeed.runtime.sampling.utils import gather_token_logprobs_torch

    device = _logprob_device()
    torch.manual_seed(0)
    vocab = 6
    # Two requests of 3 and 2 tokens; prompt logprobs for rows 1, 2 of the
    # first and row 3 (position 0) of the second.
    # Keep the four-term dot products exact so cuBLAS's batch-dependent
    # reduction order cannot change the prompt or sampled logits. This tests
    # logprob arithmetic and row selection, not GEMM batch invariance in auto.
    hidden = torch.randint(-8, 9, (5, 4), device=device).float() / 8
    weight = torch.randint(-8, 9, (vocab, 4), device=device).float() / 8
    lm_head = SimpleNamespace(weight=weight)
    rows, targets = [1, 2, 3], [5, 2, 3]
    processor = LogitsProcessor(
        config=SimpleNamespace(model_type="test", vocab_size=vocab),
        dp_lm_head_tp=False,
    )
    metadata = LogitsMetadata(
        forward_mode=ForwardMode.EXTEND,
        query_shard=None,
        gather_ids=torch.tensor([2, 4], device=device),
        input_logprob_rows=_input_logprob_rows(
            rows, targets, num_input_rows=5, chunk_tokens=chunk_tokens, device=device
        ),
    )

    out = processor(
        input_ids=None,
        hidden_states=hidden,
        lm_head=lm_head,
        logits_metadata=metadata,
    )

    logits = hidden @ weight.T
    torch.testing.assert_close(out.next_token_logits, logits[[2, 4]], rtol=0, atol=0)
    expected = gather_token_logprobs_torch(
        logits[rows], torch.tensor(targets, device=device)
    )
    assert out.input_token_logprobs.dtype == torch.float32
    torch.testing.assert_close(out.input_token_logprobs, expected, rtol=0, atol=0)


def test_input_logprobs_are_gathered_from_prefill_rows_of_a_mixed_batch():
    """Decode rows sit behind the prefill rows; the plan only names prefill
    positions, so the decode rows are never pushed through the gather."""
    device = _logprob_device()
    torch.manual_seed(1)
    vocab = 8
    hidden = torch.randn(6, 4, device=device)  # 4 prefill rows + 2 decode rows
    weight = torch.randn(vocab, 4, device=device)
    processor = LogitsProcessor(
        config=SimpleNamespace(model_type="test", vocab_size=vocab), dp_lm_head_tp=False
    )
    seen = []
    original = processor._get_logits

    def spy(hidden_states, *args, **kwargs):
        seen.append(hidden_states.shape[0])
        return original(hidden_states, *args, **kwargs)

    processor._get_logits = spy
    metadata = LogitsMetadata(
        forward_mode=ForwardMode.MIXED,
        query_shard=None,
        gather_ids=torch.tensor([3, 4, 5], device=device),
        input_logprob_rows=_input_logprob_rows(
            [0, 1, 2], [1, 2, 3], num_input_rows=6, chunk_tokens=2, device=device
        ),
    )

    out = processor(
        input_ids=None,
        hidden_states=hidden,
        lm_head=SimpleNamespace(weight=weight),
        logits_metadata=metadata,
    )

    # Two prompt chunks (2 + 1 rows) then the three sampled rows.
    assert seen == [2, 1, 3]
    logits = hidden @ weight.T
    expected = torch.log_softmax(logits[:3].float(), dim=-1)[
        torch.arange(3, device=device), torch.tensor([1, 2, 3], device=device)
    ]
    torch.testing.assert_close(out.input_token_logprobs, expected, rtol=0, atol=0)
    assert out.next_token_logits.shape == (3, vocab)


def test_input_logprobs_refuse_a_model_that_narrowed_its_logits_rows():
    device = _logprob_device()
    processor = LogitsProcessor(
        config=SimpleNamespace(model_type="test", vocab_size=4), dp_lm_head_tp=False
    )
    lm_head = SimpleNamespace(weight=torch.randn(4, 2, device=device))
    rows = _input_logprob_rows(
        [0, 1], [1, 2], num_input_rows=3, chunk_tokens=8, device=device
    )

    # A narrowing model hands over only its selected rows ...
    with pytest.raises(ValueError, match="narrowed"):
        processor(
            input_ids=None,
            hidden_states=torch.randn(1, 2, device=device),
            lm_head=lm_head,
            logits_metadata=LogitsMetadata(
                forward_mode=ForwardMode.EXTEND,
                query_shard=None,
                gather_ids=torch.tensor([0], device=device),
                logits_rows_selected=True,
                input_logprob_rows=rows,
            ),
        )
    # ... and so does any forward whose activations do not cover every input row.
    with pytest.raises(ValueError, match="narrowed"):
        processor(
            input_ids=None,
            hidden_states=torch.randn(2, 2, device=device),
            lm_head=lm_head,
            logits_metadata=LogitsMetadata(
                forward_mode=ForwardMode.EXTEND,
                query_shard=None,
                gather_ids=torch.tensor([1], device=device),
                input_logprob_rows=rows,
            ),
        )
    # A cache-only chunk without logits rows cannot provide them either.
    with pytest.raises(ValueError, match="selected logits rows"):
        processor(
            input_ids=None,
            hidden_states=torch.empty(0, 2, device=device),
            lm_head=lm_head,
            logits_metadata=LogitsMetadata(
                forward_mode=ForwardMode.EXTEND,
                query_shard=None,
                logits_rows_selected=True,
                input_logprob_rows=rows,
            ),
        )


def test_input_logprobs_bypass_the_sharded_argmax_shortcut(monkeypatch):
    """A ``do_argmax`` head keeps sampled logits TP-sharded for the fused
    argmax; the prompt-logprob gather needs the whole vocabulary."""
    proc = LogitsProcessor(
        config=SimpleNamespace(
            model_type="test", vocab_size=8, final_logit_softcapping=None
        ),
        tp_rank=0,
        tp_size=2,
        tp_group=(0, 1),
        do_argmax=True,
        dp_lm_head_tp=False,
    )
    monkeypatch.setattr(proc, "_init_dist_argmax_state", lambda lm_head: object())
    monkeypatch.setattr(proc, "_init_all_gather_state", lambda lm_head: None)
    monkeypatch.setattr(torch.cuda, "is_current_stream_capturing", lambda: False)
    monkeypatch.setattr(
        logits_processor_module,
        "distributed_argmax",
        lambda state, logits: (None, logits.argmax(dim=-1)),
    )
    gathered = []

    def collective(output, logits, group):
        gathered.append(logits.shape)
        output.copy_(torch.cat((logits, logits), dim=0))

    monkeypatch.setattr(logits_processor_module, "all_gather_single", collective)
    hidden = torch.randn(3, 2, dtype=torch.float32)
    lm_head = SimpleNamespace(weight=torch.randn(4, 2, dtype=torch.float32))
    md = LogitsMetadata(
        forward_mode=ForwardMode.EXTEND,
        query_shard=None,
        gather_ids=torch.tensor([2]),
        input_logprob_rows=_input_logprob_rows(
            [0, 1], [1, 6], num_input_rows=3, chunk_tokens=8, device="cpu"
        ),
    )

    out = proc(
        input_ids=None, hidden_states=hidden, lm_head=lm_head, logits_metadata=md
    )

    # The prompt rows were gathered to the full vocab (target id 6 lives on the
    # other shard); the sampled row kept the shortcut.
    assert gathered == [(2, 4)]
    assert out.input_token_logprobs.shape == (2,)
    assert out.next_token_logits.shape == (1, 4)


def test_input_logprob_chunks_never_take_the_multicast_gather(monkeypatch):
    """The multicast all-gather hands back a view of the TP group's shared
    buffer with no entry barrier; the next chunk's gather on a faster rank
    would overwrite it under this rank's log-softmax. The chunk loop must go
    through the NCCL collective into a private tensor, while the sampled rows
    keep the multicast path."""
    proc = LogitsProcessor(
        config=SimpleNamespace(
            model_type="test", vocab_size=8, final_logit_softcapping=None
        ),
        tp_rank=0,
        tp_size=2,
        tp_group=(0, 1),
        dp_lm_head_tp=False,
    )
    multicast_state = object()
    monkeypatch.setattr(proc, "_init_all_gather_state", lambda lm_head: multicast_state)
    monkeypatch.setattr(torch.cuda, "is_current_stream_capturing", lambda: False)
    multicast_rows: list[int] = []
    collective_rows: list[int] = []

    def multicast(state, logits, *, tp_hidden_dim, skip_entry_sync, safe):
        assert state is multicast_state and skip_entry_sync and not safe
        multicast_rows.append(logits.shape[0])
        return torch.cat((logits, logits), dim=-1)

    def collective(output, logits, group):
        collective_rows.append(logits.shape[0])
        output.copy_(torch.cat((logits, logits), dim=0))

    monkeypatch.setattr(logits_processor_module, "all_gather_inner", multicast)
    monkeypatch.setattr(logits_processor_module, "all_gather_single", collective)
    # bf16 logits: the only dtype the multicast path accepts.
    hidden = torch.randn(5, 2, dtype=torch.bfloat16)
    lm_head = SimpleNamespace(weight=torch.randn(4, 2, dtype=torch.bfloat16))
    md = LogitsMetadata(
        forward_mode=ForwardMode.EXTEND,
        query_shard=None,
        gather_ids=torch.tensor([4]),
        input_logprob_rows=_input_logprob_rows(
            [0, 1, 2, 3], [1, 6, 2, 5], num_input_rows=5, chunk_tokens=3, device="cpu"
        ),
    )

    out = proc(
        input_ids=None, hidden_states=hidden, lm_head=lm_head, logits_metadata=md
    )

    # Two prompt chunks (3 + 1 rows) through the collective; one sampled row
    # through the multicast buffer.
    assert collective_rows == [3, 1]
    assert multicast_rows == [1]
    assert out.input_token_logprobs.shape == (4,)
    assert out.next_token_logits.shape == (1, 8)


# --------------------------------------------------------------------------
# Query context parallelism: the shard is a parameter of the row selection
# --------------------------------------------------------------------------


def _sharded_processor(monkeypatch, size: int, rank: int) -> LogitsProcessor:
    """A head sharded over ``size`` ranks whose vocab all-gather is faked as
    the identity on a full-vocab head (each rank holds the whole head here);
    the shard's row gathers are recorded by the test."""
    proc = LogitsProcessor(
        config=SimpleNamespace(
            model_type="test", vocab_size=8, final_logit_softcapping=None
        ),
        tp_rank=rank,
        tp_size=size,
        tp_group=tuple(range(size)),
        dp_lm_head_tp=False,
    )
    monkeypatch.setattr(proc, "_init_all_gather_state", lambda lm_head: None)
    monkeypatch.setattr(torch.cuda, "is_current_stream_capturing", lambda: False)

    def no_gather(hidden_states, lm_head, md, embedding_bias=None, plan=None, **kw):
        return hidden_states.float() @ lm_head.weight.float().T

    monkeypatch.setattr(proc, "_get_logits", no_gather)
    return proc


def test_a_sharded_forward_gathers_the_planned_rows_then_the_sampled_rows(
    monkeypatch,
):
    """Under a query shard the processor receives the shard's rows. The head
    is vocab-sharded over the same group, so it first all-gathers the
    planned prompt rows' activations with the per-rank counts and scores the
    whole plan on every rank (one chunk schedule for the group), then gathers
    the sampled rows (never ``hidden[gather_ids]``) and runs the LM head on
    the batch's ``[bs, hidden]`` rows: prompt rows are scored before the
    sampled rows leave the shard. A FULL capture stays the shard. Every rank,
    one without planned rows included, joins both gathers."""
    from tokenspeed.runtime.execution.forward_batch_info import CaptureHiddenMode
    from tokenspeed.runtime.execution.query_shard import QueryShardPlan

    lengths = [4, 1, 5]  # rows 0..9; sampled rows 3, 4, 9; shards [3, 3, 2, 2]
    total = sum(lengths)
    gather_ids = torch.cumsum(torch.tensor(lengths), 0) - 1
    torch.manual_seed(0)
    hidden = torch.randn(total, 2)
    lm_head = SimpleNamespace(weight=torch.randn(8, 2))
    # The plan: rows 1..3 of request 0 and rows 5..7 of request 2; rank 3
    # (rows 8, 9) holds none of them.
    plan_rows = torch.tensor([1, 2, 3, 5, 6, 7])
    plan_targets = torch.tensor([3, 1, 7, 2, 2, 5])
    reference = torch.log_softmax((hidden @ lm_head.weight.T).float(), -1)[
        plan_rows, plan_targets
    ]
    gathers: list[tuple[int, str, list[int]]] = []

    def fake_gather(kind):
        def gather(tensor, group, counts):
            assert group == (0, 1, 2, 3)
            gathers.append((len(gathers), kind, list(counts)))
            # Stand in for the collective: the rows every rank would
            # contribute, in rank order.
            rows = plan_rows if kind == "planned" else gather_ids
            return hidden[rows]

        return gather

    # The planned-row gather (processor) and the sampled-row gather
    # (comm_manager.gather_sampled_rows): both byte-preserving row gathers.
    monkeypatch.setattr(
        logits_processor_module, "token_all_gather_rows", fake_gather("planned")
    )
    monkeypatch.setattr(
        comm_manager_module, "token_all_gather_rows", fake_gather("sampled")
    )

    for rank in range(4):
        plan = QueryShardPlan.from_forward(
            total_tokens=total, input_lengths=lengths, size=4, rank=rank
        )
        proc = _sharded_processor(monkeypatch, 4, rank)
        rows_per_rank = plan.rows_per_rank(plan_rows)
        assert rows_per_rank == (2, 2, 2, 0)
        local = plan.local_rows_run(rows_per_rank)
        md = LogitsMetadata(
            forward_mode=ForwardMode.EXTEND,
            capture_hidden_mode=CaptureHiddenMode.FULL,
            gather_ids=gather_ids,
            query_shard=plan,
            input_logprob_rows=InputLogprobRows(
                rows=plan_rows[local] - plan.local_start,
                targets=plan_targets,
                slots=torch.zeros(plan_rows.shape[0], dtype=torch.int64),
                num_input_rows=plan.local_rows,
                chunk_tokens=2,
                rows_per_rank=rows_per_rank,
            ),
        )
        shard = hidden[plan.local_slice]
        out = proc(
            input_ids=None, hidden_states=shard, lm_head=lm_head, logits_metadata=md
        )
        assert out.hidden_states is shard  # FULL capture: the shard's own rows
        # The planned rows leave before the sampled rows; rank 3 contributes
        # no planned row and still joins.
        assert [g[1] for g in gathers[2 * rank :]] == ["planned", "sampled"]
        assert gathers[2 * rank][2] == [2, 2, 2, 0]
        assert (
            gathers[2 * rank + 1][2]
            == list(plan.sampled_rows_per_rank)
            == [
                0,
                2,
                0,
                1,
            ]
        )
        # Every rank ends with the whole plan's logprobs and the batch's logits.
        torch.testing.assert_close(out.input_token_logprobs, reference, rtol=0, atol=0)
        torch.testing.assert_close(
            out.next_token_logits,
            (hidden[gather_ids] @ lm_head.weight.T).float(),
            rtol=0,
            atol=0,
        )


@pytest.mark.parametrize("taps", [0, 1, 3])
def test_a_last_capture_under_a_shard_is_the_gathered_sampled_rows(monkeypatch, taps):
    """A LAST hidden capture under a query shard stores the batch's gathered
    ``[bs, hidden]`` rows, whole on every rank -- the aux taps' (Eagle3) when
    the model has them, concatenated, else the final hidden's -- and each
    tap costs one sampled-row gather only on a LAST capture: a FULL or NULL
    capture never selects the taps, so the forward spends no collective on
    them. ``taps=0`` is a model without taps."""
    from tokenspeed.runtime.execution.forward_batch_info import CaptureHiddenMode
    from tokenspeed.runtime.execution.query_shard import QueryShardPlan

    lengths = [4, 1, 5]  # rows 0..9; sampled rows 3, 4, 9; shards [3, 3, 2, 2]
    total = sum(lengths)
    gather_ids = torch.cumsum(torch.tensor(lengths), 0) - 1
    torch.manual_seed(1)
    hidden = torch.randn(total, 2)
    aux = [torch.randn(total, 2) + 10 * (i + 1) for i in range(taps)]
    lm_head = SimpleNamespace(weight=torch.randn(8, 2))
    gathered: list[torch.Tensor] = []
    current: dict[str, QueryShardPlan] = {}

    def fake_gather(tensor, group, counts):
        assert group == (0, 1, 2, 3) and list(counts) == [0, 2, 0, 1]
        # The final rows are gathered first, then each tap in order; this
        # rank contributes its sampled rows of that source, and the collective
        # hands back every rank's in rank order: the batch's sampled rows.
        source = (hidden, *aux)[len(gathered)]
        plan = current["plan"]
        local = source[plan.local_slice][plan.local_sampled_ids(gather_ids)]
        assert torch.equal(tensor, local)
        gathered.append(tensor)
        return source[gather_ids]

    monkeypatch.setattr(comm_manager_module, "token_all_gather_rows", fake_gather)

    modes = (CaptureHiddenMode.LAST, CaptureHiddenMode.FULL, CaptureHiddenMode.NULL)
    for rank in range(4):
        plan = QueryShardPlan.from_forward(
            total_tokens=total, input_lengths=lengths, size=4, rank=rank
        )
        current["plan"] = plan
        shard = hidden[plan.local_slice]
        aux_shard = [a[plan.local_slice] for a in aux] or None
        for mode in modes:
            gathered.clear()
            proc = _sharded_processor(monkeypatch, 4, rank)
            out = proc(
                input_ids=None,
                hidden_states=shard,
                lm_head=lm_head,
                logits_metadata=LogitsMetadata(
                    forward_mode=ForwardMode.EXTEND,
                    capture_hidden_mode=mode,
                    gather_ids=gather_ids,
                    query_shard=plan,
                ),
                aux_hidden_states=aux_shard,
            )
            torch.testing.assert_close(
                out.next_token_logits,
                (hidden[gather_ids] @ lm_head.weight.T).float(),
                rtol=0,
                atol=0,
            )
            if mode is CaptureHiddenMode.LAST:
                assert len(gathered) == 1 + taps
                expected = (
                    torch.cat([a[gather_ids] for a in aux], dim=-1)
                    if taps
                    else hidden[gather_ids]
                )
                torch.testing.assert_close(out.hidden_states, expected, rtol=0, atol=0)
                assert out.hidden_states.shape == (3, 2 * max(taps, 1))
            else:
                assert len(gathered) == 1  # the final hidden rows only
                if mode is CaptureHiddenMode.FULL:
                    expected = torch.cat(aux_shard, dim=-1) if taps else shard
                    torch.testing.assert_close(
                        out.hidden_states, expected, rtol=0, atol=0
                    )
                else:
                    assert out.hidden_states is None


def test_a_shard_refuses_pre_selected_rows_and_a_mismatched_head(monkeypatch):
    """A model that selected its rows before the processor has no prompt
    activations left -- the refusal fires whether or not the forward is
    sharded. And the shard's group must be the LM head's vocab-shard group."""
    from tokenspeed.runtime.execution.query_shard import QueryShardPlan

    plan = QueryShardPlan.from_forward(
        total_tokens=4, input_lengths=[4], size=2, rank=0
    )
    lm_head = SimpleNamespace(weight=torch.randn(8, 2))
    rows = _input_logprob_rows(
        [0, 1],
        [1, 2],
        num_input_rows=2,
        chunk_tokens=8,
        device="cpu",
        rows_per_rank=(2, 1),
    )
    proc = _sharded_processor(monkeypatch, 2, 0)
    with pytest.raises(ValueError, match="narrowed"):
        proc(
            input_ids=None,
            hidden_states=torch.randn(1, 2),
            lm_head=lm_head,
            logits_metadata=LogitsMetadata(
                forward_mode=ForwardMode.EXTEND,
                gather_ids=torch.tensor([3]),
                logits_rows_selected=True,
                query_shard=plan,
                input_logprob_rows=rows,
            ),
        )
    # Rows staged for a shard reaching an unsharded forward, and vice versa.
    with pytest.raises(ValueError, match="staged for"):
        proc(
            input_ids=None,
            hidden_states=torch.randn(2, 2),
            lm_head=lm_head,
            logits_metadata=LogitsMetadata(
                forward_mode=ForwardMode.EXTEND,
                query_shard=None,
                gather_ids=torch.tensor([1]),
                input_logprob_rows=rows,
            ),
        )
    wide = LogitsProcessor(
        config=SimpleNamespace(model_type="test", vocab_size=8),
        tp_rank=0,
        tp_size=4,
        tp_group=(0, 1, 2, 3),
        dp_lm_head_tp=False,
    )
    with pytest.raises(ValueError, match="vocab-shard group"):
        wide(
            input_ids=None,
            hidden_states=torch.randn(2, 2),
            lm_head=lm_head,
            logits_metadata=LogitsMetadata(
                forward_mode=ForwardMode.EXTEND,
                gather_ids=torch.tensor([3]),
                query_shard=plan,
            ),
        )
    replicated = LogitsProcessor(
        config=SimpleNamespace(model_type="test", vocab_size=8),
        skip_all_gather=True,
        tp_rank=0,
        tp_size=2,
        tp_group=(0, 1),
        dp_lm_head_tp=False,
    )
    with pytest.raises(ValueError, match="replicated LM head"):
        replicated(
            input_ids=None,
            hidden_states=torch.randn(2, 2),
            lm_head=lm_head,
            logits_metadata=LogitsMetadata(
                forward_mode=ForwardMode.EXTEND,
                gather_ids=torch.tensor([3]),
                query_shard=plan,
            ),
        )
