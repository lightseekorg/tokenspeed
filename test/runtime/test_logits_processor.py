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
    metadata = LogitsMetadata(forward_mode=ForwardMode.DECODE)
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
    metadata = LogitsMetadata(forward_mode=ForwardMode.DECODE)
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
    md = LogitsMetadata(forward_mode=ForwardMode.DECODE)
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
    md = LogitsMetadata(forward_mode=ForwardMode.DECODE)
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
    md = LogitsMetadata(forward_mode=ForwardMode.DECODE)
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
    md = LogitsMetadata(forward_mode=ForwardMode.DECODE)
    out = proc._get_logits(hidden, lm_head, md, require_full_vocab=False)
    assert called.get("ag")  # gathered (softcap on full vocab), not early-returned
    assert out.shape == (4, 8)


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))


def _logprob_device() -> str:
    return "cuda" if torch.cuda.is_available() else "cpu"


def _input_logprob_rows(rows, targets, *, num_input_rows, chunk_tokens, device):
    return InputLogprobRows(
        rows=torch.tensor(rows, dtype=torch.int64, device=device),
        targets=torch.tensor(targets, dtype=torch.int64, device=device),
        slots=torch.zeros(len(rows), dtype=torch.int64, device=device),
        num_input_rows=num_input_rows,
        chunk_tokens=chunk_tokens,
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
    hidden = torch.randn(5, 4, device=device)
    weight = torch.randn(vocab, 4, device=device)
    lm_head = SimpleNamespace(weight=weight)
    rows, targets = [1, 2, 3], [5, 2, 3]
    processor = LogitsProcessor(
        config=SimpleNamespace(model_type="test", vocab_size=vocab),
        dp_lm_head_tp=False,
    )
    metadata = LogitsMetadata(
        forward_mode=ForwardMode.EXTEND,
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
        config=SimpleNamespace(model_type="test", vocab_size=vocab)
    )
    seen = []
    original = processor._get_logits

    def spy(hidden_states, *args, **kwargs):
        seen.append(hidden_states.shape[0])
        return original(hidden_states, *args, **kwargs)

    processor._get_logits = spy
    metadata = LogitsMetadata(
        forward_mode=ForwardMode.MIXED,
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
    processor = LogitsProcessor(config=SimpleNamespace(model_type="test", vocab_size=4))
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
