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


def test_argmax_full_vocab_skips_kernel(monkeypatch):
    """Full-vocab logits fall back to argmax even when a state is present."""
    proc = LogitsProcessor(
        config=SimpleNamespace(model_type="test", vocab_size=8),
        tp_rank=0,
        tp_size=2,
        tp_group=(0, 1),
    )
    proc._dist_argmax_state = object()

    monkeypatch.setattr(
        logits_processor_module,
        "distributed_argmax",
        lambda *a, **k: pytest.fail("kernel must not run on full-vocab logits"),
    )

    full = torch.randn(4, 8, dtype=torch.float32)  # width == vocab_size => not sharded
    ids = proc._argmax(full)
    assert torch.equal(ids, full.argmax(dim=-1))


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
