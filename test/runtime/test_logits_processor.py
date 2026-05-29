"""Regression tests for logits processing helpers."""

from __future__ import annotations

import os
import sys
from types import SimpleNamespace

# CI Registration (parsed via AST, runtime no-op)
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from ci_system.ci_register import register_cuda_ci

register_cuda_ci(est_time=90, suite="runtime-1gpu")

import pytest
import torch

from tokenspeed.runtime.execution.forward_batch_info import ForwardMode
from tokenspeed.runtime.layers.logits_processor import (
    LogitsMetadata,
    LogitsProcessor,
    fused_softcap,
)


@pytest.mark.parametrize("forward_mode", [ForwardMode.EXTEND, ForwardMode.MIXED])
def test_logits_processor_extend_without_gather_ids_uses_request_last_tokens(
    forward_mode,
):
    processor = LogitsProcessor(config=SimpleNamespace(model_type="test", vocab_size=3))
    hidden_states = torch.tensor(
        [
            [1.0, 0.0, 0.0],
            [2.0, 0.0, 0.0],
            [0.0, 3.0, 0.0],
            [0.0, 4.0, 0.0],
            [0.0, 5.0, 0.0],
        ],
        dtype=torch.float32,
    )
    lm_head = SimpleNamespace(weight=torch.eye(3, dtype=torch.float32))
    metadata = LogitsMetadata(
        forward_mode=forward_mode,
        gather_ids=None,
        extend_seq_lens=torch.tensor([2, 3], dtype=torch.int32),
    )

    out = processor(
        input_ids=None,
        hidden_states=hidden_states,
        lm_head=lm_head,
        logits_metadata=metadata,
    )

    torch.testing.assert_close(out.next_token_logits, hidden_states[[1, 4]])


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
