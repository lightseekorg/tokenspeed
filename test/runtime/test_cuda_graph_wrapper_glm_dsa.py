"""Focused regression tests for CUDA graph replay metadata padding."""

from __future__ import annotations

import os
import sys
from types import MethodType, SimpleNamespace

import pytest
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from ci_system.ci_register import register_cuda_ci  # noqa: E402

register_cuda_ci(est_time=5, suite="runtime-1gpu")

from tokenspeed.runtime.execution.cuda_graph_wrapper import CudaGraphWrapper  # noqa: E402
from tokenspeed.runtime.execution.forward_batch_info import ForwardMode  # noqa: E402
from tokenspeed.runtime.execution.model_executor import ModelExecutor  # noqa: E402
from tokenspeed.runtime.layers.logits_processor import LogitsProcessorOutput  # noqa: E402


class _FakeGraph:
    def __init__(self, *, fail: bool = False) -> None:
        self.fail = fail
        self.called = False

    def replay(self) -> None:
        self.called = True
        if self.fail:
            raise RuntimeError("graph replay failed")


class _FakeDeepEPAdapter:
    def replay(self) -> None:
        pass


def _make_wrapper(graph: _FakeGraph) -> CudaGraphWrapper:
    wrapper = object.__new__(CudaGraphWrapper)
    wrapper.max_tokens_per_req = 6
    wrapper.config = SimpleNamespace(enable_mamba=False, max_req_pool_size=99)
    wrapper.input_buffers = SimpleNamespace(
        seq_lens_buf=torch.tensor([11, 12, 13, 1], dtype=torch.int32),
        req_pool_indices_buf=torch.tensor([3, 4, 5, 0], dtype=torch.int64),
    )
    wrapper.graphs = {4: graph}
    wrapper.output_buffers = {
        4: (
            torch.arange(24, dtype=torch.int32),
            torch.tensor([1, 1, 1, 1], dtype=torch.int32),
            None,
            None,
        )
    }
    wrapper.deepep_adapter = _FakeDeepEPAdapter()
    wrapper.drafter = object()
    wrapper.attn_backend = object()
    wrapper._can_use_graph = MethodType(lambda self, bs, ctx: True, wrapper)
    wrapper._padded_bs = MethodType(lambda self, bs, ctx: 4, wrapper)
    return wrapper


def test_call_restores_ctx_bs_after_padding_and_replay() -> None:
    graph = _FakeGraph()
    wrapper = _make_wrapper(graph)
    seen = {}

    def init_replay_metadata(self, padded_bs, actual_bs, req_pool_indices, *args, **kw):
        seen["padded_bs"] = padded_bs
        seen["actual_bs"] = actual_bs
        seen["ctx_bs_inside"] = ctx.bs
        seen["req_pool_indices"] = req_pool_indices.clone()

    wrapper._init_replay_metadata = MethodType(init_replay_metadata, wrapper)
    ctx = SimpleNamespace(bs=3, forward_mode=ForwardMode.DECODE)

    result = CudaGraphWrapper.__call__(
        wrapper,
        bs=3,
        ctx=ctx,
        sampling_info=None,
        req_to_page=torch.zeros((1, 1), dtype=torch.int32),
    )

    assert graph.called
    assert seen["padded_bs"] == 4
    assert seen["actual_bs"] == 3
    assert seen["ctx_bs_inside"] == 4
    assert seen["req_pool_indices"].tolist() == [3, 4, 5, 99]
    assert wrapper.input_buffers.req_pool_indices_buf.tolist() == [3, 4, 5, 99]
    assert wrapper.input_buffers.seq_lens_buf.tolist() == [11, 12, 13, 1]
    assert ctx.bs == 3
    assert result[0].numel() == 18
    assert result[1].numel() == 3


def test_call_restores_ctx_bs_when_replay_fails() -> None:
    wrapper = _make_wrapper(_FakeGraph(fail=True))
    wrapper._init_replay_metadata = MethodType(
        lambda self, *args, **kwargs: None,
        wrapper,
    )
    ctx = SimpleNamespace(bs=3, forward_mode=ForwardMode.DECODE)

    with pytest.raises(RuntimeError, match="graph replay failed"):
        CudaGraphWrapper.__call__(
            wrapper,
            bs=3,
            ctx=ctx,
            sampling_info=None,
            req_to_page=torch.zeros((1, 1), dtype=torch.int32),
        )

    assert ctx.bs == 3


def _make_executor_for_capture_drafter(capture_drafter: bool) -> ModelExecutor:
    executor = object.__new__(ModelExecutor)
    executor.config = SimpleNamespace(
        capture_drafter_in_cuda_graph=capture_drafter,
        spec_algo=None,
    )
    executor.grammar_runtime = None
    executor.drafter = object()
    executor.input_buffers = SimpleNamespace(
        req_pool_indices_buf=torch.tensor([3, 4], dtype=torch.int64),
    )
    executor.runtime_states = SimpleNamespace(vocab_size=100)
    executor._run_target_forward = MethodType(
        lambda self, bs, ctx, req_pool_indices: LogitsProcessorOutput(
            next_token_logits=None,
            hidden_states=torch.ones(bs, 4),
        ),
        executor,
    )
    executor._run_sampling = MethodType(
        lambda self, logits_output, sampling_info, ctx, candidates=None: (
            torch.tensor([7, 8], dtype=torch.int32),
            torch.ones(2, dtype=torch.int32),
        ),
        executor,
    )
    executor._drafter_calls = 0

    def run_drafter(self, *args, **kwargs):
        self._drafter_calls += 1

    executor._run_drafter_and_store = MethodType(run_drafter, executor)
    return executor


def test_forward_step_captures_drafter_when_enabled(monkeypatch) -> None:
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "is_current_stream_capturing", lambda: True)
    executor = _make_executor_for_capture_drafter(capture_drafter=True)
    ctx = SimpleNamespace(bs=2, input_num_tokens=2, forward_mode=ForwardMode.DECODE)

    _, _, _, draft_hidden = ModelExecutor._forward_step(
        executor,
        bs=2,
        ctx=ctx,
        sampling_info=None,
    )

    assert executor._drafter_calls == 1
    assert draft_hidden is None


def test_forward_step_keeps_legacy_drafter_eager_during_capture(monkeypatch) -> None:
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "is_current_stream_capturing", lambda: True)
    executor = _make_executor_for_capture_drafter(capture_drafter=False)
    ctx = SimpleNamespace(bs=2, input_num_tokens=2, forward_mode=ForwardMode.DECODE)

    _, _, _, draft_hidden = ModelExecutor._forward_step(
        executor,
        bs=2,
        ctx=ctx,
        sampling_info=None,
    )

    assert executor._drafter_calls == 0
    assert torch.equal(draft_hidden, torch.ones(2, 4))
