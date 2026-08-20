"""Forced-rejection control for acceptance-independent speculative timing."""

from __future__ import annotations

import os
import sys
from types import SimpleNamespace
from unittest import mock

import torch

_TEST_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _TEST_DIR)
sys.path.insert(0, os.path.dirname(_TEST_DIR))

from ci_system.ci_register import register_cuda_ci

register_cuda_ci(est_time=5, suite="runtime-1gpu")

from test.runtime.conftest import requires_cuda

import tokenspeed.runtime.execution.input_buffer as input_buffer_module
import tokenspeed.runtime.execution.model_executor as model_executor_module
from tokenspeed.runtime.execution.input_buffer import InputBuffers
from tokenspeed.runtime.execution.model_executor import ModelExecutor


def _executor(force_mask: torch.Tensor) -> ModelExecutor:
    executor = ModelExecutor.__new__(ModelExecutor)
    executor.input_buffers = SimpleNamespace(force_single_token_verify_buf=force_mask)
    return executor


def test_local_decode_keeps_acceptance_unchanged_by_default() -> None:
    accept_lengths = torch.tensor([4, 2], dtype=torch.int32)
    executor = _executor(torch.tensor([True, True]))

    with mock.patch.object(model_executor_module, "FORCE_SINGLE_TOKEN_VERIFY", False):
        actual = executor._apply_force_single_token_verify(
            accept_lengths,
            row_offset=0,
            row_count=2,
            decode_input_ids=None,
        )

    assert actual is accept_lengths


def test_remote_recovery_mask_forces_only_marked_rows() -> None:
    executor = _executor(torch.tensor([False, True, False]))

    with mock.patch.object(model_executor_module, "FORCE_SINGLE_TOKEN_VERIFY", False):
        actual = executor._apply_force_single_token_verify(
            torch.tensor([4, 5], dtype=torch.int32),
            row_offset=1,
            row_count=2,
            decode_input_ids=[7, -1],
        )

    assert actual.tolist() == [1, 5]


def test_global_control_forces_local_decode_rows() -> None:
    executor = _executor(torch.tensor([True, True]))

    with mock.patch.object(model_executor_module, "FORCE_SINGLE_TOKEN_VERIFY", True):
        actual = executor._apply_force_single_token_verify(
            torch.tensor([8, 3], dtype=torch.int32),
            row_offset=0,
            row_count=2,
            decode_input_ids=None,
        )

    assert actual.tolist() == [1, 1]


def test_input_buffer_initializes_global_force_mask() -> None:
    with mock.patch.object(input_buffer_module, "FORCE_SINGLE_TOKEN_VERIFY", True):
        buffers = InputBuffers(
            max_bs=3,
            max_num_tokens=8,
            page_size=4,
            dummy_kv_slot=0,
            state_write_padding_pool_index=0,
            device="cpu",
        )

    assert buffers.force_single_token_verify_buf.tolist() == [True, True, True]


@requires_cuda
def test_global_force_is_cuda_graph_capturable() -> None:
    executor = _executor(torch.ones(2, dtype=torch.bool, device="cuda"))
    accept_lengths = torch.tensor([8, 3], dtype=torch.int32, device="cuda")

    with mock.patch.object(model_executor_module, "FORCE_SINGLE_TOKEN_VERIFY", True):
        # Warm up the elementwise selection before capture.
        executor._apply_force_single_token_verify(
            accept_lengths,
            row_offset=0,
            row_count=2,
            decode_input_ids=None,
        )
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            captured = executor._apply_force_single_token_verify(
                accept_lengths,
                row_offset=0,
                row_count=2,
                decode_input_ids=None,
            )

    accept_lengths.fill_(6)
    graph.replay()
    torch.cuda.synchronize()
    assert captured.tolist() == [1, 1]
