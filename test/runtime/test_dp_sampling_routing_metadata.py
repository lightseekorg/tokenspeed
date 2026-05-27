from __future__ import annotations

import re
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from tokenspeed.runtime.execution.cuda_graph_wrapper import (
    resolve_dp_sampling_min_bs,
    should_use_dp_sampling_for_bucket,
)
from tokenspeed.runtime.execution.forward_batch_info import ForwardMode
from tokenspeed.runtime.layers.logits_processor import LogitsMetadata, LogitsProcessor


def test_dp_sampling_bucket_threshold():
    assert not should_use_dp_sampling_for_bucket(
        dp_sampling_enabled=True,
        forward_mode=ForwardMode.DECODE,
        effective_bs=15,
        min_bs=16,
    )
    assert should_use_dp_sampling_for_bucket(
        dp_sampling_enabled=True,
        forward_mode=ForwardMode.DECODE,
        effective_bs=16,
        min_bs=16,
    )


def test_dp_sampling_default_threshold_covers_two_local_requests():
    min_bs = resolve_dp_sampling_min_bs(tp_size=4, configured_min_bs=None)
    assert min_bs == 8

    assert not should_use_dp_sampling_for_bucket(
        dp_sampling_enabled=True,
        forward_mode=ForwardMode.DECODE,
        effective_bs=7,
        min_bs=min_bs,
    )
    assert should_use_dp_sampling_for_bucket(
        dp_sampling_enabled=True,
        forward_mode=ForwardMode.DECODE,
        effective_bs=8,
        min_bs=min_bs,
    )
    assert not should_use_dp_sampling_for_bucket(
        dp_sampling_enabled=True,
        forward_mode=ForwardMode.EXTEND,
        effective_bs=32,
        min_bs=min_bs,
    )
    assert not should_use_dp_sampling_for_bucket(
        dp_sampling_enabled=False,
        forward_mode=ForwardMode.DECODE,
        effective_bs=32,
        min_bs=min_bs,
    )


def test_dp_sampling_lm_head_capability_check():
    assert LogitsProcessor.supports_dp_sampling_lm_head(
        SimpleNamespace(weight=torch.empty(1, 1))
    )
    assert not LogitsProcessor.supports_dp_sampling_lm_head(
        SimpleNamespace(linear_method=object())
    )


def test_dp_sampling_lm_head_vocab_size_uses_padded_local_weight():
    lm_head = SimpleNamespace(weight=torch.empty(16032, 16))

    assert LogitsProcessor.dp_sampling_lm_head_vocab_size(lm_head, tp_size=2) == 32064


def test_configure_dp_sampling_validates_lm_head_before_runtime():
    processor = LogitsProcessor(
        SimpleNamespace(vocab_size=7, model_type="unit_test"),
        tp_rank=0,
        tp_size=4,
        tp_group=(0, 1, 2, 3),
    )

    with pytest.raises(RuntimeError, match="standard LM head"):
        processor.configure_dp_sampling(
            lm_head=SimpleNamespace(linear_method=object()),
            dp_num_tokens_per_req=6,
            dp_comm=None,
        )
    assert not processor.dp_sampling_enabled

    processor.configure_dp_sampling(
        lm_head=SimpleNamespace(weight=torch.empty(7, 3)),
        dp_num_tokens_per_req=6,
        dp_comm=None,
    )
    assert processor.dp_sampling_enabled
    assert processor.dp_num_tokens_per_req == 6


def test_skip_all_gather_dp_sampling_slices_hidden_states_before_lm_head():
    processor = LogitsProcessor(
        SimpleNamespace(vocab_size=7, model_type="unit_test"),
        skip_all_gather=True,
        tp_rank=1,
        tp_size=4,
        tp_group=(0, 1, 2, 3),
        dp_sampling_enabled=True,
        dp_num_tokens_per_req=6,
    )
    hidden_states = torch.arange(5 * 6 * 3, dtype=torch.float32).view(5 * 6, 3)
    lm_head = SimpleNamespace(weight=torch.ones(7, 3))

    logits = processor._get_logits(
        hidden_states,
        lm_head,
        LogitsMetadata(forward_mode=ForwardMode.DECODE, dp_sampling=True),
    )

    assert logits.shape == (12, 7)
    expected_rows = hidden_states[12:24].sum(dim=1)
    assert torch.equal(logits[:, 0], expected_rows)


def test_skip_all_gather_logits_processors_carry_tp_metadata():
    root = Path(__file__).resolve().parents[2]
    model_files = [
        root / "python/tokenspeed/runtime/models/base/causal_lm.py",
        root / "python/tokenspeed/runtime/models/minimax_m2.py",
        root / "python/tokenspeed/runtime/models/qwen3_5_nextn.py",
        root / "python/tokenspeed/runtime/models/deepseek_nextn.py",
    ]

    for path in model_files:
        source = path.read_text()
        assert "skip_all_gather=True" in source, path
        for kwarg in ("tp_rank=", "tp_size=", "tp_group="):
            pattern = (
                r"LogitsProcessor\([\s\S]{0,240}"
                r"skip_all_gather=True[\s\S]{0,240}" + re.escape(kwarg)
            )
            assert re.search(pattern, source), f"{path}: missing {kwarg}"


def test_pdl_verify_keeps_tp_broadcast_with_fused_topk_topp():
    root = Path(__file__).resolve().parents[2]
    for relpath in (
        "python/tokenspeed/runtime/sampling/backends/flashinfer.py",
        "python/tokenspeed/runtime/sampling/backends/flashinfer_full.py",
    ):
        source = (root / relpath).read_text()
        assert re.search(
            r"(?:if|elif) pdl_enabled\(\):\s+self\.maybe_broadcast"
            r"\(predict, accept_index, accept_length\)",
            source,
        ), relpath


def test_dp_sampling_preconditions_are_capability_based():
    root = Path(__file__).resolve().parents[2]
    executor = (
        root / "python/tokenspeed/runtime/execution/model_executor.py"
    ).read_text()
    flashinfer = (
        root / "python/tokenspeed/runtime/sampling/backends/flashinfer.py"
    ).read_text()
    flashinfer_full = (
        root / "python/tokenspeed/runtime/sampling/backends/flashinfer_full.py"
    ).read_text()

    assert "backend_supports_dp" in executor
    assert "lm_head_supports_dp" in executor
    assert "_SUPPORTS_DP_VERIFY = True" in flashinfer
    assert "_SUPPORTS_DP_VERIFY = False" in flashinfer_full


def test_dp_verify_handles_grammar_masks_and_logprob_logits():
    root = Path(__file__).resolve().parents[2]
    source = (
        root / "python/tokenspeed/runtime/sampling/backends/flashinfer.py"
    ).read_text()
    comm = (
        root / "python/tokenspeed/runtime/distributed/dp_sampling_comm.py"
    ).read_text()

    assert "dp_sampling + grammar bitmask is not supported" not in source
    assert "_slice_dp_vocab_mask" in source
    assert "torch.log_softmax(logits_output.next_token_logits, dim=-1)" in source
    assert "gather_verify_logprobs" in source
    assert "gather_verify_logprobs" in comm


def test_one_sided_dp_sampling_accepts_process_group_subclasses():
    root = Path(__file__).resolve().parents[2]
    source = (
        root
        / "tokenspeed-kernel/python/tokenspeed_kernel/ops/communication/dp_sampling.py"
    ).read_text()

    assert "isinstance(" in source
    assert "group, dist.ProcessGroup" in source
