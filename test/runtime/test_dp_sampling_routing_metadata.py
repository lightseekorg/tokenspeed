from __future__ import annotations

import re
from pathlib import Path
from types import SimpleNamespace

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
