from __future__ import annotations

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


def test_dp_sampling_min_bs_ignores_env_override(monkeypatch):
    monkeypatch.setenv("TOKENSPEED_DP_SAMPLING_MIN_BS", "16")

    assert resolve_dp_sampling_min_bs(tp_size=4, configured_min_bs=None) == 8
    assert resolve_dp_sampling_min_bs(tp_size=4, configured_min_bs=12) == 12


def test_configure_dp_sampling_sets_state():
    processor = LogitsProcessor(
        SimpleNamespace(vocab_size=7, model_type="unit_test"),
        tp_rank=0,
        tp_size=4,
        tp_group=(0, 1, 2, 3),
    )

    processor.configure_dp_sampling(
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
