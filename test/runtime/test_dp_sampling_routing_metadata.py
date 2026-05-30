from __future__ import annotations

from types import SimpleNamespace

import torch

from tokenspeed.runtime.execution.context import ForwardContext
from tokenspeed.runtime.execution.cuda_graph_wrapper import (
    CudaGraphWrapper,
    dp_sampling_layout_bucket_bs,
    resolve_dp_sampling_min_bs,
    should_use_dp_sampling_for_bucket,
)
from tokenspeed.runtime.execution.forward_batch_info import ForwardMode
from tokenspeed.runtime.layers.logits_processor import LogitsMetadata, LogitsProcessor
from tokenspeed.runtime.sampling.logits_layout import LogitsLayoutPlan


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


def test_dp_sampling_layout_bucket_rounds_to_tp_size():
    assert dp_sampling_layout_bucket_bs(real_bs=32, tp_size=8) == 32
    assert dp_sampling_layout_bucket_bs(real_bs=33, tp_size=8) == 40
    assert dp_sampling_layout_bucket_bs(real_bs=24, tp_size=16) == 32
    assert dp_sampling_layout_bucket_bs(real_bs=79, tp_size=8) == 80


def test_dp_sampling_route_uses_graph_bucket_threshold():
    runner = CudaGraphWrapper.__new__(CudaGraphWrapper)
    runner.disable = False
    runner.dp_size = 1
    runner.disable_padding = False
    runner.max_bs = 32
    runner.capture_bs = [24, 32]
    runner.dp_sampling_enabled = True
    runner.dp_sampling_min_bs = 32
    runner.logits_tp_size = 8
    ctx = ForwardContext(
        attn_backend=None,
        token_to_kv_pool=None,
        bs=30,
        num_extends=0,
        input_num_tokens=30,
        forward_mode=ForwardMode.DECODE,
    )

    dp_sampling, use_graph, bucket_bs = runner.dp_sampling_route(30, ctx)

    assert dp_sampling
    assert use_graph
    assert bucket_bs == 32


def test_dp_sampling_route_pads_graph_layout_bucket_to_tp_size():
    runner = CudaGraphWrapper.__new__(CudaGraphWrapper)
    runner.disable = False
    runner.dp_size = 1
    runner.disable_padding = False
    runner.max_bs = 80
    runner.capture_bs = [72, 79, 80]
    runner.dp_sampling_enabled = True
    runner.dp_sampling_min_bs = 32
    runner.logits_tp_size = 8
    ctx = ForwardContext(
        attn_backend=None,
        token_to_kv_pool=None,
        bs=79,
        num_extends=0,
        input_num_tokens=79,
        forward_mode=ForwardMode.DECODE,
    )

    dp_sampling, use_graph, bucket_bs = runner.dp_sampling_route(79, ctx)

    assert dp_sampling
    assert use_graph
    assert bucket_bs == 80


def test_dp_sampling_route_pads_capture_bucket_above_threshold_to_tp_size():
    runner = CudaGraphWrapper.__new__(CudaGraphWrapper)
    runner.disable = False
    runner.dp_size = 1
    runner.disable_padding = False
    runner.max_bs = 32
    runner.capture_bs = [24, 32]
    runner.dp_sampling_enabled = True
    runner.dp_sampling_min_bs = 16
    runner.logits_tp_size = 16
    ctx = ForwardContext(
        attn_backend=None,
        token_to_kv_pool=None,
        bs=24,
        num_extends=0,
        input_num_tokens=24,
        forward_mode=ForwardMode.DECODE,
    )

    dp_sampling, use_graph, bucket_bs = runner.dp_sampling_route(24, ctx)

    assert dp_sampling
    assert use_graph
    assert bucket_bs == 32


def test_dp_sampling_route_keeps_graph_bucket_below_threshold_non_dp():
    runner = CudaGraphWrapper.__new__(CudaGraphWrapper)
    runner.disable = False
    runner.dp_size = 1
    runner.disable_padding = False
    runner.max_bs = 32
    runner.capture_bs = [24, 32]
    runner.dp_sampling_enabled = True
    runner.dp_sampling_min_bs = 32
    runner.logits_tp_size = 8
    ctx = ForwardContext(
        attn_backend=None,
        token_to_kv_pool=None,
        bs=23,
        num_extends=0,
        input_num_tokens=23,
        forward_mode=ForwardMode.DECODE,
    )

    dp_sampling, use_graph, bucket_bs = runner.dp_sampling_route(23, ctx)

    assert not dp_sampling
    assert use_graph
    assert bucket_bs == 24


def test_dp_sampling_eager_route_still_returns_tp_divisible_layout_bucket():
    runner = CudaGraphWrapper.__new__(CudaGraphWrapper)
    runner.disable = True
    runner.dp_sampling_enabled = True
    runner.dp_sampling_min_bs = 16
    runner.logits_tp_size = 4
    ctx = ForwardContext(
        attn_backend=None,
        token_to_kv_pool=None,
        bs=17,
        num_extends=0,
        input_num_tokens=17,
        forward_mode=ForwardMode.DECODE,
    )

    dp_sampling, use_graph, bucket_bs = runner.dp_sampling_route(17, ctx)

    assert dp_sampling
    assert not use_graph
    assert bucket_bs == 20


def test_configure_dp_sampling_sets_state():
    processor = LogitsProcessor(
        SimpleNamespace(vocab_size=7, model_type="unit_test"),
        tp_rank=0,
        tp_size=4,
        tp_group=(0, 1, 2, 3),
    )

    processor.configure_dp_sampling(
        dp_num_tokens_per_req=6,
        max_bucket_bs=8,
        vocab_size=8,
        device="cpu",
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
        LogitsMetadata(
            forward_mode=ForwardMode.DECODE,
            dp_sampling=True,
            logits_layout_plan=LogitsLayoutPlan.dp_all_to_all(
                real_bs=5,
                bucket_bs=8,
                tp_size=4,
                num_tokens_per_req=6,
            ),
        ),
    )

    assert logits.shape == (12, 7)
    expected_rows = hidden_states[12:24].sum(dim=1)
    assert torch.equal(logits[:, 0], expected_rows)
