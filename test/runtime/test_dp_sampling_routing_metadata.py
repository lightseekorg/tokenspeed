from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

from tokenspeed.runtime.execution.context import ForwardContext
from tokenspeed.runtime.execution.cuda_graph_wrapper import CudaGraphWrapper
from tokenspeed.runtime.execution.forward_batch_info import ForwardMode
from tokenspeed.runtime.execution.graph_routing import select_cuda_graph_route
from tokenspeed.runtime.layers.logits_processor import LogitsMetadata, LogitsProcessor
from tokenspeed.runtime.sampling.dp_sampling_config import (
    DpSamplingRuntimeConfig,
    DpSamplingRuntimeLimits,
    DpSamplingSupport,
    DpSamplingTopology,
    resolve_dp_sampling_support,
    resolve_dp_sampling_runtime,
    validate_dp_sampling_lm_head_vocab,
)
from tokenspeed.runtime.sampling.logits_layout import (
    LogitsLayoutPlan,
    LogitsLayoutPlanner,
    resolve_dp_sampling_min_bs,
    should_use_dp_sampling_for_bucket,
)


def _graph_route(
    bs: int,
    ctx: ForwardContext,
    *,
    disable: bool = False,
    dp_size: int = 1,
    disable_padding: bool = False,
    max_bs: int,
    capture_bs: list[int],
    max_tokens_per_req: int = 1,
) -> tuple[bool, int]:
    return select_cuda_graph_route(
        bs=bs,
        forward_mode=ctx.forward_mode,
        disable=disable,
        dp_size=dp_size,
        all_decode_or_idle=ctx.all_decode_or_idle,
        global_num_tokens=ctx.global_num_tokens,
        max_tokens_per_req=max_tokens_per_req,
        disable_padding=disable_padding,
        capture_bs=capture_bs,
        max_bs=max_bs,
        available_graph_bs=set(capture_bs),
    )


def _dp_runtime_config(
    *,
    tp_rank: int = 0,
    tp_size: int = 4,
    tp_group: tuple[int, ...] = (0, 1, 2, 3),
    num_tokens_per_req: int = 6,
    min_bs: int = 8,
    max_bucket_bs: int = 8,
    vocab_size: int = 8,
    device: torch.device | str = "cpu",
    skip_all_gather: bool = False,
) -> DpSamplingRuntimeConfig:
    return DpSamplingRuntimeConfig(
        enabled=True,
        vocab_size=vocab_size,
        max_bucket_bs=max_bucket_bs,
        min_bs=min_bs,
        num_tokens_per_req=num_tokens_per_req,
        topology=DpSamplingTopology(
            tp_rank=tp_rank,
            tp_size=tp_size,
            tp_group=tp_group,
            skip_all_gather=skip_all_gather,
        ),
        device=device,
    )


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
    assert should_use_dp_sampling_for_bucket(
        dp_sampling_enabled=True,
        forward_mode=ForwardMode.TARGET_VERIFY,
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


def test_layout_planner_dp_bucket_rounds_to_tp_size():
    planner = LogitsLayoutPlanner(
        dp_sampling_enabled=True,
        dp_sampling_min_bs=1,
        tp_size=8,
        num_tokens_per_req=6,
    )

    plan = planner.build_plan(
        forward_mode=ForwardMode.DECODE,
        real_bs=33,
        effective_bs=33,
    )

    assert plan.is_dp_all_to_all
    assert plan.bucket_bs == 40


def test_layout_planner_uses_graph_bucket_threshold():
    planner = LogitsLayoutPlanner(
        dp_sampling_enabled=True,
        dp_sampling_min_bs=32,
        tp_size=8,
        num_tokens_per_req=6,
    )
    ctx = ForwardContext(
        attn_backend=None,
        token_to_kv_pool=None,
        bs=30,
        num_extends=0,
        input_num_tokens=30,
        forward_mode=ForwardMode.DECODE,
    )

    use_graph, bucket_bs = _graph_route(30, ctx, max_bs=32, capture_bs=[24, 32])
    plan = planner.build_plan(
        forward_mode=ctx.forward_mode,
        real_bs=30,
        effective_bs=bucket_bs,
    )

    assert use_graph
    assert plan.is_dp_all_to_all
    assert plan.real_bs == 30
    assert plan.effective_bs == 32
    assert plan.bucket_bs == 32


def test_cuda_graph_wrapper_uses_shared_route_for_padding():
    wrapper = CudaGraphWrapper.__new__(CudaGraphWrapper)
    wrapper.disable = False
    wrapper.dp_size = 1
    wrapper.disable_padding = False
    wrapper.max_bs = 32
    wrapper.capture_bs = [24, 32]
    wrapper.graphs = {24, 32}
    wrapper.max_tokens_per_req = 1
    ctx = ForwardContext(
        attn_backend=None,
        token_to_kv_pool=None,
        bs=30,
        num_extends=0,
        input_num_tokens=30,
        forward_mode=ForwardMode.DECODE,
    )

    assert wrapper.graph_route(30, ctx) == (True, 32)
    assert wrapper.can_run(30, ctx)
    assert wrapper.padded_bs(30, ctx) == 32


def test_cuda_graph_route_uses_global_batch_for_dp_idle_rank():
    ctx = ForwardContext(
        attn_backend=None,
        token_to_kv_pool=None,
        bs=0,
        num_extends=0,
        input_num_tokens=0,
        forward_mode=ForwardMode.DECODE,
        global_num_tokens=[0, 17],
        all_decode_or_idle=True,
    )

    assert _graph_route(
        0,
        ctx,
        dp_size=2,
        max_bs=32,
        capture_bs=[16, 32],
        max_tokens_per_req=1,
    ) == (True, 32)


def test_cuda_graph_route_respects_disable_padding_with_global_batch():
    ctx = ForwardContext(
        attn_backend=None,
        token_to_kv_pool=None,
        bs=0,
        num_extends=0,
        input_num_tokens=0,
        forward_mode=ForwardMode.DECODE,
        global_num_tokens=[0, 17],
        all_decode_or_idle=True,
    )

    assert _graph_route(
        0,
        ctx,
        dp_size=2,
        disable_padding=True,
        max_bs=32,
        capture_bs=[16, 32],
        max_tokens_per_req=1,
    ) == (False, 0)


def test_layout_planner_pads_graph_layout_bucket_to_tp_size():
    planner = LogitsLayoutPlanner(
        dp_sampling_enabled=True,
        dp_sampling_min_bs=32,
        tp_size=8,
        num_tokens_per_req=6,
    )
    ctx = ForwardContext(
        attn_backend=None,
        token_to_kv_pool=None,
        bs=79,
        num_extends=0,
        input_num_tokens=79,
        forward_mode=ForwardMode.DECODE,
    )

    use_graph, bucket_bs = _graph_route(79, ctx, max_bs=80, capture_bs=[72, 79, 80])
    plan = planner.build_plan(
        forward_mode=ctx.forward_mode,
        real_bs=79,
        effective_bs=bucket_bs,
    )

    assert use_graph
    assert plan.is_dp_all_to_all
    assert plan.real_bs == 79
    assert plan.effective_bs == 79
    assert plan.bucket_bs == 80


def test_layout_planner_pads_capture_bucket_above_threshold_to_tp_size():
    planner = LogitsLayoutPlanner(
        dp_sampling_enabled=True,
        dp_sampling_min_bs=16,
        tp_size=16,
        num_tokens_per_req=6,
    )
    ctx = ForwardContext(
        attn_backend=None,
        token_to_kv_pool=None,
        bs=24,
        num_extends=0,
        input_num_tokens=24,
        forward_mode=ForwardMode.DECODE,
    )

    use_graph, bucket_bs = _graph_route(24, ctx, max_bs=32, capture_bs=[24, 32])
    plan = planner.build_plan(
        forward_mode=ctx.forward_mode,
        real_bs=24,
        effective_bs=bucket_bs,
    )

    assert use_graph
    assert plan.is_dp_all_to_all
    assert plan.real_bs == 24
    assert plan.effective_bs == 24
    assert plan.bucket_bs == 32


def test_layout_planner_keeps_graph_bucket_below_threshold_non_dp():
    planner = LogitsLayoutPlanner(
        dp_sampling_enabled=True,
        dp_sampling_min_bs=32,
        tp_size=8,
        num_tokens_per_req=6,
    )
    ctx = ForwardContext(
        attn_backend=None,
        token_to_kv_pool=None,
        bs=23,
        num_extends=0,
        input_num_tokens=23,
        forward_mode=ForwardMode.DECODE,
    )

    use_graph, bucket_bs = _graph_route(23, ctx, max_bs=32, capture_bs=[24, 32])
    plan = planner.build_plan(
        forward_mode=ctx.forward_mode,
        real_bs=23,
        effective_bs=bucket_bs,
    )

    assert use_graph
    assert not plan.is_dp_all_to_all
    assert plan.real_bs == 23
    assert plan.effective_bs == 24
    assert plan.bucket_bs == 24


def test_layout_planner_uses_global_decode_bucket_for_idle_rank():
    planner = LogitsLayoutPlanner(
        dp_sampling_enabled=True,
        dp_sampling_min_bs=16,
        tp_size=8,
        num_tokens_per_req=6,
    )
    ctx = ForwardContext(
        attn_backend=None,
        token_to_kv_pool=None,
        bs=0,
        num_extends=0,
        input_num_tokens=0,
        forward_mode=ForwardMode.DECODE,
        global_num_tokens=[16, 0],
        global_bs=[16, 0],
        all_decode_or_idle=True,
    )

    use_graph, bucket_bs = _graph_route(
        0,
        ctx,
        dp_size=2,
        max_bs=32,
        capture_bs=[16, 32],
        max_tokens_per_req=1,
    )
    plan = planner.build_plan(
        forward_mode=ctx.forward_mode,
        real_bs=0,
        effective_bs=bucket_bs,
    )

    assert use_graph
    assert plan.is_dp_all_to_all
    assert plan.real_bs == 0
    assert plan.effective_bs == 16
    assert plan.bucket_bs == 16


def test_layout_planner_eager_route_returns_tp_divisible_bucket():
    planner = LogitsLayoutPlanner(
        dp_sampling_enabled=True,
        dp_sampling_min_bs=16,
        tp_size=4,
        num_tokens_per_req=6,
    )
    ctx = ForwardContext(
        attn_backend=None,
        token_to_kv_pool=None,
        bs=17,
        num_extends=0,
        input_num_tokens=17,
        forward_mode=ForwardMode.DECODE,
    )

    use_graph, bucket_bs = _graph_route(
        17, ctx, disable=True, max_bs=32, capture_bs=[24, 32]
    )
    plan = planner.build_plan(
        forward_mode=ctx.forward_mode,
        real_bs=17,
        effective_bs=bucket_bs,
    )

    assert not use_graph
    assert plan.is_dp_all_to_all
    assert plan.real_bs == 17
    assert plan.effective_bs == 17
    assert plan.bucket_bs == 20


def test_configure_dp_sampling_sets_state():
    processor = LogitsProcessor(
        SimpleNamespace(vocab_size=7, model_type="unit_test"),
        tp_rank=0,
        tp_size=4,
        tp_group=(0, 1, 2, 3),
    )

    processor.configure_dp_sampling(_dp_runtime_config())
    assert processor.dp_sampling_enabled
    assert processor.dp_num_tokens_per_req == 6


def test_resolve_dp_sampling_runtime_uses_grouped_metadata():
    support = DpSamplingSupport(
        requested=True,
        enabled=True,
        infra_supports=True,
        drafter_available=True,
        backend_supports_verify=True,
        tp_size=4,
        tp_group_set=True,
    )

    runtime_config = resolve_dp_sampling_runtime(
        support=support,
        lm_head_rows=7,
        topology=DpSamplingTopology(
            tp_rank=0,
            tp_size=4,
            tp_group=(0, 1, 2, 3),
            skip_all_gather=False,
        ),
        limits=DpSamplingRuntimeLimits(
            runtime_vocab_size=7,
            max_num_seqs=17,
            data_parallel_size=1,
            num_tokens_per_req=6,
            configured_min_bs=None,
            device="cpu",
        ),
    )

    assert runtime_config.enabled
    assert runtime_config.vocab_size == 28
    assert runtime_config.max_bucket_bs == 20
    assert runtime_config.min_bs == 8
    assert runtime_config.num_tokens_per_req == 6


@pytest.mark.parametrize(
    "forward_mode",
    [ForwardMode.DECODE, ForwardMode.TARGET_VERIFY],
)
def test_logits_processor_derives_dp_layout_from_effective_hidden_states(
    forward_mode,
):
    processor = LogitsProcessor(
        SimpleNamespace(vocab_size=7, model_type="unit_test"),
        tp_rank=0,
        tp_size=4,
        tp_group=(0, 1, 2, 3),
    )
    processor.configure_dp_sampling(_dp_runtime_config(min_bs=5))

    plan = processor._resolve_logits_layout_plan(
        torch.empty(5 * 6, 3),
        LogitsMetadata(forward_mode=forward_mode),
    )

    assert plan is not None
    assert plan.real_bs == 5
    assert plan.effective_bs == 5
    assert plan.bucket_bs == 8


def test_dp_sampling_skip_all_gather_rejects_sharded_lm_head_vocab():
    with pytest.raises(RuntimeError, match="replicated/full-vocab LM head"):
        validate_dp_sampling_lm_head_vocab(
            lm_head_rows=4,
            vocab_size=7,
            tp_size=2,
            skip_all_gather=True,
            tie_word_embeddings=True,
        )


def test_resolve_dp_sampling_support_rejects_missing_preconditions():
    with pytest.raises(RuntimeError, match="backend_supports_dp_verify=False"):
        resolve_dp_sampling_support(
            requested=True,
            drafter_available=True,
            backend_supports_verify=False,
            topology=DpSamplingTopology(
                tp_rank=0,
                tp_size=4,
                tp_group=(0, 1, 2, 3),
                skip_all_gather=False,
            ),
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
    plan = LogitsLayoutPlan.dp_all_to_all(
        real_bs=5,
        effective_bs=5,
        bucket_bs=8,
        tp_size=4,
        num_tokens_per_req=6,
    )

    logits = processor._get_logits(
        hidden_states,
        lm_head,
        LogitsMetadata(forward_mode=ForwardMode.DECODE),
        plan=plan,
    )

    assert logits.shape == (12, 7)
    expected_rows = hidden_states[12:24].sum(dim=1)
    assert torch.equal(logits[:, 0], expected_rows)


def test_dp_sampling_slices_graph_effective_hidden_states_before_lm_head():
    processor = LogitsProcessor(
        SimpleNamespace(vocab_size=7, model_type="unit_test"),
        skip_all_gather=True,
        tp_rank=2,
        tp_size=4,
        tp_group=(0, 1, 2, 3),
        dp_sampling_enabled=True,
        dp_num_tokens_per_req=6,
    )
    hidden_states = torch.arange(5 * 6 * 3, dtype=torch.float32).view(5 * 6, 3)
    lm_head = SimpleNamespace(weight=torch.ones(7, 3))
    plan = LogitsLayoutPlan.dp_all_to_all(
        real_bs=4,
        effective_bs=5,
        bucket_bs=8,
        tp_size=4,
        num_tokens_per_req=6,
    )

    logits = processor._get_logits(
        hidden_states,
        lm_head,
        LogitsMetadata(forward_mode=ForwardMode.DECODE),
        plan=plan,
    )

    assert logits.shape == (12, 7)
    expected_rows = torch.cat(
        [hidden_states[24:30].sum(dim=1), torch.zeros(6, dtype=torch.float32)]
    )
    assert torch.equal(logits[:, 0], expected_rows)
