from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

from tokenspeed.runtime.execution.context import ForwardContext
from tokenspeed.runtime.execution.cuda_graph_wrapper import CudaGraphWrapper
from tokenspeed.runtime.execution.forward_batch_info import ForwardMode
from tokenspeed.runtime.layers.logits_processor import LogitsMetadata, LogitsProcessor
from tokenspeed.runtime.models.extensible import ExtensibleLM
from tokenspeed.runtime.sampling.dp_sampling_config import (
    DpSamplingRuntimeConfig,
    DpSamplingRuntimeLimits,
    DpSamplingSupport,
    DpSamplingTopology,
    resolve_dp_sampling_runtime,
    resolve_dp_sampling_support,
    validate_dp_sampling_lm_head_vocab,
)
from tokenspeed.runtime.sampling.logits_layout import LogitsLayoutPlan


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
    wrapper = CudaGraphWrapper.__new__(CudaGraphWrapper)
    wrapper.disable = disable
    wrapper.dp_size = dp_size
    wrapper.disable_padding = disable_padding
    wrapper.max_bs = max_bs
    wrapper.capture_bs = capture_bs
    wrapper.graphs = set(capture_bs)
    wrapper.max_tokens_per_req = max_tokens_per_req
    use_graph = wrapper.can_run(bs, ctx)
    return use_graph, wrapper.padded_bs(bs, ctx) if use_graph else bs


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


def test_extensible_lm_exposes_base_sampling_setup_handles():
    base = SimpleNamespace(logits_processor=object(), lm_head=object())
    ext = ExtensibleLM.__new__(ExtensibleLM)
    torch.nn.Module.__init__(ext)
    ext.base_lm = base

    assert ext.logits_processor is base.logits_processor
    assert ext.lm_head is base.lm_head


def test_logits_processor_dp_layout_threshold_and_modes():
    processor = LogitsProcessor(
        SimpleNamespace(vocab_size=7, model_type="unit_test"),
        tp_rank=0,
        tp_size=4,
        tp_group=(0, 1, 2, 3),
    )
    processor.configure_dp_logits_layout(_dp_runtime_config(min_bs=16))

    assert (
        processor._resolve_logits_layout_plan(
            torch.empty(15 * 6, 3),
            LogitsMetadata(forward_mode=ForwardMode.DECODE),
        )
        is None
    )

    decode_plan = processor._resolve_logits_layout_plan(
        torch.empty(16 * 6, 3),
        LogitsMetadata(forward_mode=ForwardMode.DECODE),
    )
    assert decode_plan is not None

    assert (
        processor._resolve_logits_layout_plan(
            torch.empty(32 * 6, 3),
            LogitsMetadata(forward_mode=ForwardMode.EXTEND),
        )
        is None
    )


def test_cuda_graph_wrapper_uses_existing_route_for_padding():
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

    assert wrapper.can_run(30, ctx)
    assert wrapper.padded_bs(30, ctx) == 32


def test_cuda_graph_req_pool_padding_keeps_attention_default_row():
    active_indices = torch.tensor([7, 8], dtype=torch.int64)

    padded_indices = CudaGraphWrapper._pad_graph_req_pool_indices(active_indices, 4)

    assert padded_indices.tolist() == [7, 8, 0, 0]


def test_cuda_graph_state_write_padding_uses_reserved_sink_row():
    wrapper = CudaGraphWrapper.__new__(CudaGraphWrapper)
    wrapper.config = SimpleNamespace(max_req_pool_size=99)
    wrapper.input_buffers = SimpleNamespace(
        state_write_req_pool_indices_buf=torch.full((4,), -1, dtype=torch.int64)
    )
    active_indices = torch.tensor([7, 8], dtype=torch.int64)

    wrapper._set_graph_state_write_indices(active_indices, 4)

    assert wrapper.input_buffers.state_write_req_pool_indices_buf.tolist() == [
        7,
        8,
        99,
        99,
    ]


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


def test_configure_dp_sampling_sets_state():
    processor = LogitsProcessor(
        SimpleNamespace(vocab_size=7, model_type="unit_test"),
        tp_rank=0,
        tp_size=4,
        tp_group=(0, 1, 2, 3),
    )

    processor.configure_dp_logits_layout(_dp_runtime_config())
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
    [ForwardMode.DECODE],
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
    processor.configure_dp_logits_layout(_dp_runtime_config(min_bs=5))

    plan = processor._resolve_logits_layout_plan(
        torch.empty(5 * 6, 3),
        LogitsMetadata(forward_mode=forward_mode),
    )

    assert plan is not None
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
    )
    processor.configure_dp_logits_layout(
        _dp_runtime_config(tp_rank=1, skip_all_gather=True, device="cpu")
    )
    hidden_states = torch.arange(5 * 6 * 3, dtype=torch.float32).view(5 * 6, 3)
    lm_head = SimpleNamespace(weight=torch.ones(7, 3))
    plan = LogitsLayoutPlan(
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
    )
    processor.configure_dp_logits_layout(
        _dp_runtime_config(tp_rank=2, skip_all_gather=True, device="cpu")
    )
    hidden_states = torch.arange(5 * 6 * 3, dtype=torch.float32).view(5 * 6, 3)
    lm_head = SimpleNamespace(weight=torch.ones(7, 3))
    plan = LogitsLayoutPlan(
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
