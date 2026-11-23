# Copyright (c) 2026 LightSeek Foundation
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

"""The ForwardStepRunner's draft-overlap decision, through its constructor.

A real runner, Eagle drafter and KV pool views; only the attention backends'
graph-state hook stands in, as the backends are model-sized.
"""

import logging
from test.ci_system.ci_register import register_cuda_ci
from test.runtime.cache_pool_test_utils import make_mha_pool_for_groups
from test.runtime.drafter_prefill_graph_test_utils import make_eagle

import pytest
import torch
from torch import nn

from tokenspeed.runtime.execution.forward_step import ForwardStepRunner
from tokenspeed.runtime.execution.model_executor import ModelExecutorConfig
from tokenspeed.runtime.layers.attention.kv_cache.mha import MHATokenToKVPool
from tokenspeed.runtime.layers.attention.kv_cache.recipes import spec
from tokenspeed.runtime.layers.moe import utils as moe_utils

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")

register_cuda_ci(est_time=10, suite="runtime-1gpu")

ENV = "TOKENSPEED_DRAFT_GRAPH_OVERLAP"


class OverlapDraft(nn.Module):
    """Stand in for model weights when checking the startup capability contract."""

    def __init__(self, *, supports_draft_overlap: bool):
        super().__init__()
        self.supports_draft_overlap = supports_draft_overlap


class Backend:
    """Only what the constructor calls on an attention backend."""

    def init_cuda_graph_state(self, *args, **kwargs) -> None:
        pass


def _config(**changes) -> ModelExecutorConfig:
    fields = dict(
        max_req_pool_size=5,
        output_length=4,
        enforce_eager=False,
        prefix_granularity=128,
        max_num_seqs=32,
        chunked_prefill_size=4096,
        input_vocab_size=32,
        vocab_size=32,
        context_len=4096,
        physical_context_len=4096,
        device="cuda",
        gpu_id=0,
        global_rank=0,
        cudagraph_capture_sizes=[1, 2, 4],
        disable_cuda_graph_padding=False,
        spec_topk=1,
        max_cudagraph_capture_size=4,
        model_is_mrope=False,
        autotune_cache_key=None,
        prefill_only=False,
        input_logprob_chunk_tokens=1024,
        decode_only_attention=False,
        prefill_graph_capture_batch_sizes=None,
        enable_speculative_sampling=False,
        enable_dp_local_cuda_graph=False,
        query_shard_size=1,
        query_shard_rank=0,
        enable_memory_saver=False,
        data_parallel_size=1,
        attention_tp_size=1,
        moe_tp_ep_size=1,
        world_size=1,
        spec_algo="MTP",
        spec_num_steps=3,
        spec_num_tokens=4,
        grammar_backend="none",
    )
    fields.update(changes)
    return ModelExecutorConfig(**fields)


def _pools(*, draft_offset: int) -> tuple[MHATokenToKVPool, MHATokenToKVPool]:
    group_specs = tuple(
        spec.CacheGroupSpec(
            group_id=f"g{i}",
            attn_kind="full_history",
            rows_per_page=128,
            entry_stride_tokens=1,
            replayable=False,
        )
        for i in range(2)
    )
    arena = make_mha_pool_for_groups(group_specs, device="cuda").arena
    target = MHATokenToKVPool(arena, torch.float16, 1, 1, 1, 0)
    draft = MHATokenToKVPool(
        arena, torch.float16, 1, 1, 1, 0, field_layer_offset=draft_offset
    )
    return target, draft


def _runner(
    *,
    config: ModelExecutorConfig | None = None,
    shared_backend: bool = False,
    draft_offset: int = 1,
    supports: bool = True,
) -> ForwardStepRunner:
    drafter, _ = make_eagle(device="cuda", spec_num_steps=3)
    drafter.draft_model_runner.model = OverlapDraft(supports_draft_overlap=supports)
    target_pool, draft_pool = _pools(draft_offset=draft_offset)
    target_backend = Backend()
    return ForwardStepRunner(
        forward_func=lambda *args, **kwargs: None,
        attn_backend=target_backend,
        token_to_kv_pool=target_pool,
        input_buffers=object(),
        config=config if config is not None else _config(),
        draft_attn_backend=target_backend if shared_backend else Backend(),
        draft_token_to_kv_pool=draft_pool,
        drafter=drafter,
    )


@pytest.fixture(autouse=True)
def _environment(monkeypatch):
    monkeypatch.setattr(moe_utils, "MOE_BACKEND", moe_utils.MoeBackend.AUTO)
    monkeypatch.delenv(ENV, raising=False)


@pytest.fixture
def overlap_log(caplog):
    # The runtime's loggers do not propagate to the root logger.
    logger = logging.getLogger("tokenspeed.runtime.execution._draft_overlap")
    logger.addHandler(caplog.handler)
    with caplog.at_level(logging.INFO, logger=logger.name):
        yield caplog
    logger.removeHandler(caplog.handler)


def test_unset_enables_and_wires_the_overlap_on_an_eligible_runner(overlap_log):
    runner = _runner()
    assert runner.draft_overlap is not None
    assert runner.drafter.decode_overlap is runner.draft_overlap
    assert "Draft graph overlap on" in overlap_log.text


def test_zero_leaves_an_eligible_runner_without_overlap(monkeypatch):
    monkeypatch.setenv(ENV, "0")
    runner = _runner()
    assert runner.draft_overlap is None
    assert runner.drafter.decode_overlap is None


@pytest.mark.parametrize(
    ("kwargs", "reason"),
    [
        (dict(shared_backend=True), "separate target/draft"),
        (dict(draft_offset=0), "separate target/draft"),
        (dict(supports=False), "does not declare support"),
        (dict(config=dict(enforce_eager=True)), "NVIDIA decode graphs"),
        (dict(config=dict(prefill_only=True)), "NVIDIA decode graphs"),
        (dict(config=dict(grammar_backend="xgrammar")), "does not support grammar"),
    ],
)
def test_unset_ineligible_runner_runs_without_overlap_and_logs(
    overlap_log, kwargs, reason
):
    if "config" in kwargs:
        kwargs = dict(config=_config(**kwargs["config"]))
    runner = _runner(**kwargs)
    assert runner.draft_overlap is None
    assert reason in overlap_log.text


@pytest.mark.parametrize(
    ("config", "reason"),
    [
        (dict(data_parallel_size=2, world_size=2), "DP1/PP1"),
        (dict(pp_size=2), "DP1/PP1"),
    ],
)
def test_one_fails_on_dp_or_pp(monkeypatch, config, reason):
    monkeypatch.setenv(ENV, "1")
    with pytest.raises(ValueError, match=reason):
        _runner(config=_config(**config))


def test_one_fails_on_hip(monkeypatch):
    monkeypatch.setenv(ENV, "1")
    monkeypatch.setattr(torch.version, "hip", "6.4")
    with pytest.raises(ValueError, match="NVIDIA decode graphs"):
        _runner()


def test_mega_moe_mask_keeps_the_overlap_off(monkeypatch, overlap_log):
    monkeypatch.setattr(moe_utils, "MOE_BACKEND", moe_utils.MoeBackend.MEGA_MOE)
    runner = _runner()
    assert runner.draft_overlap is None
    assert "shared MoE row masks" in overlap_log.text


def test_a_nonzero_rank_does_not_log(overlap_log):
    runner = _runner(config=_config(global_rank=1))
    assert runner.draft_overlap is not None
    assert "Draft graph overlap" not in overlap_log.text
