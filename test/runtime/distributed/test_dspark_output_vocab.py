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

"""Compact V4.1 DSpark heads sample across a real NCCL group."""

import os
import sys
from test.runtime.test_deepseek_v41_dspark import _draft_config, _quant
from test.runtime.test_deepseek_v41_engram import _mapping
from test.runtime.test_deepseek_v41_model import _mock_loader_hardware

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from tokenspeed.runtime.configs.deepseek_v41_config import DeepseekV41Config
from tokenspeed.runtime.configs.vocabulary import VocabularySpec
from tokenspeed.runtime.distributed.comm_backend.nccl import NcclBackend
from tokenspeed.runtime.distributed.comm_backend.registry import set_global_backend
from tokenspeed.runtime.distributed.mapping import Mapping
from tokenspeed.runtime.distributed.process_group_manager import (
    process_group_manager as pg_manager,
)
from tokenspeed.runtime.model_loader.utils import set_default_torch_dtype
from tokenspeed.runtime.models.deepseek_v41_dspark import DeepseekV41ForCausalLMDSpark
from tokenspeed.runtime.models.deepseek_v41_dspark_heads import (
    dspark_greedy_workspace,
    sample_dspark_block_greedy,
)

sys.path.insert(
    0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
)
from ci_system.ci_register import register_cuda_ci  # noqa: E402

register_cuda_ci(est_time=60, suite="runtime-2gpu")


def _check_compact_head(rank, output_size, device, *, attention_dp):
    config = _draft_config()
    config.vocab_size = 512
    config.output_vocab_size = output_size
    config = DeepseekV41Config(text_config=vars(config))
    mapping = (
        Mapping(
            rank=rank,
            world_size=2,
            attn_tp_size=1,
            attn_dp_size=2,
            dense_tp_size=1,
            dense_dp_size=2,
            moe_tp_size=1,
            moe_ep_size=2,
            moe_dp_size=1,
        )
        if attention_dp
        else _mapping(rank, 2, 2)
    )
    # Only the unused MoE backbone's hardware setup is stubbed. The model's
    # head construction, weight loaders, projections and sampling are real.
    with pytest.MonkeyPatch.context() as patch, set_default_torch_dtype(torch.bfloat16):
        _mock_loader_hardware(patch, mapping)
        adapter = DeepseekV41ForCausalLMDSpark(
            config,
            mapping,
            _quant(),
            vocabulary=VocabularySpec.from_hf_config(config, is_draft=True),
        )
    generator = torch.Generator().manual_seed(0)

    def weights(*shape):
        return torch.randint(-2, 3, shape, generator=generator).to(torch.bfloat16)

    head = weights(output_size, config.hidden_size)
    embedding = weights(512, config.text_config.dspark_markov_rank)
    projection = weights(output_size, config.text_config.dspark_markov_rank)
    hidden = weights(3, 5, config.hidden_size)
    for module, weight in (
        (adapter.lm_head, head),
        (adapter.model.markov_embedding, embedding),
        (adapter.model.markov_projection, projection),
    ):
        module.weight.weight_loader(module.weight, weight)
    adapter.lm_head.to(device)
    adapter.model.markov_head.to(device)
    logits = adapter.model.local_base_logits(hidden.to(device), adapter.lm_head.weight)
    valid = adapter.model.markov_projection.shard_indices.num_org_elements
    # Padding must lose even when its base logits are much larger than every
    # real token. output_size=1 leaves rank 1 with no valid output columns.
    logits[..., valid:] = 1000000
    anchors = torch.tensor([511, 0, 510], dtype=torch.int32, device=device)
    output = torch.empty(3, 5, dtype=torch.int32, device=device)
    candidates, partials = dspark_greedy_workspace(
        mapping.lm_head.tp_size, 3, logits.shape[-1], device
    )
    for all_masked in (False, True):
        reference_logits = hidden.float() @ head.float().T
        if all_masked:
            logits[..., :valid] = -torch.inf
            reference_logits.fill_(-torch.inf)
        expected = torch.empty(3, 5, dtype=torch.int64)
        previous = anchors.cpu().long()
        for step in range(5):
            bias = embedding[previous].float() @ projection.float().T
            previous = (reference_logits[:, step] + bias).argmax(-1)
            expected[:, step] = previous
        sample_dspark_block_greedy(
            logits,
            anchors,
            adapter.model.markov_head,
            adapter.model.markov_projection,
            mapping.lm_head.tp_group,
            candidates,
            partials,
            output,
        )
        torch.testing.assert_close(output.cpu().long(), expected, rtol=0, atol=0)


def _worker(rank, init_method):
    torch.set_num_threads(1)
    device = torch.device(f"cuda:{rank}")
    torch.cuda.set_device(device)
    dist.init_process_group("nccl", init_method=init_method, rank=rank, world_size=2)
    pg_manager.init_process_group((0, 1), backend="nccl")
    set_global_backend(NcclBackend())
    try:
        for attention_dp in (False, True):
            for output_size in (256, 257, 1):
                _check_compact_head(
                    rank, output_size, device, attention_dp=attention_dp
                )
    finally:
        dist.destroy_process_group()


@pytest.mark.skipif(torch.cuda.device_count() < 2, reason="needs 2 GPUs")
def test_compact_dspark_sampling_matches_dense_reference(tmp_path):
    mp.spawn(_worker, args=(f"file://{tmp_path / 'nccl'}",), nprocs=2, join=True)
