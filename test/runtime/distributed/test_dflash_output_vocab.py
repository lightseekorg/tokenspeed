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

from __future__ import annotations

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from transformers import PretrainedConfig

from tokenspeed.runtime.distributed.comm_backend import registry
from tokenspeed.runtime.distributed.comm_backend.nccl import NcclBackend
from tokenspeed.runtime.distributed.mapping import Mapping
from tokenspeed.runtime.distributed.process_group_manager import (
    process_group_manager as pg_manager,
)
from tokenspeed.runtime.execution.drafter.dflash import DFlash
from tokenspeed.runtime.execution.input_buffer import InputBuffers
from tokenspeed.runtime.layers.logits_processor import LogitsProcessor
from tokenspeed.runtime.layers.vocab_parallel_embedding import ParallelLMHead
from tokenspeed.runtime.utils.env import global_server_args_dict


def _worker(rank: int, init_method: str) -> None:
    torch.set_num_threads(1)
    mapping = Mapping(rank=rank, world_size=4, attn_tp_size=4)
    pg_manager.init_distributed(
        mapping, distributed_init_method=init_method, backend="gloo"
    )
    group = mapping.attn.tp_group
    pg_manager.init_process_group(group, backend="gloo")
    pg_manager.register_process_group(
        "nccl", group, pg_manager.get_process_group("gloo", group)
    )
    registry._global_backend = NcclBackend()
    global_server_args_dict["mapping"] = mapping
    global_server_args_dict["force_deterministic_rsag"] = True
    global_server_args_dict["logprob_order"] = "torch"
    try:
        drafter = DFlash.__new__(DFlash)
        drafter.spec_num_tokens = 3
        drafter.input_buffers = InputBuffers(
            max_bs=2, max_num_tokens=4, state_write_padding_pool_index=3, device="cpu"
        )
        drafter.lm_head = ParallelLMHead(
            16,
            2,
            params_dtype=torch.float32,
            padding_size=4,
            tp_rank=rank,
            tp_size=4,
            tp_group=group,
        )
        drafter.logits_processor = LogitsProcessor(
            PretrainedConfig(),
            vocab_size=5,
            tp_rank=rank,
            tp_size=4,
            tp_group=group,
            dp_lm_head_tp=False,
        )
        drafter._dist_argmax_state = None
        drafter._greedy_gathered_max = None
        drafter._greedy_gathered_ids = None
        drafter._greedy_gather_cap = 0
        assert (
            drafter._probe_dist_argmax_state(torch.float32, torch.device("cpu")) is None
        )
        hidden = torch.tensor([[1.0, 0.0], [2.0, 0.0]])
        weight = torch.zeros(16, 2)
        weight[:5, 0] = torch.arange(-5, 0)
        weight[5:, 0] = 1000
        for all_masked in (False, True):
            if all_masked:
                weight[:5, 0] = -torch.inf
            with torch.no_grad():
                drafter.lm_head.weight.copy_(weight[rank * 4 : (rank + 1) * 4])
            for precompute in (False, True):
                base_logits = hidden @ drafter.lm_head.weight.T if precompute else None
                actual = drafter._greedy_argmax_vocab_parallel(
                    hidden, base_logits=base_logits
                )
                # Ranks 2/3 have no logical output tokens. They still join the
                # collective and cannot beat either finite or -inf valid logits.
                expected = (hidden @ weight[:5].T).argmax(-1)
                torch.testing.assert_close(actual.long(), expected)
    finally:
        dist.destroy_process_group()


def test_block_draft_argmax_ignores_empty_output_shards(tmp_path):
    init_method = f"file://{tmp_path / 'gloo'}"
    mp.spawn(_worker, args=(init_method,), nprocs=4, join=True)
