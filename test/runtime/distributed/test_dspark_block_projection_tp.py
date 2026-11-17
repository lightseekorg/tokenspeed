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

"""Sampled DSpark's block projection over a vocab-sharded LM head, on CPU over gloo.

Four ranks each hold one vocab shard of the head. Sampling reads every block
position's scaled whole-vocabulary logits; the drafter projects all positions
in one call, so their rows share one vocab all-gather. Every position must
still get exactly the logits its own per-position call returns.
"""

from __future__ import annotations

import socket
from types import SimpleNamespace

import torch
import torch.distributed as dist
import torch.multiprocessing as mp

WORLD = 4
HIDDEN = 8
VOCAB = 64
ROWS = 3
POSITIONS = 5


def _get_open_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("", 0))
        return s.getsockname()[1]


def _worker(rank: int, port: int, errors) -> None:
    try:
        _run(rank, port)
    except Exception:  # pragma: no cover - reported to the parent
        import traceback

        errors[rank] = traceback.format_exc()


class _ShardIndices:
    def __init__(self, num_org: int, start: int) -> None:
        self.num_org_elements = num_org
        self.org_vocab_start_index = start


class _VocabShard:
    """This rank's rows of a vocab-parallel head, as the block projection reads them."""

    def __init__(self, weight: torch.Tensor, start: int) -> None:
        self.weight = weight
        self.shard_indices = _ShardIndices(weight.shape[0], start)
        self.org_vocab_size = VOCAB


def _run(rank: int, port: int) -> None:
    from tokenspeed.runtime.distributed.comm_backend import registry
    from tokenspeed.runtime.distributed.comm_backend.nccl import NcclBackend
    from tokenspeed.runtime.distributed.mapping import Mapping
    from tokenspeed.runtime.distributed.process_group_manager import (
        process_group_manager as pg_manager,
    )
    from tokenspeed.runtime.execution.drafter.dspark import DSpark
    from tokenspeed.runtime.execution.forward_batch_info import ForwardMode
    from tokenspeed.runtime.layers.logits_processor import (
        LogitsMetadata,
        LogitsProcessor,
    )
    from tokenspeed.runtime.utils.env import global_server_args_dict

    mapping = Mapping(rank=rank, world_size=WORLD, attn_tp_size=WORLD)
    pg_manager.init_distributed(
        mapping, distributed_init_method=f"tcp://127.0.0.1:{port}", backend="gloo"
    )
    group = mapping.attn.tp_group
    pg_manager.init_process_group(group, backend="gloo")
    pg_manager.register_process_group(
        "nccl", group, pg_manager.get_process_group("gloo", group)
    )
    global_server_args_dict["mapping"] = mapping
    registry._global_backend = NcclBackend()

    processor = LogitsProcessor(
        config=SimpleNamespace(
            model_type="test", vocab_size=VOCAB, final_logit_softcapping=None
        ),
        logit_scale=0.5,
        tp_rank=rank,
        tp_size=WORLD,
        tp_group=group,
        attn_tp_size=1,
        dp_lm_head_tp=False,
    )
    # No symmetric-memory multicast on CPU: the vocab gather is the collective.
    processor._all_gather_state = None

    generator = torch.Generator().manual_seed(3)
    weight = torch.randn(VOCAB, HIDDEN, generator=generator)
    hidden = torch.randn(ROWS, POSITIONS, HIDDEN, generator=generator)
    shard = VOCAB // WORLD
    head = _VocabShard(weight[rank * shard : (rank + 1) * shard], rank * shard)

    drafter = DSpark.__new__(DSpark)
    drafter.lm_head = head
    drafter.logits_processor = processor
    block = drafter._block_base_logits(hidden, full_vocab=True)

    assert block.shape == (POSITIONS, ROWS, VOCAB)
    metadata = LogitsMetadata(forward_mode=ForwardMode.DECODE, query_shard=None)
    for k in range(POSITIONS):
        step = processor._get_logits(
            hidden[:, k, :], head, metadata, require_full_vocab=True
        )
        # Equal up to float32 GEMM rounding across batch shapes.
        torch.testing.assert_close(block[k], step, msg=f"position {k}")
    # The whole vocabulary, scaled: not this rank's raw shard.
    torch.testing.assert_close(block[2], 0.5 * hidden[:, 2, :] @ weight.T)

    dist.barrier()
    dist.destroy_process_group()


def test_a_sampled_block_projection_gathers_each_positions_own_logits():
    port = _get_open_port()
    errors = mp.Manager().dict()
    mp.spawn(_worker, args=(port, errors), nprocs=WORLD, join=True)
    if errors:
        raise RuntimeError("\n".join(f"rank {r}: {e}" for r, e in errors.items()))
