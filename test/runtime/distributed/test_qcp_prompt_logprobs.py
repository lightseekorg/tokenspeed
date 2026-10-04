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

"""Prompt logprobs under query context parallelism, on CPU over gloo.

Four ranks hold one shard each of an extend forward's final hidden rows and
one vocab shard each of the LM head. The executor stages each rank's rows of
the prompt-logprob plan, the logits processor scores them (vocab-sharded GEMM,
vocab all-gather, the launch's ``--logprob-order``), gathers the fp32 results
over the group and only then gathers the sampled rows for the LM head. Every
rank must end with the whole plan's logprobs in row order -- the unsharded
forward's values -- and the batch's sampled logits; ranks without any planned
row, or without any row at all, join every collective.
"""

from __future__ import annotations

import socket
from types import SimpleNamespace

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

WORLD = 4
HIDDEN = 8
# Vocabulary per order: a small one for torch's log-softmax, two trainer
# blocks for the megatron fold (which needs a block multiple). Sharded over
# the four ranks' heads either way.
VOCABS = {"torch": 64, "megatron": 2 * 32768}


def _get_open_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("", 0))
        return s.getsockname()[1]


def _worker(rank: int, port: int, logprob_order: str, errors) -> None:
    try:
        _run(rank, port, logprob_order)
    except Exception:  # pragma: no cover - reported to the parent
        import traceback

        errors[rank] = traceback.format_exc()


def _staging_executor(shifted_ids: torch.Tensor, vocab: int):
    """Just enough of a ``ModelExecutor`` to run ``_input_logprob_rows``."""
    from tokenspeed.runtime.execution import model_executor as executor_module
    from tokenspeed.runtime.execution.nan_guard import NanGuard

    executor_module.is_pin_memory_available = lambda: False
    executor = object.__new__(executor_module.ModelExecutor)
    executor.device = "cpu"
    executor.runtime_states = SimpleNamespace(vocab_size=vocab)
    executor.config = SimpleNamespace(input_logprob_chunk_tokens=2)
    executor.input_buffers = SimpleNamespace(shifted_prefill_ids_buf=shifted_ids)
    executor.nan_guard = NanGuard(max_bs=8, device="cpu")
    executor.nan_guard.reset(8)
    return executor


def _forward(
    *,
    rank: int,
    processor,
    weight: torch.Tensor,
    lengths: list[int],
    plan,
    hidden: torch.Tensor,
    shifted_ids: torch.Tensor,
    logprob_order: str,
) -> None:
    """One sharded extend forward's exit on ``rank``; checks it against the
    unsharded computation and the other ranks' results."""
    from tokenspeed.runtime.execution.forward_batch_info import ForwardMode
    from tokenspeed.runtime.execution.query_shard import QueryShardPlan
    from tokenspeed.runtime.layers.logits_processor import LogitsMetadata
    from tokenspeed.runtime.sampling.utils import gather_token_logprobs

    total = sum(lengths)
    shard = QueryShardPlan.from_forward(
        total_tokens=total, input_lengths=lengths, size=WORLD, rank=rank
    )
    gather_ids = torch.cumsum(torch.tensor(lengths), 0) - 1
    vocab = weight.shape[0]
    rows = _staging_executor(shifted_ids, vocab)._input_logprob_rows(
        plan, len(lengths), total, shard
    )
    local = weight[rank * (vocab // WORLD) : (rank + 1) * (vocab // WORLD)]
    metadata = LogitsMetadata(
        forward_mode=ForwardMode.EXTEND,
        gather_ids=gather_ids,
        input_logprob_rows=rows,
        query_shard=shard,
    )
    out = processor(
        input_ids=None,
        hidden_states=hidden[shard.local_slice].clone(),
        lm_head=SimpleNamespace(weight=local),
        logits_metadata=metadata,
    )

    # The unsharded forward: every row through the whole head.
    plan_rows = torch.cat(
        [
            torch.arange(start, start + count)
            for start, count in zip(plan.row_starts, plan.counts)
        ]
    )
    logits = hidden @ weight.T
    expected = gather_token_logprobs(
        logits[plan_rows],
        shifted_ids[plan_rows].to(torch.int64),
        logprob_order=logprob_order,
    )
    assert out.input_token_logprobs.dtype == torch.float32
    assert out.input_token_logprobs.shape == (plan.num_rows,)
    torch.testing.assert_close(out.input_token_logprobs, expected)
    torch.testing.assert_close(out.next_token_logits, logits[gather_ids])
    assert rows.num_result_rows == plan.num_rows
    # Every rank holds the identical vector (the NaN audit reads it on each).
    everyone: list = [None] * WORLD
    dist.all_gather_object(everyone, out.input_token_logprobs)
    for other in everyone:
        assert torch.equal(other, out.input_token_logprobs)
    # And it is the tensor-parallel path's bit for bit: the same processor
    # over the replicated rows (no shard) meets the same operands per row.
    replicated = _staging_executor(shifted_ids, vocab)._input_logprob_rows(
        plan, len(lengths), total, None
    )
    reference = processor(
        input_ids=None,
        hidden_states=hidden.clone(),
        lm_head=SimpleNamespace(weight=local),
        logits_metadata=LogitsMetadata(
            forward_mode=ForwardMode.EXTEND,
            query_shard=None,
            gather_ids=gather_ids,
            input_logprob_rows=replicated,
        ),
    )
    assert torch.equal(reference.input_token_logprobs, out.input_token_logprobs)
    assert torch.equal(reference.next_token_logits, out.next_token_logits)


def _run(rank: int, port: int, logprob_order: str) -> None:
    from tokenspeed.runtime.distributed.comm_backend import registry
    from tokenspeed.runtime.distributed.comm_backend.nccl import NcclBackend
    from tokenspeed.runtime.distributed.mapping import Mapping
    from tokenspeed.runtime.distributed.process_group_manager import (
        process_group_manager as pg_manager,
    )
    from tokenspeed.runtime.execution.types import InputLogprobPlan
    from tokenspeed.runtime.layers.logits_processor import LogitsProcessor
    from tokenspeed.runtime.utils.env import global_server_args_dict

    mapping = Mapping(
        rank=rank, world_size=WORLD, attn_tp_size=WORLD, attn_qcp_size=WORLD
    )
    pg_manager.init_distributed(
        mapping, distributed_init_method=f"tcp://127.0.0.1:{port}", backend="gloo"
    )
    group = mapping.attn.tp_group
    assert group == mapping.attn.qcp_group
    pg_manager.init_process_group(group, backend="gloo")
    pg_manager.register_process_group(
        "nccl", group, pg_manager.get_process_group("gloo", group)
    )
    global_server_args_dict["force_deterministic_rsag"] = True
    global_server_args_dict["logprob_order"] = logprob_order
    global_server_args_dict["mapping"] = mapping
    registry._global_backend = NcclBackend()

    vocab = VOCABS[logprob_order]
    processor = LogitsProcessor(
        config=SimpleNamespace(
            model_type="test", vocab_size=vocab, final_logit_softcapping=None
        ),
        tp_rank=rank,
        tp_size=WORLD,
        tp_group=group,
        dp_lm_head_tp=False,
    )
    # No symmetric-memory multicast on CPU: the vocab gather is the collective.
    processor._all_gather_state = None

    generator = torch.Generator().manual_seed(3)
    weight = torch.randn(vocab, HIDDEN, generator=generator) * 2

    # Three requests of 5, 3 and 7 rows over shards of [4, 4, 4, 3] rows. The
    # plan wants rows 1..4 of the first (predicting prompt tokens 2..5), rows
    # 5..6 of the second and rows 10..11 of the third: ranks 0 and 1 score
    # three rows each, rank 2 two, rank 3 none.
    lengths = [5, 3, 7]
    hidden = torch.randn(sum(lengths), HIDDEN, generator=generator)
    shifted = torch.randint(0, vocab, (sum(lengths),), generator=generator).to(
        torch.int32
    )
    _forward(
        rank=rank,
        processor=processor,
        weight=weight,
        lengths=lengths,
        plan=InputLogprobPlan(
            row_starts=(1, 5, 10), counts=(4, 2, 2), position_starts=(1, 0, 0)
        ),
        hidden=hidden,
        shifted_ids=shifted,
        logprob_order=logprob_order,
    )

    # A two-row chunk over four ranks: ranks 2 and 3 hold no row at all and
    # still join the result gather and the sampled-row gather; the one
    # planned row is on rank 0, the sampled row on rank 1.
    lengths = [2]
    hidden = torch.randn(2, HIDDEN, generator=generator)
    shifted = torch.randint(0, vocab, (2,), generator=generator).to(torch.int32)
    _forward(
        rank=rank,
        processor=processor,
        weight=weight,
        lengths=lengths,
        plan=InputLogprobPlan(row_starts=(0,), counts=(1,), position_starts=(0,)),
        hidden=hidden,
        shifted_ids=shifted,
        logprob_order=logprob_order,
    )

    dist.barrier()
    dist.destroy_process_group()


@pytest.mark.parametrize("logprob_order", ["torch", "megatron"])
def test_sharded_prompt_logprobs_equal_the_unsharded_forward(logprob_order):
    port = _get_open_port()
    errors = mp.Manager().dict()
    mp.spawn(_worker, args=(port, logprob_order, errors), nprocs=WORLD, join=True)
    if errors:
        raise RuntimeError("\n".join(f"rank {r}: {e}" for r, e in errors.items()))


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
