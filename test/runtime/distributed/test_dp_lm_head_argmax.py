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

"""Cross-DP greedy heads and attention-local heads over real NCCL groups."""

import os
import subprocess
import sys
from datetime import timedelta
from pathlib import Path

import pytest
import torch
import torch.distributed as dist


def _worker():
    fused = "--fused" in sys.argv
    vocab = 32768 if fused else 32
    dtype = torch.bfloat16 if "--bf16" in sys.argv else torch.float32
    from transformers import PretrainedConfig

    from tokenspeed.runtime.distributed.comm_backend.nccl import NcclBackend
    from tokenspeed.runtime.distributed.comm_backend.registry import set_global_backend
    from tokenspeed.runtime.distributed.process_group_manager import (
        process_group_manager,
    )
    from tokenspeed.runtime.execution.forward_batch_info import ForwardMode
    from tokenspeed.runtime.layers.logits_processor import (
        LogitsMetadata,
        LogitsProcessor,
    )
    from tokenspeed.runtime.layers.vocab_parallel_embedding import ParallelLMHead

    rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(rank)
    # Compare different GEMM shapes without TF32 rounding changing the reference.
    torch.set_float32_matmul_precision("highest")
    torch.backends.cuda.matmul.allow_tf32 = False
    device = torch.device("cuda", rank)
    dist.init_process_group("nccl", device_id=device, timeout=timedelta(seconds=90))
    world = tuple(range(8))
    process_group_manager.register_process_group("nccl", world, dist.group.WORLD)
    for first in range(0, 8, 2):
        group = (first, first + 1)
        pg = dist.new_group(list(group), backend="nccl")
        process_group_manager.register_process_group("nccl", group, pg)
    set_global_backend(NcclBackend())
    weights = (
        torch.arange(vocab * 4, device=device, dtype=torch.float32).reshape(vocab, 4)
        / (vocab / 2)
        - 4
    )
    weights = weights.to(dtype)
    weights[0] = weights[-1]  # Tie across shards: lowest global ID wins.
    for tp in (8, 2):
        group = world if tp == 8 else (rank // 2 * 2, rank // 2 * 2 + 1)
        local_rank = group.index(rank)
        proc = LogitsProcessor(
            PretrainedConfig(vocab_size=vocab),
            tp_rank=local_rank,
            tp_size=tp,
            tp_group=group,
            attn_tp_size=2,
            dp_lm_head_tp=True,
            skip_all_gather=True,
            do_argmax=True,
        )
        head = (
            ParallelLMHead(
                vocab,
                4,
                params_dtype=dtype,
                padding_size=1,
                tp_rank=local_rank,
                tp_size=tp,
                tp_group=group,
            )
            .to(device)
            .requires_grad_(False)
        )
        with torch.no_grad():
            head.weight.copy_(
                weights[local_rank * (vocab // tp) : (local_rank + 1) * (vocab // tp)]
            )
        proc.prepare_communication(head)
        if fused:
            from tokenspeed_kernel.ops.sampling.cute_dsl import DistArgmaxState

            state = proc._prepared_dist_argmax_state(dtype)
            assert isinstance(state, DistArgmaxState), "requires fused argmax / NVLS"
            assert not state.skip_ping_pong
        draft = LogitsProcessor(
            PretrainedConfig(vocab_size=vocab),
            tp_rank=local_rank,
            tp_size=tp,
            tp_group=group,
            attn_tp_size=2,
            dp_lm_head_tp=True,
            skip_all_gather=True,
            do_argmax=True,
        )
        draft.prepare_communication(head)
        for dp_counts in ((2, 0, 3, 1), (0, 0, 0, 0), (2, 2, 2, 2)):
            counts = [n for n in dp_counts for _ in range(2)]
            hidden = (
                torch.arange(
                    counts[rank] * 4, device=device, dtype=torch.float32
                ).reshape(-1, 4)
                / 8
                + rank // 2
            )
            hidden = hidden.to(dtype)
            metadata = LogitsMetadata(
                forward_mode=ForwardMode.DECODE,
                query_shard=None,
                global_num_tokens=[4 * n for n in counts],
                collective_global_num_tokens=counts,
                all_decode_or_idle=True,
            )
            expected = (hidden @ weights.T).argmax(dim=-1)
            # No host synchronization between consecutive target/draft calls.
            outputs = [
                owner(None, hidden, head, metadata)
                for _ in range(8)
                for owner in (proc, draft, draft, draft)
            ]
            for result in outputs:
                torch.testing.assert_close(result.next_token_ids, expected)
            full = proc._get_logits(hidden, head, metadata, require_full_vocab=True)
            torch.testing.assert_close(full, hidden @ weights.T)
            if dp_counts == (2, 2, 2, 2):
                stream = torch.cuda.Stream()
                stream.wait_stream(torch.cuda.current_stream())
                with torch.cuda.stream(stream):
                    for _ in range(3):
                        proc(None, hidden, head, metadata)
                torch.cuda.current_stream().wait_stream(stream)
                graph = torch.cuda.CUDAGraph()
                with torch.cuda.graph(graph, stream=stream):
                    captured_round = [
                        owner(None, hidden, head, metadata)
                        for owner in (proc, draft, draft, draft)
                    ]
                for delta in (1.0, -10.0, 3.0, 10.0):
                    hidden.add_(delta)
                    graph.replay()
                    torch.cuda.synchronize()
                    for captured in captured_round:
                        torch.testing.assert_close(
                            captured.next_token_ids, (hidden @ weights.T).argmax(dim=-1)
                        )
                graph.reset()
        dist.barrier()
    dist.destroy_process_group()


@pytest.mark.parametrize("fused,bf16", [(False, False), (True, False), (True, True)])
def test_dp_lm_head_argmax_nccl(fused, bf16):
    if torch.cuda.device_count() < 8:
        pytest.skip("requires eight CUDA GPUs")
    subprocess.run(
        [
            sys.executable,
            "-m",
            "torch.distributed.run",
            "--standalone",
            "--nproc-per-node=8",
            str(Path(__file__).resolve()),
            *(["--fused"] if fused else []),
            *(["--bf16"] if bf16 else []),
        ],
        check=True,
        timeout=360,
    )


if __name__ == "__main__":
    _worker()
