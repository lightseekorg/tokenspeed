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

"""The EAGLE-4 candidate head under TP2, over a real NCCL all-gather.

The executor prepares every logits processor's distributed argmax at startup
for the vocab shard its LM head holds; the candidate head's logits are its
``num_candidates`` gathered candidates, whole on every rank, so its greedy
token must still be the best of them.

Usage:
    python -m pytest test/runtime/distributed/test_eagle4_candidate_head_tp.py -v
"""

import faulthandler
import os
import signal
import subprocess
import sys
import time

# Cover imports as well as the worker body: spawn can stall before calling
# the target, so installing the handler only inside _worker misses its stacks.
_WORKER_MODE = __name__ == "__main__" and sys.argv[1:2] == ["--worker"]
if _WORKER_MODE:
    faulthandler.register(signal.SIGUSR1, all_threads=True)

import pytest
import torch
import torch.distributed as dist
from transformers import LlamaConfig

sys.path.insert(
    0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
)
from ci_system.ci_register import register_cuda_ci

register_cuda_ci(
    est_time=60,
    suite="runtime-2gpu",
)

# Shards of 8192 tokens: wide enough that startup prepares the fused distributed argmax.
VOCAB, HIDDEN, RANK, NUM_CANDIDATES, ROWS = 16384, 32, 8, 128, 5
# The test takes under a minute; past this a rank is stuck.
JOIN_SECONDS = 300


def _check_greedy_candidate(rank, world_size, device):
    from tokenspeed.runtime.distributed.mapping import Mapping
    from tokenspeed.runtime.execution.forward_batch_info import ForwardMode
    from tokenspeed.runtime.layers.logits_processor import (
        LogitsMetadata,
        prepare_logits_communication,
    )
    from tokenspeed.runtime.layers.moe import utils as moe_utils
    from tokenspeed.runtime.models.llama_eagle4 import (
        CandidateHeads,
        LlamaForCausalLMEagle4,
    )
    from tokenspeed.runtime.utils.env import global_server_args_dict

    mapping = Mapping(rank=rank, world_size=world_size)
    global_server_args_dict.update(
        ep_num_redundant_experts=0,
        max_model_len=256,
        mapping=mapping,
        comm_fusion_max_num_tokens=256,
        chunked_prefill_size=256,
        max_prefill_tokens=256,
        max_num_seqs=ROWS,
    )
    moe_utils.MOE_BACKEND = moe_utils.MoeBackend.AUTO
    config = LlamaConfig(
        vocab_size=VOCAB,
        hidden_size=HIDDEN,
        intermediate_size=64,
        num_hidden_layers=1,
        num_attention_heads=4,
        num_key_value_heads=4,
        max_position_embeddings=256,
        rms_norm_eps=1e-6,
        tie_word_embeddings=False,
        draft_vocab_size=VOCAB,
        efficient_lm_head_config={
            "type": "low_rank",
            "rank": RANK,
            "num_candidates": NUM_CANDIDATES,
        },
    )
    torch.set_default_dtype(torch.bfloat16)
    with torch.device(device):
        model = LlamaForCausalLMEagle4(config, mapping)

    # The same full weights and activations on every rank; each rank keeps its vocab shard.
    generator = torch.Generator(device="cpu").manual_seed(0)
    lm_head = torch.randn(VOCAB, HIDDEN, generator=generator)
    scores_head = torch.randn(VOCAB, RANK, generator=generator)
    down_proj = torch.randn(RANK, HIDDEN, generator=generator)
    hidden = torch.randn(ROWS, HIDDEN, generator=generator).to(device, torch.bfloat16)
    shard = slice(rank * VOCAB // world_size, (rank + 1) * VOCAB // world_size)
    model.lm_head.weight.data.copy_(lm_head[shard])
    model.efficient_lm_head.weight.data.copy_(scores_head[shard])
    model.efficient_lm_head_down_proj.weight.data.copy_(down_proj)
    # The executor's startup step, before any forward.
    prepare_logits_communication(model)

    heads = CandidateHeads(
        lm_head=model.lm_head,
        scores_head=model.efficient_lm_head,
        down_proj=model.efficient_lm_head_down_proj,
        num_candidates=NUM_CANDIDATES,
    )
    output = model.logits_processor(
        None,
        hidden,
        heads,
        LogitsMetadata(forward_mode=ForwardMode.DECODE, query_shard=None),
    )
    logits, ids, best = (
        output.next_token_logits,
        output.candidate_ids,
        output.next_token_ids,
    )
    torch.cuda.synchronize()

    low_rank = hidden @ down_proj.to(device, torch.bfloat16).T
    scores = (low_rank @ scores_head.to(device, torch.bfloat16).T).float()
    shard_vocab = VOCAB // world_size
    # Each id is among its shard's top candidates: at least the shard's k-th
    # largest score (ties at it may be any of them).
    kth = torch.stack(
        [
            torch.topk(
                scores[:, r * shard_vocab : (r + 1) * shard_vocab], NUM_CANDIDATES
            ).values[:, -1]
            for r in range(world_size)
        ],
        dim=-1,
    )
    assert bool((scores.gather(-1, ids) >= kth.gather(-1, ids // shard_vocab)).all())
    assert bool((ids.sort(-1).values.diff(dim=-1) > 0).all())  # distinct ids
    assert torch.equal(best, ids.gather(-1, logits.argmax(-1, keepdim=True)).view(-1))
    gathered = [torch.empty_like(best) for _ in range(world_size)]
    dist.all_gather(gathered, best)
    assert all(torch.equal(g, best) for g in gathered)


def _worker(rank, world_size, rendezvous):
    # The test asks a stuck rank for its stacks before killing it.
    faulthandler.register(signal.SIGUSR1, all_threads=True)
    device = torch.device(f"cuda:{rank}")
    torch.cuda.set_device(device)
    dist.init_process_group(
        backend="nccl",
        init_method=rendezvous,
        rank=rank,
        world_size=world_size,
        device_id=device,
    )
    from tokenspeed.runtime.distributed.process_group_manager import (
        process_group_manager as pg_manager,
    )

    pg_manager.init_process_group(tuple(range(world_size)))
    _check_greedy_candidate(rank, world_size, device)
    dist.destroy_process_group()


def _wait_for_workers(processes, timeout_seconds):
    deadline = time.monotonic() + timeout_seconds
    while True:
        exit_codes = [process.poll() for process in processes]
        for rank, code in enumerate(exit_codes):
            if code is not None and code != 0:
                raise RuntimeError(
                    f"rank {rank} exited with code {code}; see worker output"
                )
        if all(code is not None for code in exit_codes):
            return
        if time.monotonic() > deadline:
            # One rank at a time, so the two dumps do not interleave.
            for rank, process in enumerate(processes):
                if process.poll() is None:
                    print(f"rank {rank} stacks:", file=sys.stderr, flush=True)
                    process.send_signal(signal.SIGUSR1)
                    time.sleep(5)
            raise RuntimeError(f"ranks still running after {timeout_seconds} s")
        time.sleep(0.1)


def _stop_workers(processes):
    for process in processes:
        if process.poll() is None:
            process.kill()
    for process in processes:
        try:
            process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            # A driver wait can survive SIGKILL. Keep the original failure and
            # still reap the other ranks before reporting this survivor.
            print(
                f"worker pid {process.pid} survived kill", file=sys.stderr, flush=True
            )


def test_worker_failure_does_not_wait_for_a_stuck_peer():
    processes = []
    try:
        processes.append(
            subprocess.Popen([sys.executable, "-c", "raise SystemExit(7)"])
        )
        processes.append(
            subprocess.Popen([sys.executable, "-c", "import signal; signal.pause()"])
        )
        with pytest.raises(RuntimeError, match="rank 0 exited with code 7"):
            _wait_for_workers(processes, 5)
        assert processes[1].poll() is None
    finally:
        _stop_workers(processes)


@pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.device_count() < 2,
    reason="needs 2 GPUs",
)
def test_two_ranks_pick_the_best_of_their_gathered_candidates(tmp_path):
    # An exec boundary lets the worker install diagnostics before importing
    # Torch. No initialized parent state or Python worker objects cross it.
    processes = []
    try:
        for rank in range(2):
            processes.append(
                subprocess.Popen(
                    [
                        sys.executable,
                        os.path.abspath(__file__),
                        "--worker",
                        str(rank),
                        "2",
                        f"file://{tmp_path / 'rendezvous'}",
                    ],
                    env={**os.environ, "PYTHONUNBUFFERED": "1"},
                )
            )
        _wait_for_workers(processes, JOIN_SECONDS)
    finally:
        _stop_workers(processes)


if __name__ == "__main__":
    if _WORKER_MODE:
        _worker(int(sys.argv[2]), int(sys.argv[3]), sys.argv[4])
    else:
        raise SystemExit(pytest.main([__file__, "-v"]))
