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
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
# THE SOFTWARE.

"""Measure AMD token all-gather/reduce-scatter with Kimi DP distributions.

Run this with ``torchrun --standalone --nproc-per-node 8``. The benchmark
reports the minimum, mean, and maximum per-rank GPU time so an imbalanced or
empty-token rank cannot disappear behind rank-zero timing.
"""

from __future__ import annotations

import argparse
import json
import os
from collections.abc import Callable

import torch
import torch.distributed as dist
from tokenspeed_kernel.ops.communication.triton import (
    all_gather,
    create_state,
    reduce_scatter,
)


def _token_cases(world_size: int) -> dict[str, tuple[list[int], int]]:
    idle = [0] * (world_size - 1)
    # Kimi K2.5 uses four target-model verification rows per request. Its
    # Eagle draft steps still process one row per request. Include sparse,
    # balanced, and ragged distributions for both paths instead of treating a
    # uniform one-token distribution as the only decode-sized workload.
    verify_ragged = ([8, 8, 4, 4] + [0] * world_size)[:world_size]
    # Keep this near the configured 8K prefill limit while varying every
    # active rank's local grid size. The canonical values target WS=8; smaller
    # jobs receive the corresponding prefix.
    prefill_ragged = ([4096, 2048, 1024, 512, 256, 128, 64, 0] + [0] * world_size)[
        :world_size
    ]
    medium_ragged = ([128, 64, 32, 16, 8, 4, 2, 0] + [0] * world_size)[:world_size]
    return {
        "single": ([1] * world_size, 100),
        "decode": ([8] * world_size, 50),
        "prefill": ([128] * world_size, 20),
        "draft_sparse": ([1] + idle, 100),
        "draft_balanced_c8": ([1] * world_size, 100),
        "draft_balanced_c16": ([2] * world_size, 100),
        "verify_sparse": ([4] + idle, 100),
        "verify_balanced_c8": ([4] * world_size, 100),
        "verify_balanced_c16": ([8] * world_size, 50),
        "verify_ragged": (verify_ragged, 50),
        "prefill_skewed_128": ([128] + idle, 20),
        "prefill_ragged_128": (medium_ragged, 20),
        "prefill_balanced_128": ([128] * world_size, 20),
        "prefill_balanced_1024": ([1024] * world_size, 10),
        "prefill_ragged_8k": (prefill_ragged, 10),
        "prefill_balanced_8192": ([8192] * world_size, 5),
        "kimi_idle": ([8192] + idle, 5),
        "kimi_root_idle": (idle + [8192], 5),
    }


def _measure(
    operation: Callable[[], torch.Tensor],
    warmups: int,
    repetitions: int,
    world_size: int,
    device: torch.device,
) -> dict[str, float]:
    for _ in range(warmups):
        operation()
    torch.cuda.synchronize(device)
    dist.barrier()

    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(repetitions):
        operation()
    end.record()
    end.synchronize()

    local_ms = start.elapsed_time(end) / repetitions
    local = torch.tensor([local_ms], dtype=torch.float64, device=device)
    per_rank = [torch.empty_like(local) for _ in range(world_size)]
    dist.all_gather(per_rank, local)
    values = torch.stack(per_rank).cpu().flatten()
    return {
        "min_rank_ms": values.min().item(),
        "mean_rank_ms": values.mean().item(),
        "max_rank_ms": values.max().item(),
    }


def _check_all_gather(
    output: torch.Tensor,
    tokens: list[int],
    hidden_size: int,
) -> None:
    assert output.shape == (sum(tokens), hidden_size)
    offset = 0
    for peer, peer_tokens in enumerate(tokens):
        if peer_tokens:
            expected = torch.tensor(
                peer + 1,
                dtype=output.dtype,
                device=output.device,
            )
            torch.testing.assert_close(output[offset, 0], expected)
        offset += peer_tokens


def _check_reduce_scatter(
    output: torch.Tensor,
    local_tokens: int,
    hidden_size: int,
    world_size: int,
) -> None:
    assert output.shape == (local_tokens, hidden_size)
    if local_tokens:
        expected = torch.tensor(
            world_size * (world_size + 1) // 2,
            dtype=output.dtype,
            device=output.device,
        )
        torch.testing.assert_close(output[0, 0], expected)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--hidden-size", type=int, default=7168)
    parser.add_argument(
        "--cases",
        nargs="+",
        default=[
            "single",
            "decode",
            "prefill",
            "kimi_idle",
            "kimi_root_idle",
        ],
    )
    parser.add_argument("--label", default="candidate")
    parser.add_argument(
        "--warmups",
        type=int,
        default=100,
        help="untimed iterations per operation and token distribution",
    )
    parser.add_argument(
        "--repetitions",
        type=int,
        help="override the per-case timed repetition count",
    )
    args = parser.parse_args()

    device = torch.device(f"cuda:{int(os.environ['LOCAL_RANK'])}")
    torch.cuda.set_device(device)
    dist.init_process_group("nccl", device_id=device)
    rank = dist.get_rank()
    world_size = dist.get_world_size()

    try:
        available_cases = _token_cases(world_size)
        unknown = sorted(set(args.cases) - set(available_cases))
        if unknown:
            raise ValueError(f"unknown cases: {unknown}")

        max_tokens = max(sum(available_cases[name][0]) for name in args.cases)
        state = create_state(
            group=dist.group.WORLD,
            rank_in_group=rank,
            max_tokens=max_tokens,
            hidden_size=args.hidden_size,
            device=device,
        )

        for name in args.cases:
            tokens, default_repetitions = available_cases[name]
            repetitions = args.repetitions or default_repetitions
            local_tokens = tokens[rank]
            gather_input = torch.full(
                (local_tokens, args.hidden_size),
                rank + 1,
                dtype=torch.bfloat16,
                device=device,
            )
            scatter_input = torch.full(
                (sum(tokens), args.hidden_size),
                rank + 1,
                dtype=torch.bfloat16,
                device=device,
            )

            gather_output = all_gather(
                state,
                gather_input,
                token_list_in_group=tokens,
                safe=False,
            )
            _check_all_gather(gather_output, tokens, args.hidden_size)
            scatter_output = reduce_scatter(
                state,
                scatter_input,
                token_list_in_group=tokens,
                safe=False,
            )
            _check_reduce_scatter(
                scatter_output,
                local_tokens,
                args.hidden_size,
                world_size,
            )

            gather_stats = _measure(
                lambda: all_gather(
                    state,
                    gather_input,
                    token_list_in_group=tokens,
                    safe=False,
                ),
                args.warmups,
                repetitions,
                world_size,
                device,
            )
            scatter_stats = _measure(
                lambda: reduce_scatter(
                    state,
                    scatter_input,
                    token_list_in_group=tokens,
                    safe=False,
                ),
                args.warmups,
                repetitions,
                world_size,
                device,
            )

            if rank == 0:
                print(
                    json.dumps(
                        {
                            "label": args.label,
                            "case": name,
                            "tokens": tokens,
                            "hidden_size": args.hidden_size,
                            "repetitions": repetitions,
                            "all_gather": gather_stats,
                            "reduce_scatter": scatter_stats,
                        },
                        sort_keys=True,
                    ),
                    flush=True,
                )
            dist.barrier()
    finally:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
