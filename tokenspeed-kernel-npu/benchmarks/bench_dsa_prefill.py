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

"""Benchmark prefill transport, native indexer and sparse MLA on token owners.

Projections, model weights, scheduler and metadata construction are excluded.
Run with torchrun; its world size is the context-parallel degree. For example:

    torchrun --standalone --nproc-per-node=8 bench_dsa_prefill.py \
        --mode latency --histories 0 131072 524288 --layers 3 --repeat 5 \
        --output /large-disk/prefill-latency

Use --mode profile for one Level1/PipeUtilization step after warmup. Use
--mode sweep --histories 524288 --sweep-step 8192 --layers 24 to reproduce
allocator growth. A reference directory compares complete indices and outputs
bitwise. Inputs are generated on CPU and their SHA256 is logged on each rank.
"""

import argparse
import dataclasses
import hashlib
import json
import os
import socket
import time
from dataclasses import dataclass
from functools import partial
from pathlib import Path

import torch
import torch.distributed as dist
import torch_npu
from tokenspeed_kernel_npu.ops.dsa import AscendDSAKernels

from tokenspeed.runtime.layers.attention.dcp.ascend import (
    AscendPrefillOptions,
    AscendPrefillPlan,
)
from tokenspeed.runtime.layers.attention.dcp.placement import CachePlacement

CHUNK_TOKENS = 8192
PAGE_TOKENS = 128
HEADS = 32
TOPK = 2048


@dataclass
class Metadata:
    extend_seq_lens_cpu: torch.Tensor
    extend_prefix_lens_cpu: torch.Tensor
    page_table: torch.Tensor

    @property
    def cum_extend_seq_lens(self):
        return torch.tensor([0, CHUNK_TOKENS], dtype=torch.int32)

    @property
    def extend_prefix_lens(self):
        return self.extend_prefix_lens_cpu


@dataclass
class Inputs:
    kv_cache: torch.Tensor
    index_cache: torch.Tensor
    query: torch.Tensor
    index_query: torch.Tensor
    index_weights: torch.Tensor
    latent: torch.Tensor
    index_key: torch.Tensor


def move_to_npu(value):
    if isinstance(value, torch.Tensor):
        return value.npu()
    if dataclasses.is_dataclass(value):
        return dataclasses.replace(
            value,
            **{
                field.name: move_to_npu(getattr(value, field.name))
                for field in dataclasses.fields(value)
            },
        )
    return value


def make_inputs(rank, local_rows, cache_rows):
    generator = torch.Generator(device="cpu").manual_seed(8403 + rank)

    def normal(shape):
        return torch.randn(shape, dtype=torch.bfloat16, generator=generator).npu() * 0.1

    return Inputs(
        kv_cache=normal((cache_rows, 576)),
        index_cache=normal((cache_rows, 128)),
        query=normal((local_rows, HEADS, 576)),
        index_query=normal((local_rows, HEADS, 128)),
        index_weights=torch.rand(
            (local_rows, HEADS), dtype=torch.float32, generator=generator
        ).npu()
        * 0.03,
        latent=normal((local_rows, 576)),
        index_key=normal((local_rows, 128)),
    )


def memory():
    free, total = torch.npu.mem_get_info()
    stats = torch.npu.memory_stats()
    return {
        "allocated": torch.npu.memory_allocated(),
        "reserved": torch.npu.memory_reserved(),
        "driver_free": free,
        "driver_total": total,
        "peak_allocated": torch.npu.max_memory_allocated(),
        "peak_reserved": torch.npu.max_memory_reserved(),
        "inactive_split": stats["inactive_split_bytes.all.current"],
        "allocator_retries": stats["num_alloc_retries"],
        "allocator_ooms": stats["num_ooms"],
    }


def check_sparse_rows(plan, runtime, inputs, indices, output, history):
    """Independently recompute two sparse-attention rows on CPU.

    This checks the native attention math against the staged keys. Full
    transport correctness additionally requires the byte-exact transport tests
    or comparison with a previously checked reference run.
    """
    checks = []
    for row in sorted({0, inputs.query.shape[0] - 1}):
        selected = indices[row][indices[row] >= 0].long()
        length = history + plan.row_start + row + 1
        assert len(selected) == min(length, TOPK)
        assert len(selected.unique()) == len(selected)
        assert int(selected.max()) < length
        pages = plan.page_table[0, selected // PAGE_TOKENS].long()
        keys = (
            runtime.kv.view(-1, PAGE_TOKENS, 576)[pages, selected % PAGE_TOKENS]
            .cpu()
            .double()
        )
        query = inputs.query[row].cpu().double()
        probability = (query @ keys.T * 192**-0.5).softmax(-1).bfloat16().double()
        expected = (probability @ keys[:, :512]).bfloat16().double()
        actual = output[:, row].cpu().double()
        delta = (actual - expected).abs()
        scale = expected.square().mean(-1, keepdim=True).sqrt()
        assert torch.all(delta <= 2**-6 * (expected.abs() + scale)), (
            history,
            row,
            float(delta.max()),
        )
        checks.append({"row": row, "max_abs": float(delta.max())})
    return checks


def run_step(kernels, plan, runtime, inputs, slots, virtual_blocks, *, history, check):
    runtime.prefetch(
        0,
        inputs.kv_cache,
        inputs.index_cache,
        batch_key=plan,
        total_rows=CHUNK_TOKENS,
        has_history=history > 0,
        history_slots=plan.gather.send_slots,
        history_counts=plan.gather.row_counts,
    )
    runtime.gather_current(0, inputs.index_key, width=128, total_rows=CHUNK_TOKENS)
    runtime.gather_current(0, inputs.latent, width=576, total_rows=CHUNK_TOKENS)
    plan.store_current(
        runtime, 0, slots, block_rows=PAGE_TOKENS, virtual_blocks=virtual_blocks
    )
    indices = kernels.prefill_index(
        inputs.index_query,
        runtime.index.view(-1, PAGE_TOKENS, 1, 128),
        inputs.index_weights,
        plan.query_cu,
        plan.key_cu,
        plan.index_table,
        TOPK,
        0,
        0,
    )
    output = kernels.prefill_attention(
        inputs.query,
        runtime.kv.view(-1, PAGE_TOKENS, 576),
        indices,
        plan.scan_count,
        plan.page_table,
        plan.query_seq,
        192**-0.5,
    )
    checks = (
        check_sparse_rows(plan, runtime, inputs, indices, output, history)
        if check
        else []
    )
    runtime.finish(0)
    return indices, output, checks


def create_profiler(output):
    return torch_npu.profiler.profile(
        activities=[
            torch_npu.profiler.ProfilerActivity.CPU,
            torch_npu.profiler.ProfilerActivity.NPU,
        ],
        experimental_config=torch_npu.profiler._ExperimentalConfig(
            profiler_level=torch_npu.profiler.ProfilerLevel.Level1,
            aic_metrics=torch_npu.profiler.AiCMetrics.PipeUtilization,
        ),
        schedule=torch_npu.profiler.schedule(wait=0, warmup=0, active=1, repeat=1),
        on_trace_ready=torch_npu.profiler.tensorboard_trace_handler(str(output)),
        record_shapes=True,
    )


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--mode", choices=("latency", "profile", "sweep"), required=True
    )
    parser.add_argument("--histories", type=int, nargs="+", required=True)
    parser.add_argument("--layers", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--repeat", type=int)
    parser.add_argument("--sweep-step", type=int)
    parser.add_argument("--reference", type=Path)
    args = parser.parse_args()
    if min(args.histories) < 0 or args.layers < 1:
        parser.error("histories must be nonnegative and layers must be positive")
    if args.mode == "latency":
        if args.repeat is None or args.repeat < 1:
            parser.error("latency requires a positive --repeat")
    elif args.repeat is not None:
        parser.error("--repeat is only used by latency")
    if args.sweep_step is not None:
        if args.mode != "sweep" or args.sweep_step < 1 or len(args.histories) != 1:
            parser.error("--sweep-step needs sweep and one maximum history")
        maximum = args.histories[0]
        args.histories = list(range(0, maximum + 1, args.sweep_step))
        if args.histories[-1] != maximum:
            args.histories.append(maximum)
    if args.mode == "sweep" and args.reference is not None:
        parser.error("sweep logs memory without saving or comparing output tensors")
    return args


@torch.inference_mode()
def main():
    args = parse_args()
    rank, world = int(os.environ["RANK"]), int(os.environ["WORLD_SIZE"])
    torch.npu.set_device(int(os.environ["LOCAL_RANK"]))
    dist.init_process_group("hccl")
    args.output.mkdir(parents=True, exist_ok=True)
    log = (args.output / f"rank{rank}.jsonl").open("w", buffering=1)

    def emit(event, **values):
        record = {"event": event, "rank": rank, **values}
        log.write(json.dumps(record) + "\n")
        print(json.dumps(record), flush=True)

    max_pages = (max(args.histories) + CHUNK_TOKENS + PAGE_TOKENS - 1) // PAGE_TOKENS
    local_pages = (max_pages + world - 1) // world + 1
    local_rows = CHUNK_TOKENS // world + (rank < CHUNK_TOKENS % world)
    inputs = make_inputs(rank, local_rows, local_pages * PAGE_TOKENS)
    emit(
        "input_fingerprints",
        sha256={
            field.name: hashlib.sha256(
                getattr(inputs, field.name).cpu().view(torch.uint8).numpy().tobytes()
            ).hexdigest()
            for field in dataclasses.fields(inputs)
        },
    )
    import flash_ops

    emit(
        "operator_environment",
        flash_ops=flash_ops.__file__,
        registered=[
            name
            for name in torch._C._dispatch_get_all_op_names()
            if name.startswith("custom::")
            and any(part in name.lower() for part in ("lightning", "sparse", "dsa"))
        ],
    )
    kernels = AscendDSAKernels()
    runtime = kernels.prepare_prefill(
        inputs.kv_cache,
        inputs.index_cache,
        ranks=tuple(range(world)),
        history_group=dist.group.WORLD,
        max_current_rows=CHUNK_TOKENS,
        max_history_rows=max_pages * PAGE_TOKENS,
        history_null_rows=PAGE_TOKENS,
    )
    emit(
        "environment",
        host=socket.gethostname(),
        torch=torch.__version__,
        torch_npu=torch_npu.__version__,
        device=torch.npu.get_device_name(),
        world_size=world,
        mask=os.environ.get("ASCEND_RT_VISIBLE_DEVICES"),
        intra_roce=os.environ.get("HCCL_INTRA_ROCE_ENABLE"),
        workspace_bytes=runtime.workspace_bytes,
        memory=memory(),
        parameters=vars(args)
        | {
            "output": str(args.output),
            "reference": str(args.reference) if args.reference is not None else None,
        },
        boundary="transport+native indexer+sparse MLA+owner cache writes; projections excluded",
    )
    try:
        for history in args.histories:
            torch.npu.reset_peak_memory_stats()
            pages = (history + CHUNK_TOKENS + PAGE_TOKENS - 1) // PAGE_TOKENS
            metadata = Metadata(
                torch.tensor([CHUNK_TOKENS], dtype=torch.int32),
                torch.tensor([history], dtype=torch.int32),
                torch.arange(1, pages + 1, dtype=torch.int32)[None],
            )
            plan = move_to_npu(
                AscendPrefillPlan.build(
                    metadata,
                    CachePlacement(
                        PAGE_TOKENS, max_pages + 1, tuple(range(world)), rank
                    ),
                    degree=world,
                    rank=rank,
                    page_size=PAGE_TOKENS,
                    topk=TOPK,
                    local_cache_pages=local_pages,
                    options=AscendPrefillOptions(
                        CHUNK_TOKENS, PAGE_TOKENS, max_pages * PAGE_TOKENS
                    ),
                )
            )
            slots = torch.arange(
                history + PAGE_TOKENS,
                history + PAGE_TOKENS + CHUNK_TOKENS,
                dtype=torch.int64,
                device="npu",
            )

            step = partial(
                run_step,
                kernels,
                plan,
                runtime,
                inputs,
                slots,
                max_pages + 1,
                history=history,
            )

            for layer in range(args.layers):
                indices, output, _ = step(check=False)
                torch.npu.synchronize()
                if layer in (0, args.layers - 1):
                    emit("warmup", history=history, layer=layer, memory=memory())
            if args.mode == "sweep":
                continue
            if args.mode == "latency":
                durations = []
                for _ in range(args.repeat):
                    dist.barrier()
                    torch.npu.synchronize()
                    start = time.perf_counter()
                    indices, output, _ = step(check=False)
                    torch.npu.synchronize()
                    durations.append((time.perf_counter() - start) * 1000)
                emit("latency", history=history, ms=durations, memory=memory())
            else:
                with create_profiler(
                    args.output / f"profile-h{history}-rank{rank}"
                ) as prof:
                    indices, output, _ = step(check=False)
                    torch.npu.synchronize()
                    prof.step()
                    prof.step()
            indices, output, checks = step(check=True)
            actual = {"indices": indices.cpu(), "output": output.cpu()}
            assert torch.isfinite(actual["output"]).all()
            torch.save(actual, args.output / f"h{history}-rank{rank}.pt")
            if args.reference is not None:
                reference = torch.load(
                    args.reference / f"h{history}-rank{rank}.pt", weights_only=True
                )
                torch.testing.assert_close(
                    actual["output"], reference["output"], rtol=0, atol=0
                )
                assert torch.equal(actual["indices"], reference["indices"])
            emit("correctness", history=history, checks=checks, memory=memory())
    finally:
        runtime.close()
        log.close()
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
