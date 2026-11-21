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

"""Muse Spark's owned-row projections on real process groups.

Workers record rank/PID/phase logs and Python stack snapshots every five minutes.
Failures retain these under .ci-artifacts for the existing upload. The suite
timeout is unchanged.

Usage:
    python -m pytest test/runtime/distributed/test_muse_owned_projections.py -v
"""

import gc
import shutil
import socket
import sys
import tempfile
from collections.abc import Callable
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from ci_system.ci_register import register_cuda_ci  # noqa: E402
from ci_system.worker_diagnostics import worker_diagnostics  # noqa: E402

register_cuda_ci(
    est_time=120,
    suite="runtime-2gpu",
    disabled_on_runners=["amd-*"],
    disabled_on_runners_reason="Muse Spark projects owned rows this way on NVIDIA only",
)

_HIDDEN, _MAX_ROWS = 256, 8192


def _integers(generator, shape, low, high, device):
    # Small integers multiply and sum exactly in FP32, so every GEMM order agrees.
    values = torch.randint(low, high + 1, shape, generator=generator)
    return values.to(device=device, dtype=torch.bfloat16)


def _all_gather(tensor):
    parts = [torch.empty_like(tensor) for _ in range(dist.get_world_size())]
    dist.all_gather(parts, tensor)
    return parts


def _project(x, weight):
    return (x.float() @ weight.float().T).to(torch.bfloat16)


def _run_worker(
    rank: int, port: int, world_size: int, phase: Callable[[str], None]
) -> None:
    phase("imports")
    from tokenspeed.runtime.distributed.comm_backend.auto import AutoBackend
    from tokenspeed.runtime.distributed.comm_backend.registry import set_global_backend
    from tokenspeed.runtime.distributed.process_group_manager import (
        process_group_manager,
    )
    from tokenspeed.runtime.models.muse_spark import (
        _OwnedRowProjections,
        _prepare_owned_projections,
    )

    phase("default_nccl_group")
    torch.cuda.set_device(rank)
    dist.init_process_group(
        backend="nccl",
        init_method=f"tcp://127.0.0.1:{port}",
        rank=rank,
        world_size=world_size,
    )
    try:
        phase("tokenspeed_groups_and_backend")
        group = tuple(range(world_size))
        process_group_manager.init_process_group(group)
        set_global_backend(AutoBackend())
        device = torch.device("cuda", rank)
        experts = 32 * world_size
        shapes = {
            "qkv": (96, _HIDDEN),
            "output": (_HIDDEN, 64),
            "pre_expert": (32, _HIDDEN),
            "post_expert": (_HIDDEN // world_size, experts),
        }

        def weights(seed):
            generator = torch.Generator().manual_seed(seed + rank)
            return {
                name: _integers(generator, shape, -1, 1, device)
                for name, shape in shapes.items()
            }

        # Only the weights' shapes and devices reach the preparation.
        sample = weights(0)
        layer = SimpleNamespace(
            self_attn=SimpleNamespace(
                qkv_proj=SimpleNamespace(weight=sample["qkv"]),
                o_proj=SimpleNamespace(weight=sample["output"]),
            ),
            mlp=SimpleNamespace(
                pre_expert_proj=SimpleNamespace(weight=sample["pre_expert"]),
                post_expert_proj=SimpleNamespace(weight=sample["post_expert"]),
            ),
        )
        owned = _OwnedRowProjections(group)
        phase("prepare_owned_projections")
        _prepare_owned_projections(owned, layer, group, owned.local_rows(_MAX_ROWS))

        def operands(rows, seed):
            local_rows = owned.local_rows(rows)
            generator = torch.Generator().manual_seed(1000 + seed + rank)
            # Every rank holds the complete expert rows, as after the EP all-reduce.
            shared = torch.Generator().manual_seed(2000 + seed)
            return {
                "owned": _integers(generator, (local_rows, _HIDDEN), -2, 2, device),
                "attended": _integers(generator, (rows, 64), -1, 1, device),
                "combined": _integers(shared, (rows, experts), -1, 1, device),
                **weights(3000 + seed),
            }

        def layer_projections(rows, values):
            return (
                owned.gather_rows(values["owned"], rows),
                owned.gather_mm(values["owned"], rows, values["qkv"], owned.qkv),
                owned.matmul_reduce_scatter(values["attended"], rows, values["output"]),
                owned.gather_mm(
                    values["owned"], rows, values["pre_expert"], owned.pre_expert
                ),
                owned.matmul_to_owners(values["combined"], rows, values["post_expert"]),
            )

        def expected(rows, values):
            local_rows = owned.local_rows(rows)
            padding = (0, 0, 0, local_rows * world_size - rows)
            mine = slice(rank * local_rows, (rank + 1) * local_rows)
            gathered = torch.cat(_all_gather(values["owned"]))[:rows]
            reduced = sum(
                _project(attended, weight)
                for attended, weight in zip(
                    _all_gather(values["attended"]), _all_gather(values["output"])
                )
            ).to(torch.bfloat16)
            columns = torch.cat(_all_gather(values["post_expert"]))
            return (
                gathered,
                _project(gathered, values["qkv"]),
                torch.nn.functional.pad(reduced, padding)[mine],
                _project(gathered, values["pre_expert"]),
                torch.nn.functional.pad(_project(values["combined"], columns), padding)[
                    mine
                ],
            )

        # Odd row counts leave the last owner's padding out.
        for seed, rows in enumerate((4097, 8192, 4099)):
            phase(f"eager_projections_and_reference rows={rows}")
            values = operands(rows, seed)
            for got, want in zip(
                layer_projections(rows, values), expected(rows, values)
            ):
                torch.testing.assert_close(got, want, atol=0, rtol=0)

        # Two captured layers replay on new rows.
        rows = 4097
        first, second = operands(rows, 10), operands(rows, 11)
        phase("pre_capture_projections")
        layer_projections(rows, first)
        phase("pre_capture_synchronize")
        torch.cuda.synchronize()
        graph = torch.cuda.CUDAGraph()
        # Destructors of discarded CUDA objects can invalidate an active capture.
        gc.collect()
        gc.disable()
        try:
            phase("capture_two_layers")
            with torch.cuda.graph(graph):
                layer_projections(rows, first)
                outputs = layer_projections(rows, second)
        finally:
            gc.enable()
        for values, seed in ((first, 12), (second, 13)):
            for name, update in operands(rows, seed).items():
                values[name].copy_(update)
        phase("graph_replay")
        graph.replay()
        phase("replay_reference_and_comparison")
        for got, want in zip(outputs, expected(rows, second)):
            torch.testing.assert_close(got, want, atol=0, rtol=0)
        # A live graph holding NCCL work blocks the process group's teardown.
        phase("graph_reset")
        graph.reset()
        phase("final_barrier")
        dist.barrier()
    finally:
        phase("destroy_process_group")
        dist.destroy_process_group()


def _worker(rank: int, port: int, world_size: int, diagnostics: str) -> None:
    with worker_diagnostics(Path(diagnostics), rank, dump_after_seconds=300) as phase:
        _run_worker(rank, port, world_size, phase)
        phase("complete")


def _spawn(world_size: int) -> None:
    if torch.cuda.device_count() < world_size:
        pytest.skip(f"requires {world_size} GPUs")
    with socket.socket() as listener:
        listener.bind(("127.0.0.1", 0))
        port = listener.getsockname()[1]
    # The existing CI failure upload retains this directory even if the suite's
    # unchanged 1800-second file timeout kills this process and its workers.
    artifacts = Path(__file__).resolve().parents[3] / ".ci-artifacts"
    artifacts.mkdir(exist_ok=True)
    diagnostics = tempfile.mkdtemp(
        prefix=f"muse-owned-projections-tp{world_size}-", dir=artifacts
    )
    print(f"Worker phase/stack diagnostics: {diagnostics}", flush=True)
    mp.spawn(
        _worker, args=(port, world_size, diagnostics), nprocs=world_size, join=True
    )
    shutil.rmtree(diagnostics)


@pytest.mark.parametrize("world_size", [2, 4])
def test_owned_projections_match_gathered_rows(world_size) -> None:
    _spawn(world_size)
