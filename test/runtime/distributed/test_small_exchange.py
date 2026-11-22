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

"""Prepared one-hop exchanges of small payloads (``prepare_small_exchange``).

``all_gather_single``, ``all_gather`` and equal-split ``all_to_all_single``
calls that fit the prepared capacity run the backend's LL exchange, a rank's
unaligned or strided operands staged locally; larger and uneven calls keep
NCCL. Both must move every bit unchanged, eager and under CUDA graph replay
with new inputs; the NCCL-only switch keeps NCCL.
"""

import gc
import sys
from datetime import timedelta
from pathlib import Path

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from ci_system.ci_register import register_cuda_ci  # noqa: E402

register_cuda_ci(
    est_time=120,
    suite="runtime-2gpu",
    disabled_on_runners=["amd-*"],
    disabled_on_runners_reason="the one-hop exchange is an NVIDIA NVLink path",
)

_CAP = 64 << 10
_TIMEOUT_NS = 60_000_000_000


def _bits(count: int, seed: int, device) -> torch.Tensor:
    generator = torch.Generator(device="cpu").manual_seed(seed)
    return torch.randint(-(2**31), 2**31 - 1, (count,), generator=generator, dtype=torch.int32).to(device)  # fmt: skip


def _worker(rank: int, degree: int, rendezvous: str) -> None:
    torch.cuda.set_device(rank)
    from tokenspeed_kernel.ops.communication.fabric import gather_fabric_map

    from tokenspeed.runtime.distributed.comm_backend.auto import AutoBackend
    from tokenspeed.runtime.distributed.comm_backend.registry import set_global_backend
    from tokenspeed.runtime.distributed.comm_ops import (
        all_gather,
        all_gather_float_words_single,
        all_gather_single,
        all_to_all_single,
        prepare_small_exchange,
    )
    from tokenspeed.runtime.distributed.process_group_manager import (
        process_group_manager,
    )
    from tokenspeed.runtime.utils.env import global_server_args_dict

    device = torch.device("cuda", rank)
    dist.init_process_group(
        "nccl",
        init_method=rendezvous,
        rank=rank,
        world_size=degree,
        device_id=device,
        timeout=timedelta(seconds=90),
    )
    try:
        gather_fabric_map()
        group = tuple(range(degree))
        process_group_manager.register_process_group("nccl", group, dist.group.WORLD)
        process_group_manager.register_process_group(
            "gloo", group, dist.new_group(backend="gloo")
        )
        backend = AutoBackend()
        set_global_backend(backend)
        nccl = backend.nccl
        global_server_args_dict["force_deterministic_rsag"] = True
        assert not prepare_small_exchange(group, _CAP, _CAP, _TIMEOUT_NS)
        global_server_args_dict["force_deterministic_rsag"] = False
        assert prepare_small_exchange(group, _CAP, _CAP, _TIMEOUT_NS)
        # A second caller of no larger capacity shares it.
        assert prepare_small_exchange(group, _CAP // 2, _CAP, _TIMEOUT_NS)
        state = backend._small_exchanges[group]

        def calls() -> int:
            """The exchange's device call count: which calls it ran."""
            torch.cuda.synchronize()
            return int(state.ctrl[0])

        def gather(words: int, seed: int, offset: int = 0, *, ll: bool) -> None:
            payload = _bits(words + offset, seed + rank, device)[offset:]
            out = torch.empty(degree * words, dtype=torch.int32, device=device)
            expected = torch.empty_like(out)
            before = calls()
            all_gather_single(out, payload, group)
            assert calls() == before + ll
            nccl.all_gather_single(expected, payload, group)
            assert torch.equal(out, expected)

        def exchange(words: int, seed: int, *, ll: bool) -> None:
            payload = _bits(degree * words, seed + rank, device)
            out = torch.empty_like(payload)
            expected = torch.empty_like(payload)
            before = calls()
            all_to_all_single(out, payload, group)
            assert calls() == before + ll
            nccl.all_to_all_single(expected, payload, group)
            assert torch.equal(out, expected)

        fits = torch.empty(4096, dtype=torch.int32, device=device)
        # Calls that fit take the exchange ...
        assert backend._small_exchange(
            torch.empty(degree * 4096, dtype=torch.int32, device=device), fits, group, True  # fmt: skip
        )
        for words, seed in ((4, 10), (4096, 20), (_CAP // 4, 30)):
            gather(words, seed, ll=True)
            exchange(words, seed + 1, ll=True)
        # ... only sizes decide, so an unaligned operand takes it too ...
        assert backend._small_exchange(
            torch.empty(degree * 8, dtype=torch.int32, device=device),
            torch.empty(9, dtype=torch.int32, device=device)[1:],
            group,
            True,
        )
        # ... oversized calls and payloads of partial units keep NCCL.
        assert not backend._small_exchange(
            torch.empty(degree * (_CAP // 4 + 4), dtype=torch.int32, device=device),
            torch.empty(_CAP // 4 + 4, dtype=torch.int32, device=device),
            group,
            True,
        )
        assert not backend._small_exchange(
            torch.empty(degree * 6, dtype=torch.int32, device=device),
            torch.empty(6, dtype=torch.int32, device=device),
            group,
            True,
        )
        gather(_CAP // 4 + 4, 40, ll=False)
        gather(6, 50, ll=False)
        gather(8, 60, offset=1, ll=True)
        exchange(_CAP // 4 + 4, 70, ll=False)
        # Unaligned and strided outputs are staged and land in place.
        payload = _bits(degree * 64, 72 + rank, device)
        expected = torch.empty_like(payload)
        nccl.all_to_all_single(expected, payload, group)
        before = calls()
        unaligned = torch.zeros(degree * 64 + 1, dtype=torch.int32, device=device)
        all_to_all_single(unaligned[1:], payload, group)
        assert torch.equal(unaligned[1:], expected) and int(unaligned[0]) == 0
        strided = torch.zeros(degree * 64, 2, dtype=torch.int32, device=device)
        assert not strided[:, 0].is_contiguous()
        all_to_all_single(strided[:, 0], payload, group)
        assert torch.equal(strided[:, 0], expected) and not strided[:, 1].any()
        assert calls() == before + 2
        # A gather along an inner dim lays the rank blocks out as NCCL's does.
        rows = _bits(3 * 256, 75 + rank, device).view(3, 256)
        for dim in (0, -1):
            before = calls()
            gathered = all_gather(rows, group, dim=dim)
            assert calls() == before + 1
            assert torch.equal(gathered, nccl.all_gather(rows, group, dim))

        # Float words take the float exchange, the sentinel word coming back
        # as another NaN; past its capacity they take the generic gather.
        def float_calls() -> int:
            torch.cuda.synchronize()
            return int(state.float_ctrl[0])

        for words, seed in ((4, 110), (_CAP // 4, 120), (_CAP // 4 + 4, 130)):
            payload = _bits(words, seed + rank, device)
            payload[::7] = -1
            out = torch.empty(degree * words, dtype=torch.int32, device=device)
            before, before_ll = float_calls(), calls()
            all_gather_float_words_single(out, payload, group)
            fits = words * 4 <= _CAP
            assert float_calls() == before + fits and calls() == before_ll
            expected = torch.empty_like(out)
            nccl.all_gather_single(expected, payload, group)
            if fits:
                expected[expected == -1] = 0x7FFFFFFF
            assert torch.equal(out, expected)

        # Graph replays of the exchange with new inputs.
        payload = _bits(1024, 80 + rank, device)
        out = torch.empty(degree * 1024, dtype=torch.int32, device=device)
        all_gather_single(out, payload, group)
        torch.cuda.synchronize()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            all_gather_single(out, payload, group)
        for seed in (90, 100):
            payload.copy_(_bits(1024, seed + rank, device))
            torch.cuda.synchronize()
            dist.barrier()
            graph.replay()
            torch.cuda.synchronize()
            expected = torch.empty_like(out)
            nccl.all_gather_single(expected, payload, group)
            assert torch.equal(out, expected)
        dist.barrier()
        graph.reset()
        set_global_backend(nccl)
        del backend
        gc.collect()
        dist.barrier()
    finally:
        dist.destroy_process_group()


@pytest.mark.parametrize("degree", [2, 4])
def test_prepared_small_exchange_moves_every_bit(degree, tmp_path):
    if torch.cuda.device_count() < degree:
        pytest.skip(f"needs {degree} GPUs")
    if torch.cuda.get_device_capability() not in ((10, 0), (10, 3)):
        pytest.skip("the one-hop exchange needs SM100 or SM103")
    rendezvous = f"file://{tmp_path / 'rendezvous'}"
    mp.spawn(_worker, args=(degree, rendezvous), nprocs=degree, join=True)
