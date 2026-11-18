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

"""Check the one-hop LL all-gather and all-to-all against NCCL.

On node-local SM100 or SM103 GPUs: every byte value at sizes from one unit to
the capacity, many calls through both parity slots, graph replays with fresh
inputs, the refusal vote and argument errors.
"""

from datetime import timedelta

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from tokenspeed_kernel.ops.communication.cute_dsl import (
    create_ll_exchange_state,
    cute_dsl_ll_all_gather,
    cute_dsl_ll_all_to_all,
)
from utils import kernel_supported

CAP_BYTES = 1 << 20
TIMEOUT_NS = 60_000_000_000
SIZES = (16, 48, 4096, 65536, 393232, CAP_BYTES)


def _init(rank, size, rendezvous):
    from tokenspeed_kernel.ops.communication.fabric import gather_fabric_map

    torch.cuda.set_device(rank)
    device = torch.device("cuda", rank)
    dist.init_process_group(
        "nccl",
        init_method=rendezvous,
        rank=rank,
        world_size=size,
        timeout=timedelta(seconds=300),
        device_id=device,
    )
    gather_fabric_map()
    return device, dist.new_group(backend="gloo")


def _bytes(count, generator, device):
    return torch.randint(0, 256, (count,), dtype=torch.uint8, device=device, generator=generator)  # fmt: skip


def _exchange_matches_nccl(rank, size, rendezvous):
    device, group = _init(rank, size, rendezvous)
    state = create_ll_exchange_state(
        group, device, cap_bytes=CAP_BYTES, timeout_ns=TIMEOUT_NS, eligible=True
    )
    assert state is not None
    generator = torch.Generator(device=device).manual_seed(1000 + rank)
    for payload in SIZES:
        for _ in range(3):
            gather_in = _bytes(payload, generator, device)
            a2a_in = _bytes(size * payload, generator, device)
            expected_gather = torch.empty(size * payload, dtype=torch.uint8, device=device)  # fmt: skip
            expected_a2a = torch.empty_like(expected_gather)
            dist.all_gather_into_tensor(expected_gather, gather_in)
            dist.all_to_all_single(expected_a2a, a2a_in)
            got_gather = cute_dsl_ll_all_gather(
                state, gather_in, torch.empty_like(expected_gather)
            )
            got_a2a = cute_dsl_ll_all_to_all(
                state, a2a_in, torch.empty_like(expected_a2a)
            )
            assert torch.equal(got_gather, expected_gather), payload
            assert torch.equal(got_a2a, expected_a2a), payload
    # Other dtypes travel as their bytes.
    words = torch.randn(size * 1024, device=device).view(torch.int32)
    got = cute_dsl_ll_all_to_all(state, words, torch.empty_like(words))
    expected = torch.empty_like(words)
    dist.all_to_all_single(expected, words)
    assert torch.equal(got, expected)
    dist.destroy_process_group()


def _graph_replays_fresh_inputs(rank, size, rendezvous):
    device, group = _init(rank, size, rendezvous)
    state = create_ll_exchange_state(
        group, device, cap_bytes=CAP_BYTES, timeout_ns=TIMEOUT_NS, eligible=True
    )
    generator = torch.Generator(device=device).manual_seed(2000 + rank)
    payloads = (4096, 65536, 16, 262144)
    gather_in = [_bytes(p, generator, device) for p in payloads]
    a2a_in = [_bytes(size * p, generator, device) for p in payloads]
    gather_out = [torch.empty(size * p, dtype=torch.uint8, device=device) for p in payloads]  # fmt: skip
    a2a_out = [torch.empty(size * p, dtype=torch.uint8, device=device) for p in payloads]  # fmt: skip
    graph = torch.cuda.CUDAGraph()
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream), torch.cuda.graph(graph, stream=stream):
        for i in range(len(payloads)):
            cute_dsl_ll_all_gather(state, gather_in[i], gather_out[i])
            cute_dsl_ll_all_to_all(state, a2a_in[i], a2a_out[i])
    for _ in range(3):
        for tensor in gather_in + a2a_in:
            tensor.random_(0, 256, generator=generator)
        torch.cuda.synchronize()
        dist.barrier()
        graph.replay()
        torch.cuda.synchronize()
        for i in range(len(payloads)):
            expected = torch.empty_like(gather_out[i])
            dist.all_gather_into_tensor(expected, gather_in[i])
            assert torch.equal(gather_out[i], expected)
            dist.all_to_all_single(expected, a2a_in[i])
            assert torch.equal(a2a_out[i], expected)
    dist.destroy_process_group()


def _refusals(rank, size, rendezvous):
    device, group = _init(rank, size, rendezvous)
    # One ineligible rank keeps every rank on its collectives.
    refused = create_ll_exchange_state(
        group, device, cap_bytes=CAP_BYTES, timeout_ns=TIMEOUT_NS, eligible=rank != 1
    )
    assert refused is None
    unaligned = create_ll_exchange_state(
        group, device, cap_bytes=CAP_BYTES + 8, timeout_ns=TIMEOUT_NS, eligible=True
    )
    assert unaligned is None
    state = create_ll_exchange_state(
        group, device, cap_bytes=4096, timeout_ns=TIMEOUT_NS, eligible=True
    )
    assert state.serves(4096) and not state.serves(4112) and not state.serves(24)
    payload = torch.zeros(4096, dtype=torch.uint8, device=device)
    out = torch.empty(size * 4096, dtype=torch.uint8, device=device)
    for bad_in, bad_out in (
        (torch.zeros(4112, dtype=torch.uint8, device=device), out),
        (payload, out[:-16]),
        (torch.zeros(4097, dtype=torch.uint8, device=device)[1:], out),
    ):
        with pytest.raises(ValueError, match="does not serve"):
            cute_dsl_ll_all_gather(state, bad_in, bad_out)
    with pytest.raises(ValueError, match="does not serve"):
        cute_dsl_ll_all_to_all(state, torch.zeros(size * 4096 + 16, dtype=torch.uint8, device=device), out)  # fmt: skip
    dist.destroy_process_group()


def _spawn(target, tmp_path):
    size = min(torch.cuda.device_count(), 8)
    size = 8 if size >= 8 else 4 if size >= 4 else 2
    mp.spawn(target, args=(size, f"file://{tmp_path}/rendezvous"), nprocs=size, join=True)  # fmt: skip


pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available()
    or torch.cuda.device_count() < 2
    or torch.cuda.get_device_capability() not in ((10, 0), (10, 3))
    or not kernel_supported("cute_dsl_ll_all_gather"),
    reason="requires two SM100 or SM103 GPUs",
)


def test_exchange_matches_nccl(tmp_path):
    _spawn(_exchange_matches_nccl, tmp_path)


def test_graph_replays_fresh_inputs(tmp_path):
    _spawn(_graph_replays_fresh_inputs, tmp_path)


def test_refusals_and_argument_errors(tmp_path):
    _spawn(_refusals, tmp_path)
