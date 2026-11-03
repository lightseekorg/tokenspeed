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

import re
import socket
from datetime import timedelta

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from tokenspeed_kernel.platform import ArchVersion, current_platform
from utils import polling_cycles, votes_with_subgroup_mask

pytestmark = pytest.mark.skipif(
    not current_platform().is_amd, reason="Iris requires AMD ROCm"
)


@pytest.fixture(autouse=True)
def _require_iris():
    pytest.importorskip("tokenspeed_kernel.ops.communication._iris")


@pytest.mark.parametrize("rank", range(8))
def test_lamport_buffer_polling_codegen(rank, tmp_path):
    """Cache flags alone do not prevent a future compiler from hoisting loads.

    Compile without launching and require all seven 16-byte peer reads to
    remain on the retry backedge. This intentionally guards the tested machine
    behavior in addition to the distributed numerical tests.
    """
    from tokenspeed_kernel._triton import gluon, triton
    from tokenspeed_kernel.ops.communication._iris.all_reduce import (
        iris_arch_policy,
        iris_lamport_allreduce,
    )

    fn = iris_lamport_allreduce
    signature = {
        name: "*i32" if name == "epochs" else "*bf16" for name in fn.arg_names[:4]
    }
    signature.update({name: "i64" for name in fn.arg_names[4:12]})
    source = gluon._runtime.GluonASTSource(
        fn,
        signature,
        constexprs={
            "RANK": rank,
            "WORLD_SIZE": 8,
            "TOTAL_ELEMENTS": 6 * 10752,
            "MAX_ELEMENTS": 6 * 10752,
            "NUM_STAGES": 3,
            "PEER_SIGNAL_STORE_CACHE": iris_arch_policy(
                ArchVersion(9, 5)
            ).peer_signal_store,
        },
        attrs={(index,): [["tt.divisibility", 16]] for index in range(12)},
    )
    kernel = triton.compile(
        source,
        target=triton.backends.compiler.GPUTarget("hip", "gfx950", 64),
        options={"num_warps": 1},
    )
    assembly = kernel.asm["amdgcn"]
    (tmp_path / f"lamport-rank{rank}.amdgcn").write_text(assembly)
    (tmp_path / f"lamport-rank{rank}.llir").write_text(kernel.asm["llir"])
    lines = assembly.splitlines()
    cycles = polling_cycles(assembly)
    assert cycles, "polling loop was removed"
    polls = [
        cycle
        for cycle in cycles
        if sum("buffer_load_dwordx4" in line and "sc0 sc1" in line for line in cycle)
        == 7
    ]
    assert polls, "all seven system-cache reads must execute again on a retry"
    assert all(
        votes_with_subgroup_mask(cycle) for cycle in polls
    ), "a retry must branch on one subgroup lane mask, not a lane reduction"
    assert (
        sum("buffer_store_dwordx4" in line and "sc0 sc1" in line for line in lines)
        == 14
    ), "publication and reset must use 16-byte system-cache stores"
    assert kernel.metadata.num_warps == 1
    assert re.search(r"\.amdhsa_private_segment_fixed_size\s+0\b", assembly)


def test_lamport_two_shot_polling_codegen(tmp_path):
    """Both two-shot exchanges must reread all seven peer mailboxes per retry."""
    from tokenspeed_kernel._triton import gluon, triton
    from tokenspeed_kernel.ops.communication._iris.all_reduce import (
        iris_arch_policy,
        iris_lamport_twoshot_bf16_allreduce,
    )

    fn = iris_lamport_twoshot_bf16_allreduce
    signature = {
        "input_ptr": "*bf16",
        "region_sym_ptr": "*bf16",
        "output_ptr": "*bf16",
        "epochs": "*i32",
        "regions": ("i64",) * 8,
        "total_elements": "i32",
    }
    attrs = {}
    for index, name in enumerate(fn.arg_names):
        if name == "regions":
            attrs.update(
                {(index, peer): [["tt.divisibility", 16]] for peer in range(8)}
            )
        elif name in signature:
            attrs[(index,)] = [["tt.divisibility", 16]]
    source = gluon._runtime.GluonASTSource(
        fn,
        signature,
        constexprs={
            "RANK": 3,
            "WORLD_SIZE": 8,
            "MAX_SHARD_ELEMENTS": 131072,
            "NUM_STAGES": 3,
            "PEER_SIGNAL_STORE_CACHE": iris_arch_policy(
                ArchVersion(9, 5)
            ).peer_signal_store,
        },
        attrs=attrs,
    )
    kernel = triton.compile(
        source,
        target=triton.backends.compiler.GPUTarget("hip", "gfx950", 64),
        options={"num_warps": 1},
    )
    assembly = kernel.asm["amdgcn"]
    (tmp_path / "two-shot.amdgcn").write_text(assembly)
    # Group system-scope peer loads by the loop that LLVM reports for them.
    loops = {}
    header = None
    for line in assembly.splitlines():
        if match := re.match(r"^\.L(BB\w+):", line):
            loop = re.search(r"Header=(BB\w+)|(=>This Inner Loop Header)", line)
            header = (loop[1] or match[1]) if loop else None
        elif header:
            loops.setdefault(header, []).append(line)
    polls = [
        body
        for body in loops.values()
        if any("buffer_load_dword " in line and "sc0 sc1" in line for line in body)
    ]
    assert sorted(
        sum("buffer_load_dword " in line and "sc0 sc1" in line for line in body)
        for body in polls
    ) == [7, 7], "each exchange must poll all seven peers per retry"
    assert all(
        votes_with_subgroup_mask(body) for body in polls
    ), "each exchange must branch on one subgroup lane mask per retry"
    # Readiness watches only the odd half of each BF16 pair, so every mailbox
    # push and reset must store the whole pair in one dword.
    stores = re.findall(r"^\s*(buffer_store_\w+)[^\n]*\bsc0 sc1\b", assembly, re.M)
    assert stores and set(stores) == {"buffer_store_dword"}, sorted(set(stores))
    assert re.search(r"\.amdhsa_private_segment_fixed_size\s+0\b", assembly)


def test_lamport_unspecialized_mailbox_polling_codegen():
    """Readiness polls must read whole 32-bit words at any mailbox alignment.

    Compile the one-shot without the mailbox pointer's alignment
    specialization, as an ahead-of-time or do_not_specialize caller would.
    BF16 element reads may then split a word into two 16-bit loads, pairing a
    stale low half with an arrived high half.
    """
    from tokenspeed_kernel._triton import gluon, triton
    from tokenspeed_kernel.ops.communication._iris.all_reduce import (
        iris_arch_policy,
        iris_lamport_bf16_allreduce,
    )

    fn = iris_lamport_bf16_allreduce
    signature = {
        "input_sym_ptr": "*bf16",
        "region_sym_ptr": "*bf16",
        "output_ptr": "*bf16",
        "epochs": "*i32",
        "regions": ("i64",) * 8,
        "total_elements": "i32",
    }
    attrs = {}
    for index, name in enumerate(fn.arg_names):
        if name == "regions":
            attrs.update(
                {(index, peer): [["tt.divisibility", 16]] for peer in range(8)}
            )
        elif name in signature and name != "region_sym_ptr":
            attrs[(index,)] = [["tt.divisibility", 16]]
    source = gluon._runtime.GluonASTSource(
        fn,
        signature,
        constexprs={
            "RANK": 3,
            "WORLD_SIZE": 8,
            "MAX_ELEMENTS": 43 * 4096,
            "NUM_STAGES": 3,
            "PEER_SIGNAL_STORE_CACHE": iris_arch_policy(
                ArchVersion(9, 5)
            ).peer_signal_store,
        },
        attrs=attrs,
    )
    kernel = triton.compile(
        source,
        target=triton.backends.compiler.GPUTarget("hip", "gfx950", 64),
        options={"num_warps": 1},
    )
    assembly = kernel.asm["amdgcn"]
    polls = [
        cycle
        for cycle in polling_cycles(assembly)
        if sum("buffer_load_dwordx4" in line and "sc0 sc1" in line for line in cycle)
        == 7
    ]
    assert polls, "all seven peers must be polled as 16-byte word groups"
    # Any sub-dword form (short, sshort, byte, the _d16 halves) splits a word.
    assert not re.search(
        r"\bbuffer_load_\w*(?:short|byte)\w*\b[^\n]*\bsc0 sc1\b", assembly
    ), "a system-scope mailbox read must not split a 32-bit word"


def _new_state(rank, device, capacity, dtype):
    from tokenspeed_kernel.ops.communication._iris.kimi_k3 import (
        prepare_k3_communication,
    )

    return prepare_k3_communication(
        enable_lamport=True,
        sharded_max_rows=0,
        group=dist.group.WORLD,
        rank_in_group=rank,
        staged_max_numel=0,
        producer_direct_max_numel=capacity,
        attnres_max_numel=0,
        attnres_max_rows=0,
        dtype=dtype,
        all_reduce=None,
        device=device,
    )


def _check_lamport_state(rank, device):

    state = _new_state(rank, device, 8 * 10752, torch.bfloat16)
    region = state.lamport.mailbox
    epochs = state.lamport.epochs
    assert region.shape == (3, 8, 6 * 10752)
    assert epochs.shape == (126,)
    torch.testing.assert_close(
        region.view(torch.int32),
        torch.full_like(region.view(torch.int32), -2147483648),
        atol=0,
        rtol=0,
    )
    # Peers may push into this mailbox as soon as they pass their own check.
    dist.barrier()
    expected_epochs = torch.zeros_like(epochs)
    for rows in range(1, 9):
        widths_cases = ((3584, 7168), (7168, 3584))
        if rows in (1, 6):
            widths_cases += ((5376, 5376),)
        for widths in widths_cases:
            shapes = tuple((rows, width) for width in widths)
            use_lamport = rows <= 6 and widths != (5376, 5376)
            inputs = state.all_reduce.acquire_outputs(shapes)
            # Distinct element/rank values, cancellation, and sentinel-valued
            # inputs. Reference uses the kernel's ascending-rank FP32 order.
            generator = torch.Generator().manual_seed(100 + rows)
            peers = torch.randn(8, rows * 10752, generator=generator).bfloat16()
            peers[:, ::31] = -0.0
            peers[1::2, 1::31] = -peers[::2, 1::31]
            peers[:, 2::31] = float("inf")
            peers[:, 3::31] = float("nan")
            expected = peers[0].float()
            for peer in range(1, 8):
                expected += peers[peer].float()
            offset = 0
            for tensor in inputs:
                tensor.copy_(
                    peers[rank, offset : offset + tensor.numel()].view(tensor.shape)
                )
                offset += tensor.numel()
            actual = state.reduce_moe(inputs)
            actual = torch.cat([tensor.flatten() for tensor in actual])
            torch.testing.assert_close(
                actual.cpu(), expected.bfloat16(), atol=0, rtol=0, equal_nan=True
            )
            if use_lamport:
                expected_epochs[: rows * 21] = (expected_epochs[: rows * 21] + 1) % 3
            torch.testing.assert_close(epochs, expected_epochs, atol=0, rtol=0)

    # Capture every row count, then alternate Lamport and pull without host
    # synchronization between replays. Skipped tiles retain independent epochs.
    captures = []
    for rows in (6, 1, 7, 3, 8, 2, 5, 4):
        inputs = state.all_reduce.acquire_outputs(((rows, 7168), (rows, 3584)))
        sources = tuple(torch.full_like(tensor, rank + 1) for tensor in inputs)
        graph = torch.cuda.CUDAGraph()
        torch.cuda.synchronize()
        dist.barrier()
        with torch.cuda.graph(graph):
            for tensor, source in zip(inputs, sources, strict=True):
                tensor.copy_(source)
            outputs = state.reduce_moe(inputs)
        captures.append((rows, graph, sources, outputs))
    snapshots = []
    before_epochs = epochs.clone()
    pointers = (
        region.data_ptr(),
        epochs.data_ptr(),
        state.all_reduce.producer.data.data_ptr(),
    )
    for iteration in range(12):
        for rows, graph, sources, outputs in captures:
            scale = iteration + 1
            for index, source in enumerate(sources, start=1):
                source.fill_(scale * index * (rank + 1))
            if rank == iteration % 8:
                torch.cuda._sleep(100_000)
            graph.replay()
            snapshots.append((scale, tuple(output.clone() for output in outputs)))
    torch.cuda.synchronize()
    assert pointers == (
        region.data_ptr(),
        epochs.data_ptr(),
        state.all_reduce.producer.data.data_ptr(),
    )
    torch.testing.assert_close(epochs, before_epochs, atol=0, rtol=0)
    for scale, outputs in snapshots:
        for index, output in enumerate(outputs, start=1):
            torch.testing.assert_close(
                output, torch.full_like(output, 36 * scale * index), atol=0, rtol=0
            )

    for capacity, expected_capacity in (
        (0, 0),
        (10751, 0),
        (10752, 10752),
        (32257, 32256),
    ):
        smaller = _new_state(rank, device, capacity, torch.bfloat16)
        assert (
            smaller.lamport.max_numel if smaller.lamport else 0
        ) == expected_capacity
        if expected_capacity:
            inputs = smaller.all_reduce.acquire_outputs(((1, 3584), (1, 7168)))
            for tensor in inputs:
                tensor.fill_(rank + 1)
            for output in smaller.reduce_moe(inputs):
                torch.testing.assert_close(
                    output, torch.full_like(output, 36), atol=0, rtol=0
                )
        else:
            assert smaller.lamport is None
    for dtype in (torch.float16, torch.float32):
        other = _new_state(rank, device, 10752, dtype)
        assert other.lamport is None
        inputs = other.all_reduce.acquire_outputs(((1, 3584), (1, 7168)))
        for tensor in inputs:
            tensor.fill_(rank + 1)
        for output in other.reduce_moe(inputs):
            torch.testing.assert_close(
                output, torch.full_like(output, 36), atol=0, rtol=0
            )


def _lamport_worker(rank, port):
    torch.cuda.set_device(rank)
    dist.init_process_group(
        backend="gloo",
        init_method=f"tcp://localhost:{port}",
        rank=rank,
        world_size=8,
        timeout=timedelta(seconds=180),
    )
    try:
        _check_lamport_state(rank, torch.device(f"cuda:{rank}"))
    finally:
        dist.destroy_process_group()


def test_lamport_world8():
    if not current_platform().is_cdna4 or torch.cuda.device_count() < 8:
        pytest.skip("requires eight CDNA4 GPUs")
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("", 0))
        port = sock.getsockname()[1]
    mp.spawn(_lamport_worker, args=(port,), nprocs=8, join=True)
