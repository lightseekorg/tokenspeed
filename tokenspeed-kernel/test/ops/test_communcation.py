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

import socket
import time
import traceback
from types import SimpleNamespace
from typing import List

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from tokenspeed_kernel.ops.communication import triton as triton_communication
from tokenspeed_kernel.ops.communication.triton import (
    all_gather,
    all_reduce,
    all_reduce_can_run,
    allreduce_residual_rmsnorm,
    create_state,
    reduce_scatter,
)
from tokenspeed_kernel.platform import current_platform


def get_open_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("", 0))
        return sock.getsockname()[1]


def test_alloc_symm_escapes_inference_mode(monkeypatch):
    process_group = object()
    handle = object()

    def fake_empty(shape, *, dtype, device):
        assert not torch.is_inference_mode_enabled()
        assert not torch.is_grad_enabled()
        return torch.empty(shape, dtype=dtype, device=device)

    def fake_rendezvous(tensor, *, group):
        assert tensor.shape == (2, 3)
        assert group is process_group
        return handle

    monkeypatch.setattr(triton_communication.symm_mem, "empty", fake_empty)
    monkeypatch.setattr(triton_communication.symm_mem, "rendezvous", fake_rendezvous)

    with torch.inference_mode():
        tensor, result_handle = triton_communication._alloc_symm(
            (2, 3), torch.float32, torch.device("cpu"), process_group
        )

    assert not tensor.is_inference()
    assert result_handle is handle
    tensor.copy_(torch.ones_like(tensor))


def token_cases(world_size: int) -> List[List[int]]:
    cases = [
        [8] * world_size,
        [8 + rank for rank in range(world_size)],
    ]
    if world_size >= 4:
        cases.append([1, 20, 3] + [0] * (world_size - 3))
    else:
        cases.append([3] + [0] * (world_size - 1))
    return cases


def worker_fn(rank, world_size, port, hidden_size, error_dict):
    try:
        worker_main(rank, world_size, port, hidden_size)
    except Exception:
        error_dict[rank] = traceback.format_exc()


def worker_main(rank: int, world_size: int, port: int, hidden_size: int) -> None:
    device = torch.device(f"cuda:{rank}")
    torch.cuda.set_device(device)
    dist.init_process_group(
        backend="nccl",
        init_method=f"tcp://localhost:{port}",
        rank=rank,
        world_size=world_size,
    )

    try:
        cases = token_cases(world_size)
        max_tokens = max(sum(tokens) for tokens in cases)
        rsag = create_state(
            group=dist.group.WORLD,
            rank_in_group=rank,
            max_tokens=max_tokens,
            hidden_size=hidden_size,
        )

        for tokens in cases:
            check_all_gather(rsag, rank, world_size, tokens, hidden_size, device)
            check_reduce_scatter(rsag, rank, world_size, tokens, hidden_size, device)

        if current_platform().is_amd:
            check_amd_rsag_reuses_single_barrier_signal(
                rsag,
                rank,
                world_size,
                hidden_size,
                device,
            )
            check_all_reduce(rank, world_size, device)
            check_allreduce_residual_rmsnorm(rank, world_size, device)
    finally:
        dist.destroy_process_group()


def check_all_gather(
    rsag, rank: int, world_size: int, tokens: List[int], hidden_size: int, device
) -> None:
    local_tokens = tokens[rank]
    local = torch.full(
        (local_tokens, hidden_size),
        rank + 1,
        dtype=torch.bfloat16,
        device=device,
    )

    result = all_gather(rsag, local, token_list_in_group=tokens)

    expected = torch.empty(
        (sum(tokens), hidden_size), dtype=torch.bfloat16, device=device
    )
    offset = 0
    for peer, peer_tokens in enumerate(tokens):
        expected[offset : offset + peer_tokens].fill_(peer + 1)
        offset += peer_tokens

    assert result.shape == expected.shape
    torch.testing.assert_close(result, expected, atol=0, rtol=0)


def check_all_reduce(rank: int, world_size: int, device) -> None:
    max_numel = 512 * 1024 // torch.empty((), dtype=torch.bfloat16).element_size()
    state = create_state(
        group=dist.group.WORLD,
        rank_in_group=rank,
        max_numel=max_numel,
        device=device,
    )

    for numel in [2880, 20160, 23040, 92160, 184320]:
        tensor = torch.full((numel,), rank + 1, dtype=torch.bfloat16, device=device)
        assert all_reduce_can_run(state, tensor)
        result = all_reduce(state, tensor)
        assert result is tensor
        expected = torch.full_like(result, world_size * (world_size + 1) // 2)
        torch.testing.assert_close(result, expected, atol=0, rtol=0)
        torch.testing.assert_close(tensor, expected, atol=0, rtol=0)

    large = torch.full((300000,), rank + 1, dtype=torch.bfloat16, device=device)
    assert not all_reduce_can_run(state, large)


def check_allreduce_residual_rmsnorm(rank: int, world_size: int, device) -> None:
    hidden = 2880
    eps = 1e-6
    weight = torch.linspace(0.5, 1.5, hidden, dtype=torch.float32, device=device)

    for tokens in [1, 8, 32]:
        x = torch.full((tokens, hidden), rank + 1, dtype=torch.bfloat16, device=device)
        residual = (
            torch.arange(tokens * hidden, dtype=torch.float32, device=device)
            .reshape(tokens, hidden)
            .mul_(0.001)
            .to(torch.bfloat16)
        )

        norm_out, residual_out, scale, partial = allreduce_residual_rmsnorm(
            input_tensor=x,
            residual=residual,
            weight=weight,
            rank=rank,
            group=dist.group.WORLD,
            eps=eps,
            max_token_num=64,
        )
        assert scale is None
        assert partial is None

        reduced = torch.full_like(residual.float(), world_size * (world_size + 1) // 2)
        ref_residual = reduced + residual.float()
        ref_norm = ref_residual * torch.rsqrt(
            ref_residual.pow(2).mean(dim=-1, keepdim=True) + eps
        )
        ref_norm = ref_norm * weight

        torch.testing.assert_close(
            residual_out.float(), ref_residual, atol=2e-2, rtol=2e-2
        )
        torch.testing.assert_close(norm_out.float(), ref_norm, atol=2e-2, rtol=2e-2)


def check_reduce_scatter(
    rsag, rank: int, world_size: int, tokens: List[int], hidden_size: int, device
) -> None:
    full = torch.full(
        (sum(tokens), hidden_size),
        rank + 1,
        dtype=torch.bfloat16,
        device=device,
    )

    result = reduce_scatter(rsag, full, token_list_in_group=tokens)
    expected = torch.full(
        (tokens[rank], hidden_size),
        world_size * (world_size + 1) // 2,
        dtype=torch.bfloat16,
        device=device,
    )

    assert result.shape == expected.shape
    assert (
        result.untyped_storage().data_ptr()
        != rsag.comm_buff.untyped_storage().data_ptr()
    )
    torch.testing.assert_close(result, expected, atol=0, rtol=0)


def check_amd_rsag_reuses_single_barrier_signal(
    rsag,
    rank: int,
    world_size: int,
    hidden_size: int,
    device,
) -> None:
    """Stress one-/multi-subgroup signal reuse with a rotating delayed rank."""
    tokens = token_cases(world_size)[-1]
    local = torch.full(
        (tokens[rank], hidden_size),
        rank + 1,
        dtype=torch.bfloat16,
        device=device,
    )
    full = torch.full(
        (sum(tokens), hidden_size),
        rank + 1,
        dtype=torch.bfloat16,
        device=device,
    )

    dist.barrier()
    scatter = None
    for iteration in range(32):
        if iteration % world_size == rank:
            time.sleep(0.001)
        all_gather(
            rsag,
            local,
            token_list_in_group=tokens,
            safe=False,
        )
        scatter = reduce_scatter(
            rsag,
            full,
            token_list_in_group=tokens,
            safe=False,
        )

    # Preserve the final gather before another collective reuses the symmetric
    # buffer. This also exercises RS-exit -> AG-exit reuse one final time.
    gather = all_gather(
        rsag,
        local,
        token_list_in_group=tokens,
        safe=True,
    )

    expected_gather = torch.empty_like(gather)
    offset = 0
    for peer, peer_tokens in enumerate(tokens):
        expected_gather[offset : offset + peer_tokens].fill_(peer + 1)
        offset += peer_tokens
    torch.testing.assert_close(gather, expected_gather, atol=0, rtol=0)

    assert scatter is not None
    expected_scatter = torch.full(
        (tokens[rank], hidden_size),
        world_size * (world_size + 1) // 2,
        dtype=torch.bfloat16,
        device=device,
    )
    torch.testing.assert_close(scatter, expected_scatter, atol=0, rtol=0)

    signal = rsag.symm_mem_hdl.get_signal_pad(rank, (3,), dtype=torch.int32)
    torch.testing.assert_close(signal, torch.zeros_like(signal), atol=0, rtol=0)


def run_rsag_test(world_size: int, hidden_size: int) -> None:
    if not torch.cuda.is_available():
        pytest.skip("CUDA/ROCm is required for TritonRSAG tests")
    if world_size > torch.cuda.device_count():
        pytest.skip(f"Need {world_size} GPUs, have {torch.cuda.device_count()}")

    port = get_open_port()
    error_dict = mp.Manager().dict()
    mp.spawn(
        worker_fn,
        args=(world_size, port, hidden_size, error_dict),
        nprocs=world_size,
        join=True,
    )

    if error_dict:
        raise RuntimeError("\n".join(f"Rank {r}: {e}" for r, e in error_dict.items()))


@pytest.mark.parametrize("world_size", [2, 4, 8])
def test_triton_communication_correctness(world_size):
    run_rsag_test(world_size=world_size, hidden_size=2880)


def test_amd_rsag_num_blocks_uses_rank_local_persistent_grid(monkeypatch):
    from tokenspeed_kernel.ops.communication.triton import amd_rsag_num_blocks

    class DeviceProperties:
        multi_processor_count = 256

    monkeypatch.setattr(
        torch.cuda,
        "get_device_properties",
        lambda _device: DeviceProperties(),
    )

    device = torch.device("cuda:0")
    assert amd_rsag_num_blocks(0, device) == 1
    assert amd_rsag_num_blocks(1024, device) == 1
    assert amd_rsag_num_blocks(1025, device) == 2
    assert amd_rsag_num_blocks(8192 * 7168, device) == 256


def test_amd_rsag_reduce_scatter_uses_one_subgroup_and_returns_kernel_output(
    monkeypatch,
):
    import tokenspeed_kernel.ops.communication.triton as communication

    class FakeKernel:
        output = None
        launch_kwargs = None

        def __getitem__(self, _grid):
            def launch(_buffer_ptrs, _signal_ptrs, output, **_kwargs):
                self.output = output
                self.launch_kwargs = _kwargs

            return launch

    kernel = FakeKernel()
    state = communication.TritonCommState(
        group=None,
        rank_in_group=0,
        world_size=1,
        device=torch.device("cpu"),
        max_token_num=1,
        hidden_dim=4,
        comm_buff=torch.empty((1, 4), dtype=torch.bfloat16),
        symm_mem_hdl=SimpleNamespace(
            buffer_ptrs_dev=None,
            signal_pad_ptrs_dev=None,
            rank=0,
            world_size=1,
        ),
    )
    hidden_states = torch.ones((1, 4), dtype=torch.bfloat16)

    monkeypatch.setattr(communication, "amd_rsag_num_blocks", lambda *_args: 1)
    monkeypatch.setattr(communication, "amd_rsag_reduce_scatter_kernel", kernel)

    result = communication.amd_rsag_reduce_scatter(
        state,
        hidden_states,
        token_list_in_group=[1],
        safe=True,
    )

    assert result is kernel.output
    # The completion release only publishes VMEM operations from its issuing
    # subgroup. A multi-subgroup payload program needs an explicit whole-program
    # global-memory fence that Triton's current scalar-atomic lowering lacks.
    assert kernel.launch_kwargs["num_warps"] == 1
    assert result.untyped_storage().data_ptr() != state.comm_buff.data_ptr()
