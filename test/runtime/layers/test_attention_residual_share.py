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

"""Sequence-parallel AttnRes mixes on a real 4-rank group: each rank mixes
its share of the tokens through multicast buffers and every rank then holds
the whole mix, with the bits of an unshared run."""

import socket
import traceback

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

WORLD_SIZE = 4


def get_open_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("", 0))
        return sock.getsockname()[1]


def worker_fn(rank, port, error_dict):
    try:
        worker_main(rank, port)
    except Exception:
        error_dict[rank] = traceback.format_exc()


def worker_main(rank: int, port: int) -> None:
    device = torch.device(f"cuda:{rank}")
    torch.cuda.set_device(device)
    dist.init_process_group(
        backend="nccl",
        init_method=f"tcp://localhost:{port}",
        rank=rank,
        world_size=WORLD_SIZE,
    )
    from tokenspeed_kernel.ops.communication.triton import MulticastRows

    from tokenspeed.runtime.layers.attention_residual import (
        AttentionResidual,
        AttentionResidualState,
        PrecisionRMSNorm,
        ResidualShare,
        norm_gains,
    )

    try:
        # Above the block-set sweep, with full layers 5 and 11 (three queries).
        tokens, width = 300, 64
        torch.manual_seed(3)
        owner = AttentionResidual(12, width, 6, [0, 1, 2, 3, 4, 6, 7, 8, 9, 10], True)
        with torch.no_grad():
            for query in owner.parameters():
                query.normal_()
        owner = owner.cuda()
        owner.derive_score_weights()
        embedding = torch.randn((tokens, width), device=device).to(torch.bfloat16)
        outputs = [
            torch.randn((tokens, width), device=device).to(torch.bfloat16)
            for _ in range(24)
        ]

        def bf16_norm():
            return PrecisionRMSNorm(width, 1e-6, torch.bfloat16, True, False).cuda()

        attention_norms = [
            tuple(bf16_norm() for _ in range(3 if layer in (5, 11) else 1))
            for layer in range(12)
        ]
        norm = bf16_norm()

        def run(share):
            state = AttentionResidualState(owner, embedding, None, share)
            mixed = []
            for layer in range(12):
                norms = attention_norms[layer]
                # A shared mix lands in a buffer the next mix of its use rewrites.
                mixed.extend(
                    x.clone()
                    for x in state.attention_input(layer, norms, norm_gains(norms))
                )
                attention, mlp = outputs[2 * layer], outputs[2 * layer + 1]
                mixed.append(state.mlp_input(layer, attention, norm).clone())
                state.finalize_layer(layer, attention, mlp)
                if layer == 7:
                    mixed.append(state.full_partial())
            mixed.append(state.compute_output(norm))
            return mixed

        expected = run(None)
        group = dist.group.WORLD
        size = -(-tokens // WORLD_SIZE)
        first = rank * size
        share = ResidualShare(
            first=first,
            count=min(size, tokens - first),
            attention=MulticastRows(group, 3 * tokens * width, device),
            mlp=MulticastRows(group, tokens * width, device),
            gather=MulticastRows(group, tokens * width, device),
        )
        actual = run(share)
        torch.cuda.synchronize()
        assert len(actual) == len(expected)
        for index, (a, b) in enumerate(zip(actual, expected)):
            assert torch.equal(a, b), f"mix {index} differs"
    finally:
        dist.destroy_process_group()


@pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.device_count() < WORLD_SIZE,
    reason=f"needs {WORLD_SIZE} GPUs",
)
def test_shared_mixes_carry_the_bits_of_an_unshared_run():
    port = get_open_port()
    error_dict = mp.Manager().dict()
    mp.spawn(worker_fn, args=(port, error_dict), nprocs=WORLD_SIZE, join=True)
    if error_dict:
        raise RuntimeError("\n".join(f"Rank {r}: {e}" for r, e in error_dict.items()))
