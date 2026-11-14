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

"""K3 attention and MoE collectives that overlap the projections around them."""

import socket
from datetime import timedelta

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from tokenspeed_kernel.platform import current_platform
from utils import is_amd

pytestmark = pytest.mark.skipif(not is_amd(), reason="Iris requires AMD ROCm")


@pytest.fixture(autouse=True)
def _require_iris():
    pytest.importorskip("tokenspeed_kernel.ops.communication._iris")


def _published_worker(rank: int, port: int) -> None:
    device = torch.device(f"cuda:{rank}")
    torch.cuda.set_device(device)
    torch.set_num_threads(1)
    dist.init_process_group(
        "gloo",
        init_method=f"tcp://127.0.0.1:{port}",
        rank=rank,
        world_size=8,
        timeout=timedelta(seconds=180),
    )
    group = dist.new_group(backend="nccl", timeout=timedelta(seconds=180))
    from tokenspeed_kernel.ops.communication._iris.kimi_k3 import (
        prepare_k3_communication,
    )
    from tokenspeed_kernel_amd.ops.gfx950.gemm.fp16.largem import (
        launch_gluon_mm_a16w16_prefill_gfx950,
    )

    k3_state = prepare_k3_communication(
        group=group,
        rank_in_group=rank,
        staged_max_numel=0,
        producer_direct_max_numel=8192 * 10752,
        attnres_max_numel=0,
        attnres_max_rows=0,
        enable_lamport=False,
        sharded_max_rows=8192,
        dtype=torch.bfloat16,
        device=device,
        all_reduce=None,
    )
    state = k3_state.all_reduce
    sharded = k3_state.sharded
    gen = torch.Generator(device=device)

    def mix(partial, residual, history, score, norm, blocks, is_sharded, published):
        result = k3_state.attention_mix(
            partial,
            residual,
            history,
            score,
            norm,
            eps=1e-6,
            out_norm_weight=norm,
            out_norm_eps=1e-6,
            num_valid_blocks=blocks,
            residual_is_sharded=is_sharded,
            published=published,
            defer_gather=False,
        )
        assert result is not None
        return result[0].clone(), result[1].clone()

    for m, k in ((4096, 1536), (6144, 1536), (8192, 1536), (8192, 768)):
        local = m // 8
        partial = state.acquire_outputs(((m, 7168),))[0]
        gen.manual_seed(500 + rank + m + k)
        activation = torch.randn(
            (m, k), device=device, dtype=torch.bfloat16, generator=gen
        )
        weight = (
            torch.randn((7168, k), device=device, dtype=torch.bfloat16, generator=gen)
            / 40
        )
        gen.manual_seed(77 + m)
        residual = torch.randn(
            (m, 7168), device=device, dtype=torch.bfloat16, generator=gen
        )
        history = torch.randn(
            (8, m, 7168), device=device, dtype=torch.bfloat16, generator=gen
        )
        score = (
            torch.randn((7168,), device=device, dtype=torch.bfloat16, generator=gen)
            / 128
        )
        norm = torch.ones_like(score)
        own = residual[rank * local : (rank + 1) * local].contiguous()
        for prefix, is_sharded, blocks in (
            (None, False, 0),
            (residual, False, 4),
            (own, True, 8),
        ):
            launch_gluon_mm_a16w16_prefill_gfx950(
                activation, weight, torch.bfloat16, out=partial
            )
            expected = mix(
                partial, prefix, history, score, norm, blocks, is_sharded, None
            )
            for repeat in range(3):
                if rank == (m // 1024 + repeat) % 8:
                    # Peers poll this rank's row tiles while its GEMM starts late.
                    torch.cuda._sleep(200_000)
                assert k3_state.project_attention(activation, weight, partial)
                published = k3_state.take_published(partial)
                assert published is not None
                assert k3_state.take_published(partial) is None
                actual = mix(
                    partial, prefix, history, score, norm, blocks, is_sharded, published
                )
                for got, want in zip(actual, expected, strict=True):
                    torch.testing.assert_close(got, want, atol=0, rtol=0)
        torch.cuda.synchronize()
        dist.barrier()
    # The scatter runs beside the forward's stream only from a high-priority one.
    assert sharded.scatter_stream.priority < 0

    # Publication counts wrap as int32; targets follow them across the wrap.
    m, k = 8192, 1536
    activation = torch.randn((m, k), device=device, dtype=torch.bfloat16, generator=gen)
    weight = (
        torch.randn((7168, k), device=device, dtype=torch.bfloat16, generator=gen) / 40
    )
    partial = state.acquire_outputs(((m, 7168),))[0]
    launch_gluon_mm_a16w16_prefill_gfx950(
        activation, weight, torch.bfloat16, out=partial
    )
    expected = mix(partial, None, history, score, norm, 0, False, None)
    torch.cuda.synchronize()
    dist.barrier()
    per_call = 8 * 28
    start = (2**31 - 1) // per_call * per_call
    sharded._published_counts = [start] * 4
    sharded.published_counters.fill_(start)
    torch.cuda.synchronize()
    dist.barrier()
    for _ in range(3):
        assert k3_state.project_attention(activation, weight, partial)
        actual = mix(
            partial,
            None,
            history,
            score,
            norm,
            0,
            False,
            k3_state.take_published(partial),
        )
        for got, want in zip(actual, expected, strict=True):
            torch.testing.assert_close(got, want, atol=0, rtol=0)
    assert sharded.published_counters[0].item() < 0
    torch.cuda.synchronize()
    dist.barrier()

    # Declines launch nothing and leave no publication behind.
    counters = sharded.published_counters.clone()
    for rows, out in (
        (2048, state.acquire_outputs(((2048, 7168),))[0]),
        (4096 + 2048 - 8, state.acquire_outputs(((6136, 7168),))[0]),
        (8192, torch.empty((8192, 7168), device=device, dtype=torch.bfloat16)),
        (8192, partial[:, :7168].float()),
    ):
        assert not k3_state.project_attention(activation[:rows], weight, out)
        assert k3_state.take_published(out) is None
    oversized = state.acquire_outputs(((10240, 7168),))[0]
    assert not k3_state.project_attention(
        torch.zeros((10240, k), device=device, dtype=torch.bfloat16), weight, oversized
    )
    assert not k3_state.project_attention(activation, weight[:, :768], partial)
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        assert not k3_state.project_attention(activation, weight, partial)
    torch.testing.assert_close(sharded.published_counters, counters, atol=0, rtol=0)
    # Peers' next publication adds to these counters.
    dist.barrier()
    # A publication is only handed over for the partial it wrote.
    partial = state.acquire_outputs(((m, 7168),))[0]
    assert k3_state.project_attention(activation, weight, partial)
    assert k3_state.take_published(partial[: m - 2048]) is None
    torch.cuda.synchronize()
    dist.barrier()
    dist.destroy_process_group()


def _moe_shared_worker(rank: int, port: int) -> None:
    device = torch.device(f"cuda:{rank}")
    torch.cuda.set_device(device)
    torch.set_num_threads(1)
    dist.init_process_group(
        "gloo",
        init_method=f"tcp://127.0.0.1:{port}",
        rank=rank,
        world_size=8,
        timeout=timedelta(seconds=180),
    )
    group = dist.new_group(backend="nccl", timeout=timedelta(seconds=180))
    from tokenspeed_kernel.ops.communication._iris.kimi_k3 import (
        prepare_k3_communication,
    )
    from tokenspeed_kernel_amd.ops.gfx950.gemm.fp16.largem import (
        launch_gluon_mm_a16w16_prefill_gfx950,
    )

    k3_state = prepare_k3_communication(
        group=group,
        rank_in_group=rank,
        staged_max_numel=0,
        producer_direct_max_numel=8192 * 10752,
        attnres_max_numel=0,
        attnres_max_rows=0,
        enable_lamport=False,
        sharded_max_rows=8192,
        dtype=torch.bfloat16,
        device=device,
        all_reduce=None,
    )
    state = k3_state.all_reduce
    gen = torch.Generator(device=device)

    def bf16(shape, scale=1.0):
        return (
            torch.randn(shape, device=device, dtype=torch.bfloat16, generator=gen)
            * scale
        )

    gen.manual_seed(41)
    up = bf16((7168, 3584), 1 / 64)
    norm = bf16((3584,)) + 1
    res_weight, rms_weight = bf16((7168,), 1 / 128), bf16((7168,)) + 1
    out_weight = bf16((7168,)) + 1
    down = bf16((7168, 768), 1 / 28)
    o_weight = bf16((7168, 1536), 1 / 40)
    for rows in (4096, 8192, 6144):
        local = rows // 8
        gen.manual_seed(900 + rank + rows)
        shared_input, routed_source = bf16((rows, 768)), bf16((rows, 3584), 1 / 8)
        attention_input = bf16((rows, 1536))
        gen.manual_seed(77 + rows)
        prefix = bf16((local, 7168))
        blocks = bf16((8, rows, 7168))
        mix = dict(
            block_residual=blocks,
            res_weight=res_weight,
            rms_weight=rms_weight,
            mix_eps=1e-6,
            out_norm_weight=out_weight,
            out_norm_eps=1e-5,
            num_valid_blocks=3,
            block_write_idx=-1,
            next_weight=None,
            gather_next_projection=False,
        )
        tail = dict(prefix_is_sharded=True, norm_weight=norm, eps=1e-6)

        def run(with_mix, publish):
            # An attention projection publishes first; both share the counters.
            partial = state.acquire_outputs(((rows, 7168),))[0]
            assert k3_state.project_attention(attention_input, o_weight, partial)
            k3_state.take_published(partial)
            state.all_reduce_symmetric((partial,))
            outputs = state.acquire_outputs(((rows, 3584), (rows, 7168)))
            record = None
            if publish:
                record = k3_state.project_moe_shared(shared_input, down, outputs)
                assert record is not None
            else:
                launch_gluon_mm_a16w16_prefill_gfx950(
                    shared_input, down, torch.bfloat16, out=outputs[1]
                )
            if rank == rows // 1024 % 8:
                torch.cuda._sleep(200_000)
            # The routed experts write their partial beside the shared reduction.
            outputs[0].copy_(routed_source)
            if with_mix:
                result = k3_state.moe_tail_mix(
                    *outputs, prefix, up, **tail, **mix, shared_reduced=record
                )
                return result[0].clone(), result[1].clone()
            return (
                k3_state.moe_tail(
                    *outputs, prefix, up, **tail, shared_reduced=record
                ).clone(),
            )

        for with_mix in (False, True):
            expected = run(with_mix, False)
            for _ in range(2):
                actual = run(with_mix, True)
                for got, want in zip(actual, expected, strict=True):
                    torch.testing.assert_close(got, want, atol=0, rtol=0)
        # A declined tail still orders after the side reduce; the next reduce sees it.
        outputs = state.acquire_outputs(((rows, 3584), (rows, 7168)))
        record = k3_state.project_moe_shared(shared_input, down, outputs)
        outputs[0].copy_(routed_source)
        assert (
            k3_state.moe_tail(*outputs, prefix[:-8], up, **tail, shared_reduced=record)
            is None
        )
        joined = state.all_reduce_symmetric(outputs)
        expected_shared = torch.empty_like(outputs[1])
        launch_gluon_mm_a16w16_prefill_gfx950(
            shared_input, down, torch.bfloat16, out=expected_shared
        )
        reference = expected_shared.float()
        dist.all_reduce(reference, group=group)
        torch.testing.assert_close(
            joined[1], reference.bfloat16(), atol=1 / 64, rtol=1 / 128
        )
        torch.cuda.synchronize()
        dist.barrier()

    # Only the producers' own (routed, shared) pair at whole 256-row owner tiles.
    outputs = state.acquire_outputs(((2048, 3584), (2048, 7168)))
    assert k3_state.project_moe_shared(shared_input[:2048], down, outputs) is None
    outputs = state.acquire_outputs(((rows, 3584), (rows, 7168)))
    assert k3_state.project_moe_shared(shared_input, down, outputs[::-1]) is None
    assert (
        k3_state.project_moe_shared(
            shared_input, down, (outputs[0], torch.empty_like(outputs[1]))
        )
        is None
    )
    assert k3_state.project_moe_shared(shared_input, down[:, :512], outputs) is None
    torch.cuda.synchronize()
    dist.barrier()
    dist.destroy_process_group()


def _deferred_gather_worker(rank: int, port: int) -> None:
    device = torch.device(f"cuda:{rank}")
    torch.cuda.set_device(device)
    torch.set_num_threads(1)
    dist.init_process_group(
        "gloo",
        init_method=f"tcp://127.0.0.1:{port}",
        rank=rank,
        world_size=8,
        timeout=timedelta(seconds=180),
    )
    group = dist.new_group(backend="nccl", timeout=timedelta(seconds=180))
    from tokenspeed_kernel.ops.communication._iris.kimi_k3 import (
        prepare_k3_communication,
    )
    from tokenspeed_kernel_amd.ops.gfx950.moe.mxfp4.routing import (
        invoke_sigmoid_bias_topk_route_prefill_gluon,
    )

    k3_state = prepare_k3_communication(
        group=group,
        rank_in_group=rank,
        staged_max_numel=0,
        producer_direct_max_numel=8192 * 10752,
        attnres_max_numel=0,
        attnres_max_rows=0,
        enable_lamport=False,
        sharded_max_rows=8192,
        dtype=torch.bfloat16,
        device=device,
        all_reduce=None,
    )
    state = k3_state.all_reduce
    gen = torch.Generator(device=device)
    gen.manual_seed(1030)
    packed_weight = (torch.randn((6016, 7168), device=device, generator=gen) / 40).to(
        torch.bfloat16
    )
    bias = torch.randn(896, device=device, generator=gen) / 100
    score = (torch.randn((7168,), device=device, generator=gen) / 128).to(
        torch.bfloat16
    )
    norm = torch.ones_like(score)

    def route(rows, logits):
        ids, weights = invoke_sigmoid_bias_topk_route_prefill_gluon(
            logits,
            bias,
            16,
            routed_scaling_factor=2.5,
            normalize_topk_weights=True,
            simulation_seed=None,
        )
        return weights, ids

    def mix(partial, residual, history, blocks, defer):
        return k3_state.attention_mix(
            partial,
            residual,
            history,
            score,
            norm,
            eps=1e-6,
            out_norm_weight=norm,
            out_norm_eps=1e-6,
            num_valid_blocks=blocks,
            residual_is_sharded=True,
            published=None,
            defer_gather=defer,
        )

    def inputs(hidden, pending):
        result = k3_state.moe_row_sharded_inputs(
            hidden,
            packed_weight,
            beta=4.0,
            linear_beta=25.0,
            route=route,
            attention_gather=pending,
        )
        assert result is not None
        (packed, scales), weights, ids, shared = result
        return tuple(t.clone() for t in (packed, scales, weights, ids, shared))

    # 8192 rows mix in the fused gather; 4096 with 8 blocks use the standalone mixer.
    for rows, blocks in ((8192, 4), (8192, 8), (4096, 8), (4096, 3)):
        local = rows // 8
        gen.manual_seed(31 + rank + rows + blocks)
        source = (torch.randn((rows, 7168), device=device, generator=gen) / 8).to(
            torch.bfloat16
        )
        gen.manual_seed(77 + rows)
        residual = torch.randn((rows, 7168), device=device, generator=gen).to(
            torch.bfloat16
        )
        own = residual[rank * local : (rank + 1) * local].contiguous()
        history = torch.randn((8, rows, 7168), device=device, generator=gen).to(
            torch.bfloat16
        )
        partial = state.acquire_outputs(((rows, 7168),))[0]
        partial.copy_(source)
        shard, activation = mix(partial, own, history, blocks, False)
        expected = (shard.clone(), activation.clone())
        expected_inputs = inputs(activation, None)
        for repeat in range(2):
            partial.copy_(source)
            if rank == (rows // 1024 + blocks + repeat) % 8:
                torch.cuda._sleep(200_000)
            shard, activation, pending = mix(partial, own, history, blocks, True)
            torch.testing.assert_close(shard, expected[0], atol=0, rtol=0)
            torch.testing.assert_close(
                pending.own_rows,
                expected[1][rank * local : (rank + 1) * local],
                atol=0,
                rtol=0,
            )
            got = inputs(activation, pending)
            for actual, want in zip(got, expected_inputs, strict=True):
                assert torch.equal(actual, want)
            torch.testing.assert_close(activation, expected[1], atol=0, rtol=0)
        # A declining shape still joins the gather before returning.
        partial.copy_(source)
        _, activation, pending = mix(partial, own, history, blocks, True)
        assert (
            k3_state.moe_row_sharded_inputs(
                activation,
                packed_weight[:-8],
                beta=4.0,
                linear_beta=25.0,
                route=route,
                attention_gather=pending,
            )
            is None
        )
        torch.testing.assert_close(activation, expected[1], atol=0, rtol=0)
        # The other path joins explicitly before reading every row.
        partial.copy_(source)
        _, activation, pending = mix(partial, own, history, blocks, True)
        assert k3_state.join_gather(pending) is activation
        assert k3_state.sharded.gather_stream.priority < 0
        torch.testing.assert_close(activation, expected[1], atol=0, rtol=0)
        torch.cuda.synchronize()
        dist.barrier()
    dist.destroy_process_group()


@pytest.mark.skipif(
    not current_platform().is_cdna4 or torch.cuda.device_count() < 8,
    reason="requires eight CDNA4 GPUs",
)
def test_deferred_attention_gather_feeds_the_moe_row_inputs():
    with socket.socket() as sock:
        sock.bind(("", 0))
        port = sock.getsockname()[1]
    mp.spawn(_deferred_gather_worker, args=(port,), nprocs=8, join=True)


@pytest.mark.skipif(
    not current_platform().is_cdna4 or torch.cuda.device_count() < 8,
    reason="requires eight CDNA4 GPUs",
)
def test_moe_shared_rows_reduce_beside_the_experts():
    with socket.socket() as sock:
        sock.bind(("", 0))
        port = sock.getsockname()[1]
    mp.spawn(_moe_shared_worker, args=(port,), nprocs=8, join=True)


@pytest.mark.skipif(
    not current_platform().is_cdna4 or torch.cuda.device_count() < 8,
    reason="requires eight CDNA4 GPUs",
)
def test_published_attention_scatter_matches_the_ordinary_scatter():
    with socket.socket() as sock:
        sock.bind(("", 0))
        port = sock.getsockname()[1]
    mp.spawn(_published_worker, args=(port,), nprocs=8, join=True)
