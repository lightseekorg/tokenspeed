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

"""Attention producer ownership and reuse of the existing MoE communication state."""

import re
import socket
from contextlib import ExitStack
from datetime import timedelta
from itertools import product

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from tokenspeed_kernel.platform import current_platform
from utils import assert_no_triton_compile, is_amd

pytestmark = pytest.mark.skipif(not is_amd(), reason="Iris requires AMD ROCm")


@pytest.fixture(autouse=True)
def _require_iris():
    pytest.importorskip("tokenspeed_kernel.ops.communication.iris")


@pytest.mark.parametrize("rank", (0, 7))
@pytest.mark.parametrize(
    "operation,has_residual", (("reduce", False), ("reduce", True), ("gather", False))
)
def test_attention_row_vector_codegen(rank, operation, has_residual, tmp_path):
    """Runtime rows must preserve vector payload traffic, including tail tiles."""
    from tokenspeed_kernel._triton import gluon, triton
    from tokenspeed_kernel.ops.communication.iris import (
        iris_attention_push_gather_gluon_kernel,
        iris_attention_reduce_scatter_gluon_kernel,
    )

    fn = (
        iris_attention_reduce_scatter_gluon_kernel
        if operation == "reduce"
        else iris_attention_push_gather_gluon_kernel
    )
    constants = {
        "RANK": rank,
        "BLOCK_ELEMENTS": 2048,
        "NUM_PROGRAMS": 24 if operation == "reduce" else 32,
        "NUM_WARPS": 4,
    }
    if operation == "reduce":
        constants["HAS_RESIDUAL"] = has_residual
    signature = {}
    for name in fn.arg_names:
        if name in constants:
            continue
        if name.startswith("heap_base_"):
            signature[name] = "i64"
        elif name in ("LOCAL_ROWS", "PARTITION_ELEMENTS"):
            signature[name] = "i32"
        elif name == "ready_flags":
            signature[name] = "*i32"
        else:
            signature[name] = "*bf16"
    source = gluon._runtime.GluonASTSource(
        fn,
        signature,
        constexprs=constants,
        # Only pointers/bases are aligned; odd runtime row counts have no hint.
        attrs={
            (index,): [["tt.divisibility", 16]]
            for index, name in enumerate(fn.arg_names)
            if name in signature and signature[name] != "i32"
        },
    )
    kernel = triton.compile(
        source,
        target=triton.backends.compiler.GPUTarget("hip", "gfx950", 64),
        options={"num_warps": 4},
    )
    assembly = kernel.asm["amdgcn"]
    (tmp_path / f"attention-{operation}-rank{rank}.amdgcn").write_text(assembly)
    lines = assembly.splitlines()
    loads = 8 + int(has_residual) if operation == "reduce" else 1
    stores = 1 if operation == "reduce" else 8
    assert sum("buffer_load_dwordx4" in line for line in lines) == loads
    assert sum("buffer_store_dwordx4" in line for line in lines) == stores
    if operation == "gather":
        assert (
            sum("buffer_store_dwordx4" in line and "sc0 sc1" in line for line in lines)
            == 8
        )
    assert not re.search(r"\bbuffer_(?:load_ushort|store_short)\b", assembly)


def _reference_mix(prefix, history, score, norm, valid_blocks):
    # An independent FP64 softmax avoids treating a different subgroup
    # reduction tree's rounding at the two BF16 boundaries as the oracle.
    output = torch.empty_like(prefix)
    for start in range(0, prefix.shape[0], 256):
        end = start + 256
        values = torch.cat(
            (history[:valid_blocks, start:end], prefix[None, start:end]), dim=0
        ).double()
        logits = (values * score.double() * norm.double()).sum(-1)
        logits *= torch.rsqrt(values.square().mean(-1) + 1e-6)
        mixed = (logits.softmax(0)[..., None] * values).sum(0)
        mixed = mixed.bfloat16().double()
        output[start:end] = (
            mixed
            * torch.rsqrt(mixed.square().mean(-1, keepdim=True) + 1e-6)
            * norm.double()
        ).bfloat16()
    return output


def _attention_worker(rank: int, port: int) -> None:
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
    from tokenspeed_kernel.ops.communication import triton as comm
    from tokenspeed_kernel.ops.communication.iris import (
        iris_attention_mix,
        iris_attention_mix_push_gluon_kernel,
        iris_attention_push_gather_gluon_kernel,
        iris_attention_reduce_scatter_gluon_kernel,
        iris_kimi3_moe_tail,
    )

    backing = comm.TritonCommState(
        group=group,
        rank_in_group=rank,
        world_size=8,
        device=device,
        attnres_max_numel=0,
        enable_lamport=True,
        moe_tail_max_rows=8192,
        max_numel=0,
        max_bytes=8192 * 10752 * 2,
        max_token_num=0,
        hidden_dim=0,
        comm_buff=None,
        symm_mem_hdl=None,
    )
    comm.initialize_all_reduce_state(backing, torch.bfloat16)
    state = comm._get_or_create_iris_state(backing, torch.bfloat16)
    buffers = (
        state._input_buf,
        state._producer_direct_scratch_buf,
        state._producer_direct_ready_flags,
        state._moe_tail_output_buf,
        state._moe_tail_ready_flags,
    )
    pointers = tuple(t.data_ptr() for t in buffers)
    gen = torch.Generator(device=device)
    held_mix = None
    warmed_variants = set()
    for m in (
        56,
        57,
        64,
        128,
        256,
        512,
        520,
        848,
        1024,
        2048,
        4096,
        6224,
        8144,
        8192,
    ):
        partial = comm.acquire_symm_outputs(backing, ((m, 7168),), torch.bfloat16)[0]
        gen.manual_seed(31729 + rank)
        source = (
            torch.randn((m, 7168), device=device, dtype=torch.bfloat16, generator=gen)
            / 8
        )
        gen.manual_seed(84003)
        residual = torch.randn(
            (m, 7168), device=device, dtype=torch.bfloat16, generator=gen
        )
        if m == 520:
            # Also cover padding between tokens, beyond the model's ordinary
            # block-major storage used by the remaining cases.
            history = torch.randn(
                (m, 12, 7168), device=device, dtype=torch.bfloat16, generator=gen
            ).transpose(0, 1)
        else:
            history = torch.randn(
                (12, m, 7168), device=device, dtype=torch.bfloat16, generator=gen
            )
        score = (
            torch.randn((7168,), device=device, dtype=torch.bfloat16, generator=gen)
            / 128
        )
        norm = torch.ones_like(score)
        reduced = source.float()
        dist.all_reduce(reduced, group=group)
        reduced = reduced.to(torch.bfloat16)
        partial.copy_(source)
        ordinary = comm.all_reduce_symmetric(backing, (partial,))[0].clone()
        # FP32 RCCL may add in another order at BF16 rounding ties. Require
        # agreement with that oracle within a BF16 ulp, and exact preservation
        # of the existing Iris tree below.
        torch.testing.assert_close(ordinary, reduced, atol=0.001953125, rtol=0.0078125)
        reduced = ordinary
        # Invalid row counts reject before history or residual specialization.
        histories = (0, 4, 7, 8, 11) if m % 8 == 0 else (0,)
        for prefix, valid_blocks in product((None, residual), histories):
            partial.copy_(source)
            if rank == m % 8:
                torch.cuda._sleep(100_000)
            expected = reduced if prefix is None else reduced + prefix
            if held_mix is not None:
                for tensor, reference in held_mix:
                    torch.testing.assert_close(tensor, reference, atol=0, rtol=0)
            if m % 8 == 0:
                reference_mixed = _reference_mix(
                    expected, history, score, norm, valid_blocks
                )
                partial.copy_(source)
                fused = (m < 1024 or m >= 4096) and (
                    valid_blocks <= 6 or (m >= 7680 and valid_blocks <= 8)
                )
                variant = (fused, valid_blocks, prefix is not None, history.stride(1))
                with ExitStack() as stack:
                    if variant in warmed_variants:
                        for kernel in (
                            iris_attention_reduce_scatter_gluon_kernel,
                            iris_attention_push_gather_gluon_kernel,
                            iris_attention_mix_push_gluon_kernel,
                        ):
                            stack.enter_context(assert_no_triton_compile(kernel))
                    mixed = iris_attention_mix(
                        partial,
                        prefix,
                        history,
                        score,
                        norm,
                        eps=1e-6,
                        out_norm_weight=norm,
                        out_norm_eps=1e-6,
                        num_valid_blocks=valid_blocks,
                        group=group,
                    )
                warmed_variants.add(variant)
                assert mixed is not None
                shard, activation = mixed
                assert shard.shape == (m // 8, 7168)
                assert activation.data_ptr() == state._moe_tail_output_buf.data_ptr()
                assert shard.data_ptr() != state._producer_direct_scratch_buf.data_ptr()
                torch.testing.assert_close(
                    shard, expected[rank * m // 8 : (rank + 1) * m // 8], atol=0, rtol=0
                )
                # FP32 reductions can round on opposite sides of a BF16
                # midpoint in both the mix and the output normalization.
                torch.testing.assert_close(
                    activation,
                    reference_mixed,
                    atol=1 / 512,
                    rtol=1 / 64,
                    msg=lambda message: (
                        f"M={m}, history={valid_blocks}, residual={prefix is not None}: {message}"
                    ),
                )
                if m in (56, 64) and valid_blocks in (4, 11):
                    # Exercise both fusion choices at the new lower boundary.
                    torch.cuda.synchronize()
                    dist.barrier()
                    small_graph = torch.cuda.CUDAGraph()
                    with torch.cuda.graph(small_graph):
                        partial.copy_(source)
                        captured = iris_attention_mix(
                            partial,
                            prefix,
                            history,
                            score,
                            norm,
                            eps=1e-6,
                            out_norm_weight=norm,
                            out_norm_eps=1e-6,
                            num_valid_blocks=valid_blocks,
                            group=group,
                        )
                    assert captured is not None
                    for sign in (-1, 1):
                        source.neg_()
                        small_graph.replay()
                        reference_prefix = reduced * sign
                        if prefix is not None:
                            reference_prefix += prefix
                        torch.testing.assert_close(
                            captured[0],
                            reference_prefix[rank * m // 8 : (rank + 1) * m // 8],
                            atol=0,
                            rtol=0,
                        )
                        torch.testing.assert_close(
                            captured[1],
                            _reference_mix(
                                reference_prefix, history, score, norm, valid_blocks
                            ),
                            atol=1 / 512,
                            rtol=1 / 64,
                        )
                held_mix = ((shard, shard.clone()), (activation, activation.clone()))
            else:
                # Peers must not enter the next valid collective while a
                # slower rank is asserting that rejection left flags intact.
                torch.cuda.synchronize()
                dist.barrier()
                flags_before = state._producer_direct_ready_flags.clone()
                assert (
                    iris_attention_mix(
                        partial,
                        prefix,
                        history,
                        score,
                        norm,
                        eps=1e-6,
                        out_norm_weight=norm,
                        out_norm_eps=1e-6,
                        num_valid_blocks=valid_blocks,
                        group=group,
                    )
                    is None
                )
                torch.testing.assert_close(
                    flags_before, state._producer_direct_ready_flags
                )
                dist.barrier()
        # The ordinary reduction overwrites both scratch and its result buffer.
        partial.copy_(source)
        comm.all_reduce_symmetric(backing, (partial,))
        if held_mix is not None:
            for tensor, reference in held_mix:
                torch.testing.assert_close(tensor, reference, atol=0, rtol=0)

    m = 512
    partial = comm.acquire_symm_outputs(backing, ((m, 7168),), torch.bfloat16)[0]
    source = source[:m].contiguous()
    residual = residual[:m].contiguous()
    reduced = reduced[:m].contiguous()
    history = history[:, :m]
    partial.copy_(source)
    flags_before = state._producer_direct_ready_flags.clone()
    for operand, prefix, owner in (
        (partial[:0], None, group),
        (partial.clone(), residual, group),
        (partial, residual, dist.group.WORLD),
        (partial, residual.float(), group),
        (partial, partial, group),
        (partial, state._moe_tail_output_buf[1 : m + 1], group),
        (partial, residual.as_strided(residual.shape, (1, m)), group),
        (partial.as_strided(partial.shape, (1, m)), None, group),
    ):
        assert (
            iris_attention_mix(
                operand,
                prefix,
                history,
                score,
                norm,
                eps=1e-6,
                out_norm_weight=norm,
                out_norm_eps=1e-6,
                num_valid_blocks=8,
                group=owner,
            )
            is None
        )
    torch.testing.assert_close(flags_before, state._producer_direct_ready_flags)
    torch.testing.assert_close(partial, source)

    # The projection capacity is wider than the result capacity. Check both
    # before publishing an entry flag, even when there is no history to read.
    oversized = comm.acquire_symm_outputs(backing, ((8200, 7168),), torch.bfloat16)[0]
    assert (
        iris_attention_mix(
            oversized,
            None,
            oversized.new_empty((0, 8200, 7168)),
            score,
            norm,
            eps=1e-6,
            out_norm_weight=norm,
            out_norm_eps=1e-6,
            num_valid_blocks=0,
            group=group,
        )
        is None
    )
    torch.testing.assert_close(flags_before, state._producer_direct_ready_flags)
    torch.testing.assert_close(partial, source)
    for blocks, scorer, epsilon, owner in (
        (history.float(), score, 1e-6, group),
        (history, score[:-1], 1e-6, group),
        (history, score.float(), 1e-6, group),
        (history, partial.view(-1)[:7168], 1e-6, group),
        (partial[None].expand(8, -1, -1), score, 1e-6, group),
        (history, score, 0.0, group),
        (history, score, float("nan"), group),
    ):
        assert (
            iris_attention_mix(
                partial,
                residual,
                blocks,
                scorer,
                norm,
                eps=epsilon,
                out_norm_weight=norm,
                out_norm_eps=1e-6,
                num_valid_blocks=8,
                group=owner,
            )
            is None
        )
    torch.testing.assert_close(flags_before, state._producer_direct_ready_flags)
    torch.testing.assert_close(partial, source)
    # Peers must not enter the MoE tail collective below while a slower rank
    # is still asserting that the rejections left its flags intact.
    dist.barrier()

    # A real MoE tail shares the input, scratch, and epochs with attention.
    gen.manual_seed(3004)
    weight = (
        torch.randn((7168, 3584), device=device, dtype=torch.bfloat16, generator=gen)
        / 64
    )
    moe_prefix = reduced + residual
    expected_activation = _reference_mix(moe_prefix, history, score, norm, 8)
    # The previous MoE result is the production residual. Each owner must finish
    # reading its rows before the attention gather overwrites those same rows.
    stored_residual = state._moe_tail_output_buf[:m]
    stored_residual.copy_(residual)
    partial.copy_(source)
    alias_result = iris_attention_mix(
        partial,
        stored_residual,
        history,
        score,
        norm,
        eps=1e-6,
        out_norm_weight=norm,
        out_norm_eps=1e-6,
        num_valid_blocks=8,
        group=group,
    )
    assert alias_result is not None
    torch.testing.assert_close(
        alias_result[0], moe_prefix[rank * m // 8 : (rank + 1) * m // 8], atol=0, rtol=0
    )
    torch.testing.assert_close(
        alias_result[1], expected_activation, atol=1 / 512, rtol=1 / 64
    )
    moe_sources = tuple(
        torch.full((m, width), (rank + 1) / 32, device=device, dtype=torch.bfloat16)
        for width in (3584, 7168)
    )
    inputs = comm.acquire_symm_outputs(backing, ((m, 3584), (m, 7168)), torch.bfloat16)
    for out, value in zip(inputs, moe_sources, strict=True):
        out.copy_(value)
    expected_moe = iris_kimi3_moe_tail(
        *inputs,
        moe_prefix,
        weight,
        prefix_is_sharded=False,
        norm_weight=None,
        eps=None,
        group=group,
    )
    assert expected_moe is not None
    expected_moe = expected_moe.clone()
    results = []

    def sequence():
        partial.copy_(source)
        first = comm.all_reduce_symmetric(backing, (partial,))[0].clone() + residual
        partial.copy_(source)
        mixed = iris_attention_mix(
            partial,
            residual,
            history,
            score,
            norm,
            eps=1e-6,
            out_norm_weight=norm,
            out_norm_eps=1e-6,
            num_valid_blocks=8,
            group=group,
        )
        assert mixed is not None
        shard, activation = mixed
        activation = activation.clone()
        inputs = comm.acquire_symm_outputs(
            backing, ((m, 3584), (m, 7168)), torch.bfloat16
        )
        for out, value in zip(inputs, moe_sources, strict=True):
            out.copy_(value)
        moe = iris_kimi3_moe_tail(
            *inputs,
            shard,
            weight,
            prefix_is_sharded=True,
            norm_weight=None,
            eps=None,
            group=group,
        )
        assert moe is not None
        partial.copy_(source)
        second = comm.all_reduce_symmetric(backing, (partial,))[0].clone()
        results[:] = [first, second, shard, activation, moe]

    sequence()
    torch.cuda.synchronize()
    dist.barrier()
    capture_stream = torch.cuda.Stream()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, stream=capture_stream):
        sequence()
    for _ in range(8):
        graph.replay()
    torch.cuda.synchronize()
    torch.testing.assert_close(results[0], reduced + residual, atol=0, rtol=0)
    torch.testing.assert_close(results[1], reduced, atol=0, rtol=0)
    torch.testing.assert_close(
        results[2], moe_prefix[rank * m // 8 : (rank + 1) * m // 8], atol=0, rtol=0
    )
    torch.testing.assert_close(
        results[3], expected_activation, atol=1 / 512, rtol=1 / 64
    )
    torch.testing.assert_close(results[4], expected_moe, atol=0, rtol=0)
    for initial_epoch in (2**31 - 3, -3):
        torch.cuda.synchronize()
        dist.barrier()
        state._producer_direct_ready_flags.fill_(initial_epoch)
        state._moe_tail_ready_flags.fill_(initial_epoch)
        torch.cuda.synchronize()
        dist.barrier()
        for repeat in range(4):
            # Change producer data between replays, and cross both epoch wraps.
            source.fill_((rank + 1) * (repeat + 1) / 128)
            residual.fill_((repeat + 1) / 64)
            reduced = torch.full_like(source, 36 * (repeat + 1) / 128)
            moe_prefix = reduced + residual
            expected_activation = _reference_mix(moe_prefix, history, score, norm, 8)
            for out, value in zip(inputs, moe_sources, strict=True):
                out.copy_(value)
            reference_moe = iris_kimi3_moe_tail(
                *inputs,
                moe_prefix,
                weight,
                prefix_is_sharded=False,
                norm_weight=None,
                eps=None,
                group=group,
            )
            assert reference_moe is not None
            expected_moe = reference_moe.clone()
            for _ in range(3):
                graph.replay()
            torch.cuda.synchronize()
            torch.testing.assert_close(results[0], moe_prefix, atol=0, rtol=0)
            torch.testing.assert_close(results[1], reduced, atol=0, rtol=0)
            torch.testing.assert_close(
                results[2],
                moe_prefix[rank * m // 8 : (rank + 1) * m // 8],
                atol=0,
                rtol=0,
            )
            torch.testing.assert_close(
                results[3], expected_activation, atol=1 / 512, rtol=1 / 64
            )
            torch.testing.assert_close(results[4], expected_moe, atol=0, rtol=0)
    assert pointers == tuple(t.data_ptr() for t in buffers)
    dist.destroy_process_group()


@pytest.mark.skipif(
    not current_platform().is_cdna4 or torch.cuda.device_count() < 8,
    reason="requires eight CDNA4 GPUs",
)
def test_attention_mix_reuses_moe_state():
    with socket.socket() as sock:
        sock.bind(("", 0))
        port = sock.getsockname()[1]
    mp.spawn(_attention_worker, args=(port,), nprocs=8, join=True)
