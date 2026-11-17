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

"""Verify the MoE tail that keeps the residual row-sharded and mixes the next input."""

import socket
from contextlib import ExitStack
from datetime import timedelta

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from tokenspeed_kernel.platform import current_platform
from utils import assert_no_triton_compile, is_amd

pytestmark = pytest.mark.skipif(not is_amd(), reason="Iris requires AMD ROCm")


@pytest.fixture(autouse=True)
def _require_iris():
    pytest.importorskip("tokenspeed_kernel.ops.communication._iris")


@pytest.mark.parametrize("rank", (0, 7))
@pytest.mark.parametrize("prefix_is_sharded", (False, True))
def test_moe_mix_gather_codegen(rank, prefix_is_sharded, tmp_path):
    """Peer pushes stay 16-byte write-through stores despite the mix's row layout."""
    from tokenspeed_kernel._triton import gluon, triton
    from tokenspeed_kernel.ops.communication._iris.all_reduce import (
        iris_arch_policy,
    )
    from tokenspeed_kernel.ops.communication._iris.kimi_k3 import allreduce_sharded
    from tokenspeed_kernel.platform import ArchVersion
    from tokenspeed_kernel_amd.ops.gfx950.attention.kda.attn_res import (
        _attn_res_mix_gfx950,
    )

    allreduce_sharded._attn_res_mix_gfx950 = _attn_res_mix_gfx950
    fn = allreduce_sharded.iris_k3moe_mix_push_gather
    constants = {
        "RANK": rank,
        "STRIDE_BLOCK_T": 7168,
        "NUM_VALID_BLOCKS": 4,
        "SCORE_EPS": 1e-6,
        "OUTPUT_EPS": 1e-5,
        "PREFIX_IS_SHARDED": prefix_is_sharded,
        "NUM_PROGRAMS": 128,
        "NUM_WARPS": 4,
        "WAIT_STORES": iris_arch_policy(ArchVersion(9, 5)).wait_stores,
    }
    signature = {}
    for name in fn.arg_names:
        if name in constants:
            continue
        if name.startswith("heap_base_") or name == "STRIDE_BLOCK_N":
            signature[name] = "i64"
        elif name == "LOCAL_ROWS":
            signature[name] = "i32"
        elif name == "ready_flags":
            signature[name] = "*i32"
        else:
            signature[name] = "*bf16"
    source = gluon._runtime.GluonASTSource(
        fn,
        signature,
        constexprs=constants,
        attrs={
            (index,): [["tt.divisibility", 16]]
            for index, name in enumerate(fn.arg_names)
            if signature.get(name, "i32") not in ("i32",)
        },
    )
    kernel = triton.compile(
        source,
        target=triton.backends.compiler.GPUTarget("hip", "gfx950", 64),
        options={"num_warps": 4},
    )
    assembly = kernel.asm["amdgcn"]
    (tmp_path / f"moe-mix-gather-rank{rank}.amdgcn").write_text(assembly)
    lines = assembly.splitlines()
    # Four 2048-element pieces of the 8192-wide row layout to each of 8 peers.
    peer_stores = [
        line for line in lines if "buffer_store" in line and "sc0 sc1" in line
    ]
    assert len(peer_stores) == 32
    assert all("buffer_store_dwordx4" in line for line in peer_stores)
    assert "scratch_" not in assembly


def _worker(rank: int, port: int) -> None:
    device = torch.device(f"cuda:{rank}")
    torch.cuda.set_device(device)
    dist.init_process_group(
        backend="gloo",
        init_method=f"tcp://127.0.0.1:{port}",
        rank=rank,
        world_size=8,
        timeout=timedelta(seconds=180),
    )
    try:
        group = dist.new_group(backend="nccl", timeout=timedelta(seconds=180))
        dist.barrier(group=group, device_ids=[rank])
        _check(rank, device, group)
    finally:
        dist.destroy_process_group()


def _check(rank: int, device: torch.device, group: dist.ProcessGroup) -> None:
    from tokenspeed_kernel.ops.communication._iris.kimi_k3 import (
        prepare_k3_communication,
    )
    from tokenspeed_kernel.ops.communication._iris.kimi_k3.allreduce_sharded import (
        iris_k3_push_gather_partition,
        iris_k3attn_pull_scatter,
        iris_k3attn_push_gather,
        iris_k3moe_mix_push_gather,
        iris_k3moe_pull_scatter,
        k3moe_residual_mix_rows,
    )
    from tokenspeed_kernel.ops.residual import attn_res_fwd
    from tokenspeed_kernel_amd.ops.gfx950.gemm.fp16.largem import (
        gluon_mm_a16w16_prefill_gfx950,
        gluon_mm_a16w16_prefill_skip_rows_gfx950,
        launch_gluon_mm_a16w16_prefill_gfx950,
    )

    k3_state = prepare_k3_communication(
        group=group,
        rank_in_group=rank,
        staged_max_numel=0,
        producer_direct_max_numel=8192 * 10752,
        attnres_max_numel=0,
        attnres_max_rows=0,
        enable_lamport=True,
        sharded_max_rows=8192,
        dtype=torch.bfloat16,
        device=device,
        all_reduce=None,
    )
    state = k3_state.all_reduce
    sharded = k3_state.sharded
    gen = torch.Generator(device=device).manual_seed(4410)

    def bf16(shape, scale=1.0, offset=0.0):
        return (torch.randn(shape, device=device, generator=gen) * scale + offset).to(
            torch.bfloat16
        )

    weight = bf16((7168, 3584), 3584**-0.5)
    norm = (torch.rand((3584,), device=device, generator=gen) + 0.5).bfloat16()
    res_weight = bf16((7168,), 1 / 16)
    rms_weight = (torch.rand((7168,), device=device, generator=gen) + 0.5).bfloat16()
    out_weight = (torch.rand((7168,), device=device, generator=gen) + 0.5).bfloat16()

    qkv_weight = bf16((2112, 7168), 7168**-0.5)
    kda_weight = bf16((6288, 7168), 7168**-0.5)

    def mix_kwargs(blocks, valid_blocks, write_idx, project=None):
        return dict(
            next_weight={None: None, "gather": qkv_weight, "local": kda_weight}[
                project
            ],
            gather_next_projection=project == "gather",
            block_residual=blocks,
            res_weight=res_weight,
            rms_weight=rms_weight,
            mix_eps=1e-6,
            out_norm_weight=out_weight,
            out_norm_eps=1e-5,
            num_valid_blocks=valid_blocks,
            block_write_idx=write_idx,
            shared_reduced=None,
        )

    warmed = set()
    for rows in (848, 1024, 2048, 8192):
        local = rows // 8
        first = rank * local
        routed, shared = state.acquire_outputs(((rows, 3584), (rows, 7168)))
        gen.manual_seed(9001 + rank)
        sources = (bf16(routed.shape, 1 / 8), bf16(shared.shape, 1 / 8))
        gen.manual_seed(9002)
        full_prefix = bf16((rows, 7168))
        blocks = bf16((8, rows, 7168))
        for prefix_is_sharded in (False, True):
            prefix = (
                full_prefix[first : first + local].clone()
                if prefix_is_sharded
                else full_prefix
            )
            for valid_blocks, write_idx, project in (
                (1, -1, None),
                (4, -1, None),
                (8, -1, None),
                (1, 1, None),
                (7, 7, None),
                (4, -1, "gather"),
                (7, 7, "gather"),
                (4, -1, "local"),
                (1, 1, "local"),
            ):
                if project == "local" and rows % 2048:
                    # The column-sharded projection tiles whole 256-row blocks.
                    assert (
                        k3_state.moe_tail_mix(
                            routed,
                            shared,
                            prefix,
                            weight,
                            prefix_is_sharded=prefix_is_sharded,
                            norm_weight=norm,
                            eps=1e-6,
                            **mix_kwargs(blocks, valid_blocks, write_idx, project),
                        )
                        is None
                    )
                    continue
                routed.copy_(sources[0])
                shared.copy_(sources[1])
                output = k3_state.moe_tail(
                    routed,
                    shared,
                    prefix,
                    weight,
                    prefix_is_sharded=prefix_is_sharded,
                    norm_weight=norm,
                    eps=1e-6,
                    shared_reduced=None,
                ).clone()
                history = blocks.clone()
                expected = attn_res_fwd(
                    output,
                    history if write_idx >= 0 else history[:valid_blocks],
                    res_weight,
                    rms_weight,
                    1e-6,
                    out_norm_weight=out_weight,
                    out_norm_eps=1e-5,
                    num_valid_blocks=valid_blocks,
                    block_write_idx=write_idx,
                )
                routed.copy_(sources[0])
                shared.copy_(sources[1])
                expected_blocks = blocks.clone()
                if write_idx >= 0:
                    # Only this rank's snapshot rows change.
                    expected_blocks[write_idx, first : first + local] = output[
                        first : first + local
                    ]
                if rank == (rows // 8 + valid_blocks) % 8:
                    torch.cuda._sleep(100_000)
                variant = (prefix_is_sharded, valid_blocks, project)
                with ExitStack() as stack:
                    if variant in warmed:
                        for kernel in (
                            iris_k3moe_pull_scatter,
                            iris_k3moe_mix_push_gather,
                            k3moe_residual_mix_rows,
                            iris_k3attn_push_gather,
                            iris_k3_push_gather_partition,
                            gluon_mm_a16w16_prefill_gfx950,
                            gluon_mm_a16w16_prefill_skip_rows_gfx950,
                        ):
                            stack.enter_context(assert_no_triton_compile(kernel))
                    result = k3_state.moe_tail_mix(
                        routed,
                        shared,
                        prefix,
                        weight,
                        prefix_is_sharded=prefix_is_sharded,
                        norm_weight=norm,
                        eps=1e-6,
                        **mix_kwargs(blocks, valid_blocks, write_idx, project),
                    )
                warmed.add(variant)
                assert result is not None
                residual, activation, projection = result
                if project == "gather":
                    assert projection.data_ptr() == sharded.qkv_output.data_ptr()
                    # Each owner projected its rows with the GEMM of its own row count.
                    for owner in range(8):
                        owned = slice(owner * local, (owner + 1) * local)
                        torch.testing.assert_close(
                            projection[owned],
                            torch.nn.functional.linear(
                                expected[owned].contiguous(), qkv_weight
                            ),
                            atol=0,
                            rtol=0,
                        )
                elif project == "local":
                    # Own rows first, the rest after the gather: the large-M GEMM's bits.
                    torch.testing.assert_close(
                        projection,
                        launch_gluon_mm_a16w16_prefill_gfx950(
                            expected, kda_weight, torch.bfloat16
                        ),
                        atol=0,
                        rtol=0,
                    )
                else:
                    assert projection is None
                assert activation.data_ptr() == sharded.output.data_ptr()
                assert residual.shape == (local, 7168)
                torch.testing.assert_close(
                    activation.view(torch.int16),
                    expected.view(torch.int16),
                    atol=0,
                    rtol=0,
                )
                torch.testing.assert_close(
                    residual, output[first : first + local], atol=0, rtol=0
                )
                if write_idx >= 0:
                    assert residual.data_ptr() == blocks[write_idx, first].data_ptr()
                torch.testing.assert_close(blocks, expected_blocks, atol=0, rtol=0)
                if write_idx >= 0:
                    # Refresh stale peer rows: later checks read every row.
                    blocks[write_idx].copy_(output)
                if prefix_is_sharded:
                    torch.testing.assert_close(
                        prefix, full_prefix[first : first + local], atol=0, rtol=0
                    )

        # A residual holding only this rank's rows reduces like the full one.
        (partial,) = state.acquire_outputs(((rows, 7168),))
        gen.manual_seed(9100 + rank)
        attn_source = bf16((rows, 7168), 1 / 8)
        attn_kwargs = dict(
            eps=1e-6,
            out_norm_weight=out_weight,
            out_norm_eps=1e-5,
            num_valid_blocks=4,
        )
        partial.copy_(attn_source)
        shard_ref, mixed_ref = k3_state.attention_mix(
            partial,
            full_prefix,
            blocks,
            res_weight,
            rms_weight,
            residual_is_sharded=False,
            published=None,
            defer_gather=False,
            **attn_kwargs,
        )
        mixed_ref = mixed_ref.clone()
        partial.copy_(attn_source)
        with ExitStack() as stack:
            if rows != 848:
                stack.enter_context(assert_no_triton_compile(iris_k3attn_pull_scatter))
            sharded_result = k3_state.attention_mix(
                partial,
                full_prefix[first : first + local].clone(),
                blocks,
                res_weight,
                rms_weight,
                residual_is_sharded=True,
                published=None,
                defer_gather=False,
                **attn_kwargs,
            )
        assert sharded_result is not None
        torch.testing.assert_close(sharded_result[0], shard_ref, atol=0, rtol=0)
        torch.testing.assert_close(sharded_result[1], mixed_ref, atol=0, rtol=0)
        assert (
            k3_state.attention_mix(
                partial,
                full_prefix,
                blocks,
                res_weight,
                rms_weight,
                residual_is_sharded=True,
                published=None,
                defer_gather=False,
                **attn_kwargs,
            )
            is None
        )

    # The captured tail replays new partials against the same storage.
    rows = 1024
    local = rows // 8
    first = rank * local
    routed, shared = state.acquire_outputs(((rows, 3584), (rows, 7168)))
    gen.manual_seed(9200 + rank)
    sources = (bf16(routed.shape, 1 / 8), bf16(shared.shape, 1 / 8))
    gen.manual_seed(9201)
    prefix = bf16((local, 7168))
    blocks = bf16((8, rows, 7168))
    kwargs = dict(prefix_is_sharded=True, norm_weight=norm, eps=1e-6)

    def tail():
        routed.copy_(sources[0])
        shared.copy_(sources[1])
        return k3_state.moe_tail_mix(
            routed, shared, prefix, weight, **kwargs, **mix_kwargs(blocks, 3, -1)
        )

    tail()
    torch.cuda.synchronize()
    dist.barrier()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        captured = tail()
    for sign in (-1, 1, -1):
        for source in sources:
            source.neg_()
        torch.cuda.synchronize()
        dist.barrier()
        reference = tail()
        assert reference[2] is None and captured[2] is None
        reference = (reference[0].clone(), reference[1].clone())
        if rank == 3:
            torch.cuda._sleep(100_000)
        graph.replay()
        torch.testing.assert_close(captured[0], reference[0], atol=0, rtol=0)
        torch.testing.assert_close(captured[1], reference[1], atol=0, rtol=0)

    # Rejected snapshot plans leave every collective's flags untouched.
    torch.cuda.synchronize()
    dist.barrier()
    routed.copy_(sources[0])
    shared.copy_(sources[1])
    flags_before = state.producer_pull.ready_flags.clone()
    gather_flags_before = sharded.gather_flags.clone()
    rejected = (
        mix_kwargs(blocks, 3, 4),
        mix_kwargs(blocks, 8, 8),
        mix_kwargs(blocks, 9, -1),
        mix_kwargs(blocks[:, :-8], 3, -1),
        mix_kwargs(blocks.transpose(0, 1).contiguous().transpose(0, 1), 3, -1),
        mix_kwargs(sharded.output[: 8 * rows].view(8, rows, 7168)[:, :rows], 3, -1),
        {**mix_kwargs(blocks, 3, -1), "res_weight": res_weight.float()},
        {**mix_kwargs(blocks, 3, -1), "mix_eps": float("nan")},
        {**mix_kwargs(blocks, 3, -1), "out_norm_eps": 0.0},
        {**mix_kwargs(blocks, 3, -1, "gather"), "next_weight": qkv_weight[:-1]},
        {**mix_kwargs(blocks, 3, -1, "gather"), "next_weight": qkv_weight.float()},
        {**mix_kwargs(blocks, 3, -1, "local"), "next_weight": kda_weight[:, :-8]},
        {
            **mix_kwargs(blocks, 3, -1, "gather"),
            "next_weight": sharded.qkv_output[:2112],
        },
    )
    for plan in rejected:
        assert (
            k3_state.moe_tail_mix(routed, shared, prefix, weight, **kwargs, **plan)
            is None
        )
    assert (
        k3_state.moe_tail_mix(
            routed,
            shared,
            blocks[3, first : first + local],
            weight,
            **kwargs,
            **mix_kwargs(blocks, 3, 3),
        )
        is None
    )
    torch.testing.assert_close(state.producer_pull.ready_flags, flags_before)
    torch.testing.assert_close(sharded.gather_flags, gather_flags_before)
    dist.barrier()


@pytest.mark.skipif(
    not current_platform().is_cdna4 or torch.cuda.device_count() < 8,
    reason="Iris MoE tail assigns token rows across eight CDNA4 GPUs",
)
def test_iris_moe_tail_mix() -> None:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        port = sock.getsockname()[1]
    mp.spawn(_worker, args=(port,), nprocs=8, join=True)


@pytest.mark.skipif(
    not current_platform().is_cdna4, reason="the K3 Iris state targets CDNA4"
)
def test_gather_stays_on_the_capture_stream() -> None:
    """Captured gathers run on the capture stream; eager ones on one side stream."""
    from tokenspeed_kernel.ops.communication._iris.kimi_k3.allreduce_sharded import (
        IrisAllReduceShardedState,
    )

    state = IrisAllReduceShardedState.__new__(IrisAllReduceShardedState)
    state.gather_stream = None
    device = torch.device("cuda")
    scratch = torch.zeros(1, device=device)
    captured = []
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        scratch.add_(1)
        captured.append(
            state._gather_side_stream(device) == torch.cuda.current_stream(device)
        )
    assert captured == [True]
    side = state.gather_stream
    assert side is not None
    assert state._gather_side_stream(device) == side
    assert side != torch.cuda.current_stream(device)
