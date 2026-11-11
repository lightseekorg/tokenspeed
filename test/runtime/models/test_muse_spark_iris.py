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

"""Muse Spark's communication runtime on the Iris collectives: QKVG and routed
weight ownership, ordinary all-reduce, expert-TP routes and the owned forward."""

import importlib.util
import os
import socket
import sys
import traceback
from functools import partial
from types import SimpleNamespace

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from tokenspeed_kernel.platform import current_platform

sys.path.insert(
    0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
)
from ci_system.ci_register import register_cuda_ci  # noqa: E402

register_cuda_ci(est_time=10, suite="runtime-1gpu")

pytestmark = pytest.mark.skipif(
    not current_platform().is_cdna4, reason="Muse Iris fusions require gfx950"
)


def _kernel_test_utils():
    kernel_test_dir = os.path.join(
        os.path.dirname(
            os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        ),
        "tokenspeed-kernel",
        "test",
    )
    spec = importlib.util.spec_from_file_location(
        "muse_spark_iris_kernel_test_utils", os.path.join(kernel_test_dir, "utils.py")
    )
    utils = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(utils)
    return utils


assert_no_triton_compile = _kernel_test_utils().assert_no_triton_compile


def _get_open_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("", 0))
        return sock.getsockname()[1]


def _skip_if_unsupported(world_size: int, reason_prefix: str) -> None:
    if not torch.cuda.is_available():
        pytest.skip(f"CUDA/ROCm is required for {reason_prefix}")
    if world_size > torch.cuda.device_count():
        pytest.skip(f"Need {world_size} GPUs, have {torch.cuda.device_count()}")
    if not current_platform().is_amd:
        pytest.skip(f"{reason_prefix} only targets AMD ROCm")
    try:
        import iris  # noqa: F401
    except ImportError:
        pytest.skip("iris is not installed")


def _spawn_and_collect(worker_fn, args, world_size: int) -> None:
    error_dict = mp.Manager().dict()
    mp.spawn(
        worker_fn,
        args=args + (error_dict,),
        nprocs=world_size,
        join=True,
    )

    if error_dict:
        raise RuntimeError("\n".join(f"Rank {r}: {e}" for r, e in error_dict.items()))


def _communication_worker(rank, world_size, port, check, backend, error_dict):
    try:
        device = torch.device("cuda", rank)
        torch.cuda.set_device(device)
        dist.init_process_group(
            backend,
            init_method=f"tcp://localhost:{port}",
            rank=rank,
            world_size=world_size,
        )
        check(rank, world_size, device)
    except Exception:
        error_dict[rank] = traceback.format_exc()
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()


def _prepare_muse_backend(world_size, projection, moe, *, expert_tp_size):
    from tokenspeed.runtime.distributed import process_group_manager as pgm
    from tokenspeed.runtime.distributed.comm_backend.registry import (
        initialize_comm_backend,
    )
    from tokenspeed.runtime.models.muse_spark import MuseSparkForConditionalGeneration

    group = tuple(range(world_size))
    pgm.process_group_manager.register_process_group("nccl", group, dist.group.WORLD)
    backend = initialize_comm_backend()
    if projection is None:
        projection = SimpleNamespace(iris_state=None)
    if moe is None:
        moe = SimpleNamespace(
            iris_state=None,
            pre_expert_proj=None,
            post_expert_proj=None,
            down_columns=None,
        )
    layer = SimpleNamespace(
        iris_state=None,
        self_attn=SimpleNamespace(iris_state=None, qkv_proj=projection),
        mlp=moe,
    )
    owner = SimpleNamespace(
        mapping=SimpleNamespace(
            attn=SimpleNamespace(tp_size=world_size, tp_group=group),
            moe=SimpleNamespace(
                tp_ep_group=group,
                tp_size=expert_tp_size,
                ep_size=world_size // expert_tp_size,
            ),
        ),
        config=SimpleNamespace(
            hidden_size=8192, moe_hidden_size=4096, num_experts_per_tok=8
        ),
        model=SimpleNamespace(iris_state=None, layers=[layer]),
    )
    assert MuseSparkForConditionalGeneration.prepare_communication_runtime(owner, 8192)
    state = owner.model.iris_state
    assert MuseSparkForConditionalGeneration.prepare_communication_runtime(owner, 8192)
    assert owner.model.iris_state is state
    assert projection.iris_state is state and moe.iris_state is state
    assert backend.get_prepared_all_gather(group, torch.bfloat16) is state.gather
    return group, state


def _muse_owned_qkvg_worker(rank, world_size, device):
    torch.set_grad_enabled(False)
    torch.set_default_dtype(torch.bfloat16)
    from tokenspeed_kernel.ops.attention.prologue import (
        HeadKVCache,
        HeadNorm,
        RopeStyle,
        Rotary,
        gqa_prologue,
    )
    from tokenspeed_kernel.ops.communication._iris.muse_spark.all_to_all import (
        iris_a2a_transpose_qkvg,
        iris_fused_a2a_transpose_qkvg_norm_rope,
    )

    from tokenspeed.runtime.models.muse_spark import (
        MUSE_OWNED_QKVG_MIN_TOKENS,
        QKVOGateParallelLinear,
    )

    group = tuple(range(world_size))
    with torch.device(device):
        projection = QKVOGateParallelLinear(
            hidden_size=8192,
            head_size=64,
            total_num_heads=128,
            total_num_kv_heads=16,
            quant_config=None,
            prefix="model.layers.1.self_attn.qkv_proj",
            tp_rank=rank,
            tp_size=world_size,
            tp_group=group,
            replicate_for_owned_rows=True,
        )
    group, state = _prepare_muse_backend(world_size, projection, None, expert_tp_size=1)
    full_weights = {}
    expected_columns = []
    for index, (name, width) in enumerate(
        (("q", 8192), ("k", 1024), ("v", 1024), ("g", 8192))
    ):
        weight = torch.zeros(width, 8192, device=device)
        columns = (torch.arange(width, device=device) * 13 + index * 17) % 8192
        weight[torch.arange(width, device=device), columns] = 1
        full_weights[name] = weight
        projection.weight_loader(projection.weight, weight, name)
        expected_columns.append(
            columns[rank * (width // world_size) : (rank + 1) * (width // world_size)]
        )
    expected_columns = torch.cat(expected_columns)
    parameter_pointer = projection.weight.data_ptr()
    local_pointer = projection._local_weight().data_ptr()

    angles = torch.arange(16384, device=device, dtype=torch.float32)[:, None] / (
        torch.arange(32, device=device, dtype=torch.float32)[None, :] + 1
    )
    rope = torch.cat((angles.cos(), angles.sin()), dim=1)
    nope = torch.empty((1, 0), device=device, dtype=torch.float32)
    eps, q_scale = 1e-5, 5.4730064863875
    norm = HeadNorm(
        torch.full((64,), q_scale, device=device, dtype=torch.float32),
        torch.ones((64,), device=device, dtype=torch.float32),
        0.0,
        eps,
    )

    def make_inputs(rows):
        local_rows = (rows + world_size - 1) // world_size
        first = rank * local_rows
        valid = min(local_rows, rows - first)
        full = (
            (
                torch.arange(rows, device=device)[:, None] % 13
                + torch.arange(8192, device=device)[None, :] % 7
            )
            / 16
        ).bfloat16()
        local = torch.full((local_rows, 8192), float("nan"), device=device)
        local[:valid].copy_(full[first : first + valid])
        poison = torch.full((rows, 8192), rank + 1, dtype=torch.bfloat16, device=device)
        positions = (
            torch.arange(rows, device=device, dtype=torch.int64) * 17 + 31
        ) % 16384
        k_cache = torch.empty(
            (rows, 16 // world_size, 64), dtype=torch.bfloat16, device=device
        )
        cache = HeadKVCache(
            k_cache,
            torch.empty_like(k_cache),
            None,
            torch.arange(rows, device=device, dtype=torch.int64),
        )
        return full, local, poison, positions, cache

    inputs = [make_inputs(rows) for rows in (2047, 2048, 8191)]

    def run(item):
        full, local, poison, positions, cache = item
        generic = state.all_reduce.acquire_outputs(
            ((state.reduction.push_max_numel + 8192,),)
        )[0]
        generic.fill_(rank + 1)
        previous = state.reduce_outputs((generic,))[0]
        fused = reference = None
        if full.shape[0] >= MUSE_OWNED_QKVG_MIN_TOKENS:
            projected = projection.project_owned_rows(local, full.shape[0])
            output = state.all_to_all_transpose_qkvg(projected, full.shape[0])
            table = rope if full.shape[0] < 4096 else nope
            q, k, v, gate = output.split(
                tuple(width // world_size for width in (8192, 1024, 1024, 8192)), dim=-1
            )
            prepared = gqa_prologue(
                q,
                k,
                v,
                norm=norm,
                rotary=(
                    Rotary(table, positions, RopeStyle.GPTJ, None)
                    if table.shape[1]
                    else None
                ),
                cache=cache,
                query_dtype=q.dtype,
                return_kv=True,
                solution="triton",
                override=None,
            )
            reference = (prepared.q, prepared.k, prepared.v, gate)
            projected = projection.project_owned_rows(local, full.shape[0])
            if rank == full.shape[0] % world_size:
                torch.cuda._sleep(20000)
            fused = state.all_to_all_transpose_qkvg_norm_rope(
                projected, full.shape[0], positions, table, eps, q_scale
            )
        else:
            normalized = state.all_gather_owned_rows(local, full.shape[0], True)
            output, _ = projection(normalized)
        # O's next producer can overwrite every old QKVG source address.
        reduced = state.reduce_scatter_owned_rows(poison)
        gathered = state.all_gather_owned_rows(reduced, full.shape[0], True)
        return output, gathered, previous, fused, reference

    def sequence():
        return [run(item) for item in inputs]

    sequence()
    state.row_gather_flags.fill_(2**32 - 2)
    torch.cuda.synchronize()
    dist.barrier()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        outputs = sequence()
    for replay in range(4):
        gain = 2 ** (replay + 1)
        for name, weight in full_weights.items():
            weight.mul_(2)
            projection.weight_loader(projection.weight, weight, name)
        assert projection.weight.data_ptr() == parameter_pointer
        assert projection._local_weight().data_ptr() == local_pointer
        for full, local, _, positions, _ in inputs:
            positions.add_(97).remainder_(16384)
            full.add_(1 / 16)
            first = rank * local.shape[0]
            valid = min(local.shape[0], full.shape[0] - first)
            local[:valid].copy_(full[first : first + valid])
        if rank == replay:
            torch.cuda._sleep(10000)
        graph.replay()
        for (full, *_), (output, gathered, previous, fused, reference) in zip(
            inputs, outputs
        ):
            if fused is not None:
                for actual, expected in zip(fused, reference):
                    assert actual.is_contiguous() and bool(torch.isfinite(actual).all())
                    torch.testing.assert_close(
                        actual.view(torch.int16),
                        expected.view(torch.int16),
                        rtol=0,
                        atol=0,
                    )
            expected = full.index_select(1, expected_columns) * gain
            torch.testing.assert_close(output, expected, rtol=0, atol=0)
            torch.testing.assert_close(
                gathered,
                torch.full_like(gathered, world_size * (world_size + 1) // 2),
                rtol=0,
                atol=0,
            )
            torch.testing.assert_close(
                previous,
                torch.full_like(previous, world_size * (world_size + 1) // 2),
                rtol=0,
                atol=0,
            )
        small = inputs[0][0][:4].contiguous()
        small_output, _ = projection(small)
        torch.testing.assert_close(
            small_output,
            small.index_select(1, expected_columns) * gain,
            rtol=0,
            atol=0,
        )
    with assert_no_triton_compile(
        iris_a2a_transpose_qkvg, iris_fused_a2a_transpose_qkvg_norm_rope
    ):
        for rows in (2051, 8190):
            output, _, _, fused, reference = run(make_inputs(rows))
            for actual, expected in zip(fused, reference):
                torch.testing.assert_close(
                    actual.view(torch.int16), expected.view(torch.int16), rtol=0, atol=0
                )
    torch.cuda.synchronize()
    dist.barrier()


def _muse_tp_routes_worker(rank, world_size, device):
    """Expert TP sums aligned partial routes before nonlinear normalization."""

    from tokenspeed.runtime.models.muse_spark import MuseSparkMoE, MuseSparkRMSNorm

    torch.set_grad_enabled(False)
    width, topk = 4096, 8
    group = tuple(range(world_size))
    moe = MuseSparkMoE.__new__(MuseSparkMoE)
    torch.nn.Module.__init__(moe)
    moe.iris_state = None
    moe.pre_expert_proj = None
    moe.post_expert_proj = None
    moe.down_columns = None
    moe.tp_group, moe.tp_size, moe.topk = group, world_size, topk
    moe.ep_group, moe.ep_size = (rank,), 1
    moe.post_expert_norm = MuseSparkRMSNorm(width, 1e-8, True, 1.0)
    norm_weight = torch.linspace(-0.25, 0.25, width, device=device).bfloat16()
    moe.post_expert_norm.weight = torch.nn.Parameter(norm_weight)
    _, state = _prepare_muse_backend(world_size, None, moe, expert_tp_size=world_size)
    dims = torch.arange(width, device=device)[None, :]

    # Full-row inputs exceed generic staging; owned inputs keep complete token
    # route groups together, including an uneven last owner and mapped outputs.
    cases = [
        (n, c, False) for n, c in ((1, False), (3, False), (16, False), (2049, False))
    ]
    cases += [(n, True, False) for n in (3, 6, 33, 44, 49, 128)]
    cases += [(64, False, False)]
    cases += [
        (192, True, True),
        (193, True, True),
        (193, False, True),
        (2049, True, True),
        (8191, False, True),
        (8192, True, True),
        (2051, True, True),
    ]
    for tokens, canonical, owned in cases:
        indices = torch.arange(tokens * topk, device=device)
        mapping = None if canonical else ((indices + rank) % indices.numel()).int()
        weights = (
            torch.linspace(0.01, 1, topk, device=device).expand(tokens, -1).contiguous()
        )
        # Decode rows completed on token owners have the expert write the
        # route staging buffer directly.
        scattered = not owned and canonical
        scattered = (
            scattered and state.can_reduce_scatter_routed_rmsnorm_sum_all_gather(tokens)
        )
        routes = (
            state.acquire_owned_route_input(tokens)
            if owned and canonical
            else (
                moe._acquire_route_destination(tokens, False)
                if scattered
                else torch.empty(
                    (indices.numel(), width if canonical else width + 16),
                    device=device,
                    dtype=torch.bfloat16,
                )
            )
        )

        def update(step):
            partial = (
                (indices[:, None] * 3 + dims * 5 + rank * 7 + step) % 9 - 4
            ).bfloat16() / 32
            routes[:, :width].copy_(
                partial if canonical else partial.roll(rank, dims=0)
            )
            routes[:, width:].fill_(torch.nan)

        def reference(step):
            local_rows = (tokens + world_size - 1) // world_size
            first = rank * local_rows if owned else 0
            count = min(local_rows, tokens - first) if owned else tokens
            route_indices = indices[first * topk : (first + count) * topk]
            full = sum(
                (
                    (route_indices[:, None] * 3 + dims * 5 + peer * 7 + step) % 9 - 4
                ).float()
                / 32
                for peer in range(world_size)
            ).reshape(count, topk, width)
            scaled = full * (norm_weight + 1.0).float()
            normalized = scaled * torch.rsqrt(
                scaled.square().mean(-1, keepdim=True) + 1e-8
            )
            expected = (
                (normalized * weights[first : first + count, :, None]).sum(1).bfloat16()
            )
            if owned and count < local_rows:
                expected = torch.nn.functional.pad(
                    expected, (0, 0, 0, local_rows - count)
                )
            return expected

        def run():
            return moe._reduce_expert_routes(
                routes.view(tokens, topk, width) if canonical else routes,
                mapping,
                tokens,
                weights,
                owned_rows=owned,
            )

        if scattered:
            assert routes.data_ptr() == state.owned_route_input.data_ptr()
        # Expert TP8 completes smaller canonical decode batches column-owned.
        columned = (
            canonical
            and not owned
            and state.can_reduce_column_routed_rmsnorm_sum(tokens)
        )
        stages = (
            state.column_routes_epochs.clone()
            if state.column_routes_epochs is not None
            else None
        )
        update(0)
        flags = state.owned_route_scatter_flags.clone()
        actual = run()
        assert columned == (
            stages is not None and not torch.equal(stages, state.column_routes_epochs)
        )
        # Only owned rows and the decode window use the route reduce-scatter.
        assert scattered or owned or torch.equal(flags, state.owned_route_scatter_flags)
        assert not scattered or not torch.equal(flags, state.owned_route_scatter_flags)
        torch.testing.assert_close(actual, reference(0), rtol=0, atol=2**-8)
        if tokens == 192 and owned and canonical:
            before = routes.clone()
            rotated = ((indices + 1) % indices.numel()).int()
            with pytest.raises(ValueError, match="must not overlap"):
                state.reduce_scatter_owned_routes(routes, rotated, tokens)
            shifted = state.owned_route_input.flatten()[
                width : width + routes.numel()
            ].view_as(routes)
            with pytest.raises(ValueError, match="must not overlap"):
                state.reduce_scatter_owned_routes(shifted, None, tokens)
            torch.testing.assert_close(routes, before, rtol=0, atol=0)
        if tokens <= 128 or owned:
            if owned:
                state.owned_route_scatter_flags.view(torch.int32).fill_(-2)
                state.all_reduce.peers.ctx.device_barrier()
            torch.cuda.synchronize()
            dist.barrier()
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                captured = run()
            for step in (1, 2):
                update(step)
                if rank == step % world_size:
                    torch.cuda._sleep(20000)
                graph.replay()
                if owned:
                    # A second reduction overwrites borrowed raw-route storage;
                    # the normalization result must retain its own contents.
                    state.reduce_scatter_owned_routes(routes, mapping, tokens)
                torch.testing.assert_close(
                    captured, reference(step), rtol=0, atol=2**-8
                )
        dist.barrier()


def _muse_tp_owned_forward_worker(
    rank, world_size, device, *, quantized: bool, down_columns: bool
):
    """Exercise the real router/pool/gather/expert/unpool caller on uneven rows.

    MXFP4 experts at expert TP8 build the column-split down copy. With
    ``down_columns`` the owned rows take it (assembled from every rank's down
    shard) and must agree with the full-row path within BF16 rounding; without
    it the test drops the copy to check the full-row path.
    """
    from tokenspeed.runtime.configs.muse_spark_config import MuseSparkConfig
    from tokenspeed.runtime.distributed import process_group_manager as pgm
    from tokenspeed.runtime.distributed.comm_backend.registry import (
        initialize_comm_backend,
    )
    from tokenspeed.runtime.distributed.mapping import Mapping
    from tokenspeed.runtime.layers.quantization.mxfp4 import Mxfp4Config
    from tokenspeed.runtime.models.muse_spark import MuseSparkMoE

    torch.set_grad_enabled(False)
    torch.set_default_dtype(torch.bfloat16)
    mapping = Mapping(
        rank=rank,
        world_size=world_size,
        attn_tp_size=world_size,
        dense_tp_size=world_size,
        moe_tp_size=world_size,
        moe_ep_size=1,
        vision_tp_size=world_size,
        linear_attn_tp_size=world_size,
        nprocs_per_node=world_size,
        nnodes=1,
    )
    group = mapping.attn.tp_group
    pgm.process_group_manager.register_process_group("nccl", group, dist.group.WORLD)
    initialize_comm_backend()
    config = MuseSparkConfig(
        text_config=dict(
            vocab_size=256,
            hidden_size=8192,
            head_dim=64,
            num_attention_heads=128,
            num_key_value_heads=16,
            num_hidden_layers=1,
            max_position_embeddings=8192,
            rms_norm_eps=1e-5,
            post_norm_eps=1e-8,
            final_logit_softcapping=20.0,
            qk_scale_factor=5.47,
            output_multiplier=0.125,
            residual_gate_temperature=0.3,
            moe_hidden_size=4096,
            moe_intermediate_size=4096 if quantized else 256,
            num_local_experts=256 if quantized else 8,
            num_experts_per_tok=8,
            num_shared_experts=0,
            routed_scaling_factor=1.0,
            layer_types=["full_attention"],
            layer_rope_theta=[0.0],
            mlp_layer_types=["sparse"],
            sliding_window=128,
            rope_parameters=dict(rope_theta=500000.0, rope_type="default"),
        ),
        vision_config={},
        audio_config={},
        image_token_id=200,
        video_token_id=201,
        audio_token_id=202,
        out_hidden_size=32,
        projector_hidden_size=32,
        projector_hidden_act="gelu",
    )
    quant_config = (
        Mxfp4Config(
            is_checkpoint_mxfp4_serialized=True,
            use_dynamic_mxfp4_activations=True,
        )
        if quantized
        else None
    )
    with torch.device(device):
        moe = MuseSparkMoE(config, mapping, quant_config, 0, "model.layers.0.mlp", None)
    assert (moe.down_columns is not None) == quantized
    if not down_columns:
        moe.down_columns = None
    torch.manual_seed(117)
    for parameter in moe.parameters():
        if parameter.is_floating_point():
            parameter.normal_(0, 0.02)
    # Sparse pool/unpool weights make the reference insensitive to GEMM tiling.
    for projection in (moe.pre_expert_proj, moe.post_expert_proj):
        projection.weight.zero_()
        index = torch.arange(projection.output_size, device=device)
        projection.weight[index, (index * 3) % projection.input_size] = 1
    torch.manual_seed(117 + rank)
    if quantized:
        moe.experts.w13_weight.random_(0, 256)
        moe.experts.w2_weight.random_(0, 256)
        moe.experts.w13_weight_scale.random_(118, 124)
        moe.experts.w2_weight_scale.random_(118, 124)
        if down_columns:
            # Assemble the full down projection from every rank's K shard.
            for local, scale in (
                (moe.experts.w2_weight, False),
                (moe.experts.w2_weight_scale, True),
            ):
                shards = [torch.empty_like(local) for _ in range(world_size)]
                dist.all_gather(shards, local.data.contiguous())
                moe.down_columns.load(
                    torch.cat(shards, dim=-1), scale=scale, expert=None
                )
            moe.down_columns.process_weights_after_loading(moe.down_columns)
        moe.experts.process_weights_after_loading(moe.experts)
    else:
        moe.experts.w13_weight.normal_(0, 0.02)
        moe.experts.w2_weight.normal_(0, 0.02)
    moe.gate.process_weights_after_loading(moe.gate)
    _, state = _prepare_muse_backend(world_size, None, moe, expert_tp_size=world_size)
    for rows in (192, 193, 1029):
        local_rows = (rows + world_size - 1) // world_size
        first = rank * local_rows
        valid = min(local_rows, rows - first)
        for step in (0, 1):
            torch.manual_seed(rows + step)
            full = torch.randn(rows, 8192, device=device)
            local = torch.zeros(local_rows, 8192, device=device)
            local[:valid].copy_(full[first : first + valid])
            expected_rows, _, _ = moe(
                full, full.float(), rows, rows, None, project=True
            )
            expected_rows = expected_rows.clone()
            expected = expected_rows[first : first + valid]
            actual, _, _ = moe(local, local.float(), rows, rows, rows, project=True)
            actual = actual.clone()
            # Post projection borrows the generic producer; retain it before reuse.
            state.all_reduce.producer.data.fill_(torch.nan)
            torch.testing.assert_close(actual[:valid], expected, atol=2**-7, rtol=2**-7)
            torch.testing.assert_close(
                actual[valid:], torch.zeros_like(actual[valid:]), atol=0, rtol=0
            )
            if quantized and rows == 192:
                # Decode batches route and quantize in two launches; each row
                # matches the same row of the 192-row batch, too large for them.
                for small in (8, 16):
                    rows_in = full[:small].contiguous()
                    router_in = rows_in.float()
                    assert (
                        moe._route_and_quantize_inputs(rows_in, router_in) is not None
                    )
                    decode, _, _ = moe(rows_in, router_in, small, small, None, True)
                    torch.testing.assert_close(
                        decode, expected_rows[:small], atol=2**-7, rtol=2**-7
                    )
        dist.barrier()


def _muse_owned_routed_wt_worker(rank, world_size, device):
    torch.set_grad_enabled(False)
    from tokenspeed_kernel.ops.communication._iris.muse_spark.routed_moe import (
        iris_routed_rmsnorm_weighted_sum_write_through,
    )
    from tokenspeed_kernel.ops.communication._iris.reduce_scatter import (
        iris_pull_reduce_scatter,
    )
    from tokenspeed_kernel.ops.moe import (
        routed_input_scale_rmsnorm_sum,
        routed_input_scale_rmsnorm_sum_gathered,
    )
    from tokenspeed_kernel.platform import pdl_enabled

    from tokenspeed.runtime.models.muse_spark import MuseSparkMoE, MuseSparkRMSNorm

    group = tuple(range(world_size))
    torch.manual_seed(181)
    norm = torch.randn(4096, device=device, dtype=torch.bfloat16) / 8
    moe = MuseSparkMoE.__new__(MuseSparkMoE)
    torch.nn.Module.__init__(moe)
    moe.iris_state = None
    moe.pre_expert_proj = None
    moe.post_expert_proj = None
    moe.down_columns = None
    moe.ep_group, moe.ep_size, moe.topk = group, world_size, 8
    moe.tp_size = 1
    moe.post_expert_norm = MuseSparkRMSNorm(4096, 1e-8, True, 1.0)
    moe.post_expert_norm.weight = torch.nn.Parameter(norm)
    group, state = _prepare_muse_backend(world_size, None, moe, expert_tp_size=1)

    def inputs(rows):
        torch.manual_seed(rows + rank * 17)
        original = torch.randn(rows * 8, 4096, device=device, dtype=torch.bfloat16)
        local_rows = (rows + world_size - 1) // world_size
        return dict(
            rows=rows,
            original=original,
            routes=torch.empty_like(original),
            index=torch.arange(rows * 8, device=device, dtype=torch.int32),
            mapping=torch.empty(rows * 8, device=device, dtype=torch.int32),
            weights=torch.full((rows, 8), 0.125, device=device),
            packet=torch.empty(local_rows, 4096, device=device, dtype=torch.bfloat16),
            ids=torch.zeros(local_rows, 8, device=device, dtype=torch.int32),
            scores=torch.ones(local_rows, 8, device=device),
            attention=torch.empty(rows, 8192, device=device, dtype=torch.bfloat16),
        )

    def update(case, step):
        owner = (case["index"] + step) % (world_size - 1 if step == 2 else world_size)
        case["mapping"].copy_(torch.where(owner == rank, case["index"], -1))
        case["routes"].copy_(
            torch.where(
                (owner == rank)[:, None],
                case["original"] * (1 + step / 8),
                torch.nan,
            )
        )
        case["packet"].fill_((rank + 1) * (step + 1) / 8)
        case["attention"].fill_((rank + 1) * (step + 1) / 16)

    def reference(routes, mapping, rows, weights):
        norm = moe.post_expert_norm
        if mapping is None:
            combined = routed_input_scale_rmsnorm_sum(
                routes,
                weights,
                norm.weight,
                norm.eps,
                norm.weight_offset,
                pdl_enabled(),
                None,
            )
            output = state.all_reduce.acquire_outputs((tuple(combined.shape),))[0]
            output.copy_(combined)
        else:
            output = state.all_reduce.acquire_outputs(((rows, 4096),))[0]
            routed_input_scale_rmsnorm_sum_gathered(
                routes,
                mapping,
                rows,
                8,
                weights,
                norm.weight,
                norm.eps,
                norm.weight_offset,
                pdl_enabled(),
                output,
            )
        return state.reduce_scatter_owned_rows(output).clone()

    def run(case):
        rows = case["rows"]
        expected = reference(case["routes"], case["mapping"], rows, case["weights"])
        producer = state.all_reduce.acquire_outputs(((rows, 8192),))[0]
        producer.copy_(case["attention"])
        state.reduce_scatter_owned_rows(producer)
        local_rows = (rows + world_size - 1) // world_size
        valid = max(0, min(local_rows, rows - rank * local_rows))
        packet, _, _ = state.gather_owned_moe_inputs(
            case["packet"], case["ids"], case["scores"], rows, valid
        )
        if rank == rows % world_size:
            torch.cuda._sleep(20000)
        actual = moe._reduce_expert_routes(
            case["routes"], case["mapping"], rows, case["weights"], owned_rows=True
        ).clone()
        upper = state.acquire_owned_post_expert_output(rows, 8192)
        upper[:, :4096].copy_(actual)
        upper[:, 4096:].copy_(actual * 2)
        full = state.all_gather_owned_rows(upper, rows, False)
        return expected, actual, packet, full

    cases = [inputs(rows) for rows in (192, 193, 8191)]
    for case in cases:
        update(case, 0)
        run(case)
    for flags in (state.row_scatter_flags, state.row_gather_flags):
        flags.view(torch.int32).fill_(-2)
    state.all_reduce.peers.ctx.device_barrier()
    torch.cuda.synchronize()
    dist.barrier()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        saved = [run(case) for case in cases]
    for step in range(4):
        for case in cases:
            update(case, step)
        graph.replay()
        for case, (expected, actual, packet, full) in zip(cases, saved):
            torch.testing.assert_close(
                actual.view(torch.int16), expected.view(torch.int16), rtol=0, atol=0
            )
            rows = case["rows"]
            local_rows = (rows + world_size - 1) // world_size
            valid = max(0, min(local_rows, rows - rank * local_rows))
            torch.testing.assert_close(
                actual[valid:], torch.zeros_like(actual[valid:]), rtol=0, atol=0
            )
            owner = torch.arange(rows, device=device) // local_rows
            torch.testing.assert_close(
                packet,
                ((owner + 1) * (step + 1) / 8)
                .to(torch.bfloat16)[:, None]
                .expand_as(packet),
                rtol=0,
                atol=0,
            )
            first = rank * local_rows
            torch.testing.assert_close(
                full[first : first + valid, :4096], actual[:valid], rtol=0, atol=0
            )
            torch.testing.assert_close(
                full[first : first + valid, 4096:],
                (actual * 2)[:valid],
                rtol=0,
                atol=0,
            )
    unseen = inputs(2051)
    update(unseen, 0)
    with assert_no_triton_compile(
        iris_routed_rmsnorm_weighted_sum_write_through, iris_pull_reduce_scatter
    ):
        expected, actual, _, _ = run(unseen)
        torch.testing.assert_close(
            actual.view(torch.int16), expected.view(torch.int16), rtol=0, atol=0
        )
    moe.post_expert_norm.weight_offset = 0.0
    expected, actual, _, _ = run(unseen)
    torch.testing.assert_close(
        actual.view(torch.int16), expected.view(torch.int16), rtol=0, atol=0
    )
    # The unmapped BF16 layer retains the portable producer and release path.
    dense = cases[1]["original"].view(193, 8, 4096)
    expected = reference(dense, None, 193, cases[1]["weights"])
    actual = moe._reduce_expert_routes(
        dense, None, 193, cases[1]["weights"], owned_rows=True
    )
    torch.testing.assert_close(
        actual.view(torch.int16), expected.view(torch.int16), rtol=0, atol=0
    )
    torch.cuda.synchronize()
    dist.barrier()


def _muse_ordinary_all_reduce_worker(rank, world_size, device):
    """Ordinary all-reduces on Muse's group, such as a draft model's per-layer
    reductions, use its prepared reduction: they match RCCL bit for bit, stay
    in caller storage across later collectives and another stream's reads,
    replay from a graph interleaved with Muse's own reductions, and hand
    payloads past the limit, views of state storage and MAX to RCCL or
    staging. Muse's own ordinary reduce takes the same policy."""
    torch.set_grad_enabled(False)
    from tokenspeed_kernel.ops.communication._iris.all_reduce import (
        ProducerAllReduce,
        iris_lamport_bf16_allreduce,
        iris_lamport_twoshot_bf16_allreduce,
        iris_pull_oneshot,
        iris_pull_twoshot,
        iris_push_twoshot_allreduce,
    )

    from tokenspeed.runtime.distributed.comm_backend.registry import (
        get_global_backend,
    )
    from tokenspeed.runtime.distributed.comm_ops import all_reduce

    group, state = _prepare_muse_backend(world_size, None, None, expert_tp_size=1)
    iris = get_global_backend().iris
    generator = torch.Generator(device=device).manual_seed(rank)

    def operands(rows, width=8192):
        # Small integers sum exactly in any order, so every protocol must
        # match RCCL bit for bit. Negative zeros look like the Lamport empty
        # marker and must still be reduced as zeros.
        values = torch.randint(
            -16, 17, (rows, width), generator=generator, device=device
        ).bfloat16()
        values[:, rank::7] = -0.0
        expected = values.clone()
        dist.all_reduce(expected)
        return values, expected

    # Lamport one-shot and (TP8) two-shot windows, the two-shot's largest
    # payload (172 rows) and one row past it, the TP4 push window, the
    # producer pull, a width outside the Lamport windows (the one-shot pull
    # at TP4, staging at TP8), and the largest payload served.
    limit_rows = ProducerAllReduce.ORDINARY_MAX_BYTES[world_size] // (8192 * 2)
    for rows, width in (
        (1, 8192),
        (8, 8192),
        (21, 8192),
        (22, 8192),
        (33, 8192),
        (64, 8192),
        (128, 8192),
        (129, 8192),
        (172, 8192),
        (173, 8192),
        (limit_rows, 8192),
        (5, 4096),
        (1, 768),
    ):
        values, expected = operands(rows, width)
        assert iris.can_run(values, group)
        result = all_reduce(values, group)
        assert result.data_ptr() == values.data_ptr()
        torch.testing.assert_close(result, expected, atol=0, rtol=0)

    values, expected = operands(limit_rows + 1)
    assert not state.reduction.can_reduce_ordinary(values, rccl_fallback=True)
    assert not iris.can_run(values, group)
    torch.testing.assert_close(all_reduce(values, group), expected, atol=0, rtol=0)
    # Muse's own reduce has no RCCL fallback and serves it through the policy.
    values, expected = operands(limit_rows + 1)
    torch.testing.assert_close(state.reduce(values), expected, atol=0, rtol=0)
    values, expected = operands(64)
    torch.testing.assert_close(state.reduce(values), expected, atol=0, rtol=0)

    values, _ = operands(8)
    maximum = values.clone()
    dist.all_reduce(maximum, op=dist.ReduceOp.MAX)
    assert not iris.can_run(values, group, op=dist.ReduceOp.MAX)
    torch.testing.assert_close(
        all_reduce(values, group, op=dist.ReduceOp.MAX), maximum, atol=0, rtol=0
    )

    # A reduce-scatter result lives in pull scratch; reducing it in place
    # through the pull would overwrite it, so the policy declines it.
    full, _ = operands(192)
    owned = state.reduce_scatter_owned_rows(full)
    # Refill the view with small integers so the reduction stays exact.
    values, expected = operands(owned.shape[0])
    owned.copy_(values)
    assert not state.reduction.can_reduce_ordinary(owned, rccl_fallback=True)
    torch.testing.assert_close(all_reduce(owned, group), expected, atol=0, rtol=0)

    # Reduce Lamport-sized and pull-sized payloads, then keep the model's
    # own reduction and more ordinary ones running on the main stream while
    # another stream, delayed, copies the first two results. A result that
    # borrowed state storage would be overwritten by then.
    aux = torch.cuda.Stream(device)
    # The last two sizes take the Lamport one-shot over 48 tiles, while
    # Muse's 5-row reduction between them uses 80 tiles of the same mailbox.
    shapes = ((64, 8192), (172, 8192), (8, 8192), (3, 8192))
    sources = [
        torch.empty(shape, dtype=torch.bfloat16, device=device) for shape in shapes
    ]
    reduced = [torch.empty_like(source) for source in sources]
    copies = [torch.empty_like(source) for source in sources[:2]]
    target_source = torch.empty((26, 8192), dtype=torch.bfloat16, device=device)
    fork = torch.cuda.Event()
    join = torch.cuda.Event()

    def step():
        first = all_reduce(reduced[0], group)
        fork.record()
        with torch.cuda.stream(aux):
            aux.wait_event(fork)
            torch.cuda._sleep(50000)
            copies[0].copy_(first)
        second = all_reduce(reduced[1], group)
        branch = state.all_reduce.acquire_outputs(((26, 8192),))[0]
        branch.copy_(target_source)
        target = state.reduce_outputs((branch,))[0].clone()
        all_reduce(reduced[2], group)
        small = state.all_reduce.acquire_outputs(((5, 8192),))[0]
        small.copy_(target_source[:5])
        small_target = state.reduce_outputs((small,))[0].clone()
        all_reduce(reduced[3], group)
        fork.record()
        with torch.cuda.stream(aux):
            aux.wait_event(fork)
            copies[1].copy_(second)
            join.record()
        torch.cuda.current_stream().wait_event(join)
        return target, small_target

    def load(step_index):
        expected = []
        for source, destination in zip(sources, reduced):
            values, total = operands(source.shape[0])
            source.copy_(values)
            destination.copy_(values)
            expected.append(total)
        target_source.fill_((rank + 1) * (step_index + 1))
        return expected

    def check(expected, targets, step_index):
        for actual, total in zip(reduced, expected):
            torch.testing.assert_close(actual, total, atol=0, rtol=0)
        for actual, total in zip(copies, expected):
            torch.testing.assert_close(actual, total, atol=0, rtol=0)
        for target in targets:
            torch.testing.assert_close(
                target,
                torch.full_like(
                    target, world_size * (world_size + 1) // 2 * (step_index + 1)
                ),
                atol=0,
                rtol=0,
            )

    expected = load(0)
    check(expected, step(), 0)
    torch.cuda.synchronize()
    dist.barrier()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        targets = step()
    for step_index in range(6):
        expected = load(step_index)
        torch.cuda.synchronize()
        if rank == step_index % world_size:
            torch.cuda._sleep(20000)
        graph.replay()
        check(expected, targets, step_index)

    # Sizes between and around the ones above reuse the compiled kernels: the
    # pulls take their workgroup and tile counts unspecialized.
    with assert_no_triton_compile(
        iris_lamport_bf16_allreduce,
        iris_lamport_twoshot_bf16_allreduce,
        iris_pull_oneshot,
        iris_pull_twoshot,
        iris_push_twoshot_allreduce,
    ):
        for rows, width in (
            (3, 8192),
            (17, 8192),
            (40, 8192),
            (80, 8192),
            (100, 8192),
            (136, 8192),
            (150, 8192),
            (200, 8192),
            (300, 8192),
            (3, 768),
        ):
            values, expected = operands(rows, width)
            torch.testing.assert_close(
                all_reduce(values, group), expected, atol=0, rtol=0
            )
    torch.cuda.synchronize()
    dist.barrier()


@pytest.mark.parametrize(
    "check,backend",
    [
        pytest.param(_muse_owned_qkvg_worker, "nccl", id="qkvg_reload_and_receiver"),
        pytest.param(_muse_owned_routed_wt_worker, "nccl", id="owned_routed_wt"),
        pytest.param(
            _muse_ordinary_all_reduce_worker, "nccl", id="ordinary_all_reduce"
        ),
        pytest.param(_muse_tp_routes_worker, "nccl", id="tp_routes"),
        pytest.param(
            partial(_muse_tp_owned_forward_worker, quantized=False, down_columns=False),
            "nccl",
            id="tp_owned_forward",
        ),
    ],
)
@pytest.mark.parametrize("world_size", [4, 8])
def test_muse_runtime_collective_replay(check, backend, world_size):
    _skip_if_unsupported(world_size, "Muse collective replay")
    _spawn_and_collect(
        _communication_worker,
        (world_size, _get_open_port(), check, backend),
        world_size,
    )


@pytest.mark.parametrize("down_columns", [False, True])
def test_muse_tp8_owned_a4w4_forward(down_columns):
    _skip_if_unsupported(8, "Muse A4W4 owned forward")
    _spawn_and_collect(
        _communication_worker,
        (
            8,
            _get_open_port(),
            partial(
                _muse_tp_owned_forward_worker,
                quantized=True,
                down_columns=down_columns,
            ),
            "nccl",
        ),
        8,
    )
