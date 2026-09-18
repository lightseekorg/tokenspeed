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

"""Pytest checks for projection configuration, dispatch, and model integration.

Distributed numerical checks use the separate torchrun entry point
validate_kimi_k3_o_proj.py.
"""

import builtins
import os
from test.runtime.distributed.kimi_k3_o_proj_helpers import dep_mapping
from types import SimpleNamespace

import pytest
import torch
import torch.distributed as dist

from tokenspeed.runtime.distributed.mapping import Mapping
from tokenspeed.runtime.distributed.process_group_manager import (
    process_group_manager as pg_manager,
)
from tokenspeed.runtime.layers.attention import o_proj as projection_ops
from tokenspeed.runtime.layers.attention.o_proj import (
    ProjectionWorkspace,
    make_output_projection,
    projection_mapping,
    validate_projection_settings,
)
from tokenspeed.runtime.utils.env import envs

ENV_NAME = envs.TOKENSPEED_KIMI_K3_O_PROJ_TP_SIZE.name
A2A_ENV_NAME = envs.TOKENSPEED_O_PROJ_A2A_BACKEND.name
RS_ENV_NAME = envs.TOKENSPEED_O_PROJ_RS_BACKEND.name


def test_projection_mapping_and_validation(monkeypatch):
    for field, expected in (
        (envs.TOKENSPEED_KIMI_K3_O_PROJ_TP_SIZE, "1"),
        (envs.TOKENSPEED_O_PROJ_A2A_BACKEND, "flashinfer"),
        (envs.TOKENSPEED_O_PROJ_RS_BACKEND, "triton_peer"),
    ):
        monkeypatch.delenv(field.name, raising=False)
        assert field.get() == expected
    for size in (1, 2, 4, 8, 16):
        for rank in range(16):
            mapping = dep_mapping(rank, 16)
            parallel = projection_mapping(mapping.rank, mapping.world_size, size)
            assert parallel.tp_group == tuple(
                range(rank // size * size, (rank // size + 1) * size)
            )
            assert parallel.tp_rank == rank % size
            assert parallel.dp_size == 16 // size
            assert mapping.attn.tp_size == mapping.linear_attn.tp_size == 1
            assert mapping.moe.ep_size == 16
    for value in ("", "bad", "0", "-1", "3", "32"):
        monkeypatch.setenv(ENV_NAME, value)
        assert envs.TOKENSPEED_KIMI_K3_O_PROJ_TP_SIZE.get() == value
        with pytest.raises(ValueError):
            validate_projection_settings(
                dep_mapping(0, 16),
                envs.TOKENSPEED_KIMI_K3_O_PROJ_TP_SIZE.get(),
                "nccl",
                "nccl",
            )
    from tokenspeed.runtime.models.kimi_k3 import _output_projection_mapping

    monkeypatch.setenv(ENV_NAME, "4")
    with pytest.raises(ValueError, match="requires"):
        _output_projection_mapping(Mapping(rank=0, world_size=4))
    monkeypatch.setenv(ENV_NAME, "1")
    monkeypatch.setenv(A2A_ENV_NAME, "nccl")
    monkeypatch.setenv(RS_ENV_NAME, "nccl")
    mapping = dep_mapping(0, 1)
    for name, valid, invalid in (
        (
            A2A_ENV_NAME,
            ("nccl", "auto", "flashinfer"),
            ("", "invalid", "NVLINK", "flashinfer_quantized"),
        ),
        (RS_ENV_NAME, ("nccl", "triton_peer"), ("", "auto", "invalid", "triton_rsag")),
    ):
        for value in valid:
            monkeypatch.setenv(name, value)
            validate_projection_settings(
                mapping,
                envs.TOKENSPEED_KIMI_K3_O_PROJ_TP_SIZE.get(),
                envs.TOKENSPEED_O_PROJ_A2A_BACKEND.get(),
                envs.TOKENSPEED_O_PROJ_RS_BACKEND.get(),
            )
        for value in invalid:
            monkeypatch.setenv(name, value)
            with pytest.raises(
                ValueError, match="A2A" if name == A2A_ENV_NAME else "reduction"
            ):
                validate_projection_settings(
                    mapping,
                    envs.TOKENSPEED_KIMI_K3_O_PROJ_TP_SIZE.get(),
                    envs.TOKENSPEED_O_PROJ_A2A_BACKEND.get(),
                    envs.TOKENSPEED_O_PROJ_RS_BACKEND.get(),
                )
        monkeypatch.setenv(name, "nccl")

    # Agreement precedes parsing, including disabled or malformed local settings.
    monkeypatch.setattr(projection_ops.dist, "is_initialized", lambda: True)
    groups = []
    monkeypatch.setattr(
        pg_manager,
        "init_process_group",
        lambda group, backend: groups.append((group, backend)),
    )
    monkeypatch.setattr(pg_manager, "get_process_group", lambda backend, group: group)

    def disagree(values, local, group):
        values[:] = [local] * len(values)
        values[-1] = ("4", "nccl", "nccl")

    monkeypatch.setattr(projection_ops.dist, "all_gather_object", disagree)
    for value in ("1", "bad"):
        with pytest.raises(ValueError, match="differ across ranks"):
            validate_projection_settings(dep_mapping(0, 4), value, "nccl", "nccl")
    assert groups == [(tuple(range(4)), "gloo")] * 2


def test_a2a_policy_and_lifecycle(monkeypatch):
    monkeypatch.setenv(A2A_ENV_NAME, "nccl")
    workspace = ProjectionWorkspace(512, 256, torch.bfloat16, torch.device("cpu"))
    parallel = projection_mapping(0, 4, 4)
    workspace.initialize_a2a(parallel, 256, backend="nccl")
    assert workspace.a2a is None
    assert not workspace.use_flashinfer(parallel, [16] * 4, 256)
    closed = []
    workspace.a2a = SimpleNamespace(close=lambda: closed.append(True))
    for rank in range(4):
        parallel = projection_mapping(rank, 4, 4)
        for counts, channels, expected in (
            ([1] * 4, 256, True),
            ([16] * 4, 256, True),
            ([17] * 4, 256, True),
            ([64] * 4, 256, True),
            ([65] * 4, 256, True),
            ([512] * 4, 256, True),
            ([513] * 4, 256, False),
            ([16, 0, 16, 16], 256, False),
            ([0] * 4, 256, False),
            ([16] * 4, 100, False),
        ):
            assert workspace.use_flashinfer(parallel, counts, channels) == expected
    workspace.close()
    workspace.close()
    assert closed == [True]


def test_bf16_dispatch_across_peer_envelope(monkeypatch):
    # Shared execution must not read Kimi settings or require a MoE layout.
    for name in (ENV_NAME, A2A_ENV_NAME, RS_ENV_NAME):
        monkeypatch.setenv(name, "invalid-model-setting")
    parallel = projection_ops.projection_mapping(0, 4, 4)
    exchange = projection_ops.DistributedOutputProjection(parallel, 512)
    workspace = ProjectionWorkspace(64, 512, torch.bfloat16, torch.device("cpu"))
    exchange.workspace = workspace
    workspace.a2a = object()
    workspace.peer_states[128] = SimpleNamespace(
        input_buffer=lambda rows: torch.empty(rows * 4, 128)
    )
    calls = []

    monkeypatch.setattr(
        projection_ops,
        "triton_projection_reduce_scatter_after_a2a",
        lambda peer, partial, rows: partial[:rows],
    )
    monkeypatch.setattr(
        projection_ops,
        "flashinfer_projection_a2a_borrowed",
        lambda state, inputs: (calls.append(("bf16", inputs.shape[0])) or inputs),
    )
    linear = SimpleNamespace(
        output_size=128,
        forward_into=lambda values, scales, output: (output, None),
    )
    for rows in range(1, 65):
        inputs = torch.empty(rows, 512, dtype=torch.bfloat16)
        assert exchange.forward(inputs, linear, [rows] * 4).shape == (rows, 128)
    assert calls == [("bf16", rows) for rows in range(1, 65)]
    ordinary_mapping = Mapping(rank=0, world_size=4)
    linear, wrapper = projection_ops.make_output_projection(
        parallel=parallel,
        input_size=512,
        output_size=128,
        quant_config=None,
        prefix="attention.o_proj",
        default_parallel=ordinary_mapping.attn,
        reduce_results=True,
    )
    assert isinstance(wrapper, projection_ops.DistributedOutputProjection)
    assert linear.weight.shape == (128, 128)
    assert not linear.reduce_results


def test_peer_policy_and_disabled_initialization(monkeypatch):
    parallel = projection_mapping(0, 4, 4)
    workspace = ProjectionWorkspace(512, 256, torch.bfloat16, torch.device("cpu"))
    monkeypatch.setenv(RS_ENV_NAME, "nccl")
    workspace.initialize_reduce_scatter(parallel, [128], backend="nccl")
    assert workspace.peer_states == {}
    # Policy depends on padded subgroup capacity, not local valid rows.
    peer = object()
    workspace.peer_states[128] = peer
    large_workspace = ProjectionWorkspace(8193, 8, torch.bfloat16, torch.device("cpu"))
    large_workspace.peer_states[128] = peer
    for rows in (512, 513, 8192, 8193):
        assert large_workspace.peer_state(128, rows) is (peer if rows <= 8192 else None)
    for rows, width in (
        (1, 128),
        (16, 128),
        (17, 128),
        (64, 128),
        (65, 128),
        (256, 128),
        (257, 128),
        (0, 128),
        (16, 256),
    ):
        assert workspace.peer_state(width, rows) is (
            peer if width == 128 and 0 < rows <= 512 else None
        )

    monkeypatch.setenv(RS_ENV_NAME, "triton_peer")
    no_peer = ProjectionWorkspace(16, 256, torch.bfloat16, torch.device("cpu"))
    no_peer.initialize_reduce_scatter(
        projection_ops.projection_mapping(0, 4, 2), [128], backend="triton_peer"
    )
    assert no_peer.peer_states == {}
    assert no_peer.borrowed_a2a is None
    bad_dtype = ProjectionWorkspace(16, 256, torch.float16, torch.device("cpu"))
    with pytest.raises(ValueError, match="BF16"):
        bad_dtype.initialize_reduce_scatter(parallel, [128], backend="triton_peer")
    bad_width = ProjectionWorkspace(16, 256, torch.bfloat16, torch.device("cpu"))
    with pytest.raises(ValueError, match="positive"):
        bad_width.initialize_reduce_scatter(parallel, [0], backend="triton_peer")


def test_peer_reduction_reuse_fence_contract(monkeypatch):
    import tokenspeed_kernel.ops.communication.triton_projection as kernels

    state = kernels.ProjectionPeerState.__new__(kernels.ProjectionPeerState)
    state.max_rows, state.p, state.r, state.hidden = 16, 4, 0, 128
    state.buffer = torch.empty(64, 128, dtype=torch.bfloat16)
    state.ptrs = None
    barriers = []
    state.handle = SimpleNamespace(barrier=lambda channel: barriers.append(channel))

    class FakeReduction:
        def __getitem__(self, grid):
            return lambda *args: args[1].zero_()

    monkeypatch.setattr(kernels, "owner_reduce", FakeReduction())
    for synchronize_reuse, expected in ((True, [0, 1]), (False, [0])):
        barriers.clear()
        output = state._reduce(state.input_buffer(16), 16, synchronize_reuse)
        assert barriers == expected
        assert output.shape == (16, 128)
        assert output.data_ptr() != state.buffer.data_ptr()


def test_optional_a2a_import_and_topology_fallback(monkeypatch):
    from tokenspeed_kernel.thirdparty.flashinfer.projection_alltoall import (
        create_projection_a2a,
    )

    group = SimpleNamespace(size=lambda: 4)

    def gather(values, value, group):
        values[:] = [value] * group.size()

    monkeypatch.setattr(dist, "all_gather_object", gather)
    original_import = builtins.__import__
    fake = None

    def optional_import(name, *args, **kwargs):
        if name == "flashinfer.comm.ulysses":
            if fake is None:
                raise ImportError("optional API absent")
            return fake
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", optional_import)
    kwargs = dict(
        group=group, max_elems=4096, dtype=torch.bfloat16, device=torch.device("cuda")
    )
    comm, reason = create_projection_a2a(**kwargs, backend="auto")
    assert comm is None and "optional API absent" in reason
    with pytest.raises(RuntimeError, match="unavailable"):
        create_projection_a2a(**kwargs, backend="flashinfer")
    closed = []
    fallback = SimpleNamespace(
        backend="nccl", fallback_reason="no NVLink", close=lambda: closed.append(True)
    )
    fake = SimpleNamespace(UlyssesCommunicator=lambda **kwargs: fallback)
    comm, reason = create_projection_a2a(**kwargs, backend="auto")
    assert comm is None and reason == "no NVLink" and closed == [True]


def test_disabled_projection_and_shard_loader(monkeypatch):
    mapping = dep_mapping(2, 4)
    monkeypatch.setenv(ENV_NAME, "1")
    local, exchange = make_output_projection(
        parallel=projection_mapping(
            mapping.rank, mapping.world_size, int(os.environ[ENV_NAME])
        ),
        input_size=32,
        output_size=16,
        quant_config=None,
        prefix="self_attn.o_proj",
        default_parallel=mapping.attn,
        reduce_results=False,
    )
    assert exchange is None
    assert local.weight.shape == (16, 32)
    monkeypatch.setenv(ENV_NAME, "4")
    sharded, exchange = make_output_projection(
        parallel=projection_mapping(
            mapping.rank, mapping.world_size, int(os.environ[ENV_NAME])
        ),
        input_size=32,
        output_size=16,
        quant_config=None,
        prefix="self_attn.o_proj",
        default_parallel=mapping.attn,
        reduce_results=False,
    )
    weight = torch.arange(16 * 32, dtype=sharded.weight.dtype).view(16, 32)
    sharded.weight.weight_loader(sharded.weight, weight)
    torch.testing.assert_close(sharded.weight, weight[:, 16:24])
    assert not sharded.reduce_results
    assert exchange is not None


def test_attention_construction_and_empty_rank_participation(monkeypatch):
    from tokenspeed.runtime.configs.kimi_k3_config import KimiLinearConfig
    from tokenspeed.runtime.models.kimi_k3 import KimiLinearKDA, KimiLinearMLAAttention

    monkeypatch.setenv(ENV_NAME, "4")
    mapping = dep_mapping(2, 4)
    config = KimiLinearConfig(
        hidden_size=64,
        num_hidden_layers=1,
        num_attention_heads=4,
        num_key_value_heads=4,
        q_lora_rank=16,
        kv_lora_rank=16,
        linear_attn_config={
            "kda_layers": [1],
            "full_attn_layers": [],
            "num_heads": 4,
            "head_dim": 16,
            "short_conv_kernel_size": 4,
            "gate_lower_bound": -5.0,
            "use_full_rank_gate": True,
        },
    )
    layers = [
        KimiLinearKDA(
            config=config, mapping=mapping, layer_id=0, quant_config=None, prefix=""
        ),
        KimiLinearMLAAttention(
            config=config,
            mapping=mapping,
            hidden_size=64,
            num_heads=4,
            qk_nope_head_dim=16,
            qk_rope_head_dim=0,
            v_head_dim=16,
            q_lora_rank=16,
            kv_lora_rank=16,
            rope_theta=10000,
            rope_scaling=None,
            max_position_embeddings=128,
            quant_config=None,
            layer_id=0,
            prefix="",
            reduce_attn_results=False,
            alt_stream=None,
        ),
    ]
    for layer in layers:
        assert layer.o_proj.weight.shape == (64, 16)
        assert layer.mapping.attn.tp_size == layer.mapping.linear_attn.tp_size == 1
        calls = []

        def exchange(inputs, linear, counts):
            calls.append((inputs.shape, counts))
            return inputs.new_empty((0, linear.output_size))

        monkeypatch.setattr(layer.output_projection_exchange, "forward", exchange)
        counts = [7, 0, 0, 0]
        result = layer(
            positions=torch.empty(0, dtype=torch.int64),
            hidden_states=torch.empty(0, 64),
            ctx=SimpleNamespace(
                collective_global_num_tokens=counts, global_num_tokens=None
            ),
            comm_manager=None,
            block_scale=None,
            attnres_partial_args=None,
        )
        assert result.shape == (0, 64)
        assert calls == [(torch.Size([0, 64]), counts)]
