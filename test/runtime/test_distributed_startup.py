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

"""Communication capacity is resolved before a first prefill can use it."""

from test.ci_system.ci_register import register_cuda_ci

import pytest

import tokenspeed.runtime.engine.rank_setup as rank_setup
from tokenspeed.runtime.execution.distributed_initializer import (
    DistributedConfig,
    DistributedInitializer,
)
from tokenspeed.runtime.utils.server_args import PortArgs, ServerArgs

register_cuda_ci(est_time=5, suite="runtime-1gpu")


class _ReductionResources:
    """Stand-in for the unavailable distributed GPU workspace allocator."""

    def __init__(self):
        self.capacities = {}

    def configure_group(self, *, rank, group, max_token_num, hidden_dim):
        self.capacities[group] = (max_token_num, hidden_dim)
        return True


class _Backend:
    def __init__(self):
        self.trtllm_ar = _ReductionResources()


class _Device:
    def set_device(self, gpu_id):
        pass


@pytest.mark.parametrize(
    "enabled,fusion_tokens,expected_capacity",
    [(True, 2048, 2048), (True, 64, 128), (False, 2048, 128)],
)
def test_startup_arms_the_complete_selected_fusion_window(
    monkeypatch, enabled, fusion_tokens, expected_capacity
):
    import tokenspeed_kernel.ops.communication.fabric as fabric
    import tokenspeed_kernel.ops.communication.iris as kernel_iris

    import tokenspeed.runtime.distributed.comm_backend as comm_backend
    import tokenspeed.runtime.execution.distributed_initializer as initializer

    args = ServerArgs(
        model="x",
        device="cuda",
        world_size=2,
        nprocs_per_node=2,
        enable_allreduce_fusion=enabled,
        comm_fusion_max_num_tokens=fusion_tokens,
        layer_boundary_norm="fused" if enabled else "unfused",
    )
    args.mapping.rank = 0
    ports = PortArgs(
        tokenizer_ipc_name="tcp://127.0.0.1:8295",
        scheduler_input_ipc_name="tcp://127.0.0.1:8299",
        nccl_port=8551,
        rpc_ipc_name="tcp://127.0.0.1:8297",
        metrics_ipc_name="tcp://127.0.0.1:8298",
        tokenizer_worker_ipc_name=None,
        dist_init_addr="127.0.0.1:8294",
    )
    config = DistributedConfig.from_server_args(
        args, ports, gpu_id=0, global_rank=0, hidden_size=8192, max_num_tokens=8192
    )
    assert config.comm_fusion_token_capacity == (fusion_tokens if enabled else 0)
    backend = _Backend()
    # Only the absent GPU driver and network are substituted. Configuration,
    # mapping, group selection and initialization execute their actual path.
    monkeypatch.setattr(initializer.torch, "get_device_module", lambda _: _Device())
    monkeypatch.setattr(initializer, "get_available_gpu_memory", lambda *a, **k: 1.0)
    monkeypatch.setattr(
        initializer, "maybe_set_numa_aware_cpu_affinity", lambda _: None
    )
    monkeypatch.setattr(
        initializer.pg_manager, "init_distributed", lambda *a, **k: None
    )
    monkeypatch.setattr(
        initializer.pg_manager, "init_process_group", lambda *a, **k: None
    )
    monkeypatch.setattr(
        initializer.pg_manager, "get_process_group", lambda *a, **k: None
    )
    monkeypatch.setattr(fabric, "gather_fabric_map", lambda: None)
    monkeypatch.setattr(kernel_iris, "discover_iris_world", lambda: True)
    monkeypatch.setattr(comm_backend, "get_global_backend", lambda: backend)

    assert DistributedInitializer.initialize(config) == 1.0
    groups = {
        config.mapping.attn.tp_group,
        config.mapping.moe.tp_ep_group,
    }
    assert groups and all(len(group) == 2 for group in groups)
    for group in groups:
        assert backend.trtllm_ar.capacities[group] == (expected_capacity, 8192)


def _amd_platform(monkeypatch, arch_version):
    from tokenspeed_kernel.platform import Platform, PlatformInfo

    monkeypatch.setattr(
        Platform,
        "_instance",
        PlatformInfo(
            vendor="amd",
            arch_version=arch_version,
            device_name="amd",
            device_count=8,
            total_memory=256 << 30,
            memory_bandwidth=8000.0,
            sm_count=256,
            max_threads_per_sm=2048,
            max_shared_memory_per_sm=160 << 10,
        ),
    )


def _fusion_args(world_size, nnodes, **overrides):
    return ServerArgs(
        model="x",
        device="cuda",
        world_size=world_size,
        attn_tp_size=world_size,
        nnodes=nnodes,
        dist_init_addr="127.0.0.1:5000",
        **overrides,
    )


@pytest.mark.parametrize(
    "arch,world_size,explicit,needs_mapped_world",
    [
        ("gfx1250", 8, False, True),
        ("gfx1250", 4, False, True),
        ("gfx950", 16, False, True),
        ("gfx950", 8, False, True),
        ("gfx1250", 8, True, False),
    ],
    ids=[
        "gfx1250-two-hosts",
        "gfx1250-one-host",
        "gfx950-two-hosts",
        "gfx950-one-host",
        "explicit",
    ],
)
def test_amd_fusion_auto_enable_waits_for_the_world_verdict(
    monkeypatch, arch, world_size, explicit, needs_mapped_world
):
    """AMD treats every launch alike: the fusion holds where one Iris heap maps
    the world, which distributed initialization decides. Four-GPU gfx1250
    trays, eight-GPU gfx950 hosts."""
    from tokenspeed_kernel.platform import ArchVersion

    versions = {"gfx1250": ArchVersion(12, 5), "gfx950": ArchVersion(9, 5)}
    _amd_platform(monkeypatch, versions[arch])
    gpus_per_host = 4 if arch == "gfx1250" else 8
    overrides = {"enable_allreduce_fusion": True} if explicit else {}
    args = _fusion_args(world_size, world_size // gpus_per_host, **overrides)
    assert args.enable_allreduce_fusion
    assert args._allreduce_fusion_needs_mapped_world is needs_mapped_world


@pytest.mark.parametrize(
    "explicit,maps_world,expected",
    [(False, True, True), (False, False, False), (True, False, True)],
    ids=["mapped", "unmapped", "explicit-unmapped"],
)
def test_ranks_keep_cross_host_fusion_only_in_a_mapped_world(
    monkeypatch, explicit, maps_world, expected
):
    """An explicit flag in an unmapped world stays on and fails at heap setup."""
    import tokenspeed_kernel.ops.communication.iris as kernel_iris
    from tokenspeed_kernel.platform import ArchVersion

    from tokenspeed.runtime.utils.env import global_server_args_dict

    _amd_platform(monkeypatch, ArchVersion(12, 5))
    overrides = {"enable_allreduce_fusion": True} if explicit else {}
    args = _fusion_args(8, 2, **overrides)
    args.mapping.rank = 0
    ports = PortArgs(
        tokenizer_ipc_name="tcp://127.0.0.1:8295",
        scheduler_input_ipc_name="tcp://127.0.0.1:8299",
        nccl_port=8551,
        rpc_ipc_name="tcp://127.0.0.1:8297",
        metrics_ipc_name="tcp://127.0.0.1:8298",
        tokenizer_worker_ipc_name=None,
        dist_init_addr="127.0.0.1:8294",
    )
    # Distributed initialization needs GPUs and peers; it decides the verdict.
    monkeypatch.setattr(
        rank_setup.DistributedInitializer, "initialize", lambda config: 1.0
    )
    monkeypatch.setattr(kernel_iris, "iris_maps_world", lambda: maps_world)

    assert (
        rank_setup.init_distributed(
            args, ports, gpu_id=0, global_rank=0, hidden_size=7168
        )
        == 1.0
    )
    assert args.enable_allreduce_fusion is expected
    assert global_server_args_dict["enable_allreduce_fusion"] is expected
