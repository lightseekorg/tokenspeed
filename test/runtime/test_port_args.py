from types import SimpleNamespace

from tokenspeed.runtime.execution.distributed_initializer import DistributedConfig
from tokenspeed.runtime.utils import server_args as server_args_module
from tokenspeed.runtime.utils.server_args import PortArgs


def test_resolved_dist_init_addr_moves_with_busy_control_port(monkeypatch):
    args = SimpleNamespace(
        port=8051,
        mapping=SimpleNamespace(nnodes=1, has_attn_dp=False),
        dist_init_addr=None,
        node_rank=0,
    )
    monkeypatch.setattr(server_args_module.random, "randint", lambda _low, _high: 500)
    monkeypatch.setattr(
        server_args_module,
        "is_port_available",
        lambda port: port != 8284,
    )

    port_args = PortArgs.init_new(args)

    assert port_args.dist_init_addr == "127.0.0.1:8294"


def test_distributed_config_uses_resolved_dist_init_addr():
    mapping = SimpleNamespace(
        world_size=1,
        nprocs_per_node=1,
        attn=SimpleNamespace(tp_rank=0, tp_size=1, dp_size=1),
        dense=SimpleNamespace(tp_size=1),
        moe=SimpleNamespace(ep_size=1, ep_rank=0),
    )
    args = SimpleNamespace(
        device="cuda",
        mapping=mapping,
        dist_init_addr="127.0.0.1:8284",
        distributed_timeout_seconds=None,
        force_deterministic_rsag=False,
    )
    port_args = PortArgs(
        tokenizer_ipc_name="tcp://127.0.0.1:8295",
        scheduler_input_ipc_name="tcp://127.0.0.1:8299",
        nccl_port=8551,
        rpc_ipc_name="tcp://127.0.0.1:8297",
        metrics_ipc_name="tcp://127.0.0.1:8298",
        tokenizer_worker_ipc_name=None,
        dist_init_addr="127.0.0.1:8294",
    )

    config = DistributedConfig.from_server_args(
        args,
        port_args,
        gpu_id=0,
        global_rank=0,
        hidden_size=0,
        max_num_tokens=0,
    )

    assert config.dist_init_addr == "127.0.0.1:8294"
