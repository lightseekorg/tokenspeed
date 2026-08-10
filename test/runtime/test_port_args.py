from types import SimpleNamespace

from tokenspeed.runtime.execution.distributed_initializer import DistributedConfig
from tokenspeed.runtime.utils import server_args as server_args_module
from tokenspeed.runtime.utils.server_args import PortArgs

LINUX_DEFAULT_EPHEMERAL_PORT_FLOOR = 32768


def test_resolved_dist_init_addr_moves_with_busy_control_port(monkeypatch):
    args = SimpleNamespace(
        port=8051,
        mapping=SimpleNamespace(nnodes=1, has_attn_dp=False),
        dist_init_addr=None,
        node_rank=0,
    )
    monkeypatch.setattr(
        server_args_module,
        "is_port_available",
        lambda port: port != 31976,
    )

    port_args = PortArgs.init_new(args)

    assert port_args.dist_init_addr == "127.0.0.1:31986"


def test_derived_dist_init_addr_without_rescan(monkeypatch):
    args = SimpleNamespace(
        port=8051,
        mapping=SimpleNamespace(nnodes=1, has_attn_dp=False),
        dist_init_addr=None,
        node_rank=0,
    )
    monkeypatch.setattr(server_args_module, "is_port_available", lambda _port: True)

    port_args = PortArgs.init_new(args)

    assert port_args.dist_init_addr == "127.0.0.1:31976"


def test_derived_control_ports_avoid_ephemeral_range(monkeypatch):
    args = SimpleNamespace(
        port=41399,
        mapping=SimpleNamespace(nnodes=1, has_attn_dp=False),
        dist_init_addr=None,
        node_rank=0,
    )
    monkeypatch.setattr(server_args_module, "is_port_available", lambda _port: True)

    port_args = PortArgs.init_new(args)

    control_ports = [
        int(port_args.dist_init_addr.rsplit(":", 1)[1]),
        int(port_args.tokenizer_ipc_name.rsplit(":", 1)[1]),
        int(port_args.scheduler_input_ipc_name.rsplit(":", 1)[1]),
        int(port_args.rpc_ipc_name.rsplit(":", 1)[1]),
        int(port_args.metrics_ipc_name.rsplit(":", 1)[1]),
        # init_new checks rpc_ipc_port even though PortArgs does not expose it.
        int(port_args.scheduler_input_ipc_name.rsplit(":", 1)[1]) + 1,
    ]
    assert max(control_ports) == 32462
    assert all(port < LINUX_DEFAULT_EPHEMERAL_PORT_FLOOR for port in control_ports)


def test_adjacent_server_ports_use_non_overlapping_control_slots(monkeypatch):
    monkeypatch.setattr(server_args_module, "is_port_available", lambda _port: True)

    def init_port(server_port):
        args = SimpleNamespace(
            port=server_port,
            mapping=SimpleNamespace(nnodes=1, has_attn_dp=False),
            dist_init_addr=None,
            node_rank=0,
        )
        return int(PortArgs.init_new(args).dist_init_addr.rsplit(":", 1)[1])

    assert init_port(41076) - init_port(41075) == 10


def test_available_explicit_dist_init_addr_is_unchanged(monkeypatch):
    args = SimpleNamespace(
        port=41075,
        mapping=SimpleNamespace(nnodes=1, has_attn_dp=False),
        dist_init_addr="127.0.0.1:41308",
        node_rank=0,
    )
    monkeypatch.setattr(server_args_module, "is_port_available", lambda _port: True)

    port_args = PortArgs.init_new(args)

    # Explicit choices are not moved solely because they are ephemeral ports.
    assert port_args.dist_init_addr == "127.0.0.1:41308"


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
