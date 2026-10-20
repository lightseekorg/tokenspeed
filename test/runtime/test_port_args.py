from tokenspeed.runtime.execution.distributed_initializer import DistributedConfig
from tokenspeed.runtime.utils import server_args as server_args_module
from tokenspeed.runtime.utils.server_args import PortArgs, ServerArgs


def test_resolved_dist_init_addr_moves_with_busy_control_port(monkeypatch):
    args = ServerArgs(model="x", port=8051)
    monkeypatch.setattr(server_args_module.random, "randint", lambda _low, _high: 500)
    monkeypatch.setattr(
        server_args_module,
        "is_port_available",
        lambda port: port != 8284,
    )

    port_args = PortArgs.init_new(args)

    assert port_args.dist_init_addr == "127.0.0.1:8294"


def test_distributed_config_uses_resolved_dist_init_addr():
    args = ServerArgs(model="x", dist_init_addr="127.0.0.1:8284")
    args.mapping.rank = 0
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
