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

"""Drive two isolated CUDA launchers through the real TokenSpeed serving wire."""

import argparse
import json
import os
import re
import signal
import subprocess
import sys
import time
from pathlib import Path

LAYOUTS = {
    "ep4": (
        2,
        4,
        [
            "--attn-tp-size",
            "1",
            "--data-parallel-size",
            "4",
            "--ep-size",
            "4",
            "--all2all-backend",
            "none",
        ],
    ),
    "dp2-tp2-ep4": (
        2,
        2,
        [
            "--attn-tp-size",
            "2",
            "--data-parallel-size",
            "2",
            "--ep-size",
            "4",
            "--all2all-backend",
            "none",
        ],
    ),
}


def node_command(args, node, layout):
    python = args.python_0 if node == 0 else args.python_1
    # The controller already runs in node 0, which also hosts the frontend.
    prefix = [] if node == 0 else ["nsenter", f"--net={args.netns_1}", "--"]
    return [
        *prefix,
        python,
        "-m",
        "tokenspeed.runtime.entrypoints.engine",
        "--model",
        str(args.model),
        "--device",
        "cuda",
        "--dtype",
        "bfloat16",
        "--seed",
        "17",
        "--nnodes",
        "2",
        "--node-rank",
        str(node),
        "--dist-init-addr",
        f"{args.host_0}:{args.store_port}",
        "--data-parallel-address",
        args.host_0,
        "--data-parallel-rpc-port",
        str(args.wire_port),
        "--zmq-engine-index",
        "0",
        "--port",
        str(args.store_port + 20),
        "--distributed-timeout-seconds",
        str(args.startup_timeout),
        "--attention-backend",
        "mha",
        "--sampling-backend",
        "greedy",
        "--enforce-eager",
        "--disable-prefill-graph",
        "--disable-autotune",
        "--disable-prefix-caching",
        "--disable-kvstore",
        "--disable-cudagraph-memory-reserve",
        "--disable-symm-mem",
        "--disable-nccl-nvls",
        "--max-model-len",
        "512",
        "--max-num-seqs",
        "4",
        "--max-total-tokens",
        "1024",
        "--chunked-prefill-size",
        "128",
        "--enable-memory-saver",
        "--weight-memory-policy",
        "cpu_backup",
        *layout,
    ]


def node_env(args, node, devices):
    env = os.environ.copy()
    for name in (
        "RANK",
        "LOCAL_RANK",
        "WORLD_SIZE",
        "LOCAL_WORLD_SIZE",
        "NODE_RANK",
        "GROUP_RANK",
        "MASTER_ADDR",
        "MASTER_PORT",
    ):
        env.pop(name, None)
    runtime_path = args.pythonpath_0 if node == 0 else args.pythonpath_1
    probe = Path(__file__).resolve().parent / "cuda_two_node_probe"
    parts = [str(probe), runtime_path or env.get("PYTHONPATH", "")]
    host = args.host_0 if node == 0 else args.host_1
    binary = args.ts_host_0 if node == 0 else args.ts_host_1
    env.update(
        PYTHONPATH=os.pathsep.join(filter(None, parts)),
        CUDA_DEVICE_ORDER="PCI_BUS_ID",
        CUDA_VISIBLE_DEVICES=",".join(devices),
        TOKENSPEED_HOST_BINARY=str(binary),
        TOKENSPEED_HOST_IP=host,
        GLOO_SOCKET_IFNAME=args.interface,
        NCCL_SOCKET_IFNAME=f"={args.interface}",
        NCCL_IB_DISABLE="1",
        NCCL_P2P_DISABLE="1",
        NCCL_SHM_DISABLE="1",
        NCCL_CUMEM_HOST_ENABLE="0",
        NCCL_NET="Socket",
        NCCL_DEBUG="INFO",
        NCCL_DEBUG_SUBSYS="INIT,NET",
        HF_HUB_OFFLINE="1",
        TRANSFORMERS_OFFLINE="1",
        PYTHONDONTWRITEBYTECODE="1",
        TS_TEST_TOPOLOGY_TRACE=str(args.output / f"node-{node}.jsonl"),
    )
    if args.preload:
        env["LD_PRELOAD"] = str(args.preload)
    env.pop("TS_TEST_TOPOLOGY_DELAY_RANK", None)
    if args.delay_rank is not None:
        env["TS_TEST_TOPOLOGY_DELAY_RANK"] = str(args.delay_rank)
    return env


def stop_processes(processes):
    signals = (signal.SIGALRM, signal.SIGTERM)
    handlers = {signum: signal.getsignal(signum) for signum in signals}
    pending = []

    def defer(signum, frame):
        if handlers[signum] != signal.SIG_IGN and not pending:
            pending.append((signum, frame))

    # A deadline must still fail the test, but only after our launchers are
    # reaped. A main-thread signal mask alone cannot protect against signals
    # delivered through the controller's background threads.
    try:
        for signum in signals:
            signal.signal(signum, defer)
        for process in processes:
            if process.poll() is None:
                process.terminate()
        deadline = time.monotonic() + 40
        for process in processes:
            try:
                process.wait(timeout=max(0, deadline - time.monotonic()))
            except subprocess.TimeoutExpired:
                pass
        # Reap the exact process group started for this launcher.
        for process in processes:
            try:
                os.killpg(process.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            process.wait(timeout=5)
        return [process.returncode for process in processes]
    finally:
        try:
            signal.signal(signal.SIGALRM, handlers[signal.SIGALRM])
        finally:
            signal.signal(signal.SIGTERM, handlers[signal.SIGTERM])
        if pending:
            signum, frame = pending[0]
            handler = handlers[signum]
            if callable(handler):
                handler(signum, frame)
            else:
                signal.raise_signal(signum)


def observe_mixed(wire, processes, timeout):
    import msgpack
    import zmq

    poller = zmq.Poller()
    poller.register(wire.handshake, zmq.POLLIN)
    poller.register(wire.input, zmq.POLLIN)
    counts = {"HELLO": 0, "READY": 0, "input": 0}
    deadline = time.monotonic() + timeout
    drained = False
    while time.monotonic() < deadline:
        events = poller.poll(250)
        for sock, _ in events:
            identity, payload = sock.recv_multipart()
            if sock == wire.input:
                counts["input"] += 1
            else:
                status = msgpack.unpackb(payload, raw=False)["status"]
                counts[status] += 1
                if status == "HELLO":
                    sock.send_multipart([identity, wire.handshake_init()])
        if all(process.poll() is not None for process in processes):
            if drained and not events:
                wire.record("observed", **counts)
                assert counts == {"HELLO": 0, "READY": 0, "input": 0}, counts
                return counts
            drained = True
    raise TimeoutError(f"mixed startup did not exit before deadline; wire={counts}")


def accepted_case(wire, register_wires, args, engines, processes):
    # Arm a parent-only watchdog; the production frontend helper has its own
    # per-request deadlines. Cleanup kills only our launchers.
    def expired(_signum, _frame):
        raise TimeoutError("serving matrix exceeded deadline")

    signal.signal(signal.SIGALRM, expired)
    signal.alarm(args.deadline)
    try:
        register_wires((wire, engines))
        prompt = [12, 25, 46, 38, 19]
        count = 8
        before = wire.generate(prompt, count)
        tags = ["kv_cache", "weights"]
        if args.nccl:
            tags.append("nccl")
        wire.all("pause_generation", [{"mode": "wait"}])
        release = time.monotonic_ns()
        release_replies = []
        wire.all(
            "release_memory_occupation",
            [{"tags": tags, "transition_id": "logical-node-release"}],
            delay=args.fanout_delay,
            check_reply=release_replies.append,
        )
        resume_replies = []
        wire.all(
            "resume_memory_occupation",
            [{"transition_id": "logical-node-resume"}],
            check_reply=resume_replies.append,
        )
        wire.all("continue_generation")
        after = wire.generate(prompt, count)
        assert before == after, (before, after)
        assert all(process.poll() is None for process in processes)
        return {
            "tokens_per_ingress": before,
            "tokens_after_resume_per_ingress": after,
            "token_count": count,
            "release_sent_ns": release,
            "release_ack_ns": release_replies,
            "resume_ack_ns": resume_replies,
            "tags": tags,
        }
    finally:
        signal.alarm(0)


def read_trace(args, cards_per_node, devices):
    rows = []
    for node in (0, 1):
        path = args.output / f"node-{node}.jsonl"
        assert path.is_file(), path
        node_rows = [json.loads(line) for line in path.read_text().splitlines()]
        expected = set(range(node * cards_per_node, (node + 1) * cards_per_node))
        mappings = [row for row in node_rows if row["event"] == "mapping"]
        assert len(mappings) == cards_per_node, mappings
        assert {row["rank"] for row in mappings} == expected
        for row in mappings:
            assert row["node"] == node, row
            rank = row["rank"]
            assert row["visible"] == devices[node], row
            assert row["local_rank"] == rank % cards_per_node, row
            assert row["ep"] == [0, 1, 2, 3], row
            if args.layout == "ep4":
                assert row["tp"] == [rank], row
                assert row["dp"] == [0, 1, 2, 3], row
            else:
                assert row["tp"] == sorted(expected), row
                assert row["dp"] == [rank % 2, rank % 2 + 2], row
        negotiations = [row for row in node_rows if row["event"] == "negotiated"]
        assert len(negotiations) == cards_per_node, negotiations
        assert {row["rank"] for row in negotiations} == expected, negotiations
        assert {row["protocol"] for row in negotiations} == {args.protocols[node]}
        rows.extend(node_rows)
    return rows


def assert_trace(args, rows, result, cards_per_node, mixed):
    all_ranks = set(range(2 * cards_per_node))
    if mixed:
        assert len(result["launcher_exits"]) == 2 and all(
            code == 1 for code in result["launcher_exits"]
        ), result["launcher_exits"]
        refused = [row for row in rows if row["event"] == "rendezvous_refused"]
        assert {row["rank"] for row in refused} == all_ranks and len(refused) == len(
            all_ranks
        )
        for row in refused:
            assert "Memory protocol mismatch" in row["error"], row
        assert not [
            row
            for row in rows
            if row["event"] in {"service_ready", "rendezvous_ok", "apply_begin"}
        ]
        return
    assert result["launcher_exits"] == [0, 0], result["launcher_exits"]
    ready = [row for row in rows if row["event"] == "service_ready"]
    assert {row["rank"] for row in ready} == all_ranks and len(ready) == len(all_ranks)
    expected_hosts = [args.host_0] * cards_per_node + [args.host_1] * cards_per_node
    for row in (r for r in rows if r["event"] == "rendezvous_ok"):
        assert [
            address.rsplit(":", 1)[0] for address in row["addresses"]
        ] == expected_hosts, row
    for release in (True, False):
        applies = [
            row
            for row in rows
            if row["event"] == "apply_complete" and row["release"] == release
        ]
        assert {row["rank"] for row in applies} == all_ranks, applies
        if args.protocols == (3, 3):
            stamps = result["release_ack_ns"] if release else result["resume_ack_ns"]
            assert max(row["ns"] for row in applies) < min(stamps), (applies, stamps)


def run(args):
    from cuda_two_node_test_wire import Wire, register_wires

    cards_per_node, engines, layout = LAYOUTS[args.layout]
    devices = [args.cards_0.split(","), args.cards_1.split(",")]
    visible = devices[0] + devices[1]
    assert len(devices[0]) == len(devices[1]) in {cards_per_node, 4}, devices
    assert all(device.strip() for device in visible), devices
    assert len(set(visible)) == len(visible), devices
    args.output = args.output.resolve()
    args.output.mkdir(parents=True)
    if os.path.samefile("/proc/self/ns/net", args.netns_1):
        raise RuntimeError("logical nodes must use distinct network namespaces")
    mixed = len(set(args.protocols)) > 1
    if args.nccl and mixed:
        raise ValueError("mixed versions stop before NCCL memory operations")
    if args.delay_rank is not None and not 0 <= args.delay_rank < 2 * cards_per_node:
        raise ValueError("delay rank must participate in this layout")
    wire = Wire(args.host_0, base_port=args.wire_port)
    processes = []
    result = {
        "layout": args.layout,
        "visible_devices": devices,
        "protocols": args.protocols,
        "participating_devices": [node[:cards_per_node] for node in devices],
    }
    try:
        for node in (0, 1):
            command = node_command(args, node, layout)
            env = node_env(args, node, devices[node])
            with (args.output / f"node-{node}.log").open("w") as stdout:
                processes.append(
                    subprocess.Popen(
                        command,
                        env=env,
                        stdout=stdout,
                        stderr=subprocess.STDOUT,
                        start_new_session=True,
                    )
                )
        if mixed:
            result["wire"] = observe_mixed(wire, processes, args.deadline)
        else:
            result.update(accepted_case(wire, register_wires, args, engines, processes))
    finally:
        result["launcher_exits"] = stop_processes(processes)
        wire.close()
        (args.output / "partial-result.json").write_text(
            json.dumps(result, indent=2) + "\n"
        )
    rows = read_trace(args, cards_per_node, devices)
    assert_trace(args, rows, result, cards_per_node, mixed)
    for node in (0, 1):
        log = (args.output / f"node-{node}.log").read_text()
        assert not re.search(r"segmentation fault|sigsegv|segfault", log, re.I), node
    result["status"] = "rejected-before-ready" if mixed else "passed"
    (args.output / "result.json").write_text(json.dumps(result, indent=2) + "\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--layout", choices=LAYOUTS, required=True)
    parser.add_argument(
        "--protocols", choices=("2,2", "2,3", "3,2", "3,3"), required=True
    )
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--netns-1", type=Path, required=True)
    parser.add_argument("--host-0", default="10.199.0.1")
    parser.add_argument("--host-1", default="10.199.0.2")
    parser.add_argument("--interface", default="tsnet")
    parser.add_argument("--store-port", type=int, default=28600)
    parser.add_argument("--wire-port", type=int, default=28610)
    parser.add_argument("--cards-0", required=True)
    parser.add_argument("--cards-1", required=True)
    parser.add_argument("--python-0", default=sys.executable)
    parser.add_argument("--python-1", default=sys.executable)
    parser.add_argument("--pythonpath-0")
    parser.add_argument("--pythonpath-1")
    parser.add_argument("--ts-host-0", type=Path, required=True)
    parser.add_argument("--ts-host-1", type=Path, required=True)
    parser.add_argument("--preload", type=Path)
    parser.add_argument("--startup-timeout", type=int, default=90)
    parser.add_argument("--deadline", type=int, default=210)
    parser.add_argument("--delay-rank", type=int)
    parser.add_argument("--fanout-delay", type=float, default=0)
    parser.add_argument("--nccl", action="store_true")
    args = parser.parse_args()
    args.protocols = tuple(map(int, args.protocols.split(",")))

    def terminated(_signum, _frame):
        raise KeyboardInterrupt("test terminated")

    signal.signal(signal.SIGTERM, terminated)
    run(args)


if __name__ == "__main__":
    main()
