#!/usr/bin/env bash
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

# Same-host dots3-note 1P1D. Run from the repository root in the runtime venv:
# MODEL_PATH=/path/to/checkpoint SPECULATIVE_ALGORITHM=NONE \
#   bash test/ci_system/serve_dots3_note_pd_1p1d.sh
# MODEL_PATH=/path/to/checkpoint SPECULATIVE_ALGORITHM=MTP SPECULATIVE_NUM_STEPS=3 \
#   bash test/ci_system/serve_dots3_note_pd_1p1d.sh
# Required: MODEL_PATH (local checkpoint), SPECULATIVE_ALGORITHM=NONE|MTP.
# MTP requires explicit SPECULATIVE_NUM_STEPS >= 1; NONE rejects that setting.
# Both workers use verify width steps+1, top-k 1, the same checkpoint and FP8 draft
# quantization for MTP. Other SPECULATIVE_* settings and all CLI args are rejected.
# No downloads or dependency installs. Source PYTHONPATH must be active.
# Optional placement: WORLD_SIZE=4, PREFILL_GPUS=0,1,2,3, DECODE_GPUS=4,5,6,7.
# Optional endpoints: PREFILL_PORT=18346, DECODE_PORT=18347,
# PREFILL_BOOTSTRAP_PORT=8998, PREFILL_DIST_INIT_ADDR=127.0.0.1:12579,
# DECODE_DIST_INIT_ADDR=127.0.0.1:13580, LB_HOST=127.0.0.1, LB_PORT=18345,
# PROMETHEUS_PORT=18422. Control-plane port clusters must not overlap.
# Optional limits: CHUNKED_PREFILL_SIZE=8192, GPU_MEMORY_UTILIZATION=0.8,
# PD_STARTUP_TIMEOUT=2400, GATEWAY_STARTUP_TIMEOUT=600,
# PD_REQUEST_TIMEOUT=600, WORKER_SHUTDOWN_TIMEOUT=30 (timeouts in seconds).
# Optional: DISAGGREGATION_IB_DEVICE, SERVED_MODEL_NAME=dots3-note,
# PD_CI_LOG_DIR=.ci-artifacts/pd-dots3-note-1p1d.
# Dots3-note is multimodal; this implementation serves only its language model.
# --language-model-only, backends, graph/overlap, prefix caching and transfer
# interval 0 are fixed; vision/audio encoders are not yet implemented.
# Set CHUNKED_PREFILL_SIZE=1024 to exercise multiple chunks in the long smoke.
set -euo pipefail
: "${MODEL_PATH:?Set MODEL_PATH to a local dots3-note checkpoint}"
if (($#)); then
  echo "This launcher accepts environment configuration, not positional arguments" >&2
  exit 2
fi

# Python reuses the CLI readiness probes and owns all subprocess sessions.
exec python3 -u - <<'PY'
import asyncio
import contextlib
import os
import signal
import socket
import subprocess
import sys
import time
from pathlib import Path


def _speculative_args(env, model):
    unsupported = sorted(
        key
        for key in env
        if key.startswith("SPECULATIVE_")
        and key not in ("SPECULATIVE_ALGORITHM", "SPECULATIVE_NUM_STEPS")
    )
    if unsupported:
        raise ValueError(f"Unsupported speculative settings: {', '.join(unsupported)}")
    algorithm = env.get("SPECULATIVE_ALGORITHM")
    if algorithm not in ("NONE", "MTP"):
        raise ValueError("Set SPECULATIVE_ALGORITHM explicitly to NONE or MTP")
    if algorithm == "NONE":
        if "SPECULATIVE_NUM_STEPS" in env:
            raise ValueError("SPECULATIVE_NUM_STEPS requires SPECULATIVE_ALGORITHM=MTP")
        # NONE is a launcher choice, not a ServerArgs CLI value.
        return []
    try:
        steps = int(env["SPECULATIVE_NUM_STEPS"])
        if steps < 1:
            raise ValueError
    except (KeyError, ValueError):
        raise ValueError(
            "MTP requires explicit SPECULATIVE_NUM_STEPS integer >= 1"
        ) from None
    return [
        "--speculative-algorithm",
        algorithm,
        "--speculative-num-steps",
        str(steps),
        "--speculative-num-draft-tokens",
        str(steps + 1),
        "--speculative-eagle-topk",
        "1",
        "--speculative-draft-model-path",
        model,
        "--speculative-draft-model-quantization",
        "fp8",
    ]


async def main():
    model = Path(os.environ["MODEL_PATH"]).resolve()
    speculative_args = _speculative_args(os.environ, str(model))
    if not (model / "config.json").is_file():
        raise ValueError("MODEL_PATH must be a local checkpoint with config.json")
    world_size = int(os.environ.get("WORLD_SIZE", "4"))
    gpus = [
        os.environ.get("PREFILL_GPUS", "0,1,2,3"),
        os.environ.get("DECODE_GPUS", "4,5,6,7"),
    ]
    gpu_ids = [part.strip() for group in gpus for part in group.split(",")]
    if (
        world_size <= 0
        or any(len(group.split(",")) != world_size for group in gpus)
        or any(not part for part in gpu_ids)
        or len(set(gpu_ids)) != 2 * world_size
    ):
        raise ValueError(
            "PREFILL_GPUS and DECODE_GPUS need WORLD_SIZE distinct GPUs each"
        )
    ports = {
        name: int(os.environ.get(name, str(default)))
        for name, default in (
            ("PREFILL_PORT", 18346),
            ("DECODE_PORT", 18347),
            ("PREFILL_BOOTSTRAP_PORT", 8998),
            ("LB_PORT", 18345),
            ("PROMETHEUS_PORT", 18422),
        )
    }
    dist_addrs = [
        os.environ.get("PREFILL_DIST_INIT_ADDR", "127.0.0.1:12579"),
        os.environ.get("DECODE_DIST_INIT_ADDR", "127.0.0.1:13580"),
    ]
    reserved_ports = list(ports.values())
    for address in dist_addrs:
        host, port = address.rsplit(":", 1)
        if host not in ("127.0.0.1", "localhost"):
            raise ValueError("Same-host rendezvous addresses must use loopback")
        reserved_ports.extend(int(port) + offset for offset in (0, 1, 3, 4, 5, 6))
    if len(set(reserved_ports)) != len(reserved_ports):
        raise ValueError(
            "Service ports and P/D control-plane clusters must be disjoint"
        )
    for port in reserved_ports:
        if not 0 < port < 65536:
            raise ValueError(f"Invalid port: {port}")
        with socket.socket() as sock:
            sock.bind(("127.0.0.1", port))

    startup_timeout = float(os.environ.get("PD_STARTUP_TIMEOUT", "2400"))
    gateway_timeout = float(os.environ.get("GATEWAY_STARTUP_TIMEOUT", "600"))
    shutdown_timeout = float(os.environ.get("WORKER_SHUTDOWN_TIMEOUT", "30"))
    request_timeout = int(os.environ.get("PD_REQUEST_TIMEOUT", "600"))
    if min(startup_timeout, gateway_timeout, shutdown_timeout, request_timeout) <= 0:
        raise ValueError("Timeouts must be positive")
    chunk_size = int(os.environ.get("CHUNKED_PREFILL_SIZE", "8192"))
    if not 0 < chunk_size <= 8192 or chunk_size % 64:
        raise ValueError("CHUNKED_PREFILL_SIZE must be a multiple of 64 in [64, 8192]")
    env = os.environ.copy()
    if env.get("TOKENSPEED_SKIP_GRPC_WARMUP", "1") != "1":
        raise ValueError("PD requires TOKENSPEED_SKIP_GRPC_WARMUP=1")
    env["TOKENSPEED_SKIP_GRPC_WARMUP"] = "1"
    env.setdefault("NO_PROXY", "*")
    env.setdefault("no_proxy", "*")
    from tokenspeed.cli._proc import wait_grpc_serving, wait_http_ready

    # Leave Mooncake's NVLink transport knobs untouched; do not force that path.
    log_dir = Path(env.get("PD_CI_LOG_DIR", ".ci-artifacts/pd-dots3-note-1p1d"))
    log_dir.mkdir(parents=True, exist_ok=True)
    common = [
        "--model",
        str(model),
        "--served-model-name",
        env.get("SERVED_MODEL_NAME", "dots3-note"),
        "--host",
        "127.0.0.1",
        "--world-size",
        str(world_size),
        "--tensor-parallel-size",
        str(world_size),
        "--data-parallel-size",
        "1",
        "--pipeline-parallel-size",
        "1",
        "--nnodes",
        "1",
        "--node-rank",
        "0",
        "--nprocs-per-node",
        str(world_size),
        "--dtype",
        "bfloat16",
        "--kv-cache-dtype",
        "bfloat16",
        "--quantization",
        "fp8",
        "--load-format",
        "auto",
        "--language-model-only",
        "--attention-backend",
        "dots3_note",
        "--moe-backend",
        "triton",
        "--sampling-backend",
        "triton",
        "--max-model-len",
        "8192",
        "--max-total-tokens",
        "32768",
        "--max-num-seqs",
        "8",
        "--chunked-prefill-size",
        str(chunk_size),
        "--max-cudagraph-capture-size",
        "8",
        "--cudagraph-capture-sizes",
        "1",
        "2",
        "4",
        "8",
        "--gpu-memory-utilization",
        env.get("GPU_MEMORY_UTILIZATION", "0.8"),
        "--prefix-granularity",
        "64",
        "--enable-prefix-caching",
        "--disable-kvstore",
        "--disable-symm-mem",
        "--comm-fusion-max-num-tokens",
        "0",
        "--disable-prefill-graph",
        "--disaggregation-transfer-backend",
        "mooncake",
        "--disaggregation-layerwise-interval",
        "0",
        *speculative_args,
    ]
    # There is only a disable-overlap CLI flag; omitting it keeps overlap on.
    # Likewise, omitting enforce-eager keeps decode graphs enabled.
    if env.get("DISAGGREGATION_IB_DEVICE"):
        common += ["--disaggregation-ib-device", env["DISAGGREGATION_IB_DEVICE"]]

    procs = []
    stop = asyncio.Event()
    exit_code = 1
    loop = asyncio.get_running_loop()

    def request_stop(signum):
        nonlocal exit_code
        exit_code = 128 + signum
        stop.set()

    for signum in (signal.SIGINT, signal.SIGTERM):
        loop.add_signal_handler(signum, request_stop, signum)

    def start(label, argv, child_env):
        log = log_dir / f"{label}.log"
        with log.open("w") as stream:
            proc = subprocess.Popen(
                argv,
                env=child_env,
                stdin=subprocess.DEVNULL,
                stdout=stream,
                stderr=subprocess.STDOUT,
                start_new_session=True,
            )
            # Register before closing the log, which can also fail on network storage.
            # No await here: a signal cannot interrupt spawn/registration.
            procs.append((label, proc))
        print(f"[dots3-pd] {label}: pid/pgid={proc.pid} log={log}", flush=True)

    async def serve():
        for role, devices, address, port in zip(
            ("prefill", "decode"),
            gpus,
            dist_addrs,
            (ports["PREFILL_PORT"], ports["DECODE_PORT"]),
        ):
            args = common + [
                "--port",
                str(port),
                "--dist-init-addr",
                address,
                "--disaggregation-mode",
                role,
            ]
            if role == "prefill":
                args += [
                    "--disaggregation-bootstrap-port",
                    str(ports["PREFILL_BOOTSTRAP_PORT"]),
                ]
            start(
                role,
                [sys.executable, "-m", "smg_grpc_servicer.tokenspeed", *args],
                {**env, "CUDA_VISIBLE_DEVICES": devices},
            )
        await asyncio.wait_for(
            asyncio.gather(
                *(
                    wait_grpc_serving(f"127.0.0.1:{port}", timeout=startup_timeout)
                    for port in (ports["PREFILL_PORT"], ports["DECODE_PORT"])
                )
            ),
            timeout=startup_timeout,
        )
        start(
            "lb",
            [
                sys.executable,
                "-m",
                "smg",
                "launch",
                "--pd-disaggregation",
                "--prefill",
                f"grpc://127.0.0.1:{ports['PREFILL_PORT']}",
                str(ports["PREFILL_BOOTSTRAP_PORT"]),
                "--decode",
                f"grpc://127.0.0.1:{ports['DECODE_PORT']}",
                "--host",
                env.get("LB_HOST", "127.0.0.1"),
                "--port",
                str(ports["LB_PORT"]),
                "--model-path",
                str(model),
                "--tokenizer-path",
                str(model),
                "--reasoning-parser",
                "passthrough",
                "--prefill-policy",
                "round_robin",
                "--decode-policy",
                "round_robin",
                "--max-concurrent-requests",
                "8",
                "--queue-size",
                "128",
                "--queue-timeout-secs",
                str(request_timeout),
                "--request-timeout-secs",
                str(request_timeout),
                "--health-check-timeout-secs",
                "120",
                "--disable-retries",
                "--disable-load-monitoring",
                "--disable-circuit-breaker",
                "--prometheus-port",
                str(ports["PROMETHEUS_PORT"]),
            ],
            env,
        )
        await asyncio.wait_for(
            wait_http_ready(
                f"http://127.0.0.1:{ports['LB_PORT']}/readiness",
                timeout=gateway_timeout,
            ),
            timeout=gateway_timeout,
        )
        print(f"[dots3-pd] ready: http://127.0.0.1:{ports['LB_PORT']}/v1", flush=True)
        await stop.wait()

    async def watch():
        while not stop.is_set():
            for label, proc in procs:
                if proc.poll() is not None:
                    raise RuntimeError(
                        f"{label} exited with {proc.returncode}; see {log_dir / f'{label}.log'}"
                    )
            await asyncio.sleep(0.2)

    tasks = [asyncio.create_task(serve()), asyncio.create_task(watch())]
    try:
        done, _ = await asyncio.wait(tasks, return_when=asyncio.FIRST_COMPLETED)
        for task in done:
            task.result()
        return exit_code
    finally:
        for task in tasks:
            task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)
        # _proc.terminate_then_kill stops only a leader. Signal every owned group
        # even if its leader already exited, so orphaned TP ranks cannot survive.
        for _, proc in reversed(procs):
            with contextlib.suppress(ProcessLookupError):
                os.killpg(proc.pid, signal.SIGTERM)
        deadline = time.monotonic() + shutdown_timeout
        for _, proc in procs:
            with contextlib.suppress(subprocess.TimeoutExpired):
                proc.wait(timeout=max(0, deadline - time.monotonic()))
        for _, proc in reversed(procs):
            with contextlib.suppress(ProcessLookupError):
                os.killpg(proc.pid, signal.SIGKILL)
        for _, proc in procs:
            proc.wait(timeout=5)
        print("[dots3-pd] owned process groups stopped", flush=True)


sys.exit(asyncio.run(main()))
PY
