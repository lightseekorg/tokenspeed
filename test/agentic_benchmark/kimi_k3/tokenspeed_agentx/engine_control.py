# MIT License
#
# Copyright (c) 2026 LightSeek Foundation <contact@lightseek.org>
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

"""Readiness and cache reset for the DWDP16 + DEP16 AgentX deployment.

wait-ready: wait until both gRPC endpoints serve and report every attention-DP
rank idle.
reset: on idle engines, flush the prefill and decode GPU caches, then empty
the shared L2 pool, so each benchmark point starts cold.
"""

import argparse
import json
import time

import grpc
from grpc_health.v1 import health_pb2, health_pb2_grpc
from smg_grpc_proto.generated import (
    common_pb2,
    tokenspeed_scheduler_pb2,
    tokenspeed_scheduler_pb2_grpc,
)

from tokenspeed.runtime.cache.shared_l2.wire import OwnerConnection


def idle_ranks(stub, dp_size):
    """Rank loads of an idle endpoint; raises if a rank is missing or busy."""
    response = stub.GetLoads(tokenspeed_scheduler_pb2.GetLoadsRequest(), timeout=30)
    ranks = sorted(row.dp_rank for row in response.loads)
    if response.dp_rank_count != dp_size or ranks != list(range(dp_size)):
        raise RuntimeError(
            f"incomplete rank telemetry: count={response.dp_rank_count}, ranks={ranks}"
        )
    for row in response.loads:
        if row.num_running_reqs or row.num_waiting_reqs or row.num_total_reqs:
            raise RuntimeError(f"rank {row.dp_rank} still has requests")
    return dp_size


def wait_ready(endpoints, timeout_s):
    deadline = time.monotonic() + timeout_s
    while True:
        pending = []
        for role, address, dp_size in endpoints:
            try:
                with grpc.insecure_channel(address) as channel:
                    health = health_pb2_grpc.HealthStub(channel).Check(
                        health_pb2.HealthCheckRequest(), timeout=10
                    )
                    if health.status != health_pb2.HealthCheckResponse.SERVING:
                        pending.append(f"{role}: not serving")
                        continue
                    idle_ranks(
                        tokenspeed_scheduler_pb2_grpc.TokenSpeedSchedulerStub(channel),
                        dp_size,
                    )
            except (grpc.RpcError, RuntimeError) as error:
                pending.append(f"{role}: {error}")
        if not pending:
            print(json.dumps({"ready": [role for role, _, _ in endpoints]}), flush=True)
            return
        if time.monotonic() > deadline:
            raise TimeoutError("; ".join(pending))
        time.sleep(20)


def reset(endpoints, owner):
    channels = [
        (role, grpc.insecure_channel(address), dp_size)
        for role, address, dp_size in endpoints
    ]
    stubs = [
        (role, tokenspeed_scheduler_pb2_grpc.TokenSpeedSchedulerStub(channel), dp_size)
        for role, channel, dp_size in channels
    ]
    for role, stub, dp_size in stubs:
        idle_ranks(stub, dp_size)
    for role, stub, _ in stubs:
        response = stub.FlushCache(
            common_pb2.FlushCacheRequest(timeout_s=60), timeout=90
        )
        if not response.success:
            raise RuntimeError(f"{role} rejected the cache flush: {response.message}")
    for role, stub, dp_size in stubs:
        idle_ranks(stub, dp_size)
    reply = OwnerConnection(owner, timeout_ms=60_000).request("clear")
    if not reply["cleared"]:
        raise RuntimeError("shared L2 pool not cleared; a rank still holds slots")
    for _, channel, _ in channels:
        channel.close()
    print(
        json.dumps(
            {"flushed": [role for role, _, _ in endpoints], "shared_l2_cleared": True}
        ),
        flush=True,
    )


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("action", choices=["wait-ready", "reset"])
    parser.add_argument(
        "--prefill", required=True, help="prefill gRPC address, host:port"
    )
    parser.add_argument(
        "--decode", required=True, help="decode gRPC address, host:port"
    )
    parser.add_argument("--prefill-dp-size", type=int, required=True)
    parser.add_argument("--decode-dp-size", type=int, required=True)
    parser.add_argument(
        "--shared-l2-owner", help="owner address for reset, tcp://host:port"
    )
    parser.add_argument("--timeout-s", type=float, help="wait-ready deadline")
    args = parser.parse_args()
    endpoints = [
        ("prefill", args.prefill, args.prefill_dp_size),
        ("decode", args.decode, args.decode_dp_size),
    ]
    if args.action == "wait-ready":
        if args.timeout_s is None:
            parser.error("wait-ready requires --timeout-s")
        wait_ready(endpoints, args.timeout_s)
    else:
        if args.shared_l2_owner is None:
            parser.error("reset requires --shared-l2-owner")
        reset(endpoints, args.shared_l2_owner)


if __name__ == "__main__":
    main()
