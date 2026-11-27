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

Both actions go through the SMG gateway's HTTP routes, which reach every
engine rank over the gateway's ZMQ lane.

wait-ready: wait until the gateway reports every attention-DP rank of both
roles idle and answers a short chat request.
reset: on idle engines, flush the prefill and decode GPU caches, then empty
the shared L2 pool, so each benchmark point starts cold.
"""

import argparse
import json
import time
import urllib.error
import urllib.request

from tokenspeed.runtime.cache.shared_l2.wire import OwnerConnection

MODEL = "kimi-k3-agentx-pd"


def http(gateway, path, body=None, timeout=60):
    data = None if body is None else json.dumps(body).encode()
    request = urllib.request.Request(
        gateway + path,
        data=data,
        method="GET" if body is None else "POST",
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return json.loads(response.read() or b"null")


def idle_ranks(gateway, dp_sizes):
    """Ranks the gateway polled, by role; raises if a rank is missing or busy."""
    report = http(gateway, "/get_loads")
    ranks = {}
    for row in report["loads"]:
        ranks.setdefault(row["worker_type"], set()).add(row["dp_rank"])
        busy = row["num_running_reqs"] or row["num_waiting_reqs"]
        if busy:
            raise RuntimeError(
                f"{row['worker_type']} rank {row['dp_rank']} still has requests"
            )
    for role, dp_size in dp_sizes.items():
        if ranks.get(role) != set(range(dp_size)):
            raise RuntimeError(
                f"incomplete {role} rank telemetry: {sorted(ranks.get(role, ()))}"
            )


def wait_ready(gateway, dp_sizes, timeout_s):
    deadline = time.monotonic() + timeout_s
    while True:
        try:
            idle_ranks(gateway, dp_sizes)
            reply = http(
                gateway,
                "/v1/chat/completions",
                {
                    "model": MODEL,
                    "messages": [{"role": "user", "content": "Hello"}],
                    "max_tokens": 8,
                },
                timeout=600,
            )
            print(
                json.dumps(
                    {"ready": True, "finish": reply["choices"][0]["finish_reason"]}
                ),
                flush=True,
            )
            return
        except (OSError, urllib.error.URLError, RuntimeError, KeyError) as error:
            if time.monotonic() > deadline:
                raise TimeoutError(str(error)) from error
            print(json.dumps({"pending": str(error)}), flush=True)
        time.sleep(20)


def reset(gateway, dp_sizes, owner):
    idle_ranks(gateway, dp_sizes)
    print(
        json.dumps({"flush_cache": http(gateway, "/flush_cache", {}, timeout=120)}),
        flush=True,
    )
    # The gateway polls loads once a second; let it see the flushed ranks.
    time.sleep(3)
    idle_ranks(gateway, dp_sizes)
    connection = OwnerConnection(owner, timeout_ms=60_000)
    try:
        reply = connection.request("clear")
    finally:
        connection.close()
    if not reply["cleared"]:
        raise RuntimeError("shared L2 pool not cleared; a rank still holds slots")
    print(json.dumps({"shared_l2_cleared": True}), flush=True)


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("action", choices=["wait-ready", "reset"])
    parser.add_argument(
        "--gateway", required=True, help="gateway URL, http://host:port"
    )
    parser.add_argument("--prefill-dp-size", type=int, required=True)
    parser.add_argument("--decode-dp-size", type=int, required=True)
    parser.add_argument(
        "--shared-l2-owner", help="owner address for reset, tcp://host:port"
    )
    parser.add_argument("--timeout-s", type=float, help="wait-ready deadline")
    args = parser.parse_args()
    dp_sizes = {"prefill": args.prefill_dp_size, "decode": args.decode_dp_size}
    if args.action == "wait-ready":
        if args.timeout_s is None:
            parser.error("wait-ready requires --timeout-s")
        wait_ready(args.gateway, dp_sizes, args.timeout_s)
    else:
        if args.shared_l2_owner is None:
            parser.error("reset requires --shared-l2-owner")
        reset(args.gateway, dp_sizes, args.shared_l2_owner)


if __name__ == "__main__":
    main()
