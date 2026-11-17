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


"""``ts reload-forward``: ask a development server to reload its edited forward code and recapture its graphs."""

from __future__ import annotations

import argparse
import json
import sys
import time
import urllib.error
import urllib.request
import uuid

# How long past its own timeout an engine may take to answer: the recapture after the wait.
_REPLY_MARGIN_S = 1800.0


def add_arguments(parser: argparse.ArgumentParser) -> None:
    """Register the command's flags on ``parser``."""
    parser.add_argument(
        "--url",
        default="http://127.0.0.1:8000",
        help="The gateway of a server started with --allow-reload-forward.",
    )
    parser.add_argument(
        "--mode",
        choices=["wait", "abort"],
        default="wait",
        help="How the drain treats running requests: let them finish, or end them.",
    )
    parser.add_argument(
        "--modules",
        nargs="+",
        default=None,
        help="Modules to re-execute; by default every module whose source changed.",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=3600.0,
        help="Seconds the engine may wait to drain and for every DP engine to hold the reload.",
    )


def _request(url: str, body: dict | None, timeout: float) -> tuple[int, dict]:
    data = None if body is None else json.dumps(body).encode()
    request = urllib.request.Request(
        url, data=data, headers={"Content-Type": "application/json"}
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return response.status, json.loads(response.read() or b"{}")
    except urllib.error.HTTPError as error:
        return error.code, json.loads(error.read() or b"{}")


def run(args: argparse.Namespace) -> int:
    """Reload every worker behind the gateway in turn; returns the exit status."""
    base = args.url.rstrip("/")
    status, listing = _request(f"{base}/v1/rl/workers", None, 60.0)
    if status != 200:
        print(f"cannot list workers at {base}/v1/rl/workers (HTTP {status}): {listing}")
        return 1
    # One id for every engine, so their gates run the same reload together.
    body: dict = {"mode": args.mode, "id": uuid.uuid4().hex, "timeout_s": args.timeout}
    if args.modules is not None:
        body["modules"] = args.modules
    failed = False
    for worker in listing["workers"]:
        started = time.monotonic()
        status, reply = _request(
            f"{base}/v1/rl/workers/{worker['id']}/engine/reload_forward",
            body,
            # The engine answers once the reload ran or timed out.
            args.timeout + _REPLY_MARGIN_S,
        )
        elapsed = time.monotonic() - started
        print(f"{worker['id']} ({elapsed:.1f} s, HTTP {status}): {json.dumps(reply)}")
        # HTTP 200 still carries a refused or undone reload, as success false.
        failed |= status != 200 or reply.get("body", {}).get("success") is not True
    return 1 if failed else 0


def main(args: argparse.Namespace) -> None:
    sys.exit(run(args))
