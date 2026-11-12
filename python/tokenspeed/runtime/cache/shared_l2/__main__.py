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

"""``python -m tokenspeed.runtime.cache.shared_l2 {owner,provider}``."""

from __future__ import annotations

import argparse
import functools
import logging
import signal
import threading


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="python -m tokenspeed.runtime.cache.shared_l2"
    )
    services = parser.add_subparsers(dest="service", required=True)
    owner = services.add_parser("owner", help="serve the pool's cache index")
    owner.add_argument("--address", required=True, help="ZMQ bind address")
    owner.add_argument(
        "--lcm-bytes",
        type=int,
        default=None,
        help="bytes of one pool LCM block; without it, the first rank's",
    )
    owner.add_argument("--providers", type=int, required=True)
    owner.add_argument("--headroom-lcm-blocks", type=int, required=True)
    owner.add_argument("--client-timeout-s", type=float, required=True)
    owner.add_argument(
        "--checkpoint-retention-requests",
        type=int,
        required=True,
        help="as the server's option, counted over every rank's lookups, hit or miss",
    )
    owner.add_argument("--checkpoint-retention-leaf-distance", type=int, required=True)
    provider = services.add_parser("provider", help="host one node's pool memory")
    provider.add_argument("--owner", required=True, help="the owner's ZMQ address")
    provider.add_argument("--hostname", required=True)
    provider.add_argument("--device-name", required=True)
    provider.add_argument("--bytes", type=int, required=True)
    args = parser.parse_args()
    if args.service == "owner" and (
        args.providers < 1 or args.headroom_lcm_blocks < 0 or args.client_timeout_s <= 0
    ):
        # No pool is built from these: said here, not to the first rank.
        parser.error(
            "the owner needs --providers of at least 1, --headroom-lcm-blocks of "
            "at least 0 and a positive --client-timeout-s"
        )
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s"
    )
    stop = threading.Event()
    for signum in (signal.SIGINT, signal.SIGTERM):
        signal.signal(signum, lambda *_: stop.set())
    if args.service == "owner":
        from tokenspeed.runtime.cache.shared_l2.owner import SharedL2Owner, make_pool

        service = SharedL2Owner(
            address=args.address,
            lcm_bytes=args.lcm_bytes,
            num_providers=args.providers,
            client_timeout_s=args.client_timeout_s,
            make_pool=functools.partial(
                make_pool,
                headroom_lcm_blocks=args.headroom_lcm_blocks,
                checkpoint_retention_requests=args.checkpoint_retention_requests,
                checkpoint_retention_leaf_distance=(
                    args.checkpoint_retention_leaf_distance
                ),
            ),
        )
        try:
            service.serve(stop)
        finally:
            service.close()
    else:
        from tokenspeed.runtime.cache.shared_l2.provider import run_provider

        run_provider(
            owner=args.owner,
            hostname=args.hostname,
            device_name=args.device_name,
            size_bytes=args.bytes,
            stop=stop,
        )


if __name__ == "__main__":
    main()
