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
    owner.add_argument("--lcm-bytes", type=int, required=True)
    owner.add_argument("--providers", type=int, required=True)
    owner.add_argument("--headroom-lcm-blocks", type=int, required=True)
    owner.add_argument("--client-timeout-s", type=float, required=True)
    provider = services.add_parser("provider", help="host one node's pool memory")
    provider.add_argument("--owner", required=True, help="the owner's ZMQ address")
    provider.add_argument("--hostname", required=True)
    provider.add_argument("--device-name", required=True)
    provider.add_argument("--bytes", type=int, required=True)
    args = parser.parse_args()
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s"
    )
    stop = threading.Event()
    for signum in (signal.SIGINT, signal.SIGTERM):
        signal.signal(signum, lambda *_: stop.set())
    if args.service == "owner":
        from tokenspeed.runtime.cache.shared_l2.owner import SharedL2Owner

        service = SharedL2Owner(
            address=args.address,
            lcm_bytes=args.lcm_bytes,
            num_providers=args.providers,
            headroom_lcm_blocks=args.headroom_lcm_blocks,
            client_timeout_s=args.client_timeout_s,
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
