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

"""Start one rank in the launching Python environment."""

import argparse
import signal
import sys

import setproctitle


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--control-fd", type=int, required=True)
    parser.add_argument(
        "--enable-async-logging", choices=("true", "false"), required=True
    )
    parser.add_argument("--process-title", required=True)
    args = parser.parse_args()
    # Rust owns shutdown. Do not chain CPython's KeyboardInterrupt handler.
    signal.signal(signal.SIGINT, signal.SIG_DFL)
    signal.signal(signal.SIGTERM, signal.SIG_DFL)
    setproctitle.setproctitle(args.process_title)
    if "tokenspeed_scheduler" in sys.modules:
        raise RuntimeError("the scheduler must be imported after host initialization")
    from tokenspeed_engine._host import run_host, scheduler

    # As in the embedded host, the external Python wrapper must use the C++
    # scheduler linked into this host, so plans share their native types.
    sys.modules["tokenspeed_scheduler.tokenspeed_scheduler_ext"] = scheduler
    run_host(args.control_fd, args.enable_async_logging == "true")


if __name__ == "__main__":
    main()
