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

"""TokenSpeed CLI entry point.

Usage:
    tokenspeed serve --model <path> [options]
    tokenspeed bench <bench_type> [options]
    tokenspeed env
    tokenspeed version

    ts serve --model <path> [options]
    ts bench <bench_type> [options]
    ts env
    ts version
"""

import argparse
import sys


def _serve(args: argparse.Namespace) -> None:
    from tokenspeed.cli.serve_native import run_native

    run_native(args)


def _bench(args: argparse.Namespace) -> None:
    from tokenspeed.bench import main as bench_main

    bench_main(args.bench_args)


def _env(args: argparse.Namespace) -> None:
    from tokenspeed.env import main as env_main

    env_main()


def _version(args: argparse.Namespace) -> None:
    from tokenspeed.version import __version__

    print(f"TokenSpeed v{__version__}")


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="tokenspeed",
        description="TokenSpeed is a speed-of-light LLM inference engine.",
    )

    subparsers = parser.add_subparsers(dest="command")

    # serve — only import heavy ServerArgs when the user actually invokes
    # the "serve" subcommand, so that lightweight commands like "version"
    # and "env" don't pull in torch / psutil / etc.
    serve_parser = subparsers.add_parser(
        "serve",
        help="Launch the TokenSpeed inference server.",
    )
    if len(sys.argv) >= 2 and sys.argv[1] == "serve":
        from tokenspeed.runtime.utils.server_args import ServerArgs

        ServerArgs.add_cli_args(serve_parser)
    serve_parser.set_defaults(func=_serve)

    # bench
    bench_parser = subparsers.add_parser(
        "bench",
        add_help=False,
        help="Run TokenSpeed benchmark commands.",
    )
    bench_parser.set_defaults(func=_bench, bench_args=[])

    # env
    env_parser = subparsers.add_parser(
        "env",
        help="Check environment configurations and dependency versions.",
    )
    env_parser.set_defaults(func=_env)

    # version
    version_parser = subparsers.add_parser(
        "version",
        help="Print the TokenSpeed version.",
    )
    version_parser.set_defaults(func=_version)

    args, extra_args = parser.parse_known_args()

    if args.command is None:
        parser.print_help()
        sys.exit(1)

    if args.func is _bench:
        args.bench_args = extra_args
    elif extra_args:
        parser.error(f"unrecognized arguments: {' '.join(extra_args)}")

    args.func(args)


if __name__ == "__main__":
    main()
