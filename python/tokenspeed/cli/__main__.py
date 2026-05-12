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
    ts serve --model <path> [options]
    ts bench <bench_type> [options]
    ts env
    ts version

``ts serve`` dispatches to the smg-fronted orchestrator, which spawns the
smg gateway and the gRPC engine as two child processes. Engine-only flags
go to the engine, gateway-only flags go to the gateway, and a small set of
flags (e.g. ``--model``) fan out to both. See ``tokenspeed/cli/serve_smg.py``
and ``_argsplit.py`` for the routing rules.
"""

import argparse
import sys


def _serve(args: argparse.Namespace, raw_argv: list[str]) -> None:
    from tokenspeed.cli.serve_smg import run_smg_from_args

    run_smg_from_args(args, raw_argv)


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

    # serve — unknown flags must fall through to the smg orchestrator
    # (which splits them between the engine and gateway subprocesses), so
    # we don't register the engine's ServerArgs on this parser. ``--help``
    # still works because argparse adds it by default.
    serve_parser = subparsers.add_parser(
        "serve",
        help="Launch the TokenSpeed inference server.",
    )
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
        args.func(args)
        return

    if args.func is _serve:
        # Hand the raw remainder to the orchestrator's own argv splitter.
        raw = list(sys.argv[2:])  # everything after ``ts serve``
        args.func(args, raw)
        return

    if extra_args:
        parser.error(f"unrecognized arguments: {' '.join(extra_args)}")
    args.func(args)


if __name__ == "__main__":
    main()
