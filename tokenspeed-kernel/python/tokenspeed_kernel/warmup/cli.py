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

from __future__ import annotations

import argparse

from tokenspeed_kernel.platform import current_platform
from tokenspeed_kernel.registry import (
    KernelRegistry,
    WarmupBehavior,
    load_builtin_kernels,
)
from tokenspeed_kernel.warmup.bundle import generate_bundle
from tokenspeed_kernel.warmup.discovery import list_config_ids, load_config
from tokenspeed_kernel.warmup.runner import validate_profile


def _parse_api(value: str) -> tuple[str, str]:
    if value.count(".") != 1:
        raise ValueError(f"API must be in family.mode form, got {value!r}")
    family, mode = value.split(".")
    if not family or not mode:
        raise ValueError(f"API must be in family.mode form, got {value!r}")
    return family, mode


def main(argv: list[str]) -> int:
    """List warmup APIs and the solutions available on this platform."""
    parser = argparse.ArgumentParser(
        prog="python -m tokenspeed_kernel.warmup",
        description="Warm registered TokenSpeed kernel APIs",
    )
    action = parser.add_mutually_exclusive_group(required=True)
    action.add_argument(
        "--list",
        dest="list_configs",
        action="store_true",
        help="List built-in warmup configurations",
    )
    action.add_argument(
        "--list-apis",
        action="store_true",
        help="List public APIs registered for warmup",
    )
    action.add_argument(
        "--list-solutions",
        metavar="API",
        help="List solutions available for one family.mode API",
    )
    action.add_argument(
        "--show",
        metavar="CONFIG",
        help="Show one built-in warmup configuration",
    )
    action.add_argument(
        "--validate",
        metavar="CONFIG",
        help="Validate one built-in warmup configuration",
    )
    action.add_argument(
        "--config",
        metavar="CONFIG",
        help="Generate a bundle from one built-in warmup configuration",
    )
    parser.add_argument(
        "--output-dir",
        metavar="PATH",
        help="Output directory for a generated warmup bundle",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Replace an existing complete output bundle",
    )
    args = parser.parse_args(argv)
    if args.config is None and (args.output_dir is not None or args.force):
        parser.error("--output-dir and --force require --config")
    if args.config is not None and args.output_dir is None:
        parser.error("--config requires --output-dir")

    if args.list_configs:
        for config_id in list_config_ids():
            print(config_id)
        return 0

    if args.show is not None:
        try:
            loaded = load_config(args.show)
        except (TypeError, ValueError) as error:
            parser.error(str(error))
        print(loaded.source, end="" if loaded.source.endswith("\n") else "\n")
        return 0

    load_builtin_kernels()
    registry = KernelRegistry.get()
    if args.list_apis:
        for spec in registry.list_apis():
            if spec.warmup_config_type is not None:
                print(spec.api)
        return 0

    if args.validate is not None:
        try:
            loaded = load_config(args.validate)
            validate_profile(loaded.profile)
        except (TypeError, ValueError) as error:
            parser.error(str(error))
        print(f"{loaded.profile.id}: valid")
        return 0

    if args.config is not None:
        try:
            loaded = load_config(args.config)
        except (TypeError, ValueError) as error:
            parser.error(str(error))
        output = generate_bundle(
            loaded=loaded,
            output_dir=args.output_dir,
            force=args.force,
            platform=current_platform(),
            command=("python", "-m", "tokenspeed_kernel.warmup", *argv),
        )
        print(output)
        return 0

    try:
        family, mode = _parse_api(args.list_solutions)
    except ValueError as error:
        parser.error(str(error))

    api_spec = registry.get_api(family, mode)
    if api_spec is None:
        parser.error(f"unknown kernel API {args.list_solutions!r}")
    if api_spec.warmup_config_type is None:
        parser.error(f"kernel API {args.list_solutions!r} has no warmup definition")

    platform = current_platform()
    solutions = sorted(
        {
            spec.solution
            for spec in registry.get_for_operator(
                family,
                mode,
                platform=platform,
            )
            if spec.warmup_behavior is not WarmupBehavior.NONE
        }
    )
    for solution in solutions:
        print(solution)
    return 0
