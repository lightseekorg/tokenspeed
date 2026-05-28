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
import json
from typing import Iterable

import torch
from tokenspeed_kernel.numerics.comparison import format_comparison
from tokenspeed_kernel.numerics.verify import verify_kernel
from tokenspeed_kernel.platform import Platform
from tokenspeed_kernel.registry import KernelRegistry, KernelSpec, load_builtin_kernels

_DTYPE_SELECTIONS: dict[str, torch.dtype] = {
    "fp32": torch.float32,
    "fp16": torch.float16,
    "bf16": torch.bfloat16,
    "fp8": Platform.get().fp8e4m3fn.dtype,
}


def _parse_shapes(raw: str | None) -> list[dict] | None:
    if raw is None:
        return None
    obj = json.loads(raw)
    if isinstance(obj, dict):
        return [obj]
    if isinstance(obj, list) and all(isinstance(item, dict) for item in obj):
        return obj
    raise ValueError("--shapes must be a JSON object or list of objects")


def _parse_op(raw: str | None) -> tuple[str, str] | None:
    if raw is None:
        return None
    if "." not in raw:
        raise ValueError("--op must be in family.mode format, e.g. gemm.mm")
    family, mode = raw.split(".", 1)
    return family, mode


def _iter_candidate_specs(
    registry: KernelRegistry,
    *,
    kernel_name: str | None,
    op_filter: tuple[str, str] | None,
    dtype_filter: torch.dtype | None,
) -> list[KernelSpec]:
    if kernel_name is not None:
        spec = registry.get_by_name(kernel_name)
        if spec is None:
            raise ValueError(f"Kernel {kernel_name!r} is not registered")
        specs = [spec]
    else:
        specs = [
            spec for spec in registry.list_kernels() if spec.solution != "reference"
        ]

    if op_filter is not None:
        family, mode = op_filter
        specs = [s for s in specs if s.family == family and s.mode == mode]

    if dtype_filter is not None:
        specs = [
            s
            for s in specs
            if s.format_signature_for_primary_storage_dtype(dtype_filter) is not None
        ]

    specs.sort(key=lambda s: (s.family, s.mode, s.name))
    return specs


def _iter_dtypes(
    spec: KernelSpec,
    dtype_filter: torch.dtype | None,
) -> Iterable[torch.dtype]:
    if dtype_filter is not None:
        return (dtype_filter,)
    return sorted(spec.primary_storage_dtypes(), key=str)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Verify kernel numerics")
    parser.add_argument("kernel_name", nargs="?", help="Filter by kernel name")
    parser.add_argument("--op", help="Filter by operator family.mode")
    parser.add_argument(
        "--dtype",
        choices=sorted(_DTYPE_SELECTIONS),
        help="Filter by dtype selection",
    )
    parser.add_argument(
        "--shapes",
        help="JSON object or list of shape objects override",
    )
    parser.add_argument("--verbose", action="store_true", help="Verbose output")
    args = parser.parse_args(argv)

    dtype_filter = _DTYPE_SELECTIONS[args.dtype] if args.dtype is not None else None
    op_filter = _parse_op(args.op)
    shapes = _parse_shapes(args.shapes)

    load_builtin_kernels()
    registry = KernelRegistry.get()
    specs = _iter_candidate_specs(
        registry,
        kernel_name=args.kernel_name,
        op_filter=op_filter,
        dtype_filter=dtype_filter,
    )

    if not specs:
        if args.verbose:
            print("[INFO] No kernels matched the provided filters")
        return 0

    failing = False
    ran = False
    for spec in specs:
        for dtype in _iter_dtypes(spec, dtype_filter):
            ran = True
            try:
                results = verify_kernel(
                    spec.name,
                    shapes=shapes,
                    dtype=dtype,
                    verbose=False,
                )
            except Exception as exc:
                print(f"[ERROR] {spec.family}.{spec.mode}:{dtype}:{spec.name}: {exc}")
                failing = True
                continue

            if not results:
                failing = True
                continue

            for i, result in enumerate(results):
                label = f"{spec.family}.{spec.mode}:{dtype}:{spec.name}[{i}]"
                print(format_comparison(result, label))
                failing = failing or (not result.passed)

    if not ran:
        if args.verbose:
            print("[INFO] No kernel+dtype combinations matched the provided filters")
        return 0
    return 1 if failing else 0
