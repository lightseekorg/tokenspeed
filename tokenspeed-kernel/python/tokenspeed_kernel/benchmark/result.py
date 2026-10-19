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

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

__all__ = [
    "BenchmarkResult",
    "export_results",
    "import_results",
]


@dataclass
class BenchmarkResult:
    """Result of benchmarking a single kernel on a single shape."""

    kernel_name: str
    op_family: str
    op_mode: str
    solution: str
    dtype: str
    platform_arch: str
    shape_params: dict[str, Any]

    median_latency_us: float
    p90_latency_us: float
    p99_latency_us: float
    min_latency_us: float
    max_latency_us: float

    tflops: float | None = None
    bandwidth_gb_s: float | None = None

    numerics_passed: bool | None = None
    max_abs_diff: float | None = None
    max_rel_diff: float | None = None

    timestamp: str = ""
    num_iters: int = 0

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "BenchmarkResult":
        return cls(**data)


def export_results(
    results: list[BenchmarkResult],
    path: str | Path,
) -> None:
    payload = [result.to_dict() for result in results]
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8"
    )


def import_results(path: str | Path) -> list[BenchmarkResult]:
    input_path = Path(path)
    raw = json.loads(input_path.read_text(encoding="utf-8"))
    if not isinstance(raw, list):
        raise ValueError(f"Expected a list of benchmark results in {input_path}")
    return [BenchmarkResult.from_dict(item) for item in raw]
