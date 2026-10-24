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

"""Run a CPU op/layer metadata example without loading model weights."""

from __future__ import annotations

import argparse
from contextlib import nullcontext
from pathlib import Path

import torch
from tokenspeed_kernel.op_profile import (
    CostRegistry,
    DenseMatmulCost,
    OpProfile,
    ProfileSession,
    profiled,
)
from tokenspeed_kernel.registry import register_kernel
from tokenspeed_kernel.signature import format_signatures


@register_kernel(
    "gemm",
    "mm",
    name="example_cpu_mm",
    solution="reference",
    signatures=format_signatures(("a", "b"), "dense", {torch.float32}),
    profile=OpProfile(cost_rule="dense_mm"),
)
def cpu_mm(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    return torch.mm(a, b)


class ExampleLayer(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.register_buffer("up", torch.ones(8, 16))
        self.register_buffer("down", torch.ones(16, 8))

    @profiled(level="layer")
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return cpu_mm(cpu_mm(x, self.up), self.down)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--backend", choices=["none", "torch"], required=True)
    args = parser.parse_args()
    layer = ExampleLayer()
    session = ProfileSession(
        run_id="cpu-example", model_id="example", rank=0, backend=args.backend
    )
    session.bind_layer(layer, layer_id=7)
    x = torch.ones(4, 8)
    collector = (
        torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU])
        if args.backend == "torch"
        else nullcontext()
    )
    with collector as profiler:
        with session.iteration(iter_id=0):
            result = layer(x)
            repeated = layer(x)
    torch.testing.assert_close(result, torch.full((4, 8), 128.0))
    torch.testing.assert_close(repeated, result)
    costs = CostRegistry()
    costs.register("dense_mm", DenseMatmulCost(lhs="a", rhs="b", output="return"))
    session.export(args.output, costs=costs)
    if args.backend == "torch":
        profiler.export_chrome_trace(str(args.output.with_suffix(".trace.json")))
    summary = costs.analyze(session)["layers"][0]
    print(
        f"layer=7 calls=2 ops={summary['op_count']} covered={summary['covered_op_count']}"
    )
    print(f"cost={summary['complete_cost']}")
    print(f"Wrote {args.output}; CPU scope timing only, device MFU/MBU unavailable")


if __name__ == "__main__":
    main()
