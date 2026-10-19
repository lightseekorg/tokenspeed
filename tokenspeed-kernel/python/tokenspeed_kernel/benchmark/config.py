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

from dataclasses import dataclass, field

from tokenspeed_kernel.profiling import ProfilingConfig

__all__ = ["BenchmarkConfig"]


@dataclass
class BenchmarkConfig:
    warmup_iters: int = 10
    bench_iters: int = 100
    verify: bool = True
    percentiles: list[float] = field(default_factory=lambda: [50.0, 90.0, 99.0])
    use_cuda_events: bool = True
    seed: int = 42
    proton_profile: bool = False
    proton_config: ProfilingConfig | None = None

    def validate(self) -> None:
        if self.warmup_iters < 0:
            raise ValueError("warmup_iters must be >= 0")
        if self.bench_iters <= 0:
            raise ValueError("bench_iters must be > 0")
        if not isinstance(self.verify, bool):
            raise ValueError("verify must be a bool")
        if not self.percentiles:
            raise ValueError("percentiles must not be empty")
        for percentile in self.percentiles:
            if percentile < 0 or percentile > 100:
                raise ValueError("percentiles must be in [0, 100]")

        if not isinstance(self.proton_profile, bool):
            raise ValueError("proton_profile must be a bool")

        if self.proton_config is None:
            return

        if not isinstance(self.proton_config, ProfilingConfig):
            raise ValueError("proton_config must be a ProfilingConfig")

        if not self.proton_config.output:
            raise ValueError("proton_config.output must be non-empty")

        if self.proton_config.data not in {"tree", "trace"}:
            raise ValueError("proton_config.data must be one of: tree, trace")

        if self.proton_config.backend not in {None, "cupti", "roctracer"}:
            raise ValueError(
                "proton_config.backend must be one of: None, cupti, roctracer"
            )

        if self.proton_config.mode not in {None, "pcsampling", "periodic_flushing"}:
            raise ValueError(
                "proton_config.mode must be one of: None, pcsampling, periodic_flushing"
            )

        if self.proton_config.hook not in {None, "triton"}:
            raise ValueError("proton_config.hook must be one of: None, triton")

        if self.proton_config.output_format not in {
            "",
            "hatchet",
            "hatchet_msgpack",
            "chrome_trace",
        }:
            raise ValueError(
                "proton_config.output_format must be one of: '', hatchet, "
                "hatchet_msgpack, chrome_trace"
            )
