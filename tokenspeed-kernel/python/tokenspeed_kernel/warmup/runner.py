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

from dataclasses import dataclass

from tokenspeed_kernel.registry import KernelApiSpec, KernelRegistry, WarmupBehavior
from tokenspeed_kernel.warmup.api import WarmupConfig
from tokenspeed_kernel.warmup.config.schema import WarmupProfile


@dataclass(frozen=True)
class ValidatedWarmupTarget:
    api_spec: KernelApiSpec
    solution: str
    config: WarmupConfig


def validate_profile(profile: WarmupProfile) -> tuple[ValidatedWarmupTarget, ...]:
    registry = KernelRegistry.get()
    validated = []
    for target in profile.targets:
        family, mode = target.api.split(".")
        api_spec = registry.get_api(family, mode)
        if api_spec is None:
            raise ValueError(f"unknown kernel API {target.api!r}")
        if api_spec.warmup_config_type is None:
            raise ValueError(f"kernel API {target.api!r} has no warmup definition")
        solution_specs = registry.get_for_operator(
            family,
            mode,
            solution=target.solution,
        )
        if not solution_specs:
            raise ValueError(
                f"unknown solution {target.solution!r} for kernel API {target.api!r}"
            )
        if all(spec.warmup_behavior is WarmupBehavior.NONE for spec in solution_specs):
            raise ValueError(
                f"solution {target.solution!r} for kernel API {target.api!r} "
                "has no warmup behavior"
            )
        config = api_spec.warmup_config_type.from_json(target.definition)
        validated.append(
            ValidatedWarmupTarget(
                api_spec=api_spec,
                solution=target.solution,
                config=config,
            )
        )
    return tuple(validated)
