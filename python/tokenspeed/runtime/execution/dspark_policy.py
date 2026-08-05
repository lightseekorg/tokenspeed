"""Conservative Kimi-K3 DSpark pool-routing policy.

The policy selects between separately launched no-spec, W4 and W8 pools. It
never changes verify width inside a CUDA graph.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

DSparkMode = Literal["no-spec", "w4", "w8"]


@dataclass(frozen=True)
class DSparkRoute:
    mode: DSparkMode
    reason: str


def route_kimi_k3_dspark(
    *,
    input_tokens: int,
    max_new_tokens: int,
    concurrency: int,
    workload: str | None = None,
    confidence: float | None = None,
    confidence_calibrated: bool = False,
) -> DSparkRoute:
    workload = (workload or "").lower()
    if input_tokens >= 32768:
        return DSparkRoute("no-spec", "32K+ context collapses draft acceptance")

    if confidence_calibrated and confidence is not None and confidence < 0.2:
        return DSparkRoute("no-spec", "calibrated confidence below 0.20")

    if workload in {"aime", "math-reasoning", "reasoning"} and concurrency >= 8:
        return DSparkRoute("w8", "reasoning c8+ uses the validated W8 pool")

    if input_tokens <= 4096 and max_new_tokens >= 256:
        return DSparkRoute("w4", "short-context generation favors W4 break-even")

    if input_tokens <= 8192 and concurrency >= 8:
        return DSparkRoute("w4", "batched context fits the validated W4 pool")

    return DSparkRoute("no-spec", "no validated >=1.30x DSpark operating point")
