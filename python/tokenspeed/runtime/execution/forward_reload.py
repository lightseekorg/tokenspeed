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

"""Development: bring edited forward code into every serving rank in lockstep, then warm its kernels and recapture the graphs."""

from __future__ import annotations

import gc
import sys
import time
import traceback
from collections.abc import Callable, Sequence
from dataclasses import dataclass

import psutil
import torch.distributed as dist
from tokenspeed_kernel.compile_monitor import (
    compile_stats,
    mark_serving,
    reopen_startup,
)
from tokenspeed_kernel.hot_reload import (
    ModuleSource,
    ReloadReport,
    SourceTracker,
    dependent_modules,
    reload_modules,
)

from tokenspeed.runtime.engine.gc_hooks import freeze_startup_heap
from tokenspeed.runtime.utils import get_colorful_logger

logger = get_colorful_logger(__name__)

# Packages a reload re-executes: the code one forward runs.
RELOADABLE_PACKAGES = (
    "tokenspeed.runtime.models",
    "tokenspeed.runtime.layers",
    "tokenspeed.runtime.moe",
    "tokenspeed.runtime.sampling",
    "tokenspeed_kernel.ops",
    "tokenspeed_kernel.thirdparty",
)
# Packages watched for edits; an edited module outside RELOADABLE_PACKAGES needs a restart.
TRACKED_PACKAGES = ("tokenspeed.runtime", "tokenspeed_kernel")


def _reloadable(name: str) -> bool:
    return any(
        name == package or name.startswith(f"{package}.")
        for package in RELOADABLE_PACKAGES
    )


@dataclass(frozen=True)
class ReloadPlan:
    """What every rank re-executes, as the planning rank read it."""

    sources: list[ModuleSource]
    changed: list[str]
    restart_required: list[str]
    refusal: str | None = None


class ForwardReloader:
    """Reloads edited forward code on every rank in the same order and recaptures the graphs, restoring the old code on failure."""

    def __init__(
        self,
        *,
        group: dist.ProcessGroup | None,
        planner_rank: int,
        is_planner: bool,
    ) -> None:
        self._group = group
        self._planner_rank = planner_rank
        self._is_planner = is_planner
        self._tracker = SourceTracker(
            TRACKED_PACKAGES, loaded_at=psutil.Process().create_time()
        )
        if is_planner:
            # What the running sources bind, so a reload sees the names an edit drops.
            self._tracker.remember_bindings(RELOADABLE_PACKAGES)

    def reload(
        self,
        modules: Sequence[str] | None,
        *,
        warm: Callable[[], None],
        recapture: Callable[[], None],
    ) -> tuple[bool, str]:
        """Reload ``modules`` (every changed module when None), ``warm`` the kernels, then ``recapture``; returns the outcome and a report."""
        started = time.perf_counter()
        plan = self._broadcast(self._safe_plan(modules) if self._is_planner else None)
        if plan.refusal is not None:
            return False, f"forward reload refused: {plan.refusal}"
        report, error = self._agree(lambda: reload_modules(plan.sources))
        if error is not None:
            if report is not None:
                # This rank reloaded where a peer could not: every rank keeps the old code.
                report.revert()
            return False, (
                f"forward reload failed and the previous definitions were restored "
                f"(top-level side effects of the failed source are not undone): {error}"
            )
        before = compile_stats()
        reopen_startup(report.modules)
        # Startup froze the old graphs out of collection; their pools free only once collectable.
        gc.unfreeze()
        try:
            warm_started = time.perf_counter()
            _, error = self._agree(warm)
            if error is not None:
                self._restore(report)
                return False, (
                    f"warming the reloaded code failed, the previous code serves on "
                    f"with its graphs: {error}"
                )
            capture_started = time.perf_counter()
            _, error = self._agree(recapture)
            if error is not None:
                self._restore(report)
                _, again = self._agree(recapture)
                if again is not None:
                    raise RuntimeError(
                        f"recapture failed with the reloaded code ({error}) and again "
                        f"with the restored code ({again})"
                    )
                return False, (
                    f"capturing the reloaded code failed, the previous code serves on "
                    f"with recaptured graphs: {error}"
                )
            finished = time.perf_counter()
        finally:
            mark_serving()
            freeze_startup_heap()
        self._tracker.mark_loaded(plan.sources)
        message = _summary(
            plan,
            report,
            before,
            reload_seconds=warm_started - started,
            warm_seconds=capture_started - warm_started,
            capture_seconds=finished - capture_started,
        )
        if self._is_planner:
            logger.info(f"forward reload: {message}")
        return True, message

    def _safe_plan(self, modules: Sequence[str] | None) -> ReloadPlan:
        """Plan on this rank; a failure becomes a refusal every rank receives instead of an exception its peers wait on."""
        try:
            return self._plan(modules)
        except Exception as exc:
            return ReloadPlan([], [], [], refusal=_describe_error(exc))

    def _plan(self, modules: Sequence[str] | None) -> ReloadPlan:
        if modules is None:
            changed = self._tracker.stale()
            reloadable = [name for name in changed if _reloadable(name)]
        else:
            missing = [name for name in modules if name not in sys.modules]
            if missing:
                return ReloadPlan([], [], [], refusal=f"not loaded: {missing}")
            changed = list(modules)
            reloadable = changed
        restart_required = [name for name in changed if name not in reloadable]
        names = dependent_modules(reloadable, RELOADABLE_PACKAGES) if reloadable else []
        return ReloadPlan(self._tracker.read(names), reloadable, restart_required)

    def _restore(self, report: ReloadReport) -> None:
        """Undo this rank's reload with every peer; a rank that cannot is fatal."""
        _, error = self._agree(report.revert)
        if error is not None:
            raise RuntimeError(f"restoring the previous code failed: {error}")

    def _broadcast(self, plan: ReloadPlan | None) -> ReloadPlan:
        if self._group is None:
            return plan
        box = [plan]
        dist.broadcast_object_list(box, src=self._planner_rank, group=self._group)
        return box[0]

    def _agree(self, step: Callable[[], object]) -> tuple[object, str | None]:
        """Run ``step`` on this rank and return its result with the first rank's error, so every rank takes the same branch."""
        result = None
        error = None
        try:
            result = step()
        except Exception as exc:
            logger.exception("forward reload step failed")
            error = _describe_error(exc)
        if self._group is None:
            return result, error
        errors: list[str | None] = [None] * dist.get_world_size(self._group)
        dist.all_gather_object(errors, error, group=self._group)
        failed = [
            f"rank {rank}: {message}"
            for rank, message in enumerate(errors)
            if message is not None
        ]
        return result, (failed[0] if failed else None)


def _describe_error(exc: BaseException) -> str:
    """The exception with the innermost frame of the code that raised it."""
    frames = traceback.extract_tb(exc.__traceback__)
    where = f" at {frames[-1].filename}:{frames[-1].lineno}" if frames else ""
    return f"{type(exc).__name__}: {exc}{where}"


def _summary(
    plan: ReloadPlan,
    report: ReloadReport,
    before,
    *,
    reload_seconds: float,
    warm_seconds: float,
    capture_seconds: float,
) -> str:
    dependents = len(report.modules) - len(plan.changed)
    if plan.changed:
        parts = [
            f"reloaded {', '.join(plan.changed)} and {dependents} dependent "
            f"module(s), {report.patched} object(s) patched in {reload_seconds:.2f} s"
        ]
    else:
        parts = ["no reloadable module changed"]
    parts.append(f"warmed in {warm_seconds:.1f} s")
    parts.append(f"recaptured in {capture_seconds:.1f} s")
    after = compile_stats()
    if before is not None and after is not None:
        compiles = after.startup_compiles - before.startup_compiles
        seconds = after.startup_seconds - before.startup_seconds
        parts.append(f"JIT-compiled {compiles} kernel(s) in {seconds:.1f} s")
    if plan.restart_required:
        parts.append(
            f"NOT reloaded, outside the forward code (restart to apply): "
            f"{', '.join(plan.restart_required)}"
        )
    if report.redirected:
        parts.append(f"redirected to new definitions: {', '.join(report.redirected)}")
    if report.reset:
        parts.append(
            f"module state returned to its import-time value: {', '.join(report.reset)}"
        )
    if report.stale:
        parts.append(
            f"left stale where referenced before the reload: {', '.join(report.stale)}"
        )
    return "; ".join(parts)
