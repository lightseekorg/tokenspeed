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

"""Host metadata collection and offline costs for registered ops and layers."""

from __future__ import annotations

import inspect
import json
import time
from contextlib import contextmanager, nullcontext
from contextvars import ContextVar
from dataclasses import asdict, dataclass
from functools import wraps
from pathlib import Path
from typing import Any, Callable, Iterator
from urllib.parse import quote

import torch


@dataclass(frozen=True)
class TensorDescription:
    shape: tuple[int, ...]
    dtype: str
    stride: tuple[int, ...]
    device: str
    logical_bytes: int


def _describe(value: Any, path: str, result: dict[str, TensorDescription]) -> None:
    if isinstance(value, torch.Tensor):
        result[path] = TensorDescription(
            tuple(value.shape),
            str(value.dtype),
            tuple(value.stride()),
            str(value.device),
            value.numel() * value.element_size(),
        )
    elif isinstance(value, (tuple, list)):
        for index, item in enumerate(value):
            _describe(item, f"{path}[{index}]", result)
    elif isinstance(value, dict):
        for key, item in value.items():
            _describe(item, f"{path}[{key!r}]", result)


@dataclass(frozen=True)
class OpProfile:
    """Opt in a registered function; cost_rule=None records metadata only."""

    cost_rule: str | None


@dataclass(frozen=True)
class OpRecord:
    iter_id: int
    layer_id: int | None
    occurrence: int | None
    inner_id: int
    op_name: str
    implementation: str
    cost_rule: str | None
    inputs: dict[str, TensorDescription]
    outputs: dict[str, TensorDescription]
    cpu_start_ns: int
    cpu_duration_ns: int
    error: str | None
    trace_label: str


@dataclass(frozen=True)
class LayerRecord:
    iter_id: int
    layer_id: int
    occurrence: int
    cpu_start_ns: int
    cpu_duration_ns: int
    error: str | None
    trace_label: str


@dataclass(frozen=True)
class _Context:
    session: ProfileSession
    iter_id: int
    layer_id: int | None
    occurrence: int | None


_current: ContextVar[_Context | None] = ContextVar("op_profile_context", default=None)
_recording_op: ContextVar[bool] = ContextVar("op_profile_recording_op", default=False)


class ProfileSession:
    """Collect host metadata within explicit, synchronous iteration windows.

    Args:
        run_id: Artifact namespace.
        model_id: Model identity, including target/draft role when applicable.
        rank: Global process rank.
        backend: Explicit scope backend: none, nvtx or torch. Collectors are external.
    """

    def __init__(self, *, run_id: str, model_id: str, rank: int, backend: str) -> None:
        if backend not in {"none", "nvtx", "torch"}:
            raise ValueError(f"Unsupported recording backend: {backend}")
        self.backend = backend
        self.run_id = run_id
        self.model_id = model_id
        self.rank = rank
        self.ops: list[OpRecord] = []
        self.layers: list[LayerRecord] = []
        self._identities: dict[int, tuple[object, int]] = {}
        self._iterations: set[int] = set()
        self._active = False
        self._counts: dict[tuple[int, int | None], int] = {}
        self._occurrences: dict[tuple[int, int], int] = {}

    def record_func(self, label: str):
        """Return the selected annotation scope; collector lifetime stays external."""
        if self.backend == "nvtx":
            return torch.cuda.nvtx.range(label)
        if self.backend == "torch":
            return torch.profiler.record_function(label)
        return nullcontext()

    def trace_label(
        self,
        *,
        level: str,
        name: str,
        iter_id: int,
        layer_id: int | None,
        occurrence: int | None,
        inner_id: int | None,
    ) -> str:
        """Return a stable serialized identity shared by metadata and trace events."""
        # Kineto rewrites quotes in event names; encode names so both backends
        # preserve exactly the identity stored in our metadata.
        identity = "ts.profile/" + "/".join(
            quote(str(value), safe="")
            for value in (
                self.run_id,
                self.model_id,
                self.rank,
                level,
                iter_id,
                layer_id,
                occurrence,
                inner_id,
            )
        )
        readable_name = quote(name, safe="._")
        scope = f"layer={layer_id}"
        scope += f" | op={readable_name}" if level == "op" else f" | {readable_name}"
        call = inner_id if level == "op" else occurrence
        return f"{scope} | rank={self.rank} iter={iter_id} call={call} | {identity}"

    def bind_layer(self, instance: object, *, layer_id: int) -> None:
        """Bind an instance to its declared global layer number; return None."""
        if self._active:
            raise RuntimeError("Bind layer identities before recording")
        if layer_id < 0:
            raise ValueError("layer_id must be nonnegative")
        self._identities[id(instance)] = (instance, layer_id)

    @contextmanager
    def iteration(self, *, iter_id: int) -> Iterator[None]:
        """Record one eager iteration; IDs must be unique within this session."""
        if self._active or _current.get() is not None:
            raise RuntimeError("Nested or concurrent session windows are unsupported")
        if iter_id in self._iterations:
            raise ValueError(f"Iteration {iter_id} was already recorded")
        self._iterations.add(iter_id)
        self._active = True
        token = _current.set(_Context(self, iter_id, None, None))
        try:
            yield
        finally:
            _current.reset(token)
            self._active = False

    def export(self, path: str | Path, *, costs: CostRegistry) -> None:
        """Write metadata and offline op/layer costs to JSON; return None."""
        payload = {
            "schema_version": 1,
            "run_id": self.run_id,
            "model_id": self.model_id,
            "rank": self.rank,
            "timing_kind": "cpu_scope",
            "backend": self.backend,
            "execution_mode": "eager",
            "ops": [asdict(record) for record in self.ops],
            "layers": [asdict(record) for record in self.layers],
            "analysis": costs.analyze(self),
        }
        output = Path(path)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(payload, indent=2) + "\n")


def profiled(*, level: str) -> Callable:
    """Decorate a synchronous layer method, preserving its result and signature.

    Args:
        level: Must be 'layer' in this initial implementation.
    Returns:
        A decorator using the active session's explicit instance identity map.
    """
    if level != "layer":
        raise ValueError("Only layer profiling is implemented")

    def decorate(fn: Callable) -> Callable:
        if inspect.iscoroutinefunction(fn) or inspect.isgeneratorfunction(fn):
            raise TypeError("Profiling requires a synchronous function")

        @wraps(fn)
        def wrapped(*args: Any, **kwargs: Any) -> Any:
            context = _current.get()
            if context is None:
                return fn(*args, **kwargs)
            session = context.session
            instance = args[0] if args else kwargs.get("self")
            identity = session._identities.get(id(instance))
            if identity is None:
                raise ValueError("Bind the layer instance before recording")
            layer_id = identity[1]
            key = (context.iter_id, layer_id)
            occurrence = session._occurrences.get(key, 0)
            session._occurrences[key] = occurrence + 1
            token = _current.set(
                _Context(session, context.iter_id, layer_id, occurrence)
            )
            label = session.trace_label(
                level="layer",
                name=f"{type(instance).__name__}.{fn.__name__}",
                iter_id=context.iter_id,
                layer_id=layer_id,
                occurrence=occurrence,
                inner_id=None,
            )
            start = time.perf_counter_ns()
            error = None
            try:
                with session.record_func(label):
                    return fn(*args, **kwargs)
            except BaseException as exc:
                error = type(exc).__name__
                raise
            finally:
                duration = time.perf_counter_ns() - start
                _current.reset(token)
                session.layers.append(
                    LayerRecord(
                        context.iter_id,
                        layer_id,
                        occurrence,
                        start,
                        duration,
                        error,
                        label,
                    )
                )

        return wrapped

    return decorate


def wrap_registered_op(
    fn: Callable, *, family: str, mode: str, name: str, profile: OpProfile
) -> Callable:
    """Wrap one registered synchronous function for optional host collection."""
    if (
        not inspect.isfunction(fn)
        or inspect.iscoroutinefunction(fn)
        or inspect.isgeneratorfunction(fn)
    ):
        raise TypeError(
            "OpProfile currently supports synchronous registered functions only"
        )
    signature = inspect.signature(fn)

    @wraps(fn)
    def wrapped(*args: Any, **kwargs: Any) -> Any:
        context = _current.get()
        if context is None or _recording_op.get():
            return fn(*args, **kwargs)
        session = context.session
        inputs: dict[str, TensorDescription] = {}
        for key, value in signature.bind(*args, **kwargs).arguments.items():
            _describe(value, key, inputs)
        counter_key = (context.iter_id, context.layer_id)
        inner_id = session._counts.get(counter_key, 0)
        session._counts[counter_key] = inner_id + 1
        outputs: dict[str, TensorDescription] = {}
        error = None
        start = time.perf_counter_ns()
        label = session.trace_label(
            level="op",
            name=f"{family}.{mode}",
            iter_id=context.iter_id,
            layer_id=context.layer_id,
            occurrence=context.occurrence,
            inner_id=inner_id,
        )
        token = _recording_op.set(True)
        try:
            with session.record_func(label):
                result = fn(*args, **kwargs)
        except BaseException as exc:
            error = type(exc).__name__
            raise
        else:
            _describe(result, "return", outputs)
            return result
        finally:
            _recording_op.reset(token)
            session.ops.append(
                OpRecord(
                    context.iter_id,
                    context.layer_id,
                    context.occurrence,
                    inner_id,
                    f"{family}.{mode}",
                    name,
                    profile.cost_rule,
                    inputs,
                    outputs,
                    start,
                    time.perf_counter_ns() - start,
                    error,
                    label,
                )
            )

    return wrapped


@dataclass(frozen=True)
class OpCost:
    useful_flops: int
    executed_flops: int | None
    logical_read_bytes: int
    logical_write_bytes: int
    estimated_hbm_bytes: int | None
    rule: str
    assumptions: tuple[str, ...]


class CostRegistry:
    """Explicit offline rule registry; missing costs remain unavailable."""

    def __init__(self) -> None:
        self._rules: dict[str, Callable[[OpRecord], OpCost]] = {}

    def register(self, name: str, rule: Callable[[OpRecord], OpCost]) -> None:
        """Register a named rule; reject duplicate names."""
        if name in self._rules:
            raise ValueError(f"Duplicate cost rule: {name}")
        self._rules[name] = rule

    def analyze(self, session: ProfileSession) -> dict[str, Any]:
        """Return per-op costs and per-layer sums with explicit coverage.

        CPU scopes cannot provide device MFU/MBU denominators. These metrics
        remain unavailable without device timing and matching hardware peaks.
        """
        rows = []
        groups: dict[tuple[int, int | None], list[dict[str, Any]]] = {}
        for op in session.ops:
            rule = self._rules.get(op.cost_rule) if op.cost_rule is not None else None
            cost = None
            reason = None
            if op.error is not None:
                reason = f"op failed: {op.error}"
            elif rule is None:
                reason = "cost rule unavailable"
            else:
                try:
                    cost = rule(op)
                except ValueError as exc:
                    reason = str(exc)
            row = {
                "iter_id": op.iter_id,
                "layer_id": op.layer_id,
                "inner_id": op.inner_id,
                "cost": asdict(cost) if cost is not None else None,
                "missing_reason": reason,
            }
            rows.append(row)
            groups.setdefault((op.iter_id, op.layer_id), []).append(row)
        layers = []
        for (iter_id, layer_id), entries in groups.items():
            covered = [entry["cost"] for entry in entries if entry["cost"] is not None]
            sums = {
                field: (
                    sum(cost[field] for cost in covered)
                    if all(cost[field] is not None for cost in covered)
                    else None
                )
                for field in (
                    "useful_flops",
                    "executed_flops",
                    "logical_read_bytes",
                    "logical_write_bytes",
                )
            }
            layers.append(
                {
                    "iter_id": iter_id,
                    "layer_id": layer_id,
                    "op_count": len(entries),
                    "covered_op_count": len(covered),
                    "covered_cost": sums,
                    "complete_cost": sums if len(covered) == len(entries) else None,
                    "mfu": None,
                    "estimated_mbu": None,
                    "utilization_missing_reason": "device timing and matched hardware peaks unavailable",
                }
            )
        return {
            "coverage_scope": "opted_in_registered_calls",
            "ops": rows,
            "layers": layers,
        }


@dataclass(frozen=True)
class DenseMatmulCost:
    """Plain dense 2D GEMM cost using explicit argument roles.

    No padding, quantization, fusion or implicit workspace is modeled. HBM
    traffic remains unknown; logical traffic assumes each operand read once.
    """

    lhs: str
    rhs: str
    output: str

    def __call__(self, op: OpRecord) -> OpCost:
        try:
            a, b, out = (
                op.inputs[self.lhs],
                op.inputs[self.rhs],
                op.outputs[self.output],
            )
        except KeyError as exc:
            raise ValueError(f"Missing GEMM tensor role: {exc}") from exc
        if len(a.shape) != 2 or len(b.shape) != 2:
            raise ValueError("DenseMatmulCost requires 2D operands")
        m, k = a.shape
        kb, n = b.shape
        if k != kb or out.shape != (m, n):
            raise ValueError("GEMM shapes do not match")
        flops = 2 * m * n * k
        return OpCost(
            flops,
            flops,
            a.logical_bytes + b.logical_bytes,
            out.logical_bytes,
            None,
            "dense_mm.v1",
            (
                "unquantized dense A@B without padding or fusion",
                "each operand read once; output written once",
                "logical bytes do not measure HBM traffic",
            ),
        )
