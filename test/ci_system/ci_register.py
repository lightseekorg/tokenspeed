"""
TokenSpeed CI Registration System

AST-based test registration for CI suites. Tests register themselves by calling
marker functions (e.g., register_cuda_ci) at module level. These calls are
parsed statically via AST - no import or execution of test files is needed.

Usage in test files:
    from ci_system.ci_register import register_cuda_ci
    register_cuda_ci(est_time=300, suite="runtime-1gpu")

Then the runner collects all registrations via:
    from ci_system.ci_register import collect_tests
    tests = collect_tests(glob.glob("**/*.py"))
"""

import ast
import warnings
from dataclasses import dataclass
from enum import Enum, auto
from typing import List, Optional

__all__ = [
    "DeviceBackend",
    "CIRegistry",
    "collect_tests",
    "auto_partition",
    "register_cuda_ci",
    "ut_parse_one_file",
]

_PARAM_ORDER = (
    "est_time",
    "suite",
    "nightly",
    "disabled",
    "disabled_on_runners",
    "disabled_on_runners_reason",
)
_UNSET = object()


class DeviceBackend(Enum):
    CUDA = auto()


@dataclass
class CIRegistry:
    backend: DeviceBackend
    filename: str
    est_time: float
    suite: str
    nightly: bool = False
    # Globally disabled. None = enabled, string = disabled-everywhere reason.
    disabled: Optional[str] = None
    # Per-runner skip: list of fnmatch patterns matched against the
    # CI_RUNNER_LABEL env var at run time. Examples: ["linux-mi35*"],
    # ["h100-1gpu", "b200-*"]. Empty / None => no per-runner restriction.
    disabled_on_runners: Optional[List[str]] = None
    disabled_on_runners_reason: Optional[str] = None


def register_cuda_ci(
    est_time: float,
    suite: str,
    nightly: bool = False,
    disabled: Optional[str] = None,
    disabled_on_runners: Optional[List[str]] = None,
    disabled_on_runners_reason: Optional[str] = None,
):
    """Marker for CUDA CI registration (parsed via AST; runtime no-op)."""
    return None


REGISTER_MAPPING = {
    "register_cuda_ci": DeviceBackend.CUDA,
}


class RegistryVisitor(ast.NodeVisitor):
    """AST visitor that extracts CI registration calls from test files."""

    def __init__(self, filename: str):
        self.filename = filename
        self.registries: list[CIRegistry] = []

    def _constant_value(self, node: ast.AST) -> object:
        if isinstance(node, ast.Constant):
            return node.value
        if isinstance(node, (ast.List, ast.Tuple)):
            values = []
            for elt in node.elts:
                v = self._constant_value(elt)
                if v is _UNSET:
                    return _UNSET
                values.append(v)
            return values
        return _UNSET

    def _parse_call_args(
        self, func_call: ast.Call
    ) -> tuple[float, str, bool, Optional[str], Optional[List[str]], Optional[str]]:
        args = {name: _UNSET for name in _PARAM_ORDER}
        seen = set()

        if any(isinstance(arg, ast.Starred) for arg in func_call.args):
            raise ValueError(
                f"{self.filename}: starred arguments are not supported in "
                f"{func_call.func.id}()"
            )
        if len(func_call.args) > len(_PARAM_ORDER):
            raise ValueError(
                f"{self.filename}: too many positional arguments in "
                f"{func_call.func.id}()"
            )

        for name, arg in zip(_PARAM_ORDER, func_call.args):
            seen.add(name)
            args[name] = self._constant_value(arg)

        for kw in func_call.keywords:
            if kw.arg is None:
                raise ValueError(
                    f"{self.filename}: **kwargs are not supported in "
                    f"{func_call.func.id}()"
                )
            if kw.arg not in args:
                raise ValueError(
                    f"{self.filename}: unknown argument '{kw.arg}' in "
                    f"{func_call.func.id}()"
                )
            if kw.arg in seen:
                raise ValueError(
                    f"{self.filename}: duplicated argument '{kw.arg}' in "
                    f"{func_call.func.id}()"
                )
            seen.add(kw.arg)
            args[kw.arg] = self._constant_value(kw.value)

        if args["est_time"] is _UNSET or args["suite"] is _UNSET:
            raise ValueError(
                f"{self.filename}: est_time and suite are required constants in "
                f"{func_call.func.id}()"
            )

        est_time, suite = args["est_time"], args["suite"]
        nightly_value = args["nightly"]

        if not isinstance(est_time, (int, float)):
            raise ValueError(
                f"{self.filename}: est_time must be a number in {func_call.func.id}()"
            )
        if not isinstance(suite, str):
            raise ValueError(
                f"{self.filename}: suite must be a string in {func_call.func.id}()"
            )

        if nightly_value is _UNSET:
            nightly = False
        elif isinstance(nightly_value, bool):
            nightly = nightly_value
        else:
            raise ValueError(
                f"{self.filename}: nightly must be a boolean in {func_call.func.id}()"
            )

        disabled = args["disabled"] if args["disabled"] is not _UNSET else None
        if disabled is not None and not isinstance(disabled, str):
            raise ValueError(
                f"{self.filename}: disabled must be a string in {func_call.func.id}()"
            )

        dor_value = args["disabled_on_runners"]
        if dor_value is _UNSET or dor_value is None:
            disabled_on_runners: Optional[List[str]] = None
        elif isinstance(dor_value, list) and all(
            isinstance(item, str) and item for item in dor_value
        ):
            disabled_on_runners = list(dor_value) or None
        else:
            raise ValueError(
                f"{self.filename}: disabled_on_runners must be a non-empty "
                f"list of non-empty string fnmatch patterns in "
                f"{func_call.func.id}()"
            )

        dor_reason_value = args["disabled_on_runners_reason"]
        if dor_reason_value is _UNSET:
            disabled_on_runners_reason: Optional[str] = None
        elif dor_reason_value is None or isinstance(dor_reason_value, str):
            disabled_on_runners_reason = dor_reason_value
        else:
            raise ValueError(
                f"{self.filename}: disabled_on_runners_reason must be a "
                f"string in {func_call.func.id}()"
            )

        return (
            float(est_time),
            suite,
            nightly,
            disabled,
            disabled_on_runners,
            disabled_on_runners_reason,
        )

    def _collect_ci_registry(self, func_call: ast.Call):
        if not isinstance(func_call.func, ast.Name):
            return None

        backend = REGISTER_MAPPING.get(func_call.func.id)
        if backend is None:
            return None

        (
            est_time,
            suite,
            nightly,
            disabled,
            disabled_on_runners,
            disabled_on_runners_reason,
        ) = self._parse_call_args(func_call)
        return CIRegistry(
            backend=backend,
            filename=self.filename,
            est_time=est_time,
            suite=suite,
            nightly=nightly,
            disabled=disabled,
            disabled_on_runners=disabled_on_runners,
            disabled_on_runners_reason=disabled_on_runners_reason,
        )

    def visit_Module(self, node):
        for stmt in node.body:
            if not isinstance(stmt, ast.Expr) or not isinstance(stmt.value, ast.Call):
                continue

            cr = self._collect_ci_registry(stmt.value)
            if cr is not None:
                self.registries.append(cr)

        self.generic_visit(node)


def ut_parse_one_file(filename: str) -> List[CIRegistry]:
    """Parse a single test file and return all CI registrations found."""
    with open(filename, "r", encoding="utf-8") as f:
        file_content = f.read()
    tree = ast.parse(file_content, filename=filename)
    visitor = RegistryVisitor(filename=filename)
    visitor.visit(tree)
    return visitor.registries


def auto_partition(files: List[CIRegistry], rank: int, size: int) -> List[CIRegistry]:
    """Partition files into `size` sublists with approximately equal sums of
    estimated times using a greedy algorithm (LPT heuristic), and return the
    partition for the specified rank.
    """
    if not files or size <= 0:
        return []

    # Sort by estimated_time descending; filename as tie-breaker for
    # deterministic partitioning regardless of glob ordering.
    sorted_files = sorted(files, key=lambda f: (-f.est_time, f.filename))

    partitions: List[List[CIRegistry]] = [[] for _ in range(size)]
    partition_sums = [0.0] * size

    # Greedily assign each file to the partition with the smallest current total time
    for file in sorted_files:
        min_sum_idx = min(range(size), key=partition_sums.__getitem__)
        partitions[min_sum_idx].append(file)
        partition_sums[min_sum_idx] += file.est_time

    if rank < size:
        return partitions[rank]
    return []


def collect_tests(files: list[str], sanity_check: bool = True) -> List[CIRegistry]:
    """Collect CI registrations from a list of test files.

    Args:
        files: List of file paths to parse.
        sanity_check: If True, raise on files without registration.
                      If False, warn and skip.
    """
    ci_tests = []
    for file in files:
        registries = ut_parse_one_file(file)
        if not registries:
            msg = f"No CI registry found in {file}"
            if sanity_check:
                raise ValueError(msg)
            else:
                warnings.warn(msg)
                continue

        ci_tests.extend(registries)

    return ci_tests
