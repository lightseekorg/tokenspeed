#!/usr/bin/env python3
"""Fail closed on hidden first-party environment configuration.

TokenSpeed runtime behaviour must come from typed configuration or an explicit
API.  A small set of external protocols still use the process environment
(authentication, launch rank, Prometheus, and vendor-library bridges).  Those
exceptions are reviewed by exact source path, variable name, and access mode;
everything else is rejected.
"""

from __future__ import annotations

import argparse
import ast
import json
import re
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from pathlib import Path

SOURCE_ROOTS = (
    Path("python/tokenspeed"),
    Path("tokenspeed-kernel/python/tokenspeed_kernel"),
    Path("tokenspeed-kernel-amd/python/tokenspeed_kernel_amd"),
    Path("tokenspeed-mla/python/tokenspeed_mla"),
    Path("tokenspeed-scheduler/python/tokenspeed_scheduler"),
)
CPP_SOURCE_ROOTS = (
    Path("tokenspeed-scheduler/csrc"),
    Path("tokenspeed-scheduler/bindings"),
    Path("csrc"),
)
CPP_SUFFIXES = frozenset(
    {".c", ".cc", ".cpp", ".cu", ".cuh", ".cxx", ".h", ".hh", ".hpp", ".hxx"}
)
EXCLUDED_PARTS = frozenset({"thirdparty", "vendor", "__pycache__"})

# External read contracts.  TokenSpeed-specific product namespaces are never
# allowed here: those must be explicit configuration instead.
READ_ALLOWLIST = frozenset(
    {
        ("python/tokenspeed/bench.py", "OPENAI_API_KEY"),
        (
            "python/tokenspeed/runtime/model_loader/weight_utils.py",
            "HF_HUB_ENABLE_HF_TRANSFER",
        ),
        ("python/tokenspeed/runtime/model_loader/weight_utils.py", "LOCAL_RANK"),
        ("python/tokenspeed/runtime/model_loader/weight_utils.py", "LOCAL_WORLD_SIZE"),
        ("python/tokenspeed/runtime/utils/common.py", "CI"),
        ("python/tokenspeed/runtime/utils/common.py", "GITHUB_ACTIONS"),
        ("python/tokenspeed/runtime/utils/common.py", "PROMETHEUS_MULTIPROC_DIR"),
        ("python/tokenspeed/runtime/utils/hostfunc.py", "CUDA_HOME"),
    }
)

# Deterministic writes that project explicit TokenSpeed configuration into an
# external library's process-level protocol.  ``setdefault`` is never allowed:
# an inherited value must not override the typed configuration.
WRITE_ALLOWLIST = frozenset(
    {
        ("python/tokenspeed/runtime/engine/async_llm.py", "TOKENIZERS_PARALLELISM"),
        (
            "python/tokenspeed/runtime/entrypoints/engine.py",
            "CUDA_DEVICE_MAX_CONNECTIONS",
        ),
        ("python/tokenspeed/runtime/entrypoints/engine.py", "CUDA_MODULE_LOADING"),
        ("python/tokenspeed/runtime/entrypoints/engine.py", "NCCL_CUMEM_ENABLE"),
        ("python/tokenspeed/runtime/entrypoints/engine.py", "NCCL_NVLS_ENABLE"),
        ("python/tokenspeed/runtime/entrypoints/engine.py", "NVIDIA_TF32_OVERRIDE"),
        ("python/tokenspeed/runtime/entrypoints/engine.py", "TF_CPP_MIN_LOG_LEVEL"),
        (
            "python/tokenspeed/runtime/entrypoints/engine.py",
            "TORCH_ALLOW_TF32_CUBLAS_OVERRIDE",
        ),
        ("python/tokenspeed/runtime/utils/common.py", "PROMETHEUS_MULTIPROC_DIR"),
        ("python/tokenspeed/runtime/utils/common.py", "TORCH_CUDA_ARCH_LIST"),
        ("python/tokenspeed/runtime/utils/server_args.py", "TLLM_LOG_LEVEL"),
        ("python/tokenspeed/runtime/utils/server_args.py", "TORCHINDUCTOR_ENABLE_PDL"),
        ("python/tokenspeed/runtime/utils/server_args.py", "TRTLLM_ENABLE_PDL"),
    }
)


@dataclass(frozen=True, order=True)
class EnvironmentAccess:
    """One environment access discovered in first-party source."""

    path: str
    line: int
    column: int
    operation: str
    key: str


def _literal_key(node: ast.AST | None) -> str | None:
    if not isinstance(node, ast.Constant):
        return None
    if isinstance(node.value, str):
        return node.value
    if isinstance(node.value, bytes):
        return node.value.decode("utf-8", errors="backslashreplace")
    return None


class _EnvironmentVisitor(ast.NodeVisitor):
    def __init__(self, path: str) -> None:
        self.path = path
        self.os_names: set[str] = set()
        self.environ_names: set[str] = set()
        self.getenv_names: set[str] = set()
        self.envs_names: set[str] = set()
        self.accesses: set[EnvironmentAccess] = set()

    def _record(self, node: ast.AST, operation: str, key: str | None) -> None:
        self.accesses.add(
            EnvironmentAccess(
                path=self.path,
                line=getattr(node, "lineno", 0),
                column=getattr(node, "col_offset", 0),
                operation=operation,
                key=key if key is not None else "<dynamic>",
            )
        )

    def _is_environ(self, node: ast.AST) -> bool:
        if isinstance(node, ast.Name):
            return node.id in self.environ_names
        return (
            isinstance(node, ast.Attribute)
            and node.attr in {"environ", "environb"}
            and isinstance(node.value, ast.Name)
            and node.value.id in self.os_names
        )

    def _is_os_getenv(self, node: ast.AST) -> bool:
        if isinstance(node, ast.Name):
            return node.id in self.getenv_names
        return (
            isinstance(node, ast.Attribute)
            and node.attr in {"getenv", "getenvb"}
            and isinstance(node.value, ast.Name)
            and node.value.id in self.os_names
        )

    def _is_os_module(self, node: ast.AST) -> bool:
        return isinstance(node, ast.Name) and node.id in self.os_names

    def _bind_alias(self, target: ast.AST, value: ast.AST) -> None:
        if not isinstance(target, ast.Name):
            return

        aliases_os = self._is_os_module(value)
        aliases_environ = self._is_environ(value)
        aliases_getenv = self._is_os_getenv(value)
        self.os_names.discard(target.id)
        self.environ_names.discard(target.id)
        self.getenv_names.discard(target.id)
        if aliases_os:
            self.os_names.add(target.id)
        if aliases_environ:
            self.environ_names.add(target.id)
        if aliases_getenv:
            self.getenv_names.add(target.id)

    def visit_Import(self, node: ast.Import) -> None:
        for alias in node.names:
            if alias.name == "os":
                self.os_names.add(alias.asname or alias.name)
        self.generic_visit(node)

    def visit_ImportFrom(self, node: ast.ImportFrom) -> None:
        if node.module == "os":
            for alias in node.names:
                local_name = alias.asname or alias.name
                if alias.name in {"environ", "environb"}:
                    self.environ_names.add(local_name)
                elif alias.name in {"getenv", "getenvb"}:
                    self.getenv_names.add(local_name)
        if node.module == "tokenspeed.runtime.utils.env":
            for alias in node.names:
                if alias.name == "envs":
                    self.envs_names.add(alias.asname or alias.name)
        self.generic_visit(node)

    def visit_Assign(self, node: ast.Assign) -> None:
        for target in node.targets:
            self._bind_alias(target, node.value)
        self.generic_visit(node)

    def visit_AnnAssign(self, node: ast.AnnAssign) -> None:
        if node.value is not None:
            self._bind_alias(node.target, node.value)
        self.generic_visit(node)

    def visit_NamedExpr(self, node: ast.NamedExpr) -> None:
        self._bind_alias(node.target, node.value)
        self.generic_visit(node)

    def visit_Call(self, node: ast.Call) -> None:
        if self._is_os_getenv(node.func):
            key = _literal_key(node.args[0]) if node.args else None
            self._record(node, "read", key)
        elif isinstance(node.func, ast.Attribute) and self._is_environ(node.func.value):
            method = node.func.attr
            key = _literal_key(node.args[0]) if node.args else None
            if method == "get":
                self._record(node, "read", key)
            elif method == "setdefault":
                self._record(node, "setdefault", key)
            elif method in {"pop", "popitem", "clear"}:
                self._record(node, "delete", key)
            elif method in {"update", "copy", "items", "keys", "values"}:
                self._record(node, "dynamic", None)
            else:
                self._record(node, f"method:{method}", key)
        elif isinstance(node.func, ast.Name) and node.func.id == "get_bool_env_var":
            key = _literal_key(node.args[0]) if node.args else None
            self._record(node, "read", key)
        elif (
            isinstance(node.func, ast.Attribute)
            and isinstance(node.func.value, ast.Name)
            and node.func.value.id in self.os_names
            and node.func.attr in {"putenv", "unsetenv"}
        ):
            key = _literal_key(node.args[0]) if node.args else None
            operation = "write" if node.func.attr == "putenv" else "delete"
            self._record(node, operation, key)
        self.generic_visit(node)

    def visit_Subscript(self, node: ast.Subscript) -> None:
        if self._is_environ(node.value):
            key = _literal_key(node.slice)
            if isinstance(node.ctx, ast.Load):
                operation = "read"
            elif isinstance(node.ctx, ast.Store):
                operation = "write"
            else:
                operation = "delete"
            self._record(node, operation, key)
        self.generic_visit(node)

    def visit_Compare(self, node: ast.Compare) -> None:
        values = [node.left, *node.comparators]
        for index, operator in enumerate(node.ops):
            if not isinstance(operator, (ast.In, ast.NotIn)):
                continue
            left, right = values[index], values[index + 1]
            if self._is_environ(right):
                self._record(node, "read", _literal_key(left))
        self.generic_visit(node)

    def visit_Attribute(self, node: ast.Attribute) -> None:
        if isinstance(node.value, ast.Name) and node.value.id in self.envs_names:
            self._record(node, "product_env_field", node.attr)
        if self._is_environ(node) and not self._has_recognized_environ_parent(node):
            self._record(node, "dynamic", None)
        self.generic_visit(node)

    def visit_Name(self, node: ast.Name) -> None:
        if (
            isinstance(node.ctx, ast.Load)
            and self._is_environ(node)
            and not self._has_recognized_environ_parent(node)
        ):
            self._record(node, "dynamic", None)
        self.generic_visit(node)

    @staticmethod
    def _is_membership_mapping_operand(node: ast.AST, parent: ast.Compare) -> bool:
        values = [parent.left, *parent.comparators]
        recognized = False
        for index, operator in enumerate(parent.ops):
            left, right = values[index], values[index + 1]
            if node is not left and node is not right:
                continue
            if node is right and isinstance(operator, (ast.In, ast.NotIn)):
                recognized = True
                continue
            # Reading the whole environment for equality, ordering, identity,
            # or as the membership needle is not a keyed lookup.
            return False
        return recognized

    def _has_recognized_environ_parent(self, node: ast.AST) -> bool:
        parent = getattr(node, "_product_env_parent", None)
        if isinstance(parent, ast.Subscript) and parent.value is node:
            return True
        if isinstance(parent, ast.Compare):
            return self._is_membership_mapping_operand(node, parent)
        if isinstance(parent, ast.Attribute) and parent.value is node:
            grandparent = getattr(parent, "_product_env_parent", None)
            return isinstance(grandparent, ast.Call) and grandparent.func is parent
        return False


def scan_source(path: Path, *, repo_root: Path) -> list[EnvironmentAccess]:
    """Return process-environment accesses found in one Python source file."""

    relative = path.relative_to(repo_root).as_posix()
    tree = ast.parse(path.read_text(), filename=relative)
    for parent in ast.walk(tree):
        for child in ast.iter_child_nodes(parent):
            child._product_env_parent = parent  # type: ignore[attr-defined]
    visitor = _EnvironmentVisitor(relative)
    visitor.visit(tree)
    return sorted(visitor.accesses)


_CPP_ENV_CALL = re.compile(
    r"(?<![A-Za-z0-9_])"
    r"(?:::)?"
    r"(?:(?:[A-Za-z_]\w*)\s*::\s*)*"
    r"(?P<function>secure_getenv|getenv|setenv|unsetenv|putenv|_putenv|_putenv_s)"
    r"\s*\("
)
_CPP_STRING_LITERAL = re.compile(r'(?:u8|u|U|L)?"((?:\\.|[^"\\])*)"')


def _strip_cpp_comments(source: str) -> str:
    """Replace C/C++ comments with spaces while preserving source positions."""

    chars = list(source)
    index = 0
    quote: str | None = None
    while index < len(chars):
        current = chars[index]
        if quote is not None:
            if current == "\\":
                index += 2
                continue
            if current == quote:
                quote = None
            index += 1
            continue
        if current in {'"', "'"}:
            quote = current
            index += 1
            continue
        if current == "/" and index + 1 < len(chars):
            following = chars[index + 1]
            if following == "/":
                chars[index] = chars[index + 1] = " "
                index += 2
                while index < len(chars) and chars[index] not in "\r\n":
                    chars[index] = " "
                    index += 1
                continue
            if following == "*":
                chars[index] = chars[index + 1] = " "
                index += 2
                while index + 1 < len(chars):
                    if chars[index] == "*" and chars[index + 1] == "/":
                        chars[index] = chars[index + 1] = " "
                        index += 2
                        break
                    if chars[index] not in "\r\n":
                        chars[index] = " "
                    index += 1
                continue
        index += 1
    return "".join(chars)


def _mask_cpp_literals(source: str) -> str:
    """Mask quoted literals so call-like text inside messages is ignored."""

    chars = list(source)
    index = 0
    while index < len(chars):
        quote = chars[index]
        if quote not in {'"', "'"}:
            index += 1
            continue
        chars[index] = " "
        index += 1
        while index < len(chars):
            current = chars[index]
            if current in "\r\n":
                index += 1
                continue
            chars[index] = " "
            if current == "\\":
                if index + 1 < len(chars) and chars[index + 1] not in "\r\n":
                    chars[index + 1] = " "
                index += 2
                continue
            index += 1
            if current == quote:
                break
    return "".join(chars)


def scan_cpp_source(path: Path, *, repo_root: Path) -> list[EnvironmentAccess]:
    """Return direct process-environment access in first-party C/C++."""

    relative = path.relative_to(repo_root).as_posix()
    source = _strip_cpp_comments(path.read_text())
    searchable = _mask_cpp_literals(source)
    accesses: list[EnvironmentAccess] = []
    for match in _CPP_ENV_CALL.finditer(searchable):
        function = match.group("function")
        argument = source[match.end() :]
        leading = len(argument) - len(argument.lstrip())
        literal = _CPP_STRING_LITERAL.match(argument, leading)
        key = None
        if literal is not None:
            key = bytes(literal.group(1), "utf-8").decode(
                "unicode_escape", errors="backslashreplace"
            )
            if function in {"putenv", "_putenv"}:
                key = key.partition("=")[0] or None
        if function in {"getenv", "secure_getenv"}:
            operation = "read"
        elif function == "unsetenv":
            operation = "delete"
        else:
            operation = "write"
        line = source.count("\n", 0, match.start()) + 1
        line_start = source.rfind("\n", 0, match.start()) + 1
        accesses.append(
            EnvironmentAccess(
                path=relative,
                line=line,
                column=match.start() - line_start,
                operation=operation,
                key=key if key is not None else "<dynamic>",
            )
        )
    return sorted(set(accesses))


def find_python_sources(repo_root: Path) -> list[Path]:
    """Find first-party Python sources covered by the environment contract."""

    sources: list[Path] = []
    for relative_root in SOURCE_ROOTS:
        root = repo_root / relative_root
        if not root.is_dir():
            continue
        sources.extend(
            path
            for path in root.rglob("*.py")
            if not EXCLUDED_PARTS.intersection(path.relative_to(root).parts)
        )
    return sorted(sources)


def find_cpp_sources(repo_root: Path) -> list[Path]:
    """Find first-party C/C++ sources covered by the environment contract."""

    sources: list[Path] = []
    for relative_root in CPP_SOURCE_ROOTS:
        root = repo_root / relative_root
        if not root.is_dir():
            continue
        sources.extend(
            path
            for path in root.rglob("*")
            if path.is_file()
            and path.suffix in CPP_SUFFIXES
            and not EXCLUDED_PARTS.intersection(path.relative_to(root).parts)
        )
    return sorted(sources)


def is_allowed(access: EnvironmentAccess) -> bool:
    """Return whether ``access`` is an exact reviewed external contract."""

    identity = (access.path, access.key)
    if access.operation == "read":
        return identity in READ_ALLOWLIST
    if access.operation == "write":
        return identity in WRITE_ALLOWLIST
    return False


def audit_repository(repo_root: Path) -> dict[str, object]:
    """Audit first-party sources and return a durable JSON-compatible result."""

    accesses: list[EnvironmentAccess] = []
    parse_failures: list[str] = []
    for path in find_python_sources(repo_root):
        try:
            accesses.extend(scan_source(path, repo_root=repo_root))
        except (OSError, SyntaxError) as error:
            parse_failures.append(f"{path.relative_to(repo_root)}: {error}")
    for path in find_cpp_sources(repo_root):
        try:
            accesses.extend(scan_cpp_source(path, repo_root=repo_root))
        except OSError as error:
            parse_failures.append(f"{path.relative_to(repo_root)}: {error}")

    accesses.sort()
    violations = [access for access in accesses if not is_allowed(access)]
    observed_reads = {
        (access.path, access.key) for access in accesses if access.operation == "read"
    }
    observed_writes = {
        (access.path, access.key) for access in accesses if access.operation == "write"
    }
    stale_allowlist = [
        {"operation": "read", "path": path, "key": key}
        for path, key in sorted(READ_ALLOWLIST - observed_reads)
    ] + [
        {"operation": "write", "path": path, "key": key}
        for path, key in sorted(WRITE_ALLOWLIST - observed_writes)
    ]
    return {
        "schema_version": 1,
        "check": "product_environment_contract",
        "ok": not parse_failures and not violations and not stale_allowlist,
        "parse_failures": parse_failures,
        "reviewed_accesses": [
            asdict(access) for access in accesses if is_allowed(access)
        ],
        "stale_allowlist": stale_allowlist,
        "violations": [asdict(access) for access in violations],
    }


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Reject hidden first-party environment configuration."
    )
    parser.add_argument(
        "--repo-root",
        type=Path,
        default=Path(__file__).resolve().parents[2],
    )
    parser.add_argument(
        "--json", action="store_true", help="Print the full JSON audit."
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    result = audit_repository(args.repo_root.resolve())
    if args.json:
        print(json.dumps(result, indent=2, sort_keys=True))
    elif result["ok"]:
        print("Product environment contract passed")
    else:
        for failure in result["parse_failures"]:
            print(f"parse failure: {failure}")
        for violation in result["violations"]:
            print(
                f"{violation['path']}:{violation['line']}: "
                f"{violation['operation']} {violation['key']}"
            )
        for stale in result["stale_allowlist"]:
            print(
                f"stale allowlist: {stale['path']} "
                f"{stale['operation']} {stale['key']}"
            )
    return 0 if result["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
