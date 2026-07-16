#!/usr/bin/env python3
"""Validate the published SMG package set before MiniMax-M3 GPU startup.

The active multimodal release job must exercise the exact private packages
declared by TokenSpeed, not similarly named public packages or a source-tree
overlay.  All package contents are located through ``importlib.metadata``;
this check never falls back to an SMG source checkout.
"""

from __future__ import annotations

import argparse
import ast
import importlib.util
import json
import os
import re
import site
import subprocess
import sys
import tomllib
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass
from importlib import metadata
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

from packaging.requirements import Requirement
from packaging.utils import canonicalize_name

EXPECTED_DISTRIBUTIONS = {
    "tokenspeed-smg": "smg",
    "tokenspeed-smg-grpc-proto": "smg_grpc_proto",
    "tokenspeed-smg-grpc-servicer": "smg_grpc_servicer",
}
CONFLICTING_PUBLIC_DISTRIBUTIONS = (
    "smg",
    "smg-grpc-proto",
    "smg-grpc-servicer",
)
SERVICER_FILES = {
    "encoder": "smg_grpc_servicer/tokenspeed/encoder_servicer.py",
    "launcher": "smg_grpc_servicer/tokenspeed/scheduler_launcher.py",
    "rdma_config": "smg_grpc_servicer/tokenspeed/rdma_config.py",
    "server": "smg_grpc_servicer/tokenspeed/server.py",
    "servicer": "smg_grpc_servicer/tokenspeed/servicer.py",
}
TEXT_MARKERS = (
    (
        "encoder_explicit_epd_pixel_shm",
        "encoder",
        re.compile(r"server_args\s*\.\s*epd_pixel_shm"),
    ),
    (
        "encoder_explicit_epd_ingest_offloop",
        "encoder",
        re.compile(r"server_args\s*\.\s*epd_ingest_offloop"),
    ),
    (
        "encoder_explicit_unlink_mm_shm_after_read",
        "encoder",
        re.compile(r"server_args\s*\.\s*unlink_mm_shm_after_read"),
    ),
    (
        "launcher_disables_nested_signal_management",
        "launcher",
        re.compile(r"manage_signals\s*=\s*False"),
    ),
    (
        "servicer_awaits_engine_close",
        "servicer",
        re.compile(r"await\s+self\s*\.\s*async_llm\s*\.\s*close\s*\(\s*\)"),
    ),
    (
        "server_uses_explicit_grpc_message_limit",
        "server",
        re.compile(r"_grpc_max_message_bytes\s*\(\s*server_args\s*\)"),
    ),
)
MAX_PIP_OUTPUT_CHARS = 8_192
PYPI_INDEX_URL = "https://pypi.org/simple"
PYPI_FILE_HOSTS = frozenset({"files.pythonhosted.org"})
PIP_OVERRIDE_KEYS = frozenset(
    {
        "PIP_BREAK_SYSTEM_PACKAGES",
        "PIP_CONFIG_FILE",
        "PIP_EXTRA_INDEX_URL",
        "PIP_FIND_LINKS",
        "PIP_INDEX_URL",
        "PIP_NO_INDEX",
        "PIP_PREFIX",
        "PIP_TARGET",
        "PIP_TRUSTED_HOST",
        "PIP_USER",
    }
)
PRODUCT_ENV_PREFIXES = ("TOKENSPEED_", "SMG_", "EPD_", "TS_")
SERVICER_TOKENSPEED_SOURCE_PREFIX = "smg_grpc_servicer/tokenspeed/"
# The TokenSpeed adapter imports this shared root module directly for image
# RDMA.  Keep the root-file list exact so unrelated SMG adapters do not turn
# their own configuration contracts into MiniMax-M3 release false positives.
SERVICER_REQUIRED_ROOT_SOURCES = frozenset({"smg_grpc_servicer/mm_rdma.py"})
# These multimodal configuration keys used to be read by the private Rust
# binding.  Source AST checks cannot inspect a compiled extension, so the
# installed wheel is also rejected when any legacy key remains embedded in the
# binding.  The list is intentionally exact: unrelated SMG features are outside
# the MiniMax-M3 release surface and may have their own configuration contract.
FORBIDDEN_BINDING_ENV_KEYS = frozenset(
    {
        "EPD_INGEST_OFFLOOP",
        "EPD_PIXEL_SHM",
        "SMG_IMAGE_MAX_INPUT_BYTES",
        "SMG_LOG_MM_TIMING",
        "SMG_MM_PIXEL_CACHE_MB",
        "SMG_MM_PIXEL_RDMA",
        "SMG_MM_SHM_MIN_BYTES",
        "SMG_MM_TENSOR_TRANSPORT",
        "SMG_RDMA_LANDING_SLOTS",
        "SMG_RDMA_LANDING_WAIT_S",
        "SMG_RDMA_LISTEN_IP",
        "SMG_RDMA_LISTEN_PORT",
        "SMG_RDMA_POOL_SLOTS",
        "SMG_RDMA_READ_TIMEOUT_S",
        "SMG_RDMA_SEND_MD",
        "SMG_RDMA_SLOT_BYTES",
        "SMG_RDMA_SLOT_TTL_S",
        "SMG_TOKENSPEED_AUDIO_ENCODER_INPUT_DTYPE",
        "SMG_TOKENSPEED_ENCODER_INPUT_DTYPE",
        "SMG_TOKENSPEED_IMAGE_ENCODER_INPUT_DTYPE",
        "SMG_TOKENSPEED_MM_SHM_MIN_BYTES",
        "SMG_TOKENSPEED_MM_TENSOR_TRANSPORT",
        "SMG_TOKENSPEED_VIDEO_ENCODER_INPUT_DTYPE",
        "TOKENSPEED_GRPC_MAX_MESSAGE_BYTES",
        "TOKENSPEED_HEALTH_CHECK_TIMEOUT",
        "TOKENSPEED_LOG_MM_TENSOR_DATA",
        "TOKENSPEED_LOG_MM_TIMING",
        "TOKENSPEED_SKIP_GRPC_WARMUP",
        "TOKENSPEED_UNLINK_MM_SHM_AFTER_READ",
    }
)


@dataclass(frozen=True)
class PipCheckResult:
    """Result of invoking ``python -m pip check``."""

    returncode: int
    stdout: str = ""
    stderr: str = ""


@dataclass(frozen=True)
class PipInstallResult:
    """Result of installing the exact published wheel set."""

    returncode: int
    stdout: str = ""
    stderr: str = ""


def _static_env_key(node: ast.AST | None) -> str | None:
    if isinstance(node, ast.Constant):
        if isinstance(node.value, str):
            return node.value
        if isinstance(node.value, bytes):
            return node.value.decode("utf-8", errors="backslashreplace")
        return None
    if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Add):
        left = _static_env_key(node.left)
        right = _static_env_key(node.right)
        if left is not None and right is not None:
            return left + right
    return None


class _ProductEnvironmentAccessVisitor(ast.NodeVisitor):
    """Find forbidden or non-static access through ``os`` environment APIs."""

    _ENVIRON_READ_METHODS = frozenset(
        {
            "__delitem__",
            "__getitem__",
            "__setitem__",
            "get",
            "pop",
            "setdefault",
        }
    )
    _ENVIRON_DYNAMIC_METHODS = frozenset(
        {
            "__iter__",
            "clear",
            "copy",
            "items",
            "keys",
            "popitem",
            "update",
            "values",
        }
    )

    def __init__(self) -> None:
        # Treat the conventional name fail-closed even if an import appears
        # textually after a function definition that references it.
        self.os_names: set[str] = {"os"}
        self.environ_names: set[str] = set()
        self.getenv_names: set[str] = set()
        self.putenv_names: set[str] = set()
        self.unsetenv_names: set[str] = set()
        self.environ_reader_names: set[str] = set()
        self.handled_environ_nodes: set[int] = set()
        self.violations: list[dict[str, Any]] = []

    def _is_os_module(self, node: ast.AST) -> bool:
        return isinstance(node, ast.Name) and node.id in self.os_names

    def _is_environ(self, node: ast.AST) -> bool:
        if isinstance(node, ast.Name):
            return node.id in self.environ_names
        return (
            isinstance(node, ast.Attribute)
            and node.attr in {"environ", "environb"}
            and self._is_os_module(node.value)
        )

    def _is_getenv(self, node: ast.AST) -> bool:
        if isinstance(node, ast.Name):
            return node.id in self.getenv_names
        return (
            isinstance(node, ast.Attribute)
            and node.attr in {"getenv", "getenvb"}
            and self._is_os_module(node.value)
        )

    def _is_os_environment_function(
        self, node: ast.AST, *, name: str, aliases: set[str]
    ) -> bool:
        if isinstance(node, ast.Name):
            return node.id in aliases
        return (
            isinstance(node, ast.Attribute)
            and node.attr == name
            and self._is_os_module(node.value)
        )

    def _record(self, node: ast.AST, api: str, key_node: ast.AST | None) -> None:
        key = _static_env_key(key_node)
        if key is not None and not key.startswith(PRODUCT_ENV_PREFIXES):
            return
        self.violations.append(
            {
                "api": api,
                "column": node.col_offset,
                "key": key if key is not None else "<dynamic>",
                "line": node.lineno,
            }
        )

    def _bind_alias(self, target: ast.AST, value: ast.AST) -> None:
        if not isinstance(target, ast.Name):
            return
        if self._is_os_module(value):
            self.os_names.add(target.id)
        if self._is_environ(value):
            self.environ_names.add(target.id)
            self.handled_environ_nodes.add(id(value))
        if self._is_getenv(value):
            self.getenv_names.add(target.id)
        if self._is_os_environment_function(
            value, name="putenv", aliases=self.putenv_names
        ):
            self.putenv_names.add(target.id)
        if self._is_os_environment_function(
            value, name="unsetenv", aliases=self.unsetenv_names
        ):
            self.unsetenv_names.add(target.id)
        if (
            isinstance(value, ast.Attribute)
            and self._is_environ(value.value)
            and value.attr in self._ENVIRON_READ_METHODS
        ):
            self.environ_reader_names.add(target.id)
            self.handled_environ_nodes.add(id(value.value))
        if isinstance(value, ast.Name) and value.id in self.environ_reader_names:
            self.environ_reader_names.add(target.id)

    def scan(self, tree: ast.AST) -> None:
        """Collect aliases to a fixed point, then inspect all accesses."""

        nodes = list(ast.walk(tree))
        for node in nodes:
            if isinstance(node, ast.Import):
                for alias in node.names:
                    if alias.name == "os":
                        self.os_names.add(alias.asname or "os")
            elif isinstance(node, ast.ImportFrom) and node.module == "os":
                for alias in node.names:
                    local_name = alias.asname or alias.name
                    if alias.name in {"environ", "environb"}:
                        self.environ_names.add(local_name)
                    elif alias.name in {"getenv", "getenvb"}:
                        self.getenv_names.add(local_name)
                    elif alias.name == "putenv":
                        self.putenv_names.add(local_name)
                    elif alias.name == "unsetenv":
                        self.unsetenv_names.add(local_name)

        changed = True
        while changed:
            before = (
                len(self.os_names),
                len(self.environ_names),
                len(self.getenv_names),
                len(self.environ_reader_names),
                len(self.putenv_names),
                len(self.unsetenv_names),
            )
            for node in nodes:
                if isinstance(node, ast.Assign):
                    for target in node.targets:
                        self._bind_alias(target, node.value)
                elif isinstance(node, ast.AnnAssign) and node.value is not None:
                    self._bind_alias(node.target, node.value)
                elif isinstance(node, ast.NamedExpr):
                    self._bind_alias(node.target, node.value)
            after = (
                len(self.os_names),
                len(self.environ_names),
                len(self.getenv_names),
                len(self.environ_reader_names),
                len(self.putenv_names),
                len(self.unsetenv_names),
            )
            changed = after != before

        self.visit(tree)
        for node in nodes:
            if (
                self._is_environ(node)
                and isinstance(node.ctx, ast.Load)
                and id(node) not in self.handled_environ_nodes
            ):
                self._record(node, "dynamic os.environ", None)

    def visit_Import(self, node: ast.Import) -> None:
        for alias in node.names:
            if alias.name == "os":
                self.os_names.add(alias.asname or "os")

    def visit_ImportFrom(self, node: ast.ImportFrom) -> None:
        if node.module != "os":
            return
        for alias in node.names:
            local_name = alias.asname or alias.name
            if alias.name in {"environ", "environb"}:
                self.environ_names.add(local_name)
            elif alias.name in {"getenv", "getenvb"}:
                self.getenv_names.add(local_name)
            elif alias.name == "putenv":
                self.putenv_names.add(local_name)
            elif alias.name == "unsetenv":
                self.unsetenv_names.add(local_name)
            elif alias.name == "*":
                self._record(node, "from os import *", None)

    def visit_Assign(self, node: ast.Assign) -> None:
        for target in node.targets:
            self._bind_alias(target, node.value)
            if self._is_environ(target):
                self.handled_environ_nodes.add(id(target))
                self._record(target, "assign os.environ", None)
        self.generic_visit(node)

    def visit_AnnAssign(self, node: ast.AnnAssign) -> None:
        if node.value is not None:
            self._bind_alias(node.target, node.value)
        self.generic_visit(node)

    def visit_NamedExpr(self, node: ast.NamedExpr) -> None:
        self._bind_alias(node.target, node.value)
        self.generic_visit(node)

    def visit_Call(self, node: ast.Call) -> None:
        if self._is_getenv(node.func):
            self._record(node, "os.getenv", node.args[0] if node.args else None)
        elif self._is_os_environment_function(
            node.func, name="putenv", aliases=self.putenv_names
        ):
            self._record(node, "os.putenv", node.args[0] if node.args else None)
        elif self._is_os_environment_function(
            node.func, name="unsetenv", aliases=self.unsetenv_names
        ):
            self._record(node, "os.unsetenv", node.args[0] if node.args else None)
        elif (
            isinstance(node.func, ast.Name)
            and node.func.id in self.environ_reader_names
        ):
            self._record(node, "os.environ reader", node.args[0] if node.args else None)
        elif isinstance(node.func, ast.Attribute) and self._is_environ(node.func.value):
            self.handled_environ_nodes.add(id(node.func.value))
            if node.func.attr in self._ENVIRON_READ_METHODS:
                self._record(
                    node,
                    f"os.environ.{node.func.attr}",
                    node.args[0] if node.args else None,
                )
            elif node.func.attr in self._ENVIRON_DYNAMIC_METHODS:
                self._record(node, f"os.environ.{node.func.attr}", None)
        else:
            environ_arguments = [
                argument for argument in node.args if self._is_environ(argument)
            ] + [
                keyword.value
                for keyword in node.keywords
                if self._is_environ(keyword.value)
            ]
            if environ_arguments:
                self.handled_environ_nodes.update(
                    id(argument) for argument in environ_arguments
                )
                self._record(node, "dynamic os.environ argument", None)
        self.generic_visit(node)

    def visit_Subscript(self, node: ast.Subscript) -> None:
        if self._is_environ(node.value):
            self.handled_environ_nodes.add(id(node.value))
            operation = {
                ast.Load: "os.environ[]",
                ast.Store: "os.environ[]=",
                ast.Del: "del os.environ[]",
            }.get(type(node.ctx), "os.environ[] access")
            self._record(node, operation, node.slice)
        self.generic_visit(node)

    def visit_AugAssign(self, node: ast.AugAssign) -> None:
        if self._is_environ(node.target):
            self.handled_environ_nodes.add(id(node.target))
            self._record(node, "update os.environ", None)
        self.generic_visit(node)

    def visit_Delete(self, node: ast.Delete) -> None:
        for target in node.targets:
            if self._is_environ(target):
                self.handled_environ_nodes.add(id(target))
                self._record(target, "delete os.environ", None)
        self.generic_visit(node)

    def visit_Compare(self, node: ast.Compare) -> None:
        values = [node.left, *node.comparators]
        for index, operator in enumerate(node.ops):
            if isinstance(operator, (ast.In, ast.NotIn)) and self._is_environ(
                values[index + 1]
            ):
                self.handled_environ_nodes.add(id(values[index + 1]))
                self._record(node, "key in os.environ", values[index])
        self.generic_visit(node)

    def visit_For(self, node: ast.For) -> None:
        if self._is_environ(node.iter):
            self.handled_environ_nodes.add(id(node.iter))
            self._record(node, "iterate os.environ", None)
        self.generic_visit(node)

    def visit_comprehension(self, node: ast.comprehension) -> None:
        if self._is_environ(node.iter):
            self.handled_environ_nodes.add(id(node.iter))
            self._record(node, "iterate os.environ", None)
        self.generic_visit(node)


def _write_json_atomic(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)


def _metadata_path(distribution: metadata.Distribution) -> str:
    files = distribution.files or ()
    metadata_file = next(
        (entry for entry in files if str(entry).endswith(".dist-info/METADATA")),
        None,
    )
    if metadata_file is None:
        return ""
    return str(Path(distribution.locate_file(metadata_file)).absolute())


def _distribution_name(distribution: metadata.Distribution) -> str:
    return distribution.metadata.get("Name", "")


def _distribution_instances(
    distributions: Iterable[metadata.Distribution], name: str
) -> list[metadata.Distribution]:
    normalized = canonicalize_name(name)
    return [
        distribution
        for distribution in distributions
        if canonicalize_name(_distribution_name(distribution)) == normalized
    ]


def _parse_exact_pins(pyproject: Path) -> tuple[dict[str, str], list[str]]:
    failures: list[str] = []
    pins: dict[str, str] = {}
    try:
        document = tomllib.loads(pyproject.read_text())
        dependencies = document["project"]["dependencies"]
    except (OSError, KeyError, TypeError, tomllib.TOMLDecodeError) as error:
        return {}, [f"cannot read project dependencies from {pyproject}: {error}"]

    expected = {canonicalize_name(name): name for name in EXPECTED_DISTRIBUTIONS}
    matches: dict[str, list[Requirement]] = {name: [] for name in expected.values()}
    for dependency in dependencies:
        try:
            requirement = Requirement(dependency)
        except Exception as error:
            failures.append(f"invalid project dependency {dependency!r}: {error}")
            continue
        canonical_name = canonicalize_name(requirement.name)
        if canonical_name in expected:
            matches[expected[canonical_name]].append(requirement)

    for name, requirements in matches.items():
        if len(requirements) != 1:
            failures.append(
                f"{name} must appear exactly once in project.dependencies; "
                f"found {len(requirements)} entries"
            )
            continue
        requirement = requirements[0]
        specifiers = list(requirement.specifier)
        exact = (
            requirement.url is None
            and requirement.marker is None
            and not requirement.extras
            and len(specifiers) == 1
            and specifiers[0].operator == "=="
            and "*" not in specifiers[0].version
        )
        if not exact:
            failures.append(
                f"{name} must use one unconditional, exact == pin without extras"
            )
            continue
        pins[name] = specifiers[0].version

    return pins, failures


def _read_venv_config(prefix: Path) -> tuple[dict[str, str], str]:
    path = prefix / "pyvenv.cfg"
    try:
        lines = path.read_text().splitlines()
    except OSError as error:
        return {}, str(error)
    settings: dict[str, str] = {}
    for line in lines:
        key, separator, value = line.partition("=")
        if separator:
            settings[key.strip().lower()] = value.strip().lower()
    return settings, ""


def inspect_python_environment() -> dict[str, Any]:
    """Describe Python isolation properties used by the package audit."""

    prefix = Path(sys.prefix).absolute()
    config, config_error = _read_venv_config(prefix)
    return {
        # A standard venv executable is commonly a symlink to the base Python.
        # Preserve its invoked path here; sys.prefix/base_prefix prove that the
        # interpreter actually entered venv mode.
        "executable": str(Path(sys.executable).absolute()),
        "prefix": str(prefix),
        "base_prefix": str(Path(sys.base_prefix).absolute()),
        "venv_config": str(prefix / "pyvenv.cfg"),
        "venv_config_error": config_error,
        "include_system_site_packages": config.get("include-system-site-packages"),
        "user_site_enabled": site.ENABLE_USER_SITE,
        "pythonpath": os.environ.get("PYTHONPATH", ""),
        "pip_override_keys": sorted(PIP_OVERRIDE_KEYS.intersection(os.environ)),
        "sys_path": list(sys.path),
    }


def _resolved(path: str | Path) -> Path:
    return Path(path).absolute().resolve()


def _is_inside(path: str | Path, root: Path) -> bool:
    try:
        return _resolved(path).is_relative_to(root)
    except (OSError, RuntimeError):
        return False


def _is_lexically_inside(path: str | Path, root: Path) -> bool:
    return Path(path).absolute().is_relative_to(Path(root).absolute())


def _audit_python_environment(
    environment: Mapping[str, Any],
    distributions: Sequence[metadata.Distribution],
) -> tuple[dict[str, Any], list[str]]:
    details = dict(environment)
    pythonpath_present = bool(str(details.pop("pythonpath", "")).strip())
    details["pythonpath_present"] = pythonpath_present
    failures: list[str] = []
    try:
        prefix = _resolved(str(environment["prefix"]))
        base_prefix = _resolved(str(environment["base_prefix"]))
    except KeyError as error:
        return details, [f"Python environment audit is missing {error.args[0]}"]

    if prefix == base_prefix:
        failures.append("Python is not running inside a virtual environment")
    if environment.get("venv_config_error"):
        failures.append(
            "cannot read the virtual environment configuration: "
            f"{environment['venv_config_error']}"
        )
    if environment.get("include_system_site_packages") != "false":
        failures.append("virtual environment must disable system site-packages")
    if environment.get("user_site_enabled") is not False:
        failures.append("Python user site-packages must be disabled")
    if pythonpath_present:
        failures.append("PYTHONPATH must be unset for the release package audit")
    pip_override_keys = environment.get("pip_override_keys", [])
    if pip_override_keys:
        failures.append(
            "pip override environment variables must be unset: "
            + ", ".join(str(key) for key in pip_override_keys)
        )
    executable = environment.get("executable", "")
    lexical_prefix = Path(str(environment["prefix"])).absolute()
    if not executable or not _is_lexically_inside(str(executable), lexical_prefix):
        failures.append("Python executable is outside the isolated environment")

    external_shadows: list[dict[str, str]] = []
    for raw_entry in environment.get("sys_path", []):
        entry = _resolved(raw_entry or Path.cwd())
        if entry == prefix or entry.is_relative_to(prefix):
            continue
        for import_name in EXPECTED_DISTRIBUTIONS.values():
            candidate = entry / import_name
            if candidate.exists():
                external_shadows.append(
                    {"import_name": import_name, "path": str(candidate)}
                )
    details["external_import_shadows"] = external_shadows
    for shadow in external_shadows:
        failures.append(
            f"external sys.path entry can shadow {shadow['import_name']}: "
            f"{shadow['path']}"
        )

    outside_metadata = []
    for distribution in distributions:
        if canonicalize_name(_distribution_name(distribution)) not in {
            canonicalize_name(name) for name in EXPECTED_DISTRIBUTIONS
        }:
            continue
        path = _metadata_path(distribution)
        if not path or not _is_inside(path, prefix):
            outside_metadata.append(path)
    details["private_metadata_outside_prefix"] = outside_metadata
    if outside_metadata:
        failures.append(
            "private distribution metadata is outside the isolated environment"
        )

    details["ok"] = not failures
    return details, failures


def _audit_pip_install_report(
    report_path: Path | None, pins: Mapping[str, str]
) -> tuple[dict[str, Any], list[str]]:
    details: dict[str, Any] = {
        "path": str(report_path.absolute()) if report_path is not None else None,
        "packages": {},
    }
    failures: list[str] = []
    if report_path is None:
        details["ok"] = False
        return details, ["a pip install report is required for release provenance"]
    try:
        document = json.loads(report_path.read_text())
        entries = document["install"]
        if not isinstance(entries, list):
            raise TypeError("install must be a list")
    except (OSError, json.JSONDecodeError, KeyError, TypeError) as error:
        details["ok"] = False
        return details, [f"cannot read pip install report {report_path}: {error}"]

    expected_names = {canonicalize_name(name): name for name in EXPECTED_DISTRIBUTIONS}
    reported: dict[str, list[Mapping[str, Any]]] = {
        name: [] for name in EXPECTED_DISTRIBUTIONS
    }
    unexpected: list[str] = []
    for entry in entries:
        if not isinstance(entry, Mapping):
            unexpected.append(repr(entry))
            continue
        package_metadata = entry.get("metadata", {})
        if not isinstance(package_metadata, Mapping):
            unexpected.append("<invalid metadata>")
            continue
        name = str(package_metadata.get("name", ""))
        expected_name = expected_names.get(canonicalize_name(name))
        if expected_name is None:
            unexpected.append(name or "<missing name>")
        else:
            reported[expected_name].append(entry)
    if unexpected:
        failures.append(
            "pip install report contains unexpected distributions: "
            + ", ".join(sorted(unexpected))
        )

    for name in EXPECTED_DISTRIBUTIONS:
        package_entries = reported[name]
        package_details: dict[str, Any] = {"entry_count": len(package_entries)}
        details["packages"][name] = package_details
        if len(package_entries) != 1:
            failures.append(
                f"pip install report must contain {name} exactly once; "
                f"found {len(package_entries)}"
            )
            continue
        entry = package_entries[0]
        package_metadata = entry.get("metadata", {})
        download_info = entry.get("download_info", {})
        if not isinstance(package_metadata, Mapping):
            package_metadata = {}
        if not isinstance(download_info, Mapping):
            download_info = {}
        archive_info = download_info.get("archive_info", {})
        if not isinstance(archive_info, Mapping):
            archive_info = {}
        hashes = archive_info.get("hashes", {})
        if not isinstance(hashes, Mapping):
            hashes = {}
        version = str(package_metadata.get("version", ""))
        url = str(download_info.get("url", ""))
        parsed_url = urlparse(url)
        sha256 = str(hashes.get("sha256", ""))
        package_details.update(
            {
                "version": version,
                "version_matches_pin": version == pins.get(name),
                "url": url,
                "is_direct": entry.get("is_direct"),
                "requested": entry.get("requested"),
                "sha256": sha256,
            }
        )
        if version != pins.get(name):
            failures.append(
                f"pip install report version for {name} is {version!r}, "
                f"expected {pins.get(name)!r}"
            )
        if (
            parsed_url.scheme != "https"
            or parsed_url.hostname not in PYPI_FILE_HOSTS
            or not parsed_url.path.lower().endswith(".whl")
        ):
            failures.append(
                f"pip install report does not prove an official PyPI wheel for {name}: "
                f"{url!r}"
            )
        if entry.get("is_direct") is not False:
            failures.append(f"pip install report marks {name} as a direct install")
        if entry.get("requested") is not True:
            failures.append(f"pip install report does not mark {name} as requested")
        if re.fullmatch(r"[0-9a-fA-F]{64}", sha256) is None:
            failures.append(f"pip install report lacks a valid SHA-256 for {name}")

    details["ok"] = not failures
    return details, failures


def _owned_files(distribution: metadata.Distribution) -> dict[str, Path]:
    return {
        str(entry).replace("\\", "/"): Path(distribution.locate_file(entry))
        for entry in distribution.files or ()
    }


def _is_wheel_install(distribution: metadata.Distribution) -> bool:
    return any(
        str(entry).endswith(".dist-info/WHEEL") for entry in distribution.files or ()
    )


def _is_source_install(distribution: metadata.Distribution) -> bool:
    direct_url_file = next(
        (
            entry
            for entry in distribution.files or ()
            if str(entry).endswith(".dist-info/direct_url.json")
        ),
        None,
    )
    if direct_url_file is None:
        return False
    try:
        direct_url = json.loads(distribution.read_text("direct_url.json") or "{}")
    except (json.JSONDecodeError, OSError):
        # An unreadable provenance record is not trustworthy release evidence.
        return True
    return "dir_info" in direct_url


def _uses_symlink_overlay(distribution: metadata.Distribution, path: Path) -> bool:
    root = Path(distribution.locate_file("")).absolute()
    current = path.absolute()
    if not current.is_relative_to(root):
        return True
    while current != root:
        if current.is_symlink():
            return True
        current = current.parent
    return False


def _check_import_ownership(
    distribution: metadata.Distribution,
    import_name: str,
    find_spec: Callable[[str], Any],
) -> tuple[bool, str, str]:
    expected_relative = f"{import_name}/__init__.py"
    expected_path = _owned_files(distribution).get(expected_relative)
    if expected_path is None:
        return False, "", f"distribution RECORD does not own {expected_relative}"
    if _uses_symlink_overlay(distribution, expected_path):
        return (
            False,
            str(expected_path.absolute()),
            f"{expected_relative} uses a symlink or escapes the installation root",
        )

    try:
        spec = find_spec(import_name)
    except (ImportError, AttributeError, ValueError) as error:
        return False, "", f"cannot resolve import {import_name}: {error}"
    if spec is None or spec.origin is None:
        return False, "", f"cannot resolve import {import_name} to a concrete file"

    actual = Path(spec.origin).absolute()
    expected = expected_path.absolute()
    if actual != expected:
        return (
            False,
            str(actual),
            f"import {import_name} resolves outside the private distribution RECORD",
        )
    return True, str(actual), ""


def _read_owned_file(
    distribution: metadata.Distribution, relative_path: str
) -> tuple[bytes | None, str]:
    path = _owned_files(distribution).get(relative_path)
    if path is None:
        return None, f"distribution RECORD does not contain {relative_path}"
    if _uses_symlink_overlay(distribution, path):
        return None, f"distribution file uses a symlink overlay: {relative_path}"
    try:
        return path.read_bytes(), ""
    except OSError as error:
        return None, f"cannot read installed distribution file {relative_path}: {error}"


def _audit_servicer_product_environment_accesses(
    distribution: metadata.Distribution,
) -> tuple[dict[str, Any], list[str]]:
    owned_files = _owned_files(distribution)
    tokenspeed_source_files = {
        relative
        for relative in owned_files
        if relative.startswith(SERVICER_TOKENSPEED_SOURCE_PREFIX)
        and relative.endswith(".py")
    }
    # Always attempt the required root sources so a wheel whose RECORD omits
    # mm_rdma.py fails closed instead of silently narrowing the audit.
    source_files = sorted(tokenspeed_source_files | SERVICER_REQUIRED_ROOT_SOURCES)
    details: dict[str, Any] = {
        "files": [],
        "source_file_count": len(source_files),
        "violation_count": 0,
    }
    failures: list[str] = []
    if not tokenspeed_source_files:
        failures.append(
            "tokenspeed-smg-grpc-servicer RECORD contains no TokenSpeed adapter "
            "Python source"
        )

    for relative_path in source_files:
        file_details: dict[str, Any] = {
            "file": relative_path,
            "ok": False,
            "violations": [],
        }
        details["files"].append(file_details)
        contents, read_failure = _read_owned_file(distribution, relative_path)
        if contents is None:
            file_details["error"] = read_failure
            failures.append(read_failure)
            continue
        try:
            source = contents.decode("utf-8")
            tree = ast.parse(source, filename=relative_path)
        except (UnicodeDecodeError, SyntaxError) as error:
            message = f"cannot parse installed servicer source {relative_path}: {error}"
            file_details["error"] = message
            failures.append(message)
            continue

        visitor = _ProductEnvironmentAccessVisitor()
        visitor.scan(tree)
        file_details["violations"] = visitor.violations
        file_details["ok"] = not visitor.violations
        details["violation_count"] += len(visitor.violations)
        for violation in visitor.violations:
            failures.append(
                f"{relative_path}:{violation['line']}: forbidden "
                f"{violation['api']} product environment access "
                f"({violation['key']})"
            )

    details["ok"] = not failures
    return details, failures


def _audit_binding_product_environment_keys(
    binding: bytes | None,
    *,
    relative_path: str,
    read_failure: str,
) -> tuple[dict[str, Any], list[str]]:
    """Reject legacy MiniMax-M3 environment keys embedded in the Rust binding."""

    matched_keys = (
        []
        if binding is None
        else sorted(
            key for key in FORBIDDEN_BINDING_ENV_KEYS if key.encode("ascii") in binding
        )
    )
    failures: list[str] = []
    if binding is None:
        failures.append(read_failure)
    elif matched_keys:
        failures.extend(
            f"{relative_path}: compiled binding contains forbidden product "
            f"environment key {key}"
            for key in matched_keys
        )
    return (
        {
            "file": relative_path,
            "forbidden_keys": matched_keys,
            "ok": binding is not None and not matched_keys,
        },
        failures,
    )


def _run_pip_check() -> PipCheckResult:
    completed = subprocess.run(
        [sys.executable, "-m", "pip", "check"],
        check=False,
        capture_output=True,
        text=True,
        timeout=120,
    )
    return PipCheckResult(completed.returncode, completed.stdout, completed.stderr)


def _install_published_packages(
    pins: Mapping[str, str], report_path: Path
) -> PipInstallResult:
    report_path.parent.mkdir(parents=True, exist_ok=True)
    requirements = [f"{name}=={pins[name]}" for name in EXPECTED_DISTRIBUTIONS]
    completed = subprocess.run(
        [
            sys.executable,
            "-m",
            "pip",
            "install",
            "--isolated",
            "--disable-pip-version-check",
            "--force-reinstall",
            "--no-deps",
            "--no-cache-dir",
            "--only-binary=:all:",
            "--index-url",
            PYPI_INDEX_URL,
            "--report",
            str(report_path),
            *requirements,
        ],
        check=False,
        capture_output=True,
        text=True,
        timeout=600,
    )
    return PipInstallResult(completed.returncode, completed.stdout, completed.stderr)


def _bounded(value: str) -> tuple[str, bool]:
    if len(value) <= MAX_PIP_OUTPUT_CHARS:
        return value, False
    return value[:MAX_PIP_OUTPUT_CHARS], True


def audit_release_packages(
    pyproject: Path,
    *,
    distributions: Iterable[metadata.Distribution] | None = None,
    find_spec: Callable[[str], Any] = importlib.util.find_spec,
    pip_check: Callable[[], PipCheckResult] = _run_pip_check,
    pip_install_report: Path | None = None,
    python_environment: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Audit isolated Python, exact SMG wheels, provenance, and release markers."""

    installed = list(
        metadata.distributions() if distributions is None else distributions
    )
    pins, failures = _parse_exact_pins(pyproject)
    environment_details, environment_failures = _audit_python_environment(
        (
            inspect_python_environment()
            if python_environment is None
            else python_environment
        ),
        installed,
    )
    failures.extend(environment_failures)
    report_details, report_failures = _audit_pip_install_report(
        pip_install_report, pins
    )
    failures.extend(report_failures)
    packages: dict[str, Any] = {}
    selected: dict[str, metadata.Distribution] = {}

    for name, import_name in EXPECTED_DISTRIBUTIONS.items():
        instances = _distribution_instances(installed, name)
        details: dict[str, Any] = {
            "expected_version": pins.get(name),
            "import_name": import_name,
            "installed_instances": [
                {
                    "metadata_path": _metadata_path(instance),
                    "version": instance.version,
                }
                for instance in instances
            ],
        }
        packages[name] = details
        if len(instances) != 1:
            failures.append(
                f"{name} must have exactly one installed instance; found {len(instances)}"
            )
            continue

        distribution = instances[0]
        selected[name] = distribution
        expected_version = pins.get(name)
        version_matches = (
            expected_version is not None and distribution.version == expected_version
        )
        wheel_install = _is_wheel_install(distribution)
        source_install = _is_source_install(distribution)
        import_owned, import_origin, ownership_failure = _check_import_ownership(
            distribution, import_name, find_spec
        )
        details.update(
            {
                "import_origin": import_origin,
                "import_owned_by_distribution": import_owned,
                "source_install": source_install,
                "version_matches_pin": version_matches,
                "wheel_metadata_present": wheel_install,
            }
        )
        if not version_matches:
            failures.append(
                f"{name} installed version {distribution.version} does not match exact pin "
                f"{expected_version!r}"
            )
        if not wheel_install:
            failures.append(f"{name} is missing wheel installation metadata")
        if source_install:
            failures.append(f"{name} is installed from a source directory")
        if ownership_failure:
            failures.append(f"{name}: {ownership_failure}")
        prefix = Path(str(environment_details.get("prefix", "/"))).resolve()
        if import_origin and not _is_inside(import_origin, prefix):
            failures.append(
                f"{name}: import origin is outside the isolated environment"
            )

    public_packages: dict[str, list[dict[str, str]]] = {}
    for name in CONFLICTING_PUBLIC_DISTRIBUTIONS:
        instances = _distribution_instances(installed, name)
        public_packages[name] = [
            {"metadata_path": _metadata_path(instance), "version": instance.version}
            for instance in instances
        ]
        if instances:
            failures.append(
                f"conflicting public distribution {name} is installed ({len(instances)} instance(s))"
            )

    marker_checks: list[dict[str, Any]] = []
    binding_environment_scan: dict[str, Any] = {
        "file": "",
        "forbidden_keys": [],
        "ok": False,
        "skipped": True,
    }
    smg_distribution = selected.get("tokenspeed-smg")
    if smg_distribution is not None:
        binding_files = sorted(
            relative
            for relative in _owned_files(smg_distribution)
            if relative.startswith("smg/smg_rs") and relative.endswith(".so")
        )
        binding_ok = False
        binding: bytes | None = None
        marker_failure = "binding .so is missing from the distribution RECORD"
        marker_file = binding_files[0] if len(binding_files) == 1 else ""
        if len(binding_files) > 1:
            marker_failure = f"expected one SMG binding .so; found {len(binding_files)}"
        elif len(binding_files) == 1:
            binding, read_failure = _read_owned_file(smg_distribution, binding_files[0])
            binding_ok = binding is not None and b"minimax_m3_vl" in binding
            marker_failure = read_failure or "binding .so lacks minimax_m3_vl"
        marker_checks.append(
            {
                "file": marker_file,
                "name": "binding_supports_minimax_m3_vl",
                "ok": binding_ok,
            }
        )
        if not binding_ok:
            failures.append(marker_failure)
        binding_environment_scan, binding_environment_failures = (
            _audit_binding_product_environment_keys(
                binding,
                relative_path=marker_file,
                read_failure=marker_failure,
            )
        )
        failures.extend(binding_environment_failures)

    servicer_environment_scan: dict[str, Any] = {
        "files": [],
        "ok": False,
        "skipped": True,
        "source_file_count": 0,
        "violation_count": 0,
    }
    servicer_distribution = selected.get("tokenspeed-smg-grpc-servicer")
    if servicer_distribution is not None:
        servicer_environment_scan, environment_access_failures = (
            _audit_servicer_product_environment_accesses(servicer_distribution)
        )
        failures.extend(environment_access_failures)

        rdma_contents, rdma_failure = _read_owned_file(
            servicer_distribution, SERVICER_FILES["rdma_config"]
        )
        rdma_ok = rdma_contents is not None
        marker_checks.append(
            {
                "file": SERVICER_FILES["rdma_config"],
                "name": "servicer_contains_rdma_config",
                "ok": rdma_ok,
            }
        )
        if not rdma_ok:
            failures.append(rdma_failure)

        text_cache: dict[str, str | None] = {}
        read_failures: dict[str, str] = {}
        for _, file_key, _ in TEXT_MARKERS:
            if file_key in text_cache:
                continue
            contents, read_failure = _read_owned_file(
                servicer_distribution, SERVICER_FILES[file_key]
            )
            if contents is None:
                text_cache[file_key] = None
                read_failures[file_key] = read_failure
                continue
            try:
                text_cache[file_key] = contents.decode("utf-8")
            except UnicodeDecodeError as error:
                text_cache[file_key] = None
                read_failures[file_key] = (
                    f"installed source is not UTF-8: {SERVICER_FILES[file_key]}: {error}"
                )

        for marker_name, file_key, pattern in TEXT_MARKERS:
            source = text_cache[file_key]
            marker_ok = source is not None and pattern.search(source) is not None
            marker_checks.append(
                {
                    "file": SERVICER_FILES[file_key],
                    "name": marker_name,
                    "ok": marker_ok,
                }
            )
            if not marker_ok:
                failures.append(
                    read_failures.get(
                        file_key,
                        f"installed source lacks marker {marker_name}: {SERVICER_FILES[file_key]}",
                    )
                )

    try:
        pip_result = pip_check()
        pip_stdout, stdout_truncated = _bounded(pip_result.stdout)
        pip_stderr, stderr_truncated = _bounded(pip_result.stderr)
        pip_details = {
            "ok": pip_result.returncode == 0,
            "returncode": pip_result.returncode,
            "stderr": pip_stderr,
            "stderr_truncated": stderr_truncated,
            "stdout": pip_stdout,
            "stdout_truncated": stdout_truncated,
        }
        if pip_result.returncode != 0:
            failures.append("python -m pip check reported an inconsistent environment")
    except (OSError, subprocess.SubprocessError) as error:
        pip_details = {"ok": False, "error": str(error)}
        failures.append(f"cannot run python -m pip check: {error}")

    return {
        "schema_version": 4,
        "check": "minimax_m3_smg_package_preflight",
        "ok": not failures,
        "pyproject": str(pyproject.absolute()),
        "python_environment": environment_details,
        "pip_install_report": report_details,
        "packages": packages,
        "conflicting_public_distributions": public_packages,
        "marker_checks": marker_checks,
        "binding_product_environment_scan": binding_environment_scan,
        "servicer_product_environment_scan": servicer_environment_scan,
        "pip_check": pip_details,
        "failures": failures,
    }


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Validate published SMG wheels required by MiniMax-M3 active-MM CI."
    )
    parser.add_argument(
        "--pyproject",
        type=Path,
        default=Path("python/pyproject.toml"),
        help="TokenSpeed pyproject containing the exact private SMG pins.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Path for the durable JSON audit artifact.",
    )
    parser.add_argument(
        "--pip-install-report",
        type=Path,
        required=True,
        help="pip --report JSON proving the exact wheels came from official PyPI.",
    )
    parser.add_argument(
        "--install-published",
        action="store_true",
        help=(
            "Force-reinstall the exact private pins as wheels from official PyPI "
            "and create --pip-install-report before auditing."
        ),
    )
    return parser.parse_args(argv)


def main(
    argv: Sequence[str] | None = None,
    *,
    distributions: Iterable[metadata.Distribution] | None = None,
    find_spec: Callable[[str], Any] = importlib.util.find_spec,
    pip_check: Callable[[], PipCheckResult] = _run_pip_check,
    pip_install: Callable[
        [Mapping[str, str], Path], PipInstallResult
    ] = _install_published_packages,
    python_environment: Mapping[str, Any] | None = None,
) -> int:
    args = parse_args(argv)
    effective_python_environment = (
        inspect_python_environment()
        if python_environment is None
        else python_environment
    )
    install_details: dict[str, Any] = {"performed": args.install_published}
    install_failure = ""
    if args.install_published:
        args.pip_install_report.parent.mkdir(parents=True, exist_ok=True)
        pins, pin_failures = _parse_exact_pins(args.pyproject)
        _, isolation_failures = _audit_python_environment(
            effective_python_environment, []
        )
        if isolation_failures:
            install_result = PipInstallResult(
                1,
                "",
                "refusing to install outside a strict isolated Python: "
                + "; ".join(isolation_failures),
            )
        elif pin_failures:
            install_result = PipInstallResult(1, "", "; ".join(pin_failures))
        else:
            try:
                install_result = pip_install(pins, args.pip_install_report)
            except (OSError, subprocess.SubprocessError) as error:
                install_result = PipInstallResult(1, "", str(error))
        install_stdout, stdout_truncated = _bounded(install_result.stdout)
        install_stderr, stderr_truncated = _bounded(install_result.stderr)
        install_details.update(
            {
                "ok": install_result.returncode == 0,
                "returncode": install_result.returncode,
                "stdout": install_stdout,
                "stdout_truncated": stdout_truncated,
                "stderr": install_stderr,
                "stderr_truncated": stderr_truncated,
            }
        )
        if install_result.returncode != 0:
            install_failure = (
                "official PyPI wheel installation failed with exit code "
                f"{install_result.returncode}"
            )

    result = audit_release_packages(
        args.pyproject,
        distributions=distributions,
        find_spec=find_spec,
        pip_check=pip_check,
        pip_install_report=args.pip_install_report,
        python_environment=effective_python_environment,
    )
    result["published_install"] = install_details
    if install_failure:
        result["failures"].append(install_failure)
        result["ok"] = False
    _write_json_atomic(args.output, result)

    if result["ok"]:
        print(f"MiniMax-M3 SMG package preflight passed: {args.output}")
        return 0

    print("MiniMax-M3 SMG package preflight failed:")
    for failure in result["failures"]:
        print(f"- {failure}")
    print(f"Audit artifact: {args.output}")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
