#!/usr/bin/env python3
"""Validate the published SMG package set before MiniMax-M3 GPU startup.

The active multimodal release job must exercise the exact private packages
declared by TokenSpeed, not similarly named public packages or a source-tree
overlay.  All package contents are located through ``importlib.metadata``;
this check never falls back to an SMG source checkout.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import re
import subprocess
import sys
import tomllib
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass
from importlib import metadata
from pathlib import Path
from typing import Any

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


@dataclass(frozen=True)
class PipCheckResult:
    """Result of invoking ``python -m pip check``."""

    returncode: int
    stdout: str = ""
    stderr: str = ""


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


def _run_pip_check() -> PipCheckResult:
    completed = subprocess.run(
        [sys.executable, "-m", "pip", "check"],
        check=False,
        capture_output=True,
        text=True,
        timeout=120,
    )
    return PipCheckResult(completed.returncode, completed.stdout, completed.stderr)


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
) -> dict[str, Any]:
    """Audit exact SMG pins, installed wheel ownership, and release markers."""

    installed = list(
        metadata.distributions() if distributions is None else distributions
    )
    pins, failures = _parse_exact_pins(pyproject)
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
    smg_distribution = selected.get("tokenspeed-smg")
    if smg_distribution is not None:
        binding_files = sorted(
            relative
            for relative in _owned_files(smg_distribution)
            if relative.startswith("smg/smg_rs") and relative.endswith(".so")
        )
        binding_ok = False
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

    servicer_distribution = selected.get("tokenspeed-smg-grpc-servicer")
    if servicer_distribution is not None:
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
        "schema_version": 1,
        "check": "minimax_m3_smg_package_preflight",
        "ok": not failures,
        "pyproject": str(pyproject.absolute()),
        "packages": packages,
        "conflicting_public_distributions": public_packages,
        "marker_checks": marker_checks,
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
    return parser.parse_args(argv)


def main(
    argv: Sequence[str] | None = None,
    *,
    distributions: Iterable[metadata.Distribution] | None = None,
    find_spec: Callable[[str], Any] = importlib.util.find_spec,
    pip_check: Callable[[], PipCheckResult] = _run_pip_check,
) -> int:
    args = parse_args(argv)
    result = audit_release_packages(
        args.pyproject,
        distributions=distributions,
        find_spec=find_spec,
        pip_check=pip_check,
    )
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
