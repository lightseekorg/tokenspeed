#!/usr/bin/env python3
"""Verify preinstalled ROCm CI package stacks before refreshing them."""

import importlib
import importlib.metadata
import re
import sys


def _installed_version(distribution: str) -> str | None:
    try:
        return importlib.metadata.version(distribution)
    except importlib.metadata.PackageNotFoundError:
        return None


def _version_matches(actual: str, expected: str) -> bool:
    if "+" in expected:
        return actual == expected
    return actual == expected or actual.startswith(f"{expected}+")


def _require_exact(errors: list[str], distribution: str, expected: str) -> None:
    actual = _installed_version(distribution)
    if actual != expected:
        errors.append(
            f"{distribution}: expected {expected}, found {actual or 'missing'}"
        )


def _verify_mi450_sdk(uv_version: str, rocm_version: str) -> list[str]:
    errors: list[str] = []
    _require_exact(errors, "uv", uv_version)
    for distribution in (
        "rocm",
        "rocm-sdk-core",
        "rocm-sdk-devel",
        "rocm-sdk-libraries",
        "rocm-sdk-device-gfx1250",
    ):
        _require_exact(errors, distribution, rocm_version)

    pytest_version = _installed_version("pytest")
    match = re.match(r"(\d+)", pytest_version or "")
    if match is None or int(match.group(1)) < 9:
        errors.append(f"pytest: expected >=9, found {pytest_version or 'missing'}")
    for distribution in ("pytest-reportlog", "pytest-timeout", "pytest-xdist"):
        if _installed_version(distribution) is None:
            errors.append(f"{distribution}: missing")
    return errors


def _verify_torch(
    torch_version: str, torchvision_version: str, device_requirement: str
) -> list[str]:
    errors: list[str] = []
    for module_name, expected in (
        ("torch", torch_version),
        ("torchvision", torchvision_version),
    ):
        try:
            module = importlib.import_module(module_name)
        except Exception as exc:
            errors.append(f"{module_name}: import failed: {exc}")
            continue
        actual = getattr(module, "__version__", "")
        if not _version_matches(actual, expected):
            errors.append(f"{module_name}: expected {expected}, found {actual}")

    if device_requirement:
        distribution, separator, expected = device_requirement.partition("==")
        if not separator or not distribution or not expected:
            errors.append(
                f"device package must be an exact requirement: {device_requirement}"
            )
        else:
            _require_exact(errors, distribution, expected)
    return errors


def main(argv: list[str]) -> int:
    if len(argv) == 4 and argv[1] == "mi450-sdk":
        errors = _verify_mi450_sdk(argv[2], argv[3])
    elif len(argv) == 5 and argv[1] == "torch":
        errors = _verify_torch(argv[2], argv[3], argv[4])
    else:
        print(
            "usage: verify_rocm_stack.py mi450-sdk <uv-version> <rocm-version>\n"
            "   or: verify_rocm_stack.py torch <torch-version> "
            "<torchvision-version> <device-requirement>",
            file=sys.stderr,
        )
        return 2

    if errors:
        print("\n".join(errors), file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
