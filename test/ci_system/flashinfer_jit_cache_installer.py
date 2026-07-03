from __future__ import annotations

import argparse
import re
import sys
from importlib import metadata
from pathlib import Path

JIT_CACHE_DIST = "flashinfer-jit-cache"
FLASHINFER_PYTHON_DIST = "flashinfer-python"
JIT_CACHE_RUNNER_PREFIXES = ("b200", "gb200")
_QUERY_INSTALLED = object()


def read_exact_pin(requirements_path: Path, package: str) -> str:
    pattern = re.compile(rf"^\s*{re.escape(package)}==([^#\s]+)")
    for line in requirements_path.read_text().splitlines():
        match = pattern.match(line)
        if match:
            return match.group(1)
    raise ValueError(f"{package} exact pin not found in {requirements_path}")


def installed_distribution_version(distribution: str = JIT_CACHE_DIST) -> str | None:
    try:
        return metadata.version(distribution)
    except metadata.PackageNotFoundError:
        return None


def should_sync_jit_cache(runner_label: str | None) -> bool:
    return bool(runner_label and runner_label.startswith(JIT_CACHE_RUNNER_PREFIXES))


def expected_jit_cache_version(flashinfer_version: str, cuda_index: str) -> str:
    return f"{flashinfer_version}+cu{cuda_index}"


def jit_cache_wheel_url(
    flashinfer_version: str,
    cuda_index: str,
    *,
    platform_tag: str = "manylinux_2_28_aarch64",
) -> str:
    wheel = (
        f"flashinfer_jit_cache-{flashinfer_version}+cu{cuda_index}"
        f"-cp39-abi3-{platform_tag}.whl"
    )
    return (
        "https://github.com/flashinfer-ai/flashinfer/releases/download/"
        f"v{flashinfer_version}/{wheel}"
    )


def install_url_if_needed(
    requirements_path: Path,
    cuda_index: str,
    installed_version: str | None | object = _QUERY_INSTALLED,
) -> tuple[str | None, str, str | None]:
    flashinfer_version = read_exact_pin(requirements_path, FLASHINFER_PYTHON_DIST)
    expected_version = expected_jit_cache_version(flashinfer_version, cuda_index)
    current_version = (
        installed_distribution_version()
        if installed_version is _QUERY_INSTALLED
        else installed_version
    )
    if current_version == expected_version:
        return None, expected_version, current_version
    return (
        jit_cache_wheel_url(flashinfer_version, cuda_index),
        expected_version,
        current_version,
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--requirements", type=Path, required=True)
    parser.add_argument("--cuda-index", required=True)
    parser.add_argument("--runner-label")
    args = parser.parse_args(argv)

    if args.runner_label is not None and not should_sync_jit_cache(args.runner_label):
        print(
            f"skipping {JIT_CACHE_DIST} sync on runner {args.runner_label}",
            file=sys.stderr,
        )
        return 0

    url, expected_version, installed_version = install_url_if_needed(
        args.requirements,
        args.cuda_index,
    )
    if url is None:
        print(
            f"{JIT_CACHE_DIST} {installed_version} already matches "
            f"{expected_version}",
            file=sys.stderr,
        )
        return 0

    current = installed_version or "not installed"
    print(
        f"{JIT_CACHE_DIST} {current} does not match {expected_version}; "
        "installing matching wheel",
        file=sys.stderr,
    )
    print(url)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
