from __future__ import annotations

import argparse
import platform
import re
import sys
import urllib.request
import zipfile
from email.parser import Parser
from importlib import metadata
from pathlib import Path
from urllib.parse import unquote, urljoin, urlparse

from packaging.requirements import Requirement

JIT_CACHE_DIST = "flashinfer-jit-cache"
FLASHINFER_PYTHON_DIST = "flashinfer-python"
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


def jit_cache_dependencies_satisfied() -> bool:
    try:
        requirements = metadata.requires(JIT_CACHE_DIST) or []
    except metadata.PackageNotFoundError:
        return False
    for spec in requirements:
        requirement = Requirement(spec)
        if requirement.marker and not requirement.marker.evaluate():
            continue
        installed = installed_distribution_version(requirement.name)
        print(f"Installed {requirement.name}=={installed}", file=sys.stderr)
        if installed is None or installed not in requirement.specifier:
            return False
    return True


def expected_jit_cache_version(flashinfer_version: str, cuda_index: str) -> str:
    return f"{flashinfer_version}+cu{cuda_index}"


def release_tag(flashinfer_version: str) -> str:
    """GitHub release tag: vX.Y.Z[rcN], or nightly-vX.Y.Z-YYYYMMDD for
    nightly versions of the form X.Y.Z.devYYYYMMDD."""
    match = re.fullmatch(r"(\d+\.\d+\.\d+)\.dev(\d{8})", flashinfer_version)
    if match:
        return f"nightly-v{match.group(1)}-{match.group(2)}"
    return f"v{flashinfer_version}"


def default_platform_tag() -> str:
    return f"manylinux_2_28_{platform.machine()}"


def jit_cache_wheel_url(
    flashinfer_version: str,
    cuda_index: str,
    *,
    platform_tag: str | None = None,
) -> str:
    if platform_tag is None:
        platform_tag = default_platform_tag()
    wheel = (
        f"flashinfer_jit_cache-{flashinfer_version}+cu{cuda_index}"
        f"-cp39-abi3-{platform_tag}.whl"
    )
    return (
        "https://github.com/flashinfer-ai/flashinfer/releases/download/"
        f"{release_tag(flashinfer_version)}/{wheel}"
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
    if current_version == expected_version and jit_cache_dependencies_satisfied():
        return None, expected_version, current_version
    return (
        jit_cache_wheel_url(flashinfer_version, cuda_index),
        expected_version,
        current_version,
    )


def simple_index_url(cuda_index: str, flashinfer_version: str) -> str:
    """FlashInfer wheel index carrying the jit-cache wheels and their hashes."""
    if re.fullmatch(r"\d+\.\d+\.\d+\.dev\d{8}", flashinfer_version):
        return f"https://flashinfer.ai/whl/nightly/cu{cuda_index}"
    return f"https://flashinfer.ai/whl/cu{cuda_index}"


def index_wheel_url(
    index_url: str,
    distribution: str,
    version: str,
    *,
    platform_tag: str | None = None,
) -> tuple[str, str]:
    """Resolve distribution==version to its wheel URL and sha256 from a
    PEP 503 index page."""
    if platform_tag is None:
        platform_tag = default_platform_tag()
    page_url = f"{index_url}/{distribution}/"
    wheel_name = (
        f"{distribution.replace('-', '_')}-{version}-cp39-abi3-{platform_tag}.whl"
    )
    with urllib.request.urlopen(page_url, timeout=30) as response:
        page = response.read().decode()
    for link in re.findall(r'href="([^"]+)"', page):
        if not unquote(urlparse(link).path).endswith(f"/{wheel_name}"):
            continue
        match = re.search(r"#sha256=([0-9a-f]{64})", link)
        if match:
            # Decode %2B et al.: pip rejects percent-encoded wheel filenames.
            url = unquote(urljoin(page_url, link)).split("#", 1)[0]
            return url, match.group(1)
    raise ValueError(f"{wheel_name} with a sha256 fragment not found in {page_url}")


def read_wheel_requires(wheel_path: Path) -> list[str]:
    """Requires-Dist entries from a local wheel's METADATA."""
    with zipfile.ZipFile(wheel_path) as archive:
        for name in archive.namelist():
            if name.endswith(".dist-info/METADATA"):
                text = archive.read(name).decode()
                break
        else:
            raise ValueError(f"METADATA not found in {wheel_path}")
    return Parser().parsestr(text).get_all("Requires-Dist", [])


def exact_requirement_version(requirement: Requirement) -> str:
    for specifier in requirement.specifier:
        if specifier.operator == "==":
            return specifier.version
    raise ValueError(f"{requirement} does not pin an exact version")


def unsatisfied_providers(wheel_path: Path) -> list[Requirement]:
    """Architecture provider requirements of the metapackage wheel that the
    active markers need but the installed distributions do not satisfy."""
    unsatisfied = []
    for spec in read_wheel_requires(wheel_path):
        requirement = Requirement(spec)
        if requirement.marker and not requirement.marker.evaluate():
            continue
        installed = installed_distribution_version(requirement.name)
        if installed is None or installed not in requirement.specifier:
            unsatisfied.append(requirement)
    return unsatisfied


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--requirements", type=Path, required=True)
    parser.add_argument("--cuda-index", required=True)
    parser.add_argument(
        "--resolve-metapackage",
        action="store_true",
        help="print the wheel URL and its sha256 from the FlashInfer index "
        "instead of the bare GitHub release URL",
    )
    parser.add_argument(
        "--list-providers",
        type=Path,
        metavar="WHEEL",
        help="print the URL and sha256 of each unsatisfied architecture "
        "provider required by this local metapackage wheel",
    )
    args = parser.parse_args(argv)

    flashinfer_version = read_exact_pin(args.requirements, FLASHINFER_PYTHON_DIST)
    index_url = simple_index_url(args.cuda_index, flashinfer_version)
    try:
        if args.list_providers is not None:
            for requirement in unsatisfied_providers(args.list_providers):
                url, sha256 = index_wheel_url(
                    index_url,
                    requirement.name,
                    exact_requirement_version(requirement),
                )
                print(url, sha256)
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
            f"{JIT_CACHE_DIST} {current} or its dependencies do not match "
            f"{expected_version}; installing matching wheel and dependencies",
            file=sys.stderr,
        )
        if args.resolve_metapackage:
            url, sha256 = index_wheel_url(index_url, JIT_CACHE_DIST, expected_version)
            print(url, sha256)
        else:
            print(url)
        return 0
    except (OSError, ValueError) as error:
        raise SystemExit(f"error: {error}") from error


if __name__ == "__main__":
    raise SystemExit(main())
