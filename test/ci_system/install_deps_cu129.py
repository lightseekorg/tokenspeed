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

"""Experimental H100/H200 + CUDA 12.9 source install into an active Python 3.11 venv."""

import importlib.metadata
import os
import platform
import re
import runpy
import subprocess
import sys
import tempfile
import tomllib
from pathlib import Path

PYPI = "https://pypi.org/simple"
PYTORCH = "https://download.pytorch.org/whl/cu129"
WHEELS = "https://lightseek.org/whl/cu129/"
NATIVE_PACKAGES = {
    "tokenspeed-deepep",
    "tokenspeed-deepgemm",
    "tokenspeed-flashmla",
    "tokenspeed-fast-hadamard-transform",
    "tokenspeed-fa3",
    "tokenspeed-trtllm-kernel",
    "tokenspeed-flashkda",
}


CUDA13_UNSUFFIXED_PACKAGES = {
    "tokenspeed-cutedsl-kda",  # Current AOT distribution is CUDA-13-only.
    "nvidia-cublas",
    "nvidia-cuda-runtime",
    "nvidia-cuda-nvrtc",
}
CUDA_VERSIONED_PACKAGES = {"cuda-toolkit", "cuda-python", "cuda-bindings"}


def validate_environment(workspace: Path, cuda_home: Path) -> None:
    """Reject an unsupported or mixed CUDA environment before installing packages.

    Args:
        workspace: Checkout whose packages and constraints will be installed.
        cuda_home: Host-supplied CUDA Toolkit directory.

    Returns:
        None when Linux, Python 3.11, an active venv and CUDA 12.9 are present.
    """
    if platform.system() != "Linux":
        raise RuntimeError(
            "The Hopper/cu129 installer requires Linux; no packages were changed"
        )
    if sys.version_info[:2] != (3, 11):
        raise RuntimeError(
            "Activate a Python 3.11 venv before running the Hopper/cu129 installer"
        )
    if sys.prefix == sys.base_prefix:
        raise RuntimeError(
            "Activate a Python venv before running the Hopper/cu129 installer"
        )
    for relative in (
        "requirements/nvidia-cu129-constraints.txt",
        "tokenspeed-kernel/python/setup.py",
        "tokenspeed-scheduler/pyproject.toml",
        "python/pyproject.toml",
    ):
        if not (workspace / relative).is_file():
            raise RuntimeError(
                f"Incomplete TokenSpeed checkout: missing {workspace / relative}"
            )

    conflicts = []
    for distribution in importlib.metadata.distributions():
        name = re.sub(r"[-_.]+", "-", distribution.metadata.get("Name", "")).lower()
        version = distribution.version
        if (
            (name.startswith("nvidia-") and name.endswith("-cu13"))
            or name in CUDA13_UNSUFFIXED_PACKAGES
            or (name in CUDA_VERSIONED_PACKAGES and re.match(r"^13(?:\.|$)", version))
            or (name in {"torch", "torchvision"} and re.search(r"\+cu13[0-9]", version))
        ):
            conflicts.append(f"{name}=={version}")
    if conflicts:
        raise RuntimeError(
            "CUDA 13 packages are already installed: "
            + ", ".join(sorted(conflicts))
            + ". Use a fresh Python 3.11 venv for cu129; this installer will not "
            "uninstall or replace an existing CUDA 13 environment."
        )
    nvcc = cuda_home / "bin" / "nvcc"
    try:
        result = subprocess.run(
            [str(nvcc), "--version"],
            check=True,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError) as exc:
        raise RuntimeError(
            f"CUDA_HOME must contain a working CUDA 12.9 nvcc: {nvcc}"
        ) from exc
    match = re.search(r"\brelease\s+(\d+)\.(\d+)", result.stdout)
    if match is None or tuple(map(int, match.groups())) != (12, 9):
        raise RuntimeError(
            f"CUDA_HOME must select CUDA Toolkit 12.9, got {result.stdout.strip()!r}"
        )


def kernel_requirements(workspace: Path) -> list[str]:
    # Evaluate metadata from the checkout, without running any build command.
    import setuptools

    captured = {}
    original = setuptools.setup
    try:
        setuptools.setup = lambda **kwargs: captured.update(kwargs)
        runpy.run_path(str(workspace / "tokenspeed-kernel" / "python" / "setup.py"))
    finally:
        setuptools.setup = original
    return captured["install_requires"]


def pip_install(arguments: list[str], index: str) -> None:
    subprocess.run(
        [sys.executable, "-m", "pip", "install", "--index-url", index, *arguments],
        check=True,
    )


def main() -> None:
    workspace = Path(os.environ.get("WORKSPACE", Path(__file__).resolve().parents[2]))
    workspace = workspace.resolve()
    cuda_home = Path(os.environ.get("CUDA_HOME", "/usr/local/cuda")).resolve()
    validate_environment(workspace, cuda_home)
    os.environ.update(
        CUDA_HOME=str(cuda_home),
        FLASHINFER_NVCC=str(cuda_home / "bin" / "nvcc"),
        TOKENSPEED_KERNEL_BACKEND="cuda",
        TOKENSPEED_KERNEL_CUDA_VARIANT="cu129",
        FLASHINFER_CUDA_ARCH_LIST="9.0a",
        MAX_JOBS=os.environ.get("BUILD_AND_DOWNLOAD_PARALLEL", "16"),
        PIP_CONSTRAINT=str(workspace / "requirements" / "nvidia-cu129-constraints.txt"),
        PIP_CONFIG_FILE=os.devnull,
        PIP_INDEX_URL=PYPI,
    )
    # Identical native package versions exist on PyPI with a different CUDA
    # ABI. Each install below uses one source, including in configured hosts.
    for name in (
        "PIP_EXTRA_INDEX_URL",
        "PIP_FIND_LINKS",
        # A requirements file can itself add indexes or wheel directories.
        "PIP_REQUIREMENT",
        "PIP_NO_INDEX",
        "PIP_NO_DEPS",
        "PIP_FORCE_REINSTALL",
        "PIP_IGNORE_INSTALLED",
        "PIP_UPGRADE",
        "PIP_TARGET",
        "PIP_PREFIX",
        "PIP_USER",
    ):
        os.environ.pop(name, None)
    print(
        f"Experimental Hopper/cu129 install: {workspace}, Python: {sys.executable}",
        flush=True,
    )

    pip_install(["--upgrade", "pip", "setuptools==83.0.0", "wheel", "packaging"], PYPI)
    requirements = kernel_requirements(workspace)
    pip_install(["torch", "torchvision"], PYTORCH)

    native = [req for req in requirements if req.split("==")[0] in NATIVE_PACKAGES]
    runtime = tomllib.loads((workspace / "python" / "pyproject.toml").read_text())
    mooncake = [
        req
        for req in runtime["project"]["dependencies"]
        if req.startswith("tokenspeed-mooncake>=")
    ]
    pip_install(["--force-reinstall", "--no-deps", *native, *mooncake], WHEELS)

    flashinfer = next(
        req.split("==")[1]
        for req in requirements
        if req.startswith("flashinfer-python==")
    )
    cubin = (
        "https://github.com/flashinfer-ai/flashinfer/releases/download/"
        f"v{flashinfer}/flashinfer_cubin-{flashinfer}-py3-none-any.whl"
    )
    # Resolve native wheels' transitive dependencies after the right binaries
    # are installed. Matching public versions (including local suffixes) stay.
    pip_install([*requirements, *mooncake, cubin], PYPI)
    pip_install(
        [
            "--no-build-isolation",
            "--no-deps",
            "-e",
            str(workspace / "tokenspeed-kernel" / "python"),
        ],
        PYPI,
    )
    with tempfile.TemporaryDirectory(prefix="tokenspeed-cu129-scheduler-") as build_dir:
        pip_install(
            [
                str(workspace / "tokenspeed-scheduler"),
                f"--config-settings=build-dir={build_dir}",
            ],
            PYPI,
        )
    pip_install(["--no-build-isolation", "-e", str(workspace / "python")], PYPI)
    subprocess.run([sys.executable, "-m", "pip", "check"], check=True)


if __name__ == "__main__":
    main()
