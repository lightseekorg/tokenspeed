# Copyright (c) 2026 LightSeek Foundation
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

from __future__ import annotations

import importlib.util
import os
import shutil
import subprocess
import sys
import zipfile
from pathlib import Path
from unittest.mock import patch

from tokenspeed_kernel.registry import KernelRegistry, load_builtin_kernels

ROOT = Path(__file__).resolve().parents[1]
PYTHON_ROOT = ROOT / "python"
SETUP_PY = PYTHON_ROOT / "setup.py"
BUILD_ARTIFACTS = (
    PYTHON_ROOT / "build",
    PYTHON_ROOT / "tokenspeed_kernel.egg-info",
    PYTHON_ROOT / "tokenspeed_kernel_nvidia.egg-info",
    PYTHON_ROOT / "tokenspeed_kernel_amd.egg-info",
)
FIXED_ENV = {
    "TOKENSPEED_KERNEL_VERSION_DATE": "20260101",
    "TOKENSPEED_KERNEL_GIT_SHA": "abcdef12",
    "TOKENSPEED_KERNEL_GIT_BRANCH": "main",
}

CORE_FORBIDDEN_VENDOR_PATHS = (
    "tokenspeed_kernel/ops/attention/gluon/mha_decode_fp16_gfx950.py",
    "tokenspeed_kernel/ops/attention/gluon/mha_prefill_fp16_gfx950.py",
    "tokenspeed_kernel/ops/gemm/cute_dsl.py",
    "tokenspeed_kernel/ops/gemm/flashinfer.py",
    "tokenspeed_kernel/ops/moe/gluon.py",
    "tokenspeed_kernel/ops/moe/flashinfer.py",
    "tokenspeed_kernel/thirdparty/cute_dsl/argmax.py",
    "tokenspeed_kernel/thirdparty/cute_dsl/nvfp4_gemm_swiglu_nvfp4_quant.py",
)


def _clean_build_artifacts() -> None:
    for path in BUILD_ARTIFACTS:
        if path.is_dir():
            shutil.rmtree(path)
        elif path.exists():
            path.unlink()


def _build_wheel(mode: str, dist_dir: Path) -> Path:
    _clean_build_artifacts()
    dist_dir.mkdir()
    env = os.environ.copy()
    env.update(FIXED_ENV)
    env["TOKENSPEED_KERNEL_PACKAGE"] = mode
    try:
        subprocess.run(
            [
                sys.executable,
                "setup.py",
                "bdist_wheel",
                "--dist-dir",
                str(dist_dir),
            ],
            cwd=PYTHON_ROOT,
            env=env,
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
        )
    finally:
        _clean_build_artifacts()
    wheels = sorted(dist_dir.glob("*.whl"))
    assert len(wheels) == 1
    return wheels[0]


def _wheel_names(wheel: Path) -> list[str]:
    with zipfile.ZipFile(wheel) as archive:
        return archive.namelist()


def _load_setup_kwargs(mode: str) -> dict:
    captured = {}

    def fake_setup(**kwargs):
        captured.update(kwargs)

    old_env = os.environ.copy()
    old_cwd = Path.cwd()
    os.chdir(PYTHON_ROOT)
    os.environ.update(FIXED_ENV)
    os.environ["TOKENSPEED_KERNEL_PACKAGE"] = mode
    try:
        spec = importlib.util.spec_from_file_location(
            f"tokenspeed_kernel_setup_{mode}", SETUP_PY
        )
        assert spec is not None and spec.loader is not None
        module = importlib.util.module_from_spec(spec)
        with patch("setuptools.setup", fake_setup):
            spec.loader.exec_module(module)
    finally:
        os.chdir(old_cwd)
        os.environ.clear()
        os.environ.update(old_env)
    return captured


def test_package_mode_distribution_names() -> None:
    assert _load_setup_kwargs("core")["name"] == "tokenspeed-kernel"
    assert _load_setup_kwargs("nvidia")["name"] == "tokenspeed-kernel-nvidia"
    assert _load_setup_kwargs("amd")["name"] == "tokenspeed-kernel-amd"


def test_dependency_direction() -> None:
    core = _load_setup_kwargs("core")
    nvidia = _load_setup_kwargs("nvidia")
    amd = _load_setup_kwargs("amd")

    version = core["version"]
    assert f"tokenspeed-kernel-nvidia=={version}" in core["install_requires"]
    assert f"tokenspeed-kernel-amd=={version}" in core["install_requires"]
    assert not any(
        req.startswith("tokenspeed-kernel") for req in nvidia["install_requires"]
    )
    assert not any(
        req.startswith("tokenspeed-kernel") for req in amd["install_requires"]
    )


def test_package_mode_selects_vendor_requirements() -> None:
    nvidia_requires = _load_setup_kwargs("nvidia")["install_requires"]
    amd_requires = _load_setup_kwargs("amd")["install_requires"]

    assert any(req.startswith("nvidia-cutlass-dsl==") for req in nvidia_requires)
    assert any(req.startswith("flashinfer-python==") for req in nvidia_requires)
    assert not any("+rocm" in req for req in nvidia_requires)
    assert any("+rocm" in req for req in amd_requires)
    assert not any(req.startswith("nvidia-cutlass-dsl==") for req in amd_requires)


def test_package_mode_boundaries() -> None:
    for mode, prefix in (
        ("core", "tokenspeed_kernel"),
        ("nvidia", "tokenspeed_kernel_nvidia"),
        ("amd", "tokenspeed_kernel_amd"),
    ):
        packages = _load_setup_kwargs(mode)["packages"]
        assert packages
        assert all(pkg == prefix or pkg.startswith(prefix + ".") for pkg in packages)


def test_wheel_artifact_boundaries(tmp_path) -> None:
    wheels = {
        mode: _build_wheel(mode, tmp_path / mode) for mode in ("nvidia", "amd", "core")
    }

    core_names = _wheel_names(wheels["core"])
    assert any(name.startswith("tokenspeed_kernel/") for name in core_names)
    assert not any(
        name.startswith(("tokenspeed_kernel_nvidia/", "tokenspeed_kernel_amd/"))
        for name in core_names
    )
    assert not any(
        name.startswith("tokenspeed_kernel/thirdparty/") for name in core_names
    )
    for forbidden_path in CORE_FORBIDDEN_VENDOR_PATHS:
        assert forbidden_path not in core_names

    for mode, prefix in (
        ("nvidia", "tokenspeed_kernel_nvidia/"),
        ("amd", "tokenspeed_kernel_amd/"),
    ):
        names = _wheel_names(wheels[mode])
        assert any(name.startswith(prefix) for name in names)
        assert not any(name.startswith("tokenspeed_kernel/") for name in names)


def test_load_builtin_kernels_smoke(fresh_registry) -> None:
    load_builtin_kernels()
    names = {spec.name for spec in KernelRegistry.get().list_kernels()}
    assert "torch_mm" in names
    assert "triton_quantize_fp8" in names


def test_core_sources_do_not_duplicate_vendor_implementations() -> None:
    for forbidden_path in CORE_FORBIDDEN_VENDOR_PATHS:
        assert not (PYTHON_ROOT / forbidden_path).exists()
    assert not (PYTHON_ROOT / "tokenspeed_kernel/thirdparty").exists()


def test_vendor_thirdparty_paths_import_from_vendor_namespace() -> None:
    from tokenspeed_kernel_nvidia.thirdparty import trtllm
    from tokenspeed_kernel_nvidia.thirdparty.cuda.merge_state import merge_state

    assert trtllm.__name__ == "tokenspeed_kernel_nvidia.thirdparty.trtllm"
    assert (
        merge_state.__module__ == "tokenspeed_kernel_nvidia.thirdparty.cuda.merge_state"
    )


def test_vendor_top_level_imports_without_core() -> None:
    env = os.environ.copy()
    env["PYTHONPATH"] = str(PYTHON_ROOT)
    code = (
        "import sys; "
        "import tokenspeed_kernel_nvidia, tokenspeed_kernel_amd; "
        'assert "tokenspeed_kernel" not in sys.modules'
    )
    subprocess.run([sys.executable, "-c", code], env=env, check=True)
