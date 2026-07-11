from __future__ import annotations

import runpy
from pathlib import Path

import setuptools

SETUP_PY = Path(__file__).parents[1] / "python" / "setup.py"


def test_cuda_include_dirs_prefer_complete_toolkit_headers(
    tmp_path, monkeypatch
) -> None:
    monkeypatch.setenv("TOKENSPEED_KERNEL_BACKEND", "cuda")
    monkeypatch.setattr(setuptools, "setup", lambda **_kwargs: None)
    setup_namespace = runpy.run_path(str(SETUP_PY))

    cuda_root = tmp_path / "cuda"
    cuda_include = cuda_root / "include"
    cuda_include.mkdir(parents=True)
    (cuda_include / "cuda_runtime.h").touch()
    (cuda_include / "cublas_v2.h").touch()

    site_packages = tmp_path / "site-packages"
    wheel_include = site_packages / "nvidia" / "cu13" / "include"
    wheel_include.mkdir(parents=True)
    (wheel_include / "cuda_runtime.h").touch()
    (wheel_include / "cublas_v2.h").touch()

    builder = setup_namespace["CudaKernelBuilder"]([], verbose=False)
    monkeypatch.setattr(builder, "_cuda_toolkit_roots", lambda: iter([cuda_root]))
    monkeypatch.setattr(builder, "_site_paths", lambda: iter([site_packages]))

    include_dirs = builder._resolve_include_dirs()

    assert str(cuda_include) in include_dirs
    assert str(wheel_include) not in include_dirs


def test_cuda_include_dirs_fall_back_from_partial_toolkit(
    tmp_path, monkeypatch
) -> None:
    monkeypatch.setenv("TOKENSPEED_KERNEL_BACKEND", "cuda")
    monkeypatch.setattr(setuptools, "setup", lambda **_kwargs: None)
    setup_namespace = runpy.run_path(str(SETUP_PY))

    cuda_root = tmp_path / "cuda"
    cuda_include = cuda_root / "include"
    cuda_include.mkdir(parents=True)
    (cuda_include / "cuda_runtime.h").touch()

    site_packages = tmp_path / "site-packages"
    wheel_include = site_packages / "nvidia" / "cu13" / "include"
    wheel_include.mkdir(parents=True)
    (wheel_include / "cuda_runtime.h").touch()
    (wheel_include / "cublas_v2.h").touch()

    builder = setup_namespace["CudaKernelBuilder"]([], verbose=False)
    monkeypatch.setattr(builder, "_cuda_toolkit_roots", lambda: iter([cuda_root]))
    monkeypatch.setattr(builder, "_site_paths", lambda: iter([site_packages]))

    include_dirs = builder._resolve_include_dirs()

    assert str(cuda_include) not in include_dirs
    assert str(wheel_include) in include_dirs
