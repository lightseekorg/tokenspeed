from pathlib import Path

import pytest
from flashinfer_jit_cache_installer import (
    expected_jit_cache_version,
    install_url_if_needed,
    jit_cache_wheel_url,
    read_exact_pin,
)


def test_read_exact_pin_ignores_other_requirements(tmp_path: Path):
    requirements = tmp_path / "cuda.txt"
    requirements.write_text(
        "-r common.txt\n" "torch==2.11.0\n" "flashinfer-python==0.6.15.post1\n"
    )

    assert read_exact_pin(requirements, "flashinfer-python") == "0.6.15.post1"


def test_read_exact_pin_requires_exact_pin(tmp_path: Path):
    requirements = tmp_path / "cuda.txt"
    requirements.write_text("flashinfer-python>=0.6\n")

    with pytest.raises(ValueError, match="flashinfer-python exact pin not found"):
        read_exact_pin(requirements, "flashinfer-python")


def test_jit_cache_url_tracks_flashinfer_and_cuda_versions():
    assert expected_jit_cache_version("0.6.15.post1", "130") == "0.6.15.post1+cu130"
    assert jit_cache_wheel_url("0.6.15.post1", "130") == (
        "https://github.com/flashinfer-ai/flashinfer/releases/download/"
        "v0.6.15.post1/flashinfer_jit_cache-0.6.15.post1+cu130-cp39-abi3-"
        "manylinux_2_28_aarch64.whl"
    )


def test_install_url_if_needed_skips_matching_version(tmp_path: Path):
    requirements = tmp_path / "cuda.txt"
    requirements.write_text("flashinfer-python==0.6.15.post1\n")

    url, expected, installed = install_url_if_needed(
        requirements,
        "130",
        installed_version="0.6.15.post1+cu130",
    )

    assert url is None
    assert expected == "0.6.15.post1+cu130"
    assert installed == "0.6.15.post1+cu130"


def test_install_url_if_needed_reinstalls_missing_or_stale_version(tmp_path: Path):
    requirements = tmp_path / "cuda.txt"
    requirements.write_text("flashinfer-python==0.6.15.post1\n")

    missing_url, _, missing_installed = install_url_if_needed(
        requirements,
        "130",
        installed_version=None,
    )
    stale_url, _, stale_installed = install_url_if_needed(
        requirements,
        "130",
        installed_version="0.6.11.post3+cu130",
    )

    expected_url = jit_cache_wheel_url("0.6.15.post1", "130")
    assert missing_url == expected_url
    assert missing_installed is None
    assert stale_url == expected_url
    assert stale_installed == "0.6.11.post3+cu130"
