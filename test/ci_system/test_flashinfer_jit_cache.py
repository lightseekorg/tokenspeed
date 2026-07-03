from pathlib import Path

import pytest
from flashinfer_jit_cache_installer import (
    expected_jit_cache_version,
    install_url_if_needed,
    jit_cache_wheel_url,
    read_exact_pin,
    should_sync_jit_cache,
)


def test_read_exact_pin_ignores_other_requirements(tmp_path: Path):
    requirements = tmp_path / "cuda.txt"
    requirements.write_text(
        "-r common.txt\n"
        "torch==2.11.0\n"
        "flashinfer-python==0.6.13\n"
        "flashinfer-cubin==0.6.13\n"
    )

    assert read_exact_pin(requirements, "flashinfer-python") == "0.6.13"


def test_read_exact_pin_requires_exact_pin(tmp_path: Path):
    requirements = tmp_path / "cuda.txt"
    requirements.write_text("flashinfer-python>=0.6\n")

    with pytest.raises(ValueError, match="flashinfer-python exact pin not found"):
        read_exact_pin(requirements, "flashinfer-python")


def test_jit_cache_url_tracks_flashinfer_and_cuda_versions():
    assert expected_jit_cache_version("0.6.13", "130") == "0.6.13+cu130"
    assert jit_cache_wheel_url("0.6.13", "130") == (
        "https://github.com/flashinfer-ai/flashinfer/releases/download/"
        "v0.6.13/flashinfer_jit_cache-0.6.13+cu130-cp39-abi3-"
        "manylinux_2_28_aarch64.whl"
    )


@pytest.mark.parametrize(
    "runner_label,expected",
    [
        ("b200-4gpu", True),
        ("b200v2-4gpu", True),
        ("gb200-4gpu", True),
        ("gb200-4gpu-perf", True),
        ("b300-4gpu", False),
        ("h100-1gpu", False),
        ("", False),
        (None, False),
    ],
)
def test_should_sync_jit_cache_only_on_b200_and_gb200(runner_label, expected):
    assert should_sync_jit_cache(runner_label) is expected


def test_install_url_if_needed_skips_matching_version(tmp_path: Path):
    requirements = tmp_path / "cuda.txt"
    requirements.write_text("flashinfer-python==0.6.13\n")

    url, expected, installed = install_url_if_needed(
        requirements,
        "130",
        installed_version="0.6.13+cu130",
    )

    assert url is None
    assert expected == "0.6.13+cu130"
    assert installed == "0.6.13+cu130"


def test_install_url_if_needed_reinstalls_missing_or_stale_version(tmp_path: Path):
    requirements = tmp_path / "cuda.txt"
    requirements.write_text("flashinfer-python==0.6.13\n")

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

    expected_url = jit_cache_wheel_url("0.6.13", "130")
    assert missing_url == expected_url
    assert missing_installed is None
    assert stale_url == expected_url
    assert stale_installed == "0.6.11.post3+cu130"
