from pathlib import Path

import flashinfer_jit_cache_installer as installer
import pytest
from flashinfer_jit_cache_installer import (
    expected_jit_cache_version,
    install_url_if_needed,
    jit_cache_wheel_url,
    read_exact_pin,
)


def test_read_exact_pin_ignores_other_requirements(tmp_path: Path):
    requirements = tmp_path / "cuda.txt"
    requirements.write_text("torch==2.11.0\n" "flashinfer-python==0.6.18\n")

    assert read_exact_pin(requirements, "flashinfer-python") == "0.6.18"


def test_read_exact_pin_requires_exact_pin(tmp_path: Path):
    requirements = tmp_path / "cuda.txt"
    requirements.write_text("flashinfer-python>=0.6\n")

    with pytest.raises(ValueError, match="flashinfer-python exact pin not found"):
        read_exact_pin(requirements, "flashinfer-python")


def test_jit_cache_url_tracks_flashinfer_and_cuda_versions():
    assert expected_jit_cache_version("0.6.18", "130") == "0.6.18+cu130"
    assert jit_cache_wheel_url(
        "0.6.18", "130", platform_tag="manylinux_2_28_aarch64"
    ) == (
        "https://github.com/flashinfer-ai/flashinfer/releases/download/"
        "v0.6.18/flashinfer_jit_cache-0.6.18+cu130-cp39-abi3-"
        "manylinux_2_28_aarch64.whl"
    )


def test_install_url_if_needed_skips_matching_version(tmp_path: Path, monkeypatch):
    monkeypatch.setattr(installer.metadata, "requires", lambda _: [])
    requirements = tmp_path / "cuda.txt"
    requirements.write_text("flashinfer-python==0.6.18\n")

    url, expected, installed = install_url_if_needed(
        requirements,
        "130",
        installed_version="0.6.18+cu130",
    )

    assert url is None
    assert expected == "0.6.18+cu130"
    assert installed == "0.6.18+cu130"


def test_install_url_if_needed_reinstalls_missing_or_stale_version(tmp_path: Path):
    requirements = tmp_path / "cuda.txt"
    requirements.write_text("flashinfer-python==0.6.18\n")

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

    expected_url = jit_cache_wheel_url("0.6.18", "130")
    assert missing_url == expected_url
    assert missing_installed is None
    assert stale_url == expected_url
    assert stale_installed == "0.6.11.post3+cu130"


@pytest.mark.parametrize(
    "provider_version", [None, "0.6.18+cu130", "0.7.0+cu129", "0.7.0+cu130"]
)
def test_matching_shim_requires_matching_providers(
    tmp_path, monkeypatch, provider_version
):
    requirements = tmp_path / "cuda.txt"
    requirements.write_text("flashinfer-python==0.7.0\n")
    monkeypatch.setattr(
        installer.metadata,
        "requires",
        lambda _: ["flashinfer-jit-cache-sm100a==0.7.0+cu130"],
    )
    monkeypatch.setattr(
        installer, "installed_distribution_version", lambda _: provider_version
    )

    url, expected, installed = install_url_if_needed(
        requirements, "130", installed_version="0.7.0+cu130"
    )

    assert expected == installed == "0.7.0+cu130"
    assert url == (
        None
        if provider_version == "0.7.0+cu130"
        else jit_cache_wheel_url("0.7.0", "130")
    )


def test_matching_shim_checks_every_required_provider(tmp_path, monkeypatch):
    requirements = tmp_path / "cuda.txt"
    requirements.write_text("flashinfer-python==0.7.0\n")
    monkeypatch.setattr(
        installer.metadata,
        "requires",
        lambda _: [
            "flashinfer-jit-cache-sm100a==0.7.0+cu130",
            "flashinfer-jit-cache-sm103a==0.7.0+cu130",
        ],
    )
    versions = {"flashinfer-jit-cache-sm100a": "0.7.0+cu130"}
    monkeypatch.setattr(installer, "installed_distribution_version", versions.get)

    url, _, _ = install_url_if_needed(
        requirements, "130", installed_version="0.7.0+cu130"
    )

    assert url == jit_cache_wheel_url("0.7.0", "130")


def test_jit_cache_ignores_inactive_dependency_markers(monkeypatch):
    monkeypatch.setattr(
        installer.metadata,
        "requires",
        lambda _: ['flashinfer-jit-cache-sm100a==0.7.0+cu130; python_version < "3.0"'],
    )
    monkeypatch.setattr(installer, "installed_distribution_version", lambda _: None)

    assert installer.jit_cache_dependencies_satisfied()
