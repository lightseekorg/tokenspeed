from pathlib import Path

import pytest
from flashinfer_jit_cache_installer import (
    expected_jit_cache_version,
    install_url_if_needed,
    jit_cache_wheel_url,
    main,
    platform_tag_for_runner,
    read_exact_pin,
)


@pytest.mark.parametrize(
    ("runner", "platform_tag"),
    [
        ("gb200-1gpu", "manylinux_2_28_aarch64"),
        ("b200-8gpu", "manylinux_2_28_x86_64"),
    ],
)
def test_platform_tag_for_sm100_runners(runner: str, platform_tag: str):
    assert platform_tag_for_runner(runner) == platform_tag


@pytest.mark.parametrize(
    "runner", ["h100-1gpu", "b300-1gpu", "gb300-1gpu", "amd-mi350-1gpu"]
)
def test_platform_tag_for_other_runners_is_disabled(runner: str):
    assert platform_tag_for_runner(runner) is None


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


def test_install_url_if_needed_uses_x86_64_tag_for_b200(tmp_path: Path):
    requirements = tmp_path / "cuda.txt"
    requirements.write_text("flashinfer-python==0.6.13\n")

    url, _, _ = install_url_if_needed(
        requirements,
        "130",
        platform_tag="manylinux_2_28_x86_64",
        installed_version=None,
    )

    assert url == jit_cache_wheel_url(
        "0.6.13", "130", platform_tag="manylinux_2_28_x86_64"
    )
    assert url.endswith("manylinux_2_28_x86_64.whl")


def test_main_prints_only_url_to_stdout_for_enabled_runner(tmp_path: Path, capsys):
    requirements = tmp_path / "cuda.txt"
    requirements.write_text("flashinfer-python==0.6.13\n")

    rc = main(
        [
            "--requirements",
            str(requirements),
            "--cuda-index",
            "130",
            "--runner",
            "b200-8gpu",
        ]
    )

    out = capsys.readouterr().out
    assert rc == 0
    assert out.strip() == jit_cache_wheel_url(
        "0.6.13", "130", platform_tag="manylinux_2_28_x86_64"
    )
    assert "\n" not in out.strip()


def test_main_skips_disabled_runner_before_reading_requirements(tmp_path: Path, capsys):
    requirements = tmp_path / "cuda.txt"
    requirements.write_text("flashinfer-python>=0.6\n")

    rc = main(
        [
            "--requirements",
            str(requirements),
            "--cuda-index",
            "130",
            "--runner",
            "h100-1gpu",
        ]
    )

    assert rc == 0
    assert capsys.readouterr().out == ""


def test_install_url_if_needed_skips_matching_version(tmp_path: Path):
    requirements = tmp_path / "cuda.txt"
    requirements.write_text("flashinfer-python==0.6.13\n")

    url, expected, installed = install_url_if_needed(
        requirements,
        "130",
        platform_tag="manylinux_2_28_aarch64",
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
        platform_tag="manylinux_2_28_aarch64",
        installed_version=None,
    )
    stale_url, _, stale_installed = install_url_if_needed(
        requirements,
        "130",
        platform_tag="manylinux_2_28_aarch64",
        installed_version="0.6.11.post3+cu130",
    )

    expected_url = jit_cache_wheel_url("0.6.13", "130")
    assert missing_url == expected_url
    assert missing_installed is None
    assert stale_url == expected_url
    assert stale_installed == "0.6.11.post3+cu130"
