import os
import re
import subprocess
import sys
from pathlib import Path

import pytest


CHECKER = Path(__file__).with_name("verify_rocm_stack.py")


def write_distribution(root: Path, name: str, version: str) -> None:
    normalized = re.sub(r"[-_.]+", "_", name)
    dist_info = root / f"{normalized}-{version}.dist-info"
    dist_info.mkdir()
    (dist_info / "METADATA").write_text(
        f"Metadata-Version: 2.1\nName: {name}\nVersion: {version}\n"
    )


def run_checker(root: Path, *args: str) -> subprocess.CompletedProcess[str]:
    env = {**os.environ, "PYTHONPATH": str(root)}
    return subprocess.run(
        [sys.executable, "-S", str(CHECKER), *args],
        env=env,
        capture_output=True,
        text=True,
    )


def populate_mi450_stack(root: Path, rocm_version: str) -> None:
    versions = {
        "uv": "0.9.26",
        "pytest": "9.2.0",
        "pytest-reportlog": "1.0.0",
        "pytest-timeout": "2.4.0",
        "pytest-xdist": "3.8.0",
        "rocm": rocm_version,
        "rocm-sdk-core": rocm_version,
        "rocm-sdk-devel": rocm_version,
        "rocm-sdk-libraries": rocm_version,
        "rocm-sdk-device-gfx1250": rocm_version,
    }
    for name, version in versions.items():
        write_distribution(root, name, version)


@pytest.mark.parametrize("rocm_version", ["10.3.0a20261107", "10.3.0a20261108"])
def test_mi450_sdk_requires_the_exact_rocm_nightly(tmp_path, rocm_version):
    populate_mi450_stack(tmp_path, rocm_version)

    result = run_checker(tmp_path, "mi450-sdk", "0.9.26", "10.3.0a20261107")

    assert (result.returncode == 0) == (rocm_version == "10.3.0a20261107")
    if result.returncode:
        assert "expected 10.3.0a20261107" in result.stderr


def populate_torch_stack(
    root: Path, torch_version: str, torchvision_version: str
) -> None:
    (root / "torch.py").write_text(f'__version__ = "{torch_version}"\n')
    (root / "torchvision.py").write_text(f'__version__ = "{torchvision_version}"\n')
    write_distribution(root, "amd-torch-device-gfx1250", torch_version)


def test_torch_accepts_a_local_rocm_version_for_a_public_version_pin(tmp_path):
    populate_torch_stack(
        tmp_path,
        "2.14.0+rocm10.3.0a20261107",
        "0.29.0+rocm10.3.0a20261107",
    )

    result = run_checker(tmp_path, "torch", "2.14.0", "0.29.0", "")

    assert result.returncode == 0, result.stderr


def test_torch_requires_an_exact_local_version_and_device_package(tmp_path):
    populate_torch_stack(
        tmp_path,
        "2.14.0+rocm10.3.0a20261107",
        "0.29.0+rocm10.3.0a20261107",
    )

    result = run_checker(
        tmp_path,
        "torch",
        "2.14.0+rocm10.3.0a20261108",
        "0.29.0+rocm10.3.0a20261108",
        "amd-torch-device-gfx1250==2.14.0+rocm10.3.0a20261108",
    )

    assert result.returncode == 1
    assert "torch: expected 2.14.0+rocm10.3.0a20261108" in result.stderr
    assert "amd-torch-device-gfx1250" in result.stderr
