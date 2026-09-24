"""Check ROCm install scopes without downloading or installing packages."""

import os
import subprocess
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
INSTALL = REPO_ROOT / "test/ci_system/install_deps_rocm.sh"
DISPATCH = REPO_ROOT / "test/ci_system/install_deps.sh"


def _fake_installer(tmp_path: Path) -> tuple[dict[str, str], Path]:
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    calls = tmp_path / "calls"
    for name in ("sudo", "pip", "pip3", "python3"):
        command = bin_dir / name
        command.write_text(
            '#!/bin/bash\nprintf "%s %s\\n" "${0##*/}" "$*" >> "$FAKE_INSTALL_CALLS"\n'
        )
        command.chmod(0o755)
    env = {
        **os.environ,
        "PATH": f"{bin_dir}:{os.environ['PATH']}",
        "FAKE_INSTALL_CALLS": str(calls),
        "WORKSPACE": str(REPO_ROOT),
    }
    return env, calls


@pytest.mark.parametrize("scope", ["kernel", "full"])
def test_rocm_install_scope(tmp_path: Path, scope: str) -> None:
    env, calls = _fake_installer(tmp_path)
    result = subprocess.run(
        ["bash", str(INSTALL), scope],
        cwd=REPO_ROOT,
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stdout + result.stderr
    commands = calls.read_text()
    assert "pip3 install tokenspeed-kernel/python/" in commands
    assert ("pip3 install -e ./python" in commands) == (scope == "full")
    assert ("tokenspeed-scheduler/" in commands) == (scope == "full")


@pytest.mark.parametrize("args", [[], ["invalid"], ["kernel", "full"]])
def test_rocm_install_rejects_missing_or_invalid_scope(
    tmp_path: Path, args: list[str]
) -> None:
    env, calls = _fake_installer(tmp_path)
    result = subprocess.run(
        ["bash", str(INSTALL), *args],
        cwd=REPO_ROOT,
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 2
    assert not calls.exists()


def test_amd_dispatch_preserves_explicit_scope(tmp_path: Path) -> None:
    env, calls = _fake_installer(tmp_path)
    env["CI_RUNNER_LABEL"] = "amd-mi35x-1gpu-test"
    result = subprocess.run(
        ["bash", str(DISPATCH), "full"],
        cwd=REPO_ROOT,
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stdout + result.stderr
    assert "pip3 install -e ./python" in calls.read_text()


@pytest.mark.parametrize(
    ("args", "expected_code"),
    [(["kernel"], 2), (["invalid"], 2), (["full", "kernel"], 2), (["full"], 0)],
)
def test_cu129_dispatch_validates_scope_before_install(
    tmp_path: Path, args: list[str], expected_code: int
) -> None:
    env, calls = _fake_installer(tmp_path)
    env["CUDA_VARIANT"] = "cu129"
    result = subprocess.run(
        ["bash", str(DISPATCH), *args],
        cwd=REPO_ROOT,
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == expected_code, result.stdout + result.stderr
    assert calls.exists() == (expected_code == 0)
