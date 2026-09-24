import json
import os
import subprocess
from pathlib import Path

import mi450_sim_environment as environment
import pytest


@pytest.fixture
def installed_simulator(tmp_path, monkeypatch):
    root = tmp_path / "sim"
    root.mkdir()
    manifest = {
        "schema_version": 1,
        "rocm_systems_ref": "sim-ref",
        "rocm_sdk_version": "sdk-version",
        "uv_version": "uv-version",
    }
    (root / "preinstalled.json").write_text(json.dumps(manifest))
    launcher = root / "rocjitsu-build/tools/rocjitsu/rocjitsu"
    launcher.parent.mkdir(parents=True)
    launcher.touch(mode=0o755)
    (root / "rocjitsu-build/librocjitsu.so").touch()
    config = root / "rocm-systems/emulation/rocjitsu/configs/gfx1250_mi455x.json"
    config.parent.mkdir(parents=True)
    config.write_text('{"max_ticks": 0}')
    sdk = tmp_path / "sdk"
    (sdk / "lib").mkdir(parents=True)
    (sdk / "lib/libamdhip64.so").touch()
    env = {
        "TORCH_VERSION": "torch-version",
        "TORCHVISION_VERSION": "vision-version",
        "TORCH_DEVICE_PACKAGE": "amd-torch-device-gfx1250==device-version",
    }
    versions = {
        "rocm": "sdk-version",
        "rocm-sdk-core": "sdk-version",
        "rocm-sdk-devel": "sdk-version",
        "rocm-sdk-libraries": "sdk-version",
        "rocm-sdk-device-gfx1250": "sdk-version",
        "torch": "torch-version",
        "torchvision": "vision-version",
        "amd-torch-device-gfx1250": "device-version",
        "uv": "uv-version",
        "pyyaml": "1",
        "pytest-timeout": "1",
        "pytest-xdist": "1",
        "pytest-reportlog": "1",
    }
    monkeypatch.setattr(environment.importlib.metadata, "version", versions.__getitem__)
    monkeypatch.setattr(environment.shutil, "which", lambda command, path: command)
    monkeypatch.setattr(
        environment.subprocess,
        "check_output",
        lambda command, **kwargs: "sim-ref\n" if command[0] == "git" else f"{sdk}\n",
    )
    return root, env, versions


def validate(root, env):
    environment.validate_environment(root, "sim-ref", "sdk-version", "uv-version", env)


def test_matching_image_is_reused(installed_simulator):
    root, env, _ = installed_simulator
    validate(root, env)


@pytest.mark.parametrize(
    "field", ["schema_version", "rocm_systems_ref", "rocm_sdk_version", "uv_version"]
)
def test_manifest_mismatch_uses_fallback(installed_simulator, field):
    root, env, _ = installed_simulator
    path = root / "preinstalled.json"
    manifest = json.loads(path.read_text())
    manifest[field] = "different"
    path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="manifest"):
        validate(root, env)


@pytest.mark.parametrize(
    "package",
    [
        "rocm",
        "rocm-sdk-core",
        "rocm-sdk-devel",
        "rocm-sdk-libraries",
        "rocm-sdk-device-gfx1250",
        "torch",
        "torchvision",
        "amd-torch-device-gfx1250",
        "uv",
    ],
)
def test_installed_package_mismatch_uses_fallback(installed_simulator, package):
    root, env, versions = installed_simulator
    versions[package] = "different"
    with pytest.raises(ValueError, match=package):
        validate(root, env)


@pytest.mark.parametrize(
    "package", ["", "amd-torch-device-gfx950==1", "amd-torch-device-gfx1250>=1"]
)
def test_device_package_must_match(installed_simulator, package):
    root, env, _ = installed_simulator
    env["TORCH_DEVICE_PACKAGE"] = package
    with pytest.raises(ValueError, match="TORCH_DEVICE_PACKAGE"):
        validate(root, env)


@pytest.mark.parametrize(
    "missing",
    [
        "preinstalled.json",
        "rocjitsu-build/librocjitsu.so",
        "rocjitsu-build/tools/rocjitsu/rocjitsu",
    ],
)
def test_missing_image_file_uses_fallback(installed_simulator, missing):
    root, env, _ = installed_simulator
    (root / missing).unlink()
    with pytest.raises((OSError, ValueError)):
        validate(root, env)


def test_config_patch_is_required(installed_simulator):
    root, env, _ = installed_simulator
    config = root / "rocm-systems/emulation/rocjitsu/configs/gfx1250_mi455x.json"
    config.write_text('{"max_ticks": 100}')
    with pytest.raises(ValueError, match="max_ticks"):
        validate(root, env)


def test_sdk_must_be_initialized(installed_simulator):
    root, env, _ = installed_simulator
    (root.parent / "sdk/lib/libamdhip64.so").unlink()
    with pytest.raises(ValueError, match="initialized"):
        validate(root, env)


def test_source_mismatch_uses_fallback(installed_simulator, monkeypatch):
    root, env, _ = installed_simulator
    monkeypatch.setattr(
        environment.subprocess, "check_output", lambda *a, **kw: "different"
    )
    with pytest.raises(ValueError, match="checkout"):
        validate(root, env)


def test_preinstalled_installer_still_installs_checkout(tmp_path):
    workspace = tmp_path / "checkout"
    scripts = workspace / "test/ci_system"
    scripts.mkdir(parents=True)
    (scripts / "setup_mi450_sim.sh").write_text("#!/bin/bash\nexit 0\n")
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    log = tmp_path / "commands"
    for command in ("pip", "pip3", "python3", "sudo"):
        stub = bin_dir / command
        stub.write_text(
            f'#!/bin/bash\nprintf "%s\\n" "{command} $*" >> "$COMMAND_LOG"\n'
            + ("exit 99\n" if command == "sudo" else "exit 0\n")
        )
        stub.chmod(0o755)
    result = subprocess.run(
        ["bash", str(Path(__file__).with_name("install_deps_rocm.sh"))],
        env=os.environ
        | {
            "PATH": f"{bin_dir}:{os.environ['PATH']}",
            "WORKSPACE": str(workspace),
            "GFX_ARCH": "gfx1250",
            "COMMAND_LOG": str(log),
        },
        capture_output=True,
        text=True,
        timeout=10,
    )
    assert result.returncode == 0, result.stderr
    commands = log.read_text()
    assert "sudo" not in commands
    assert "torch==" not in commands
    assert "--force-reinstall --no-deps" in commands
    assert "tokenspeed-kernel-amd" in commands
    assert "tokenspeed-kernel/python/" in commands
    assert "tokenspeed-scheduler/" in commands
    assert "install -e ./python --no-build-isolation" in commands


def test_check_on_old_image_is_read_only(tmp_path):
    # A missing preinstalled root must return immediately, without apt/pip/git.
    result = subprocess.run(
        ["bash", str(Path(__file__).with_name("setup_mi450_sim.sh")), "--check"],
        env=os.environ | {"TOKENSPEED_MI450_SIM_ROOT": str(tmp_path / "missing")},
        capture_output=True,
        text=True,
        timeout=10,
    )
    assert result.returncode == 1
    assert "unavailable" in result.stdout
    assert not (tmp_path / "missing").exists()
