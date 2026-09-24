"""Read-only validation of the optional preinstalled MI450 simulator toolchain."""

import argparse
import importlib.metadata
import json
import os
import shutil
import subprocess
from pathlib import Path
from typing import Mapping


def validate_environment(
    root: Path,
    rocm_systems_ref: str,
    rocm_sdk_version: str,
    uv_version: str,
    env: Mapping[str, str],
) -> None:
    """Raise on missing/stale image contents; never install or initialize anything."""
    manifest = json.loads((root / "preinstalled.json").read_text())
    expected = {
        "schema_version": 1,
        "rocm_systems_ref": rocm_systems_ref,
        "rocm_sdk_version": rocm_sdk_version,
        "uv_version": uv_version,
    }
    if manifest != expected:
        raise ValueError(f"simulator image manifest does not match {expected}")

    device_package, separator, device_version = env.get(
        "TORCH_DEVICE_PACKAGE", ""
    ).partition("==")
    if (
        device_package != "amd-torch-device-gfx1250"
        or not separator
        or not device_version
    ):
        raise ValueError("TORCH_DEVICE_PACKAGE must pin amd-torch-device-gfx1250")
    packages = {
        "rocm": rocm_sdk_version,
        "rocm-sdk-core": rocm_sdk_version,
        "rocm-sdk-devel": rocm_sdk_version,
        "rocm-sdk-libraries": rocm_sdk_version,
        "rocm-sdk-device-gfx1250": rocm_sdk_version,
        "torch": env["TORCH_VERSION"],
        "torchvision": env["TORCHVISION_VERSION"],
        device_package: device_version,
        "uv": uv_version,
    }
    for package, version in packages.items():
        installed = importlib.metadata.version(package)
        if installed != version:
            raise ValueError(f"{package}: installed {installed}, requested {version}")
    for package in ("pyyaml", "pytest-timeout", "pytest-xdist", "pytest-reportlog"):
        importlib.metadata.version(package)
    for command in ("clang", "cmake", "ninja", "mpicc", "pkg-config", "rocm-sdk"):
        if shutil.which(command, path=env.get("PATH")) is None:
            raise ValueError(f"missing image dependency: {command}")

    source = root / "rocm-systems"
    actual_ref = subprocess.check_output(
        ["git", "-C", str(source), "rev-parse", "HEAD"],
        env=env,
        text=True,
        stderr=subprocess.DEVNULL,
        timeout=10,
    ).strip()
    if actual_ref != rocm_systems_ref:
        raise ValueError(
            f"rocJITsu checkout is {actual_ref}, expected {rocm_systems_ref}"
        )
    launcher = root / "rocjitsu-build/tools/rocjitsu/rocjitsu"
    if not launcher.is_file() or not os.access(launcher, os.X_OK):
        raise ValueError(f"missing executable: {launcher}")
    if not (root / "rocjitsu-build/librocjitsu.so").is_file():
        raise ValueError("missing rocJITsu runtime")
    config = json.loads(
        (source / "emulation/rocjitsu/configs/gfx1250_mi455x.json").read_text()
    )
    if config.get("max_ticks") != 0:
        raise ValueError("MI455X config must have max_ticks=0")
    sdk_root = Path(
        subprocess.check_output(
            ["rocm-sdk", "path", "--root"], env=env, text=True, timeout=10
        ).strip()
    )
    if not (sdk_root / "lib/libamdhip64.so").is_file():
        raise ValueError("ROCm SDK has not been initialized")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    parser.add_argument("rocm_systems_ref")
    parser.add_argument("rocm_sdk_version")
    parser.add_argument("uv_version")
    args = parser.parse_args()
    try:
        validate_environment(
            args.root,
            args.rocm_systems_ref,
            args.rocm_sdk_version,
            args.uv_version,
            os.environ,
        )
    except (
        OSError,
        ValueError,
        KeyError,
        importlib.metadata.PackageNotFoundError,
        subprocess.SubprocessError,
    ) as exc:
        print(f"MI450 preinstalled environment unavailable: {exc}")
        return 1
    print(f"Using validated preinstalled MI450 simulator: {args.root}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
