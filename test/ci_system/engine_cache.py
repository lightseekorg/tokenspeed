# Copyright (c) 2026 LightSeek Foundation
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

"""Install the checkout's engine, reusing a wheel only for identical build inputs."""

import argparse
import fcntl
import hashlib
import importlib.metadata
import json
import os
import platform
import re
import shlex
import subprocess
import sys
import sysconfig
import tempfile
import time
import tomllib
from pathlib import Path


def build_requirements(workspace: Path) -> list[str]:
    with (workspace / "tokenspeed-engine/pyproject.toml").open("rb") as stream:
        return tomllib.load(stream)["build-system"]["requires"]


def source_digest(workspace: Path) -> str:
    """Hash real sources, including the scheduler behind the engine's symlink.

    Slurm uses git archives, so this must also work without a .git directory.
    Generated build/install files must not invalidate the next checkout's hit.
    """
    digest = hashlib.sha256()
    for name in ("tokenspeed-engine", "tokenspeed-scheduler", ".cargo"):
        for directory, dirs, files in os.walk(workspace / name):
            dirs[:] = sorted(
                entry
                for entry in dirs
                if entry not in {"target", "build", "dist", "__pycache__"}
                and not entry.endswith(".egg-info")
                and Path(directory, entry).relative_to(workspace).as_posix()
                != "tokenspeed-engine/python/tokenspeed_engine/bin"
            )
            for entry in sorted(
                files + [d for d in dirs if Path(directory, d).is_symlink()]
            ):
                path = Path(directory, entry)
                if path.suffix in {".pyc", ".so", ".o"}:
                    continue
                content = (
                    os.readlink(path).encode()
                    if path.is_symlink()
                    else path.read_bytes()
                )
                digest.update(path.relative_to(workspace).as_posix().encode() + b"\0")
                digest.update(hashlib.sha256(content).digest())
    # Changes to cache/install semantics invalidate already published wheels.
    for name in ("engine_cache.py", "install_tokenspeed_engine.sh"):
        digest.update(Path(__file__).with_name(name).read_bytes())
    return digest.hexdigest()


def environment_digest(requirements: list[str]) -> str:
    def output(command: list[str]) -> str:
        return subprocess.check_output(command, text=True).strip()

    # Native wheels link the system's libpython, OpenSSL and C++ runtime. The
    # installed native package versions distinguish CI image updates as well
    # as architectures. These CI images are Debian/Ubuntu based.
    tools = [
        [*shlex.split(os.environ.get("RUSTC", "rustc")), "-vV"],
        [*shlex.split(os.environ.get("CC", "cc")), "--version"],
        [*shlex.split(os.environ.get("CXX", "c++")), "--version"],
        ["dpkg-query", "-W", "-f=${Package}=${Version}\n"],
    ]
    packages = [re.split(r"[<>=!~\[;\s]", req, maxsplit=1)[0] for req in requirements]
    environment = {
        "platform": [platform.system(), platform.machine(), platform.libc_ver()],
        "python": [sys.version, sys.base_prefix, sysconfig.get_config_vars()],
        "packages": {name: importlib.metadata.version(name) for name in packages},
        "tools": [output(command) for command in tools],
        "env": {
            key: value
            for key, value in os.environ.items()
            if key.startswith(
                ("CARGO", "RUST", "CMAKE", "PYO3", "OPENSSL", "PKG_CONFIG")
            )
            or key
            in {
                "CC",
                "CXX",
                "CFLAGS",
                "CXXFLAGS",
                "CPPFLAGS",
                "LDFLAGS",
                "CPATH",
                "C_INCLUDE_PATH",
                "CPLUS_INCLUDE_PATH",
                "LIBRARY_PATH",
                "LD_LIBRARY_PATH",
            }
        },
    }
    return hashlib.sha256(json.dumps(environment, sort_keys=True).encode()).hexdigest()


def wheel_cache_root() -> Path | None:
    # Fork code must not populate a cache subsequently consumed by trusted jobs.
    # Local source overrides are outside the checkout's content hash.
    root = os.environ.get("CI_WHEEL_CACHE_DIR")
    if (
        not root
        or os.environ.get("TOKENSPEED_CI_FORK_PR") == "true"
        or os.environ.get("FETCHCONTENT_SOURCE_DIR_XGRAMMAR")
    ):
        return None
    return Path(root) / "tokenspeed-engine"


def pip(*args: str) -> None:
    subprocess.run([sys.executable, "-m", "pip", *args], check=True)


def install(source: Path) -> None:
    pip(
        "install", "--force-reinstall", "--no-deps", "--no-build-isolation", str(source)
    )
    subprocess.run(
        [
            sys.executable,
            "-c",
            "import os, tokenspeed_engine; "
            "assert os.access(tokenspeed_engine.host_binary(), os.X_OK)",
        ],
        check=True,
    )


def install_engine(workspace: Path, requirements: list[str]) -> None:
    # Resolve the declared build dependencies first, then key their actual
    # versions. PEP 517 isolation would hide a different set from this key.
    pip("install", *requirements)
    start = time.monotonic()
    root = wheel_cache_root()
    if root is None:
        print("Engine wheel cache disabled; building checkout", flush=True)
        install(workspace / "tokenspeed-engine")
    else:
        key = f"{environment_digest(requirements)}-{source_digest(workspace)}"
        root.mkdir(parents=True, exist_ok=True)
        destination = root / key
        with (root / f"{key}.lock").open("a") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            if destination.is_dir():
                print(f"Engine wheel cache hit: {key}", flush=True)
                (wheel,) = destination.glob("*.whl")
                install(wheel)
            else:
                print(f"Engine wheel cache miss: {key}", flush=True)
                with tempfile.TemporaryDirectory(
                    dir=root, prefix="build-"
                ) as temporary:
                    directory = Path(temporary)
                    pip(
                        "wheel",
                        "--no-deps",
                        "--no-build-isolation",
                        "--wheel-dir",
                        str(directory),
                        str(workspace / "tokenspeed-engine"),
                    )
                    (wheel,) = directory.glob("*.whl")
                    install(wheel)
                    # Publish only a complete wheel that passed the import check.
                    directory.rename(destination)
    print(f"Engine build/install elapsed: {time.monotonic() - start:.1f}s", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--workspace", type=Path, default=Path(__file__).resolve().parents[2]
    )
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--build-requirements", action="store_true")
    mode.add_argument("--environment-key", action="store_true")
    args = parser.parse_args()
    requirements = build_requirements(args.workspace)
    if args.build_requirements:
        print("\n".join(requirements))
    elif args.environment_key:
        print(environment_digest(requirements))
    else:
        install_engine(args.workspace.resolve(), requirements)


if __name__ == "__main__":
    main()
