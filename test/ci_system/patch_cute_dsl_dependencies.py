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

"""Migrate pinned CI wheels to CuTe DSL's supported APIs before importing them."""

import hashlib
import importlib.metadata
import json
import shutil
import subprocess
from pathlib import Path

PATCH_DIR = Path(__file__).with_name("cute_dsl_patches")
SCHEDULER = (
    Path(__file__).resolve().parents[2]
    / "tokenspeed-kernel/python/tokenspeed_kernel/thirdparty/cute_dsl/static_persistent_tile_scheduler.py"
)


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    manifest = json.loads((PATCH_DIR / "manifest.json").read_text())
    version = importlib.metadata.version("nvidia-cutlass-dsl")
    if version != manifest["cutlass_version"]:
        raise RuntimeError(f"Review CuTe DSL patches before using {version}")

    # Validate every package before changing any installed file.
    pending = []
    for package in manifest["packages"]:
        distribution = importlib.metadata.distribution(package["name"])
        if distribution.version != package["version"]:
            raise RuntimeError(
                f"Review patches for {package['name']}=={distribution.version}"
            )
        root = Path(distribution.locate_file(""))
        states = set()
        for file in package["files"]:
            actual = digest(root / file["path"])
            if actual == file["before"]:
                states.add("before")
            elif actual == file["after"]:
                states.add("after")
            else:
                raise RuntimeError(f"Unexpected source for {file['path']}")
        if len(states) != 1:
            raise RuntimeError(f"Partially patched {package['name']}")
        command = [
            "patch",
            "-p0",
            "--fuzz=0",
            "--forward",
            "--batch",
            "--input",
            str(PATCH_DIR / package["patch"]),
        ]
        if states == {"before"}:
            subprocess.run(
                [*command, "--dry-run"], cwd=root, check=True, capture_output=True
            )
        pending.append((package, root, command, states))

    for package, root, command, states in pending:
        # Each wheel gets its own helper, avoiding a runtime dependency cycle.
        shutil.copyfile(
            SCHEDULER,
            root / package["root"] / "_tokenspeed_static_persistent_tile_scheduler.py",
        )
        if states == {"before"}:
            subprocess.run(command, cwd=root, check=True, capture_output=True)
        for file in package["files"]:
            if digest(root / file["path"]) != file["after"]:
                raise RuntimeError(f"Failed to migrate {file['path']}")
        print(f"CuTe DSL APIs verified: {package['name']}=={package['version']}")


if __name__ == "__main__":
    main()
