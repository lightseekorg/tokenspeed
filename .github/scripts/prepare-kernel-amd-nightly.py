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

import argparse
import re
import tomllib
from datetime import datetime
from pathlib import Path


def prepare_nightly(root: Path, date: str, target: str) -> str:
    if not re.fullmatch(r"[0-9]{8}", date):
        raise ValueError("Nightly date must be YYYYMMDD")
    datetime.strptime(date, "%Y%m%d")
    pyproject = root / "tokenspeed-kernel-amd/pyproject.toml"
    original = pyproject.read_text()
    project = tomllib.loads(original)["project"]
    if project["name"] != "tokenspeed-kernel-amd":
        raise ValueError("Expected tokenspeed-kernel-amd project")
    base = project["version"]
    if not re.fullmatch(r"[0-9]+\.[0-9]+\.[0-9]+", base):
        raise ValueError("Nightly requires a stable AMD package version on main")
    version = f"{base}.post{date}"

    if target == "amd":
        updated, count = re.subn(
            r'(?m)^version = "[^"\n]+"$', f'version = "{version}"', original
        )
        if count != 1:
            raise ValueError("Expected one AMD package version")
        pyproject.write_text(updated)
    elif target == "rocm":
        requirements = (
            root / "tokenspeed-kernel/python/requirements/rocm-thirdparty.txt"
        )
        original = requirements.read_text()
        updated, count = re.subn(
            r"(?m)^tokenspeed-kernel-amd>=[^\s]+$",
            f"tokenspeed-kernel-amd=={version}",
            original,
        )
        if count != 1:
            raise ValueError("Expected one ROCm AMD dependency floor")
        requirements.write_text(updated)
    elif target != "version":
        raise ValueError(f"Unsupported target: {target}")
    return version


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--target", choices=("amd", "rocm", "version"), required=True)
    parser.add_argument("root", type=Path)
    parser.add_argument("date")
    args = parser.parse_args()
    print(prepare_nightly(args.root, args.date, args.target))
