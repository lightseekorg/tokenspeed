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

"""Prepare TokenSpeed nightly metadata in the workflow checkout."""

import argparse
import re
import tomllib
from datetime import datetime
from pathlib import Path

from packaging.version import Version


def prepare_nightly(root: Path, date: str, kernel_version: str) -> str:
    if not re.fullmatch(r"[0-9]{8}", date):
        raise ValueError("Nightly date must be YYYYMMDD")
    datetime.strptime(date, "%Y%m%d")
    kernel_version = str(Version(kernel_version))
    if Version(kernel_version).post != int(date):
        raise ValueError("Kernel nightly must have the same date as TokenSpeed")

    pyproject = root / "python/pyproject.toml"
    original = pyproject.read_text()
    project = tomllib.loads(original)["project"]
    if project["name"] != "tokenspeed":
        raise ValueError("Expected the tokenspeed project")
    version = f"{Version(project['version']).base_version}.post{date}"
    updated, versions = re.subn(
        r'^version = "[^"\n]+"$', f'version = "{version}"', original, flags=re.MULTILINE
    )
    updated, dependencies = re.subn(
        r'"tokenspeed-kernel[^"\n]*"', f'"tokenspeed-kernel=={kernel_version}"', updated
    )
    version_path = root / "python/tokenspeed/version.py"
    version_source, runtime_versions = re.subn(
        r'^__version__ = "[^"\n]+"$',
        f'__version__ = "{version}"',
        version_path.read_text(),
        flags=re.MULTILINE,
    )
    if (versions, dependencies, runtime_versions) != (1, 1, 1):
        raise ValueError(
            "Expected one project version, kernel dependency and runtime version"
        )
    pyproject.write_text(updated)
    version_path.write_text(version_source)
    return version


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    parser.add_argument("date")
    parser.add_argument("kernel_version")
    args = parser.parse_args()
    print(prepare_nightly(args.root, args.date, args.kernel_version))
