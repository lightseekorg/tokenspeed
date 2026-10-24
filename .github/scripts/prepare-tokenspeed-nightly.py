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
import json
import re
import sys
import tomllib
import urllib.request
from datetime import datetime
from pathlib import Path

from packaging.version import InvalidVersion, Version

# One SMG nightly build publishes these together under one .postYYYYMMDD date.
SMG_PACKAGES = (
    "tokenspeed-smg",
    "tokenspeed-smg-grpc-proto",
    "tokenspeed-smg-grpc-servicer",
)


def nightly_date(version: Version) -> int | None:
    """The YYYYMMDD date of a dated nightly version, else None."""
    if (
        version.post is None
        or version.pre is not None
        or version.dev is not None
        or version.local is not None
    ):
        return None
    try:
        datetime.strptime(str(version.post), "%Y%m%d")
    except ValueError:
        return None
    return version.post


def newest_smg_nightly(releases: dict[str, list[str]], date: str) -> dict[str, str]:
    """Pick the newest SMG nightly published on or before ``date``.

    ``releases`` maps each SMG package to its published versions. A date counts
    only when every SMG package has a release with it, so the gateway, servicer
    and protocol all come from one SMG build.
    """
    by_date: dict[str, dict[int, Version]] = {}
    for package in SMG_PACKAGES:
        by_date[package] = {}
        for text in releases[package]:
            try:
                version = Version(text)
            except InvalidVersion:
                continue
            day = nightly_date(version)
            if day is None or day > int(date):
                continue
            if day not in by_date[package] or version > by_date[package][day]:
                by_date[package][day] = version
    common = set.intersection(*(set(days) for days in by_date.values()))
    if not common:
        raise ValueError(f"No SMG nightly of every SMG package on or before {date}")
    newest = max(common)
    return {package: str(by_date[package][newest]) for package in SMG_PACKAGES}


def pypi_releases(json_api: str, package: str) -> list[str]:
    """Versions of ``package`` with at least one non-yanked file on a PyPI JSON API."""
    with urllib.request.urlopen(f"{json_api}/{package}/json", timeout=60) as response:
        releases = json.load(response)["releases"]
    return [
        version
        for version, files in releases.items()
        if any(not file.get("yanked", False) for file in files)
    ]


def prepare_nightly(
    root: Path, date: str, kernel_version: str, smg_versions: dict[str, str]
) -> str:
    if not re.fullmatch(r"[0-9]{8}", date):
        raise ValueError("Nightly date must be YYYYMMDD")
    datetime.strptime(date, "%Y%m%d")
    kernel_version = str(Version(kernel_version))
    if Version(kernel_version).post != int(date):
        raise ValueError("Kernel nightly must have the same date as TokenSpeed")
    if set(smg_versions) != set(SMG_PACKAGES):
        raise ValueError(f"Expected versions for {', '.join(SMG_PACKAGES)}")
    smg_dates = {nightly_date(Version(v)) for v in smg_versions.values()}
    if len(smg_dates) != 1 or None in smg_dates or smg_dates.pop() > int(date):
        raise ValueError(
            "SMG packages must come from one nightly on or before TokenSpeed's"
        )

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
    for package, smg_version in smg_versions.items():
        updated, pins = re.subn(
            rf'"{re.escape(package)}==[^"\n]*"', f'"{package}=={smg_version}"', updated
        )
        if pins != 1:
            raise ValueError(f"Expected one {package} dependency")
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
    parser.add_argument(
        "--smg-pypi-json",
        required=True,
        help="PyPI JSON API base URL to read SMG nightly versions from",
    )
    args = parser.parse_args()
    smg_versions = newest_smg_nightly(
        {p: pypi_releases(args.smg_pypi_json, p) for p in SMG_PACKAGES}, args.date
    )
    pins = ", ".join(f"{p}=={v}" for p, v in smg_versions.items())
    print(f"TokenSpeed nightly {args.date} requires {pins}", file=sys.stderr)
    print(prepare_nightly(args.root, args.date, args.kernel_version, smg_versions))
