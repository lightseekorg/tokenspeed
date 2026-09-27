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

"""Validate the source-only tokenspeed-mla release wheel without GPU dependencies."""

import argparse
import tomllib
import zipfile
from email.parser import BytesParser
from pathlib import Path

from packaging.version import Version


def read_version(package_dir: Path) -> str:
    project = tomllib.loads((package_dir / "pyproject.toml").read_text())["project"]
    if project["name"] != "tokenspeed-mla":
        raise ValueError("expected project name tokenspeed-mla")
    version = project["version"]
    parsed = Version(version)
    if parsed.local is not None or str(parsed) != version:
        raise ValueError(
            f"expected a normalized public release version, found {version}"
        )
    return version


def check_release(package_dir: Path, dist_dir: Path) -> Path:
    version = read_version(package_dir)
    artifacts = list(dist_dir.iterdir())
    expected_name = f"tokenspeed_mla-{version}-py3-none-any.whl"
    if len(artifacts) != 1 or artifacts[0].name != expected_name:
        raise ValueError(f"expected exactly one artifact: {expected_name}")
    wheel_path = artifacts[0]
    dist_info = f"tokenspeed_mla-{version}.dist-info"
    source_root = package_dir / "python"
    sources = {
        path.relative_to(source_root).as_posix(): path
        for path in (source_root / "tokenspeed_mla").rglob("*.py")
    }
    required = {
        "tokenspeed_mla/__init__.py",
        "tokenspeed_mla/fmha.py",
        "tokenspeed_mla/mla_prefill.py",
    }
    if not required.issubset(sources):
        raise ValueError("package source is missing required JIT modules")

    with zipfile.ZipFile(wheel_path) as wheel:
        names = wheel.namelist()
        if len(names) != len(set(names)):
            raise ValueError("wheel contains duplicate entries")
        for name in names:
            if (
                name.startswith("tokenspeed_mla/objs/")
                or name == "tokenspeed_mla/fmha_binary.py"
                or name.endswith(
                    (".so", ".a", ".o", ".cubin", ".fatbin", ".ptx", ".dll", ".dylib")
                )
            ):
                raise ValueError(
                    f"wheel contains a bundled binary or AOT module: {name}"
                )

        metadata = BytesParser().parsebytes(wheel.read(f"{dist_info}/METADATA"))
        if metadata.get_all("Name") != ["tokenspeed-mla"] or metadata.get_all(
            "Version"
        ) != [version]:
            raise ValueError(
                "wheel metadata does not match the project name and version"
            )
        tags = BytesParser().parsebytes(wheel.read(f"{dist_info}/WHEEL"))
        if tags.get_all("Root-Is-Purelib") != ["true"] or tags.get_all("Tag") != [
            "py3-none-any"
        ]:
            raise ValueError("wheel must contain pure Python with the py3-none-any tag")

        packaged_sources = {
            name for name in names if name.startswith("tokenspeed_mla/")
        }
        if packaged_sources != sources.keys():
            raise ValueError("wheel package contents do not match the source modules")
        for name, source_path in sources.items():
            if wheel.read(name) != source_path.read_bytes():
                raise ValueError(f"wheel source differs from the checkout: {name}")
        for license_name in ("LICENSE", "THIRDPARTYNOTICES"):
            candidates = {
                f"{dist_info}/{license_name}",
                f"{dist_info}/licenses/{license_name}",
            }.intersection(names)
            if len(candidates) != 1:
                raise ValueError(f"wheel must include {license_name} exactly once")
            if (
                wheel.read(candidates.pop())
                != (package_dir / license_name).read_bytes()
            ):
                raise ValueError(f"wheel {license_name} differs from the checkout")
    return wheel_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--package-dir", type=Path, required=True)
    parser.add_argument("--dist-dir", type=Path, required=True)
    args = parser.parse_args()
    print(f"Validated {check_release(args.package_dir, args.dist_dir)}")
