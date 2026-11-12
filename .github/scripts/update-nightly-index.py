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

"""Add published wheels to the Simple API indexes."""

import argparse
import json
import re
from html import escape
from pathlib import Path


def _package_contents(
    previous: str, expected: set[str], assets: dict, replace: bool
) -> str:
    entries = []
    for name in sorted(expected):
        asset = assets[name]
        url = asset["browser_download_url"]
        digest = asset["digest"]
        if not url.startswith("https://github.com/lightseekorg/whl/releases/download/"):
            raise ValueError("Wheel URL must belong to lightseekorg/whl")
        if not re.fullmatch(r"sha256:[0-9a-f]{64}", digest or ""):
            raise ValueError(f"Missing SHA256 digest for {name}")
        existing = re.search(
            rf'<a href="[^"]+#sha256=([0-9a-f]{{64}})">{re.escape(escape(name))}</a><br>\n',
            previous,
        )
        if existing is not None:
            if existing.group(1) == digest.removeprefix("sha256:"):
                continue
            if not replace:
                raise ValueError(
                    f"Published digest changed for {name}; replacement required"
                )
            previous = previous.replace(existing.group(0), "")
            # A fresh URL also invalidates pip's cached HTTP response body.
            url += f"?sha256={digest.removeprefix('sha256:')}"
        entries.append(
            f'<a href="{escape(url)}#sha256={digest.removeprefix("sha256:")}">'
            f"{escape(name)}</a><br>\n"
        )

    return previous + "".join(entries)


def update_index(
    wheelhouse: Path,
    release: dict,
    distributions: Path,
    package: str,
    variant: str,
    replace: bool = False,
) -> list[str]:
    nightly = wheelhouse / "nightly"
    if package == "tokenspeed-engine":
        if replace:
            raise ValueError("Engine release assets must not be replaced")
        if variant == "engine-release":
            roots = [wheelhouse]
        elif variant == "engine-nightly":
            roots = [nightly, nightly / "rocm7.2"]
        else:
            raise ValueError(f"Unsupported engine index variant: {variant}")
    else:
        if variant not in ("cu130", "rocm72"):
            raise ValueError(f"Unsupported nightly variant: {variant}")
        if package == "tokenspeed-kernel-amd" and variant != "rocm72":
            raise ValueError("AMD kernel nightlies require the ROCm index")
        roots = [nightly / "rocm7.2" if variant == "rocm72" else nightly]
    pattern = {
        "tokenspeed": "tokenspeed-dist/*.whl",
        "tokenspeed-engine": "tokenspeed_engine-*.whl",
        "tokenspeed-kernel-amd": "tokenspeed-kernel-amd-dist/*.whl",
        "tokenspeed-kernel": (
            "tokenspeed-kernel-wheel-cu130-*/*.whl"
            if variant == "cu130"
            else "tokenspeed-kernel-rocm72-wheel-*/*.whl"
        ),
    }[package]
    expected = {path.name for path in distributions.glob(pattern)}
    assets = {asset["name"]: asset for asset in release["assets"]}
    if not expected or not expected <= assets.keys():
        raise ValueError("Release is missing expected wheels")

    updates = {}
    for root in roots:
        index = root / package / "index.html"
        previous = index.read_text() if index.exists() else "<!DOCTYPE html>\n"
        updates[index] = _package_contents(previous, expected, assets, replace)
        package_list = root / "index.html"
        contents = (
            package_list.read_text() if package_list.exists() else "<!DOCTYPE html>\n"
        )
        package_link = f'<a href="{package}/">{package}</a><br>\n'
        updates[package_list] = (
            contents if package_link in contents else contents + package_link
        )

    if variant in ("rocm72", "engine-nightly"):
        nightly_root = wheelhouse / "nightly" / "index.html"
        previous_nightly_root = updates.get(
            nightly_root,
            nightly_root.read_text() if nightly_root.exists() else "<!DOCTYPE html>\n",
        )
        rocm_link = '<a href="rocm7.2/">rocm7.2</a><br>\n'
        if rocm_link not in previous_nightly_root:
            updates[nightly_root] = previous_nightly_root + rocm_link

    # Validate every index before writing, including the existing ROCm alias.
    for path, contents in updates.items():
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(contents)
    return [str(path.relative_to(wheelhouse)) for path in updates]


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--package",
        choices=(
            "tokenspeed",
            "tokenspeed-engine",
            "tokenspeed-kernel",
            "tokenspeed-kernel-amd",
        ),
        required=True,
    )
    parser.add_argument(
        "--variant",
        choices=("cu130", "rocm72", "engine-release", "engine-nightly"),
        required=True,
    )
    parser.add_argument("--replace", action="store_true")
    parser.add_argument("wheelhouse", type=Path)
    parser.add_argument("release_json", type=Path)
    parser.add_argument("distributions", type=Path)
    args = parser.parse_args()
    paths = update_index(
        args.wheelhouse,
        json.loads(args.release_json.read_text()),
        args.distributions,
        args.package,
        args.variant,
        args.replace,
    )
    print("\n".join(paths))
