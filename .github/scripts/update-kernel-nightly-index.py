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

"""Add published CUDA 13 kernel wheels to the nightly Simple API index."""

import argparse
import json
import re
from html import escape
from pathlib import Path


def update_index(wheelhouse: Path, release: dict, distributions: Path) -> None:
    expected = {
        path.name
        for path in distributions.glob("tokenspeed-kernel-wheel-cu130-*/*.whl")
    }
    assets = {asset["name"]: asset for asset in release["assets"]}
    if not expected or not expected <= assets.keys():
        raise ValueError("Release is missing expected nightly wheels")

    entries = []
    for name in sorted(expected):
        asset = assets[name]
        url = asset["browser_download_url"]
        digest = asset["digest"]
        if not url.startswith("https://github.com/lightseekorg/whl/releases/download/"):
            raise ValueError("Nightly wheel URL must belong to lightseekorg/whl")
        if not re.fullmatch(r"sha256:[0-9a-f]{64}", digest or ""):
            raise ValueError(f"Missing SHA256 digest for {name}")
        entries.append(
            f'<a href="{escape(url)}#sha256={digest.removeprefix("sha256:")}">'
            f"{escape(name)}</a><br>\n"
        )

    index = wheelhouse / "nightly" / "tokenspeed-kernel" / "index.html"
    index.parent.mkdir(parents=True, exist_ok=True)
    previous = index.read_text() if index.exists() else "<!DOCTYPE html>\n"
    # Published assets are immutable; reruns add only missing links.
    additions = "".join(entry for entry in entries if entry not in previous)
    index.write_text(previous + additions)

    root = wheelhouse / "nightly" / "index.html"
    previous_root = root.read_text() if root.exists() else "<!DOCTYPE html>\n"
    package_link = '<a href="tokenspeed-kernel/">tokenspeed-kernel</a><br>\n'
    if package_link not in previous_root:
        root.write_text(previous_root + package_link)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("wheelhouse", type=Path)
    parser.add_argument("release_json", type=Path)
    parser.add_argument("distributions", type=Path)
    args = parser.parse_args()
    update_index(
        args.wheelhouse, json.loads(args.release_json.read_text()), args.distributions
    )
