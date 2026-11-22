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

"""Check the coordinated nightly's versions and existing release provenance."""

import argparse
import json
import re
import subprocess
import tomllib
from datetime import datetime
from pathlib import Path


def check_source(body: str, source_sha: str, replace: bool) -> None:
    sources = set(
        re.findall(r"lightseekorg/tokenspeed@([0-9a-f]{40})(?![0-9a-f])", body)
    )
    if not replace and sources != {source_sha}:
        raise ValueError(
            "Nightly version already exists with different or missing source provenance; enable replace_nightly to replace it"
        )


def plan(
    root: Path,
    date: str,
    source_sha: str,
    releases: list[dict],
    replace: bool,
    release_candidate: str,
) -> dict:
    if not re.fullmatch(r"[0-9]{8}", date):
        raise ValueError("Nightly date must be YYYYMMDD")
    datetime.strptime(date, "%Y%m%d")
    if not re.fullmatch(r"[0-9a-f]{40}", source_sha):
        raise ValueError("Expected a full source commit")
    rc = release_candidate.removeprefix("rc")
    if rc and not re.fullmatch(r"[1-9][0-9]*", rc):
        raise ValueError("release_candidate must be a positive number or rcN")
    suffix = f"rc{rc}.post{date}" if rc else f".post{date}"
    kernel = re.search(
        r'^BASE_VERSION = "([0-9]+\.[0-9]+\.[0-9]+)"$',
        (root / "tokenspeed-kernel/python/setup.py").read_text(),
        re.M,
    )
    if kernel is None:
        raise ValueError("Cannot resolve kernel base version")

    def base(path: str) -> str:
        version = tomllib.loads((root / path).read_text())["project"]["version"]
        if not re.fullmatch(r"[0-9]+\.[0-9]+\.[0-9]+", version):
            raise ValueError(f"Nightly requires a stable base version: {path}")
        return version

    tags = [
        f"tokenspeed-kernel-nightly-v{kernel[1]}{suffix}-cu130",
        f"tokenspeed-nightly-v{base('python/pyproject.toml')}{suffix}",
    ]
    engine = f"{base('tokenspeed-engine/pyproject.toml')}{suffix}"
    tags.append(f"tokenspeed-engine-v{engine}")
    if not rc:
        tags += [
            f"tokenspeed-kernel-amd-nightly-v{base('tokenspeed-kernel-amd/pyproject.toml')}.post{date}",
            f"tokenspeed-kernel-nightly-v{kernel[1]}{suffix}-rocm72",
        ]
        if any(
            r["tag_name"] == f"tokenspeed-engine-nightly-v{engine}" for r in releases
        ):
            raise ValueError(
                "This engine date uses a legacy release tag; choose another date"
            )
    existing = {r["tag_name"]: r for r in releases}
    for tag in tags:
        if tag in existing:
            check_source(existing[tag].get("body") or "", source_sha, replace)
    return {
        "version_date": date,
        "source_sha": source_sha,
        "release_candidate": rc,
        "replace_nightly": replace,
        "tags": tags,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-sha", required=True)
    parser.add_argument("--replace", action="store_true")
    commands = parser.add_subparsers(dest="command", required=True)
    prepare = commands.add_parser("plan")
    prepare.add_argument("--date", required=True)
    prepare.add_argument("--release-candidate", default="")
    prepare.add_argument("--releases", type=Path, required=True)
    prepare.add_argument("root", type=Path)
    check = commands.add_parser("check")
    check.add_argument("tag")
    args = parser.parse_args()
    if args.command == "check":
        result = subprocess.run(
            [
                "gh",
                "release",
                "view",
                args.tag,
                "--repo",
                "lightseekorg/whl",
                "--json",
                "body",
                "--jq",
                ".body",
            ],
            text=True,
            capture_output=True,
            check=True,
        )
        check_source(result.stdout, args.source_sha, args.replace)
    else:
        pages = json.loads(args.releases.read_text())
        print(
            json.dumps(
                plan(
                    args.root,
                    args.date,
                    args.source_sha,
                    [release for page in pages for release in page],
                    args.replace,
                    args.release_candidate,
                )
            )
        )


if __name__ == "__main__":
    main()
