"""Resolve an engine wheel version and optionally apply it to a build checkout."""

import argparse
import re
import tomllib
from datetime import datetime, timedelta
from pathlib import Path


def prepare_release(
    root: Path, kind: str, date: str, target: str, release_tags: set[str]
) -> str:
    pyproject = root / "tokenspeed-engine/pyproject.toml"
    original = pyproject.read_text()
    project = tomllib.loads(original)["project"]
    if project["name"] != "tokenspeed-engine":
        raise ValueError("Expected tokenspeed-engine project")
    base = project["version"]
    if not re.fullmatch(r"[0-9]+\.[0-9]+\.[0-9]+", base):
        raise ValueError("Release requires a stable engine base version")

    if kind == "release":
        if date:
            raise ValueError("A stable release does not accept a date")
        version = base
    elif kind in ("post", "rc"):
        if not re.fullmatch(r"[0-9]{8}", date):
            raise ValueError("Release date must be YYYYMMDD")
        release_date = datetime.strptime(date, "%Y%m%d")
        if kind == "post" and target == "version":
            # Older nightly releases used a different tag prefix.
            prefixes = ("tokenspeed-engine-v", "tokenspeed-engine-nightly-v")
            while any(
                f"{prefix}{base}.post{date}" in release_tags for prefix in prefixes
            ):
                release_date += timedelta(days=1)
                date = release_date.strftime("%Y%m%d")
        suffix = ".post" if kind == "post" else "rc"
        version = f"{base}{suffix}{date}"
    else:
        raise ValueError(f"Unsupported release kind: {kind}")

    if target == "engine":
        updated, count = re.subn(
            r'(?m)^version = "[^"\n]+"$', f'version = "{version}"', original
        )
        if count != 1:
            raise ValueError("Expected one engine package version")
        pyproject.write_text(updated)
    elif target != "version":
        raise ValueError(f"Unsupported target: {target}")
    return version


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--target", choices=("version", "engine"), required=True)
    parser.add_argument("--kind", choices=("release", "post", "rc"), required=True)
    parser.add_argument("--date", required=True)
    parser.add_argument("--release-tags", type=Path)
    parser.add_argument("root", type=Path)
    args = parser.parse_args()
    resolve_post = args.target == "version" and args.kind == "post"
    if resolve_post != (args.release_tags is not None):
        parser.error("--release-tags is required only when resolving a post version")
    tags = set(args.release_tags.read_text().splitlines()) if resolve_post else set()
    print(prepare_release(args.root, args.kind, args.date, args.target, tags))
