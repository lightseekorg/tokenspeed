#!/usr/bin/env python3
"""Fetch the immutable image fixtures used by MiniMax-M3 active-MM CI."""

from __future__ import annotations

import argparse
import hashlib
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import BinaryIO
from urllib.request import urlopen

SMG_FIXTURE_COMMIT = "90803e7e90c11509b99bace83e7881011d43bdac"
SMG_RAW_BASE = (
    "https://raw.githubusercontent.com/lightseekorg/smg/"
    f"{SMG_FIXTURE_COMMIT}/e2e_test/fixtures/images"
)


@dataclass(frozen=True)
class Fixture:
    name: str
    sha256: str

    @property
    def url(self) -> str:
        return f"{SMG_RAW_BASE}/{self.name}"


FIXTURES = (
    Fixture(
        name="dog.jpg",
        sha256="e1cd91db28149f21f3c410ffa074d0fb8bc8950740ba140c7eaae130f0493464",
    ),
    Fixture(
        name="pug.jpg",
        sha256="9601bd492fd14ccea9adfe401135794adbf682f3354b3cdfed099fc20d6b3510",
    ),
)


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def fetch_fixture(
    fixture: Fixture,
    output_dir: Path,
    *,
    opener: Callable[..., BinaryIO] = urlopen,
) -> Path:
    """Fetch one fixture, verify its digest, and replace the target atomically."""
    output_dir.mkdir(parents=True, exist_ok=True)
    target = output_dir / fixture.name
    if target.is_file() and _sha256(target.read_bytes()) == fixture.sha256:
        return target

    with opener(fixture.url, timeout=60) as response:
        data = response.read()
    digest = _sha256(data)
    if digest != fixture.sha256:
        raise ValueError(
            f"{fixture.name} SHA256 mismatch: expected {fixture.sha256}, got {digest}"
        )

    temporary = target.with_suffix(f"{target.suffix}.tmp")
    temporary.write_bytes(data)
    temporary.replace(target)
    return target


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    for fixture in FIXTURES:
        path = fetch_fixture(fixture, args.output_dir)
        print(f"{fixture.sha256}  {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
