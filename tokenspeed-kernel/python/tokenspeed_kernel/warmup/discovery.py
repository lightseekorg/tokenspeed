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

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from importlib.resources import files
from importlib.resources.abc import Traversable
from pathlib import PurePosixPath

from tokenspeed_kernel.warmup.config.schema import WarmupProfile


@dataclass(frozen=True)
class LoadedWarmupProfile:
    profile: WarmupProfile
    source: str
    sha256: str


def _config_root() -> Traversable:
    return files("tokenspeed_kernel.warmup.config")


def _walk_json(root: Traversable, prefix: tuple[str, ...]) -> list[str]:
    result = []
    for child in root.iterdir():
        if child.is_dir():
            result.extend(_walk_json(child, (*prefix, child.name)))
        elif child.name.endswith(".json"):
            result.append("/".join((*prefix, child.name.removesuffix(".json"))))
    return result


def list_config_ids() -> list[str]:
    return sorted(_walk_json(_config_root(), ()))


def _config_parts(config_id: str) -> tuple[str, ...]:
    if not config_id or "\\" in config_id or config_id.endswith(".json"):
        raise ValueError(f"invalid warmup configuration ID {config_id!r}")
    path = PurePosixPath(config_id)
    parts = path.parts
    if path.is_absolute() or not parts or any(part in {"", ".", ".."} for part in parts):
        raise ValueError(f"invalid warmup configuration ID {config_id!r}")
    if path.as_posix() != config_id:
        raise ValueError(f"invalid warmup configuration ID {config_id!r}")
    return parts


def _reject_duplicate_keys(pairs: list[tuple[str, object]]) -> dict[str, object]:
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON field {key!r}")
        result[key] = value
    return result


def load_config(config_id: str) -> LoadedWarmupProfile:
    parts = _config_parts(config_id)
    resource = _config_root().joinpath(*parts[:-1], f"{parts[-1]}.json")
    if not resource.is_file():
        raise ValueError(f"unknown warmup configuration {config_id!r}")
    source = resource.read_text(encoding="utf-8")
    try:
        raw = json.loads(source, object_pairs_hook=_reject_duplicate_keys)
    except json.JSONDecodeError as error:
        raise ValueError(f"invalid JSON in warmup configuration {config_id!r}") from error
    if not isinstance(raw, dict):
        raise TypeError(f"warmup configuration {config_id!r} must contain an object")
    profile = WarmupProfile.from_json(raw)
    if profile.id != config_id:
        raise ValueError(
            f"warmup configuration ID {profile.id!r} does not match path {config_id!r}"
        )
    return LoadedWarmupProfile(
        profile=profile,
        source=source,
        sha256=hashlib.sha256(source.encode()).hexdigest(),
    )
