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

import importlib
import json
from dataclasses import dataclass
from pathlib import Path

from tokenspeed_kernel.ops.tuning import set_autotune_max_num_tokens

__all__ = ["LoadedWarmupBundle", "load_warmup_bundle"]


@dataclass(frozen=True)
class LoadedWarmupBundle:
    path: Path
    source_id: str
    source_sha256: str
    maximum_num_tokens: int
    profile_count: int


def _object(value: object, context: str) -> dict[str, object]:
    if not isinstance(value, dict) or not all(isinstance(key, str) for key in value):
        raise TypeError(f"{context} must be an object")
    return value


def _fields(raw: dict[str, object], names: set[str], context: str) -> None:
    missing = sorted(names - set(raw))
    unknown = sorted(set(raw) - names)
    if missing:
        raise ValueError(f"{context} is missing fields: {', '.join(missing)}")
    if unknown:
        raise ValueError(f"{context} has unknown fields: {', '.join(unknown)}")


def _string(raw: dict[str, object], name: str, context: str) -> str:
    value = raw.get(name)
    if not isinstance(value, str) or not value:
        raise TypeError(f"{context}.{name} must be a non-empty string")
    return value


def _integer(raw: dict[str, object], name: str, context: str) -> int:
    value = raw.get(name)
    if type(value) is not int:
        raise TypeError(f"{context}.{name} must be an integer")
    return value


def load_warmup_bundle(path: str, expected_max_num_tokens: int) -> LoadedWarmupBundle:
    """Validate and load a generated FlashInfer warmup bundle.

    Args:
        path: Bundle directory containing manifest.json and flashinfer.json.
        expected_max_num_tokens: Largest token count serving may send to a
            warmed kernel.

    Returns:
        The validated bundle identity and coverage.
    """
    if expected_max_num_tokens <= 0:
        raise ValueError("expected_max_num_tokens must be positive")
    root = Path(path).expanduser().resolve()
    manifest_path = root / "manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(f"warmup bundle manifest not found: {manifest_path}")
    with manifest_path.open() as file:
        manifest = json.load(file)
    manifest = _object(manifest, "warmup manifest")
    _fields(
        manifest,
        {
            "schema_version",
            "complete",
            "source",
            "provider",
            "maximum_num_tokens",
            "environment",
            "targets",
            "flashinfer",
            "command",
            "created_at",
        },
        "warmup manifest",
    )
    if manifest.get("schema_version") != 1:
        raise ValueError("warmup manifest schema_version must be 1")
    if manifest.get("complete") is not True:
        raise ValueError("warmup bundle is incomplete")
    if manifest.get("provider") != "flashinfer":
        raise ValueError("warmup bundle provider must be 'flashinfer'")

    maximum_num_tokens = _integer(manifest, "maximum_num_tokens", "warmup manifest")
    if expected_max_num_tokens > maximum_num_tokens:
        raise ValueError(
            f"warmup bundle covers at most {maximum_num_tokens} tokens, "
            f"but serving requires {expected_max_num_tokens}"
        )
    source = _object(manifest.get("source"), "warmup manifest.source")
    _fields(source, {"id", "sha256"}, "warmup manifest.source")
    flashinfer = _object(manifest.get("flashinfer"), "warmup manifest.flashinfer")
    _fields(
        flashinfer,
        {"path", "profile_count", "profile_keys"},
        "warmup manifest.flashinfer",
    )
    cache_name = _string(flashinfer, "path", "warmup manifest.flashinfer")
    if cache_name != "flashinfer.json":
        raise ValueError("warmup manifest FlashInfer path must be 'flashinfer.json'")
    profile_count = _integer(flashinfer, "profile_count", "warmup manifest.flashinfer")
    if profile_count <= 0:
        raise ValueError("warmup bundle contains no FlashInfer profiles")
    profile_keys = flashinfer["profile_keys"]
    if not isinstance(profile_keys, list) or not all(
        isinstance(key, str) and key for key in profile_keys
    ):
        raise TypeError("warmup manifest.flashinfer.profile_keys must be strings")
    if len(profile_keys) != profile_count or len(set(profile_keys)) != profile_count:
        raise ValueError("warmup manifest FlashInfer profile count is inconsistent")
    cache_path = root / cache_name
    if not cache_path.is_file():
        raise FileNotFoundError(f"warmup bundle cache not found: {cache_path}")
    with cache_path.open() as file:
        native_cache = json.load(file)
    native_cache = _object(native_cache, "FlashInfer cache")
    actual_keys = sorted(key for key in native_cache if not key.startswith("_"))
    if actual_keys != profile_keys:
        raise ValueError("warmup manifest does not match the FlashInfer cache keys")

    set_autotune_max_num_tokens(maximum_num_tokens)
    autotuner = importlib.import_module("flashinfer.autotuner")
    tuner = autotuner.AutoTuner.get()
    tuner.clear_cache()
    if not tuner.load_configs(str(cache_path)):
        tuner.clear_cache()
        raise RuntimeError("FlashInfer rejected the warmup bundle cache")
    return LoadedWarmupBundle(
        path=root,
        source_id=_string(source, "id", "warmup manifest.source"),
        source_sha256=_string(source, "sha256", "warmup manifest.source"),
        maximum_num_tokens=maximum_num_tokens,
        profile_count=profile_count,
    )
