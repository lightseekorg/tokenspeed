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

from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType


def _require_fields(
    raw: Mapping[str, object], required: frozenset[str], context: str
) -> None:
    keys = frozenset(raw)
    missing = sorted(required - keys)
    unknown = sorted(keys - required)
    if missing:
        raise ValueError(f"{context} is missing fields: {', '.join(missing)}")
    if unknown:
        raise ValueError(f"{context} has unknown fields: {', '.join(unknown)}")


def _require_mapping(
    raw: Mapping[str, object], field: str, context: str
) -> Mapping[str, object]:
    value = raw[field]
    if not isinstance(value, Mapping):
        raise TypeError(f"{context}.{field} must be an object")
    if not all(isinstance(key, str) for key in value):
        raise TypeError(f"{context}.{field} keys must be strings")
    return value


def _require_int(raw: Mapping[str, object], field: str, context: str) -> int:
    value = raw[field]
    if type(value) is not int:
        raise TypeError(f"{context}.{field} must be an integer")
    return value


def _require_str(raw: Mapping[str, object], field: str, context: str) -> str:
    value = raw[field]
    if not isinstance(value, str) or not value:
        raise TypeError(f"{context}.{field} must be a non-empty string")
    return value


def _require_nullable_str(
    raw: Mapping[str, object], field: str, context: str
) -> str | None:
    value = raw[field]
    if value is not None and (not isinstance(value, str) or not value):
        raise TypeError(f"{context}.{field} must be a non-empty string or null")
    return value


@dataclass(frozen=True)
class WarmupPlatform:
    vendor: str
    minimum_compute_capability: int
    maximum_compute_capability: int

    @classmethod
    def from_json(cls, raw: Mapping[str, object]) -> WarmupPlatform:
        _require_fields(
            raw,
            frozenset(
                {
                    "vendor",
                    "minimum_compute_capability",
                    "maximum_compute_capability",
                }
            ),
            "platform",
        )
        vendor = _require_str(raw, "vendor", "platform")
        minimum = _require_int(raw, "minimum_compute_capability", "platform")
        maximum = _require_int(raw, "maximum_compute_capability", "platform")
        if vendor != "nvidia":
            raise ValueError("the first warmup schema supports vendor 'nvidia' only")
        if minimum <= 0 or maximum < minimum:
            raise ValueError("platform compute capability range is invalid")
        return cls(
            vendor=vendor,
            minimum_compute_capability=minimum,
            maximum_compute_capability=maximum,
        )


@dataclass(frozen=True)
class WarmupProvenance:
    model: str
    model_revision: str | None
    quantization: str

    @classmethod
    def from_json(cls, raw: Mapping[str, object]) -> WarmupProvenance:
        _require_fields(
            raw,
            frozenset({"model", "model_revision", "quantization"}),
            "provenance",
        )
        return cls(
            model=_require_str(raw, "model", "provenance"),
            model_revision=_require_nullable_str(raw, "model_revision", "provenance"),
            quantization=_require_str(raw, "quantization", "provenance"),
        )


@dataclass(frozen=True)
class WarmupTarget:
    api: str
    solution: str
    definition: Mapping[str, object]

    @classmethod
    def from_json(cls, raw: Mapping[str, object], index: int) -> WarmupTarget:
        context = f"target {index}"
        _require_fields(raw, frozenset({"api", "solution", "definition"}), context)
        api = _require_str(raw, "api", context)
        if api.count(".") != 1 or api.startswith(".") or api.endswith("."):
            raise ValueError(f"{context}.api must use family.mode form")
        definition = _require_mapping(raw, "definition", context)
        return cls(
            api=api,
            solution=_require_str(raw, "solution", context),
            definition=MappingProxyType(dict(definition)),
        )


@dataclass(frozen=True)
class WarmupProfile:
    schema_version: int
    id: str
    provider: str
    platform: WarmupPlatform
    provenance: WarmupProvenance
    targets: tuple[WarmupTarget, ...]

    @classmethod
    def from_json(cls, raw: Mapping[str, object]) -> WarmupProfile:
        _require_fields(
            raw,
            frozenset(
                {
                    "schema_version",
                    "id",
                    "provider",
                    "platform",
                    "provenance",
                    "targets",
                }
            ),
            "warmup profile",
        )
        schema_version = _require_int(raw, "schema_version", "warmup profile")
        if schema_version != 1:
            raise ValueError("warmup profile schema_version must be 1")
        provider = _require_str(raw, "provider", "warmup profile")
        if provider != "flashinfer":
            raise ValueError(
                "the first warmup schema supports provider 'flashinfer' only"
            )
        raw_targets = raw["targets"]
        if not isinstance(raw_targets, list) or not raw_targets:
            raise TypeError("warmup profile.targets must be a non-empty list")
        targets = []
        for index, raw_target in enumerate(raw_targets):
            if not isinstance(raw_target, Mapping):
                raise TypeError(f"target {index} must be an object")
            targets.append(WarmupTarget.from_json(raw_target, index))
        profile_id = _require_str(raw, "id", "warmup profile")
        platform = WarmupPlatform.from_json(
            _require_mapping(raw, "platform", "warmup profile")
        )
        if not profile_id.startswith(f"{platform.vendor}/{provider}/"):
            raise ValueError(
                "warmup profile.id must begin with its platform vendor and provider"
            )
        return cls(
            schema_version=schema_version,
            id=profile_id,
            provider=provider,
            platform=platform,
            provenance=WarmupProvenance.from_json(
                _require_mapping(raw, "provenance", "warmup profile")
            ),
            targets=tuple(targets),
        )
