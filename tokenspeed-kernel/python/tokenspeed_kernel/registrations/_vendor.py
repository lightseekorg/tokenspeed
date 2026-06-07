# Copyright (c) 2026 LightSeek Foundation
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.

from __future__ import annotations

import importlib
from collections.abc import Iterable, Mapping
from types import ModuleType
from typing import Any

from tokenspeed_kernel.platform import current_platform
from tokenspeed_kernel.registry import error_fn

_VENDOR_REGISTRATION_MODULES = (
    "tokenspeed_kernel_nvidia.registration",
    "tokenspeed_kernel_amd.registration",
)


def false_fn(*args, **kwargs) -> bool:
    return False


def noop_fn(*args, **kwargs) -> None:
    return None


def _matches_current_platform(vendor: str) -> bool:
    try:
        return current_platform().vendor == vendor
    except RuntimeError:
        return False


def _normalize_missing_symbol(value: Any, fallback: Any) -> Any:
    if value is None:
        return fallback
    value_module = getattr(value, "__module__", "")
    value_name = getattr(value, "__name__", "")
    if value_module in _VENDOR_REGISTRATION_MODULES and value_name in {
        "ErrorClass",
        "error_fn",
    }:
        return fallback
    return value


def import_vendor_module(
    vendor: str,
    module_name: str,
    *,
    fallback: Any = error_fn,
) -> ModuleType | Any:
    if not _matches_current_platform(vendor):
        return fallback
    try:
        return importlib.import_module(module_name)
    except ImportError:
        return fallback


def export_vendor_symbols(
    vendor: str,
    module_name: str,
    names: Iterable[str],
    *,
    fallback_by_name: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    fallbacks = dict(fallback_by_name or {})
    module = import_vendor_module(vendor, module_name, fallback=None)
    exports: dict[str, Any] = {}
    for name in names:
        fallback = fallbacks.get(name, error_fn)
        if module is None:
            exports[name] = fallback
            continue
        exports[name] = _normalize_missing_symbol(
            getattr(module, name, fallback), fallback
        )
    return exports
