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

"""Safeguard tests for expected kernel selection at each call site."""

from __future__ import annotations

import ast
import importlib
from pathlib import Path
from typing import Any, Optional

import pytest

# GEMM
import tokenspeed_kernel.numerics.reference.gemm
import tokenspeed_kernel.ops.gemm as _gemm_pkg
import tokenspeed_kernel.ops.gemm.deep_gemm
import tokenspeed_kernel.ops.gemm.flashinfer as _gemm_flashinfer
import tokenspeed_kernel.ops.gemm.triton as _gemm_triton

# MoE
import tokenspeed_kernel.ops.moe as _moe_pkg
import tokenspeed_kernel.ops.moe.flashinfer as _moe_flashinfer
import tokenspeed_kernel.ops.moe.gluon as _moe_gluon
import tokenspeed_kernel.ops.moe.triton as _moe_triton
import torch
from tokenspeed_kernel.ops.moe.flashinfer import (
    cutedsl_deepep_nvfp4 as _moe_cutedsl_deepep_nvfp4,
)
from tokenspeed_kernel.ops.moe.flashinfer import cutlass_fp8 as _moe_cutlass_fp8
from tokenspeed_kernel.ops.moe.flashinfer import cutlass_nvfp4 as _moe_cutlass_nvfp4
from tokenspeed_kernel.ops.moe.flashinfer import cutlass_unquant as _moe_cutlass_unquant
from tokenspeed_kernel.ops.moe.flashinfer import trtllm_mxfp4 as _moe_trtllm_mxfp4
from tokenspeed_kernel.ops.moe.flashinfer import trtllm_nvfp4 as _moe_trtllm_nvfp4
from tokenspeed_kernel.ops.moe.flashinfer import trtllm_unquant as _moe_trtllm_unquant
from tokenspeed_kernel.ops.moe.gluon import mxfp4 as _moe_gluon_mxfp4
from tokenspeed_kernel.ops.moe.triton import mxfp4 as _moe_triton_mxfp4
from tokenspeed_kernel.registry import KernelRegistry
from tokenspeed_kernel.selection import select_kernel
from tokenspeed_kernel.signature import FormatSignature

# -- Pre-import so they can be reloaded into the fresh registry. --


# ---------------------------------------------------------------------------
# 1. Real kernel registration via importlib.reload
# ---------------------------------------------------------------------------

_RELOAD_MODULES = [
    # MoE
    _moe_cutedsl_deepep_nvfp4,
    _moe_cutlass_fp8,
    _moe_cutlass_nvfp4,
    _moe_cutlass_unquant,
    _moe_trtllm_mxfp4,
    _moe_trtllm_nvfp4,
    _moe_trtllm_unquant,
    _moe_flashinfer,
    _moe_gluon_mxfp4,
    _moe_gluon,
    _moe_triton_mxfp4,
    _moe_triton,
    _moe_pkg,
    # GEMM
    tokenspeed_kernel.numerics.reference.gemm,
    tokenspeed_kernel.ops.gemm.deep_gemm,
    _gemm_flashinfer,
    _gemm_triton,
    _gemm_pkg,
]


@pytest.fixture(autouse=True)
def _kernel_registry(fresh_registry):
    """Reload real kernel registrations into a clean registry."""
    for mod in _RELOAD_MODULES:
        importlib.reload(mod)


# ---------------------------------------------------------------------------
# 2. AST-based call-site scanner
# ---------------------------------------------------------------------------

# Maps ``tokenspeed_kernel.<attr>(...)`` to ``(family, mode)``.
_API_MAP: dict[str, tuple[str, str]] = {
    "mm": ("gemm", "mm"),
}

_TORCH_DTYPE_MAP: dict[str, torch.dtype] = {
    "bfloat16": torch.bfloat16,
    "float16": torch.float16,
    "float32": torch.float32,
    "uint8": torch.uint8,
    "int32": torch.int32,
    "float8_e4m3fn": torch.float8_e4m3fn,
}


def _try_literal(node: ast.AST) -> tuple[Any, bool]:
    """Return ``(value, True)`` if *node* is a compile-time literal."""
    try:
        return ast.literal_eval(node), True
    except (ValueError, TypeError):
        return None, False


def _try_torch_dtype(node: ast.AST) -> Optional[torch.dtype]:
    """Resolve ``torch.<dtype>`` attribute access."""
    if isinstance(node, ast.Attribute) and isinstance(node.value, ast.Name):
        if node.value.id == "torch":
            return _TORCH_DTYPE_MAP.get(node.attr)
    return None


def _extract_literal_dict(node: ast.AST) -> Optional[dict[str, Any]]:
    """Extract a dict literal, keeping only keys/values resolvable at compile time."""
    if not isinstance(node, ast.Dict):
        return None
    result: dict[str, Any] = {}
    for key, value in zip(node.keys, node.values):
        k, k_ok = _try_literal(key)
        if not k_ok or not isinstance(k, str):
            continue
        v, v_ok = _try_literal(value)
        if v_ok:
            result[k] = v
    return result if result else None


def _extract_features(node: ast.AST) -> Optional[set[str]]:
    """Extract a set-literal of feature strings."""
    val, ok = _try_literal(node)
    if ok and isinstance(val, set):
        return val
    return None


# A ``CallSite`` tuple: (family, mode, dtype|None, features|None, traits, weight_format, expected_name, location)
CallSite = tuple[
    str, str, Optional[torch.dtype], Optional[set], dict, Optional[str], str, str
]


def _collect_call_sites(search_dir: Path) -> list[CallSite]:
    """Scan *search_dir* for ``tokenspeed_kernel.<api>(...)`` calls.

    Returns one entry per call whose ``expected_kernel_name`` is a string
    literal.  Calls with a variable or missing ``expected_kernel_name`` are
    silently skipped — they should be covered by ``_MANUAL_CALL_SITES``.
    """
    sites: list[CallSite] = []

    for py_path in sorted(search_dir.rglob("*.py")):
        source = py_path.read_text()
        try:
            tree = ast.parse(source, filename=str(py_path))
        except SyntaxError:
            continue

        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            func = node.func
            if not isinstance(func, ast.Attribute):
                continue
            if func.attr not in _API_MAP:
                continue
            if not (
                isinstance(func.value, ast.Name)
                and func.value.id == "tokenspeed_kernel"
            ):
                continue

            kwargs: dict[str, ast.AST] = {}
            for kw in node.keywords:
                if kw.arg is not None:
                    kwargs[kw.arg] = kw.value

            ekn_node = kwargs.get("expected_kernel_name")
            if ekn_node is None:
                continue
            expected, ok = _try_literal(ekn_node)
            if not ok or not isinstance(expected, str):
                continue

            family, mode = _API_MAP[func.attr]

            # -- dtype --
            dtype_node = kwargs.get("dtype")
            dtype: Optional[torch.dtype] = None
            if dtype_node is not None:
                dtype = _try_torch_dtype(dtype_node)

            # -- features --
            features: Optional[set[str]] = None
            feat_node = kwargs.get("features")
            if feat_node is not None:
                features = _extract_features(feat_node)

            # -- traits --
            traits: dict[str, Any] = {}
            traits_node = kwargs.get("traits")
            if traits_node is not None:
                parsed = _extract_literal_dict(traits_node)
                if parsed is not None:
                    traits = parsed

            weight_format: Optional[str] = None
            weight_format_node = kwargs.get("weight_format")
            if weight_format_node is not None:
                value, ok = _try_literal(weight_format_node)
                if ok and isinstance(value, str):
                    weight_format = value

            lineno = getattr(node, "lineno", 0)
            rel = py_path.relative_to(search_dir.parent.parent.parent)
            location = f"{rel}:{lineno}"

            sites.append(
                (
                    family,
                    mode,
                    dtype,
                    features,
                    traits,
                    weight_format,
                    expected,
                    location,
                )
            )

    return sites


# ---------------------------------------------------------------------------
# 3. Discover call sites
# ---------------------------------------------------------------------------

_SEARCH_DIR = (
    Path(__file__).resolve().parent.parent.parent / "python" / "tokenspeed" / "runtime"
)

_AUTO_SITES = _collect_call_sites(_SEARCH_DIR)


# Call sites that cannot be statically extracted (partial wrappers, variable
# expected_kernel_name, plan-mediated selection, etc.)
def _moe_apply_site(
    expected: str,
    traits: dict[str, Any],
    location: str,
    dtype: torch.dtype = torch.bfloat16,
) -> CallSite:
    return (
        "moe",
        "apply",
        dtype,
        None,
        traits,
        None,
        expected,
        location,
    )


_MANUAL_CALL_SITES: list[CallSite] = [
    _moe_apply_site(
        "flashinfer_trtllm_unquant_moe_apply",
        {
            "weight_dtype": "unquant",
            "activation": "swiglu",
            "supports_deferred_finalize": True,
            "supports_all_to_all_ep": False,
            "supports_ep": True,
            "ispp": 128,
            "internal_activation_dtype": "input",
        },
        "manual:runtime/layers/moe/expert.py:moe_plan/unquant_deferred",
    ),
    _moe_apply_site(
        "flashinfer_cutlass_fp8_moe_apply",
        {
            "weight_dtype": "fp8",
            "activation": "silu",
            "supports_all_to_all_ep": False,
            "supports_ep": True,
            "ispp": 128,
            "fp8_scale_block_shape": (128, 128),
            "internal_activation_dtype": "input",
        },
        "manual:runtime/layers/moe/expert.py:moe_plan/fp8",
    ),
    _moe_apply_site(
        "flashinfer_trtllm_nvfp4_moe_apply",
        {
            "weight_dtype": "nvfp4",
            "activation": "swiglu",
            "supports_deferred_finalize": True,
            "supports_all_to_all_ep": False,
            "supports_ep": True,
            "ispp": 128,
            "internal_activation_dtype": "fp4",
        },
        "manual:runtime/layers/moe/expert.py:moe_plan/nvfp4_deferred",
    ),
    _moe_apply_site(
        "flashinfer_cutedsl_deepep_nvfp4_moe_apply",
        {
            "weight_dtype": "nvfp4",
            "activation": "swiglu",
            "supports_all_to_all_ep": True,
            "supports_ep": True,
            "ispp": 128,
            "internal_activation_dtype": "fp4",
        },
        "manual:runtime/layers/moe/expert.py:moe_plan/nvfp4_deepep",
    ),
    _moe_apply_site(
        "flashinfer_trtllm_mxfp4_moe_apply",
        {
            "weight_dtype": "mxfp4",
            "activation": "swiglu",
            "supports_all_to_all_ep": False,
            "supports_ep": True,
            "ispp": 128,
            "internal_activation_dtype": "mxfp8",
            "supports_bias": True,
        },
        "manual:runtime/layers/moe/expert.py:moe_plan/mxfp4_trtllm",
    ),
    _moe_apply_site(
        "triton_mxfp4_moe_apply",
        {
            "weight_dtype": "mxfp4",
            "activation": "swiglu",
            "supports_all_to_all_ep": False,
            "supports_ep": False,
            "ispp": 128,
            "internal_activation_dtype": "fp8",
            "supports_bias": True,
        },
        "manual:runtime/layers/moe/expert.py:moe_plan/mxfp4_fp8",
    ),
]

_ALL_SITES = _AUTO_SITES + _MANUAL_CALL_SITES


_DTYPE_PREFERENCE = [
    torch.bfloat16,
    torch.float16,
    torch.float32,
    torch.int32,
    torch.uint8,
    torch.float8_e4m3fn,
]


def _all_storage_dtypes(spec) -> frozenset[torch.dtype]:
    return frozenset(
        tensor_format.storage_dtype
        for signature in spec.format_signatures
        for _role, tensor_format in signature.roles
    )


def _format_signature_for_any_role(spec, dtype: torch.dtype) -> FormatSignature | None:
    matches = tuple(
        signature
        for signature in sorted(spec.format_signatures, key=str)
        if any(
            tensor_format.storage_dtype == dtype
            for _role, tensor_format in signature.roles
        )
    )
    if len(matches) > 1:
        pytest.skip(f"Kernel {spec.name!r} has multiple signatures for {dtype}")
    return matches[0] if matches else None


def _infer_dtype(expected_name: str) -> torch.dtype:
    spec = KernelRegistry.get().get_by_name(expected_name)
    if spec is not None:
        storage_dtypes = _all_storage_dtypes(spec)
        for dt in _DTYPE_PREFERENCE:
            if dt in storage_dtypes:
                return dt
        if storage_dtypes:
            return next(iter(storage_dtypes))
    return torch.bfloat16


def _infer_format_signature(
    family: str,
    mode: str,
    dtype: torch.dtype,
    weight_format: Optional[str],
    spec,
) -> FormatSignature:
    signature = _format_signature_for_any_role(spec, dtype)
    if signature is None:
        pytest.skip(f"Kernel {spec.name!r} has no signature for {dtype}")
    return signature


def _site_id(site: CallSite) -> str:
    """Generate a readable test-id from a call-site tuple."""
    expected = site[6]
    location = site[7]
    return f"{expected}@{location}"


# ---------------------------------------------------------------------------
# 4. Parametrized test
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "site",
    _ALL_SITES,
    ids=[_site_id(s) for s in _ALL_SITES],
)
@pytest.mark.parametrize(
    "platform_name",
    [
        "h100_platform",
        "b200_platform",
        "mi350_platform",
    ],
)
def test_kernel_selection(site, platform_name, request):
    platform = request.getfixturevalue(platform_name)
    family, mode, raw_dtype, features, traits, weight_format, expected, location = site

    reg = KernelRegistry.get()
    spec = reg.get_by_name(expected)
    if spec is None:
        pytest.skip(f"Kernel {expected!r} not registered (dependency missing?)")
    if not spec.capability.satisfied_by(platform):
        pytest.skip(
            f"Kernel {expected!r} requires capability not satisfied by "
            f"{platform.device_name} ({platform.arch_version})"
        )

    dtype = raw_dtype or _infer_dtype(expected)
    signature = _infer_format_signature(family, mode, dtype, weight_format, spec)

    result = select_kernel(
        family,
        mode,
        signature,
        features=frozenset(features) if features else None,
        traits=traits,
        platform=platform,
    )
    assert result.name == expected, (
        f"Expected '{expected}' but got '{result.name}' "
        f"at {location} on {platform.device_name} "
        f"for {family}.{mode}(signature={signature}, features={features}, traits={traits})"
    )
