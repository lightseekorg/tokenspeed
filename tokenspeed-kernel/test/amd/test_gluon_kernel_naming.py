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

"""Source-level check that Gluon kernel names agree across the stack.

Proton names a kernel launch after the ``@gluon.jit`` function that was
compiled, while ``override=`` strings, ``describe_kernel`` and the dispatcher
scopes use the ``register_kernel(name=...)`` value. Keeping the two identical
(together with the registered Python ``def``) makes profiles, overrides and
source line up without a lookup table. This test walks both source trees so it
runs without a GPU and without importing the kernel packages.
"""

from __future__ import annotations

import ast
from collections.abc import Iterator
from pathlib import Path

import pytest

_KERNEL_OPS = Path(__file__).resolve().parents[2] / "python" / "tokenspeed_kernel"
_AMD_PACKAGE = (
    Path(__file__).resolve().parents[3]
    / "tokenspeed-kernel-amd"
    / "python"
    / "tokenspeed_kernel_amd"
)

# Registrations whose work is done by kernels shared with other registered
# entries, so no single kernel can carry the registration name. Each entry must
# name the shared kernel family it rides on; new registrations that cannot
# comply are added here deliberately, not by omission.
_SHARED_KERNEL_REGISTRATIONS: dict[str, str] = {
    # Dense DSA attention: one launcher family serves decode and prefill.
    "gluon_dsa_decode_gfx950": "dsa dense/packed kv kernels",
    "gluon_dsa_decode_gfx1250": "dsa selected dense/packed kernels",
    "gluon_dsa_prefill_gfx950": "dsa dense/packed kv kernels",
    "gluon_dsa_prefill_gfx1250": "dsa selected dense/packed kernels",
    "gluon_dsa_prefill_fp8_dense_gfx950": "dsa dense/packed kv kernels",
    "gluon_dsa_prefill_fp8_dense_gfx1250": "dsa selected dense/packed kernels",
    # DSv4 mxfp4 logits kernel serves both decode and prefill top-k.
    "gluon_dsv4_decode_topk_mxfp4_gfx950": "dsv4 mxfp4 logits + radix top-k",
    "gluon_dsv4_decode_topk_mxfp4_gfx1250": "dsv4 mxfp4 logits + radix top-k",
    "gluon_dsv4_prefill_topk_mxfp4_gfx950": "dsv4 mxfp4 logits + radix top-k",
    "gluon_dsv4_prefill_topk_mxfp4_gfx1250": "dsv4 mxfp4 logits + radix top-k",
    # Host-side planning, no kernel launch.
    "gluon_dsv4_plan_gfx950": "no kernel",
    "gluon_dsv4_plan_gfx1250": "no kernel",
    # MLA decode: trait-specialized registrations over one kernel.
    "gluon_mla_decode_bf16xbf16_gfx950_bh16bn64": "mla decode gfx950",
    "gluon_mla_decode_bf16xbf16_gfx950_bh64": "mla decode gfx950",
    "gluon_mla_decode_bf16xbf16_gfx950_bh16_multiblock": "mla decode gfx950",
    "gluon_mla_decode_bf16xbf16_gfx950_bh64_small": "mla decode gfx950",
    "gluon_mla_decode_bf16xfp8_gfx950_bh16bn128": "mla decode gfx950",
    "gluon_mla_decode_fp8xfp8_gfx950_bh16bn128": "mla decode gfx950",
    "gluon_mla_decode_projected_value_gfx950": "mla decode gfx950",
    "gluon_mla_decode_gfx1250": "mla decode gfx1250",
    "gluon_mla_decode_projected_value_gfx1250": "mla decode gfx1250",
    # Shared with the a16w4 situ MoE pipeline.
    "gluon_latent_expert_shared_gfx950": "a16w4 situ warp-decode stages",
    # Dense WMMA GEMV kernel also backs unregistered MLA/KDA projection entries.
    "gluon_wmma_dense_gemv_gfx1250": "wmma tdm dense m16 kernel",
    # MoE apply pipelines compose routing, sorting, quantize, GEMM stages and
    # reduce kernels that are shared across the family.
    "gluon_bf16_precomputed_moe_apply": "moe pipeline",
    "gluon_fp8_block_precomputed_moe_apply": "moe pipeline",
    "gluon_mxfp4_moe_apply": "moe pipeline",
    "gluon_mxfp4_dynamic_moe_apply": "moe pipeline",
    "gluon_mxfp4_precomputed_moe_apply": "moe pipeline",
    "gluon_mxfp4_gfx1250_precomputed_moe_apply": "moe pipeline",
    "gluon_mxfp4_a8w4_situ_precomputed_moe_apply": "moe pipeline",
    "gluon_mxfp4_a8w4_situ_ep_precomputed_moe_apply": "moe pipeline",
    "gluon_mxfp4_a8w4_situ_gfx1250_precomputed_moe_apply": "moe pipeline",
    "gluon_mxfp4_a16w4_situ_ep_precomputed_moe_apply": "moe pipeline",
    "gluon_mxfp4_a16w4_swiglu_ep_precomputed_moe_apply": "moe pipeline",
}


def _keyword(call: ast.Call, name: str) -> ast.expr | None:
    for keyword in call.keywords:
        if keyword.arg == name:
            return keyword.value
    return None


def _is_register_kernel(call: ast.expr) -> bool:
    return isinstance(call, ast.Call) and ast.unparse(call.func).endswith(
        "register_kernel"
    )


def _gluon_registrations() -> Iterator[tuple[str, str, str]]:
    """Yield ``(registered name, decorated def name, location)`` for Gluon."""
    for path in sorted(_KERNEL_OPS.rglob("*.py")):
        source = path.read_text()
        if "register_kernel" not in source:
            continue
        tree = ast.parse(source)
        for node in ast.walk(tree):
            call: ast.Call | None = None
            target: str | None = None
            if isinstance(node, ast.FunctionDef):
                for decorator in node.decorator_list:
                    if _is_register_kernel(decorator):
                        call, target = decorator, node.name
            elif (
                isinstance(node, ast.Call)
                and _is_register_kernel(node.func)
                and node.args
                and isinstance(node.args[0], ast.Name)
            ):
                # ``register_kernel(...)(fn)`` call form.
                call, target = node.func, node.args[0].id
            if call is None or target is None:
                continue
            solution = _keyword(call, "solution")
            if not (isinstance(solution, ast.Constant) and solution.value == "gluon"):
                continue
            name = _keyword(call, "name")
            assert isinstance(name, ast.Constant), f"{path}:{node.lineno}"
            location = f"{path.relative_to(_KERNEL_OPS)}:{node.lineno}"
            yield name.value, target, location


def _amd_top_level_defs() -> tuple[dict[str, list[str]], dict[str, list[str]]]:
    """Return ``(jit kernels, plain functions)`` keyed by name -> locations."""
    jit: dict[str, list[str]] = {}
    plain: dict[str, list[str]] = {}
    for path in sorted(_AMD_PACKAGE.rglob("*.py")):
        tree = ast.parse(path.read_text())
        location = str(path.relative_to(_AMD_PACKAGE))
        for node in tree.body:
            if not isinstance(node, ast.FunctionDef):
                continue
            decorators = [
                ast.unparse(d.func if isinstance(d, ast.Call) else d)
                for d in node.decorator_list
            ]
            bucket = jit if any(d.endswith("jit") for d in decorators) else plain
            bucket.setdefault(node.name, []).append(f"{location}:{node.lineno}")
    return jit, plain


@pytest.fixture(scope="module")
def registrations() -> list[tuple[str, str, str]]:
    found = list(_gluon_registrations())
    assert found, "no Gluon registrations found"
    return found


@pytest.fixture(scope="module")
def amd_defs() -> tuple[dict[str, list[str]], dict[str, list[str]]]:
    if not _AMD_PACKAGE.is_dir():
        pytest.skip("tokenspeed-kernel-amd source tree not available")
    return _amd_top_level_defs()


def test_registered_def_matches_registration_name(registrations) -> None:
    mismatched = [
        f"{location}: name={name!r} def={target}"
        for name, target, location in registrations
        if name != target
    ]
    assert not mismatched, "\n".join(mismatched)


def test_gluon_kernel_carries_registration_name(registrations, amd_defs) -> None:
    jit, _ = amd_defs
    missing = [
        f"{location}: no @jit kernel named {name!r} in tokenspeed_kernel_amd"
        for name, _, location in registrations
        if name not in _SHARED_KERNEL_REGISTRATIONS and name not in jit
    ]
    assert not missing, "\n".join(missing)


def test_amd_launcher_never_takes_registration_name(registrations, amd_defs) -> None:
    _, plain = amd_defs
    clashes = [
        f"{name!r} is a plain function at {', '.join(plain[name])}; "
        f"launchers are named launch_{name}"
        for name, _, _ in registrations
        if name in plain
    ]
    assert not clashes, "\n".join(clashes)


def test_shared_kernel_exemptions_are_live(registrations) -> None:
    registered = {name for name, _, _ in registrations}
    stale = sorted(set(_SHARED_KERNEL_REGISTRATIONS) - registered)
    assert not stale, f"exemptions without a registration: {stale}"
