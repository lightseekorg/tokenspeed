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

import ast
import runpy
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace
from unittest.mock import patch

import pytest

ROOT = Path(__file__).resolve().parents[2]
ADAPTER = ROOT / "tokenspeed-kernel/python/tokenspeed_kernel/_triton.py"


def _load_adapter(*, is_npu):
    names = ["torch"]
    if is_npu:
        names += ["tokenspeed_kernel_npu._triton"]
    else:
        names += [
            "tokenspeed_triton.experimental.gluon.language",
            "tokenspeed_triton.language.core",
            "tokenspeed_triton.language.extra",
            "tokenspeed_triton.tools.tensor_descriptor",
            "tokenspeed_triton.profiler",
        ]
    modules = {}
    for name in names:
        parts = name.split(".")
        for depth in range(1, len(parts) + 1):
            full_name = ".".join(parts[:depth])
            if full_name not in modules:
                module = ModuleType(full_name)
                module.__path__ = []
                modules[full_name] = module
                if depth > 1:
                    setattr(
                        modules[".".join(parts[: depth - 1])], parts[depth - 1], module
                    )
    modules["torch"].npu = SimpleNamespace(is_available=lambda: is_npu)
    if is_npu:
        provider = modules["tokenspeed_kernel_npu._triton"]
        for name in ("libdevice", "proton", "tl", "triton"):
            setattr(provider, name, object())
        modules["tokenspeed_triton"] = None
    else:
        modules["tokenspeed_triton.language.core"]._aggregate = object()
        modules["tokenspeed_triton.language.extra"].libdevice = object()
        modules["tokenspeed_triton.tools.tensor_descriptor"].TensorDescriptor = object()
    with patch.dict(sys.modules, modules):
        adapter = runpy.run_path(str(ADAPTER))
    return adapter, modules


class TestGluonAdapter:
    def test_ascend_decorates_gpu_kernel_without_importing_gluon(self, subtests):
        adapter, modules = _load_adapter(is_npu=True)
        assert adapter["triton"] is modules["tokenspeed_kernel_npu._triton"].triton
        path = (
            ROOT / "tokenspeed-kernel/python/tokenspeed_kernel/ops/gemm/_gluon/gemv.py"
        )
        source = ast.parse(path.read_text())
        kernel = next(
            node
            for node in source.body
            if isinstance(node, ast.FunctionDef) and node.name == "_simt_gemm_kernel"
        )
        # Execute the real decorated definition, retaining its deferred Gluon
        # annotations. Its GPU body must never execute under the Ascend adapter.
        definition = ast.Module(
            body=[
                ast.ImportFrom(
                    module="__future__", names=[ast.alias(name="annotations")], level=0
                ),
                kernel,
            ],
            type_ignores=[],
        )
        exec(compile(ast.fix_missing_locations(definition), str(path), "exec"), adapter)
        decorated = adapter["_simt_gemm_kernel"]
        original = decorated.fn
        options = {"do_not_specialize": ["M"], "launch_metadata": object()}
        configured = adapter["gluon"].jit(**options)(original)
        assert configured.fn is original
        assert configured.options == options
        for candidate in (decorated, configured, adapter["gluon"].jit()(original)):
            with subtests.test(options=candidate.options):
                with pytest.raises(RuntimeError, match="Gluon.*Ascend"):
                    candidate()
                with pytest.raises(RuntimeError, match="Gluon.*Ascend"):
                    candidate[(1,)]()

    def test_gpu_keeps_native_gluon_provider(self):
        adapter, modules = _load_adapter(is_npu=False)
        assert adapter["triton"] is modules["tokenspeed_triton"]
        assert adapter["gluon"] is modules["tokenspeed_triton.experimental.gluon"]
        assert adapter["gl"] is modules["tokenspeed_triton.experimental.gluon.language"]


def test_kernel_sources_use_tokenspeed_triton_adapter():
    source_root = ROOT / "tokenspeed-kernel/python/tokenspeed_kernel"
    adapter = source_root / "_triton.py"
    # These helpers intentionally interact with framework-owned stock Triton;
    # neither defines TokenSpeed Triton kernels.
    stock_triton_interop = {
        source_root / "ops/attention/kda/_triton/fla.py",
        source_root / "thirdparty/msa/cute/src/common/cute_dsl_utils.py",
    }
    stock_triton_roots = {
        source_root / "thirdparty/gluon_petit",
    }
    direct_imports = {}

    for path in source_root.rglob("*.py"):
        if (
            path == adapter
            or path in stock_triton_interop
            or any(root in path.parents for root in stock_triton_roots)
        ):
            continue
        tree = ast.parse(path.read_text(), filename=str(path))
        lines = []
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                names = (alias.name for alias in node.names)
            elif isinstance(node, ast.ImportFrom) and node.module is not None:
                names = (node.module,)
            else:
                continue
            if any(
                name == "triton"
                or name.startswith("triton.")
                or name == "tokenspeed_triton"
                or name.startswith("tokenspeed_triton.")
                for name in names
            ):
                lines.append(node.lineno)
        if lines:
            direct_imports[str(path.relative_to(ROOT))] = lines

    assert direct_imports == {}, (
        "TokenSpeed kernel sources must import Triton through "
        f"tokenspeed_kernel._triton: {direct_imports}"
    )
