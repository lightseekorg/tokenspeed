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

import importlib.util
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace


def test_triton_adapter_allows_optional_modules_to_be_missing(monkeypatch) -> None:
    triton = ModuleType("triton")
    triton.jit = lambda function: function
    language = ModuleType("triton.language")
    language.extra = SimpleNamespace()
    triton.language = language
    extra = ModuleType("triton.language.extra")
    extra.libdevice = object()
    monkeypatch.setitem(sys.modules, "triton", triton)
    monkeypatch.setitem(sys.modules, "triton.language", language)
    monkeypatch.setitem(sys.modules, "triton.language.extra", extra)

    path = Path(__file__).parents[1] / "python" / "tokenspeed_kernel_npu" / "_triton.py"
    spec = importlib.util.spec_from_file_location("_test_npu_triton", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    assert module.proton is None
    assert module.tl.extra.cuda.gdc_wait is module._unsupported_pdl_noop
    assert module.tl.extra.cuda.gdc_launch_dependents is module._unsupported_pdl_noop
