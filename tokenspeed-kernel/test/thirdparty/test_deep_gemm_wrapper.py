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
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

from __future__ import annotations

import importlib
import sys
import types

import pytest


def test_deep_gemm_wrapper_tolerates_missing_optional_symbols(monkeypatch):
    wrapper_name = "tokenspeed_kernel.thirdparty.deep_gemm"
    package_name = "tokenspeed_kernel.thirdparty"
    old_wrapper = sys.modules.pop(wrapper_name, None)
    old_package_attr = getattr(sys.modules.get(package_name), "deep_gemm", None)

    fake_deep_gemm = types.ModuleType("deep_gemm")
    fake_deep_gemm.ceil_div = object()
    fake_deep_gemm.fp8_paged_mqa_logits = object()
    fake_deep_gemm.get_num_sms = object()
    monkeypatch.setitem(sys.modules, "deep_gemm", fake_deep_gemm)

    try:
        wrapper = importlib.import_module(wrapper_name)

        assert wrapper.ceil_div is fake_deep_gemm.ceil_div
        assert wrapper.fp8_paged_mqa_logits is fake_deep_gemm.fp8_paged_mqa_logits
        assert wrapper.get_num_sms is fake_deep_gemm.get_num_sms
        assert "fp8_einsum" not in wrapper.__all__
        with pytest.raises(AttributeError, match="fp8_einsum"):
            wrapper.fp8_einsum
    finally:
        sys.modules.pop(wrapper_name, None)
        if old_wrapper is not None:
            sys.modules[wrapper_name] = old_wrapper
            package = sys.modules.get(package_name)
            if package is not None:
                setattr(package, "deep_gemm", old_wrapper)
        elif old_package_attr is not None:
            package = sys.modules.get(package_name)
            if package is not None:
                setattr(package, "deep_gemm", old_package_attr)
