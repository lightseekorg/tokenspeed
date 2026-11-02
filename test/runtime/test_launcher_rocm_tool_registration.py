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

"""ROCm worker processes default to no profiler-tool registration with ROCr.

ROCm 7.2's queue interception, active once any tool registers, corrupts the
host heap under deep dispatch backlogs; see
``launcher.set_rocm_tool_registration_default``. CPU-only: the decision reads
the torch build and the environment, nothing else.
"""

from __future__ import annotations

import os
import sys

import pytest
import torch

# CI Registration (parsed via AST, runtime no-op)
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from ci_system.ci_register import register_cuda_ci  # noqa: E402

from tokenspeed.runtime.utils import launcher  # noqa: E402

register_cuda_ci(est_time=5, suite="runtime-1gpu")

KEY = "HSA_TOOLS_DISABLE_REGISTER"


@pytest.mark.parametrize(
    "hip, environ, expected",
    [
        ("7.2", {}, "1"),
        ("7.2", {KEY: "0"}, "0"),
        ("7.2", {"ROCP_TOOL_LIBRARIES": "librocprofiler-sdk-tool.so"}, None),
        (None, {}, None),
    ],
    ids=["rocm", "explicit-keep", "attached-tool", "cuda"],
)
def test_rocm_tool_registration_default(monkeypatch, hip, environ, expected):
    monkeypatch.setattr(torch.version, "hip", hip)
    for name in (KEY, "ROCP_TOOL_LIBRARIES"):
        monkeypatch.delenv(name, raising=False)
    for name, value in environ.items():
        monkeypatch.setenv(name, value)

    launcher.set_rocm_tool_registration_default()

    assert os.environ.get(KEY) == expected
