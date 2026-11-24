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

"""Discovering GPU-only models must not prevent Ascend startup."""

import subprocess
import sys

import pytest
import torch

pytestmark = pytest.mark.skipif(
    not hasattr(torch, "npu") or not torch.npu.is_available(), reason="requires Ascend"
)


@pytest.mark.parametrize("module", ["inkling2_monokernel", "registry"])
def test_model_discovery_on_npu(module):
    # Use a fresh interpreter so cached model imports cannot hide a regression.
    code = f"""
import importlib

importlib.import_module("tokenspeed.runtime.models.{module}")
from tokenspeed.runtime.models.inkling2 import Inkling2ForConditionalGeneration
from tokenspeed.runtime.models.registry import ModelRegistry

assert ModelRegistry.resolve_model_cls("Inkling2ForConditionalGeneration") == (
    Inkling2ForConditionalGeneration,
    "Inkling2ForConditionalGeneration",
)
"""
    result = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, timeout=120
    )
    assert result.returncode == 0, result.stdout + result.stderr
