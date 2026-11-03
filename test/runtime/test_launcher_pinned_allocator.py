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

"""Exercise launcher defaults against the real installed allocator parser."""

import os
import subprocess
import sys

import pytest

from tokenspeed.runtime.utils.launcher import set_pinned_allocator_default


@pytest.fixture
def allocator_env(monkeypatch):
    for key in (
        "PYTORCH_CUDA_ALLOC_CONF",
        "PYTORCH_HIP_ALLOC_CONF",
        "PYTORCH_ALLOC_CONF",
    ):
        monkeypatch.delenv(key, raising=False)


def test_default_is_accepted_by_installed_torch(allocator_env, monkeypatch):
    monkeypatch.setenv("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:False")
    set_pinned_allocator_default()
    assert os.environ["PYTORCH_CUDA_ALLOC_CONF"].startswith("expandable_segments:False")
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            "import os,torch; torch.cuda.memory._set_allocator_settings(os.environ['PYTORCH_CUDA_ALLOC_CONF']); assert not torch.cuda.is_initialized()",
        ],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr


def test_explicit_threshold_is_preserved(allocator_env, monkeypatch):
    # Explicit options remain the user's responsibility, even on older builds.
    value = "expandable_segments:False,pinned_max_round_threshold_mb:2048"
    monkeypatch.setenv("PYTORCH_ALLOC_CONF", value)
    set_pinned_allocator_default()
    assert os.environ["PYTORCH_ALLOC_CONF"] == value


def test_effective_legacy_variable_keeps_precedence(allocator_env, monkeypatch):
    monkeypatch.setenv("PYTORCH_CUDA_ALLOC_CONF", "pinned_max_round_threshold_mb:2048")
    monkeypatch.setenv("PYTORCH_ALLOC_CONF", "expandable_segments:False")
    set_pinned_allocator_default()
    assert os.environ["PYTORCH_ALLOC_CONF"] == "expandable_segments:False"
    assert os.environ["PYTORCH_CUDA_ALLOC_CONF"] == "pinned_max_round_threshold_mb:2048"
