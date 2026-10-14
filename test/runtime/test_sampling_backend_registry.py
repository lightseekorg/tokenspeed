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

"""Regression tests for sampling backend registry defaults."""

import os
import sys
from types import SimpleNamespace
from unittest import mock

import pytest

# CI Registration (parsed via AST, runtime no-op)
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from ci_system.ci_register import register_cuda_ci  # noqa: E402

register_cuda_ci(est_time=10, suite="runtime-1gpu")

from tokenspeed.runtime.sampling import registry  # noqa: E402


class TestSamplingBackendRegistryDefaults:
    def test_unresolved_default_uses_flashinfer_on_nvidia(self):
        platform = SimpleNamespace(is_nvidia=True, is_npu=False)
        server_args = SimpleNamespace(sampling_backend=None)

        with mock.patch.object(registry, "current_platform", return_value=platform):
            assert (registry._resolve_backend_name(server_args)) == ("flashinfer")

    def test_unresolved_default_uses_greedy_on_non_nvidia_non_npu(self):
        platform = SimpleNamespace(is_nvidia=False, is_npu=False)
        server_args = SimpleNamespace(sampling_backend=None)

        with mock.patch.object(registry, "current_platform", return_value=platform):
            assert (registry._resolve_backend_name(server_args)) == ("greedy")

    def test_unresolved_default_uses_ascend_full_on_npu(self):
        platform = SimpleNamespace(is_nvidia=False, is_npu=True)
        server_args = SimpleNamespace(sampling_backend=None)

        with mock.patch.object(registry, "current_platform", return_value=platform):
            assert (registry._resolve_backend_name(server_args)) == ("ascend_full")

    def test_explicit_backend_is_preserved(self):
        server_args = SimpleNamespace(sampling_backend="flashinfer")

        with mock.patch.object(
            registry,
            "current_platform",
            side_effect=AssertionError("unexpected lookup"),
        ):
            assert (registry._resolve_backend_name(server_args)) == ("flashinfer")


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, *sys.argv[1:]]))
