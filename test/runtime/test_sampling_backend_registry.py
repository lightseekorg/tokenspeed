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
        platform = SimpleNamespace(is_nvidia=True)
        server_args = SimpleNamespace(sampling_backend=None)

        with mock.patch.object(registry, "current_platform", return_value=platform):
            assert (registry._resolve_backend_name(server_args)) == ("flashinfer")

    def test_unresolved_default_uses_greedy_on_non_nvidia(self):
        platform = SimpleNamespace(is_nvidia=False)
        server_args = SimpleNamespace(sampling_backend=None)

        with mock.patch.object(registry, "current_platform", return_value=platform):
            assert (registry._resolve_backend_name(server_args)) == ("greedy")

    def test_explicit_backend_is_preserved(self):
        platform = SimpleNamespace(is_nvidia=False)
        server_args = SimpleNamespace(sampling_backend="flashinfer")

        with mock.patch.object(registry, "current_platform", return_value=platform):
            assert (registry._resolve_backend_name(server_args)) == ("flashinfer")


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, *sys.argv[1:]]))
