"""Regression tests for sampling backend registry defaults."""

import os
import sys
import unittest
from types import SimpleNamespace
from unittest import mock

# CI Registration (parsed via AST, runtime no-op)
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from ci_system.ci_register import register_cuda_ci  # noqa: E402

register_cuda_ci(est_time=10, suite="runtime-1gpu")

from tokenspeed.runtime.sampling import registry  # noqa: E402
from tokenspeed.runtime.utils import server_args as server_args_module  # noqa: E402


class TestSamplingBackendRegistryDefaults(unittest.TestCase):
    def test_unresolved_default_uses_flashinfer_on_nvidia(self):
        platform = SimpleNamespace(is_nvidia=True, is_amd=False)
        server_args = SimpleNamespace(sampling_backend=None)

        with mock.patch.object(registry, "current_platform", return_value=platform):
            self.assertEqual(registry._resolve_backend_name(server_args), "flashinfer")

    def test_unresolved_default_uses_triton_on_amd(self):
        platform = SimpleNamespace(is_nvidia=False, is_amd=True)
        server_args = SimpleNamespace(sampling_backend=None)

        with mock.patch.object(registry, "current_platform", return_value=platform):
            self.assertEqual(registry._resolve_backend_name(server_args), "triton")

    def test_unresolved_default_uses_greedy_on_other_platforms(self):
        platform = SimpleNamespace(is_nvidia=False, is_amd=False)
        server_args = SimpleNamespace(sampling_backend=None)

        with mock.patch.object(registry, "current_platform", return_value=platform):
            self.assertEqual(registry._resolve_backend_name(server_args), "greedy")

    def test_explicit_backend_is_preserved(self):
        platform = SimpleNamespace(is_nvidia=False, is_amd=True)
        server_args = SimpleNamespace(sampling_backend="flashinfer")

        with mock.patch.object(registry, "current_platform", return_value=platform):
            self.assertEqual(registry._resolve_backend_name(server_args), "flashinfer")

    def test_server_args_resolves_amd_default_to_triton(self):
        platform = SimpleNamespace(is_nvidia=False, is_amd=True)
        args = SimpleNamespace(sampling_backend=None)

        with mock.patch.object(
            server_args_module, "current_platform", return_value=platform
        ):
            server_args_module.ServerArgs.resolve_kernel_backends(args)

        self.assertEqual(args.sampling_backend, "triton")
