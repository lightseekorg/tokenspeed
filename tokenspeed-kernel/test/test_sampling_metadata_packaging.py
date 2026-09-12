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

"""CPU tests for optional extension compatibility and package build metadata."""

import ast
import hashlib
import json
import runpy
import shutil
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import patch

from setuptools import Distribution

ROOT = Path(__file__).resolve().parents[1] / "python"
LOADER = ROOT / "tokenspeed_kernel/thirdparty/cuda/sampling_metadata.py"


class MetadataPackagingTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.directory = Path(self.temp.name)
        self.loader = self.directory / "sampling_metadata.py"
        shutil.copy2(LOADER, self.loader)
        self.output = self.directory / "objs/sampling_metadata"
        self.output.mkdir(parents=True)
        self.torch = types.ModuleType("torch")
        self.torch.__version__ = "2.13.0+cu130"

    def load(self, spec):
        with patch.dict(sys.modules, {"torch": self.torch}), patch(
            "importlib.util.spec_from_file_location", spec
        ):
            return runpy.run_path(str(self.loader))

    def record(self):
        binary = self.output / "_sampling_metadata.so"
        binary.write_bytes(b"binary fixture; never executed")
        record = {
            "torch": self.torch.__version__,
            "python": list(sys.version_info[:2]),
            "sha256": hashlib.sha256(binary.read_bytes()).hexdigest(),
        }
        (self.output / "build.json").write_text(json.dumps(record))
        return record

    def unavailable(self):
        def forbidden(*args, **kwargs):
            raise AssertionError("incompatible binary must not be loaded")

        module = self.load(forbidden)
        self.assertIsNone(module["NativeLaunchPair"])
        self.assertIsNone(module["validate_metadata"](*([None] * 8)))

    def test_source_install_without_native_binary_retains_fallback(self):
        self.unavailable()

    def test_build_abi_and_integrity_mismatch_never_loads_binary(self):
        for field, value in (
            ("torch", "different"),
            ("python", [1, 2]),
            ("sha256", "invalid"),
        ):
            with self.subTest(field=field):
                record = self.record()
                record[field] = value
                (self.output / "build.json").write_text(json.dumps(record))
                self.unavailable()
        for value in ("[]", "not-json"):
            with self.subTest(record=value):
                (self.output / "build.json").write_text(value)
                self.unavailable()

    def test_compatible_binary_preserves_native_bound_method(self):
        self.record()
        marker = object()

        class Validator:
            def validate_metadata(self, *args):
                return marker

        native = types.ModuleType("_sampling_metadata")
        native.MetadataValidator = Validator
        native.NativeLaunchPair = object()

        class Loader:
            def exec_module(self, module):
                self.module = module

        loader = Loader()
        spec = types.SimpleNamespace(loader=loader)
        with patch("importlib.util.module_from_spec", return_value=native):
            module = self.load(lambda *args: spec)
        self.assertIs(module["NativeLaunchPair"], native.NativeLaunchPair)
        self.assertIs(module["validate_metadata"](*([None] * 8)), marker)
        self.assertIs(loader.module, native)

    def test_dynamic_loader_failure_preserves_fallback(self):
        self.record()

        def broken(*args, **kwargs):
            raise ImportError("incompatible optional shared object")

        module = self.load(broken)
        self.assertIsNone(module["validate_metadata"](*([None] * 8)))

    def test_build_records_actual_binary_abi_and_avoids_gpu_compilation(self):
        tree = ast.parse((ROOT / "setup.py").read_text())
        fn = next(
            n
            for n in tree.body
            if isinstance(n, ast.FunctionDef) and n.name == "_build_sampling_metadata"
        )
        compiled = self.directory / "compiled.so"
        compiled.write_bytes(b"compiler fixture output")
        calls = []

        def compile_host(**kwargs):
            calls.append(kwargs)
            return types.SimpleNamespace(__file__=str(compiled))

        utils = types.ModuleType("torch.utils")
        cpp = types.ModuleType("torch.utils.cpp_extension")
        cpp.load = compile_host
        namespace = {
            "ROOT": self.directory,
            "CUDA_OBJS_DIR": self.directory / "objects",
            "CUDA_CSRC_DIR": self.directory / "csrc",
            "CUDA_HOME": "/selected/cuda",
            "Path": Path,
            "shutil": shutil,
            "json": json,
            "hashlib": hashlib,
            "sys": sys,
        }
        exec(
            compile(ast.Module(body=[fn], type_ignores=[]), "setup.py", "exec"),
            namespace,
        )
        with patch.dict(
            sys.modules,
            {
                "torch": self.torch,
                "torch.utils": utils,
                "torch.utils.cpp_extension": cpp,
            },
        ):
            namespace["_build_sampling_metadata"](False)
        record = json.loads(
            (self.directory / "objects/sampling_metadata/build.json").read_text()
        )
        self.assertEqual(
            record["sha256"], hashlib.sha256(compiled.read_bytes()).hexdigest()
        )
        self.assertEqual(record["torch"], self.torch.__version__)
        self.assertEqual(record["python"], list(sys.version_info[:2]))
        self.assertEqual(len(calls), 1)
        self.assertFalse(calls[0]["with_cuda"])
        self.assertEqual(calls[0]["extra_include_paths"], ["/selected/cuda/include"])
        self.assertEqual(calls[0]["extra_ldflags"], ["-ldl"])
        self.assertFalse(
            Path(calls[0]["build_directory"]).is_relative_to(self.directory / "objects")
        )

    def test_only_cuda_wheel_requires_cpython_platform_tag(self):
        tree = ast.parse((ROOT / "setup.py").read_text())
        cls = next(
            n
            for n in tree.body
            if isinstance(n, ast.ClassDef) and n.name == "KernelDistribution"
        )
        for backend, expected in (("cuda", True), ("rocm", False)):
            with self.subTest(backend=backend):
                namespace = {
                    "Distribution": Distribution,
                    "_selected_backend": lambda: backend,
                }
                exec(
                    compile(
                        ast.Module(body=[cls], type_ignores=[]), "setup.py", "exec"
                    ),
                    namespace,
                )
                self.assertEqual(
                    namespace["KernelDistribution"]().has_ext_modules(), expected
                )


if __name__ == "__main__":
    unittest.main()
