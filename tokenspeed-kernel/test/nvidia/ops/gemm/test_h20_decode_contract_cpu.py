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

"""CPU dispatch contracts using the real candidate and fake vendor dependencies.

Run this file directly with Python; no pytest, conftest, Torch, or GPU is needed.
The candidate and CPU registry/signature/selection modules execute unchanged.
Platform data classes and the generic entry, selector, signature constants and
original registration decorators execute from their actual source ASTs. Torch,
platform discovery, vendor drivers and kernel launches remain fake; this does
not exercise the complete Triton module import, GPU numerics or performance.
"""

from __future__ import annotations

import ast
import contextlib
import functools
import importlib.util
import itertools
import math
import os
import sys
import types
import unittest
from collections.abc import Mapping
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from unittest.mock import patch

KERNEL_ROOT = Path(__file__).resolve().parents[4]
GEMM_SOURCE = KERNEL_ROOT / "python/tokenspeed_kernel/ops/gemm"
CANDIDATE_SOURCE = GEMM_SOURCE / "_cute_dsl/h20_decode.py"

# Independently stated contract; do not derive expected cases from the candidate.
EXPECTED_ROUTES = {
    (1, 2624, 6144): "ll_bf16",
    (1, 4096, 2048): "skinny",
    (1, 160, 6144): "ll_bf16",
    (1, 512, 2048): "ll_bf16",
    (1, 512, 6144): "ll_bf16",
    (1, 128, 6144): "ll_bf16",
    (1, 1152, 6144): "ll_bf16",
}
BF16 = "fake_bfloat16"
FP32 = "fake_float32"


@dataclass(frozen=True)
class FakeDevice:
    type: str = "cuda"
    index: int | None = 0


@dataclass(frozen=True)
class FakeStorage:
    pointer: int

    def data_ptr(self):
        return self.pointer


class FakeTensor:
    _pointers = itertools.count(4096, 4096)

    def __init__(
        self,
        shape,
        *,
        dtype=BF16,
        device=FakeDevice(),
        contiguous=True,
        pointer=None,
        storage_pointer=None,
        strides=None,
    ):
        self.shape = tuple(shape)
        self.ndim = len(self.shape)
        self.dtype = dtype
        self.device = device
        self.is_cuda = device.type == "cuda"
        self.contiguous = contiguous
        self.pointer = next(self._pointers) if pointer is None else pointer
        self.storage_pointer = (
            self.pointer if storage_pointer is None else storage_pointer
        )
        if strides is None:
            computed = []
            stride = 1
            for dimension in reversed(self.shape):
                computed.append(stride)
                stride *= dimension
            self.strides = tuple(reversed(computed))
        else:
            self.strides = tuple(strides)

    def is_contiguous(self):
        return self.contiguous

    def data_ptr(self):
        return self.pointer

    def stride(self, dim=None):
        return self.strides if dim is None else self.strides[dim]

    def numel(self):
        return math.prod(self.shape)

    def untyped_storage(self):
        return FakeStorage(self.storage_pointer)

    def detach(self):
        return self

    def t(self):
        return FakeTensor(
            tuple(reversed(self.shape)), dtype=self.dtype, device=self.device
        )

    def float(self):
        return FakeTensor(self.shape, dtype=FP32, device=self.device)

    def view(self, *shape):
        if shape == (-1,):
            shape = (self.numel(),)
        return FakeTensor(
            shape,
            dtype=self.dtype,
            device=self.device,
            pointer=self.pointer,
            storage_pointer=self.storage_pointer,
        )

    def __matmul__(self, other):
        return FakeTensor(
            (self.shape[0], other.shape[1]), dtype=self.dtype, device=self.device
        )


@dataclass(frozen=True)
class FakeSkinnyConfig:
    num_rows: int
    block_size: int
    outputs_per_block: int
    k_unroll: int
    vector_width: int


class FakeDriver:
    def __init__(self, backend):
        self.backend = backend
        self.events = []
        self.supports_result = True
        self.available = True
        self.fail_call = False

    def supports(self, *args):
        self.events.append(("supports", args))
        return self.supports_result

    def is_available(self):
        self.events.append(("available",))
        return self.available

    def __call__(self, *args, **kwargs):
        self.events.append(("call", args, kwargs))
        if self.fail_call:
            raise RuntimeError("fake vendor launch failed")
        x, weight = args[:2]
        out = args[2] if self.backend == "ll_bf16" else kwargs["out"]
        return (
            out
            if out is not None
            else FakeTensor((x.shape[0], weight.shape[0]), device=x.device)
        )

    def calls(self):
        return [event for event in self.events if event[0] == "call"]


class CandidateHarness:
    """Inject only dependency interfaces; load the complete production module."""

    def __init__(self):
        self.serving = False
        self.capturing = False
        self.pdl = False
        self.current_device = 0
        self.capture_by_device = {}
        self.devices = {0: ((9, 0), "NVIDIA H20"), 1: ((9, 0), "H20")}
        self.drivers = []
        self.registrations = []
        self.fallback_events = []
        self.torch = types.ModuleType("torch")
        self.torch.Tensor = FakeTensor
        self.torch.bfloat16 = BF16
        self.torch.float32 = FP32
        self.torch.cuda = types.SimpleNamespace(
            get_device_capability=lambda index: self.devices[index][0],
            get_device_name=lambda index: self.devices[index][1],
            current_device=lambda: self.current_device,
            is_current_stream_capturing=lambda: self.capture_by_device.get(
                self.current_device, self.capturing
            ),
            device=self.device_context,
        )
        self.torch.mm = self.mm
        self.torch.empty = lambda *shape, dtype, device: FakeTensor(
            shape, dtype=dtype, device=device
        )
        self.torch.promote_types = lambda left, right: left if left == right else FP32
        functional = types.ModuleType("torch.nn.functional")
        functional.linear = self.linear
        nn = types.ModuleType("torch.nn")
        nn.functional = functional
        self.torch.nn = nn
        modules = {
            "torch": self.torch,
            "torch.nn": nn,
            "torch.nn.functional": functional,
        }
        dependencies = {
            "tokenspeed_kernel.compile_monitor": {"is_serving": lambda: self.serving},
            "tokenspeed_kernel.platform": {
                "ArchVersion": lambda *parts: parts,
                "CapabilityRequirement": lambda **fields: fields,
                "pdl_enabled": lambda: self.pdl,
            },
            "tokenspeed_kernel.registry": {
                "Priority": types.SimpleNamespace(SPECIALIZED=100),
                "register_kernel": self.register,
            },
            "tokenspeed_kernel.signature": {
                "dense_tensor_format": lambda dtype: dtype,
                "format_signature": lambda **fields: tuple(sorted(fields.items())),
            },
            "tokenspeed_kernel.thirdparty.cute_dsl.ll_bf16": {
                "LLBf16Router": lambda: self.driver("ll_bf16")
            },
            "tokenspeed_kernel.thirdparty.cute_dsl.skinny_gemm": {
                "ShapeDynamicSkinnyGemm": lambda: self.driver("skinny"),
                "SkinnyGemmConfig": FakeSkinnyConfig,
            },
        }
        for name, fields in dependencies.items():
            module = types.ModuleType(name)
            module.__dict__.update(fields)
            modules[name] = module
            components = name.split(".")
            for length in range(1, len(components)):
                parent = ".".join(components[:length])
                if parent not in modules:
                    modules[parent] = types.ModuleType(parent)
                    modules[parent].__path__ = []
        self.modules = modules
        spec = importlib.util.spec_from_file_location(
            "_h20_decode_cpu_contract_candidate", CANDIDATE_SOURCE
        )
        self.module = importlib.util.module_from_spec(spec)
        with patch.dict(sys.modules, modules):
            spec.loader.exec_module(self.module)

    def driver(self, backend):
        driver = FakeDriver(backend)
        self.drivers.append(driver)
        return driver

    @contextmanager
    def device_context(self, device):
        previous = self.current_device
        index = device.index if isinstance(device, FakeDevice) else device
        self.current_device = previous if index is None else index
        try:
            yield
        finally:
            self.current_device = previous

    def register(self, *args, **kwargs):
        def decorate(function):
            self.registrations.append((args, kwargs, function))
            return function

        return decorate

    def linear(self, x, weight):
        self.fallback_events.append(("linear", x, weight))
        return FakeTensor((x.shape[0], weight.shape[0]), dtype=x.dtype, device=x.device)

    def mm(self, x, weight, *, out):
        self.fallback_events.append(("mm", x, weight, out))
        return out

    def vendor_calls(self):
        return [event for driver in self.drivers for event in driver.calls()]

    def vendor_events(self):
        return [event for driver in self.drivers for event in driver.events]

    def generic_entry_function(self):
        """Execute the real function AST, not the Triton-owning module import."""
        path = GEMM_SOURCE / "triton_gemv.py"
        tree = ast.parse(path.read_text())
        function = next(
            node
            for node in tree.body
            if isinstance(node, ast.FunctionDef) and node.name == "use_decode_gemv"
        )
        module = ast.Module(body=[function], type_ignores=[])
        namespace = {
            "torch": self.torch,
            "h20_decode_supported": self.module.h20_decode_supported,
            "flashinfer_joint_bf16_supported": lambda *args: False,
            "BF16_GEMM_MAX_M": 32,
            "is_serving": lambda: self.serving,
            "current_platform": lambda: types.SimpleNamespace(
                is_cdna4=False, is_cdna5=False
            ),
        }
        exec(compile(module, str(path), "exec"), namespace)
        return namespace["use_decode_gemv"]


class FakeRowCtaKernel:
    def __init__(self):
        self.launches = []

    def __getitem__(self, grid):
        def launch(*args, **kwargs):
            self.launches.append((grid, args, kwargs))

        return launch


class ActualSelectionHarness(CandidateHarness):
    """Execute real CPU selection; fake only device discovery and kernel work."""

    def __init__(self):
        super().__init__()
        self.rowcta_kernel = FakeRowCtaKernel()
        self.selected_impls = []
        self.autotune_calls = []
        self.platform = None
        package = KERNEL_ROOT / "python/tokenspeed_kernel"
        amd_name = "tokenspeed_kernel_amd.ops.gfx1250.gemm.fp16.mm"
        amd_policy = types.ModuleType(amd_name)
        self.execute_ast(
            KERNEL_ROOT.parent
            / "tokenspeed-kernel-amd/python/tokenspeed_kernel_amd/ops/gfx1250/gemm/fp16/mm.py",
            amd_policy.__dict__,
            {
                "_DENSE_MAX_M",
                "_DENSE_M64_MIN_M",
                "_DENSE_M64_MAX_N",
                "_use_dense_m64_tile",
                "use_gluon_wmma_dense_gfx1250",
            },
        )
        self.modules[amd_name] = amd_policy
        with patch.dict(sys.modules, self.modules):
            platform = types.ModuleType("tokenspeed_kernel.platform")
            platform.__dict__.update(
                dataclass=dataclass,
                field=field,
                Mapping=Mapping,
                contextlib=contextlib,
                torch=self.torch,
                current_platform=lambda: self.platform,
                pdl_enabled=lambda: self.pdl,
            )
            sys.modules[platform.__name__] = platform
            self.execute_ast(
                package / "platform.py",
                platform.__dict__,
                {
                    "ArchVersion",
                    "InterconnectInfo",
                    "PlatformInfo",
                    "CapabilityRequirement",
                },
            )
            self.actual_platform = platform
            self.signature = self.load_cpu_module("signature", package)
            self.registry = self.load_cpu_module("registry", package)
            self.selection = self.load_cpu_module("selection", package)
            self.registry.KernelRegistry.reset()
            self.generic = {
                "__name__": "_actual_decode_cpu_functions",
                "torch": self.torch,
                "functools": functools,
                "ArchVersion": platform.ArchVersion,
                "CapabilityRequirement": platform.CapabilityRequirement,
                "KernelRegistry": self.registry.KernelRegistry,
                "Priority": self.registry.Priority,
                "register_kernel": self.registry.register_kernel,
                "dense_tensor_format": self.signature.dense_tensor_format,
                "format_signature": self.signature.format_signature,
                "spec_matches_traits": self.selection.spec_matches_traits,
                "spec_matches_shape_traits": self.selection.spec_matches_shape_traits,
                "current_platform": lambda: self.platform,
                "is_serving": lambda: self.serving,
                "flashinfer_joint_bf16_supported": lambda *args: False,
                "BF16_GEMM_MAX_M": 32,
                "autotune_bf16_gemm": lambda *args: self.autotune_calls.append(args),
                "triton_bf16x3_gemm_fp32": lambda *args: None,
                "_rowcta_gemv_kernel": self.rowcta_kernel,
            }
            self.execute_ast(
                GEMM_SOURCE / "triton_gemv.py",
                self.generic,
                {
                    "_BF16_SIG",
                    "_FP32_SIG",
                    "_BF16_FP32_SIG",
                    "_SIGNATURES",
                    "triton_rowcta_gemv",
                    "_gfx1250_wmma_dense_problem",
                    "gluon_wmma_dense_gemv_gfx1250",
                    "torch_decode_gemv",
                    "_select",
                    "use_decode_gemv",
                    "decode_gemv",
                },
            )
            registry = self.registry.KernelRegistry.get()
            self.specs_before_helper = tuple(registry.list_kernels())
            self.drivers = []
            spec = importlib.util.spec_from_file_location(
                "_h20_actual_selection_candidate", CANDIDATE_SOURCE
            )
            self.module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(self.module)
            self.specs_after_helper = tuple(registry.list_kernels())
        self.generic.update(
            h20_decode_supported=self.module.h20_decode_supported,
            cute_dsl_h20_decode=self.module.cute_dsl_h20_decode,
        )
        self.select = self.generic["_select"]
        self.generic["_select"] = self.record_selection
        self.set_platform("NVIDIA H20", vendor="nvidia", arch=(9, 0))

    @staticmethod
    def execute_ast(path, namespace, names):
        tree = ast.parse(path.read_text())
        nodes = []
        found = set()
        for node in tree.body:
            if isinstance(node, ast.ImportFrom) and node.module == "__future__":
                nodes.append(node)
            elif (
                isinstance(node, (ast.FunctionDef, ast.ClassDef)) and node.name in names
            ):
                nodes.append(node)
                found.add(node.name)
            elif isinstance(node, ast.Assign):
                targets = {
                    target.id for target in node.targets if isinstance(target, ast.Name)
                }
                if targets & names:
                    nodes.append(node)
                    found.update(targets & names)
        if found != names:
            raise AssertionError(
                f"Missing actual source nodes in {path}: {names - found}"
            )
        exec(
            compile(ast.Module(body=nodes, type_ignores=[]), str(path), "exec"),
            namespace,
        )

    @staticmethod
    def load_cpu_module(name, package):
        spec = importlib.util.spec_from_file_location(
            f"tokenspeed_kernel.{name}", package / f"{name}.py"
        )
        module = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = module
        spec.loader.exec_module(module)
        return module

    def set_platform(self, name, *, vendor, arch):
        self.platform = self.actual_platform.PlatformInfo(
            vendor=vendor,
            arch_version=self.actual_platform.ArchVersion(*arch),
            device_name=name,
            device_count=1,
            total_memory=0,
            memory_bandwidth=0,
            sm_count=0,
            max_threads_per_sm=0,
            max_shared_memory_per_sm=0,
        )
        self.devices[0] = (arch, name)
        self.current_device = 0
        self.module._is_h20_device.cache_clear()
        # This fixture changes emulated hardware. The real selector caches no
        # platform key, so fixture transitions must clear its separate cache.
        self.select.cache_clear()
        self.registry.KernelRegistry.get().clear_cache()

    def record_selection(self, *args):
        implementation = self.select(*args)
        self.selected_impls.append(implementation)
        return implementation

    def select_shape(self, shape, *, h20_eligible):
        with patch.dict(sys.modules, self.modules):
            return self.select(*shape, True, BF16, BF16, False, h20_eligible)

    def register_candidate_fault(self, *, extra_traits):
        """Inject a broad SM90 priority-14 decorator through real registry."""
        capability = self.actual_platform.CapabilityRequirement(
            vendors=frozenset({"nvidia"}),
            min_arch_version=self.actual_platform.ArchVersion(9, 0),
            max_arch_version=self.actual_platform.ArchVersion(9, 0),
        )
        self.registry.register_kernel(
            "gemm",
            "decode_gemv",
            name="cute_dsl_h20_decode",
            solution="cute_dsl",
            capability=capability,
            signatures=self.generic["_BF16_SIG"],
            traits={
                "m": frozenset({1}),
                "mnk_problem_filter": frozenset({self.module._shape_supported}),
                **extra_traits,
            },
            priority=self.registry.Priority.SPECIALIZED + 2,
        )(self.module.cute_dsl_h20_decode)
        self.select.cache_clear()
        return self.registry.KernelRegistry.get().get_by_name("cute_dsl_h20_decode")


class H20DecodeContractCPU(unittest.TestCase):
    def setUp(self):
        self.env = CandidateHarness()
        self.module = self.env.module

    def inputs(self, shape=(1, 512, 6144), *, device=FakeDevice()):
        m, n, k = shape
        return FakeTensor((m, k), device=device), FakeTensor((n, k), device=device)

    def test_exact_seven_shapes_and_no_global_candidate_registration(self):
        self.assertEqual(dict(self.module.H20_DECODE_ROUTES), EXPECTED_ROUTES)
        self.assertEqual(self.env.registrations, [])
        predicate = self.module._shape_supported
        for shape in EXPECTED_ROUTES:
            self.assertTrue(predicate(*shape))
        self.assertFalse(predicate(1, 4096, 6144))

    def test_each_shape_calls_its_expected_vendor(self):
        for shape, backend in EXPECTED_ROUTES.items():
            with self.subTest(shape=shape):
                x, weight = self.inputs(shape)
                self.assertTrue(self.module.h20_decode_supported(x, weight))
                result = self.module.cute_dsl_h20_decode(x, weight)
                selected = next(driver for driver in self.env.drivers if driver.calls())
                self.assertEqual(selected.backend, backend)
                self.assertEqual(len(self.env.vendor_calls()), 1)
                self.assertEqual(result.shape, (1, shape[1]))
                self.assertEqual(result.dtype, BF16)
                for driver in self.env.drivers:
                    driver.events.clear()
        self.assertFalse(self.env.fallback_events)

    def test_m_boundaries_refuse_and_fallback(self):
        for m in (0, 2, 4, 8, 16, 32, 33):
            with self.subTest(m=m):
                x, weight = self.inputs((m, 512, 6144))
                self.assertFalse(self.module.h20_decode_supported(x, weight))
                self.module.cute_dsl_h20_decode(x, weight)
        self.assertFalse(self.env.vendor_events())
        self.assertEqual(len(self.env.fallback_events), 7)

    def test_m_zero_with_null_storage_output_falls_back_without_alias_error(self):
        x = FakeTensor((0, 6144), pointer=0, storage_pointer=0)
        weight = FakeTensor((512, 6144))
        out = FakeTensor((0, 512), pointer=0, storage_pointer=0)
        self.assertEqual(x.numel(), 0)
        self.assertEqual(out.numel(), 0)
        self.assertFalse(self.module.h20_decode_supported(x, weight, out))
        self.assertIs(self.module.cute_dsl_h20_decode(x, weight, out), out)
        self.assertEqual(self.env.fallback_events[-1][0], "mm")
        self.assertFalse(self.env.vendor_events())

    def test_other_shapes_and_incompatible_dimensions_refuse(self):
        for shape in ((1, 513, 6144), (1, 4096, 6144), (1, 512, 2049)):
            with self.subTest(shape=shape):
                self.assertFalse(self.module.h20_decode_supported(*self.inputs(shape)))
        x, weight = self.inputs()
        self.assertFalse(
            self.module.h20_decode_supported(FakeTensor((1, 2048)), weight)
        )
        self.assertFalse(self.env.vendor_events())

    def test_other_hardware_refuses(self):
        hardware = (
            ((9, 0), "NVIDIA H100 80GB HBM3"),
            ((9, 0), "NVIDIA H200"),
            ((10, 0), "NVIDIA B200"),
            ((10, 0), "NVIDIA H20"),
            ((9, 0), "AMD Instinct MI350X"),
            ((9, 0), "NVIDIA H20 NVL"),
        )
        for index, description in enumerate(hardware, start=2):
            with self.subTest(hardware=description):
                self.env.devices[index] = description
                self.env.current_device = index
                inputs = self.inputs(device=FakeDevice(index=index))
                self.assertFalse(self.module.h20_decode_supported(*inputs))
        inputs = self.inputs(device=FakeDevice("cpu", None))
        self.assertFalse(self.module.h20_decode_supported(*inputs))
        self.assertFalse(self.env.vendor_events())

    def test_input_dtype_device_rank_stride_alignment_guards(self):
        bad_x = (
            FakeTensor((1, 6144), dtype=FP32),
            FakeTensor((1, 6144), device=FakeDevice(index=1)),
            FakeTensor((6144,)),
            FakeTensor((1, 1, 6144)),
            FakeTensor((1, 6144), contiguous=False),
            FakeTensor((1, 6144), strides=(9999, 1)),
            FakeTensor((1, 6144), pointer=4098),
        )
        bad_weight = (
            FakeTensor((512, 6144), dtype=FP32),
            FakeTensor((512, 6144), device=FakeDevice("cpu", None)),
            FakeTensor((512,)),
            FakeTensor((1, 512, 6144)),
            FakeTensor((512, 6144), contiguous=False),
            FakeTensor((512, 6144), strides=(6145, 1)),
            FakeTensor((512, 6144), pointer=4098),
        )
        x, weight = self.inputs()
        for tensor in bad_x:
            with self.subTest(side="x", properties=vars(tensor)):
                self.assertFalse(self.module.h20_decode_supported(tensor, weight))
        for tensor in bad_weight:
            with self.subTest(side="weight", properties=vars(tensor)):
                self.assertFalse(self.module.h20_decode_supported(x, tensor))
        self.assertFalse(self.env.vendor_events())

    def test_invalid_output_shape_dtype_device_refuses_and_raises(self):
        x, weight = self.inputs()
        for out in (
            FakeTensor((512,)),
            FakeTensor((1, 511)),
            FakeTensor((1, 512), dtype=FP32),
            FakeTensor((1, 512), device=FakeDevice(index=1)),
            FakeTensor((1, 512), device=FakeDevice("cpu", None)),
        ):
            with self.subTest(properties=vars(out)):
                self.assertFalse(self.module.h20_decode_supported(x, weight, out))
                with self.assertRaises(ValueError):
                    self.module.cute_dsl_h20_decode(x, weight, out)
        self.assertFalse(self.env.vendor_events())
        self.assertFalse(self.env.fallback_events)

    def test_noncontiguous_or_unaligned_output_uses_torch_out(self):
        x, weight = self.inputs()
        for out in (
            FakeTensor((1, 512), contiguous=False),
            FakeTensor((1, 512), strides=(9999, 1)),
            FakeTensor((1, 512), pointer=4098),
        ):
            with self.subTest(properties=vars(out)):
                self.assertFalse(self.module.h20_decode_supported(x, weight, out))
                self.assertIs(self.module.cute_dsl_h20_decode(x, weight, out), out)
                self.assertEqual(self.env.fallback_events[-1][0], "mm")
        self.assertFalse(self.env.vendor_events())

    def test_output_shared_storage_refuses_even_with_different_data_pointer(self):
        x, weight = self.inputs()
        for source in (x, weight):
            with self.subTest(source_shape=source.shape):
                out = FakeTensor(
                    (1, 512),
                    pointer=source.pointer + 32,
                    storage_pointer=source.storage_pointer,
                )
                self.assertNotEqual(out.data_ptr(), source.data_ptr())
                self.assertFalse(self.module.h20_decode_supported(x, weight, out))
                with self.assertRaises(ValueError):
                    self.module.cute_dsl_h20_decode(x, weight, out)
        self.assertFalse(self.env.vendor_events())
        self.assertFalse(self.env.fallback_events)

    def test_cold_serving_and_capture_never_touch_vendor(self):
        for serving, capturing in ((True, False), (False, True), (True, True)):
            self.env.serving, self.env.capturing = serving, capturing
            for shape in EXPECTED_ROUTES:
                with self.subTest(serving=serving, capturing=capturing, shape=shape):
                    x, weight = self.inputs(shape)
                    self.assertFalse(self.module.h20_decode_supported(x, weight))
                    self.module.cute_dsl_h20_decode(x, weight)
        self.assertFalse(self.env.vendor_events())
        self.assertEqual(len(self.env.fallback_events), 21)
        self.assertFalse(self.module._warmed)

    def test_foreign_device_capture_cannot_authorize_cold_vendor(self):
        self.env.current_device = 0
        self.env.capture_by_device = {0: False, 1: True}
        for shape in ((1, 512, 6144), (1, 4096, 2048)):
            with self.subTest(shape=shape):
                x, weight = self.inputs(shape, device=FakeDevice(index=1))
                self.assertFalse(self.module.h20_decode_supported(x, weight))
                self.module.cute_dsl_h20_decode(x, weight)
        self.assertFalse(self.env.vendor_events())
        self.assertFalse(self.module._warmed)
        self.assertEqual(self.env.current_device, 0)

    def test_successful_eager_call_warms_serving_and_capture(self):
        for shape in EXPECTED_ROUTES:
            x, weight = self.inputs(shape)
            self.module.cute_dsl_h20_decode(x, weight)
        for serving, capturing in ((True, False), (False, True), (True, True)):
            self.env.serving, self.env.capturing = serving, capturing
            for shape in EXPECTED_ROUTES:
                with self.subTest(serving=serving, capturing=capturing, shape=shape):
                    x, weight = self.inputs(shape)
                    self.assertTrue(self.module.h20_decode_supported(x, weight))
                    self.module.cute_dsl_h20_decode(x, weight)
        self.assertEqual(len(self.env.vendor_calls()), 28)
        self.assertFalse(self.env.fallback_events)
        self.assertEqual(len(self.module._warmed), 7)

    def test_failed_eager_call_does_not_mark_shape_warm(self):
        for shape in ((1, 512, 6144), (1, 4096, 2048)):
            with self.subTest(shape=shape):
                x, weight = self.inputs(shape)
                for driver in self.env.drivers:
                    driver.fail_call = True
                self.env.serving = False
                self.env.capturing = False
                with self.assertRaisesRegex(RuntimeError, "fake vendor launch failed"):
                    self.module.cute_dsl_h20_decode(x, weight)
                self.assertFalse(self.module._warmed)
                self.env.serving = True
                self.assertFalse(self.module.h20_decode_supported(x, weight))
                before = len(self.env.vendor_calls())
                self.module.cute_dsl_h20_decode(x, weight)
                self.assertEqual(len(self.env.vendor_calls()), before)
                self.env.serving = False
                self.env.capturing = True
                self.assertFalse(self.module.h20_decode_supported(x, weight))
                self.module.cute_dsl_h20_decode(x, weight)
                self.assertEqual(len(self.env.vendor_calls()), before)

    def test_pdl_uses_separate_drivers_and_warm_keys(self):
        self.assertEqual(len({id(driver) for driver in self.env.drivers}), 4)
        for shape in ((1, 512, 6144), (1, 4096, 2048)):
            with self.subTest(shape=shape):
                x, weight = self.inputs(shape)
                self.env.serving = False
                self.env.pdl = False
                self.module.cute_dsl_h20_decode(x, weight)
                self.env.serving = True
                self.env.pdl = True
                self.assertFalse(self.module.h20_decode_supported(x, weight))
                self.env.serving = False
                self.module.cute_dsl_h20_decode(x, weight)
                self.env.serving = True
                self.assertTrue(self.module.h20_decode_supported(x, weight))
        for driver in self.env.drivers:
            self.assertEqual(len(driver.calls()), 1)
        self.assertEqual(len(self.module._warmed), 4)

    def test_warm_keys_are_isolated_by_device(self):
        for shape in ((1, 512, 6144), (1, 4096, 2048)):
            with self.subTest(shape=shape):
                self.env.current_device = 0
                self.env.serving = False
                self.module.cute_dsl_h20_decode(*self.inputs(shape))
                self.env.serving = True
                other_inputs = self.inputs(shape, device=FakeDevice(index=1))
                self.env.current_device = 1
                self.assertFalse(self.module.h20_decode_supported(*other_inputs))
                self.env.serving = False
                self.module.cute_dsl_h20_decode(*other_inputs)
                self.env.serving = True
                self.assertTrue(self.module.h20_decode_supported(*other_inputs))
        self.assertEqual(len(self.module._warmed), 4)

    def test_noncurrent_device_refuses_even_outside_capture(self):
        inputs = self.inputs(device=FakeDevice(index=1))
        self.assertFalse(self.module.h20_decode_supported(*inputs))
        self.module.cute_dsl_h20_decode(*inputs)
        self.assertFalse(self.env.vendor_events())
        self.assertFalse(self.module._warmed)
        self.env.current_device = 1
        self.assertTrue(self.module.h20_decode_supported(*inputs))

    def test_warming_one_shape_does_not_authorize_another_shape(self):
        self.module.cute_dsl_h20_decode(*self.inputs((1, 512, 6144)))
        self.env.serving = True
        self.assertFalse(self.module.h20_decode_supported(*self.inputs((1, 128, 6144))))
        self.env.serving = False
        self.env.capturing = True
        self.assertFalse(self.module.h20_decode_supported(*self.inputs((1, 128, 6144))))
        self.assertEqual(len(self.env.vendor_calls()), 1)

    def test_device_without_index_uses_current_device_in_warm_key(self):
        inputs = self.inputs(device=FakeDevice(index=None))
        self.env.current_device = 0
        self.module.cute_dsl_h20_decode(*inputs)
        self.env.serving = True
        self.assertTrue(self.module.h20_decode_supported(*inputs))
        self.env.current_device = 1
        self.assertFalse(self.module.h20_decode_supported(*inputs))

    def test_rejected_or_unavailable_vendor_does_not_warm(self):
        for driver in self.env.drivers:
            driver.supports_result = False
        for shape in EXPECTED_ROUTES:
            self.assertFalse(self.module.h20_decode_supported(*self.inputs(shape)))
            self.module.cute_dsl_h20_decode(*self.inputs(shape))
        for driver in self.env.drivers:
            driver.supports_result = True
            driver.available = False
        skinny_inputs = self.inputs((1, 4096, 2048))
        self.assertFalse(self.module.h20_decode_supported(*skinny_inputs))
        self.module.cute_dsl_h20_decode(*skinny_inputs)
        self.assertFalse(self.env.vendor_calls())
        self.assertFalse(self.module._warmed)

    def test_missing_shared_toolchain_also_refuses_otherwise_available_skinny(self):
        for driver in self.env.drivers:
            driver.available = driver.backend != "ll_bf16"
        for pdl in (False, True):
            self.env.pdl = pdl
            for shape in ((1, 512, 6144), (1, 4096, 2048)):
                with self.subTest(pdl=pdl, shape=shape):
                    inputs = self.inputs(shape)
                    self.assertFalse(self.module.h20_decode_supported(*inputs))
                    self.module.cute_dsl_h20_decode(*inputs)
        self.assertFalse(self.env.vendor_calls())
        self.assertFalse(self.module._warmed)
        self.assertEqual(len(self.env.fallback_events), 4)

    def test_caller_output_is_forwarded_and_returned_for_both_backends(self):
        for shape in ((1, 512, 6144), (1, 4096, 2048)):
            with self.subTest(shape=shape):
                x, weight = self.inputs(shape)
                out = FakeTensor((1, shape[1]))
                self.assertIs(self.module.cute_dsl_h20_decode(x, weight, out), out)
                for driver in self.env.drivers:
                    if driver.calls():
                        _, args, kwargs = driver.calls()[-1]
                        forwarded = (
                            args[2] if driver.backend == "ll_bf16" else kwargs["out"]
                        )
                        self.assertIs(forwarded, out)
                        if driver.backend == "ll_bf16":
                            self.assertEqual(kwargs["out_dtype"], BF16)
                        else:
                            self.assertEqual(args[2], FakeSkinnyConfig(1, 64, 2, 2, 8))
                    driver.events.clear()
        self.assertFalse(self.env.fallback_events)

    def test_cold_out_fallback_preserves_caller_output(self):
        self.env.serving = True
        x, weight = self.inputs()
        out = FakeTensor((1, 512))
        self.assertIs(self.module.cute_dsl_h20_decode(x, weight, out), out)
        self.assertIs(self.env.fallback_events[-1][-1], out)
        self.assertFalse(self.env.vendor_events())

    def test_mixed_dtype_fallback_promotes_x_and_validates_out_dtype(self):
        x, _ = self.inputs()
        weight = FakeTensor((512, 6144), dtype=FP32)
        result = self.module.cute_dsl_h20_decode(x, weight)
        self.assertEqual(result.dtype, FP32)
        out = FakeTensor((1, 512), dtype=FP32)
        self.assertIs(self.module.cute_dsl_h20_decode(x, weight, out), out)
        with self.assertRaises(ValueError):
            self.module.cute_dsl_h20_decode(x, weight, FakeTensor((1, 512)))
        self.assertFalse(self.env.vendor_events())

    def test_mixed_dtype_fallback_rejects_output_sharing_original_input_storage(self):
        x, _ = self.inputs()
        weight = FakeTensor((512, 6144), dtype=FP32)
        out = FakeTensor(
            (1, 512),
            dtype=FP32,
            pointer=x.pointer + 32,
            storage_pointer=x.storage_pointer,
        )
        with self.assertRaises(ValueError):
            self.module.cute_dsl_h20_decode(x, weight, out)
        self.assertFalse(self.env.vendor_events())
        self.assertFalse(self.env.fallback_events)

    def test_ast_extracted_generic_entry_selects_candidate_and_honors_cold_gate(self):
        generic_entry = self.env.generic_entry_function()
        for shape in EXPECTED_ROUTES:
            x, weight = self.inputs(shape)
            self.assertTrue(generic_entry(x, weight))
            self.env.serving = True
            self.assertFalse(generic_entry(x, weight))
            self.env.serving = False
            self.module.cute_dsl_h20_decode(x, weight)
            self.env.serving = True
            self.assertTrue(generic_entry(x, weight))
            self.env.serving = False
        for m in (0, 2, 4, 8, 16, 32, 33):
            self.assertFalse(generic_entry(*self.inputs((m, 512, 6144))))


class ActualRegistrySelectionCPU(unittest.TestCase):
    def setUp(self):
        self.env = ActualSelectionHarness()
        self.registry = self.env.registry.KernelRegistry.get()
        self.rowcta = self.env.generic["triton_rowcta_gemv"]
        self.torch_impl = self.env.generic["torch_decode_gemv"]
        self.candidate = self.env.module.cute_dsl_h20_decode

    @staticmethod
    def inputs(shape):
        m, n, k = shape
        return FakeTensor((m, k)), FakeTensor((n, k))

    def register_plugin(self, *, priority, traits):
        marker = object()

        def plugin(x, weight, out=None):
            return marker

        self.env.registry.register_kernel(
            "gemm",
            "decode_gemv",
            name="test_decode_plugin",
            solution="plugin",
            signatures=self.env.generic["_BF16_SIG"],
            traits=traits,
            priority=priority,
        )(plugin)
        self.env.select.cache_clear()
        return plugin, marker

    def test_helper_import_leaves_actual_registry_specs_unchanged(self):
        self.assertEqual(self.env.specs_before_helper, self.env.specs_after_helper)
        self.assertEqual(
            {spec.name for spec in self.registry.list_kernels()},
            {
                "triton_rowcta_gemv",
                "gluon_wmma_dense_gemv_gfx1250",
                "torch_decode_gemv",
            },
        )
        self.assertIsNone(self.registry.get_by_name("cute_dsl_h20_decode"))

    def test_baseline_seven_argument_selector_call_retains_original_rowcta(self):
        self.assertIs(
            self.env.select(1, 512, 6144, True, BF16, BF16, False), self.rowcta
        )

    def test_h100_h200_all_seven_shapes_keep_actual_rowcta_selection_and_dispatch(self):
        for name in ("NVIDIA H100 80GB HBM3", "NVIDIA H200"):
            self.env.set_platform(name, vendor="nvidia", arch=(9, 0))
            for shape in EXPECTED_ROUTES:
                with self.subTest(device=name, shape=shape):
                    x, weight = self.inputs(shape)
                    self.assertFalse(self.env.module.h20_decode_supported(x, weight))
                    self.assertFalse(self.env.generic["use_decode_gemv"](x, weight))
                    selected = self.env.select_shape(shape, h20_eligible=False)
                    self.assertIs(selected, self.rowcta)
                    out = FakeTensor((shape[0], shape[1]))
                    self.assertIs(self.env.generic["decode_gemv"](x, weight, out), out)
                    self.assertIs(self.env.selected_impls[-1], self.rowcta)
        self.assertEqual(len(self.env.rowcta_kernel.launches), 14)
        self.assertFalse(self.env.vendor_calls())
        self.assertFalse(self.env.fallback_events)

    def test_eligible_h20_all_seven_shapes_select_unregistered_local_candidate(self):
        for shape in EXPECTED_ROUTES:
            with self.subTest(shape=shape):
                x, weight = self.inputs(shape)
                self.assertTrue(self.env.module.h20_decode_supported(x, weight))
                self.assertIs(
                    self.env.select_shape(shape, h20_eligible=True), self.candidate
                )
                out = FakeTensor((shape[0], shape[1]))
                self.assertIs(self.env.generic["decode_gemv"](x, weight, out), out)
                self.assertIs(self.env.selected_impls[-1], self.candidate)
        self.assertEqual(len(self.env.vendor_calls()), 7)
        self.assertFalse(self.env.rowcta_kernel.launches)
        self.assertIsNone(self.registry.get_by_name("cute_dsl_h20_decode"))

    def test_real_decode_recomputes_eligibility_across_cached_cold_warm_and_reject(
        self,
    ):
        x, weight = self.inputs((1, 512, 6144))
        self.env.serving = True
        self.env.generic["decode_gemv"](x, weight)
        self.assertIs(self.env.selected_impls[-1], self.rowcta)
        self.env.serving = False
        self.env.generic["decode_gemv"](x, weight)
        self.assertIs(self.env.selected_impls[-1], self.candidate)
        self.env.serving = True
        self.env.generic["decode_gemv"](x, weight)
        self.assertIs(self.env.selected_impls[-1], self.candidate)
        # The same input shape remains hot in _select's cache. An unsuitable
        # destination must still make eligibility false before that lookup.
        out = FakeTensor((1, 512), pointer=4098)
        self.env.generic["decode_gemv"](x, weight, out)
        self.assertIs(self.env.selected_impls[-1], self.rowcta)
        self.assertEqual(len(self.env.rowcta_kernel.launches), 2)
        self.assertEqual(len(self.env.vendor_calls()), 2)
        self.assertEqual(self.env.select.cache_info().currsize, 2)

    def test_actual_capability_and_shape_traits_filter_original_specs(self):
        amd_spec = self.registry.get_by_name("gluon_wmma_dense_gemv_gfx1250")
        self.assertFalse(amd_spec.capability.satisfied_by(self.env.platform))
        self.assertIs(
            self.env.select_shape((2, 512, 6144), h20_eligible=False), self.torch_impl
        )
        self.env.set_platform("AMD gfx1250", vendor="amd", arch=(12, 5))
        self.assertTrue(amd_spec.capability.satisfied_by(self.env.platform))
        self.assertIs(
            self.env.select_shape((2, 512, 6144), h20_eligible=False),
            self.env.generic["gluon_wmma_dense_gemv_gfx1250"],
        )
        self.assertIs(
            self.env.select_shape((2, 513, 6144), h20_eligible=False), self.torch_impl
        )
        self.env.set_platform("NVIDIA H100", vendor="nvidia", arch=(9, 0))
        row_spec = self.registry.get_by_name("triton_rowcta_gemv")
        for shape in ((1, 127, 6144), (1, 512, 127)):
            traits = dict(zip(("m", "n", "k"), shape))
            self.assertTrue(self.env.selection.spec_matches_traits(row_spec, traits))
            self.assertFalse(
                self.env.selection.spec_matches_shape_traits(row_spec, traits)
            )
            self.assertIs(
                self.env.select_shape(shape, h20_eligible=False), self.torch_impl
            )

    def test_matching_priority16_plugin_and_priority14_tie_precede_local_candidate(
        self,
    ):
        shape = (1, 512, 6144)
        for priority in (16, 14):
            with self.subTest(priority=priority):
                plugin, marker = self.register_plugin(
                    priority=priority, traits={"m": frozenset({1})}
                )
                self.assertIs(self.env.select_shape(shape, h20_eligible=True), plugin)
                self.assertIs(
                    self.env.generic["decode_gemv"](*self.inputs(shape)), marker
                )
        self.assertFalse(self.env.vendor_calls())

    def test_lower_priority_or_mismatched_plugin_does_not_block_local_candidate(self):
        shape = (1, 512, 6144)
        cases = (
            (13, {"m": frozenset({1})}),
            (16, {"m": frozenset({2})}),
            (16, {"m": frozenset({1}), "n_min": frozenset({1024})}),
        )
        for priority, traits in cases:
            with self.subTest(priority=priority, traits=traits):
                self.register_plugin(priority=priority, traits=traits)
                self.assertIs(
                    self.env.select_shape(shape, h20_eligible=True), self.candidate
                )
                self.env.generic["decode_gemv"](*self.inputs(shape))
                self.assertIs(self.env.selected_impls[-1], self.candidate)
        self.assertEqual(len(self.env.vendor_calls()), 3)

    def test_global_priority14_fault_detects_all_14_rowcta_regressions(self):
        injected = self.env.register_candidate_fault(extra_traits={})
        self.assertEqual(injected.priority, 14)
        self.assertEqual(self.registry.get_by_name("triton_rowcta_gemv").priority, 12)
        detected = 0
        for name in ("NVIDIA H100 80GB HBM3", "NVIDIA H200"):
            self.env.set_platform(name, vendor="nvidia", arch=(9, 0))
            self.assertTrue(injected.capability.satisfied_by(self.env.platform))
            for shape in EXPECTED_ROUTES:
                with self.subTest(device=name, shape=shape):
                    selected = self.env.select_shape(shape, h20_eligible=False)
                    # The invariant used by the preceding hardware regression
                    # test must fail after restoring the old broad decorator.
                    with self.assertRaises(AssertionError):
                        self.assertIs(selected, self.rowcta)
                    self.assertIs(selected, self.candidate)
                    self.env.generic["decode_gemv"](*self.inputs(shape))
                    self.assertEqual(self.env.fallback_events[-1][0], "linear")
                    detected += 1
        self.assertEqual(detected, 14)
        self.assertFalse(self.env.rowcta_kernel.launches)
        self.assertFalse(self.env.vendor_calls())
        print(
            f"FAULT SENSITIVITY: global priority-14 registration breaks rowcta in {detected}/14 H100/H200 cases; wrapper takes Torch.",
            flush=True,
        )

    def test_missing_ordinary_boolean_trait_cannot_gate_h20_product(self):
        self.env.set_platform("NVIDIA H100", vendor="nvidia", arch=(9, 0))
        spec = self.env.register_candidate_fault(
            extra_traits={"h20_only": frozenset({True})}
        )
        traits = {"m": 1, "n": 512, "k": 6144, "weight_split": False}
        self.assertTrue(self.env.selection.spec_matches_traits(spec, traits))
        self.assertTrue(self.env.selection.spec_matches_shape_traits(spec, traits))
        self.assertFalse(
            self.env.selection.spec_matches_traits(spec, {**traits, "h20_only": False})
        )
        self.assertIs(
            self.env.select_shape((1, 512, 6144), h20_eligible=False), self.candidate
        )
        torch_spec = self.registry.get_by_name("torch_decode_gemv")
        self.assertTrue(self.env.selection.spec_matches_traits(torch_spec, traits))
        self.assertFalse(
            self.env.selection.spec_matches_traits(
                torch_spec, traits, require_all_traits=True
            )
        )

    def test_public_solution_and_override_selection_keeps_original_registry_only(self):
        signature = self.env.generic["_SIGNATURES"][(BF16, BF16)]
        traits = {"m": 1, "n": 512, "k": 6144, "weight_split": False}
        env_key = "TOKENSPEED_KERNEL_OVERRIDE_GEMM_DECODE_GEMV"
        with patch.dict(os.environ, {env_key: ""}):
            for solution, expected in (
                ("torch", self.torch_impl),
                ("triton", self.rowcta),
            ):
                selected = self.env.selection.select_kernel(
                    "gemm", "decode_gemv", signature, traits=traits, solution=solution
                )
                self.assertIs(selected.impl, expected)
            for override in ("torch_decode_gemv", "torch"):
                selected = self.env.selection.select_kernel(
                    "gemm", "decode_gemv", signature, traits=traits, override=override
                )
                self.assertIs(selected.impl, self.torch_impl)
            with self.assertRaises(self.env.selection.NoKernelFoundError):
                self.env.selection.select_kernel(
                    "gemm", "decode_gemv", signature, traits=traits, solution="cute_dsl"
                )

    def test_direct_dispatch_preserves_baseline_context_and_env_override_boundary(self):
        # Baseline decode_gemv/_select has no override argument and does not
        # consult public selection overrides. Do not claim new override support.
        signature = self.env.generic["_SIGNATURES"][(BF16, BF16)]
        traits = {"m": 1, "n": 512, "k": 6144, "weight_split": False}
        env_key = "TOKENSPEED_KERNEL_OVERRIDE_GEMM_DECODE_GEMV"
        with patch.dict(os.environ, {env_key: ""}):
            with self.env.selection.kernel_override(
                "gemm", "decode_gemv", "torch_decode_gemv"
            ):
                selected = self.env.selection.select_kernel(
                    "gemm", "decode_gemv", signature, traits=traits
                )
                self.assertIs(selected.impl, self.torch_impl)
                self.env.generic["decode_gemv"](*self.inputs((1, 512, 6144)))
                self.assertIs(self.env.selected_impls[-1], self.candidate)
        with patch.dict(os.environ, {env_key: "torch"}):
            selected = self.env.selection.select_kernel(
                "gemm", "decode_gemv", signature, traits=traits
            )
            self.assertIs(selected.impl, self.torch_impl)
            self.env.generic["decode_gemv"](*self.inputs((1, 512, 6144)))
            self.assertIs(self.env.selected_impls[-1], self.candidate)


if __name__ == "__main__":
    print(
        "CPU-only contract harness: real candidate module; fake Torch/platform/"
        "vendor drivers; actual registry/signature/selection modules and actual "
        "platform data classes, generic decorators, selector and entry ASTs. "
        "No GPU correctness, performance, or full Triton-module import is tested.",
        flush=True,
    )
    unittest.main(verbosity=2)
