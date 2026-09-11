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

import os
import subprocess
import sys
from pathlib import Path

import pytest


@pytest.mark.parametrize("entrypoint", ["public", "builtins"])
def test_dsv41_registration_from_empty_registry(entrypoint):
    # A subprocess keeps implementation imports during collection from masking
    # a missing public-family or builtin-loader registration import.
    script = """
import importlib
import sys

sys.path.insert(0, sys.argv[2])
from tokenspeed_kernel.registry import KernelRegistry, load_builtin_kernels

family = "tokenspeed_kernel.ops.attention.dsv41"
operations = (
    "cache_pack", "cache_unpack", "cache_scatter", "cache_gather",
    "index_q_quantize", "selected_attention", "index_score", "index_topk",
)
for _ in range(2):
    KernelRegistry.reset()
    assert not KernelRegistry.get().list_kernels(family=None, mode=None)
    if sys.argv[1] == "public":
        for name in tuple(sys.modules):
            if name == family or name.startswith(family + "."):
                del sys.modules[name]
        importlib.import_module(family)
    else:
        load_builtin_kernels()
    registry = KernelRegistry.get()
    for operation in operations:
        name = "triton_dsv41_" + operation
        spec = registry.get_by_name(name)
        assert spec is not None, name
        assert (spec.family, spec.mode, spec.solution) == (
            "attention", "dsv41_" + operation, "triton"
        )
        implementation = registry.get_impl(name)
        assert callable(implementation), name
        assert implementation.__module__ == family + ".triton"
"""
    result = subprocess.run(
        [
            sys.executable,
            "-B",
            "-c",
            script,
            entrypoint,
            str(Path(__file__).resolve().parents[1] / "python"),
        ],
        env={
            **os.environ,
            "CUDA_VISIBLE_DEVICES": "",
            "HIP_VISIBLE_DEVICES": "",
            "ROCR_VISIBLE_DEVICES": "",
            "PYTHONDONTWRITEBYTECODE": "1",
        },
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert result.returncode == 0, result.stdout + result.stderr
