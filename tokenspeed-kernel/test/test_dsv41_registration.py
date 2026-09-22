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
    "compressor_tail_scatter", "compressor_pool", "swa_rope_scatter",
    "rope_inplace", "rope_pad_query", "dspark_rows", "dspark_anchors",
    "dspark_block",
)
registrations = [
    ("triton_dsv41_" + op, "dsv41_" + op, "triton", "triton")
    for op in operations
] + [
    ("flashmla_dsv41_selected_attention", "dsv41_selected_attention", "flashmla", "flash_mla"),
    ("deep_gemm_dsv41_index_topk", "dsv41_index_topk", "deep_gemm", "deep_gemm"),
    ("deepselect_dsv41_select_candidates", "dsv41_select_candidates", "deepselect", "deep_select"),
    ("deepselect_dsv41_select_topk", "dsv41_select_topk", "deepselect", "deep_select"),
]
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
    for name, mode, solution, module in registrations:
        spec = registry.get_by_name(name)
        assert spec is not None, name
        assert (spec.family, spec.mode, spec.solution) == (
            "attention", mode, solution
        )
        implementation = registry.get_impl(name)
        assert callable(implementation), name
        assert implementation.__module__ == family + "." + module
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
            "PYTHONDONTWRITEBYTECODE": "1",
        },
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert result.returncode == 0, result.stdout + result.stderr
