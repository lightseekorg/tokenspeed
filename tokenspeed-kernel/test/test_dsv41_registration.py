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
import faulthandler
import importlib
import os
import sys
import time

started = time.monotonic()

def log(phase):
    try:
        print(
            f"[dsv41 {sys.argv[1]} +{time.monotonic() - started:.3f}s] {phase}",
            file=sys.stderr,
            flush=True,
        )
    except (OSError, ValueError):
        # Diagnostic output must not interrupt the registration checks.
        pass

# Capture the child stack before the parent's 60-second timeout kills it.
# Keep the watchdog armed through interpreter shutdown as well as imports.
faulthandler.enable()
faulthandler.dump_traceback_later(45)

# Read inherited settings without importing dependencies or probing the GPU.
environment = {
    key: os.environ.get(key)
    for key in (
        "PYTEST_XDIST_WORKER",
        "CUDA_VISIBLE_DEVICES",
        "CUDA_HOME",
        "LD_LIBRARY_PATH",
        "LD_PRELOAD",
        "TOKENSPEED_KERNEL_PROFILE",
        "TOKENSPEED_KERNEL_CAPTURE_SHAPES",
    )
}
log(f"child started: pid={os.getpid()}, environment={environment!a}")

sys.path.insert(0, sys.argv[2])
log("import registry: start")
from tokenspeed_kernel.registry import KernelRegistry, load_builtin_kernels
log("import registry: done")

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
    ("triton_dsv41_global_rope_scatter", "dsv41_global_rope_scatter", "triton", "_triton.rope"),
    ("triton_dsv41_index_key_norm_rope", "dsv41_index_key_norm_rope", "triton", "_triton.rope"),
    ("triton_dsv41_index_query_rope_weights", "dsv41_index_query_rope_weights", "triton", "_triton.rope"),
    ("triton_dsv41_index_query_rope_quant", "dsv41_index_query_rope_quant", "triton", "_triton.rope"),
    ("triton_dsv41_index_key_norm_rope_scatter", "dsv41_index_key_norm_rope_scatter", "triton", "_triton.rope"),
    ("flashmla_dsv41_selected_attention", "dsv41_selected_attention", "flashmla", "flash_mla"),
    ("deep_gemm_dsv41_index_topk", "dsv41_index_topk", "deep_gemm", "deep_gemm"),
    ("deepselect_dsv41_select_candidates", "dsv41_select_candidates", "deepselect", "deep_select"),
    ("deepselect_dsv41_select_topk", "dsv41_select_topk", "deepselect", "deep_select"),
]
for round_index in range(2):
    log(f"round {round_index + 1}: reset registry")
    KernelRegistry.reset()
    assert not KernelRegistry.get().list_kernels(family=None, mode=None)
    log(f"round {round_index + 1}: load {sys.argv[1]}: start")
    if sys.argv[1] == "public":
        for name in tuple(sys.modules):
            if name == family or name.startswith(family + "."):
                del sys.modules[name]
        importlib.import_module(family)
    else:
        load_builtin_kernels()
    log(f"round {round_index + 1}: load {sys.argv[1]}: done")
    registry = KernelRegistry.get()
    log(f"round {round_index + 1}: check {len(registrations)} registrations")
    for name, mode, solution, module in registrations:
        spec = registry.get_by_name(name)
        assert spec is not None, name
        assert (spec.family, spec.mode, spec.solution) == (
            "attention", mode, solution
        )
        implementation = registry.get_impl(name)
        assert callable(implementation), name
        assert implementation.__module__ == family + "." + module
    log(f"round {round_index + 1}: checks done")
log("registration checks complete")
"""
    try:
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
    except subprocess.TimeoutExpired as exc:
        # TimeoutExpired does not display captured output, even with text=True.
        for stream, output in (("stdout", exc.stdout), ("stderr", exc.stderr)):
            if output:
                if isinstance(output, bytes):
                    output = output.decode(errors="replace")
                print(f"Child {entrypoint} {stream}:\n{output}", file=sys.stderr)
        raise
    assert result.returncode == 0, result.stdout + result.stderr
