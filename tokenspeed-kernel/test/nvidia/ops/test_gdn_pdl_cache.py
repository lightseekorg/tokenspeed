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

"""PDL and ordinary CuTe executors must never share persisted artifacts."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch
from tokenspeed_kernel.platform import current_platform

if not current_platform().is_hopper_plus:
    pytest.skip("PDL requires NVIDIA SM90+", allow_module_level=True)

import cutlass.cute as cute
from cutlass.cute.runtime import from_dlpack
from flashinfer.jit import env as jit_env
from flashinfer.jit.cute_dsl_core import build_and_load_cute_dsl_kernel
from tokenspeed_kernel.ops.attention.gdn._flashinfer import pdl

_compile_cache = {}
_compile_calls = []


@cute.kernel
def _copy_kernel(source, output):
    index, _, _ = cute.arch.thread_idx()
    output[index] = source[index]


@cute.jit
def _launch_copy(source, output):
    _copy_kernel(source, output).launch(grid=(1, 1, 1), block=(32, 1, 1))


def _copy(source, output, *, source_file):
    if "compiled" not in _compile_cache:

        def compile_kernel():
            _compile_calls.append(True)
            return cute.compile(
                _launch_copy,
                from_dlpack(source),
                from_dlpack(output),
                options="--enable-tvm-ffi",
            )

        _compile_cache["compiled"] = build_and_load_cute_dsl_kernel(
            "tokenspeed_test_gdn_pdl",
            "copy",
            compile_kernel,
            extra_key_files=(__file__, str(source_file)),
        )
    _compile_cache["compiled"](source, output)


@pytest.mark.parametrize("pdl_first", [False, True])
def test_pdl_persistent_cache_isolation(pdl_first, tmp_path, monkeypatch):
    monkeypatch.setattr(jit_env, "FLASHINFER_JIT_DIR", tmp_path / "cache")
    monkeypatch.delenv("FLASHINFER_CUTE_DSL_DISABLE_CACHE", raising=False)
    # Mutate temporary source copies to test invalidation without changing
    # installed code or any shared cache directory.
    adapter_sources = [tmp_path / name for name in ("pdl.py", "adapter.py")]
    for target in adapter_sources:
        target.write_text(Path(pdl.__file__).with_name(target.name).read_text())
    monkeypatch.setattr(pdl, "__file__", str(adapter_sources[0]))
    upstream_source = tmp_path / "upstream.py"
    upstream_source.write_text("upstream version 1")
    monkeypatch.setattr(sys.modules[__name__], "_compile_cache", {})
    monkeypatch.setattr(sys.modules[__name__], "_compile_calls", [])
    adapted = pdl._adapt_module(
        sys.modules[__name__],
        kernels=("_copy_kernel",),
        launchers=("_launch_copy",),
        entrypoints=("_copy",),
        caches=("_compile_cache",),
        overrides={},
    )
    runners = (_copy, adapted["_copy"])
    order = (1, 0) if pdl_first else (0, 1)
    source = torch.arange(32, device="cuda", dtype=torch.float32)
    output = torch.empty_like(source)

    def run_both():
        # A second invocation must consult disk rather than an in-memory hit.
        _compile_cache.clear()
        adapted["_compile_cache"].clear()
        for index in order:
            output.fill_(float("nan"))
            runners[index](source, output, source_file=upstream_source)
            torch.testing.assert_close(output, source, rtol=0, atol=0)

    run_both()
    assert len(_compile_calls) == 2
    assert len(list((tmp_path / "cache").rglob("*.o"))) == 2
    run_both()
    assert len(_compile_calls) == 2, "persisted executors should be reused"

    # Local adapter changes invalidate only the PDL artifact.
    for count, path in enumerate(adapter_sources, start=3):
        path.write_text(path.read_text() + "\n# revised adapter\n")
        run_both()
        assert len(_compile_calls) == count
        run_both()
        assert len(_compile_calls) == count

    # Upstream changes invalidate both variants.
    upstream_source.write_text("upstream version 2")
    run_both()
    assert len(_compile_calls) == 6
    run_both()
    assert len(_compile_calls) == 6
