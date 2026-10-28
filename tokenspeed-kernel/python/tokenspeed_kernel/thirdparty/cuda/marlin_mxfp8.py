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

"""Standalone dense Marlin MXFP8 from pinned public vLLM v0.29 sources."""

from __future__ import annotations

import hashlib
import inspect
from functools import cache
from pathlib import Path

import torch

_UPSTREAM_COMMIT = "98dff2a81d747d1dba01a47f939f48c3526d4206"
_CXX_FLAGS = ("-std=c++20",)
_CUDA_FLAGS = (
    "-std=c++20",
    "--ftz=false",
    "--prec-div=true",
    "--prec-sqrt=true",
    "--fmad=true",
)


@cache
def load_marlin_mxfp8_module():
    """Compile the fixed W8A16 specializations once, never during graph capture."""
    if torch.cuda.is_current_stream_capturing():
        raise RuntimeError("Prepare Marlin MXFP8 weights before CUDA graph capture")
    from flashinfer.jit.core import gen_jit_spec

    if "use_fast_math" not in inspect.signature(gen_jit_spec).parameters:
        raise RuntimeError(
            "Dense Marlin requires FlashInfer JIT with use_fast_math support"
        )
    csrc = Path(__file__).resolve().parent / "csrc"
    source_root = csrc / "marlin_mxfp8_dense"
    sources = sorted(
        p for p in source_root.rglob("*") if p.suffix in (".h", ".cuh", ".hpp", ".cu")
    )
    digest = hashlib.sha256(_UPSTREAM_COMMIT.encode())
    for path in [*sources, csrc / "tvm_ffi_utils.h"]:
        digest.update(path.relative_to(csrc).as_posix().encode())
        digest.update(path.read_bytes())
    digest.update("\n".join((*_CXX_FLAGS, *_CUDA_FLAGS)).encode())
    return gen_jit_spec(
        f"tokenspeed_marlin_mxfp8_{digest.hexdigest()[:16]}",
        [source_root / "marlin_mxfp8_binding.cu"],
        extra_include_paths=[source_root, csrc],
        extra_cflags=list(_CXX_FLAGS),
        extra_cuda_cflags=list(_CUDA_FLAGS),
        use_fast_math=False,
    ).build_and_load()
