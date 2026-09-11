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

"""JIT-only SwiGLU correction for the pinned, optional DeepGEMM dependency.

See README.md for upstream provenance and the import-before-JIT requirement.
"""

from __future__ import annotations

import hashlib
import os
import tempfile
from importlib import import_module
from pathlib import Path

_HEADER = Path("deep_gemm/impls/sm100_fp8_fp4_mega_moe.cuh")
# Public upstream: deepseek-ai/DeepGEMM@66081d4c9c7d7c44f13fea402e5b622aa0f409c2.
# Include the preceding line to exclude the SiTU assignment of the same name.
_SWIGLU = """                            const auto up = __bfloat1622float2(bf16_up);
                            activation_values[i][k] = __fmul2_rn(__fmul2_rn(gate, up), weights);"""
_BF16_SWIGLU = """                            const auto up = __bfloat1622float2(bf16_up);
                            // V4/V4.1 round weighted SwiGLU to BF16 before amax and FP8.
                            activation_values[i][k] = __bfloat1622float2(__float22bfloat162_rn(
                                __fmul2_rn(__fmul2_rn(gate, up), weights)));"""


def _patch_swiglu(source: str) -> str:
    if source.count(_SWIGLU) != 1:
        raise RuntimeError(
            "Unsupported DeepGEMM MegaMoE header: expected one weighted SwiGLU "
            "epilogue. Update the BF16-before-FP8 patch for this DeepGEMM version."
        )
    return source.replace(_SWIGLU, _BF16_SWIGLU, 1)


def _make_include_overlay(include: Path, cache: Path) -> Path:
    patched = _patch_swiglu((include / _HEADER).read_text())
    digest = hashlib.sha256((str(include) + patched).encode()).hexdigest()
    root = cache / "mega_moe_bf16" / digest
    if (root / "include" / _HEADER).is_file():
        return root
    root.parent.mkdir(parents=True, exist_ok=True)
    # Publish a complete tree atomically; EP ranks may initialize concurrently.
    with tempfile.TemporaryDirectory(dir=root.parent) as tmp:
        staging = Path(tmp) / "root"
        for relative in (Path(), Path("deep_gemm"), Path("deep_gemm/impls")):
            destination = staging / "include" / relative
            destination.mkdir(parents=True, exist_ok=True)
            for entry in (include / relative).iterdir():
                if relative / entry.name in (
                    Path("deep_gemm"),
                    Path("deep_gemm/impls"),
                    _HEADER,
                ):
                    continue
                (destination / entry.name).symlink_to(entry)
        (staging / "include" / _HEADER).write_text(patched)
        try:
            staging.rename(root)
        except OSError:
            if not (root / "include" / _HEADER).is_file():
                raise
    return root


def prepare_mega_moe_bf16_jit() -> None:
    """Select corrected JIT headers before any DeepGEMM kernel is compiled.

    Only the weighted SwiGLU epilogue changes; all other installed headers are
    symlinked unchanged. DeepGEMM hashes the include contents and include path,
    so cached uncorrected cubins cannot be reused. Returns nothing.
    """
    deep_gemm = import_module("deep_gemm")
    if deep_gemm.__file__ is None:
        raise RuntimeError(
            "DeepGEMM JIT correction requires a file-backed installation"
        )

    include = Path(deep_gemm.__file__).resolve().parent / "include"
    cache = Path(os.environ.get("DG_JIT_CACHE_DIR", str(Path.home() / ".deep_gemm")))
    root = _make_include_overlay(include, cache)
    deep_gemm._C.init(str(root), deep_gemm._find_cuda_home())
