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

import glob
import os
import site
import sys


def _prepare_deep_gemm_cuda_home() -> None:
    """Set CUDA_HOME before importing deep_gemm so its C++ init sees nvcc."""

    requested_cuda_home = os.environ.get("CUDA_HOME") or os.environ.get("CUDA_PATH")
    site_paths = []
    try:
        site_paths.extend(site.getsitepackages())
    except Exception:
        pass
    site_paths.extend(sys.path)
    site_paths.extend(glob.glob("/tmp/*/lib/python*/site-packages"))

    candidates = []
    if requested_cuda_home:
        candidates.append(requested_cuda_home)
    candidates.append("/usr/local/cuda")
    for base in site_paths:
        candidates.extend(
            sorted(glob.glob(os.path.join(base, "nvidia", "cu*")), reverse=True)
        )

    cuda_home = None
    for candidate in candidates:
        candidate = os.path.abspath(candidate)
        if os.path.exists(
            os.path.join(candidate, "include", "cuda_runtime.h")
        ) and os.path.exists(os.path.join(candidate, "bin", "nvcc")):
            cuda_home = candidate
            break
    if cuda_home is None:
        return

    os.environ["CUDA_HOME"] = cuda_home
    include_dir = os.path.join(cuda_home, "include")
    cpath = os.environ.get("CPATH", "")
    cpath_entries = [entry for entry in cpath.split(os.pathsep) if entry]
    if include_dir not in cpath_entries:
        os.environ["CPATH"] = os.pathsep.join([include_dir] + cpath_entries)

    bin_dir = os.path.join(cuda_home, "bin")
    path = os.environ.get("PATH", "")
    path_entries = [entry for entry in path.split(os.pathsep) if entry]
    if bin_dir not in path_entries:
        os.environ["PATH"] = os.pathsep.join([bin_dir] + path_entries)


_prepare_deep_gemm_cuda_home()

from deep_gemm import (
    ceil_div,
    fp8_fp4_mqa_logits,
    fp8_fp4_paged_mqa_logits,
    fp8_gemm_nt,
    fp8_mqa_logits,
    fp8_paged_mqa_logits,
    get_mn_major_tma_aligned_tensor,
    get_num_sms,
    get_paged_mqa_logits_metadata,
    m_grouped_fp8_gemm_nt_contiguous,
    m_grouped_fp8_gemm_nt_masked,
    set_num_sms,
    transform_sf_into_required_layout,
)

__all__ = [
    "ceil_div",
    "fp8_fp4_mqa_logits",
    "fp8_fp4_paged_mqa_logits",
    "fp8_gemm_nt",
    "get_num_sms",
    "m_grouped_fp8_gemm_nt_contiguous",
    "m_grouped_fp8_gemm_nt_masked",
    "set_num_sms",
    "transform_sf_into_required_layout",
    "get_paged_mqa_logits_metadata",
    "fp8_paged_mqa_logits",
    "fp8_mqa_logits",
]
