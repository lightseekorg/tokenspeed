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

"""Binary FMHA prefill backend for tokenspeed-mla.

Loads a pre-compiled AOT TVM-FFI .so calls the exported function.

SO files are looked up in:
  <package>/objs/cute_dsl_fmha_fp8_e4m3_to_bf16_hd192_{arch}.so

Set TOKENSPEED_MLA_FMHA_BINARY_SO to override with a custom .so path.

Note on LSE layout: the binary kernel writes LSE in (1, h_k, h_r, total_q)
layout (row-major), which differs from the CuteDSL backend's (total_q, h_q).
The caller (mla_prefill) allocates the binary-layout buffer and reshapes the
result to (total_q, h_q) before returning.
"""

from __future__ import annotations

import functools
import logging
import os
import platform
from pathlib import Path

import torch

logger = logging.getLogger(__name__)

# Exported TVM-FFI function names inside the bundled SO, keyed by (is_causal, return_lse).
# Names follow compile_dsl_fmha.py::get_variant_name() with:
#   in_dtype=Float8E4M3FN, out_dtype=BFloat16, head_dim=192,
#   varlen=True, is_persistent=False, enable_skip_softmax=False, enable_tvm_ffi=True
_FUNC_NAMES: dict[tuple[bool, bool], str] = {
    (
        True,
        True,
    ): "cute_dsl_fmha_fp8_e4m3_bf16_hd192_causal_nonpersistent_varlen_lse_tvmffi",
    (
        True,
        False,
    ): "cute_dsl_fmha_fp8_e4m3_bf16_hd192_causal_nonpersistent_varlen_tvmffi",
    (
        False,
        True,
    ): "cute_dsl_fmha_fp8_e4m3_bf16_hd192_nocausal_nonpersistent_varlen_lse_tvmffi",
    (
        False,
        False,
    ): "cute_dsl_fmha_fp8_e4m3_bf16_hd192_nocausal_nonpersistent_varlen_tvmffi",
}


def _objs_dir() -> Path:
    return Path(__file__).resolve().parent / "objs"


def _resolve_so_path() -> Path:
    """Return the SO path: env override first, then arch-specific default."""
    env = os.environ.get("TOKENSPEED_MLA_FMHA_BINARY_SO")
    if env:
        return Path(env)
    props = torch.cuda.get_device_properties(torch.cuda.current_device())
    machine = platform.machine().lower()
    if machine == "amd64":
        machine = "x86_64"
    elif machine == "arm64":
        machine = "aarch64"
    arch = f"sm_{props.major}{props.minor}a_{machine}"
    name = f"cute_dsl_fmha_fp8_e4m3_to_bf16_hd192_{arch}.so"
    return _objs_dir() / name


@functools.cache
def _load_module(so_path_str: str):
    """Load a TVM-FFI module from the given .so path (result is cached per path)."""
    import cutlass.cute as cute

    so_path = Path(so_path_str)
    if not so_path.exists():
        raise RuntimeError(
            f"tokenspeed_mla binary FMHA SO not found: {so_path}. "
            "Place the compiled .so there or set TOKENSPEED_MLA_FMHA_BINARY_SO."
        )
    logger.info("Loading binary FMHA module from %s", so_path)
    return cute.runtime.load_module(str(so_path), enable_tvm_ffi=True)


def has_binary_prefill() -> bool:
    """Return True if a binary FMHA SO is present and loadable for the current GPU."""
    try:
        _load_module(str(_resolve_so_path()))
        return True
    except Exception:
        return False


def call_binary_prefill(
    q_ct,
    k_ct,
    v_ct,
    o_ct,
    lse_ct,
    problem_size: tuple,
    cum_q_ct,
    cum_k_ct,
    scale_softmax_log2: float,
    scale_softmax: float,
    is_causal: bool,
    return_lse: bool,
) -> None:
    """Invoke the pre-compiled binary FMHA prefill kernel via TVM-FFI.

    Tensor layout requirements (must match the AOT-compiled binary):
      Q  : cute.Tensor, shape (1, total_q, h_k, h_r, d_qk), row-major
      K  : cute.Tensor, shape (1, total_kv, h_k, 1,   d_qk), row-major
      V  : cute.Tensor, shape (1, total_kv, h_k, 1,   d_v),  row-major
      O  : cute.Tensor, shape (1, total_q, h_k, h_r, d_v),  row-major
      lse: cute.Tensor, shape (1, h_k, h_r, total_q), row-major — or None
      cum_q_ct / cum_k_ct: cute.Tensor of int32 cumulative sequence lengths
    """
    import tvm_ffi

    module = _load_module(str(_resolve_so_path()))
    func = getattr(module, _FUNC_NAMES[(is_causal, return_lse)])
    window_size_right = 0 if is_causal else None

    with tvm_ffi.use_torch_stream():
        func(
            q_ct,
            k_ct,
            v_ct,
            o_ct,
            problem_size,
            cum_q_ct,
            cum_k_ct,
            lse_ct,
            scale_softmax_log2,
            scale_softmax,
            1.0,  # scale_output
            None,  # skip_softmax_threshold_log2 (disabled)
            None,  # window_size_left
            window_size_right,
            None,  # skip_softmax_count
            None,  # total_softmax_count
        )
