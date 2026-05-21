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

"""RoPE (Rotary Positional Embedding) kernel wrapper.

Provides apply_rope_with_cos_sin_cache_inplace with support for output_q_rope /
output_k_rope (out-of-place mode) and fused KV-buffer scatter.
"""

import functools
from pathlib import Path
from typing import Any, Optional

import torch


def _objs_dir() -> Path:
    return Path(__file__).resolve().parent / "objs"


@functools.cache
def _load_rope_module():
    """Load the pre-compiled rope shared library via TVM FFI."""
    import tvm_ffi

    so_path = _objs_dir() / "rope" / "rope.so"
    if not so_path.exists():
        raise RuntimeError(
            f"tokenspeed_kernel rope library not found at {so_path}. "
            "Run `pip install -e tokenspeed_kernel/python/` to build."
        )
    return tvm_ffi.load_module(str(so_path))


def apply_rope_with_cos_sin_cache_inplace(
    positions: torch.Tensor,
    query: torch.Tensor,
    key: torch.Tensor,
    head_size: int,
    cos_sin_cache: torch.Tensor,
    is_neox: bool = True,
    fused_set_kv_buffer_arg: Any = None,
    output_q_rope: Optional[torch.Tensor] = None,
    output_k_rope: Optional[torch.Tensor] = None,
    enable_pdl: bool = False,
) -> None:
    """Apply rotary embedding with precomputed cos/sin cache.

    Supports both in-place and out-of-place (via output_q_rope / output_k_rope).
    Optionally fuses with KV-buffer scatter when fused_set_kv_buffer_arg is provided.
    """
    if head_size not in [64, 128, 256, 512]:
        raise ValueError("Unsupported head_size, only 64/128/256/512 are supported")

    if cos_sin_cache.dtype != torch.float32:
        raise ValueError("cos_sin_cache should be float32")

    if fused_set_kv_buffer_arg is not None:
        a = fused_set_kv_buffer_arg
        if a.k_scale is not None or a.v_scale is not None:
            raise ValueError("k_scale/v_scale are not supported yet")
        if a.cache_loc is None:
            raise ValueError("fused_set_kv_buffer_arg.cache_loc is required")
        if a.cache_loc.dtype not in (torch.int32, torch.int64):
            raise ValueError(
                f"cache_loc must be int32 or int64, got {a.cache_loc.dtype}"
            )

    def _view_3d(x: torch.Tensor) -> torch.Tensor:
        return x.view(x.shape[0], -1, head_size)

    def _view_3d_value(x: torch.Tensor) -> torch.Tensor:
        return x.view(x.shape[0], -1, x.shape[-1])

    q_rope = output_q_rope if output_q_rope is not None else query
    k_rope = output_k_rope if output_k_rope is not None else key

    pos_ids = positions.to(torch.int64)
    mod = _load_rope_module()

    if fused_set_kv_buffer_arg is None:
        mod.apply_rope_pos_ids_cos_sin_cache_fused(
            _view_3d(query),
            _view_3d(key),
            _view_3d(q_rope),
            _view_3d(k_rope),
            cos_sin_cache,
            pos_ids,
            not is_neox,  # interleave = not is_neox
            None,  # v
            None,  # k_buffer
            None,  # v_buffer
            None,  # kv_cache_loc
            enable_pdl,
        )
        return

    a = fused_set_kv_buffer_arg
    mod.apply_rope_pos_ids_cos_sin_cache_fused(
        _view_3d(query),
        _view_3d(key),
        _view_3d(q_rope),
        _view_3d(k_rope),
        cos_sin_cache,
        pos_ids,
        not is_neox,
        _view_3d_value(a.value),
        _view_3d(a.k_buffer),
        _view_3d(a.v_buffer),
        a.cache_loc,
        enable_pdl,
    )
