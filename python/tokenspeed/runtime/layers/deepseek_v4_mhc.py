# Copyright (c) 2026 LightSeek Foundation
#
# Portions copyright the vLLM project contributors under Apache-2.0.

from __future__ import annotations

import glob
import importlib
import math
import os
import site
import sys
from functools import cache
from typing import Any

import torch

try:
    import tilelang
    import tilelang.language as T
except Exception:  # pragma: no cover - availability depends on deployment image
    tilelang = None
    T = None


def _cdiv(x: int, y: int) -> int:
    return (x + y - 1) // y


@cache
def _compute_num_split(block_k: int, k: int | None, grid_size: int) -> int:
    device_props = torch.cuda.get_device_properties(0)
    split_k = device_props.multi_processor_count // grid_size
    if k is not None:
        num_block_k = _cdiv(k, block_k)
        split_k = min(split_k, num_block_k // 4)
    return max(split_k, 1)


def _prepare_deep_gemm_jit_env() -> None:
    cuda_home = os.environ.get("CUDA_HOME") or os.environ.get("CUDA_PATH")
    if not cuda_home and os.path.exists("/usr/local/cuda/include/cuda_runtime.h"):
        cuda_home = "/usr/local/cuda"
        os.environ["CUDA_HOME"] = cuda_home
    if not cuda_home:
        return

    include_dir = os.path.join(cuda_home, "include")
    if os.path.exists(os.path.join(include_dir, "cuda_runtime.h")):
        cpath = os.environ.get("CPATH", "")
        paths = [path for path in cpath.split(os.pathsep) if path]
        if include_dir not in paths:
            os.environ["CPATH"] = os.pathsep.join([include_dir] + paths)

    path_entries = [
        entry for entry in os.environ.get("PATH", "").split(os.pathsep) if entry
    ]
    candidates = []
    site_paths = []
    try:
        site_paths.extend(site.getsitepackages())
    except Exception:
        pass
    site_paths.extend(sys.path)
    for base in site_paths:
        candidates.extend(
            sorted(glob.glob(os.path.join(base, "nvidia", "cu*", "bin")), reverse=True)
        )
    candidates.extend(
        [
            os.path.join(os.path.dirname(torch.__file__), "bin"),
            os.path.join(
                os.path.dirname(torch.__file__),
                "..",
                "triton",
                "backends",
                "nvidia",
                "bin",
            ),
        ]
    )

    ptxas_dirs: list[str] = []
    for candidate in candidates:
        candidate = os.path.abspath(candidate)
        if (
            os.path.exists(os.path.join(candidate, "ptxas"))
            and candidate not in ptxas_dirs
        ):
            ptxas_dirs.append(candidate)
    for candidate in reversed(ptxas_dirs):
        if candidate not in path_entries:
            path_entries.insert(0, candidate)
    if path_entries:
        os.environ["PATH"] = os.pathsep.join(path_entries)


@cache
def _get_deep_gemm() -> Any | None:
    _prepare_deep_gemm_jit_env()
    for module_name in (
        "deep_gemm",
        "tokenspeed_kernel.thirdparty.deep_gemm",
    ):
        try:
            module = importlib.import_module(module_name)
        except Exception:
            continue
        if hasattr(module, "tf32_hc_prenorm_gemm"):
            return module
    return None


if tilelang is not None:

    @tilelang.jit(
        pass_configs={
            tilelang.PassConfigKey.TL_DISABLE_WARP_SPECIALIZED: True,
            tilelang.PassConfigKey.TL_DISABLE_TMA_LOWER: True,
            tilelang.PassConfigKey.TL_PTXAS_REGISTER_USAGE_LEVEL: 10,
        },
    )
    def _mhc_pre_big_fuse_tilelang(
        gemm_out_mul,
        gemm_out_sqrsum,
        hc_scale,
        hc_base,
        residual,
        post_mix,
        comb_mix,
        layer_input,
        hidden_size: int,
        rms_eps: float,
        hc_pre_eps: float,
        hc_sinkhorn_eps: float,
        hc_post_mult_value: float,
        sinkhorn_repeat: int,
        n_splits: int = 16,
        hc_mult: int = 4,
    ):
        num_tokens = T.dynamic("num_tokens")
        hc_mult3 = hc_mult * (2 + hc_mult)
        hidden_block = math.gcd(512, hidden_size)

        gemm_out_mul: T.Tensor[[n_splits, num_tokens, hc_mult3], T.float32]
        gemm_out_sqrsum: T.Tensor[[n_splits, num_tokens], T.float32]
        hc_scale: T.Tensor[[3], T.float32]
        hc_base: T.Tensor[[hc_mult3], T.float32]
        residual: T.Tensor[[num_tokens, hc_mult, hidden_size], T.bfloat16]
        post_mix: T.Tensor[[num_tokens, hc_mult], T.float32]
        comb_mix: T.Tensor[[num_tokens, hc_mult * hc_mult], T.float32]
        layer_input: T.Tensor[[num_tokens, hidden_size], T.bfloat16]

        with T.Kernel(num_tokens, threads=96) as i:
            T.pdl_sync()
            rms = T.alloc_fragment(1, T.float32)
            mixes = T.alloc_fragment(hc_mult3, T.float32)
            T.clear(mixes)
            rms[0] = 0
            for i_split in T.serial(n_splits):
                rms[0] += gemm_out_sqrsum[i_split, i]
            rms[0] = T.rsqrt(rms[0] / (hc_mult * hidden_size) + rms_eps)
            for j in T.Parallel(hc_mult3):
                mixes[j] = 0
                for i_split in T.serial(n_splits):
                    mixes[j] += gemm_out_mul[i_split, i, j]
                mixes[j] *= rms[0]
            mixes_shared = T.alloc_shared(hc_mult3, T.float32)
            T.copy(mixes, mixes_shared)

            if T.get_thread_binding() < 32:
                cm = T.alloc_fragment((hc_mult, hc_mult), T.float32)
                for j in T.Parallel(hc_mult):
                    post_mix[i, j] = (
                        T.sigmoid(
                            mixes_shared[j + hc_mult] * hc_scale[1]
                            + hc_base[j + hc_mult]
                        )
                        * hc_post_mult_value
                    )
                for j, k in T.Parallel(hc_mult, hc_mult):
                    cm[j, k] = (
                        mixes_shared[j * hc_mult + k + hc_mult * 2] * hc_scale[2]
                        + hc_base[j * hc_mult + k + hc_mult * 2]
                    )

                row_sum = T.alloc_fragment(hc_mult, T.float32)
                col_sum = T.alloc_fragment(hc_mult, T.float32)
                row_max = T.alloc_fragment(hc_mult, T.float32)
                T.reduce_max(cm, row_max, dim=1)
                for j, k in T.Parallel(hc_mult, hc_mult):
                    cm[j, k] = T.exp(cm[j, k] - row_max[j])
                T.reduce_sum(cm, row_sum, dim=1)
                for j, k in T.Parallel(hc_mult, hc_mult):
                    cm[j, k] = cm[j, k] / row_sum[j] + hc_sinkhorn_eps

                T.reduce_sum(cm, col_sum, dim=0)
                for j, k in T.Parallel(hc_mult, hc_mult):
                    cm[j, k] = cm[j, k] / (col_sum[k] + hc_sinkhorn_eps)

                for _ in T.serial(sinkhorn_repeat - 1):
                    T.reduce_sum(cm, row_sum, dim=1)
                    for j, k in T.Parallel(hc_mult, hc_mult):
                        cm[j, k] = cm[j, k] / (row_sum[j] + hc_sinkhorn_eps)

                    T.reduce_sum(cm, col_sum, dim=0)
                    for j, k in T.Parallel(hc_mult, hc_mult):
                        cm[j, k] = cm[j, k] / (col_sum[k] + hc_sinkhorn_eps)

                for j, k in T.Parallel(hc_mult, hc_mult):
                    comb_mix[i, j * hc_mult + k] = cm[j, k]
            else:
                pre_mix_shared = T.alloc_shared(hc_mult, T.float32)
                for j in T.Parallel(hc_mult):
                    pre_mix_shared[j] = (
                        T.sigmoid(mixes_shared[j] * hc_scale[0] + hc_base[j])
                        + hc_pre_eps
                    )
                for i0_h in T.Pipelined(hidden_size // hidden_block, num_stages=2):
                    xs = T.alloc_shared((hc_mult, hidden_block), T.float32)
                    xl = T.alloc_fragment((hc_mult, hidden_block), T.float32)
                    T.copy(residual[i, 0, i0_h * hidden_block], xs)
                    T.copy(xs, xl)

                    ol = T.alloc_fragment(hidden_block, T.float32)
                    T.clear(ol)
                    for i_hc in T.serial(hc_mult):
                        pre = pre_mix_shared[i_hc]
                        for i1_h in T.Parallel(hidden_block):
                            ol[i1_h] += pre * xl[i_hc, i1_h]
                    T.copy(ol, layer_input[i, i0_h * hidden_block])
            T.pdl_trigger()

    @tilelang.jit(
        pass_configs={
            tilelang.PassConfigKey.TL_DISABLE_WARP_SPECIALIZED: True,
            tilelang.PassConfigKey.TL_DISABLE_TMA_LOWER: True,
            tilelang.PassConfigKey.TL_PTXAS_REGISTER_USAGE_LEVEL: 10,
        },
    )
    def _mhc_post_tilelang(
        a,
        b,
        c,
        d,
        x,
        hc: int,
        hidden: int,
        n_thr: int = 128,
        h_blk: int = 1024,
    ):
        n = T.dynamic("num_tokens")
        h_blk = math.gcd(hidden, h_blk)
        a: T.Tensor((n, hc, hc), T.float32)
        b: T.Tensor((n, hc, hidden), T.bfloat16)
        c: T.Tensor((n, hc), T.float32)
        d: T.Tensor((n, hidden), T.bfloat16)
        x: T.Tensor((n, hc, hidden), T.bfloat16)

        with T.Kernel(n, threads=n_thr) as i_n:
            x_shared = T.alloc_shared((hc, h_blk), T.bfloat16)
            b_shared = T.alloc_shared((hc, h_blk), T.bfloat16)
            d_shared = T.alloc_shared(h_blk, T.bfloat16)
            x_local = T.alloc_fragment((hc, h_blk), T.float32)
            b_local = T.alloc_fragment((hc, h_blk), T.float32)
            d_local = T.alloc_fragment(h_blk, T.float32)
            a_local = T.alloc_fragment((hc, hc), T.float32)
            c_local = T.alloc_fragment(hc, T.float32)

            T.pdl_sync()
            T.copy(a[i_n, 0, 0], a_local)
            T.copy(c[i_n, 0], c_local)
            for i0_h in T.Pipelined(T.ceildiv(hidden, h_blk), num_stages=2):
                T.copy(b[i_n, 0, i0_h * h_blk], b_shared)
                T.copy(d[i_n, i0_h * h_blk], d_shared)
                T.copy(b_shared, b_local)
                T.copy(d_shared, d_local)
                for i_hco, i1_h in T.Parallel(hc, h_blk):
                    x_local[i_hco, i1_h] = c_local[i_hco] * d_local[i1_h]
                    for i_hci in T.serial(hc):
                        x_local[i_hco, i1_h] += (
                            a_local[i_hci, i_hco] * b_local[i_hci, i1_h]
                        )
                T.copy(x_local, x_shared)
                T.copy(x_shared, x[i_n, 0, i0_h * h_blk])
            T.pdl_trigger()

else:
    _mhc_pre_big_fuse_tilelang = None
    _mhc_post_tilelang = None


def mhc_pre(
    residual: torch.Tensor,
    fn: torch.Tensor,
    hc_scale: torch.Tensor,
    hc_base: torch.Tensor,
    rms_eps: float,
    hc_eps: float,
    sinkhorn_iters: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if tilelang is None or _mhc_pre_big_fuse_tilelang is None:
        raise RuntimeError("tilelang is unavailable")
    if residual.dtype != torch.bfloat16 or fn.dtype != torch.float32:
        raise RuntimeError("fast mHC requires bf16 residual and fp32 weights")
    if not residual.is_cuda:
        raise RuntimeError("fast mHC requires CUDA tensors")

    deep_gemm = _get_deep_gemm()
    if deep_gemm is None:
        raise RuntimeError("deep_gemm.tf32_hc_prenorm_gemm is unavailable")

    hc_mult = residual.shape[-2]
    hidden_size = residual.shape[-1]
    hc_mult2 = hc_mult * hc_mult
    hc_mult3 = hc_mult * 2 + hc_mult2
    hc_hidden_size = hc_mult * hidden_size
    outer_shape = residual.shape[:-2]
    residual_flat = residual.view(-1, hc_mult, hidden_size)
    num_tokens = residual_flat.shape[0]
    if num_tokens == 0:
        return (
            residual.new_empty(*outer_shape, hidden_size),
            torch.empty(
                *outer_shape,
                hc_mult,
                1,
                dtype=torch.float32,
                device=residual.device,
            ),
            torch.empty(
                *outer_shape,
                hc_mult,
                hc_mult,
                dtype=torch.float32,
                device=residual.device,
            ),
        )

    block_k = 64
    block_m = 64
    n_splits = _compute_num_split(block_k, hc_hidden_size, _cdiv(num_tokens, block_m))

    post_mix = torch.empty(
        num_tokens, hc_mult, dtype=torch.float32, device=residual.device
    )
    comb_mix = torch.empty(
        num_tokens, hc_mult2, dtype=torch.float32, device=residual.device
    )
    layer_input = torch.empty(
        num_tokens, hidden_size, dtype=torch.bfloat16, device=residual.device
    )
    gemm_out_mul = torch.empty(
        n_splits, num_tokens, hc_mult3, dtype=torch.float32, device=residual.device
    )
    gemm_out_sqrsum = torch.empty(
        n_splits, num_tokens, dtype=torch.float32, device=residual.device
    )

    deep_gemm.tf32_hc_prenorm_gemm(
        residual_flat.view(num_tokens, hc_hidden_size),
        fn,
        gemm_out_mul,
        gemm_out_sqrsum,
        n_splits,
    )
    _mhc_pre_big_fuse_tilelang(
        gemm_out_mul,
        gemm_out_sqrsum,
        hc_scale,
        hc_base,
        residual_flat,
        post_mix,
        comb_mix,
        layer_input,
        hidden_size,
        rms_eps,
        hc_eps,
        hc_eps,
        2.0,
        sinkhorn_iters,
        n_splits,
        hc_mult,
    )

    return (
        layer_input.view(*outer_shape, hidden_size),
        post_mix.view(*outer_shape, hc_mult, 1),
        comb_mix.view(*outer_shape, hc_mult, hc_mult),
    )


def mhc_post(
    hidden_states: torch.Tensor,
    residual: torch.Tensor,
    post: torch.Tensor,
    comb: torch.Tensor,
) -> torch.Tensor:
    if tilelang is None or _mhc_post_tilelang is None:
        raise RuntimeError("tilelang is unavailable")
    if not hidden_states.is_cuda:
        raise RuntimeError("fast mHC requires CUDA tensors")
    if residual.numel() == 0:
        return torch.empty_like(residual)
    out = torch.empty_like(residual)
    _mhc_post_tilelang(
        comb,
        residual,
        post.squeeze(-1),
        hidden_states,
        out,
        residual.shape[-2],
        residual.shape[-1],
    )
    return out
