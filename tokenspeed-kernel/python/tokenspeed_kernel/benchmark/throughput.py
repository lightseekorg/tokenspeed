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

from __future__ import annotations

from typing import Any

import torch

__all__ = ["ThroughputCalculator"]


def _dtype_nbytes(dtype: torch.dtype) -> int:
    return torch.empty([], dtype=dtype).element_size()


def _shape_int(shape_params: dict[str, Any], *keys: str) -> int | None:
    for key in keys:
        if key not in shape_params:
            continue
        try:
            return int(shape_params[key])
        except (TypeError, ValueError):
            continue
    return None


class ThroughputCalculator:
    """Computes throughput metrics for benchmark results."""

    @staticmethod
    def gemm_mm_flops(M: int, N: int, K: int) -> int:
        return 2 * M * N * K

    @staticmethod
    def gemm_mm_bytes(M: int, N: int, K: int, dtype: torch.dtype) -> int:
        element_size = _dtype_nbytes(dtype)
        return (M * K + K * N + M * N) * element_size

    @staticmethod
    def gemm_bmm_flops(batch: int, M: int, N: int, K: int) -> int:
        return batch * ThroughputCalculator.gemm_mm_flops(M, N, K)

    @staticmethod
    def gemm_bmm_bytes(
        batch: int,
        M: int,
        N: int,
        K: int,
        dtype: torch.dtype,
    ) -> int:
        element_size = _dtype_nbytes(dtype)
        return (batch * M * K + batch * N * K + batch * M * N) * element_size

    @staticmethod
    def attn_flops(
        batch: int,
        seq_len: int,
        num_q_heads: int,
        num_kv_heads: int,
        head_dim: int,
    ) -> int:
        _ = num_kv_heads
        return 4 * batch * num_q_heads * seq_len * head_dim

    @staticmethod
    def attn_bytes(
        batch: int,
        seq_len: int,
        num_q_heads: int,
        num_kv_heads: int,
        head_dim: int,
        dtype: torch.dtype,
    ) -> int:
        element_size = _dtype_nbytes(dtype)
        q_elements = batch * num_q_heads * head_dim
        kv_elements = batch * seq_len * num_kv_heads * head_dim
        out_elements = batch * num_q_heads * head_dim
        return (q_elements + 2 * kv_elements + out_elements) * element_size

    @staticmethod
    def compute(
        op_family: str,
        op_mode: str,
        shape_params: dict[str, Any],
        latency_us: float,
        *,
        dtype: torch.dtype,
    ) -> tuple[float | None, float | None]:
        if latency_us <= 0:
            return None, None

        if op_family == "gemm" and op_mode == "mm":
            M = int(shape_params["M"])
            N = int(shape_params["N"])
            K = int(shape_params["K"])

            flops = ThroughputCalculator.gemm_mm_flops(M, N, K)
            bytes_moved = ThroughputCalculator.gemm_mm_bytes(M, N, K, dtype)
            seconds = latency_us * 1e-6

            tflops = flops / seconds / 1e12
            bandwidth = bytes_moved / seconds / 1e9
            return tflops, bandwidth

        if op_family == "gemm" and op_mode == "bmm":
            batch = _shape_int(shape_params, "B", "batch")
            M = _shape_int(shape_params, "M")
            N = _shape_int(shape_params, "N")
            K = _shape_int(shape_params, "K")
            if any(value is None for value in (batch, M, N, K)):
                return None, None

            flops = ThroughputCalculator.gemm_bmm_flops(batch, M, N, K)
            bytes_moved = ThroughputCalculator.gemm_bmm_bytes(
                batch,
                M,
                N,
                K,
                dtype,
            )
            seconds = latency_us * 1e-6

            tflops = flops / seconds / 1e12
            bandwidth = bytes_moved / seconds / 1e9
            return tflops, bandwidth

        if op_family in {"attention", "attn"} and op_mode == "decode":
            batch = _shape_int(
                shape_params,
                "batch",
                "batch_size",
            )
            seq_len = _shape_int(
                shape_params,
                "seq_len",
                "max_seq_len",
                "context_len",
            )
            num_q_heads = _shape_int(
                shape_params,
                "num_q_heads",
                "heads",
                "num_heads",
            )
            head_dim = _shape_int(shape_params, "head_dim")
            num_kv_heads = _shape_int(
                shape_params,
                "num_kv_heads",
            )
            if num_q_heads is not None and num_kv_heads is None:
                num_kv_heads = num_q_heads

            if any(
                value is None
                for value in (batch, seq_len, num_q_heads, num_kv_heads, head_dim)
            ):
                return None, None

            if any(
                value <= 0
                for value in (batch, seq_len, num_q_heads, num_kv_heads, head_dim)
            ):
                return None, None

            flops = ThroughputCalculator.attn_flops(
                batch,
                seq_len,
                num_q_heads,
                num_kv_heads,
                head_dim,
            )
            bytes_moved = ThroughputCalculator.attn_bytes(
                batch,
                seq_len,
                num_q_heads,
                num_kv_heads,
                head_dim,
                dtype,
            )
            seconds = latency_us * 1e-6

            tflops = flops / seconds / 1e12
            bandwidth = bytes_moved / seconds / 1e9
            return tflops, bandwidth

        return None, None
