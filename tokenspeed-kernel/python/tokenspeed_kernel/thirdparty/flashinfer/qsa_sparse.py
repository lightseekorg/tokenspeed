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

"""Stateful FlashInfer FA2 block-sparse runner for QSA attention."""

from __future__ import annotations

from dataclasses import dataclass

import torch

_DEFAULT_WORKSPACE_BYTES = 128 * 1024 * 1024
_BITS_PER_BYTE = 8


@dataclass(frozen=True)
class _PlanKey:
    rows: int
    width: int
    cache_slots: int
    num_q_heads: int
    num_kv_heads: int
    head_dim: int
    q_dtype: torch.dtype
    kv_dtype: torch.dtype
    output_dtype: torch.dtype
    softmax_scale: float


@dataclass
class FlashInferQSAPlan:
    """Static FA2 plan plus mutable GPU slot and mask buffers."""

    wrapper: object
    indptr: torch.Tensor
    indices: torch.Tensor
    packed_mask: torch.Tensor


class FlashInferQSASparseRunner:
    """Cache FlashInfer FA2 plans while sharing one workspace per device."""

    def __init__(
        self,
        device: torch.device,
        *,
        workspace_bytes: int = _DEFAULT_WORKSPACE_BYTES,
    ) -> None:
        if device.type != "cuda":
            raise ValueError("FlashInfer QSA sparse attention requires a CUDA device")
        from flashinfer.sparse import BlockSparseAttentionWrapper

        self.device = device
        self._wrapper_type = BlockSparseAttentionWrapper
        self.workspace = torch.empty(
            (int(workspace_bytes),), dtype=torch.uint8, device=device
        )
        self._plans: dict[_PlanKey, FlashInferQSAPlan] = {}
        self._scalar_scales: dict[tuple[int, float], torch.Tensor] = {}

    @property
    def plan_count(self) -> int:
        """Return the number of tensor geometries planned by this runner."""

        return len(self._plans)

    def plan(
        self,
        q: torch.Tensor,
        k_cache: torch.Tensor,
        v_cache: torch.Tensor,
        selected_width: int,
        *,
        softmax_scale: float,
    ) -> FlashInferQSAPlan:
        """Return a cached FA2 plan for the supplied QSA tensor geometry."""

        rows, num_q_heads, head_dim = q.shape
        cache_slots, num_kv_heads, cache_head_dim = k_cache.shape
        if cache_head_dim != head_dim:
            raise ValueError("QSA query and key head dimensions must match")
        if k_cache.shape[:2] != v_cache.shape[:2]:
            raise ValueError("QSA key/value cache geometry must match")
        if v_cache.shape[-1] != head_dim:
            raise ValueError("FlashInfer FA2 requires equal Q/K/V head dimensions")
        if k_cache.dtype != v_cache.dtype:
            raise TypeError("QSA key and value caches must share one dtype")
        if q.device != k_cache.device or q.device != v_cache.device:
            raise ValueError("QSA query and caches must share one device")
        if num_q_heads % num_kv_heads:
            raise ValueError("QSA query heads must be divisible by KV heads")

        key = _PlanKey(
            rows=int(rows),
            width=int(selected_width),
            cache_slots=int(cache_slots),
            num_q_heads=int(num_q_heads),
            num_kv_heads=int(num_kv_heads),
            head_dim=int(head_dim),
            q_dtype=q.dtype,
            kv_dtype=k_cache.dtype,
            output_dtype=q.dtype,
            softmax_scale=float(softmax_scale),
        )
        cached = self._plans.get(key)
        if cached is not None:
            return cached

        indices = torch.zeros(
            (rows, selected_width), dtype=torch.int32, device=self.device
        )
        initial_mask = torch.zeros(
            (rows * selected_width, 1, 1), dtype=torch.bool, device=self.device
        )
        indptr = torch.arange(rows + 1, dtype=torch.int32, device=self.device)
        indptr *= int(selected_width)
        wrapper = self._wrapper_type(self.workspace, backend="fa2")
        wrapper.plan(
            indptr,
            indices.view(-1),
            int(rows),
            int(cache_slots),
            1,
            1,
            int(num_q_heads),
            int(num_kv_heads),
            int(head_dim),
            mask=initial_mask,
            causal=False,
            pos_encoding_mode="NONE",
            sm_scale=float(softmax_scale),
            q_data_type=q.dtype,
            kv_data_type=k_cache.dtype,
            o_data_type=q.dtype,
        )
        # FlashInfer 0.6.x converts custom-mask offsets from bits to bytes only
        # while planning from a bool mask. Reuse the resulting packed buffer so
        # dynamic selected slots remain CUDA-graph capturable without re-plan.
        if wrapper._backend != "fa2":
            raise RuntimeError(
                f"FlashInfer QSA requested FA2 but planned {wrapper._backend!r}"
            )
        packed_mask = wrapper._packed_mask_buf
        expected_mask_bytes = rows * (
            (selected_width + _BITS_PER_BYTE - 1) // _BITS_PER_BYTE
        )
        if packed_mask is None or packed_mask.numel() != expected_mask_bytes:
            raise RuntimeError("FlashInfer produced an invalid QSA packed-mask buffer")
        cached = FlashInferQSAPlan(
            wrapper=wrapper,
            indptr=indptr,
            indices=indices,
            packed_mask=packed_mask,
        )
        self._plans[key] = cached
        return cached

    def _scale_tensor(
        self,
        value: float | torch.Tensor | None,
        num_heads: int,
    ) -> torch.Tensor | None:
        if value is None:
            return None
        if isinstance(value, torch.Tensor):
            tensor = value.to(device=self.device, dtype=torch.float32).reshape(-1)
            if tensor.numel() == num_heads:
                return tensor.contiguous()
            if tensor.numel() != 1:
                raise ValueError(
                    "FlashInfer QSA scale must be scalar or have one value per KV head"
                )
            if num_heads == 1:
                return tensor
            return tensor.expand(num_heads).contiguous()

        scalar = float(value)
        key = (int(num_heads), scalar)
        tensor = self._scalar_scales.get(key)
        if tensor is None:
            tensor = torch.full(
                (num_heads,), scalar, dtype=torch.float32, device=self.device
            )
            self._scalar_scales[key] = tensor
        return tensor

    def run(
        self,
        plan: FlashInferQSAPlan,
        q: torch.Tensor,
        k_cache: torch.Tensor,
        v_cache: torch.Tensor,
        *,
        k_scale: float | torch.Tensor | None,
        v_scale: float | torch.Tensor | None,
        enable_pdl: bool,
    ) -> torch.Tensor:
        """Run a prepared FA2 plan and return the attention output."""

        fp8_dtypes = {
            torch.float8_e4m3fn,
            torch.float8_e5m2,
            torch.float8_e4m3fnuz,
            torch.float8_e5m2fnuz,
        }
        if k_cache.dtype in fp8_dtypes and (k_scale is None or v_scale is None):
            raise ValueError("FP8 FlashInfer QSA requires both K and V scales")
        out = q.new_empty((q.shape[0], q.shape[1], v_cache.shape[-1]))
        return plan.wrapper.run(
            q,
            k_cache,
            v_cache,
            scale_k=self._scale_tensor(k_scale, k_cache.shape[1]),
            scale_v=self._scale_tensor(v_scale, v_cache.shape[1]),
            out=out,
            enable_pdl=enable_pdl,
        )


_runners: dict[tuple[str, int | None], FlashInferQSASparseRunner] = {}


def get_flashinfer_qsa_sparse_runner(
    device: torch.device | str,
) -> FlashInferQSASparseRunner:
    """Return the process-local QSA FA2 runner for one CUDA device."""

    normalized = torch.device(device)
    index = normalized.index
    if index is None:
        index = torch.cuda.current_device()
        normalized = torch.device("cuda", index)
    key = (normalized.type, index)
    runner = _runners.get(key)
    if runner is None:
        runner = FlashInferQSASparseRunner(normalized)
        _runners[key] = runner
    return runner


__all__ = [
    "FlashInferQSAPlan",
    "FlashInferQSASparseRunner",
    "get_flashinfer_qsa_sparse_runner",
]
