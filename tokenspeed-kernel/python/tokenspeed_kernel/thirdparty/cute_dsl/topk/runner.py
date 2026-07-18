# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# Vendored into tokenspeed-kernel from NVIDIA TensorRT-LLM:
#   https://github.com/NVIDIA/TensorRT-LLM/blob/main/tensorrt_llm/_torch/custom_ops/cute_dsl_custom_ops.py
# The CuteDSLTopKDecode* runner classes and their helper glue
# (``_get_num_sms``, ``_TORCH_TO_CUTLASS_DTYPE``) are adapted from the
# TensorRT-LLM source with logic unchanged (formatting follows the repo style).
# ``_ScratchBuffers`` / ``get_memory_buffers`` replace
# TensorRT-LLM's native ``memory_buffer_utils.get_memory_buffers`` with a
# self-contained, CUDA-graph-friendly arena so the runners carry no dependency
# on the TensorRT-LLM native stack.
"""Class-level runners driving the single-pass multi-CTA radix top-k kernels.

The runners own compilation caching, the SM-aware chunk heuristic, and the
scratch ``row_states`` buffer used for inter-CTA coordination. Two variants are
exposed:

    * :class:`CuteDSLTopKDecodeSinglePassMultiCTARunner` -- global-memory atomic
      histogram merging (the "distributed" variant).
    * :class:`CuteDSLTopKDecodeSinglePassMultiCTAClusterRunner` -- Blackwell
      cluster barriers + DSMEM histogram merging, falling back with
      ``(None, None)`` when the problem size exceeds cluster capacity.
"""

from __future__ import annotations

import logging
import math
from typing import Optional

import cutlass
import cutlass.cute as cute
import torch

from .top_k.single_pass_multi_cta_radix_topk import (
    STATE_SIZE as DISTRIBUTED_TOPK_STATE_SIZE,
)
from .top_k.single_pass_multi_cta_radix_topk import (
    SinglePassMultiCTARadixTopKKernel,
)
from .top_k.single_pass_multi_cta_radix_topk_cluster import (
    STATE_SIZE as CLUSTER_TOPK_STATE_SIZE,
)
from .top_k.single_pass_multi_cta_radix_topk_cluster import (
    SinglePassMultiCTARadixTopKClusterKernel,
    _query_max_cluster_size,
)

logger = logging.getLogger(__name__)


class _ScratchBuffers:
    """Self-contained replacement for TensorRT-LLM's ``get_memory_buffers()``.

    The runners request named scratch tensors via
    ``get_buffer(shape, dtype, buffer_name=..., reserve_buffer=...)``. The real
    implementation reuses a growable arena keyed by name; this mirrors that so a
    hot decode loop does not re-allocate every call, and so the returned tensor
    keeps a stable device address across calls (required for CUDA-graph replay).
    """

    def __init__(self) -> None:
        self._cache: dict = {}

    def get_buffer(self, shape, dtype, buffer_name="_default", reserve_buffer=False):
        numel = 1
        for s in shape:
            numel *= int(s)
        key = (buffer_name, dtype)
        buf = self._cache.get(key)
        if buf is None or buf.numel() < numel:
            buf = torch.empty(numel, dtype=dtype, device="cuda")
            self._cache[key] = buf
        return buf[:numel].view(*shape)


_SCRATCH_BUFFERS = _ScratchBuffers()


def get_memory_buffers() -> _ScratchBuffers:
    """Return the process-wide scratch arena singleton."""
    return _SCRATCH_BUFFERS


def _get_num_sms() -> int:
    """Return the number of SMs on the current device (cached)."""
    if not hasattr(_get_num_sms, "_value"):
        _get_num_sms._value = torch.cuda.get_device_properties().multi_processor_count
    return _get_num_sms._value


# Module-level dtype mapping (avoid recreating per call)
_TORCH_TO_CUTLASS_DTYPE = {
    torch.float16: cutlass.Float16,
    torch.bfloat16: cutlass.BFloat16,
    torch.float32: cutlass.Float32,
}


class CuteDSLTopKDecodeSinglePassMultiCTARunner:
    """Runner for single-pass multi-CTA radix top-k (FlashInfer-style fused multi-CTA).

    All CTAs in a group cooperatively find the global pivot via multi-round
    radix select with global histogram merging, then each CTA collects
    results from its own chunk.  Single kernel launch, no intermediate
    buffer, no merge kernel.

    All methods are class-level -- no instantiation needed.

    Attributes:
        kernel_cache: Class-level dict mapping config tuples to compiled
                     kernels.
    """

    kernel_cache = dict()
    buffers = get_memory_buffers()
    _row_states_initialized = False
    _row_states_buffer_name = "sp_mcta_row_states"
    _buf_prefix = "sp_mcta_"
    _kernel_class = SinglePassMultiCTARadixTopKKernel
    _state_size = DISTRIBUTED_TOPK_STATE_SIZE

    @classmethod
    def _compile(
        cls,
        dtype,
        chunk_size,
        top_k,
        next_n,
        num_copy_bits,
        ctas_per_group,
        num_sms,
        return_val,
    ):
        """Compile and cache a single-pass multi-CTA radix top-k kernel."""
        key = (
            dtype,
            chunk_size,
            top_k,
            next_n,
            num_copy_bits,
            ctas_per_group,
            num_sms,
            return_val,
        )
        if key in cls.kernel_cache:
            return
        n_rows = cute.sym_int()
        n_cols = cute.sym_int()
        n_batch = cute.sym_int()
        n_groups = cute.sym_int()

        input_fake = cute.runtime.make_fake_compact_tensor(
            dtype,
            (n_rows, n_cols),
            stride_order=(1, 0),
            assumed_align=32,
        )
        row_states_fake = cute.runtime.make_fake_compact_tensor(
            cutlass.Int32,
            (n_groups, cls._state_size),
            stride_order=(1, 0),
            assumed_align=32,
        )
        seqlen_fake = cute.runtime.make_fake_compact_tensor(
            cutlass.Int32,
            (n_batch,),
            stride_order=(0,),
        )
        output_indices_fake = cute.runtime.make_fake_compact_tensor(
            cutlass.Int32,
            (n_rows, top_k),
            stride_order=(1, 0),
        )
        if return_val:
            output_values_fake = cute.runtime.make_fake_compact_tensor(
                dtype,
                (n_rows, top_k),
                stride_order=(1, 0),
            )
        else:
            output_values_fake = None
        fake_stream = cute.runtime.make_fake_stream(use_tvm_ffi_env_stream=True)

        kernel_obj = cls._kernel_class(
            dtype=dtype,
            chunk_size=chunk_size,
            top_k=top_k,
            next_n=next_n,
            num_copy_bits=num_copy_bits,
            ctas_per_group=ctas_per_group,
            num_sms=num_sms,
        )
        compiled_kernel = cute.compile(
            kernel_obj,
            input_fake,
            row_states_fake,
            seqlen_fake,
            output_indices_fake,
            output_values_fake,
            stream=fake_stream,
            options="--enable-tvm-ffi",
        )
        cls.kernel_cache[key] = compiled_kernel

    @classmethod
    def _compute_max_chunk(cls, dtype, num_copy_bits: int = 256):
        """Compute the maximum chunk_size a single CTA can handle."""
        max_smem = cutlass.utils.get_smem_capacity_in_bytes()
        # Fixed shared memory overhead (excludes shared_ordered[chunk_size]):
        # local_histogram[256]*4 + prefix_buf[256]*4 + scalars[4]*4 + warp_sums[8]*4
        overhead = 256 * 4 * 2 + 4 * 4 + 8 * 4
        if dtype == cutlass.Float32:
            ordered_elem_size = 4
        else:
            ordered_elem_size = 2
        vec_size = num_copy_bits // dtype.width
        max_chunk = (max_smem - overhead) // ordered_elem_size
        max_chunk = (max_chunk // vec_size) * vec_size
        return max_chunk, vec_size

    @classmethod
    def _get_chunk_config(
        cls,
        dtype,
        num_cols: int,
        chunk_size: Optional[int] = None,
        num_copy_bits: int = 256,
        num_rows: int = 1,
    ):
        """Resolve chunk_size and ctas_per_group.

        If chunk_size is provided, use it (clamped and aligned).
        Otherwise use an SM-aware heuristic that targets
        total_ctas approx num_sms by balancing parallelism against
        per-CTA reduce overhead.

        Returns:
            (chunk_size, ctas_per_group, vec_size)
        """
        max_chunk, vec_size = cls._compute_max_chunk(dtype, num_copy_bits)

        if chunk_size is not None:
            # User-specified: clamp and align
            chunk_size = min(chunk_size, max_chunk)
            chunk_size = (chunk_size // vec_size) * vec_size
            if chunk_size < vec_size:
                chunk_size = vec_size
        else:
            # Auto: SM-aware heuristic
            num_sms = _get_num_sms()

            # Target total_ctas approx num_sms
            ideal_ctas_per_group = max(1, num_sms // max(num_rows, 1))

            if ideal_ctas_per_group <= 1:
                # Large batch: use FlashInfer-style logic --
                # minimize ctas_per_group based on max_chunk capacity
                ctas_per_group = math.ceil(num_cols / max_chunk)
                if ctas_per_group < 1:
                    ctas_per_group = 1
                chunk_size = math.ceil(num_cols / ctas_per_group)
                chunk_size = ((chunk_size + vec_size - 1) // vec_size) * vec_size
                if chunk_size > max_chunk:
                    chunk_size = max_chunk
            else:
                chunk_size = math.ceil(num_cols / ideal_ctas_per_group)

                # Minimum chunk to avoid per-CTA overhead dominating
                chunk_size = max(chunk_size, 8192)

                # Avoid ctas_per_group=2 with small chunks: reduce
                # overhead (~5us) exceeds 2-way parallelism benefit
                ctas_per_group = math.ceil(num_cols / chunk_size)
                if ctas_per_group == 2 and chunk_size < 32768:
                    chunk_size = num_cols

                # Snap to power-of-2 for JIT cache friendliness
                snap_up = 1 << math.ceil(math.log2(max(chunk_size, 1)))
                if snap_up > max_chunk:
                    snap_up = 1 << int(math.log2(max_chunk))
                chunk_size = snap_up

        ctas_per_group = math.ceil(num_cols / chunk_size)
        return chunk_size, ctas_per_group, vec_size

    @classmethod
    def _get_possible_chunk_sizes(cls, dtype, num_copy_bits: int = 256):
        """Return all possible chunk_size values the auto heuristic can produce.

        These are powers of 2 from 8192 up to the largest power of 2
        that fits within max_chunk (for the SM-aware multi-CTA path).
        """
        max_chunk, _ = cls._compute_max_chunk(dtype, num_copy_bits)
        sizes = []
        cs = 8192
        while cs <= max_chunk:
            sizes.append(cs)
            cs *= 2
        return sizes

    @classmethod
    def forward(
        cls,
        input_values: torch.Tensor,
        seq_lens: torch.Tensor,
        top_k: int,
        next_n: int,
        return_val: bool = False,
        num_copy_bits: int = 256,
        chunk_size: Optional[int] = None,
        output_indices: Optional[torch.Tensor] = None,
    ):
        """Execute single-pass multi-CTA radix top-k selection.

        Args:
            chunk_size: Optional chunk size per CTA. If None, uses the
                maximum chunk that fits in shared memory. Smaller values
                increase ctas_per_group (more parallelism) at the cost of
                more inter-CTA synchronization.
        """
        torch_dtype = input_values.dtype
        dtype = _TORCH_TO_CUTLASS_DTYPE[torch_dtype]
        num_rows, num_cols = input_values.shape
        num_sms = _get_num_sms()

        chunk_size, ctas_per_group, _ = cls._get_chunk_config(
            dtype, num_cols, chunk_size, num_copy_bits, num_rows=num_rows
        )

        num_groups = min(num_sms // ctas_per_group, num_rows)
        if num_groups < 1:
            num_groups = 1

        key = (
            dtype,
            chunk_size,
            top_k,
            next_n,
            num_copy_bits,
            ctas_per_group,
            num_sms,
            return_val,
        )
        cls._compile(*key)
        compiled_kernel = cls.kernel_cache[key]
        reserve = torch.cuda.is_current_stream_capturing()

        # Allocate row_states once with num_sms rows -- large enough for
        # any ctas_per_group config because group_id < num_groups
        # <= num_sms // ctas_per_group <= num_sms.  The kernel resets
        # the slots it used at end-of-kernel, so the buffer stays clean
        # across calls without re-zeroing (FlashInfer pattern).
        # extra buffer: 148 * 770 * 4 bytes = 452960 bytes = 440 KB
        buf_name = cls._row_states_buffer_name
        row_states = cls.buffers.get_buffer(
            [num_sms, cls._state_size],
            torch.int32,
            buffer_name=buf_name,
            reserve_buffer=reserve,
        )
        if not cls._row_states_initialized:
            row_states.zero_()
            cls._row_states_initialized = True

        # Allocate outputs
        if output_indices is not None:
            output_indices_torch = output_indices
        else:
            output_indices_torch = cls.buffers.get_buffer(
                [num_rows, top_k],
                torch.int32,
                buffer_name=cls._buf_prefix + "output_indices",
                reserve_buffer=reserve,
            )
        if return_val:
            output_values = cls.buffers.get_buffer(
                [num_rows, top_k],
                torch_dtype,
                buffer_name=cls._buf_prefix + "output_values",
                reserve_buffer=reserve,
            )
        else:
            output_values = None

        compiled_kernel(
            input_values,
            row_states,
            seq_lens,
            output_indices_torch,
            output_values,
        )

        return output_indices_torch, output_values


class CuteDSLTopKDecodeSinglePassMultiCTAClusterRunner(
    CuteDSLTopKDecodeSinglePassMultiCTARunner
):
    """Runner for cluster-accelerated single-pass multi-CTA radix top-k.

    Uses Blackwell cluster barriers and DSMEM for inter-CTA histogram
    merging instead of global memory atomics.  Only 1 int32 per group
    is needed in global memory (the output counter).

    Inherits compile, chunk heuristics, and forward from the base runner;
    overrides _get_chunk_config (cluster-size clamping) and forward
    (unsupported-size fallback).
    """

    kernel_cache = dict()
    buffers = get_memory_buffers()
    _row_states_initialized = False
    _row_states_buffer_name = "sp_mcta_cluster_row_states"
    _buf_prefix = "sp_mcta_cluster_"
    _kernel_class = SinglePassMultiCTARadixTopKClusterKernel
    _state_size = CLUSTER_TOPK_STATE_SIZE

    @classmethod
    def _get_chunk_config(
        cls,
        dtype,
        num_cols: int,
        chunk_size: Optional[int] = None,
        num_copy_bits: int = 256,
        num_rows: int = 1,
    ):
        """Resolve chunk_size and ctas_per_group, clamped to hw max cluster.

        Returns:
            (chunk_size, ctas_per_group, vec_size) or (None, None, None)
        """
        chunk_size, ctas_per_group, vec_size = super()._get_chunk_config(
            dtype, num_cols, chunk_size, num_copy_bits, num_rows
        )

        hw_max_cluster = _query_max_cluster_size()
        if ctas_per_group > hw_max_cluster:
            max_chunk, vec_size = cls._compute_max_chunk(dtype, num_copy_bits)
            chunk_size = math.ceil(num_cols / hw_max_cluster)
            chunk_size = ((chunk_size + vec_size - 1) // vec_size) * vec_size
            if chunk_size > max_chunk:
                logger.warning(
                    f"Cluster top-k: num_cols={num_cols} requires "
                    f"chunk_size={chunk_size} which exceeds max shared "
                    f"memory capacity ({max_chunk}). Cannot handle this "
                    f"problem size with cluster kernel."
                )
                return None, None, None
            ctas_per_group = math.ceil(num_cols / chunk_size)

        return chunk_size, ctas_per_group, vec_size

    @classmethod
    def forward(
        cls,
        input_values: torch.Tensor,
        seq_lens: torch.Tensor,
        top_k: int,
        next_n: int,
        return_val: bool = False,
        num_copy_bits: int = 256,
        chunk_size: Optional[int] = None,
        output_indices: Optional[torch.Tensor] = None,
    ):
        """Execute cluster-accelerated single-pass multi-CTA radix top-k.

        Returns (None, None) if the problem size exceeds what the cluster
        kernel can handle (caller should fall back to the non-cluster runner).
        """
        torch_dtype = input_values.dtype
        dtype = _TORCH_TO_CUTLASS_DTYPE[torch_dtype]
        num_cols = input_values.shape[1]

        max_chunk, _ = cls._compute_max_chunk(dtype, num_copy_bits)
        hw_max_cluster = _query_max_cluster_size()
        max_supported_cols = max_chunk * hw_max_cluster
        if num_cols > max_supported_cols:
            logger.warning(
                f"Cluster top-k does not support num_cols={num_cols} "
                f"(max supported: {max_supported_cols} = "
                f"max_chunk={max_chunk} x max_cluster={hw_max_cluster} "
                f"for dtype={torch_dtype}). "
                f"Falling back to non-cluster runner."
            )
            return None, None

        result = super().forward(
            input_values,
            seq_lens,
            top_k,
            next_n,
            return_val,
            num_copy_bits,
            chunk_size,
            output_indices,
        )
        if result[0] is None:
            return None, None
        return result
