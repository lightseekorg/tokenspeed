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

"""Opt-in Triton AR+RMSNorm example using PyTorch symmetric memory.

The backend targets a single gfx950 node with replicated residual and RMSNorm
weight tensors. Each rank owns a persistent PyTorch symmetric-memory input. A
one-shot kernel pulls peer inputs for world sizes one and two; world sizes four
and eight use a two-shot row-sharded reduction and push the result to peers.

State creation and rendezvous are initialization-time collective operations.
Every local allocation reports its status before the next collective so a
rank-local allocation failure makes every rank decline the backend together.
"""

import logging
from collections.abc import Callable
from typing import Any, TypeVar, cast

import tokenspeed_kernel.ops.communication._triton_shmem_kernels as _kernels
import torch
import torch.distributed as dist
import torch.distributed._symmetric_memory as symm_mem
from tokenspeed_kernel._triton import triton
from tokenspeed_kernel.platform import current_platform
from tokenspeed_kernel.profiling import kernel_scope

logger = logging.getLogger(__name__)

_SUPPORTED_WORLD_SIZES = {1, 2, 4, 8}
_BLOCK_SIZE = 512
_WHOLE_ROW_GRID_CAP = 64
_BLOCKED_GRID_CAP = 256

_T = TypeVar("_T")
_UNSET = object()


class _CollectiveInitializationError(RuntimeError):
    """A local initialization step failed on at least one rank."""


def _initialize_consistently(
    group: dist.ProcessGroup,
    description: str,
    initialize: Callable[[], _T],
) -> _T:
    """Run local initialization, then exchange status before ranks proceed."""
    value: _T | object = _UNSET
    local_error = None
    try:
        value = initialize()
    except Exception as exc:  # noqa: BLE001 - rank-consistent fallback protocol
        local_error = f"{type(exc).__name__}: {exc}"

    reports: list[str | None] = [None] * group.size()
    dist.all_gather_object(reports, local_error, group=group)
    errors = [
        f"rank {rank}: {error}"
        for rank, error in enumerate(reports)
        if error is not None
    ]
    if errors:
        raise _CollectiveInitializationError(
            f"{description} failed on one or more ranks: {'; '.join(errors)}"
        )
    assert value is not _UNSET
    return cast(_T, value)


def _allocate_symmetric(
    group: dist.ProcessGroup,
    shape: tuple[int, ...],
    dtype: torch.dtype,
    device: torch.device,
) -> tuple[torch.Tensor, Any]:
    def allocate() -> torch.Tensor:
        # State may be initialized from a forward running under inference_mode.
        with torch.inference_mode(False), torch.no_grad():
            return symm_mem.empty(shape, dtype=dtype, device=device)

    tensor = _initialize_consistently(
        group,
        "symmetric-memory allocation",
        allocate,
    )
    # rendezvous is itself a PyTorch collective. Once ranks enter it, an outer
    # status collective cannot safely turn a rank-local failure into fallback:
    # peers may still be inside rendezvous's handle exchange. Keep failures
    # fatal until PyTorch provides rank-consistent rendezvous failure handling.
    try:
        handle = symm_mem.rendezvous(tensor, group=group)
    except Exception as exc:
        raise RuntimeError(
            "PyTorch symmetric-memory rendezvous failed; the collective cannot "
            "safely fall back"
        ) from exc
    return tensor, handle


def runtime_context_supported(
    *,
    single_node: bool,
    has_tensor_parallel: bool,
    has_data_parallel: bool,
    speculative: bool,
) -> bool:
    """Return whether runtime scheduling keeps ranks in matching calls.

    Args:
        single_node: Whether all communication ranks are on one host.
        has_tensor_parallel: Whether an attention tensor-parallel group exists.
        has_data_parallel: Whether attention data parallelism can diverge ranks.
        speculative: Whether speculative scheduling can diverge call shapes.

    Returns:
        Whether this example backend's same-shape, same-order contract holds.
    """
    return (
        single_node
        and has_tensor_parallel
        and not has_data_parallel
        and not speculative
    )


def _configuration_errors(
    *,
    arch: str,
    world_size: int,
    max_token_num: int,
    hidden_dim: int,
    dtype: torch.dtype,
) -> list[str]:
    errors = []
    if arch != "gfx950":
        errors.append(f"arch={arch!r} is unsupported; expected 'gfx950'")
    if world_size not in _SUPPORTED_WORLD_SIZES:
        errors.append(f"world_size={world_size} is unsupported")
    if max_token_num <= 0:
        errors.append(f"max_token_num={max_token_num} must be positive")
    if hidden_dim <= 0:
        errors.append(f"hidden_dim={hidden_dim} must be positive")
    if dtype != torch.bfloat16:
        errors.append(f"dtype={dtype} is unsupported; expected torch.bfloat16")
    return errors


class TritonShmemAllReduceResidualRMSNorm:
    """Persistent symmetric state for fused all-reduce, residual, and RMSNorm."""

    def __init__(
        self,
        group: dist.ProcessGroup,
        rank_in_group: int,
        max_token_num: int,
        hidden_dim: int,
        dtype: torch.dtype = torch.bfloat16,
        device: torch.device | None = None,
    ) -> None:
        self.rank_in_group = rank_in_group
        self.world_size = group.size()
        self.max_token_num = max_token_num
        self.hidden_dim = hidden_dim
        self.dtype = dtype
        self.device = device or torch.device(f"cuda:{torch.cuda.current_device()}")

        def configure_device() -> int:
            group_rank = dist.get_rank(group)
            if group_rank != rank_in_group:
                raise ValueError(
                    f"rank_in_group={rank_in_group} does not match process-group "
                    f"rank {group_rank}"
                )
            properties = torch.cuda.get_device_properties(self.device)
            required_signal_bytes = self.world_size * 4
            if symm_mem.get_signal_pad_size() < required_signal_bytes:
                symm_mem.set_signal_pad_size(required_signal_bytes)
            return properties.multi_processor_count

        self._max_programs = _initialize_consistently(
            group,
            "device and signal-pad initialization",
            configure_device,
        )

        shape = (max_token_num, hidden_dim)
        self._input, self._input_handle = _allocate_symmetric(
            group, shape, dtype, self.device
        )

        if self.world_size <= 2:
            self.kernel = (
                "oneshot_wholerow"
                if hidden_dim & (hidden_dim - 1) == 0
                else "oneshot_blocked"
            )
            if self.kernel == "oneshot_blocked":
                scratch_rows = min(
                    max_token_num,
                    self._max_programs,
                    _BLOCKED_GRID_CAP,
                )
                self._scratch = _initialize_consistently(
                    group,
                    "one-shot scratch allocation",
                    lambda: torch.empty(
                        (scratch_rows, hidden_dim),
                        dtype=torch.float32,
                        device=self.device,
                    ),
                )
            else:
                self._scratch = None
        else:
            self.kernel = "twoshot_blocked"
            self._output, self._output_handle = _allocate_symmetric(
                group, shape, dtype, self.device
            )
            self._residual_output, self._residual_output_handle = _allocate_symmetric(
                group, shape, dtype, self.device
            )
            scratch_rows = triton.cdiv(max_token_num, self.world_size)
            self._scratch = _initialize_consistently(
                group,
                "two-shot scratch allocation",
                lambda: torch.empty(
                    (scratch_rows, hidden_dim),
                    dtype=torch.float32,
                    device=self.device,
                ),
            )

        logger.info(
            "triton_shmem AR+RMSNorm state: kernel=%s ws=%d max_tokens=%d hidden=%d",
            self.kernel,
            self.world_size,
            max_token_num,
            hidden_dim,
        )

    def _grid(self, rows: int) -> int:
        cap = (
            _WHOLE_ROW_GRID_CAP
            if self.kernel == "oneshot_wholerow"
            else _BLOCKED_GRID_CAP
        )
        return max(1, min(rows, cap, self._max_programs))

    def fused(
        self,
        input_tensor: torch.Tensor,
        residual: torch.Tensor,
        weight: torch.Tensor,
        eps: float,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Run the fused operation."""
        if not _call_is_supported(self, input_tensor, residual, weight):
            raise ValueError("unsupported triton_shmem AR+RMSNorm input")
        norm_out = torch.empty_like(input_tensor)
        residual_out = torch.empty_like(residual)

        rows = input_tensor.shape[0]
        self._input[:rows].copy_(input_tensor)
        self._input_handle.barrier()

        if self.kernel == "oneshot_wholerow":
            programs = self._grid(rows)
            _kernels.fused_ar_rmsnorm_oneshot_wholerow_kernel[(programs,)](
                self._input,
                self._input_handle.buffer_ptrs_dev,
                residual,
                weight,
                norm_out,
                residual_out,
                rows,
                EPS=eps,
                HIDDEN_SIZE=self.hidden_dim,
                RANK=self.rank_in_group,
                WORLD_SIZE=self.world_size,
                NUM_PROGRAMS=programs,
                num_warps=8,
            )
        elif self.kernel == "oneshot_blocked":
            programs = self._grid(rows)
            _kernels.fused_ar_rmsnorm_oneshot_blocked_kernel[(programs,)](
                self._input,
                self._input_handle.buffer_ptrs_dev,
                residual,
                weight,
                self._scratch,
                norm_out,
                residual_out,
                rows,
                EPS=eps,
                HIDDEN_SIZE=self.hidden_dim,
                BLOCK_SIZE=min(
                    _BLOCK_SIZE,
                    triton.next_power_of_2(self.hidden_dim),
                ),
                RANK=self.rank_in_group,
                WORLD_SIZE=self.world_size,
                NUM_PROGRAMS=programs,
                num_warps=4,
            )
        else:
            shard_rows = triton.cdiv(rows, self.world_size)
            programs = self._grid(shard_rows)
            _kernels.fused_ar_rmsnorm_twoshot_blocked_kernel[(programs,)](
                self._input,
                self._input_handle.buffer_ptrs_dev,
                residual,
                self._output,
                self._output_handle.buffer_ptrs_dev,
                self._residual_output,
                self._residual_output_handle.buffer_ptrs_dev,
                weight,
                self._scratch,
                rows,
                EPS=eps,
                HIDDEN_SIZE=self.hidden_dim,
                BLOCK_SIZE=min(
                    _BLOCK_SIZE,
                    triton.next_power_of_2(self.hidden_dim),
                ),
                RANK=self.rank_in_group,
                WORLD_SIZE=self.world_size,
                NUM_PROGRAMS=programs,
                num_warps=4,
            )

        self._input_handle.barrier()
        if self.kernel == "twoshot_blocked":
            norm_out.copy_(self._output[:rows])
            residual_out.copy_(self._residual_output[:rows])
        return norm_out, residual_out


def _call_is_supported(
    state: TritonShmemAllReduceResidualRMSNorm,
    input_tensor: torch.Tensor,
    residual: torch.Tensor,
    weight: torch.Tensor,
) -> bool:
    return (
        input_tensor.is_cuda
        and residual.is_cuda
        and weight.is_cuda
        and input_tensor.is_contiguous()
        and residual.is_contiguous()
        and weight.is_contiguous()
        and input_tensor.dtype == state.dtype
        and residual.dtype == state.dtype
        and weight.dtype in (torch.bfloat16, torch.float16, torch.float32)
        and input_tensor.dim() == 2
        and input_tensor.shape == residual.shape
        and 0 < input_tensor.shape[0] <= state.max_token_num
        and input_tensor.shape[1] == state.hidden_dim
        and weight.shape == (state.hidden_dim,)
        and input_tensor.device == state.device
        and residual.device == state.device
        and weight.device == state.device
    )


TRITON_SHMEM_AR_RMSNORM_STATES: dict[
    tuple, TritonShmemAllReduceResidualRMSNorm | None
] = {}


def create_triton_shmem_ar_rmsnorm_state(
    group: dist.ProcessGroup,
    rank_in_group: int,
    max_token_num: int,
    hidden_dim: int,
    dtype: torch.dtype = torch.bfloat16,
    device: torch.device | None = None,
) -> TritonShmemAllReduceResidualRMSNorm | None:
    """Create state, declining only after rank-consistent local setup failure."""
    if not current_platform().is_amd:
        return None
    resolved_device = device or torch.device(f"cuda:{torch.cuda.current_device()}")
    try:
        arch = torch.cuda.get_device_properties(resolved_device).gcnArchName.split(":")[
            0
        ]
        local_errors = _configuration_errors(
            arch=arch,
            world_size=group.size(),
            max_token_num=max_token_num,
            hidden_dim=hidden_dim,
            dtype=dtype,
        )
    except Exception as exc:  # noqa: BLE001 - report capability errors collectively
        local_errors = [f"device query failed: {type(exc).__name__}: {exc}"]

    reports: list[list[str] | None] = [None] * group.size()
    dist.all_gather_object(reports, local_errors, group=group)
    errors = [
        f"rank {rank}: {error}"
        for rank, rank_errors in enumerate(reports)
        for error in rank_errors or ()
    ]
    if errors:
        logger.warning(
            "triton_shmem declined before state creation: %s",
            "; ".join(errors),
        )
        return None

    try:
        return TritonShmemAllReduceResidualRMSNorm(
            group=group,
            rank_in_group=rank_in_group,
            max_token_num=max_token_num,
            hidden_dim=hidden_dim,
            dtype=dtype,
            device=resolved_device,
        )
    except _CollectiveInitializationError as exc:
        logger.warning("triton_shmem declined during state creation: %s", exc)
        return None


def triton_shmem_can_run(
    state: TritonShmemAllReduceResidualRMSNorm,
    input_tensor: torch.Tensor,
    residual: torch.Tensor,
    weight: torch.Tensor,
) -> bool:
    """Return whether this eager call satisfies the backend contract."""
    if torch.cuda.is_current_stream_capturing():
        return False
    return _call_is_supported(state, input_tensor, residual, weight)


def triton_shmem_allreduce_residual_rmsnorm(
    state: TritonShmemAllReduceResidualRMSNorm,
    input_tensor: torch.Tensor,
    residual: torch.Tensor,
    weight: torch.Tensor,
    eps: float = 1e-6,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Run fused all-reduce, residual addition, and RMSNorm.

    Args:
        state: Initialized backend state owning communication workspaces.
        input_tensor: Per-rank partial input with shape ``[M, N]``.
        residual: Replicated residual tensor with shape ``[M, N]``.
        weight: Replicated RMSNorm weight with shape ``[N]``.
        eps: RMSNorm epsilon.

    Returns:
        The normalized output and all-reduced-plus-residual output.
    """
    with kernel_scope(
        "communication",
        "allreduce_residual_rmsnorm",
        input_tensor.dtype,
        kernel_name=state.kernel,
        M=input_tensor.shape[0],
        N=input_tensor.shape[1],
        world_size=state.world_size,
    ):
        return state.fused(input_tensor, residual, weight, eps)


__all__ = [
    "TRITON_SHMEM_AR_RMSNORM_STATES",
    "TritonShmemAllReduceResidualRMSNorm",
    "create_triton_shmem_ar_rmsnorm_state",
    "runtime_context_supported",
    "triton_shmem_allreduce_residual_rmsnorm",
    "triton_shmem_can_run",
]
