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

"""Fused block-FP8 DP column and row projections on GB300."""

from typing import TYPE_CHECKING

import torch
import torch.distributed as dist
from tokenspeed_kernel.platform import (
    ArchVersion,
    CapabilityRequirement,
    current_platform,
)
from tokenspeed_kernel.registry import Priority, error_fn, register_kernel
from tokenspeed_kernel.signature import (
    ScaleFormat,
    dense_tensor_format,
    format_signature,
    tensor_format,
)

if TYPE_CHECKING:
    from tokenspeed_kernel.ops.communication._cute_dsl.fused_tp_projection import (
        FusedTPProjectionState,
    )

cute_dsl_fused_fp8_dp_column_linear = error_fn
cute_dsl_fused_fp8_dp_row_linear = error_fn


def create_fused_tp_column_projection_state(
    group: dist.ProcessGroup,
    device: torch.device,
    max_rows: int,
    input_size: int,
    output_size: int,
) -> "FusedTPProjectionState":
    """Allocate and compile column-projection state before graph capture.

    Args:
        group: Two, four, or eight node-local peers, in output-shard order.
        device: This peer's GB300 CUDA device.
        max_rows: Maximum physical rows on any owner, used to prepare row buckets.
        input_size: Full input width, divisible by 128.
        output_size: Padded output width, divisible by group size times 128.

    Returns:
        Model-owned state; serialize calls and close after destroying graphs.
    """
    from tokenspeed_kernel.ops.communication._cute_dsl.fused_tp_projection import (
        FusedTPProjectionState,
    )

    return FusedTPProjectionState(
        group, device, max_rows, input_size, output_size, row_parallel=False
    )


def create_fused_tp_row_projection_state(
    group: dist.ProcessGroup,
    device: torch.device,
    max_rows: int,
    input_size: int,
    output_size: int,
) -> "FusedTPProjectionState":
    """Collectively allocate and compile a workspace before graph capture.

    Args:
        group: Two or four node-local peers, in input-shard order.
        device: This peer's GB300 CUDA device.
        max_rows: Maximum physical rows on any owner, used to prepare row buckets.
        input_size: Full input width, divisible by group size times 128.
        output_size: Padded output width, divisible by group size times 128.

    Returns:
        Model-owned state; serialize calls and close after destroying graphs.
    """
    from tokenspeed_kernel.ops.communication._cute_dsl.fused_tp_projection import (
        FusedTPProjectionState,
    )

    return FusedTPProjectionState(
        group, device, max_rows, input_size, output_size, row_parallel=True
    )


def fused_tp_column_projection_supported(
    group_size: int,
    input_size: int,
    output_size: int,
    max_rows: int,
    dtype: torch.dtype,
    device: torch.device,
) -> bool:
    """Return whether the layout and device support fused column projection.

    Args:
        group_size: Number of ranks in a node-local projection group.
        input_size: Full input channel count.
        output_size: Full padded output channel count.
        max_rows: Positive prepared capacity per owner.
        dtype: Unquantized activation/output dtype.
        device: Device owning the workspace.

    Returns:
        Whether this fixed schedule can be prepared. The caller must separately
        agree on node locality and eligibility across the group.
    """
    return (
        current_platform().is_nvidia
        and device.type == "cuda"
        and group_size in (2, 4, 8)
        and input_size == 7168
        and output_size
        == ((49376 + group_size * 128 - 1) // (group_size * 128)) * group_size * 128
        and max_rows >= 1
        and dtype == torch.bfloat16
        and torch.cuda.get_device_capability(device) == (10, 3)
        and torch.cuda.get_device_properties(device).multi_processor_count == 152
    )


def fused_tp_row_projection_supported(
    group_size: int,
    input_size: int,
    output_size: int,
    max_rows: int,
    dtype: torch.dtype,
    device: torch.device,
) -> bool:
    """Return whether the static layout and device match the kernel constraints.

    Args:
        group_size: Number of ranks in a node-local projection group.
        input_size: Full input channel count.
        output_size: Full padded output channel count.
        max_rows: Positive prepared capacity per owner.
        dtype: Unquantized activation/output dtype.
        device: Device owning the workspace.

    Returns:
        Whether this fixed schedule can be prepared. The caller must separately
        agree on node locality and eligibility across the group.
    """
    return (
        current_platform().is_nvidia
        and device.type == "cuda"
        and group_size in (2, 4)
        and input_size == 12288
        and output_size == 7168
        and max_rows >= 1
        and dtype == torch.bfloat16
        and torch.cuda.get_device_capability(device) == (10, 3)
        and torch.cuda.get_device_properties(device).multi_processor_count == 152
    )


if current_platform().is_nvidia:

    @register_kernel(
        "communication",
        "dp_column_linear",
        name="cute_dsl_fused_fp8_dp_column_linear",
        solution="cute_dsl",
        capability=CapabilityRequirement(
            min_arch_version=ArchVersion(10, 3),
            max_arch_version=ArchVersion(10, 3),
            vendors=frozenset({"nvidia"}),
        ),
        signatures={
            format_signature(
                inputs=dense_tensor_format(torch.bfloat16),
                weight=tensor_format(
                    "mxfp8",
                    torch.float8_e4m3fn,
                    scale=ScaleFormat(
                        storage_dtype=torch.float32,
                        granularity="block",
                        block_shape=(128, 128),
                    ),
                ),
                out=dense_tensor_format(torch.bfloat16),
            )
        },
        priority=Priority.SPECIALIZED,
    )
    def cute_dsl_fused_fp8_dp_column_linear(
        state: "FusedTPProjectionState",
        inputs: torch.Tensor,
        weight: torch.Tensor,
        weight_scales: torch.Tensor,
        owner_rows: tuple[int, ...],
        out: torch.Tensor,
    ) -> torch.Tensor:
        """Run fused block-FP8 DP column-parallel linear on locally owned rows.

        Fuses input all-gather, quantization, GEMM, and inverse all-to-all in one
        kernel per rank, returning every output channel to its token owner.

        Args:
            state: Collectively prepared workspace, shared only by serial calls.
            inputs: Contiguous BF16 [R, K] containing this owner's actual rows.
            weight: Contiguous FP8 E4M3 [N/P, K] local output-channel shard,
                where P is the prepared group size and N the padded output width.
            weight_scales: Contiguous FP32 [N/(P*128), K/128] block scales.
            owner_rows: Physical row counts in group-rank order, identical on all
                peers. Counts may differ or be zero; at least one must be positive.
            out: Caller-owned contiguous BF16 [R, N] destination.

        Returns:
            out, valid across subsequent calls. All peers must participate
            in the same order. Destroy referencing graphs before closing state.
        """
        return state.run(inputs, weight, weight_scales, owner_rows, out)

    @register_kernel(
        "communication",
        "dp_row_linear",
        name="cute_dsl_fused_fp8_dp_row_linear",
        solution="cute_dsl",
        capability=CapabilityRequirement(
            min_arch_version=ArchVersion(10, 3),
            max_arch_version=ArchVersion(10, 3),
            vendors=frozenset({"nvidia"}),
        ),
        signatures={
            format_signature(
                inputs=inputs_format,
                weight=tensor_format(
                    "mxfp8",
                    torch.float8_e4m3fn,
                    scale=ScaleFormat(
                        storage_dtype=torch.float32,
                        granularity="block",
                        block_shape=(128, 128),
                    ),
                ),
                out=dense_tensor_format(torch.bfloat16),
            )
            for inputs_format in (
                dense_tensor_format(torch.bfloat16),
                tensor_format(
                    "mxfp8",
                    torch.float8_e4m3fn,
                    scale=ScaleFormat(
                        storage_dtype=torch.float32,
                        granularity="block",
                        block_shape=(1, 128),
                    ),
                ),
            )
        },
        priority=Priority.SPECIALIZED,
    )
    def cute_dsl_fused_fp8_dp_row_linear(
        state: "FusedTPProjectionState",
        inputs: torch.Tensor,
        weight: torch.Tensor,
        weight_scales: torch.Tensor,
        owner_rows: tuple[int, ...],
        out: torch.Tensor,
        *,
        input_scales: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Run fused block-FP8 DP row-parallel linear on locally owned rows.

        Fuses input all-to-all, quantization, GEMM, and reduce-scatter in one
        kernel per rank, returning every output channel to its token owner.

        Args:
            state: Collectively prepared workspace, shared only by serial calls.
            inputs: Contiguous BF16 or E4M3 [R, K] containing this owner's rows.
            input_scales: Contiguous FP32 [R, K/128] for E4M3 input; None for BF16.
                Supplied FP8 values and scales bypass quantization.
            weight: Contiguous FP8 E4M3 [N, K/P] local input-channel shard,
                where P is the prepared group size and N the padded output width.
            weight_scales: Contiguous FP32 [N/128, K/(P*128)] block scales.
            owner_rows: Physical row counts in group-rank order, identical on all
                peers. Counts may differ or be zero; at least one must be positive.
            out: Caller-owned contiguous BF16 [R, N] destination.

        Returns:
            out, valid across subsequent calls. All peers must participate
            in the same order. Destroy referencing graphs before closing state.
        """
        return state.run(
            inputs, weight, weight_scales, owner_rows, out, input_scales=input_scales
        )
