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

"""Fused block-FP8 DP column and row projections and the fused paged-row
all-gather on GB300."""

import logging
from typing import TYPE_CHECKING

import torch
import torch.distributed as dist
from tokenspeed_kernel.ops.communication.fabric import multicast_reachable
from tokenspeed_kernel.ops.communication.multimem import multimem_available
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
    format_signatures,
    tensor_format,
)

if TYPE_CHECKING:
    from tokenspeed_kernel.ops.communication._cute_dsl.fused_tp_projection import (
        FusedTPProjectionState,
    )
    from tokenspeed_kernel.ops.communication._cute_dsl.paged_row_all_gather import (
        PagedRowAllGatherState,
    )

logger = logging.getLogger(__name__)

cute_dsl_fused_fp8_dp_column_linear = error_fn
cute_dsl_fused_fp8_dp_row_linear = error_fn
cute_dsl_paged_row_all_gather = error_fn

# The paged-row all-gather's modes, in the order a mode code numbers them.
_PAGED_ROW_ALL_GATHER_MODES = ("multicast", "peer")


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


def create_paged_row_all_gather_state(
    group: dist.ProcessGroup,
    device: torch.device,
    *,
    row_bytes: int,
    cap_bytes: int,
    timeout_ns: int,
    modes: tuple[str, ...],
    eligible: bool,
) -> "PagedRowAllGatherState | None":
    """Collectively build, compile and warm a paged-row all-gather, or agree
    that the group keeps its existing all-gather.

    Every rank of the group calls this at the same program point, before
    graph capture and before the KV cache is sized. Every refusal is voted on
    before any rank returns, so all ranks take the same branch; what is voted
    on, what is fatal instead, and the contract a caller keeps are in the
    ``communication`` README.

    Args:
        group: CPU (gloo) process group over the 2, 4 or 8 gather ranks; the
            CUDA symmetric-memory backend exchanges handles through its store.
            Without the fabric map (``gather_fabric_map``, which the server's
            distributed initializer gathers) the group keeps its all-gather.
        device: This rank's CUDA device, with its index.
        row_bytes: Bytes per cache row, a multiple of 16; equal on every rank.
        cap_bytes: Receive bytes per rank and slot, a whole number of rows;
            equal on every rank.
        timeout_ns: Bound of every in-kernel wait; a peer that does not reach
            the same window within it fails the kernel with a trap.
        modes: Modes in preference order, from ``multicast`` (one NVLink
            multicast store reaches every peer) and ``peer`` (a store per
            peer); equal on every rank.
        eligible: The caller's own verdict for this rank, voted on with the
            rest.

    Returns:
        A state whose first window (epoch 1) has run, or None on every rank.
    """
    # Bad arguments are a caller's bug; they are voted on like any refusal and
    # raised afterwards, so no peer waits in a vote this rank never joins.
    modes_ok = (
        bool(modes)
        and len(set(modes)) == len(modes)
        and all(mode in _PAGED_ROW_ALL_GATHER_MODES for mode in modes)
    )
    degree = group.size()
    ranks = dist.get_process_group_ranks(group)
    local_ok = (
        modes_ok
        and eligible
        and current_platform().is_nvidia
        and device.type == "cuda"
        and device.index is not None
        and torch.cuda.get_device_capability(device) == (10, 3)
        and degree in (2, 4, 8)
        and row_bytes > 0
        and row_bytes % 16 == 0
        and cap_bytes > 0
        and cap_bytes % row_bytes == 0
        # Store offsets inside a rank's part of a slot stay int32 in the kernel.
        and degree * cap_bytes < 2**31
    )
    if local_ok:
        # A rendezvous over hosts the fabric cannot map would hang.
        local_ok = multicast_reachable(group)
    multicast = False
    if local_ok:
        with torch.cuda.device(device):
            multicast = multimem_available()
        # Imported only where the platform can load it, and before the vote:
        # a rank failing after it would leave its peers waiting in the next.
        try:
            from tokenspeed_kernel.ops.communication._cute_dsl.paged_row_all_gather import (
                build_paged_row_all_gather_state,
            )
        except Exception as exc:  # noqa: BLE001 - local failure is voted on
            logger.warning(
                f"paged_row_all_gather: kernel module import failed: {exc!r}"
            )
            local_ok = False
    mode_code = (
        sum(
            (_PAGED_ROW_ALL_GATHER_MODES.index(mode) + 1) * 3**i
            for i, mode in enumerate(modes)
        )
        if modes_ok
        else -1
    )
    # MIN over the group; a negated value gives its MAX, so equal parameters
    # show as MIN == MAX.
    votes = torch.tensor(
        [
            int(local_ok),
            int(multicast),
            row_bytes,
            -row_bytes,
            cap_bytes,
            -cap_bytes,
            mode_code,
            -mode_code,
        ],
        dtype=torch.int64,
    )
    dist.all_reduce(votes, op=dist.ReduceOp.MIN, group=group)
    ok, all_multicast, *bounds = votes.tolist()
    if not modes_ok:
        raise ValueError(
            f"modes must be distinct, non-empty and drawn from "
            f"{_PAGED_ROW_ALL_GATHER_MODES}, got {modes}"
        )
    if ok and any(low != -high for low, high in zip(bounds[::2], bounds[1::2])):
        raise ValueError(
            "paged_row_all_gather: row_bytes, cap_bytes or modes differ across "
            f"ranks {ranks}"
        )
    if not ok:
        if group.rank() == 0:
            logger.info(
                f"paged_row_all_gather: ranks {ranks} keep their all-gather (a rank "
                "is not eligible, or lacks the GB300 platform, group size, fabric "
                "or kernel module)"
            )
        return None
    return build_paged_row_all_gather_state(
        group,
        device,
        row_bytes=row_bytes,
        cap_bytes=cap_bytes,
        timeout_ns=timeout_ns,
        modes=modes,
        multicast=bool(all_multicast),
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

    @register_kernel(
        "communication",
        "paged_row_all_gather",
        name="cute_dsl_paged_row_all_gather",
        solution="cute_dsl",
        capability=CapabilityRequirement(
            min_arch_version=ArchVersion(10, 3),
            max_arch_version=ArchVersion(10, 3),
            vendors=frozenset({"nvidia"}),
        ),
        signatures=format_signatures(("fetch_slots",), "dense", {torch.int32}),
        priority=Priority.SPECIALIZED,
    )
    def cute_dsl_paged_row_all_gather(
        state: "PagedRowAllGatherState",
        plane: torch.Tensor,
        fetch_slots: torch.Tensor,
        owned: int,
        send_rows: int,
        header: int,
    ) -> tuple[int, torch.cuda.Event]:
        """All-gather paged cache rows, packed, into the next epoch's slot.

        Every rank stores its ``fetch_slots[:owned]`` rows of ``plane`` into
        rows ``[rank * send_rows, rank * send_rows + owned)`` of every rank's
        slot, on ``state.stream`` after the current stream's work so far;
        rows ``[owned, send_rows)`` of each rank are not written. The
        caller's contract (the same windows in the same order on every rank,
        one epoch per slot in use) is in the ``communication`` README.

        Args:
            state: From ``create_paged_row_all_gather_state``.
            plane: ``[rows, *row]`` cache rows of any dtype on the state's
                device, contiguous, ``row_bytes`` per row, 16-byte aligned.
            fetch_slots: 1-D int32 tensor on the state's device: this rank's
                plane rows in send order, at least ``max(owned, 1)`` entries.
            owned: This rank's row count, ``0 <= owned <= send_rows``.
            send_rows: Rows reserved per rank, equal on every rank, with
                ``send_rows * row_bytes <= cap_bytes``.
            header: Window identity, equal on every rank, below 2**63; a
                mismatch fails the kernel.

        Returns:
            The window's epoch, whose slot ``state.receive_view(epoch)`` is,
            and the event recorded after the kernel, which the slot's readers
            wait for; the launch of ``epoch + 2`` records it again.
        """
        return state.launch(plane, fetch_slots, owned, send_rows, header)
