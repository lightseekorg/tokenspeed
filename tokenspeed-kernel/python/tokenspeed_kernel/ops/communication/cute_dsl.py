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

"""CuTe DSL projection and communication kernels for NVIDIA Blackwell."""

import logging
from typing import TYPE_CHECKING

import torch
import torch.distributed as dist
from tokenspeed_kernel.ops.communication.fabric import symm_mem_reachable
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
    from tokenspeed_kernel.ops.communication._cute_dsl.gemm_reduce_scatter import (
        GemmReduceScatterState,
    )
    from tokenspeed_kernel.ops.communication._cute_dsl.ll_exchange import (
        LLExchangeState,
    )
    from tokenspeed_kernel.ops.communication._cute_dsl.paged_row_all_gather import (
        PagedRowAllGatherState,
    )

logger = logging.getLogger(__name__)

cute_dsl_fused_fp8_dp_column_linear = error_fn
cute_dsl_fused_fp8_dp_row_linear = error_fn
cute_dsl_fused_bf16_dp_column_linear = error_fn
cute_dsl_fused_bf16_dp_row_linear = error_fn
cute_dsl_gemm_reduce_scatter = error_fn
cute_dsl_paged_row_all_gather = error_fn
cute_dsl_ll_all_gather = error_fn
cute_dsl_ll_all_gather_floats = error_fn
cute_dsl_ll_all_to_all = error_fn

# The paged-row all-gather's modes, in the order a mode code numbers them.
_PAGED_ROW_ALL_GATHER_MODES = ("multicast", "peer")
# A paged-row all-gather slot (degree * cap_bytes) stays under this many bytes,
# which bounds the receive workspace every rank reserves (two slots) before
# the KV cache is sized. It also keeps the kernel's int32 store offsets in
# range, which need only a rank's part of a slot (cap_bytes) under 2**31.
PAGED_ROW_ALL_GATHER_SLOT_BYTES_LIMIT = 2**31


def create_fused_tp_column_projection_state(
    group: dist.ProcessGroup,
    device: torch.device,
    max_rows: int,
    input_size: int,
    output_size: int,
    *,
    weight_dtype: torch.dtype,
) -> "FusedTPProjectionState":
    """Allocate and compile column-projection state before graph capture.

    Args:
        group: Two, four, or eight NVLink-connected peers, in output-shard order.
        device: This peer's SM100 or SM103 CUDA device.
        max_rows: Maximum physical rows on any owner, used to prepare row buckets.
        input_size: Full input width, divisible by 128.
        output_size: Padded output width, divisible by group size times tile alignment.
        weight_dtype: BF16 or E4M3 weights; fixes the prepared GEMM variant.

    Returns:
        Model-owned state; serialize calls and close after destroying graphs.
    """
    from tokenspeed_kernel.ops.communication._cute_dsl.fused_tp_projection import (
        FusedTPProjectionState,
    )

    if weight_dtype not in (torch.bfloat16, torch.float8_e4m3fn):
        raise ValueError("Fused TP projection requires BF16 or E4M3 weights")
    return FusedTPProjectionState(
        group,
        device,
        max_rows,
        input_size,
        output_size,
        row_parallel=False,
        quantized=weight_dtype == torch.float8_e4m3fn,
    )


def create_fused_tp_row_projection_state(
    group: dist.ProcessGroup,
    device: torch.device,
    max_rows: int,
    input_size: int,
    output_size: int,
    *,
    weight_dtype: torch.dtype,
) -> "FusedTPProjectionState":
    """Collectively allocate and compile a workspace before graph capture.

    Args:
        group: Two or four NVLink-connected peers, in input-shard order.
        device: This peer's SM100 or SM103 CUDA device.
        max_rows: Maximum physical rows on any owner, used to prepare row buckets.
        input_size: Full input width, divisible by group size times 128.
        output_size: Padded output width, divisible by group size times tile alignment.
        weight_dtype: BF16 or E4M3 weights; fixes the prepared GEMM variant.

    Returns:
        Model-owned state; serialize calls and close after destroying graphs.
    """
    from tokenspeed_kernel.ops.communication._cute_dsl.fused_tp_projection import (
        FusedTPProjectionState,
    )

    if weight_dtype not in (torch.bfloat16, torch.float8_e4m3fn):
        raise ValueError("Fused TP projection requires BF16 or E4M3 weights")
    return FusedTPProjectionState(
        group,
        device,
        max_rows,
        input_size,
        output_size,
        row_parallel=True,
        quantized=weight_dtype == torch.float8_e4m3fn,
    )


def fused_tp_column_projection_supported(
    group_size: int,
    input_size: int,
    output_size: int,
    max_rows: int,
    dtype: torch.dtype,
    device: torch.device,
    *,
    weight_dtype: torch.dtype,
) -> bool:
    """Return whether the layout and device support fused column projection.

    Args:
        group_size: Number of ranks in an NVLink-connected projection group.
        input_size: Full input channel count.
        output_size: Full padded output channel count.
        max_rows: Positive prepared capacity per owner.
        dtype: Unquantized activation/output dtype.
        device: Device owning the workspace.
        weight_dtype: Prepared BF16 or block-FP8 weight storage.

    Returns:
        Whether this fixed schedule can be prepared. The caller must separately
        agree on peer-memory reachability and eligibility across the group.
    """
    if not (
        current_platform().is_nvidia
        and device.type == "cuda"
        and group_size in (2, 4, 8)
        and max_rows >= 1
        and dtype == torch.bfloat16
        and input_size >= 128
        and input_size % 128 == 0
        and output_size >= group_size * 128
    ):
        return False
    device_arch = torch.cuda.get_device_capability(device)
    if device_arch not in ((10, 0), (10, 3)):
        return False
    aligned_output = output_size % (group_size * 128) == 0
    if weight_dtype == torch.float8_e4m3fn:
        return aligned_output
    # BF16 TP4 DSA QKV-A uses 96-channel alignment for each output shard.
    dsa_tp4_shape = group_size == 4 and (input_size, output_size) == (7168, 14592)
    return (
        weight_dtype == torch.bfloat16
        and group_size in (2, 4)
        and (aligned_output or dsa_tp4_shape)
    )


def fused_tp_row_projection_supported(
    group_size: int,
    input_size: int,
    output_size: int,
    max_rows: int,
    dtype: torch.dtype,
    device: torch.device,
    *,
    weight_dtype: torch.dtype,
) -> bool:
    """Return whether the static layout and device match the kernel constraints.

    Args:
        group_size: Number of ranks in an NVLink-connected projection group.
        input_size: Full input channel count.
        output_size: Full padded output channel count.
        max_rows: Positive prepared capacity per owner.
        dtype: Unquantized activation/output dtype.
        device: Device owning the workspace.
        weight_dtype: Prepared BF16 or block-FP8 weight storage.

    Returns:
        Whether this fixed schedule can be prepared. The caller must separately
        agree on peer-memory reachability and eligibility across the group.
    """
    return (
        current_platform().is_nvidia
        and device.type == "cuda"
        and group_size in (2, 4)
        and max_rows >= 1
        and dtype == torch.bfloat16
        and weight_dtype in (torch.bfloat16, torch.float8_e4m3fn)
        and input_size >= group_size * 128
        and input_size % (group_size * 128) == 0
        and output_size >= group_size * 128
        and output_size % (group_size * 128) == 0
        and torch.cuda.get_device_capability(device) in ((10, 0), (10, 3))
    )


def gemm_reduce_scatter_supported(
    group_size: int,
    out_features: int,
    in_features: int,
    dtype: torch.dtype,
    device: torch.device,
) -> bool:
    """Return whether ``create_gemm_reduce_scatter_state`` can prepare this layout.

    Args:
        group_size: Number of ranks in an NVLink-connected group.
        out_features: Weight rows N.
        in_features: Weight columns K of this rank's shard.
        dtype: Operand and output dtype.
        device: Device owning the workspace.

    Returns:
        Whether the kernel serves the layout. The caller must separately agree
        on NVLink multicast support across the group.
    """
    return (
        current_platform().is_nvidia
        and device.type == "cuda"
        and group_size in (2, 4, 8)
        and out_features % 256 == 0
        and in_features % 64 == 0
        and dtype == torch.bfloat16
        and torch.cuda.get_device_capability(device) in ((10, 0), (10, 3))
    )


def create_gemm_reduce_scatter_state(
    group: dist.ProcessGroup,
    device: torch.device,
    max_rows: int,
    out_features: int,
    in_features: int,
) -> "GemmReduceScatterState":
    """Collectively allocate the workspace and compile before graph capture.

    Args:
        group: Two, four, or eight NVLink-connected ranks with multicast.
        device: This rank's SM100 or SM103 CUDA device.
        max_rows: Most rows of all ranks' blocks one call takes.
        out_features: Weight rows N, divisible by 256.
        in_features: Weight columns K, divisible by 64.

    Returns:
        State for ``cute_dsl_gemm_reduce_scatter``, shared only by serial calls.
    """
    from tokenspeed_kernel.ops.communication._cute_dsl.gemm_reduce_scatter import (
        GemmReduceScatterState,
    )

    return GemmReduceScatterState(group, device, max_rows, out_features, in_features)


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
        cap_bytes: Receive bytes per rank and slot, a whole number of rows,
            with the slot (``degree * cap_bytes``) under
            ``PAGED_ROW_ALL_GATHER_SLOT_BYTES_LIMIT``; equal on every rank.
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
        and torch.cuda.get_device_capability(device) in ((10, 0), (10, 3))
        and degree in (2, 4, 8)
        and row_bytes > 0
        and row_bytes % 16 == 0
        and cap_bytes > 0
        and cap_bytes % row_bytes == 0
        and degree * cap_bytes < PAGED_ROW_ALL_GATHER_SLOT_BYTES_LIMIT
    )
    if local_ok:
        # A rendezvous over hosts the fabric cannot map would hang.
        local_ok = symm_mem_reachable(group)
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
                "is not eligible, or lacks an SM100 or SM103 GPU, group size, a "
                f"slot under {PAGED_ROW_ALL_GATHER_SLOT_BYTES_LIMIT >> 30} GiB, "
                "fabric or kernel module)"
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


def _check_ll_tensors(
    state, input: torch.Tensor, out: torch.Tensor, gather: bool, floats: bool
) -> None:
    """Raise ValueError unless this state's LL exchange serves the operands.

    ``input`` and ``out`` must be contiguous, 16-byte aligned payloads of the
    sizes it serves; ``floats`` checks a float-word gather's size.
    """
    payload = input.numel() * input.element_size()
    if not gather:
        payload //= state.ranks
    if (
        input.device != state.device
        or out.device != state.device
        or not input.is_contiguous()
        or not out.is_contiguous()
        or input.data_ptr() % 16
        or out.data_ptr() % 16
        or not state.serves(payload, floats=floats)
        or (
            not gather and payload * state.ranks != input.numel() * input.element_size()
        )
        or out.numel() * out.element_size() != payload * state.ranks
    ):
        raise ValueError("the LL exchange does not serve these tensors")


def create_ll_exchange_state(
    group: dist.ProcessGroup,
    device: torch.device,
    *,
    cap_bytes: int,
    float_cap_bytes: int,
    timeout_ns: int,
    eligible: bool,
) -> "LLExchangeState | None":
    """Collectively build the one-hop LL exchange for small payloads.

    The state serves an all-gather and an all-to-all, or every rank agrees
    that the group keeps its collectives.

    Every rank of the group calls this at the same program point, before graph
    capture and before the KV cache is sized. Every refusal is voted on before
    any rank returns, so all ranks take the same branch.

    Args:
        group: CPU (gloo) process group over the 2, 4 or 8 ranks; the CUDA
            symmetric-memory backend exchanges handles through its store.
        device: This rank's CUDA device, with its index.
        cap_bytes: Largest per-rank payload, a positive multiple of 16; equal
            on every rank. The workspace holds ``4 * ranks * cap_bytes``.
        float_cap_bytes: Largest per-rank payload of a float-word gather
            (``cute_dsl_ll_all_gather_floats``), a multiple of 16, or 0 for
            none; equal on every rank. Its workspace holds ``2 * ranks *
            float_cap_bytes``.
        timeout_ns: Bound of every in-kernel wait; a peer that does not reach
            the same call within it fails the kernel with a trap.
        eligible: The caller's own verdict for this rank, voted on with the
            rest.

    Returns:
        A state serving ``cute_dsl_ll_all_gather`` and
        ``cute_dsl_ll_all_to_all``, or None on every rank.
    """
    degree = group.size()
    local_ok = (
        eligible
        and current_platform().is_nvidia
        and device.type == "cuda"
        and device.index is not None
        and torch.cuda.get_device_capability(device) in ((10, 0), (10, 3))
        and degree in (2, 4, 8)
        and cap_bytes > 0
        and cap_bytes % 16 == 0
        and 4 * degree * cap_bytes < 2**31
        and float_cap_bytes >= 0
        and float_cap_bytes % 16 == 0
        and 2 * degree * float_cap_bytes < 2**31
        and timeout_ns > 0
    )
    if local_ok:
        # A rendezvous over hosts the fabric cannot map would hang.
        local_ok = symm_mem_reachable(group)
    multicast = False
    if local_ok:
        with torch.cuda.device(device):
            multicast = multimem_available()
        # Imported before the vote: a rank failing after it would strand its peers.
        try:
            from tokenspeed_kernel.ops.communication._cute_dsl.ll_exchange import (
                build_ll_exchange_state,
            )
        except Exception as exc:  # noqa: BLE001 - local failure is voted on
            logger.warning(f"ll_exchange: kernel module import failed: {exc!r}")
            local_ok = False
    # MIN over the group; a negated value gives its MAX.
    votes = torch.tensor(
        [int(local_ok), int(multicast), cap_bytes, -cap_bytes]
        + [float_cap_bytes, -float_cap_bytes],
        dtype=torch.int64,
    )
    dist.all_reduce(votes, op=dist.ReduceOp.MIN, group=group)
    ok, all_multicast, low, high, float_low, float_high = votes.tolist()
    for name, least, most in (
        ("cap_bytes", low, -high),
        ("float_cap_bytes", float_low, -float_high),
    ):
        if ok and least != most:
            raise ValueError(
                f"ll_exchange: {name} differs across ranks "
                f"{dist.get_process_group_ranks(group)}: {least} to {most}"
            )
    if not ok:
        return None
    return build_ll_exchange_state(
        group,
        device,
        cap_bytes=cap_bytes,
        float_cap_bytes=float_cap_bytes,
        multicast=bool(all_multicast),
        timeout_ns=timeout_ns,
    )


if current_platform().is_nvidia:

    @register_kernel(
        "communication",
        "dp_column_linear",
        name="cute_dsl_fused_fp8_dp_column_linear",
        solution="cute_dsl",
        capability=CapabilityRequirement(
            min_arch_version=ArchVersion(10, 0),
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
            min_arch_version=ArchVersion(10, 0),
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
        "dp_column_linear",
        name="cute_dsl_fused_bf16_dp_column_linear",
        solution="cute_dsl",
        capability=CapabilityRequirement(
            min_arch_version=ArchVersion(10, 0),
            max_arch_version=ArchVersion(10, 3),
            vendors=frozenset({"nvidia"}),
        ),
        signatures={
            format_signature(
                inputs=dense_tensor_format(torch.bfloat16),
                weight=dense_tensor_format(torch.bfloat16),
                out=dense_tensor_format(torch.bfloat16),
            )
        },
        priority=Priority.SPECIALIZED,
    )
    def cute_dsl_fused_bf16_dp_column_linear(
        state: "FusedTPProjectionState",
        inputs: torch.Tensor,
        weight: torch.Tensor,
        weight_scales: None,
        owner_rows: tuple[int, ...],
        out: torch.Tensor,
    ) -> torch.Tensor:
        """Run BF16 all-gather, column GEMM and inverse all-to-all together.

        Args:
            state: Collectively prepared BF16 state, used serially on one stream.
            inputs: Contiguous BF16 [R, K] rows owned by this rank.
            weight: Contiguous BF16 [N/P, K] output-channel shard.
            weight_scales: Explicit None; BF16 operands are not quantized.
            owner_rows: Physical row counts in group-rank order on every peer;
                counts may differ or be zero.
            out: Caller-owned contiguous BF16 [R, N] destination.

        Returns:
            out, valid across later calls and state teardown. Destroy graphs
            referencing state before closing it collectively.
        """
        return state.run(inputs, weight, weight_scales, owner_rows, out)

    @register_kernel(
        "communication",
        "dp_row_linear",
        name="cute_dsl_fused_bf16_dp_row_linear",
        solution="cute_dsl",
        capability=CapabilityRequirement(
            min_arch_version=ArchVersion(10, 0),
            max_arch_version=ArchVersion(10, 3),
            vendors=frozenset({"nvidia"}),
        ),
        signatures={
            format_signature(
                inputs=dense_tensor_format(torch.bfloat16),
                weight=dense_tensor_format(torch.bfloat16),
                out=dense_tensor_format(torch.bfloat16),
            )
        },
        priority=Priority.SPECIALIZED,
    )
    def cute_dsl_fused_bf16_dp_row_linear(
        state: "FusedTPProjectionState",
        inputs: torch.Tensor,
        weight: torch.Tensor,
        weight_scales: None,
        owner_rows: tuple[int, ...],
        out: torch.Tensor,
    ) -> torch.Tensor:
        """Run BF16 all-to-all, row GEMM and reduce-scatter together.

        Args:
            state: Collectively prepared BF16 state, used serially on one stream.
            inputs: Contiguous BF16 [R, K] rows owned by this rank.
            weight: Contiguous BF16 [N, K/P] input-channel shard.
            weight_scales: Explicit None; BF16 operands are not quantized.
            owner_rows: Physical row counts in group-rank order on every peer;
                counts may differ or be zero.
            out: Caller-owned contiguous BF16 [R, N] destination.

        Returns:
            out. GEMM partials round to BF16 before a rank-ordered FP32 sum and
            final BF16 rounding. Destroy graphs before collective state teardown.
        """
        return state.run(inputs, weight, weight_scales, owner_rows, out)

    @register_kernel(
        "communication",
        "gemm_reduce_scatter",
        name="cute_dsl_gemm_reduce_scatter",
        solution="cute_dsl",
        capability=CapabilityRequirement(
            min_arch_version=ArchVersion(10, 0),
            max_arch_version=ArchVersion(10, 3),
            vendors=frozenset({"nvidia"}),
        ),
        signatures=format_signatures(("x", "weight"), "dense", {torch.bfloat16}),
        priority=Priority.SPECIALIZED,
    )
    def cute_dsl_gemm_reduce_scatter(
        state: "GemmReduceScatterState", x: torch.Tensor, weight: torch.Tensor
    ) -> torch.Tensor:
        """This rank's block of ``reduce_scatter(x @ weight.T)`` in one kernel.

        Each rank's BF16 partial tiles are summed in FP32 by their block's
        owner and rounded once, so the result is at least as precise as a
        BF16 GEMM followed by a ring reduce-scatter.

        Args:
            state: From ``create_gemm_reduce_scatter_state``.
            x: Contiguous BF16 ``[ranks * B, K]``, rank ``r``'s block rows
                ``[r * B, (r + 1) * B)``, equal shape on every rank.
            weight: Contiguous BF16 ``[N, K]``.

        Returns:
            BF16 ``[B, N]``: block ``rank`` of the summed product.
        """
        return state.run(x, weight)

    @register_kernel(
        "communication",
        "paged_row_all_gather",
        name="cute_dsl_paged_row_all_gather",
        solution="cute_dsl",
        capability=CapabilityRequirement(
            min_arch_version=ArchVersion(10, 0),
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

    @register_kernel(
        "communication",
        "ll_all_gather",
        name="cute_dsl_ll_all_gather",
        solution="cute_dsl",
        capability=CapabilityRequirement(
            min_arch_version=ArchVersion(10, 0),
            max_arch_version=ArchVersion(10, 3),
            vendors=frozenset({"nvidia"}),
        ),
        signatures=format_signatures(("input",), "dense", {torch.uint8}),
        priority=Priority.SPECIALIZED,
    )
    def cute_dsl_ll_all_gather(
        state: "LLExchangeState", input: torch.Tensor, out: torch.Tensor
    ) -> torch.Tensor:
        """Gather every rank's ``input`` bytes into ``out`` in rank order.

        Args:
            state: From ``create_ll_exchange_state``.
            input: Contiguous 16-byte aligned payload of any dtype on the
                state's device, ``state.serves`` its byte size, equal on
                every rank.
            out: Contiguous 16-byte aligned destination of ``ranks`` times
                the payload; rank ``r``'s bytes land at offset ``r`` times it.

        Returns:
            ``out``, bit for bit what an all-gather of ``input`` gives.
        """
        _check_ll_tensors(state, input, out, gather=True, floats=False)
        state.launch(input, out, gather=True, floats=False)
        return out

    @register_kernel(
        "communication",
        "ll_all_gather_floats",
        name="cute_dsl_ll_all_gather_floats",
        solution="cute_dsl",
        capability=CapabilityRequirement(
            min_arch_version=ArchVersion(10, 0),
            max_arch_version=ArchVersion(10, 3),
            vendors=frozenset({"nvidia"}),
        ),
        signatures=format_signatures(("input",), "dense", {torch.uint8}),
        priority=Priority.SPECIALIZED,
    )
    def cute_dsl_ll_all_gather_floats(
        state: "LLExchangeState", input: torch.Tensor, out: torch.Tensor
    ) -> torch.Tensor:
        """Gather every rank's float words into ``out`` in rank order.

        The words travel as themselves rather than as LL elements, half the
        bytes: an armed slot holds ``0xFFFFFFFF``, which a sender stores as
        ``0x7FFFFFFF``. Both are NaN as one FP32, as two BF16 halves' high
        half, and as four E4M3 bytes' top byte, so a payload of FP32, BF16, or
        FP8 words comes back with only that NaN's bits changed.

        Args:
            state: From ``create_ll_exchange_state``.
            input: Contiguous 16-byte aligned float words on the state's
                device; ``state.serves`` its byte size with ``floats``, equal
                on every rank.
            out: Contiguous 16-byte aligned destination of ``ranks`` times
                the payload; rank ``r``'s words land at offset ``r`` times it.

        Returns:
            ``out``.
        """
        _check_ll_tensors(state, input, out, gather=True, floats=True)
        state.launch(input, out, gather=True, floats=True)
        return out

    @register_kernel(
        "communication",
        "ll_all_to_all",
        name="cute_dsl_ll_all_to_all",
        solution="cute_dsl",
        capability=CapabilityRequirement(
            min_arch_version=ArchVersion(10, 0),
            max_arch_version=ArchVersion(10, 3),
            vendors=frozenset({"nvidia"}),
        ),
        signatures=format_signatures(("input",), "dense", {torch.uint8}),
        priority=Priority.SPECIALIZED,
    )
    def cute_dsl_ll_all_to_all(
        state: "LLExchangeState", input: torch.Tensor, out: torch.Tensor
    ) -> torch.Tensor:
        """Send ``input``'s ``r``-th equal part to rank ``r``; receive into ``out``.

        Args:
            state: From ``create_ll_exchange_state``.
            input: Contiguous 16-byte aligned ``ranks`` equal parts of any
                dtype; ``state.serves`` one part's byte size, equal on every rank.
            out: Contiguous 16-byte aligned destination of the same size; the
                part rank ``r`` sent lands at part ``r``.

        Returns:
            ``out``, bit for bit what an equal-split all-to-all gives.
        """
        _check_ll_tensors(state, input, out, gather=False, floats=False)
        state.launch(input, out, gather=False, floats=False)
        return out
