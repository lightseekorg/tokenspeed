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

"""TokenSpeed bit-preserving, intra-node Lamport A2A.

CUDA C++ expresses the system-scoped 64-bit packet transactions explicitly.
Projection callers own topology admission, padding and NCCL fallback.

Both packet and chunk exchange currently require exactly four GPUs per
process group on one host. Peer indexing and scratch layouts specialize for
four peers; other group sizes are rejected, with no implicit NCCL fallback.
"""

import socket
from pathlib import Path

import torch
import torch.distributed as dist
import torch.distributed._symmetric_memory as symm
from tokenspeed_kernel.registry import register_kernel
from tokenspeed_kernel.signature import format_signatures


class TokenSpeedA2ALamportState:
    """Persistent TP4 scratch, initialized collectively before graph capture.

    The supplied process group must contain exactly four GPUs, irrespective
    of the total number of GPUs in the job.

    All ranks must call in the same order, with equal physical shapes, on one
    serialized stream. Outputs are borrowed unless the caller supplies storage.
    Empty logical owners must participate using zero-padded physical rows.
    Three generations protect readers from faster peers; each 64-bit packet
    embeds a 32-bit epoch and 32 payload bits, preserving signed zero and NaN
    payloads exactly.
    """

    def __init__(self, group, max_rows, channels, device, blocks):
        from tokenspeed_kernel.thirdparty.flashinfer.jit import build_cuda_module

        contracts = [None] * group.size()
        dist.all_gather_object(
            contracts, (socket.gethostname(), max_rows, channels, blocks), group=group
        )
        if any(contract != contracts[0] for contract in contracts):
            raise ValueError(
                "A2A requires one host and identical capacity, channels, and grid on all ranks"
            )
        if group.size() != 4 or max_rows < 1 or channels < 8 or channels % 8:
            raise ValueError("Requires TP4, positive capacity, channels divisible by 8")
        if 3 * max_rows * channels // 2 > 2**31 - 1:
            raise ValueError("A2A workspace exceeds 32-bit indexing")
        if (
            not 1
            <= blocks
            <= torch.cuda.get_device_properties(device).multi_processor_count
        ):
            raise ValueError("Grid must be positive and no larger than the SM count")
        self.group = group
        self.rank = group.rank()
        self.max_rows = max_rows
        self.channels = channels
        self.blocks = blocks
        self.words = max_rows * channels // 2
        self.chunk_threshold_bytes = None
        self.chunk_scratch = self.chunk_handle = self.chunk_peers = None
        self.chunk_flags = self.chunk_flag_handle = self.chunk_flag_peers = None
        self.chunk_control = self.chunk_module = None
        self.fp8_output = self.fp8_scales = None
        self.chunk_capacity = self.words // 2
        with torch.inference_mode(False):
            self.scratch = symm.empty(
                (3 * self.words,), dtype=torch.int64, device=device
            )
        self.scratch.zero_()
        self.handle = symm.rendezvous(self.scratch, group=group)
        self.peers = torch.tensor(
            [
                self.handle.get_buffer(
                    r, self.scratch.shape, self.scratch.dtype
                ).data_ptr()
                for r in range(4)
            ],
            dtype=torch.int64,
            device=device,
        )
        self.control = torch.tensor([1, 0], dtype=torch.int32, device=device)
        self.output = torch.empty(
            max_rows * channels, dtype=torch.bfloat16, device=device
        )
        self.module = build_cuda_module(
            "tokenspeed_lamport_a2a_v1",
            [Path(__file__).with_name("lamport_a2a.cu")],
        )
        torch.cuda.synchronize(device)
        dist.barrier(group=group)

    def prepare_fp8_quantization(self):
        """Allocate borrowed FP8 values and MN-major FP32 scales before capture.

        Only the forward [M,K] -> [4*M,K/4] exchange is quantized. Every
        channel shard must contain complete 128-element quantization groups.
        Packet/chunk storage and generations remain shared with BF16 calls.
        """
        if self.channels % 512:
            raise ValueError("Fused A2A quantization requires K divisible by 512")
        if self.fp8_output is not None:
            return
        self.fp8_output = torch.empty(
            (4 * self.max_rows, self.channels // 4),
            dtype=torch.float8_e4m3fn,
            device=self.output.device,
        )
        self.fp8_scales = torch.empty(
            self.max_rows * self.channels // 128,
            dtype=torch.float32,
            device=self.output.device,
        )

    def prepare_chunk_exchange(self, threshold_bytes):
        """Collectively enable vectorized chunk publication before capture.

        Inputs at least threshold_bytes use chunk flags; smaller inputs retain
        the packet kernel. All peers must agree. Channels must be divisible by
        32. Separate scratch/generations are essential: raw chunk payload must
        never be interpreted as packet readiness tags after a size transition.
        Extra scratch is three payload buffers plus per-CTA flags.
        """
        from tokenspeed_kernel.thirdparty.flashinfer.jit import build_cuda_module

        thresholds = [None] * self.group.size()
        dist.all_gather_object(thresholds, threshold_bytes, group=self.group)
        if any(value != threshold_bytes for value in thresholds):
            raise ValueError("All A2A peers must agree on the chunk threshold")
        if threshold_bytes <= 0 or self.channels % 32:
            raise ValueError(
                "Chunk exchange requires a positive threshold and K divisible by 32"
            )
        if self.chunk_threshold_bytes is not None:
            if self.chunk_threshold_bytes != threshold_bytes:
                raise ValueError("A prepared chunk threshold cannot be changed")
            return
        device = self.output.device
        with torch.inference_mode(False):
            self.chunk_scratch = symm.empty(
                (3 * self.chunk_capacity,), dtype=torch.int64, device=device
            )
            self.chunk_flags = symm.empty(
                (3 * 4 * self.blocks,), dtype=torch.int64, device=device
            )
        self.chunk_flags.zero_()
        self.chunk_handle = symm.rendezvous(self.chunk_scratch, group=self.group)
        self.chunk_flag_handle = symm.rendezvous(self.chunk_flags, group=self.group)
        self.chunk_peers = torch.tensor(
            [
                self.chunk_handle.get_buffer(
                    r, self.chunk_scratch.shape, torch.int64
                ).data_ptr()
                for r in range(4)
            ],
            dtype=torch.int64,
            device=device,
        )
        self.chunk_flag_peers = torch.tensor(
            [
                self.chunk_flag_handle.get_buffer(
                    r, self.chunk_flags.shape, torch.int64
                ).data_ptr()
                for r in range(4)
            ],
            dtype=torch.int64,
            device=device,
        )
        self.chunk_control = torch.tensor([1, 0], dtype=torch.int32, device=device)
        self.chunk_module = build_cuda_module(
            "tokenspeed_chunk_a2a_v1",
            [Path(__file__).with_name("chunk_a2a.cu")],
        )
        torch.cuda.synchronize(device)
        dist.barrier(group=self.group)
        self.chunk_threshold_bytes = threshold_bytes


@register_kernel(
    family="communication",
    mode="all_to_all",
    name="cuda_tokenspeed_a2a_lamport",
    solution="cuda",
    signatures=format_signatures(("inputs",), "dense", {torch.bfloat16}),
)
def tokenspeed_a2a_lamport(state, inputs, inverse, out: torch.Tensor | None):
    """Exchange BF16 channel shards while preserving every input bit.

    Forward: [M,K] -> [4*M,K/4]. Inverse: [4*M,K/4] -> [M,K].
    inputs must be contiguous BF16 with a 16-byte-aligned address.
    state owns persistent scratch; inverse explicitly chooses layout.
    out=None returns borrowed state.output, valid until the next call. A supplied
    out must have the exact result shape, be contiguous BF16 on the same device,
    and have a 16-byte-aligned address without aliasing inputs or state storage.
    The kernel writes directly into out and returns it, without an extra copy.
    Every peer must use the same shape/direction on one serialized stream.
    Keep state and output storage alive until their queued consumers finish.
    """
    if inputs.ndim != 2:
        raise ValueError("A2A input must be a matrix")
    if inputs.untyped_storage().data_ptr() in (
        state.output.untyped_storage().data_ptr(),
        state.scratch.untyped_storage().data_ptr(),
    ):
        raise ValueError("Input must not alias A2A output or communication scratch")
    if (
        state.chunk_scratch is not None
        and inputs.untyped_storage().data_ptr()
        == state.chunk_scratch.untyped_storage().data_ptr()
    ):
        raise ValueError("Input must not alias chunk communication scratch")
    rows = inputs.shape[0] // 4 if inverse else inputs.shape[0]
    shape = (4 * rows, state.channels // 4) if inverse else (rows, state.channels)
    if (
        tuple(inputs.shape) != shape
        or not 1 <= rows <= state.max_rows
        or inputs.dtype != torch.bfloat16
        or inputs.device != state.output.device
        or not inputs.is_contiguous()
        or inputs.data_ptr() % 16
    ):
        raise ValueError(
            "Input shape, dtype, device, contiguity, or 16-byte alignment "
            "violates the A2A contract"
        )
    result_shape = (
        (rows, state.channels) if inverse else (4 * rows, state.channels // 4)
    )
    if out is None:
        output = state.output[: inputs.numel()].view(result_shape)
    else:
        if (
            tuple(out.shape) != result_shape
            or out.dtype != torch.bfloat16
            or out.device != inputs.device
            or not out.is_contiguous()
            or out.data_ptr() % 16
        ):
            raise ValueError(
                "Output must match the A2A result shape, dtype, device, "
                "contiguity, and 16-byte alignment"
            )
        # A separate destination preserves ownership across subsequent calls.
        # Reject shared storage conservatively, including protocol metadata.
        output_storage = out.untyped_storage().data_ptr()
        if any(
            buffer is not None and output_storage == buffer.untyped_storage().data_ptr()
            for buffer in (
                inputs,
                state.output,
                state.scratch,
                state.control,
                state.peers,
                state.chunk_scratch,
                state.chunk_flags,
                state.chunk_control,
                state.chunk_peers,
                state.chunk_flag_peers,
            )
        ):
            raise ValueError("Output must not alias A2A inputs or state storage")
        output = out
    if (
        state.chunk_threshold_bytes is not None
        and inputs.numel() * inputs.element_size() >= state.chunk_threshold_bytes
    ):
        state.chunk_module.exchange_chunk(
            state.chunk_flag_peers,
            inputs,
            output,
            state.chunk_peers,
            state.chunk_control,
            state.chunk_capacity,
            rows,
            state.channels,
            state.rank,
            state.blocks,
            inverse,
        )
        return output
    state.module.exchange(
        inputs,
        output,
        state.peers,
        state.control,
        state.words,
        rows,
        state.channels,
        state.rank,
        state.blocks,
        inverse,
    )
    return output


@register_kernel(
    family="communication",
    mode="all_to_all_fp8_quantize",
    name="cuda_tokenspeed_a2a_lamport_fp8_quantize",
    solution="cuda",
    signatures=format_signatures(("inputs",), "dense", {torch.bfloat16}),
)
def tokenspeed_a2a_lamport_fp8_quantize(state, inputs):
    """Exchange BF16 shards and quantize ready 128-element groups in one kernel.

    Args:
        state: TokenSpeedA2ALamportState with FP8 buffers prepared before capture.
        inputs: Contiguous BF16 [M,K], with equal positive physical M on all peers.

    Returns:
        Borrowed FP8 [4*M,K/4] and MN-major FP32 scales [K/512,4*M], valid until
        the next quantized call. Consumers use ordinary stream ordering; no
        external tile-readiness protocol or additional quantization is needed.
    """
    if (
        state.fp8_output is None
        or inputs.ndim != 2
        or inputs.dtype != torch.bfloat16
        or inputs.device != state.output.device
        or not inputs.is_contiguous()
        or inputs.shape[1] != state.channels
        or not 0 < inputs.shape[0] <= state.max_rows
        or inputs.data_ptr() % 16
    ):
        raise ValueError("Invalid fused A2A quantization input or unprepared state")
    storage = inputs.untyped_storage().data_ptr()
    if any(
        buffer is not None and storage == buffer.untyped_storage().data_ptr()
        for buffer in (
            state.output,
            state.scratch,
            state.control,
            state.peers,
            state.chunk_scratch,
            state.chunk_flags,
            state.chunk_control,
            state.chunk_peers,
            state.chunk_flag_peers,
            state.fp8_output,
            state.fp8_scales,
        )
    ):
        raise ValueError("Fused A2A input must not alias state storage")
    rows = inputs.shape[0]
    values = state.fp8_output[: 4 * rows]
    scales = state.fp8_scales[: rows * state.channels // 128].view(-1, 4 * rows)
    if (
        state.chunk_threshold_bytes is not None
        and inputs.numel() * inputs.element_size() >= state.chunk_threshold_bytes
    ):
        state.chunk_module.exchange_chunk_fp8(
            state.chunk_flag_peers,
            inputs,
            values,
            scales,
            state.chunk_peers,
            state.chunk_control,
            state.chunk_capacity,
            rows,
            state.channels,
            state.rank,
            state.blocks,
        )
    else:
        state.module.exchange_fp8(
            inputs,
            values,
            scales,
            state.peers,
            state.control,
            state.words,
            rows,
            state.channels,
            state.rank,
            state.blocks,
        )
    return values, scales
