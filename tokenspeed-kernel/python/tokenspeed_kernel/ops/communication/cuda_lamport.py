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

"""Experimental bit-preserving, intra-node Lamport packet A2A.

CUDA C++ is used for this protocol prototype to make the system-scoped
64-bit packet transactions explicit. No model backend is changed.

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


class CudaLamportA2AState:
    """Persistent TP4 scratch, initialized collectively before graph capture.

    The supplied process group must contain exactly four GPUs, irrespective
    of the total number of GPUs in the job.

    All ranks must call in the same order, with equal physical shapes, on one
    serialized stream. Outputs are borrowed until the next call. Empty logical
    owners must participate using zero-padded physical rows. Three generations
    protect readers from faster peers; each 64-bit packet embeds a 32-bit epoch
    and 32 payload bits, preserving signed zero and NaN payloads exactly.
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
            [Path(__file__).with_name("_cuda") / "lamport_a2a.cu"],
        )
        torch.cuda.synchronize(device)
        dist.barrier(group=group)

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
            [Path(__file__).with_name("_cuda") / "chunk_a2a.cu"],
        )
        torch.cuda.synchronize(device)
        dist.barrier(group=self.group)
        self.chunk_threshold_bytes = threshold_bytes


@register_kernel(
    family="communication",
    mode="all_to_all",
    solution="cuda",
    signatures=format_signatures(("inputs",), "dense", {torch.bfloat16}),
)
def cuda_lamport_a2a(state, inputs, inverse):
    """Exchange BF16 channel shards while preserving every input bit.

    Forward: [M,K] -> [4*M,K/4]. Inverse: [4*M,K/4] -> [M,K].
    state owns persistent scratch/output; inverse explicitly chooses layout.
    Every peer must use the same shape/direction and serialize consumers before
    its next call. Returned output aliases state.output, not remote scratch.
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
    ):
        raise ValueError(
            "Input shape, dtype, device, or contiguity violates the A2A contract"
        )
    result_shape = (
        (rows, state.channels) if inverse else (4 * rows, state.channels // 4)
    )
    output = state.output[: inputs.numel()].view(result_shape)
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
