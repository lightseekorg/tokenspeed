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

"""KV cache kernel entry points."""

import torch


def zero_byte_ranges(
    backing: torch.Tensor, ranges: list[tuple[int, int]], *, device: str | torch.device
) -> None:
    """Zero byte ranges using the allocation's placement and execution device.

    Args:
        backing: Contiguous uint8 allocation; CUDA execution requires pinned
            memory when the allocation is on the host.
        ranges: (byte offset, byte count) pairs relative to the tensor's data_ptr.
        device: Execution device. GPU work follows its current stream; callers
            retain host allocations until that work completes. CPU execution
            clears CPU allocations synchronously.
    """
    if not ranges:
        return
    if backing.dtype != torch.uint8 or not backing.is_contiguous():
        raise ValueError("backing must be a contiguous uint8 tensor")
    if any(
        offset < 0 or size <= 0 or offset + size > backing.numel()
        for offset, size in ranges
    ):
        raise ValueError("ranges must be non-empty and lie within backing")

    execution_device = torch.device(device)
    if execution_device.type == "cpu":
        if backing.device.type != "cpu":
            raise ValueError("CPU zeroing requires host storage")
        for offset, size in ranges:
            backing[offset : offset + size].zero_()
    elif backing.device.type == "cpu":
        from tokenspeed_kernel.ops.kvcache.cute_dsl import (
            cute_dsl_zero_host_byte_ranges,
        )

        cute_dsl_zero_host_byte_ranges(backing, ranges, device=execution_device)
    else:
        from tokenspeed_kernel.ops.kvcache.triton import (
            zero_byte_ranges as zero_device_byte_ranges,
        )

        zero_device_byte_ranges(backing, ranges)
