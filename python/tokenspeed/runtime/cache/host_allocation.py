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

"""Pinned Host buffers honoring the user's PyTorch allocator policy."""

HOST_BUFFER_ALIGNMENT_BYTES = 4096


def host_register_enabled() -> bool:
    """Whether the native CUDA host allocator currently uses registration."""
    import torch

    if torch.version.cuda is None or torch.cuda.get_allocator_backend() != "native":
        return False
    return torch.cuda.memory._snapshot()["allocator_settings"][
        "pinned_use_cuda_host_register"
    ]


def allocate_host_buffer(nbytes: int, *, align: bool):
    """Allocate pinned uint8 storage, optionally returning a 4 KiB-aligned view.

    Allocation, registration and freeing always follow PyTorch's current policy.
    The aligned view retains its backing allocation.
    """
    import torch

    if nbytes <= 0:
        raise ValueError("Host buffer size must be positive")
    padding = HOST_BUFFER_ALIGNMENT_BYTES - 1 if align else 0
    backing = torch.empty(nbytes + padding, dtype=torch.uint8, pin_memory=True)
    if not align:
        return backing
    offset = (-backing.data_ptr()) % HOST_BUFFER_ALIGNMENT_BYTES
    return backing[offset : offset + nbytes]
