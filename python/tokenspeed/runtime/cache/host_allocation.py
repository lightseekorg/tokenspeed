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

"""Scoped CUDA host allocation settings for startup Host L2 storage."""

from contextlib import contextmanager


def validate_host_register_threads(threads: int) -> None:
    if (
        type(threads) is not int
        or threads <= 0
        or threads > 128
        or threads & (threads - 1)
    ):
        raise ValueError(
            "Host registration threads must be a power of two from 1 to 128"
        )


@contextmanager
def host_register_settings(threads: int):
    """Override only registration settings, restoring them even on failure.

    PyTorch settings are process-wide; use during synchronous startup allocation,
    before other threads allocate pinned storage.
    """
    import torch

    validate_host_register_threads(threads)
    before = torch.cuda.memory._snapshot()["allocator_settings"]
    original = before["PYTORCH_CUDA_ALLOC_CONF"]
    base = original.rstrip(",")
    override = ",".join(
        filter(
            None,
            [
                base,
                f"pinned_use_cuda_host_register:True,pinned_num_register_threads:{threads}",
            ],
        )
    )
    restore = ",".join(
        filter(
            None,
            [
                base,
                (
                    f"pinned_use_cuda_host_register:{before['pinned_use_cuda_host_register']},"
                    f"pinned_num_register_threads:{before['pinned_num_register_threads']}"
                ),
            ],
        )
    )
    try:
        torch.cuda.memory._set_allocator_settings(override)
        yield
    finally:
        # Omitted pinned options stay sticky, so restore effective values first.
        torch.cuda.memory._set_allocator_settings(restore)
        torch.cuda.memory._set_allocator_settings(original)


def allocate_host_buffer(nbytes: int, *, register_threads: int | None):
    """Allocate pinned uint8 storage, optionally overriding startup registration.

    With an override, the returned view is 4 KiB aligned and retains its PyTorch
    allocation. None preserves the caller's allocator policy and allocation size.
    """
    import torch

    if nbytes <= 0:
        raise ValueError("Host buffer size must be positive")
    if register_threads is None:
        return torch.empty(nbytes, dtype=torch.uint8, pin_memory=True)
    if not torch.cuda.is_available() or torch.version.cuda is None:
        raise ValueError("Parallel Host registration requires CUDA")
    with host_register_settings(register_threads):
        backing = torch.empty(nbytes + 4095, dtype=torch.uint8, pin_memory=True)
    offset = (-backing.data_ptr()) % 4096
    return backing[offset : offset + nbytes]
