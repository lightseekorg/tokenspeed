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

"""NVIDIA 2D device-to-device copy."""

from __future__ import annotations

from tokenspeed_kernel.platform import current_platform


def memcpy_2d_async(
    *,
    dst: int,
    dst_pitch: int,
    src: int,
    src_pitch: int,
    width: int,
    height: int,
    stream_ptr: int,
) -> None:
    """Copy a 2D device rectangle asynchronously.

    Args:
        dst: Destination device pointer.
        dst_pitch: Destination row pitch in bytes.
        src: Source device pointer.
        src_pitch: Source row pitch in bytes.
        width: Bytes copied per row.
        height: Number of rows.
        stream_ptr: CUDA stream handle.

    Returns:
        None.

    Raises:
        RuntimeError: If the platform is not NVIDIA or the runtime copy fails.
    """
    if not current_platform().is_nvidia:
        raise RuntimeError("memcpy_2d_async is NVIDIA-only")
    from tokenspeed_kernel.thirdparty.cuda.memcpy_2d import cuda_memcpy_2d_async

    cuda_memcpy_2d_async(
        dst=dst,
        dst_pitch=dst_pitch,
        src=src,
        src_pitch=src_pitch,
        width=width,
        height=height,
        stream_ptr=stream_ptr,
    )
