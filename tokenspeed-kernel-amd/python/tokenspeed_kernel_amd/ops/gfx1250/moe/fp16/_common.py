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

"""Shared checks for the gfx1250 latent-MoE input projections."""

from __future__ import annotations


def _is_packed_projection_view(packed, router, routed, shared) -> bool:
    """Whether ``packed`` is exactly the three weights as consecutive rows.

    The kernels read only ``packed``, so weights that do not live inside it
    would be silently ignored. Checked here rather than through
    ``tokenspeed_kernel``: this package must not depend on it.
    """
    parts = (router, routed, shared)
    storage = packed.untyped_storage()
    if any(part.untyped_storage().data_ptr() != storage.data_ptr() for part in parts):
        return False
    address = packed.data_ptr()
    row_bytes = packed.shape[1] * packed.element_size()
    for part in parts:
        if part.data_ptr() != address:
            return False
        address += part.shape[0] * row_bytes
    return address == packed.data_ptr() + packed.shape[0] * row_bytes
