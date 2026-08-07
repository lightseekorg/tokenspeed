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
"""CPU-only unit tests for MORI all-to-all backend acceptance in the kernel layer.

Covers ``_validate_a2a_backend`` gating (``moe/__init__.py``). The
``a2a_backend='mori' -> solution='mori'`` mapping in ``moe_plan`` requires full
kernel selection (gfx950) and is exercised by the distributed gfx950 test instead.
"""

from __future__ import annotations

import pytest
from tokenspeed_kernel.ops.moe import _validate_a2a_backend


@pytest.mark.parametrize("backend", [None, "none", "deepep", "mori"])
def test_validate_accepts_supported_backends(backend) -> None:
    # Must not raise — "mori" is an accepted backend.
    _validate_a2a_backend(backend)


@pytest.mark.parametrize("backend", ["nccl", "foo", "MORI", "flashinfer", ""])
def test_validate_rejects_unsupported_backends(backend) -> None:
    with pytest.raises(NotImplementedError):
        _validate_a2a_backend(backend)
