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

from functools import cache
from pathlib import Path

_MODULE_PATH = Path(__file__).parent / "objs" / "ple_host_gather" / "ple_host_gather.so"


@cache
def _load_module():
    import tvm_ffi

    return tvm_ffi.load_module(str(_MODULE_PATH))


def ple_staged_gather():
    return _load_module().ple_gather_staged


def validate_ple_hot_cache_geometry(
    heads: int, head_dim: int, vocab_start: int, vocab_end: int
) -> None:
    """Validate the PLE hot cache geometry before the first gather.

    Cache construction uses the same checks as the ``ple_gather_staged`` launcher.

    Args:
        heads: Number of n-gram heads. Must be 16.
        head_dim: Number of elements per head. Must be 160 or 512.
        vocab_start: Inclusive first ID of the local shard. Must be nonnegative.
        vocab_end: Exclusive end of the local shard. The span must be positive
            and fit in a signed int32.

    Returns:
        None if the geometry is supported.

    Raises:
        RuntimeError: If the geometry is unsupported.
    """
    _load_module().ple_validate_hot_cache_geometry(
        heads, head_dim, vocab_start, vocab_end
    )
