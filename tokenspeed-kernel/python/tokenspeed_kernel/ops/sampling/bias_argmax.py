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

"""Allocation-free, exact bias plus argmax for the SM120 proposal path."""

from __future__ import annotations

from dataclasses import dataclass, field

import torch
from tokenspeed_kernel._triton import triton
from tokenspeed_kernel.platform import current_platform
from tokenspeed_kernel.selection import NoKernelFoundError, select_kernel
from tokenspeed_kernel.signature import dense_tensor_format, format_signature
from tokenspeed_kernel.thirdparty.cuda.sampling_metadata import validate_metadata

BIAS_ARGMAX_TILE = 4096
BIAS_ARGMAX_MAX_VOCAB = 262144
_FLOAT_DTYPES = (torch.float16, torch.bfloat16, torch.float32)
_ELEMENT_BYTES = {
    torch.float16: 2,
    torch.bfloat16: 2,
    torch.float32: 4,
    torch.int32: 4,
    torch.int64: 8,
}
_INDEX_MAX = {torch.int32: 2**31 - 1, torch.int64: 2**63 - 1}
_SCALAR_DTYPES = {
    3: torch.int32,
    4: torch.int64,
    5: torch.float16,
    6: torch.float32,
    15: torch.bfloat16,
}
_SIGNATURES = {
    dtype: format_signature(logits=dense_tensor_format(dtype))
    for dtype in _FLOAT_DTYPES
}
_SCALAR_SIGNATURES = {
    code: _SIGNATURES[dtype]
    for code, dtype in _SCALAR_DTYPES.items()
    if dtype in _SIGNATURES
}


@dataclass(frozen=True)
class BiasArgmaxWorkspace:
    """Caller-owned partials, allocated before any graph captures.

    The same workspace may be reused by sequential calls. Concurrent calls
    on different streams require separate workspaces or explicit ordering.
    """

    values: torch.Tensor
    indices: torch.Tensor
    _device: torch.device | None = field(init=False, repr=False)
    _capability: tuple[int, int] | None = field(init=False, repr=False)

    def __post_init__(self) -> None:
        # A CUDA device's architecture is immutable for this process. Bind its
        # identity when constructing the workspace, including direct callers
        # and dataclasses.replace; no tensor descriptors or pointers are cached.
        device = self.values.device if isinstance(self.values, torch.Tensor) else None
        capability = (
            torch.cuda.get_device_capability(device)
            if device is not None
            and device.type == "cuda"
            and current_platform().is_nvidia
            else None
        )
        object.__setattr__(self, "_device", device)
        object.__setattr__(self, "_capability", capability)


def create_bias_argmax_workspace(
    max_rows: int, vocab_size: int, device: torch.device | str | None
) -> BiasArgmaxWorkspace | None:
    """Allocate stable partial buffers for a bounded SM120 bias argmax.

    Args:
        max_rows: Maximum number of proposal rows across capture buckets.
        vocab_size: Maximum vocabulary width, in the range 1 through 262144.
        device: Device owning the logits; None disables this candidate.

    Returns:
        Contiguous float32 values and int32 indices, or None for unsupported
        hardware or sizes. The bound is this candidate's tested scope.
    """
    if device is None or max_rows <= 0 or not 0 < vocab_size <= BIAS_ARGMAX_MAX_VOCAB:
        return None
    device = torch.device(device)
    if device.type != "cuda" or not current_platform().is_nvidia:
        return None
    if torch.cuda.get_device_capability(device) != (12, 0):
        return None
    tiles = (vocab_size + BIAS_ARGMAX_TILE - 1) // BIAS_ARGMAX_TILE
    return BiasArgmaxWorkspace(
        values=torch.empty((max_rows, tiles), dtype=torch.float32, device=device),
        indices=torch.empty((max_rows, tiles), dtype=torch.int32, device=device),
    )


def _byte_span(tensor: torch.Tensor) -> tuple[int, int]:
    """Conservative byte interval for a nonempty positive-strided view."""
    start = tensor.data_ptr()
    extent = 1 + sum(
        (size - 1) * stride for size, stride in zip(tensor.shape, tensor.stride())
    )
    return start, start + extent * tensor.element_size()


def try_bias_argmax(
    logits: torch.Tensor,
    bias: torch.Tensor,
    out: torch.Tensor,
    workspace: BiasArgmaxWorkspace | None,
    global_offset: int,
) -> bool:
    """Write exact native-rounded bias argmax IDs without allocating tensors.

    Args:
        logits: Float16, bfloat16 or float32 matrix [M, V].
        bias: Same-shaped matrix in any of those floating dtypes.
        out: Int32 or int64 vector [M], including positive-strided column views.
        workspace: Persistent partial buffers from the factory, sized for M,V.
        global_offset: Nonnegative original-vocabulary start. Every possible
            resulting ID must be representable by out.dtype.

    Returns:
        True when the registered SM120 kernel ran; False without any writes
        for unsupported metadata. Scores are round_d(logits + round_d(bias)),
        where d is logits.dtype. NaNs win at their first index, and equal
        values (including signed zeros and infinities) choose the first index.
        Inputs may alias each other; output and scratch must not overlap any
        other read/write region. Overlapping byte spans are conservatively
        rejected, even if an interleaved view's live elements do not overlap.
    """
    # An active cache hook may suppress a cold JIT compilation/launch. Reject
    # before either stage: finalizing stale partials would change token IDs.
    if (
        triton.knobs.runtime.interpret
        or triton.knobs.runtime.jit_cache_hook is not None
    ):
        return False
    if not isinstance(workspace, BiasArgmaxWorkspace):
        return False
    if (
        workspace._device is None
        or workspace._device.type != "cuda"
        or workspace._capability != (12, 0)
    ):
        return False
    metadata = validate_metadata(
        logits,
        bias,
        out,
        workspace.values,
        workspace.indices,
        torch.cuda.current_device(),
        workspace._device.index,
        global_offset,
    )
    if metadata is None:
        return False
    # Native scalar tags already distinguish every launcher cache class.
    # Pass the checked current tuple directly without rebuilding dtype fields.
    signature = _SCALAR_SIGNATURES[metadata[3]]
    try:
        kernel = select_kernel(
            "sampling", "bias_argmax", signature, solution="triton", override=None
        )
    except NoKernelFoundError:
        return False
    prepared = getattr(getattr(kernel, "impl", None), "_from_checked_metadata", None)
    if prepared is None:
        # Selection overrides keep their ordinary public five-argument call.
        kernel(logits, bias, out, workspace, global_offset)
    else:
        prepared(logits, bias, out, workspace, global_offset, metadata)
    return True
