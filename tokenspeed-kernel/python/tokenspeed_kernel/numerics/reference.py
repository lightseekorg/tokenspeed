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

"""Select the ground truth a test compares an operator's kernel against."""

from __future__ import annotations

from typing import Any

import tokenspeed_kernel.numerics.torch  # noqa: F401  (registers the references)
from tokenspeed_kernel.registry import KernelRegistry
from tokenspeed_kernel.selection import (
    NoKernelFoundError,
    SelectedKernel,
    select_kernel,
    select_reference,
)
from tokenspeed_kernel.signature import FormatSignature


def reference_kernel(
    family: str,
    mode: str,
    format_signature: FormatSignature,
    *,
    traits: dict[str, Any] | None,
    kernel: str | None,
) -> SelectedKernel:
    """The ``"reference"`` solution to check a kernel of ``family.mode`` against.

    Args:
        family: Operator family.
        mode: Operator mode.
        format_signature: Format signature of the call.
        traits: Traits of the call, as the operator's dispatcher passes them,
            or None.
        kernel: Name of the kernel under test, or None for the kernel
            :func:`~tokenspeed_kernel.selection.select_kernel` dispatches for
            this call.

    Returns:
        The reference :func:`~tokenspeed_kernel.selection.select_reference`
        picks for the kernel under test; it takes that kernel's arguments.

    Raises:
        NoKernelFoundError: No kernel other than the one under test matches.
    """
    if kernel is None:
        kernel = select_kernel(family, mode, format_signature, traits=traits).name
    spec = KernelRegistry.get().get_by_name(kernel)
    if spec is None:
        raise NoKernelFoundError(f"Kernel {kernel!r} is not registered")
    reference = select_reference(spec, format_signature, traits=traits)
    if reference is None:
        raise NoKernelFoundError(
            f"No reference for {kernel!r} ({family}.{mode}, {format_signature})"
        )
    return reference
