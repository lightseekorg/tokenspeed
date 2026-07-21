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

"""Backend-independent operation contracts."""

from tokenspeed_kernel.contracts.ops import MHA_PREFILL
from tokenspeed_kernel.operation import OperationRegistry, OperationSchema

__all__ = [
    "MHA_PREFILL",
    "get_operation_schema",
    "list_operation_schemas",
]


def get_operation_schema(family: str, mode: str) -> OperationSchema:
    """Return the published contract for one family and mode."""
    return OperationRegistry.get().lookup(family, mode)


def list_operation_schemas() -> tuple[OperationSchema, ...]:
    """Return every contract published by this package."""
    return OperationRegistry.get().schemas()
