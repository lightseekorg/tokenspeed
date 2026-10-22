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

"""Storage facts distinguish validity, object failure and native retirement."""

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class StorageIOResult:
    success: tuple[bool, ...]
    retired: bool
    # Confirmed absent or invalid objects, never policy/queue/transport errors.
    object_failures: tuple[bool, ...]


@dataclass(frozen=True, slots=True)
class StorageCompletion:
    op_id: int
    success: tuple[bool, ...]
    retired: bool
    object_failures: tuple[bool, ...]


@dataclass(frozen=True, slots=True)
class StoragePoll:
    """Local completion facts and bounded pressure observations for one round."""

    completions: tuple[StorageCompletion, ...]
    queued_operations: int
    queued_bytes: int
    oldest_operation_us: int
