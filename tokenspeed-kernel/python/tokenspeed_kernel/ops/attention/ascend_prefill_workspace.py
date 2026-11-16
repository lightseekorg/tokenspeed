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

"""Static storage geometry for the Ascend prefill transport.

One allocation serves successive layers. History capacity is also the scheduler's
batch KV bound, including page rounding and the current chunk's cache pages.
"""

from dataclasses import dataclass


@dataclass(frozen=True)
class AscendPrefillWorkspaceLayout:
    history_capacity: int
    current_capacity: int
    degree: int
    null_rows: int
    has_index: bool

    def __post_init__(self):
        if (
            min(
                self.history_capacity,
                self.current_capacity,
                self.degree,
                self.null_rows,
            )
            <= 0
        ):
            raise ValueError("Prefill workspace capacities must be positive")

    @property
    def width(self) -> int:
        return 704 if self.has_index else 576

    @property
    def current_padded_rows(self) -> int:
        alignment = self.degree * 128
        return (self.current_capacity + alignment - 1) // alignment * alignment

    @property
    def plane_rows(self) -> int:
        return self.null_rows + self.history_capacity + self.current_padded_rows

    @property
    def send_rows(self) -> int:
        if self.degree == 1:
            return self.current_capacity
        # Page placement need not balance the owners in one prefill batch.
        return max(
            self.history_capacity,
            (self.current_capacity + self.degree - 1) // self.degree,
        )

    @property
    def receive_rows(self) -> int:
        # History collectives receive directly into their final planes.
        return self.current_capacity if self.degree > 1 else 0

    @property
    def nbytes(self) -> int:
        return (self.plane_rows + self.send_rows + self.receive_rows) * self.width * 2
