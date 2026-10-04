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

from collections.abc import Sequence
from dataclasses import dataclass


@dataclass(frozen=True)
class ForwardOutputLayout:
    """Output rows for every forward, including ordinary and compact batches.

    Prefill outputs form a request prefix; decode keeps its fixed verify
    width. Ordinary batches include every extend request in that prefix.
    Backends that omit incomplete prefills shorten it without moving
    request/cache rows. This host-only value is frozen before grammar work
    is queued; consumers always use it, never infer a layout from absence.
    """

    num_extends: int
    num_prefill_outputs: int
    num_decodes: int
    decode_width: int

    def __post_init__(self) -> None:
        if not 0 <= self.num_prefill_outputs <= self.num_extends:
            raise ValueError("prefill outputs must be a prefix of extend requests")
        if self.num_decodes < 0 or self.decode_width < 1:
            raise ValueError("invalid decode output geometry")

    @classmethod
    def from_prefill(
        cls,
        *,
        prefix_lengths: Sequence[int],
        input_lengths: Sequence[int],
        prompt_lengths: Sequence[int],
        num_decodes: int,
        decode_width: int,
    ) -> "ForwardOutputLayout":
        complete = []
        for prefix, count, target in zip(
            prefix_lengths, input_lengths, prompt_lengths, strict=True
        ):
            if prefix < 0 or count < 0 or prefix + count > target:
                raise ValueError("invalid prefill input range")
            complete.append(prefix + count == target)
        outputs = sum(complete)
        if complete != [True] * outputs + [False] * (len(complete) - outputs):
            raise ValueError("completed prefills must form a prefix")
        return cls(len(complete), outputs, num_decodes, decode_width)

    @property
    def num_output_tokens(self) -> int:
        return self.num_prefill_outputs + self.num_decodes * self.decode_width

    @property
    def prefill_slice(self) -> slice:
        """Completing prefills share the same request and output prefix."""
        return slice(0, self.num_prefill_outputs)

    @property
    def decode_request_slice(self) -> slice:
        """Decode rows in request-indexed parameters and state."""
        return slice(self.num_extends, self.num_extends + self.num_decodes)

    @property
    def decode_output_slice(self) -> slice:
        """Decode rows in compact logits and fixed-width token storage."""
        return slice(self.num_prefill_outputs, self.num_output_tokens)

    def output_width(self, request: int) -> int:
        """Stored output rows for a request, independent of accepted length."""
        if not 0 <= request < self.num_extends + self.num_decodes:
            raise IndexError("output request is outside the batch")
        if request < self.num_extends:
            return int(request < self.num_prefill_outputs)
        return self.decode_width

    def token_offset(self, request: int) -> int:
        if not 0 <= request < self.num_extends + self.num_decodes:
            raise IndexError("output request is outside the batch")
        if request < self.num_extends:
            return min(request, self.num_prefill_outputs)
        return (
            self.num_prefill_outputs + (request - self.num_extends) * self.decode_width
        )
