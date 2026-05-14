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

from enum import IntEnum, auto


class SpeculativeAlgorithm(IntEnum):
    NONE = auto()
    EAGLE3 = auto()
    MTP = auto()
    NGRAM = auto()

    def is_none(self) -> bool:
        return self == SpeculativeAlgorithm.NONE

    def needs_draft_decode_prealloc(self) -> bool:
        # NGRAM has no draft model and therefore no draft KV slots to reserve.
        return self in (SpeculativeAlgorithm.EAGLE3, SpeculativeAlgorithm.MTP)

    @staticmethod
    def from_string(name: str | None) -> "SpeculativeAlgorithm":
        name_map = {
            "EAGLE3": SpeculativeAlgorithm.EAGLE3,
            "MTP": SpeculativeAlgorithm.MTP,
            "NGRAM": SpeculativeAlgorithm.NGRAM,
            None: SpeculativeAlgorithm.NONE,
        }
        if name is not None:
            name = name.upper()
        return name_map[name]
