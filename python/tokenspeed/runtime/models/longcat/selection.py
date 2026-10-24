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

"""Selection shared by the two LongCat attention branches."""

from __future__ import annotations

from dataclasses import dataclass

from tokenspeed.runtime.models.glm5 import GlmDsaDecodeTopK as DSADecodeSelection
from tokenspeed.runtime.models.glm5 import GlmDsaPrefillTopK as DSAPrefillSelection


@dataclass
class DSASelection:
    """Forward-local sparse selection shared by one owner/reuse layer pair."""

    owner_layer_id: int
    prefill: DSAPrefillSelection | None = None
    decode: DSADecodeSelection | None = None

    def require_consumer(self, consumer_layer_id: int) -> None:
        expected_consumer = self.owner_layer_id + 1
        if consumer_layer_id != expected_consumer:
            raise RuntimeError(
                f"DSA selection from owner layer {self.owner_layer_id} requires consumer layer {expected_consumer}, got consumer layer {consumer_layer_id}."
            )
