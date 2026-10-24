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

"""LongCat argmax policy on the shared logits and drafter paths."""

from tokenspeed_kernel.ops.sampling.cute_dsl import DistArgmaxState

from tokenspeed.runtime.layers.logits_processor import LogitsProcessor


class LongCatLogitsProcessor(LogitsProcessor):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # A generic state at the same key carries a different kernel policy.
        self._LOGITS_DIST_ARGMAX_STATES = {}

    def _create_dist_argmax_state(self, **kwargs) -> DistArgmaxState | None:
        from tokenspeed_kernel.ops.longcat.argmax import try_create_argmax_state

        return try_create_argmax_state(**kwargs)
