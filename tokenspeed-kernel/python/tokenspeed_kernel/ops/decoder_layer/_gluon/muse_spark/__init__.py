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

PLAN_REQUESTS = (1, 2)
"""Requests per step this leaf serves (8 and 16 rows at verify width 8)."""


def plan_variants():
    """Every plan key this leaf compiles: requests x layer kind x next kind."""
    from tokenspeed_kernel.ops.decoder_layer import LAYER_KINDS, PlanKey

    return tuple(
        PlanKey(requests, kind, nxt)
        for requests in PLAN_REQUESTS
        for kind in LAYER_KINDS
        for nxt in LAYER_KINDS
    ) + tuple(
        PlanKey(requests, kind, "regular")
        for requests in PLAN_REQUESTS
        for kind in LAYER_KINDS
    )
