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

"""The ``--ep-dispatch-algorithm`` vocabulary, defined once.

An algorithm says how routing picks among a logical expert's physical replicas
(``moe/expert_location.py``). The ``static`` ones map a logical expert to one
fixed replica (per rank under all-to-all EP, per token row on replicated-input
EP); the ``dynamic`` ones draw a replica at random per route. The
``*_with_zero_expert`` variants leave zero-expert routes (ids outside the
routed experts) unmapped.
"""

from __future__ import annotations

EP_DISPATCH_ALGORITHMS: tuple[str, ...] = (
    "static",
    "dynamic",
    "fake",
    "static_with_zero_expert",
    "dynamic_with_zero_expert",
)

STATIC_EP_DISPATCH_ALGORITHMS: frozenset[str] = frozenset(
    {"static", "static_with_zero_expert"}
)

_ZERO_EXPERT_SUFFIX = "_with_zero_expert"


def has_zero_expert(algorithm: str) -> bool:
    """Whether ``algorithm`` keeps zero-expert routes out of the replica map."""
    if algorithm not in EP_DISPATCH_ALGORITHMS:
        raise ValueError(f"unknown --ep-dispatch-algorithm {algorithm!r}")
    return algorithm.endswith(_ZERO_EXPERT_SUFFIX)
