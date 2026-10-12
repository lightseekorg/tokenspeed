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

"""Decision-style typed output (Jev-like) on top of the Score API.

See docs/design/scoring.md for the contract and ownership split.
"""

from tokenspeed.runtime.decision.adapter import (
    DecisionAdapter,
    GenericDecisionAdapter,
)
from tokenspeed.runtime.decision.registry import (
    get_decision_adapter,
    register_decision_adapter,
)
from tokenspeed.runtime.decision.types import (
    ALL_STYLES,
    STYLE_FUSED_CHOICE,
    STYLE_POINTWISE_YESNO,
    DecisionRequest,
    DecisionResult,
    ScoreCall,
)

__all__ = [
    "ALL_STYLES",
    "STYLE_FUSED_CHOICE",
    "STYLE_POINTWISE_YESNO",
    "DecisionAdapter",
    "DecisionRequest",
    "DecisionResult",
    "GenericDecisionAdapter",
    "ScoreCall",
    "get_decision_adapter",
    "register_decision_adapter",
]
