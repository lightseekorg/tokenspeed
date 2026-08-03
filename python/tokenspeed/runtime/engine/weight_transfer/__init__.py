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

"""RL online weight-transfer control plane.

Implements the HTTP weight-transfer API that RL trainers (verl / slime / AReaL /
miles) drive to update model weights in place during online serving.

Modules:
    config:   ``WeightTransferConfig``.
    manager:  ``WeightTransferManager`` lifecycle state machine that the HTTP
              handlers call into; delegates weight movement to ``AsyncLLM``.
"""

from tokenspeed.runtime.engine.weight_transfer.config import (
    SUPPORTED_BACKENDS,
    WeightTransferConfig,
)

__all__ = ["WeightTransferConfig", "SUPPORTED_BACKENDS"]
