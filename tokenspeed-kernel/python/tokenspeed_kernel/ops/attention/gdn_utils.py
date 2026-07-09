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

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

import torch


class GdnCheckpointLayout(str, Enum):
    """Backend-native checkpoint layout returned by GDN chunk prefill."""

    NONE = "none"
    FLA = "fla"
    FLASHINFER = "flashinfer"


@dataclass(frozen=True)
class GdnChunkPrefillResult:
    """Structured result for GDN chunk prefill.

    Args:
        out: GDN output tensor.
        final_state: Final recurrent state, when requested.
        h: Optional backend-native intermediate recurrent checkpoints.
        h_cu_starts: Optional cumulative checkpoint starts for FlashInfer layout.
        h_layout: Layout of ``h``.
    """

    out: torch.Tensor
    final_state: torch.Tensor | None
    h: torch.Tensor | None = None
    h_cu_starts: torch.Tensor | None = None
    h_layout: GdnCheckpointLayout = GdnCheckpointLayout.NONE
