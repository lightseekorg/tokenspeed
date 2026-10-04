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

"""The draft trees a sampling backend verifies (--speculative-eagle-topk > 1)."""

from __future__ import annotations

from dataclasses import dataclass

import torch

__all__ = ["TreeVerifyBatch", "accepted_path_rows"]


@dataclass(frozen=True)
class TreeVerifyBatch:
    """This step's draft trees, one per request of the verify window.

    Attributes:
        parents: ``[bs, N]`` int32 parent node, ``-1`` for the root; a parent
            precedes its children.
        depths: ``[bs, N]`` int32 node depth, 0 for the root.
    """

    parents: torch.Tensor
    depths: torch.Tensor


def accepted_path_rows(path: torch.Tensor) -> torch.Tensor:
    """``[bs * N]`` window row of each packed prediction: the accepted path
    first, then identity.

    Args:
        path: ``[bs, N]`` accepted path, root first, ``-1`` past it.
    """
    bs, n = path.shape
    path = path.long()
    nodes = torch.arange(n, device=path.device)
    local = torch.where(path >= 0, path, nodes)
    return (local + torch.arange(bs, device=path.device)[:, None] * n).view(-1)
