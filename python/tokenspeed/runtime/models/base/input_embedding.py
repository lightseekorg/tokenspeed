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

"""Token lookup at a decoder's supplied-embedding boundary."""

from dataclasses import dataclass

import torch
from torch import nn


@dataclass(frozen=True)
class DecoderInputEmbedding:
    """Lookup plus an optional dtype-preserving transform before media scatter.

    Args:
        lookup: Token embedding with weight, vocabulary size, and output width.
        transform: The model's lookup-only transform, or None for raw lookup.

    Calling this object returns rows ready for the model's ``input_embeds``
    branch. Normalization shared by text and media stays inside that branch.
    The adapter does not register or copy parameters. Metadata follows the
    lookup's current weight, including after a weight update.
    """

    lookup: nn.Module
    transform: nn.Module | None

    @property
    def weight(self) -> nn.Parameter:
        return self.lookup.weight

    @property
    def num_embeddings(self) -> int:
        return self.lookup.num_embeddings

    @property
    def embedding_dim(self) -> int:
        return self.lookup.embedding_dim

    def __call__(self, input_ids: torch.Tensor) -> torch.Tensor:
        embeds = self.lookup(input_ids)
        return self.transform(embeds) if self.transform is not None else embeds
