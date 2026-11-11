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

"""The shared model embedding boundary for eager and graphed prefill."""

from __future__ import annotations

from typing import TYPE_CHECKING, Protocol, runtime_checkable

if TYPE_CHECKING:
    import torch

    from tokenspeed.runtime.execution.context import ForwardContext
    from tokenspeed.runtime.execution.request_token_history import (
        RequestTokenHistoryView,
    )


@runtime_checkable
class InputEmbeddingModel(Protocol):
    """An inner model whose input embedding consumes all request-history inputs.

    Ordinary forward and graph replay share this method. It returns one
    embedding row per input token and performs any history appends. Graph
    replay calls it eagerly, before padding. With ``input_embeds`` supplied,
    the inner stack and the outer model's ``prepare_model_kwargs`` must
    neither require nor append history.
    """

    def compute_input_embeddings(
        self,
        input_ids: torch.Tensor,
        ctx: ForwardContext,
        *,
        request_token_history: RequestTokenHistoryView | None,
    ) -> torch.Tensor:
        """Embed the live input rows using the runtime-owned history view."""
        ...
