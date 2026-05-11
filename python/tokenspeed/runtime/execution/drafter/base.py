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

from abc import abstractmethod
from typing import TYPE_CHECKING

import torch

from tokenspeed.runtime.execution.model_runner import ModelRunner

if TYPE_CHECKING:
    from tokenspeed.runtime.execution.context import ForwardContext
    from tokenspeed.runtime.execution.input_buffer import InputBuffers
    from tokenspeed.runtime.execution.runtime_states import RuntimeStates
    from tokenspeed.runtime.layers.attention.backends.base import AttentionBackend
    from tokenspeed.runtime.layers.attention.kv_cache.base import BaseTokenToKVPool
    from tokenspeed.runtime.layers.logits_processor import LogitsProcessorOutput


class BaseDrafter:
    def __init__(
        self,
        spec_num_tokens: int,
        spec_num_steps: int | None = None,
        draft_model_runner: ModelRunner | None = None,
        runtime_states: RuntimeStates | None = None,
        input_buffers: InputBuffers | None = None,
        page_size: int | None = None,
        req_to_page: torch.Tensor | None = None,
        attn_backend: AttentionBackend | None = None,
        token_to_kv_pool: BaseTokenToKVPool | None = None,
        vocab_size: int | None = None,
    ):
        self.spec_num_tokens = spec_num_tokens
        self.spec_num_steps = spec_num_steps
        self.draft_model_runner = draft_model_runner
        self.runtime_states = runtime_states
        self.input_buffers = input_buffers
        self.page_size = page_size
        self.req_to_page = req_to_page
        self.attn_backend = attn_backend
        self.token_to_kv_pool = token_to_kv_pool
        self.vocab_size = vocab_size

    @abstractmethod
    def get_candidates(
        self,
        base_ctx: ForwardContext,
    ) -> torch.Tensor | None:
        raise NotImplementedError

    @abstractmethod
    def run(
        self,
        base_ctx: ForwardContext,
        logits_output: LogitsProcessorOutput,
        output_tokens: torch.Tensor,
        accept_lengths: torch.Tensor,
    ) -> torch.Tensor:
        raise NotImplementedError

    @abstractmethod
    def draft(self, *args, **kwargs) -> torch.Tensor | None:
        raise NotImplementedError
