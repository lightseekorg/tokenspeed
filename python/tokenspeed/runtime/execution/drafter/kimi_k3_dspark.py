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

"""K3 model wiring around the ordinary DSpark proposal path."""

from __future__ import annotations

import torch

from tokenspeed.runtime.execution.context import ForwardContext
from tokenspeed.runtime.execution.drafter.dspark import DSpark
from tokenspeed.runtime.layers.logits_processor import LogitsProcessorOutput


class K3DSpark(DSpark):
    """K3 owns capture configuration; candidate generation stays on DSpark."""

    @staticmethod
    def configure_target(target_model, draft_model) -> None:
        """Configure K3 taps once after loading, including non-drafting PP stages."""
        capture_layer_ids = list(draft_model.target_capture_layer_ids)
        stream = draft_model.config.aux_hidden_stream
        if draft_model.mapping.has_pp:
            target_model.set_prefill_context_capture(
                capture_layer_ids, stream, draft_model.hidden_size
            )
        else:
            target_model.set_dflash_layers_to_capture(capture_layer_ids)
            target_model.set_dflash_aux_hidden_stream(stream)

    def wire_target(self, target_model) -> None:
        """Bind execution resources without replacing the configured K3 taps."""
        language_model = getattr(target_model, "language_model", target_model)
        self.target_model = target_model
        self.target_language_model = language_model
        self.embed_tokens = (
            self.model.embed_tokens
            if self.model.mapping.has_pp
            else target_model.get_input_embeddings()
        )
        self.lm_head = target_model.lm_head
        self.logits_processor = language_model.logits_processor

    def _update_native_cache_from_target(
        self,
        base_ctx: ForwardContext,
        logits_output: LogitsProcessorOutput,
        accept_lengths: torch.Tensor,
    ) -> None:
        if not self.model.mapping.has_pp:
            super()._update_native_cache_from_target(
                base_ctx, logits_output, accept_lengths
            )
            return
        # PP capture already wrote this chunk's context on our stream. Seed
        # the ordinary draft block at the complete prefix, including cached
        # and chunked history, without projecting those taps a second time.
        self.draft_seq_lens_buf[: base_ctx.bs].copy_(
            self.input_buffers.seq_lens_buf[: base_ctx.bs]
        )
