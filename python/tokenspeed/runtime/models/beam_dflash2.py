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

"""Beam's DFlash2 drafter checkpoint, served by the DFlash2 draft model."""

from __future__ import annotations

import copy

from tokenspeed.runtime.distributed.mapping import Mapping
from tokenspeed.runtime.layers.quantization.base_config import QuantizationConfig
from tokenspeed.runtime.models.dflash2 import DFlash2DraftModel
from tokenspeed.runtime.utils.env import global_server_args_dict
from tokenspeed.runtime.utils.hf_transformers_utils import get_context_length
from tokenspeed.runtime.utils.server_args import spec_rope_positions
from tokenspeed.runtime.utils.spec_block_geometry import read_checkpoint_block_size


def draft_rope_positions(config, context: int) -> int:
    """How many positions the draft's RoPE table needs for a context
    (``spec_rope_positions``; a DFlash verify width is the checkpoint's block
    size)."""
    return spec_rope_positions(context, read_checkpoint_block_size(config))


class DFlash2BeamDraftModel(DFlash2DraftModel):
    """The DFlash2 draft graph under the architecture name Beam's drafter
    checkpoint declares. Its config, weights and target contract are the
    DFlash2 ones; the drafter routes to ``DFlash2`` by class. Only its RoPE
    table is longer (``draft_rope_positions``), and it refuses a context
    that table does not cover (``configure_target``)."""

    def __init__(
        self,
        config,
        mapping: Mapping,
        quant_config: QuantizationConfig | None = None,
        prefix: str = "",
    ) -> None:
        # The checkpoint's max_position_embeddings is also the longest
        # --max-model-len the draft accepts; --max-model-len exceeds it only
        # under TOKENSPEED_ALLOW_OVERWRITE_LONGER_CONTEXT_LEN.
        context = max(
            int(config.max_position_embeddings),
            int(global_server_args_dict["max_model_len"] or 0),
        )
        # While the model is built, only the RoPE tables read
        # max_position_embeddings. The server builds them on the GPU, where
        # cos and sin run per element, so the rows below the old length stay
        # bit for bit. On the CPU, torch's vectorized cos/sin can round an
        # element one ulp differently when the thread split moves the vector
        # boundaries. The loaded config keeps its value.
        rope_config = copy.copy(config)
        rope_config.max_position_embeddings = draft_rope_positions(config, context)
        super().__init__(
            config=rope_config,
            mapping=mapping,
            quant_config=quant_config,
            prefix=prefix,
        )
        self.config = config
        self.rope_positions = rope_config.max_position_embeddings

    def configure_target(self, target_model, target_config) -> None:
        """Refuse a context the RoPE table does not cover, then wire the
        target as DFlash2 does.

        This is the first point where the draft sees the target's config.
        Without --max-model-len the server takes the target's own context
        (Beam's is 1048576), and no config check compares it with the
        draft's.
        """
        context = global_server_args_dict["max_model_len"] or get_context_length(
            target_config
        )
        if draft_rope_positions(self.config, context) > self.rope_positions:
            limit = int(self.config.max_position_embeddings)
            raise ValueError(
                f"Beam's DFlash2 drafter supports up to {limit} tokens of "
                f"context, but the server would use {context}; pass "
                f"--max-model-len {limit} or less."
            )
        super().configure_target(target_model, target_config)


EntryClass = [DFlash2BeamDraftModel]
