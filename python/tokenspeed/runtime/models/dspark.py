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

"""Static DSpark draft model.

The backbone, target-hidden projection, KV injection and block forward are 
all inherited unchanged from ``DFlashDraftModel``; the only addition is a 
per-position Markov head that turns the parallel mask-forward proposal into 
a semi-autoregressive one by adding a bigram-style bias to the base logits before sampling.

This module implements the minimal static configuration: the ``vanilla``
Markov head only (no confidence head, no gated/rnn heads, no
confidence-scheduled ragged verify).
"""

from __future__ import annotations

from typing import Any

import torch
from torch import nn

from tokenspeed.runtime.distributed.mapping import Mapping
from tokenspeed.runtime.layers.quantization.base_config import QuantizationConfig
from tokenspeed.runtime.models.dflash import DFlashDraftModel

SUPPORTED_MARKOV_HEAD_TYPES = ("vanilla",)


class VanillaMarkov(nn.Module):

    markov_head_type = "vanilla"

    def __init__(self, *, vocab_size: int, markov_rank: int) -> None:
        super().__init__()
        self.vocab_size = int(vocab_size)
        self.markov_rank = int(markov_rank)
        if self.markov_rank <= 0:
            raise ValueError(
                f"VanillaMarkov requires markov_rank > 0, got {self.markov_rank}."
            )
        self.markov_w1 = nn.Embedding(self.vocab_size, self.markov_rank)
        self.markov_w2 = nn.Linear(self.markov_rank, self.vocab_size, bias=False)

    def get_prev_latent(self, token_ids: torch.Tensor) -> torch.Tensor:
        """Look up the rank-space latent for the previous token(s)."""
        return self.markov_w1(token_ids.long())

    def project_bias(self, latent_states: torch.Tensor) -> torch.Tensor:
        return self.markov_w2(latent_states)

    def compute_step_bias(self, token_ids: torch.Tensor) -> torch.Tensor:
        """Full-vocab bias for one block position given the previous token."""
        return self.project_bias(self.get_prev_latent(token_ids))


def _get_markov_params(config: Any) -> tuple[int, str]:
    dspark_cfg = getattr(config, "dspark_config", None) or {}

    def pick(key: str, default: Any = None) -> Any:
        if isinstance(dspark_cfg, dict) and key in dspark_cfg:
            return dspark_cfg[key]
        return getattr(config, key, default)

    markov_rank = int(pick("markov_rank", 0) or 0)
    markov_head_type = str(pick("markov_head_type", "vanilla") or "vanilla").lower()
    return markov_rank, markov_head_type


class DSparkDraftModel(DFlashDraftModel):
    """DFlash draft backbone augmented with a (vanilla) Markov head.

    Everything except the Markov head is inherited from ``DFlashDraftModel``.
    The Markov head weights (``markov_head.markov_w1`` / ``markov_head.markov_w2``)
    are plain replicated modules, so the inherited ``load_weights`` loads them
    via the default weight loader without extra routing.
    """

    def __init__(
        self,
        config,
        mapping: Mapping,
        quant_config: QuantizationConfig | None = None,
        prefix: str = "",
    ) -> None:
        super().__init__(
            config=config,
            mapping=mapping,
            quant_config=quant_config,
            prefix=prefix,
        )
        markov_rank, markov_head_type = _get_markov_params(config)
        if markov_rank <= 0:
            raise ValueError(
                "DSpark draft requires markov_rank > 0 (the Markov head is the "
                f"core of the semi-AR draft); got markov_rank={markov_rank}."
            )
        if markov_head_type not in SUPPORTED_MARKOV_HEAD_TYPES:
            raise ValueError(
                f"Unsupported DSpark markov_head_type={markov_head_type!r}; this "
                f"static build only supports {SUPPORTED_MARKOV_HEAD_TYPES}."
            )
        vocab_size = getattr(config, "vocab_size", None)
        if vocab_size is None:
            raise ValueError(
                "DSpark draft config must define vocab_size for the Markov head."
            )
        self.markov_head = VanillaMarkov(
            vocab_size=int(vocab_size), markov_rank=markov_rank
        )


EntryClass = [DSparkDraftModel]
