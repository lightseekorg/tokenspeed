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

"""Static DSpark draft models.

The backbone, KV injection and block forward are inherited from
``DFlashDraftModel``. A per-position Markov head turns the parallel mask-forward
proposal into a semi-autoregressive one by adding a bigram-style bias to the
base logits before sampling. HyperDSpark first reduces each captured target HC
residual with its trained gate, then uses the same FC and output norm as DSpark.

This module implements the minimal static configuration: the ``vanilla``
Markov head only (no confidence head, no gated/rnn heads, no
confidence-scheduled ragged verify).
"""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any

import torch
from torch import nn

from tokenspeed.runtime.distributed.mapping import Mapping
from tokenspeed.runtime.layers.logits_processor import LogitsProcessor
from tokenspeed.runtime.layers.quantization.base_config import QuantizationConfig
from tokenspeed.runtime.layers.vocab_parallel_embedding import (
    ParallelLMHead,
    VocabParallelEmbedding,
)
from tokenspeed.runtime.models.dflash import (
    DFlashDraftModel,
    _cfg_get,
    _get_text_config,
)
from tokenspeed.runtime.utils import add_prefix

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
    dflash_cfg = getattr(config, "dflash_config", None) or {}

    def pick(key: str, default: Any = None) -> Any:
        if isinstance(dspark_cfg, dict) and key in dspark_cfg:
            return dspark_cfg[key]
        if isinstance(dflash_cfg, dict) and key in dflash_cfg:
            return dflash_cfg[key]
        return getattr(config, key, default)

    markov_rank = int(pick("markov_rank", 0) or 0)
    markov_head_type = str(pick("markov_head_type", "vanilla") or "vanilla").lower()
    return markov_rank, markov_head_type


class DSparkDraftModel(DFlashDraftModel):
    """DFlash draft backbone augmented with a (vanilla) Markov head.

    The decoder and context projection are inherited from ``DFlashDraftModel``.
    The Markov head weights (``markov_head.markov_w1`` / ``markov_head.markov_w2``)
    are plain replicated modules, so the inherited ``load_weights`` loads them
    via the default weight loader. Exported embedding and LM-head tables are
    optional; absent tables are shared with the target during execution wiring.
    """

    def _configure_qwen4_target(
        self, target_model, target_config, *, capture_hc: bool
    ) -> None:
        """Capture completed target layers in the checkpoint's feature space."""
        if self.aux_hidden_stream != "prefix":
            raise ValueError(
                "Qwen4-Exp DSpark captures the completed HC residual; "
                f"aux_hidden_stream={self.aux_hidden_stream!r} is unsupported."
            )
        hidden_size = int(_cfg_get(target_config, "hidden_size"))
        if hidden_size != int(self.config.hidden_size):
            raise ValueError(
                f"Qwen4-Exp target hidden_size={hidden_size} differs from the "
                f"DSpark projector hidden_size={self.config.hidden_size}."
            )
        checkpoint_hidden_size = getattr(self.config, "target_hidden_size", None)
        if (
            checkpoint_hidden_size is not None
            and int(checkpoint_hidden_size) != hidden_size
        ):
            raise ValueError(
                f"DSpark target_hidden_size={checkpoint_hidden_size} differs "
                f"from Qwen4-Exp hidden_size={hidden_size}."
            )
        target_vocab_size = int(_cfg_get(target_config, "vocab_size"))
        if int(self.config.vocab_size) != target_vocab_size:
            raise ValueError(
                f"DSpark vocab_size={self.config.vocab_size} differs from "
                f"Qwen4-Exp vocab_size={target_vocab_size}; the draft and target "
                "require the same full vocabulary."
            )
        draft_vocab_size = getattr(self.config, "draft_vocab_size", None)
        if draft_vocab_size is not None and int(draft_vocab_size) != target_vocab_size:
            raise ValueError(
                f"DSpark draft_vocab_size={draft_vocab_size} requires a reduced "
                "vocabulary mapping, which Qwen4-Exp DSpark does not support."
            )
        num_target_layers = getattr(self.config, "num_target_layers", None)
        if num_target_layers is not None and int(num_target_layers) != int(
            _cfg_get(target_config, "num_hidden_layers")
        ):
            raise ValueError(
                f"DSpark num_target_layers={num_target_layers} differs from "
                f"Qwen4-Exp num_hidden_layers={_cfg_get(target_config, 'num_hidden_layers')}."
            )
        target_model.set_dspark_layers_to_capture(
            list(self.target_layer_ids), capture_hc=capture_hc
        )

    def configure_target(self, target_model, target_config) -> None:
        target_text_config = _get_text_config(target_config)
        if _cfg_get(target_text_config, "model_type") in (
            "qwen4_exp",
            "qwen4_exp_text",
        ):
            self._configure_qwen4_target(
                target_model, target_text_config, capture_hc=False
            )
            return
        super().configure_target(target_model, target_config)

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
        self.prefix = prefix
        self.embed_tokens: VocabParallelEmbedding | None = None
        self.lm_head: ParallelLMHead | None = None
        self.logits_processor: LogitsProcessor | None = None
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

    def _load_optional_vocab_weight(self, name: str, weight: torch.Tensor) -> None:
        """Materialize only vocabulary tables actually supplied by the checkpoint."""
        expected = (int(self.config.vocab_size), int(self.config.hidden_size))
        if tuple(weight.shape) != expected:
            raise ValueError(
                f"DSpark {name}.weight requires full-vocabulary shape {expected}, "
                f"got {tuple(weight.shape)}."
            )
        if name == "embed_tokens":
            if self.embed_tokens is None:
                self.embed_tokens = VocabParallelEmbedding(
                    *expected,
                    params_dtype=self.context_dtype,
                    quant_config=None,
                    prefix=add_prefix("embed_tokens", self.prefix),
                    tp_rank=self.mapping.attn.tp_rank,
                    tp_size=self.mapping.attn.tp_size,
                    tp_group=self.mapping.attn.tp_group,
                ).to(device=self.fc.weight.device)
            self.embed_tokens.weight_loader(self.embed_tokens.weight, weight)
            return
        if self.lm_head is None:
            self.lm_head = ParallelLMHead(
                *expected,
                bias=False,
                params_dtype=self.context_dtype,
                quant_config=None,
                prefix=add_prefix("lm_head", self.prefix),
                tp_rank=self.mapping.lm_head.tp_rank,
                tp_size=self.mapping.lm_head.tp_size,
                tp_group=self.mapping.lm_head.tp_group,
            ).to(device=self.fc.weight.device)
            self.logits_processor = LogitsProcessor(
                self.config,
                skip_all_gather=self.mapping.attn.has_dp,
                do_argmax=False,
                logit_scale=getattr(self.config, "logit_scale", None),
                tp_rank=self.mapping.lm_head.tp_rank,
                tp_size=self.mapping.lm_head.tp_size,
                tp_group=self.mapping.lm_head.tp_group,
                dp_lm_head_tp=self.mapping.attn.has_dp and self.mapping.lm_head.has_tp,
            )
        self.lm_head.weight_loader(self.lm_head.weight, weight)

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]):
        def backbone_weights():
            for name, loaded_weight in weights:
                normalized = name.removeprefix("model.")
                root = normalized.split(".", 1)[0]
                if root in ("d2t", "draft_id_to_target_id"):
                    raise ValueError(
                        f"DSpark checkpoint tensor {name!r} requires a reduced "
                        "vocabulary mapping, which is unsupported."
                    )
                if root == "t2d":
                    continue
                if normalized in ("embed_tokens.weight", "lm_head.weight"):
                    self._load_optional_vocab_weight(root, loaded_weight)
                    continue
                yield name, loaded_weight

        return super().load_weights(backbone_weights())


class InheritedHCGateReducer(nn.Module):
    """Reduce one captured HC residual with the checkpoint's trained gate."""

    def __init__(
        self,
        hidden_size: int,
        hc_count: int,
        hc_lowrank: int,
        eps: float,
        dtype: torch.dtype,
    ) -> None:
        super().__init__()
        self.hidden_size = int(hidden_size)
        self.hc_count = int(hc_count)
        self.hc_lowrank = int(hc_lowrank)
        self.eps = float(eps)
        if self.hc_count <= 1 or self.hc_lowrank <= 0:
            raise ValueError(
                "HyperDSpark requires hc_count > 1 and hc_lowrank > 0, got "
                f"hc_count={self.hc_count}, hc_lowrank={self.hc_lowrank}."
            )
        wide = self.hidden_size * self.hc_count
        # This is a Gemma affine: the checkpoint stores an offset from one.
        # Keep the exported names and the per-branch FP32 normalization order.
        self.hc_norm_weight = nn.Parameter(torch.zeros(wide, dtype=dtype))
        self.input_mix_weight_down = nn.Linear(
            wide, self.hc_lowrank, bias=False, dtype=dtype
        )
        self.input_mix_weight_up = nn.Linear(
            self.hc_lowrank, wide, bias=False, dtype=dtype
        )

    def forward(self, hyper: torch.Tensor) -> torch.Tensor:
        expected = self.hc_count * self.hidden_size
        if hyper.shape[-1] != expected:
            raise ValueError(
                f"HyperDSpark reducer expects {expected} HC features, "
                f"got {hyper.shape[-1]}."
            )
        grouped = hyper.float().unflatten(-1, (self.hc_count, self.hidden_size))
        normalized = grouped * torch.rsqrt(
            grouped.square().mean(dim=-1, keepdim=True) + self.eps
        )
        normalized = (normalized.flatten(-2) * (1.0 + self.hc_norm_weight.float())).to(
            hyper.dtype
        )
        activation = torch.nn.functional.silu(
            self.input_mix_weight_down(normalized) / self.hc_count
        )
        gate = torch.sigmoid(self.input_mix_weight_up(activation))
        mixed = gate.unflatten(-1, (self.hc_count, self.hidden_size)) * (
            normalized.unflatten(-1, (self.hc_count, self.hidden_size))
        )
        return mixed.mean(dim=-2)


class HyperDSparkDraftModel(DSparkDraftModel):
    """DSpark with one trained reducer per raw target HC capture."""

    @property
    def supports_incremental_target_projection(self) -> bool:
        # Splitting the FC along raw HC features bypasses the nonlinear gate.
        return False

    @property
    def context_in_features(self) -> int:
        return self.num_context_features * self.hc_count * int(self.config.hidden_size)

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
        nested = getattr(config, "dflash_config", {}) or {}
        hc_count = nested.get("hc_count", getattr(config, "hc_count", None))
        hc_lowrank = nested.get("hc_lowrank", getattr(config, "hc_lowrank", None))
        if hc_count is None or hc_lowrank is None:
            raise ValueError("HyperDSpark requires explicit hc_count and hc_lowrank.")
        self.hc_count = int(hc_count)
        self.hc_lowrank = int(hc_lowrank)
        if self.num_context_features != len(self.target_layer_ids):
            raise ValueError("HyperDSpark target taps do not match the FC input width.")
        self.hc_reducers = nn.ModuleList(
            [
                InheritedHCGateReducer(
                    hidden_size=int(config.hidden_size),
                    hc_count=self.hc_count,
                    hc_lowrank=self.hc_lowrank,
                    eps=float(getattr(config, "rms_norm_eps", 1e-6)),
                    dtype=self.context_dtype,
                )
                for _ in self.target_layer_ids
            ]
        )

    def configure_target(self, target_model, target_config) -> None:
        target_text_config = _get_text_config(target_config)
        if _cfg_get(target_text_config, "model_type") not in (
            "qwen4_exp",
            "qwen4_exp_text",
        ):
            raise ValueError(
                "HyperDSpark requires a Qwen4-Exp target with HC captures."
            )
        target_hc_count = int(_cfg_get(target_text_config, "hc_count"))
        if target_hc_count != self.hc_count:
            raise ValueError(
                f"HyperDSpark hc_count={self.hc_count} differs from "
                f"Qwen4-Exp hc_count={target_hc_count}."
            )
        self._configure_qwen4_target(target_model, target_text_config, capture_hc=True)

    def project_target_hidden(self, target_hidden: torch.Tensor) -> torch.Tensor:
        if target_hidden.shape[-1] != self.context_in_features:
            raise ValueError(
                f"HyperDSpark expects {self.context_in_features} captured features, "
                f"got {target_hidden.shape[-1]}."
            )
        per_tap = self.hc_count * int(self.config.hidden_size)
        squeeze = target_hidden.ndim == 1
        if squeeze:
            target_hidden = target_hidden.unsqueeze(0)
        reduced = torch.cat(
            [
                reducer(tap)
                for reducer, tap in zip(
                    self.hc_reducers, target_hidden.split(per_tap, dim=-1), strict=True
                )
            ],
            dim=-1,
        )
        projected = super().project_target_hidden(reduced)
        return projected.squeeze(0) if squeeze else projected


EntryClass = [DSparkDraftModel, HyperDSparkDraftModel]
