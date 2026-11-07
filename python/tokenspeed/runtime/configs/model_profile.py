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

"""The facts a model declares about itself before the runtime builds it."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from types import MappingProxyType
from typing import TYPE_CHECKING

from tokenspeed.runtime.configs.numerics import NUMERICS_ENVELOPES

if TYPE_CHECKING:
    from tokenspeed.runtime.configs.model_config import ModelConfig
    from tokenspeed.runtime.layers.attention.configs.base import AttnConfig
    from tokenspeed.runtime.utils.server_args import ServerArgs


@dataclass(frozen=True, kw_only=True)
class ModelProfile:
    """Model-owned answers to the questions architecture tables used to answer.

    A registered model class exposes ``model_profile(hf_config)`` as a
    classmethod returning one of these. ``ModelConfig`` resolves it once, and
    every resolver that would otherwise match the architecture name against a
    table reads the profile instead. Required facts are explicit. Optional
    history lookback defaults to the complete history, preserving readers
    that declare no bound.

    Attributes:
        configure_attention: Writes the attention geometry onto the
            ``ModelConfig`` (``attention_arch``, head dimensions, scaling,
            a DSA model's ``index_k_format``), e.g. ``configure_mla_attention``
            or ``configure_dsa_attention``. Receives the resolved launch too,
            so a hook can key a choice on it (the index-K plane a model scores
            under ``server_args.numerics``, say); the in-tree hooks read none
            of it.
        cache_family: Registered cache recipe and pool family that owns this
            model's per-request state.
        linear_attention: Registered linear-attention backend serving the
            model's linear layers, or None for a model without them.
        default_attention_backend: Attention backend used when the launch
            names none, or None to keep the architecture default.
        request_token_history: Whether the model reads each request's
            committed token history (``ForwardContext.request_token_history``).
        request_token_history_lookback: Committed tokens behind the write
            frontier the model reads (an n-gram hash reads its order minus
            one), so a resuming extend seeds only that tail of its prefix.
            Omitting it (None) preserves the complete history. Zero means the
            model reads no committed tokens behind the frontier.
        is_multimodal: Whether this checkpoint has an active-capable media encoder.
        tokenizer_kwargs: Extra keyword arguments for the model's tokenizer.
        attention_instances_per_layer: Attention modules per decoder layer:
            1 for the ordinary stack, 2 for paired layouts (two attention
            branches sharing one layer's MLP block, e.g. LongCat's ScMoE) so
            the cache plans one plane per branch. Cache geometry, so it has
            no fallback: an omitted count would undersize the cache.
        numerics_envelopes: The ``--numerics`` envelopes the model is verified
            under, always including ``"auto"``. A model lists ``"rl-bitwise"``
            only once the bitwise invariance harness and the teacher-forced
            logprob comparison against the trainer pass for it; launching an
            unlisted envelope is refused.
        create_attention_config: Model-owned construction of the
            softmax attention configuration from the resolved launch, model
            config, and draft flag. Explicitly pass None to keep the
            architecture's generator.
            Returns an ``AttnConfig``; the runtime adds the profile's linear
            component afterwards. The factory owns validation of its cache
            format and execution constraints; its errors are not retried
            through the architecture's generator.
    """

    configure_attention: Callable[[ModelConfig, ServerArgs], None]
    cache_family: str
    linear_attention: str | None
    default_attention_backend: str | None
    request_token_history: bool
    is_multimodal: bool
    tokenizer_kwargs: Mapping[str, object]
    attention_instances_per_layer: int
    numerics_envelopes: frozenset[str]
    create_attention_config: (
        Callable[[ServerArgs, ModelConfig, bool], AttnConfig] | None
    )
    request_token_history_lookback: int | None = None

    def __post_init__(self) -> None:
        if self.create_attention_config is not None and not callable(
            self.create_attention_config
        ):
            raise TypeError(
                "ModelProfile.create_attention_config must be callable or None"
            )
        if self.request_token_history_lookback is not None and (
            not self.request_token_history
            or type(self.request_token_history_lookback) is not int
            or self.request_token_history_lookback < 0
        ):
            raise ValueError(
                "ModelProfile.request_token_history_lookback needs a model that "
                "reads request token history and a non-negative integer count"
            )
        if not self.cache_family:
            raise ValueError("ModelProfile.cache_family must name a cache family")
        if self.linear_attention == "":
            raise ValueError("ModelProfile.linear_attention must be None or a name")
        if self.attention_instances_per_layer < 1:
            raise ValueError(
                "ModelProfile.attention_instances_per_layer must be >= 1, got "
                f"{self.attention_instances_per_layer}"
            )
        envelopes = frozenset(self.numerics_envelopes)
        unknown = envelopes - frozenset(NUMERICS_ENVELOPES)
        if "auto" not in envelopes or unknown:
            raise ValueError(
                "ModelProfile.numerics_envelopes must include 'auto' and name "
                f"only {list(NUMERICS_ENVELOPES)}, got {sorted(envelopes)}"
            )
        object.__setattr__(self, "numerics_envelopes", envelopes)
        # Profiles are shared by every ModelConfig of the model; keep the one
        # mutable field read-only so no consumer can edit it for the others.
        object.__setattr__(
            self, "tokenizer_kwargs", MappingProxyType(dict(self.tokenizer_kwargs))
        )


def resolve_history_seed_lookback(*profiles: ModelProfile | None) -> int | None:
    """Bound seeds for all history readers, including the draft model.

    Unprofiled models and profiles that read no history add no requirement.
    Any reader without a declared bound requires the complete prefix;
    otherwise the largest lookback covers every reader. No readers returns
    None, with history allocation and seeding still disabled by the caller.
    """
    lookbacks = [
        profile.request_token_history_lookback
        for profile in profiles
        if profile is not None and profile.request_token_history
    ]
    if not lookbacks or any(value is None for value in lookbacks):
        return None
    return max(value for value in lookbacks if value is not None)


__all__ = ["ModelProfile", "resolve_history_seed_lookback"]
