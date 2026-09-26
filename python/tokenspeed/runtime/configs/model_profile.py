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

if TYPE_CHECKING:
    from tokenspeed.runtime.configs.model_config import ModelConfig


@dataclass(frozen=True, kw_only=True)
class ModelProfile:
    """Model-owned answers to the questions architecture tables used to answer.

    A registered model class exposes ``model_profile(hf_config)`` as a
    classmethod returning one of these. ``ModelConfig`` resolves it once, and
    every resolver that would otherwise match the architecture name against a
    table reads the profile instead. Every field is explicit: a profile that
    omits a fact is a construction error, never a silent fallback.

    Attributes:
        configure_attention: Writes the attention geometry onto the
            ``ModelConfig`` (``attention_arch``, head dimensions, scaling),
            e.g. ``configure_mla_attention`` or ``configure_dsa_attention``.
        cache_family: Registered cache recipe and pool family that owns this
            model's per-request state.
        linear_attention: Registered linear-attention backend serving the
            model's linear layers, or None for a model without them.
        default_attention_backend: Attention backend used when the launch
            names none, or None to keep the architecture default.
        default_prefix_granularity: Prefix granularity used when the launch
            keeps the server default, or None to keep that default.
        request_token_history: Whether the model reads each request's
            committed token history (``ForwardContext.request_token_history``).
        tokenizer_kwargs: Extra keyword arguments for the model's tokenizer.
        attention_instances_per_layer: Attention modules per decoder layer.
            Paired layouts (two attention branches sharing one layer's MLP
            block, e.g. LongCat's ScMoE) declare 2 so the cache plans one
            plane per branch; the default 1 is the ordinary stack.
    """

    configure_attention: Callable[[ModelConfig], None]
    cache_family: str
    linear_attention: str | None
    default_attention_backend: str | None
    default_prefix_granularity: int | None
    request_token_history: bool
    tokenizer_kwargs: Mapping[str, object]
    attention_instances_per_layer: int = 1

    def __post_init__(self) -> None:
        if not self.cache_family:
            raise ValueError("ModelProfile.cache_family must name a cache family")
        if self.linear_attention == "":
            raise ValueError("ModelProfile.linear_attention must be None or a name")
        if self.attention_instances_per_layer < 1:
            raise ValueError(
                "ModelProfile.attention_instances_per_layer must be >= 1, got "
                f"{self.attention_instances_per_layer}"
            )
        if self.default_prefix_granularity is not None and (
            self.default_prefix_granularity <= 0
        ):
            raise ValueError(
                "ModelProfile.default_prefix_granularity must be positive, got "
                f"{self.default_prefix_granularity}"
            )
        # Profiles are shared by every ModelConfig of the model; keep the one
        # mutable field read-only so no consumer can edit it for the others.
        object.__setattr__(
            self, "tokenizer_kwargs", MappingProxyType(dict(self.tokenizer_kwargs))
        )


__all__ = ["ModelProfile"]
