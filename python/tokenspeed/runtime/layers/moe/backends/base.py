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

from abc import ABC, abstractmethod
from collections.abc import Callable
from functools import partial
from typing import ClassVar

import torch
from torch import nn

from tokenspeed.runtime.layers.moe.backends.weight_loaders import (
    load_group_weight_scale,
    load_model_weight,
    load_per_channel_weight_scale,
    load_per_tensor_weight_scale,
)
from tokenspeed.runtime.layers.moe.core.types import BackendKey, MoELayerSpec
from tokenspeed.runtime.layers.moe.topk import TopKOutputFormat


class MoEBackend(ABC):
    # Static hardware capability declaration. Keep dynamic shape/quant checks in
    # supports(spec, quant_config).
    supported_arches: ClassVar[frozenset[str]] = frozenset({"any"})

    def __init__(
        self,
        key: BackendKey,
        spec: MoELayerSpec,
        quant_config: object,
        routing_config: dict | None = None,
    ):
        self.key = key
        self.spec = spec
        self.quant_config = quant_config
        self.routing_config = routing_config

    @classmethod
    @abstractmethod
    def supports(cls, spec: MoELayerSpec, quant_config: object) -> bool:
        raise NotImplementedError

    @abstractmethod
    def create_layer_weights(
        self, layer: nn.Module, *, with_bias: bool = False
    ) -> None:
        raise NotImplementedError

    def process_weights_after_loading(self, layer: nn.Module) -> None:
        pass

    @abstractmethod
    def forward(
        self,
        layer: nn.Module,
        hidden_states: torch.Tensor,
        topk_output: object,
        num_global_tokens: int,
        max_num_tokens_per_gpu: int,
    ) -> torch.Tensor:
        raise NotImplementedError

    @property
    def apply_routed_scaling_factor_on_output(self) -> bool:
        return True

    @property
    def supports_deferred_finalize(self) -> bool:
        """Whether this backend can return the raw (unfinalized) MoE output
        as a 3-tuple ``(gemm2_out, expert_weights,
        expanded_idx_to_permuted_idx)`` when called with ``do_finalize=False``.
        Only backends that set this to True are called with the
        ``do_finalize`` kwarg; the default finalized path is otherwise
        unchanged.
        """
        return False

    @property
    def topk_output_format(self) -> TopKOutputFormat:
        return TopKOutputFormat.STANDARD

    def _make_weight_loader(
        self,
        *,
        is_bias: bool = False,
        do_transpose: bool = False,
        use_presharded_weights: bool = False,
    ) -> Callable:
        return partial(
            load_model_weight,
            tp_rank=self.spec.tp_rank,
            is_bias=is_bias,
            use_presharded_weights=use_presharded_weights,
            do_transpose=do_transpose,
            tp_size=self.spec.tp_size,
        )

    def _make_group_scale_loader(self, *, do_transpose: bool = False) -> Callable:
        return partial(
            load_group_weight_scale,
            tp_rank=self.spec.tp_rank,
            do_transpose=do_transpose,
        )

    def _make_per_channel_scale_loader(self, *, do_transpose: bool = False) -> Callable:
        return partial(
            load_per_channel_weight_scale,
            tp_rank=self.spec.tp_rank,
            do_transpose=do_transpose,
        )

    @staticmethod
    def _per_tensor_scale_loader() -> Callable:
        return load_per_tensor_weight_scale
