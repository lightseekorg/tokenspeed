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

"""Extensible wrappers for injecting custom input and output processors."""

import importlib
from typing import Any

from torch import Tensor, nn

from tokenspeed.runtime.execution.context import ForwardContext
from tokenspeed.runtime.layers.logits_processor import (
    LogitsMetadata,
    LogitsProcessorOutput,
)
from tokenspeed.runtime.utils import get_colorful_logger

logger = get_colorful_logger(__name__)


# Used for Input/Output Processor sharing
class ContextBase:
    """Base shared context for extensible input and output processors."""

    def __init__(self, base_lm, config_dict: dict[str, Any]):
        pass


class InputProcessorBase(nn.Module):
    """Default input processor that falls back to token embeddings."""

    def __init__(self, base_lm, ctx, config_dict: dict[str, Any]):
        super().__init__()
        self.base_lm = base_lm

    def forward(
        self,
        input_ids: Tensor,
        positions: Tensor,
        ctx: ForwardContext,
        out_cache_loc: Tensor,
        input_embeds: Tensor = None,
    ) -> Tensor:
        if input_embeds is not None:
            return input_embeds
        return self.base_lm.model.embed_tokens(input_ids)


class OutputProcessorBase(nn.Module):
    """Default output processor that routes hidden states to logits."""

    def __init__(self, base_lm, ctx, config_dict: dict[str, Any]):
        super().__init__()
        self.base_lm = base_lm

    def forward(
        self,
        input_ids: Tensor,
        positions: Tensor,
        ctx: ForwardContext,
        output_hidden_states: Tensor,
    ) -> LogitsProcessorOutput:
        logits_metadata = LogitsMetadata.from_forward_context(ctx)
        return self.base_lm.logits_processor(
            input_ids,
            output_hidden_states,
            self.base_lm.lm_head,
            logits_metadata,
        )


_EXT_CLS_REGISTRY: dict[str, type] = {}


def register_ext_cls(name: str, cls: type) -> None:
    global _EXT_CLS_REGISTRY
    _EXT_CLS_REGISTRY[name] = cls


def get_ext_cls(name: str) -> type:
    if name not in _EXT_CLS_REGISTRY:
        raise ValueError(
            f"Input module {name} not found in registry. {_EXT_CLS_REGISTRY=}"
        )
    return _EXT_CLS_REGISTRY[name]


register_ext_cls("ContextBase", ContextBase)
register_ext_cls("InputProcessorBase", InputProcessorBase)
register_ext_cls("OutputProcessorBase", OutputProcessorBase)


class ExtensibleLM(nn.Module):
    """Wrap a base LM with pluggable context, input, and output processors."""

    def __init__(
        self,
        base_lm: nn.Module,
        ext_config: dict[str, Any],
    ) -> None:
        super().__init__()
        self.base_lm = base_lm

        if "ext_def_file" in ext_config:
            import os
            import sys
            from pathlib import Path

            ext_def_file = ext_config["ext_def_file"]
            ext_def_dir = os.path.dirname(os.path.abspath(ext_def_file))
            sys.path.insert(0, ext_def_dir)
            ext_def_module = f"{Path(ext_def_file).stem}"
            logger.info(
                "\x1b[32m[[ExtensibleLM] Loading ext_def_dir=%r, ext_def_module=%r]\x1b[0m",
                ext_def_dir,
                ext_def_module,
            )
            importlib.import_module(ext_def_module)

        ctx_config = ext_config["context"]
        ctx_name = ctx_config.pop("cls")
        ctx_cls = get_ext_cls(ctx_name)
        self.ctx: ContextBase = ctx_cls(base_lm, ctx_config)

        input_processor_config = ext_config["input_processor"]
        input_processor_name = input_processor_config.pop("cls")
        input_processor_cls = get_ext_cls(input_processor_name)
        self.input_processor: InputProcessorBase = input_processor_cls(
            self.base_lm,
            self.ctx,
            input_processor_config,
        ).eval()

        output_processor_config = ext_config["output_processor"]
        output_processor_name = output_processor_config.pop("cls")
        output_processor_cls = get_ext_cls(output_processor_name)
        self.output_processor: OutputProcessorBase = output_processor_cls(
            self.base_lm,
            self.ctx,
            output_processor_config,
        ).eval()
        self.step = 0

    def forward(
        self,
        ctx: ForwardContext,
        input_ids: Tensor,
        positions: Tensor,
        out_cache_loc: Tensor,
        input_embeds: Tensor = None,
    ) -> LogitsProcessorOutput:
        # input processor: get input hidden states
        input_embeds = self.input_processor(
            input_ids, positions, ctx, out_cache_loc, input_embeds
        )

        # base model forward
        out_hidden_states, _ = self.base_lm.model(
            input_ids=None,
            positions=positions,
            ctx=ctx,
            out_cache_loc=out_cache_loc,
            input_embeds=input_embeds,
        )

        # output processor: lm hidden states to logits
        logits_output: LogitsProcessorOutput = self.output_processor(
            input_ids, positions, ctx, out_hidden_states
        )
        self.step += 1
        return logits_output
