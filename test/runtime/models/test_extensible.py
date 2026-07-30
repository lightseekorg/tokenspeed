from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch
from torch import nn

from tokenspeed.runtime.models.extensible import (
    ExtensibleLM,
    _forward_accepts_kwarg,
)


class _BaseModel(nn.Module):
    def forward(self, **kwargs):
        return kwargs["input_embeds"], None


class _InputProcessor(nn.Module):
    def forward(self, input_ids, positions, ctx, out_cache_loc, input_embeds):
        return input_embeds


class _LegacyOutputProcessor(nn.Module):
    def forward(self, input_ids, positions, ctx, output_hidden_states):
        return output_hidden_states


class _GatherOutputProcessor(nn.Module):
    def __init__(self):
        super().__init__()
        self.gather_ids = None

    def forward(
        self,
        input_ids,
        positions,
        ctx,
        output_hidden_states,
        gather_ids=None,
    ):
        self.gather_ids = gather_ids
        return output_hidden_states


def _make_extensible(output_processor: nn.Module) -> ExtensibleLM:
    model = ExtensibleLM.__new__(ExtensibleLM)
    nn.Module.__init__(model)
    model.base_lm = SimpleNamespace(model=_BaseModel())
    model.input_processor = _InputProcessor()
    model.output_processor = output_processor
    model._output_processor_accepts_gather_ids = _forward_accepts_kwarg(
        output_processor, "gather_ids"
    )
    model.step = 0
    return model


@pytest.mark.parametrize("gather_ids", [None, torch.tensor([1])])
def test_extensible_lm_preserves_legacy_output_processor_contract(gather_ids):
    model = _make_extensible(_LegacyOutputProcessor())
    hidden_states = torch.randn(2, 3)

    output = model(
        ctx=object(),
        input_ids=torch.tensor([1, 2]),
        positions=torch.tensor([0, 1]),
        out_cache_loc=torch.tensor([0, 1]),
        input_embeds=hidden_states,
        gather_ids=gather_ids,
    )

    assert output is hidden_states


def test_extensible_lm_forwards_gather_ids_when_supported():
    processor = _GatherOutputProcessor()
    model = _make_extensible(processor)
    gather_ids = torch.tensor([1])

    model(
        ctx=object(),
        input_ids=torch.tensor([1, 2]),
        positions=torch.tensor([0, 1]),
        out_cache_loc=torch.tensor([0, 1]),
        input_embeds=torch.randn(2, 3),
        gather_ids=gather_ids,
    )

    assert processor.gather_ids is gather_ids
