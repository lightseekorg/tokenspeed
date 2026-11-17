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

"""Native target feature capture through the real model and logits processor."""

from test.runtime.models.test_inkling2_model import _mapping, _model
from test.runtime.models.test_inkling2_mtp import _config as _draft_config
from test.runtime.models.test_inkling2_mtp import _native_weights

import pytest
import torch

from tokenspeed.runtime.configs.inkling2 import Inkling2MTPConfig
from tokenspeed.runtime.execution.context import ForwardContext
from tokenspeed.runtime.execution.forward_batch_info import (
    CaptureHiddenMode,
    ForwardMode,
)
from tokenspeed.runtime.execution.output_layout import ForwardOutputLayout
from tokenspeed.runtime.layers.attention_residual import AttentionResidualState
from tokenspeed.runtime.models.inkling2 import (
    Inkling2DecoderLayer,
    Inkling2ForConditionalGeneration,
)
from tokenspeed.runtime.models.inkling2_mtp import Inkling2ForConditionalGenerationMTP
from tokenspeed.runtime.models.inkling2_sequence import SequenceShare


def _models() -> (
    tuple[Inkling2ForConditionalGeneration, Inkling2ForConditionalGenerationMTP]
):
    target = _model()
    target.config.mtp_config = _draft_config().to_dict()
    target.config.mtp_config["target_layer_ids"] = [0, 1, 2]
    draft = Inkling2ForConditionalGenerationMTP(
        Inkling2MTPConfig.from_target(target.config), _mapping(), None
    )
    # Serving loads the draft before wiring it to the target.
    draft.load_weights(_native_weights(draft))
    return target, draft


def _context(rows: int) -> ForwardContext:
    return ForwardContext(
        attn_backend=None,
        token_to_kv_pool=None,
        bs=rows,
        num_extends=0,
        input_num_tokens=rows,
        forward_mode=ForwardMode.DECODE if rows else ForwardMode.IDLE,
        output_layout=ForwardOutputLayout(0, 0, rows, 1),
        capture_hidden_mode=CaptureHiddenMode.FULL,
    )


class _CaptureSink:
    def __init__(self) -> None:
        self.rows: list[tuple[int, torch.Tensor]] = []

    def on_target_capture(self, index: int, hidden: torch.Tensor) -> None:
        self.rows.append((index, hidden.clone()))


def test_native_capture_selects_raw_entries_then_final_norm(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    target, draft = _models()

    def cpu_layer(
        layer: Inkling2DecoderLayer,
        state: AttentionResidualState,
        ctx: ForwardContext,
        tau: torch.Tensor | None,
        mailbox: object,
        share: SequenceShare | None,
    ) -> None:
        # Replace only GPU decoder work. The actual loop and residual owner run.
        attention = torch.full_like(
            state.partial, layer.layer_id + 0.51, dtype=torch.bfloat16
        )
        mlp = torch.full_like(
            state.partial, (layer.layer_id + 1) * 0.003, dtype=torch.bfloat16
        )
        state.finalize_layer(layer.layer_id, attention, mlp)

    monkeypatch.setattr(Inkling2DecoderLayer, "forward", cpu_layer)
    with torch.no_grad():
        target.model.norm.weight.fill_(3)
    embeds = torch.full((2, 32), 0.75, dtype=torch.bfloat16)
    ids, positions = torch.zeros(2, dtype=torch.long), torch.arange(2)
    baseline, no_taps = target.model(ids, positions, _context(2), embeds)
    assert no_taps is None
    draft.configure_target(target, target.text_config)
    sink = _CaptureSink()
    ctx = _context(2)
    ctx.target_capture_sink = sink
    output, taps = target.model(ids, positions, ctx, embeds)
    assert torch.equal(output, baseline)
    assert len(taps) == 3
    assert torch.equal(taps[0], embeds)
    partial = (
        torch.full((2, 32), 0.51, dtype=torch.bfloat16).float()
        + torch.full((2, 32), 0.003, dtype=torch.bfloat16).float()
    )
    assert torch.equal(taps[1], partial.bfloat16())
    assert not torch.equal(taps[1], (embeds.float() + partial).bfloat16())
    assert torch.equal(taps[2], torch.full_like(output, 3))
    assert torch.equal(taps[2], output)
    assert all(tap.dtype == torch.bfloat16 for tap in taps)
    assert [index for index, _ in sink.rows] == [0, 1, 2]
    assert all(
        torch.equal(recorded, expected)
        for (_, recorded), expected in zip(sink.rows, taps)
    )


def test_idle_capture_preserves_packed_width_through_model_exit() -> None:
    target, draft = _models()
    ids = torch.empty(0, dtype=torch.long)
    embeds = torch.empty(0, 32, dtype=torch.bfloat16)
    plain = target(_context(0), ids, ids, input_embeds=embeds)
    assert plain.hidden_states.shape == (0, 32)
    draft.configure_target(target, target.text_config)
    captured = target(_context(0), ids, ids, input_embeds=embeds)
    assert captured.hidden_states.shape == (0, 96)
    assert captured.hidden_states.dtype == torch.bfloat16
    assert captured.next_token_logits.shape == (0, 60)


def test_repeated_capture_setup_copies_the_trained_selection() -> None:
    target, draft = _models()
    draft.configure_target(target, target.text_config)
    draft.configure_target(target, target.text_config)
    selection = [0, 1, 2]
    target.set_dspark_layers_to_capture(selection)
    selection.clear()
    output = target(
        _context(0),
        torch.empty(0, dtype=torch.long),
        torch.empty(0, dtype=torch.long),
        input_embeds=torch.empty(0, 32, dtype=torch.bfloat16),
    )
    assert output.hidden_states.shape == (0, 96)


@pytest.mark.parametrize("ids", [[0, 2], [1, 0, 2], [0, 1, 1, 2]])
def test_wrong_tap_selection_does_not_change_existing_capture(ids: list[int]) -> None:
    target, draft = _models()
    draft.configure_target(target, target.text_config)
    with pytest.raises(ValueError, match="checkpoint target_layer_ids"):
        target.set_dspark_layers_to_capture(ids)
    output = target(
        _context(0),
        torch.empty(0, dtype=torch.long),
        torch.empty(0, dtype=torch.long),
        input_embeds=torch.empty(0, 32, dtype=torch.bfloat16),
    )
    assert output.hidden_states.shape == (0, 96)


@pytest.mark.parametrize(
    "field",
    [
        "hidden_size",
        "target_num_hidden_layers",
        "padded_vocab_size",
        "unpadded_vocab_size",
    ],
)
def test_incompatible_draft_geometry_fails_before_capture_changes(field: str) -> None:
    target, draft = _models()
    values = draft.config.to_dict()
    values[field] += 1
    draft.config = Inkling2MTPConfig(**values)
    with pytest.raises(ValueError, match="target geometry"):
        draft.configure_target(target, target.text_config)
    output = target(
        _context(0),
        torch.empty(0, dtype=torch.long),
        torch.empty(0, dtype=torch.long),
        input_embeds=torch.empty(0, 32, dtype=torch.bfloat16),
    )
    assert output.hidden_states.shape == (0, 32)


def test_generic_draft_capture_contracts_remain_rejected() -> None:
    target, _ = _models()
    with pytest.raises(NotImplementedError):
        target.set_dflash_layers_to_capture([0])
    with pytest.raises(NotImplementedError):
        target.set_eagle3_layers_to_capture([0])


def test_native_draft_rejects_a_different_target_family() -> None:
    target, draft = _models()
    with pytest.raises(TypeError, match="Inkling2 target"):
        draft.configure_target(torch.nn.Identity(), target.text_config)
