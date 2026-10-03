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

"""The model weight-update session on a tiny stand-in MLA model (CPU only).

Live updates stream a checkpoint through many partial ``load_weights`` calls.
The session must (1) derive post-load state once, after the last chunk, (2)
keep derived tensors such as ``w_kc``/``w_vc`` at their captured addresses,
and (3) apply in-place one-shot transforms (the LoRA norm scale) exactly once
per reloaded value.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch
from torch import nn

from tokenspeed.runtime.model_loader.weight_utils import bind_or_copy
from tokenspeed.runtime.models.base import BaseCausalLM
from tokenspeed.runtime.models.base.weight_update import weight_update_session
from tokenspeed.runtime.models.deepseek_v3 import _prepare_mla_kv_b_proj_weights

HEADS, NOPE, VDIM, LATENT = 2, 3, 2, 4
SCALE = 3.0


class _TinyAttention(nn.Module):
    """``kv_b_proj`` plus the absorbed pair, like ``DeepseekV3AttentionMLA``."""

    def __init__(self) -> None:
        super().__init__()
        self.kv_b_proj = nn.Linear(LATENT, HEADS * (NOPE + VDIM), bias=False)
        self.q_a_layernorm = nn.Module()
        self.q_a_layernorm.weight = nn.Parameter(torch.ones(LATENT))
        self.qk_nope_head_dim = NOPE
        self.v_head_dim = VDIM
        self.w_kc: torch.Tensor | None = None
        self.w_vc: torch.Tensor | None = None


class _TinyLM(BaseCausalLM):
    """The in-tree MLA convention: ``load_weights`` ends in ``post_load_weights``,
    which rebuilds the absorbed pair and folds a scale into the norm weight."""

    def __init__(self) -> None:
        super().__init__(
            config=SimpleNamespace(), mapping=SimpleNamespace(), encoder_only=True
        )
        self.layers = nn.ModuleList([_TinyAttention(), _TinyAttention()])
        self.post_load_calls = 0

    def post_load_weights(self) -> None:
        self.post_load_calls += 1
        reloaded = self._weight_update_loaded_names
        names = {id(param): name for name, param in self.named_parameters()}
        for attn in self.layers:
            attn.w_kc, attn.w_vc = _prepare_mla_kv_b_proj_weights(
                attn.kv_b_proj.weight.detach(), attn
            )
            norm = attn.q_a_layernorm.weight
            if reloaded is None or names[id(norm)] in reloaded:
                norm.data *= SCALE


def _checkpoint(seed: int) -> list[tuple[str, torch.Tensor]]:
    gen = torch.Generator().manual_seed(seed)
    stream = []
    for i in range(2):
        stream.append(
            (
                f"layers.{i}.kv_b_proj.weight",
                torch.randn(HEADS * (NOPE + VDIM), LATENT, generator=gen),
            )
        )
        stream.append((f"layers.{i}.q_a_layernorm.weight", torch.ones(LATENT)))
    return stream


def _expected_pair(weight: torch.Tensor):
    probe = _TinyAttention()
    return _prepare_mla_kv_b_proj_weights(weight, probe)


@pytest.fixture
def model() -> _TinyLM:
    lm = _TinyLM()
    loaded = lm.load_weights(_checkpoint(0))
    assert loaded == {name for name, _ in _checkpoint(0)}
    assert lm.post_load_calls == 1
    return lm


def test_initial_load_derives_once_and_scales_every_norm(model):
    for attn in model.layers:
        assert torch.equal(attn.q_a_layernorm.weight, torch.full((LATENT,), SCALE))
        w_kc, w_vc = _expected_pair(attn.kv_b_proj.weight.detach())
        assert torch.equal(attn.w_kc, w_kc)
        assert torch.equal(attn.w_vc, w_vc)


def test_partial_streams_in_a_session_derive_once_at_the_end(model):
    stream = _checkpoint(1)
    first, second = stream[:2], stream[2:]
    pointers = [(a.w_kc.data_ptr(), a.w_vc.data_ptr()) for a in model.layers]

    with weight_update_session([model]):
        model.load_weights(first)
        # Nothing derived mid-stream.
        assert model.post_load_calls == 1
        model.load_weights(second)
        assert model.post_load_calls == 1
    assert model.post_load_calls == 2

    for attn, (kc_ptr, vc_ptr) in zip(model.layers, pointers):
        # Same storage, new values: captured graphs keep valid addresses.
        assert (attn.w_kc.data_ptr(), attn.w_vc.data_ptr()) == (kc_ptr, vc_ptr)
        w_kc, w_vc = _expected_pair(attn.kv_b_proj.weight.detach())
        assert torch.equal(attn.w_kc, w_kc)
        assert torch.equal(attn.w_vc, w_vc)
        # Reloaded as ones, scaled exactly once.
        assert torch.equal(attn.q_a_layernorm.weight, torch.full((LATENT,), SCALE))


def test_norm_scale_applies_only_to_reloaded_layers(model):
    stream = [
        entry for entry in _checkpoint(2) if entry[0].startswith("layers.0.")
    ]

    with weight_update_session([model]):
        model.load_weights(stream)

    assert torch.equal(
        model.layers[0].q_a_layernorm.weight, torch.full((LATENT,), SCALE)
    )
    # Layer 1 kept its initial-load value; a blanket re-scale would square it.
    assert torch.equal(
        model.layers[1].q_a_layernorm.weight, torch.full((LATENT,), SCALE)
    )


def test_outside_a_session_each_call_derives_as_before(model):
    model.load_weights(_checkpoint(3)[:2])
    assert model.post_load_calls == 2
    model.load_weights(_checkpoint(3)[2:])
    assert model.post_load_calls == 3


def test_a_failed_update_leaves_no_session_behind_and_skips_derivation(model):
    with pytest.raises(RuntimeError, match="store"):
        with weight_update_session([model]):
            model.load_weights(_checkpoint(4)[:1])
            raise RuntimeError("store unreachable")
    assert model.post_load_calls == 1
    assert not model._weight_update_active
    # The next session starts clean.
    with weight_update_session([model]):
        model.load_weights(_checkpoint(4))
    assert model.post_load_calls == 2


def test_nested_sessions_are_rejected(model):
    with weight_update_session([model]):
        with pytest.raises(RuntimeError, match="already active"):
            model.begin_weight_update()
    with pytest.raises(RuntimeError, match="no weight-update session"):
        model.end_weight_update()


def test_models_outside_the_protocol_are_left_alone():
    class _Plain(nn.Module):
        def __init__(self):
            super().__init__()
            self.calls = 0

        def load_weights(self, weights):
            self.calls += len(list(weights))

    plain = _Plain()
    with weight_update_session([plain]):
        plain.load_weights([("a", torch.ones(1))])
    assert plain.calls == 1


def test_bind_or_copy_keeps_storage_only_for_matching_geometry():
    existing = torch.zeros(2, 3)
    same = bind_or_copy(existing, torch.ones(2, 3))
    assert same is existing and torch.equal(existing, torch.ones(2, 3))
    fresh = torch.ones(3, 2)
    assert bind_or_copy(existing, fresh) is fresh
    assert bind_or_copy(None, fresh) is fresh
    other_dtype = torch.ones(2, 3, dtype=torch.float64)
    assert bind_or_copy(existing, other_dtype) is other_dtype
