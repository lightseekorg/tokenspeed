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

"""The model weight-update session on tiny stand-in MLA models (CPU only).

Live updates stream a checkpoint through many partial ``load_weights`` calls.
The session must (1) derive post-load state once, after the last chunk, (2)
keep derived tensors such as ``w_kc``/``w_vc`` at their captured addresses,
(3) apply in-place one-shot transforms (the LoRA norm scale) exactly once per
reloaded value, (4) work for subclasses that override ``load_weights`` /
``post_load_weights`` without session awareness or skip the base ``__init__``
(the speculative drafts), and (5) keep multi-tensor pairing state across
chunks.
"""

from __future__ import annotations

from types import SimpleNamespace
from unittest import mock

import pytest
import torch
from torch import nn

from tokenspeed.runtime.model_loader.weight_utils import bind_or_copy
from tokenspeed.runtime.models import deepseek_nextn
from tokenspeed.runtime.models.base import BaseCausalLM
from tokenspeed.runtime.models.base.weight_update import weight_update_session
from tokenspeed.runtime.models.deepseek_nextn import DeepseekV3ForCausalLMNextN
from tokenspeed.runtime.models.deepseek_v3 import (
    Eagle3DeepseekV2ForCausalLM,
    _prepare_mla_kv_b_proj_weights,
)
from tokenspeed.runtime.models.glm_moe_dsa_nextn import GlmMoeDsaForCausalLMNextN
from tokenspeed.runtime.models.llama_eagle3 import LlamaForCausalLMEagle3

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
    stream = [entry for entry in _checkpoint(2) if entry[0].startswith("layers.0.")]

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


def test_a_non_unit_kv_scale_is_loaded_derived_and_then_rejected(model):
    """The session screens every chunk like the initial load screens the
    checkpoint, whoever drives the calls (the Model Updater SDK streams
    straight into ``load_weights``). The update still completes and derives,
    so the model stays consistent, and is rejected at the end."""
    stream = _checkpoint(5)
    scales = [
        ("layers.0.attn.k_scale", torch.tensor([2.0])),
        ("layers.1.attn.v_scale", torch.ones(1)),  # unit: fine
    ]

    with pytest.raises(ValueError, match="layers.0.attn.k_scale") as info:
        with weight_update_session([model]):
            model.load_weights(stream[:2] + scales)
            model.load_weights(stream[2:])
    assert "v_scale" not in str(info.value)
    assert model.post_load_calls == 2
    assert not model._weight_update_active
    for attn in model.layers:
        w_kc, w_vc = _expected_pair(attn.kv_b_proj.weight.detach())
        assert torch.equal(attn.w_kc, w_kc)
    # The next session starts clean and a unit scale passes.
    with weight_update_session([model]):
        model.load_weights(_checkpoint(6) + scales[1:])
    assert model.post_load_calls == 3


def test_kv_scales_are_not_screened_outside_a_session(model):
    # The initial load screens the checkpoint itself (require_unit_kv_scales).
    model.load_weights([("layers.0.attn.k_scale", torch.tensor([2.0]))])
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


def test_bind_or_copy_keeps_storage_and_rejects_a_geometry_change():
    existing = torch.zeros(2, 3)
    same = bind_or_copy(existing, torch.ones(2, 3))
    assert same is existing and torch.equal(existing, torch.ones(2, 3))
    fresh = torch.ones(3, 2)
    assert bind_or_copy(None, fresh) is fresh
    # A live update rewrites values, never geometry.
    with pytest.raises(ValueError, match="geometry"):
        bind_or_copy(existing, fresh)
    with pytest.raises(ValueError, match="geometry"):
        bind_or_copy(existing, torch.ones(2, 3, dtype=torch.float64))


# ---------------------------------------------------------------------------
# Base-owned mechanics: overrides without session awareness, drafts that skip
# the base __init__.
# ---------------------------------------------------------------------------


class _NaiveLM(BaseCausalLM):
    """The pre-session convention, verbatim: ``load_weights`` ends in
    ``post_load_weights`` and knows nothing about sessions. Built the way the
    speculative drafts are, without ``BaseCausalLM.__init__``."""

    def __init__(self) -> None:
        nn.Module.__init__(self)
        self.layer = _TinyAttention()
        self.post_load_calls = 0
        self.seen_at_derive: list[set[str] | None] = []

    def load_weights(self, weights):
        params_dict = dict(self.named_parameters())
        loaded: set[str] = set()
        for name, weight in weights:
            params_dict[name].data.copy_(weight)
            loaded.add(name)
        self.post_load_weights()
        return loaded

    def post_load_weights(self):
        self.post_load_calls += 1
        self.seen_at_derive.append(
            None
            if self._weight_update_loaded_names is None
            else set(self._weight_update_loaded_names)
        )
        self.layer.w_kc, self.layer.w_vc = _prepare_mla_kv_b_proj_weights(
            self.layer.kv_b_proj.weight.detach(), self.layer
        )


def _naive_stream(seed: int) -> list[tuple[str, torch.Tensor]]:
    gen = torch.Generator().manual_seed(seed)
    return [
        (
            "layer.kv_b_proj.weight",
            torch.randn(HEADS * (NOPE + VDIM), LATENT, generator=gen),
        ),
        ("layer.q_a_layernorm.weight", torch.full((LATENT,), float(seed))),
    ]


def test_naive_overrides_derive_once_per_session_and_record_names():
    lm = _NaiveLM()
    lm.load_weights(_naive_stream(0))
    assert lm.post_load_calls == 1
    assert lm.seen_at_derive == [None]

    stream = _naive_stream(1)
    with weight_update_session([lm]):
        lm.load_weights(stream[:1])
        lm.load_weights(stream[1:])
        assert lm.post_load_calls == 1
    assert lm.post_load_calls == 2
    # The base class recorded the returned names and exposed them to the
    # deferred derivation.
    assert lm.seen_at_derive[-1] == {name for name, _ in stream}
    assert lm._weight_update_loaded_names is None
    w_kc, w_vc = _expected_pair(lm.layer.kv_b_proj.weight.detach())
    assert torch.equal(lm.layer.w_kc, w_kc) and torch.equal(lm.layer.w_vc, w_vc)


def test_a_delegating_override_reports_each_non_unit_scale_once():
    """A subclass loader that hands the stream to ``super()`` runs the screen
    twice (both ``load_weights`` are wrapped); the rejection names each
    scale once."""

    class _Delegating(_TinyLM):
        def load_weights(self, weights):
            return super().load_weights((name, weight) for name, weight in weights)

    lm = _Delegating()
    lm.load_weights(_checkpoint(0))
    with pytest.raises(ValueError) as info:
        with weight_update_session([lm]):
            lm.load_weights(
                _checkpoint(7) + [("layers.0.attn.k_scale", torch.tensor([0.5]))]
            )
    assert str(info.value).count("k_scale") == 1
    assert lm.post_load_calls == 2


def test_a_session_without_any_load_derives_nothing():
    lm = _NaiveLM()
    lm.load_weights(_naive_stream(0))
    with weight_update_session([lm]):
        pass
    assert lm.post_load_calls == 1


@pytest.mark.parametrize(
    "draft_cls",
    [
        Eagle3DeepseekV2ForCausalLM,
        DeepseekV3ForCausalLMNextN,
        GlmMoeDsaForCausalLMNextN,
        LlamaForCausalLMEagle3,
    ],
)
def test_drafts_built_without_the_base_init_take_part_in_sessions(draft_cls):
    # These drafts call ``nn.Module.__init__`` and set their own fields; the
    # session state must still exist on them.
    shell = draft_cls.__new__(draft_cls)
    nn.Module.__init__(shell)
    assert not shell._weight_update_active
    shell.record_loaded_weights({"outside.a.session"})
    assert shell._weight_update_loaded_names is None
    BaseCausalLM.begin_weight_update(shell)
    shell.record_loaded_weights({"model.x"})
    assert shell._weight_update_loaded_names == {"model.x"}
    BaseCausalLM.abort_weight_update(shell)
    assert not shell._weight_update_active


def test_end_of_one_model_raising_still_closes_the_others():
    first, second, third = _NaiveLM(), _NaiveLM(), _NaiveLM()
    for lm in (first, second, third):
        lm.load_weights(_naive_stream(0))

    def _boom():
        raise RuntimeError("derive failed")

    with pytest.raises(RuntimeError, match="derive failed"):
        with weight_update_session([first, second, third]):
            for lm in (first, second, third):
                lm.load_weights(_naive_stream(2))
            # The deferred derivation of ``second`` fails at end_weight_update.
            second.post_load_weights = _boom
    assert first.post_load_calls == 2
    # ``third`` was aborted, not derived, and every session is closed.
    assert third.post_load_calls == 1
    for lm in (first, second, third):
        assert not lm._weight_update_active
        assert lm._weight_update_loaded_names is None
    with weight_update_session([first, second, third]):
        pass


# ---------------------------------------------------------------------------
# Pairing state across chunks: the NextN draft's fused q/kv a-projection.
# ---------------------------------------------------------------------------

HID, QL, KVL, ROPE = 8, 4, 6, 2


class _DraftAttention(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.fused_qkv_a_proj_with_mqa = nn.Linear(HID, QL + KVL + ROPE, bias=False)
        self.kv_b_proj = nn.Linear(KVL, HEADS * (NOPE + VDIM), bias=False)
        self.qk_nope_head_dim = NOPE
        self.v_head_dim = VDIM
        self.w_kc: torch.Tensor | None = None
        self.w_vc: torch.Tensor | None = None


class _DraftModel(nn.Module):
    """Stands in for ``DeepseekModelNextN``: one decoder with MLA attention."""

    def __init__(self, config, mapping, quant_config) -> None:
        super().__init__()
        self.decoder = nn.Module()
        self.decoder.self_attn = _DraftAttention()


@pytest.fixture
def nextn() -> DeepseekV3ForCausalLMNextN:
    config = SimpleNamespace(
        q_lora_rank=QL,
        kv_lora_rank=KVL,
        num_nextn_predict_layers=1,
        num_hidden_layers=3,
        n_routed_experts=4,
        hidden_size=HID,
        vocab_size=16,
    )
    attn = SimpleNamespace(has_dp=True, tp_rank=0, tp_size=1, tp_group=None)
    # Attention DP with the default replicated LM head (no --lm-head-tp-size).
    lm_head = SimpleNamespace(has_tp=False, tp_rank=0, tp_size=1, tp_group=None)
    mapping = SimpleNamespace(
        attn=attn, lm_head=lm_head, moe=SimpleNamespace(ep_rank=0, ep_size=1)
    )
    with mock.patch.object(deepseek_nextn, "DeepseekModelNextN", _DraftModel):
        return DeepseekV3ForCausalLMNextN(config, mapping, None)


def _a_proj_pair(seed: int):
    gen = torch.Generator().manual_seed(seed)
    prefix = "model.layers.3.self_attn"
    return (
        (f"{prefix}.q_a_proj.weight", torch.randn(QL, HID, generator=gen)),
        (
            f"{prefix}.kv_a_proj_with_mqa.weight",
            torch.randn(KVL + ROPE, HID, generator=gen),
        ),
    )


def test_nextn_pairs_q_a_and_kv_a_across_chunks(nextn):
    attn = nextn.model.decoder.self_attn
    q_a, kv_a = _a_proj_pair(0)
    nextn.load_weights([q_a, kv_a])
    assert torch.equal(
        attn.fused_qkv_a_proj_with_mqa.weight, torch.cat([q_a[1], kv_a[1]])
    )
    pointers = (attn.w_kc.data_ptr(), attn.w_vc.data_ptr())

    q_a, kv_a = _a_proj_pair(1)
    with weight_update_session([nextn]):
        nextn.load_weights([q_a])
        # Half a pair: the fused projection keeps waiting.
        assert nextn._pending_a_proj
        nextn.load_weights([kv_a])
        assert not nextn._pending_a_proj
    assert torch.equal(
        attn.fused_qkv_a_proj_with_mqa.weight, torch.cat([q_a[1], kv_a[1]])
    )
    assert (attn.w_kc.data_ptr(), attn.w_vc.data_ptr()) == pointers


def test_nextn_rejects_an_update_whose_partner_never_arrived(nextn):
    attn = nextn.model.decoder.self_attn
    q_a, kv_a = _a_proj_pair(0)
    nextn.load_weights([q_a, kv_a])
    before = attn.fused_qkv_a_proj_with_mqa.weight.clone()

    with pytest.raises(RuntimeError, match="q_a_proj"):
        with weight_update_session([nextn]):
            nextn.load_weights([_a_proj_pair(1)[0]])
    assert torch.equal(attn.fused_qkv_a_proj_with_mqa.weight, before)
    assert not nextn._pending_a_proj
    assert not nextn._weight_update_active
    # The next session starts clean.
    q_a, kv_a = _a_proj_pair(2)
    with weight_update_session([nextn]):
        nextn.load_weights([q_a, kv_a])
    assert torch.equal(
        attn.fused_qkv_a_proj_with_mqa.weight, torch.cat([q_a[1], kv_a[1]])
    )


# ---------------------------------------------------------------------------
# GLM's FP8 fused QKV-A pad replaces the parameter; the loader must survive.
# ---------------------------------------------------------------------------


def test_fp8_qkv_a_pad_keeps_the_loader_attributes_for_live_updates():
    from tokenspeed.runtime.layers.linear import ReplicatedLinear
    from tokenspeed.runtime.models.glm5 import (
        pad_fused_qkv_a_proj_weight_for_fp8_blockscale,
    )

    n, k = 2624, 16  # GLM-5.1's unaligned N (q_lora + kv_lora + rope)
    proj = nn.Module()
    proj.weight = nn.Parameter(
        torch.ones(n, k).to(torch.float8_e4m3fn), requires_grad=False
    )
    loader = ReplicatedLinear.weight_loader.__get__(proj)
    proj.weight.weight_loader = loader
    proj.weight.output_dim = 0
    attn = SimpleNamespace(fused_qkv_a_proj_with_mqa=proj)

    pad_fused_qkv_a_proj_weight_for_fp8_blockscale(attn)

    padded = proj.weight
    assert padded.shape == (2688, k)
    assert padded.weight_loader is loader and padded.output_dim == 0
    assert torch.all(padded[n:].float() == 0)
    # The next live update's fused branch streams a shard by row offset.
    shard = torch.full((4, k), 2.0).to(torch.float8_e4m3fn)
    padded.weight_loader(padded, shard, begin_size=8)
    assert torch.all(padded[8:12].float() == 2.0)
    assert torch.all(padded[n:].float() == 0)
    # Aligned now: a re-run (every load_weights ends in the pad) is a no-op.
    pad_fused_qkv_a_proj_weight_for_fp8_blockscale(attn)
    assert proj.weight is padded
