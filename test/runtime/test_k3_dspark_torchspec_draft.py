"""Tests for the two fields a TorchSpec-trained K3 DSpark draft adds.

``fc_norm`` -- a per-tap RMSNorm applied to each target hidden-state chunk
before concatenation and ``context_proj``.

``aux_hidden_stream: attn_res`` -- the target taps read the pre-norm AttnRes
mixture rather than the running prefix sum.
"""

from __future__ import annotations

import copy
from types import SimpleNamespace
from unittest import mock

import pytest
import torch

from tokenspeed.runtime.configs.kimi_k3_dspark_config import (
    KimiK3DSparkConfig,
    validate_k3_dspark_config,
)
from tokenspeed.runtime.execution.drafter.dflash import (
    AUX_HIDDEN_STREAM_ENV,
    DFlash,
    _resolve_aux_hidden_stream,
)

# The Inferact/Kimi-K3-DSpark config.json, which declares neither field.
INFERACT_CONFIG = dict(
    hidden_size=7168,
    num_hidden_layers=5,
    num_attention_heads=64,
    q_lora_rank=1536,
    vocab_size=163840,
    num_target_layers=5,
    target_hidden_size=7168,
    target_num_hidden_layers=93,
    target_layer_ids=[2, 23, 47, 71, 89],
    mask_token_id=163837,
    markov_rank=256,
)

# Ours: exports/native-iter_0006891.
TORCHSPEC_OVERRIDES = dict(
    target_layer_ids=[7, 31, 47, 63, 87],
    fc_norm=True,
    aux_hidden_stream="attn_res",
)


def make_config(**overrides) -> KimiK3DSparkConfig:
    fields = copy.deepcopy(INFERACT_CONFIG)
    fields.update(overrides)
    return KimiK3DSparkConfig(**fields)


# --------------------------------------------------------------------------
# Config contract
# --------------------------------------------------------------------------


def test_the_published_draft_keeps_the_historical_defaults() -> None:
    """Neither field appears in Inferact's config.json, so both must default."""
    config = make_config()
    assert config.fc_norm is False
    assert config.aux_hidden_stream == "prefix"
    validate_k3_dspark_config(config)


def test_the_torchspec_draft_validates() -> None:
    config = make_config(**TORCHSPEC_OVERRIDES)
    assert config.fc_norm is True
    assert config.aux_hidden_stream == "attn_res"
    validate_k3_dspark_config(config)


def test_an_unknown_aux_stream_is_rejected_rather_than_ignored() -> None:
    with pytest.raises(ValueError, match="aux_hidden_stream"):
        validate_k3_dspark_config(make_config(aux_hidden_stream="attnres"))


# --------------------------------------------------------------------------
# Resolving the stream: checkpoint declares, env overrides
# --------------------------------------------------------------------------


def test_stream_defaults_to_prefix_when_undeclared() -> None:
    assert _resolve_aux_hidden_stream(SimpleNamespace()) == "prefix"


def test_stream_comes_from_the_draft_config() -> None:
    assert _resolve_aux_hidden_stream(make_config(**TORCHSPEC_OVERRIDES)) == "attn_res"


def test_stream_is_also_read_from_a_nested_dflash_config() -> None:
    cfg = SimpleNamespace(dflash_config={"aux_hidden_stream": "attn_res"})
    assert _resolve_aux_hidden_stream(cfg) == "attn_res"


def test_the_env_var_overrides_the_checkpoint_for_ab_runs(monkeypatch) -> None:
    monkeypatch.setenv(AUX_HIDDEN_STREAM_ENV, "PREFIX")
    assert _resolve_aux_hidden_stream(make_config(**TORCHSPEC_OVERRIDES)) == "prefix"


# --------------------------------------------------------------------------
# Wiring the target
# --------------------------------------------------------------------------


def _drafter(config) -> SimpleNamespace:
    holder = SimpleNamespace(model=SimpleNamespace(config=config))
    holder._wire_aux_hidden_stream = DFlash._wire_aux_hidden_stream.__get__(holder)
    return holder


def test_the_resolved_stream_is_pushed_to_the_target() -> None:
    holder = _drafter(make_config(**TORCHSPEC_OVERRIDES))
    target = mock.Mock(spec=["set_dflash_aux_hidden_stream"])
    holder._wire_aux_hidden_stream(target)
    target.set_dflash_aux_hidden_stream.assert_called_once_with("attn_res")


def test_a_target_that_cannot_supply_the_stream_fails_at_startup() -> None:
    """Silently serving 'prefix' to a draft trained on 'attn_res' is the bug."""
    holder = _drafter(make_config(**TORCHSPEC_OVERRIDES))
    with pytest.raises(ValueError, match="set_dflash_aux_hidden_stream"):
        holder._wire_aux_hidden_stream(SimpleNamespace())


def test_a_prefix_draft_still_works_on_a_target_without_the_hook() -> None:
    holder = _drafter(make_config())
    holder._wire_aux_hidden_stream(SimpleNamespace())


def test_incremental_projection_stands_down_for_an_fc_norm_draft() -> None:
    """That path re-projects from pre-split ``fc`` columns and would skip fc_norm."""
    holder = SimpleNamespace(
        _fused_kv_enabled=True,
        _kv_aux_stream=object(),
        target_layer_ids=[7, 31, 47, 63, 87],
        draft_model_runner=SimpleNamespace(
            model=SimpleNamespace(fc_norm=[object()], fc=object())
        ),
    )
    DFlash._init_incremental_proj.__get__(holder)()
    assert holder._incremental_proj_enabled is False


# --------------------------------------------------------------------------
# The target side of the switch
# --------------------------------------------------------------------------


def _causal_lm(num_layers: int = 93, attn_res_block_size: int | None = 12):
    from tokenspeed.runtime.models.kimi_k3 import KimiLinearForCausalLM

    holder = SimpleNamespace(
        config=SimpleNamespace(attn_res_block_size=attn_res_block_size),
        model=SimpleNamespace(
            layers=[object()] * num_layers,
            layers_to_capture=[7, 31, 47, 63, 87],
            dflash_aux_stream="prefix",
        ),
    )
    holder.set_dflash_aux_hidden_stream = (
        KimiLinearForCausalLM.set_dflash_aux_hidden_stream.__get__(holder)
    )
    return holder


def test_selecting_attn_res_records_it_on_the_model() -> None:
    holder = _causal_lm()
    holder.set_dflash_aux_hidden_stream("attn_res")
    assert holder.model.dflash_aux_stream == "attn_res"


def test_an_unknown_stream_name_is_rejected() -> None:
    with pytest.raises(ValueError, match="prefix"):
        _causal_lm().set_dflash_aux_hidden_stream("mixture")


def test_attn_res_needs_a_target_that_has_attn_res() -> None:
    holder = _causal_lm(attn_res_block_size=None)
    with pytest.raises(ValueError, match="AttnRes"):
        holder.set_dflash_aux_hidden_stream("attn_res")


# --------------------------------------------------------------------------
# The capture itself
# --------------------------------------------------------------------------


def _k3_model(num_layers: int = 93, stream: str = "attn_res"):
    from tokenspeed.runtime.models.kimi_k3 import KimiLinearModel

    layers = [
        SimpleNamespace(
            self_attention_res_proj=f"proj{i}",
            self_attention_res_norm=f"norm{i}",
            prev_valid_blocks=i // 12,
        )
        for i in range(num_layers)
    ]
    holder = SimpleNamespace(
        layers=layers,
        dflash_aux_stream=stream,
        output_attn_res_proj="out_proj",
        output_attn_res_norm="out_norm",
        config=SimpleNamespace(
            num_hidden_layers=num_layers, attn_res_block_size=12
        ),
    )
    holder._dspark_capture_stream = KimiLinearModel._dspark_capture_stream.__get__(
        holder
    )
    return holder


def test_prefix_mode_hands_back_a_copy_not_the_live_stream() -> None:
    """Later layers write the same storage in place."""
    holder = _k3_model(stream="prefix")
    prefix_sum = torch.ones(2, 4)
    out = holder._dspark_capture_stream(7, prefix_sum, torch.zeros(8, 2, 4))
    assert out is not prefix_sum
    torch.testing.assert_close(out, prefix_sum)


def test_attn_res_mode_mixes_with_the_weights_the_consumer_will_use() -> None:
    """Tap after layer 7 -> layer 8 reads it, so layer 8's mix defines it."""
    from tokenspeed.runtime.models import kimi_k3 as kimi_k3_module

    holder = _k3_model()
    prefix_sum = torch.ones(2, 4)
    block_residual = torch.zeros(8, 2, 4)
    sentinel = torch.full((2, 4), 3.0)

    with mock.patch.object(
        kimi_k3_module, "_apply_attn_res", return_value=sentinel
    ) as mixer:
        out = holder._dspark_capture_stream(7, prefix_sum, block_residual)

    assert out is sentinel
    mixer.assert_called_once_with(prefix_sum, block_residual, "proj8", "norm8", 0)


def test_a_tap_on_the_last_layer_uses_the_model_output_mix() -> None:
    from tokenspeed.runtime.models import kimi_k3 as kimi_k3_module

    holder = _k3_model()
    prefix_sum = torch.ones(2, 4)
    with mock.patch.object(
        kimi_k3_module, "_apply_attn_res", return_value=torch.zeros(2, 4)
    ) as mixer:
        holder._dspark_capture_stream(92, prefix_sum, torch.zeros(8, 2, 4))

    _, _, proj, norm, num_blocks = mixer.call_args[0]
    assert (proj, norm) == ("out_proj", "out_norm")
    # ceil_div(93, 12)
    assert num_blocks == 8


def test_an_unmixed_tap_is_still_copied() -> None:
    """With no committed blocks the mixer returns its input; the caller keeps it."""
    from tokenspeed.runtime.models import kimi_k3 as kimi_k3_module

    holder = _k3_model()
    prefix_sum = torch.ones(2, 4)
    with mock.patch.object(
        kimi_k3_module, "_apply_attn_res", side_effect=lambda p, *a: p
    ):
        out = holder._dspark_capture_stream(0, prefix_sum, torch.zeros(8, 2, 4))
    assert out is not prefix_sum
    torch.testing.assert_close(out, prefix_sum)


# --------------------------------------------------------------------------
# fc_norm: per-tap normalization before the projection
# --------------------------------------------------------------------------


class _Norm(torch.nn.Module):
    def __init__(self, factor: float) -> None:
        super().__init__()
        self.weight = torch.nn.Parameter(torch.full((3,), factor))
        self.variance_epsilon = 1e-6


def _draft(fc_norm, width: int = 3, taps: int = 4):
    from tokenspeed.runtime.models.kimi_k3_dspark import K3DSparkModel

    holder = SimpleNamespace(
        fc_norm=fc_norm,
        _fc_norm_weight=None,
        config=SimpleNamespace(target_hidden_size=width),
        context_proj=lambda hidden_states: (hidden_states, None),
        context_norm=lambda hidden_states: hidden_states,
    )
    holder.project_target_hidden = K3DSparkModel.project_target_hidden.__get__(holder)
    return holder


def test_without_fc_norm_the_taps_reach_the_projection_untouched() -> None:
    concat = torch.arange(12, dtype=torch.float32).reshape(1, 12)
    torch.testing.assert_close(_draft(None).project_target_hidden(concat), concat)


def test_each_tap_is_normalized_on_its_own_before_the_projection() -> None:
    """Per-chunk, in ascending target-layer order -- the order fc_norm trained in."""
    from tokenspeed.runtime.models import kimi_k3_dspark as dspark_module

    fc_norm = torch.nn.ModuleList([_Norm(f) for f in (1.0, 10.0, 100.0, 1000.0)])
    concat = torch.arange(1, 25, dtype=torch.float32).reshape(2, 12)

    def reference(x, weight, eps):
        x_float = x.float()
        return (
            x_float
            * torch.rsqrt(x_float.pow(2).mean(dim=-1, keepdim=True) + eps)
            * weight
        ).to(x.dtype)

    with mock.patch.object(
        dspark_module, "segmented_rmsnorm", side_effect=reference
    ) as fused:
        out = _draft(fc_norm).project_target_hidden(concat)

    expected = reference(
        concat.unflatten(-1, (4, 3)),
        torch.stack([norm.weight for norm in fc_norm]),
        1e-6,
    ).flatten(-2)
    torch.testing.assert_close(out, expected)
    fused.assert_called_once()
    assert out.shape == concat.shape


def test_a_tap_count_that_disagrees_with_fc_norm_is_an_error() -> None:
    """strict=True: a 5-norm draft handed 4 taps must not silently normalize 4."""
    fc_norm = torch.nn.ModuleList([_Norm(1.0) for _ in range(5)])
    with pytest.raises(ValueError, match="fc_norm expects 5 target taps"):
        _draft(fc_norm).project_target_hidden(torch.ones(2, 12))
