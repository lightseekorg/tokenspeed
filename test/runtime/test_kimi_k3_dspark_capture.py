"""Kimi-K3 target hidden-state capture for the DSpark draft.

K3's residual is not a plain pre-norm sum: ``prefix_sum`` is one candidate in a
learned softmax over block-residual snapshots (AttnRes), so the value a layer's
successor actually reads is the *mixture*, not ``prefix_sum``. These tests pin
that the capture reproduces the consumer's mixture exactly, and that the tap
ordering -- which is positional in the draft's ``context_proj`` concat -- is
stable.
"""

from __future__ import annotations

import pytest
import torch

from tokenspeed.runtime.models.kimi_k3 import KimiLinearModel, _apply_attn_res

# The AttnRes kernel is bf16-only, which is also the dtype K3's stream runs in.
DTYPE = torch.bfloat16


class _Norm:
    def __init__(self, hidden: int, eps: float = 1e-5) -> None:
        self.weight = torch.ones(hidden, dtype=DTYPE)
        self.variance_epsilon = eps


class _Proj:
    def __init__(self, hidden: int, seed: int) -> None:
        generator = torch.Generator().manual_seed(seed)
        self.weight = torch.randn(1, hidden, generator=generator).to(DTYPE)


class _Layer:
    """Only the AttnRes attributes the capture reads."""

    def __init__(self, hidden: int, prev_valid_blocks: int, seed: int) -> None:
        self.self_attention_res_proj = _Proj(hidden, seed)
        self.self_attention_res_norm = _Norm(hidden)
        self.prev_valid_blocks = prev_valid_blocks


class _Config:
    def __init__(self, num_hidden_layers: int, attn_res_block_size: int) -> None:
        self.num_hidden_layers = num_hidden_layers
        self.attn_res_block_size = attn_res_block_size


def _make_model(num_layers: int = 8, hidden: int = 16, block: int = 3):
    """A bare KimiLinearModel shell carrying only what the capture touches."""
    model = KimiLinearModel.__new__(KimiLinearModel)
    model.config = _Config(num_layers, block)
    model.layers = [
        _Layer(hidden, prev_valid_blocks=(i + block - 1) // block, seed=100 + i)
        for i in range(num_layers)
    ]
    model.output_attn_res_proj = _Proj(hidden, seed=999)
    model.output_attn_res_norm = _Norm(hidden)
    model.layers_to_capture = []
    model._dflash_capture_idx_map = {}
    model._dflash_incremental_callback = None
    model._dflash_slot_bufs = None
    return model


def _stream(hidden: int = 16, tokens: int = 4, blocks: int = 3, seed: int = 7):
    generator = torch.Generator().manual_seed(seed)
    prefix_sum = torch.randn(tokens, hidden, generator=generator).to(DTYPE)
    block_residual = torch.randn(blocks, tokens, hidden, generator=generator).to(DTYPE)
    return prefix_sum, block_residual


def test_capture_matches_what_the_next_layer_consumes() -> None:
    """The captured tensor equals the successor's own AttnRes mixture."""
    model = _make_model()
    prefix_sum, block_residual = _stream()
    layer_idx = 3
    consumer = model.layers[layer_idx + 1]

    captured = model._dspark_capture_stream(layer_idx, prefix_sum, block_residual)
    expected = _apply_attn_res(
        prefix_sum,
        block_residual,
        consumer.self_attention_res_proj,
        consumer.self_attention_res_norm,
        consumer.prev_valid_blocks,
    )
    torch.testing.assert_close(captured, expected)


def test_capture_is_not_the_bare_prefix_sum() -> None:
    """Guards the whole point: prefix_sum alone is a tensor no layer reads."""
    model = _make_model()
    prefix_sum, block_residual = _stream()
    captured = model._dspark_capture_stream(3, prefix_sum, block_residual)
    assert not torch.allclose(captured, prefix_sum)


def test_last_layer_capture_uses_the_model_output_mixing() -> None:
    model = _make_model(num_layers=8, block=3)
    prefix_sum, block_residual = _stream()
    last = len(model.layers) - 1

    captured = model._dspark_capture_stream(last, prefix_sum, block_residual)
    expected = _apply_attn_res(
        prefix_sum,
        block_residual,
        model.output_attn_res_proj,
        model.output_attn_res_norm,
        # ceil_div(num_layers, block) -- every block is valid at the output.
        3,
    )
    torch.testing.assert_close(captured, expected)


def test_capture_excludes_the_consumers_own_norm() -> None:
    """The draft's context_proj wants the pre-norm mixture, as vLLM/SGLang do."""
    model = _make_model()
    prefix_sum, block_residual = _stream()
    consumer = model.layers[4]

    captured = model._dspark_capture_stream(3, prefix_sum, block_residual)
    normed = _apply_attn_res(
        prefix_sum,
        block_residual,
        consumer.self_attention_res_proj,
        consumer.self_attention_res_norm,
        consumer.prev_valid_blocks,
        out_norm=consumer.self_attention_res_norm,
    )
    assert not torch.allclose(captured, normed)


def test_capture_survives_later_in_place_writes() -> None:
    """A capture aliasing prefix_sum would be silently rewritten by later layers."""
    model = _make_model(num_layers=8, block=3)
    prefix_sum, block_residual = _stream()
    # prev_valid_blocks == 0 for layer 0's consumer is the aliasing case.
    model.layers[1].prev_valid_blocks = 0

    captured = model._dspark_capture_stream(0, prefix_sum, block_residual)
    before = captured.clone()
    prefix_sum.add_(1.0)
    torch.testing.assert_close(captured, before)


# --------------------------------------------------------------------------
# Tap selection contract
# --------------------------------------------------------------------------


class _CausalLM:
    """Minimal stand-in exposing set_dflash_layers_to_capture."""

    set_dflash_layers_to_capture = None  # replaced below

    def __init__(self, model) -> None:
        self.model = model
        self.capture_aux_hidden_states = False


def _bind_setter():
    from tokenspeed.runtime.models.kimi_k3 import KimiLinearForCausalLM

    _CausalLM.set_dflash_layers_to_capture = (
        KimiLinearForCausalLM.set_dflash_layers_to_capture
    )


def test_taps_are_stored_ascending_for_positional_concat() -> None:
    _bind_setter()
    holder = _CausalLM(_make_model(num_layers=93, block=12))
    holder.set_dflash_layers_to_capture([89, 2, 71, 23, 47])
    assert holder.model.layers_to_capture == [2, 23, 47, 71, 89]
    assert holder.model._dflash_capture_idx_map == {2: 0, 23: 1, 47: 2, 71: 3, 89: 4}
    assert holder.capture_aux_hidden_states is True


def test_the_published_k3_taps_are_all_capturable() -> None:
    _bind_setter()
    holder = _CausalLM(_make_model(num_layers=93, block=12))
    holder.set_dflash_layers_to_capture([2, 23, 47, 71, 89])
    assert holder.model.layers_to_capture == [2, 23, 47, 71, 89]


def test_duplicate_taps_are_rejected() -> None:
    _bind_setter()
    holder = _CausalLM(_make_model(num_layers=93, block=12))
    with pytest.raises(ValueError, match="unique"):
        holder.set_dflash_layers_to_capture([2, 2, 47, 71, 89])


def test_the_final_layer_cannot_be_a_tap() -> None:
    """Layer 92's output has no successor to define its stream mixture."""
    _bind_setter()
    holder = _CausalLM(_make_model(num_layers=93, block=12))
    with pytest.raises(ValueError, match="invalid ids"):
        holder.set_dflash_layers_to_capture([2, 23, 47, 71, 92])


def test_negative_taps_are_rejected() -> None:
    _bind_setter()
    holder = _CausalLM(_make_model(num_layers=93, block=12))
    with pytest.raises(ValueError, match="invalid ids"):
        holder.set_dflash_layers_to_capture([-1, 23])
