"""Kimi-K3 target hidden-state capture for the DSpark draft."""

from __future__ import annotations

import pytest
import torch


def _make_model(num_layers: int = 8):
    model = type("Model", (), {})()
    model.layers = [object() for _ in range(num_layers)]
    model.layers_to_capture = []
    model._dflash_capture_idx_map = {}
    # The setter re-marks the AttnRes consumers; the stub has no layers to mark.
    model._refresh_dflash_capture_fallback = lambda: None
    return model


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
    holder = _CausalLM(_make_model(num_layers=93))
    holder.set_dflash_layers_to_capture([89, 2, 71, 23, 47])
    assert holder.model.layers_to_capture == [2, 23, 47, 71, 89]
    assert holder.model._dflash_capture_idx_map == {2: 0, 23: 1, 47: 2, 71: 3, 89: 4}
    assert holder.capture_aux_hidden_states is True


def test_duplicate_taps_are_rejected() -> None:
    _bind_setter()
    holder = _CausalLM(_make_model(num_layers=93))
    with pytest.raises(ValueError, match="unique"):
        holder.set_dflash_layers_to_capture([2, 2, 47, 71, 89])


def test_the_final_layer_can_be_a_completed_layer_tap() -> None:
    _bind_setter()
    holder = _CausalLM(_make_model(num_layers=93))
    holder.set_dflash_layers_to_capture([2, 23, 47, 71, 92])
    assert holder.model.layers_to_capture[-1] == 92


def test_negative_taps_are_rejected() -> None:
    _bind_setter()
    holder = _CausalLM(_make_model(num_layers=93))
    with pytest.raises(ValueError, match="invalid ids"):
        holder.set_dflash_layers_to_capture([-1, 23])


# --------------------------------------------------------------------------
# Latent KV injection
# --------------------------------------------------------------------------


@pytest.mark.skipif(
    not torch.cuda.is_available(), reason="the rope kernel is device-only"
)
def test_latent_rope_rotates_the_tail_without_self_assignment() -> None:
    """The rope tail slices to a view of the latent, so it must be copied out.

    Rotating in place and writing back onto the same storage is a
    self-assignment torch rejects outright -- which is exactly how this failed
    on the first real launch.
    """
    from tokenspeed.runtime.layers.rotary_embedding import get_rope
    from tokenspeed.runtime.models.kimi_k3_dspark import K3DSparkAttention

    kv_lora_rank, qk_rope_head_dim, tokens = 8, 4, 3
    attn = K3DSparkAttention.__new__(K3DSparkAttention)
    torch.nn.Module.__init__(attn)
    attn.kv_lora_rank = kv_lora_rank
    attn.qk_rope_head_dim = qk_rope_head_dim
    attn.rotary_emb = get_rope(
        qk_rope_head_dim,
        rotary_dim=qk_rope_head_dim,
        max_position=64,
        base=10000,
        rope_scaling=None,
        is_neox_style=False,
    )

    device = torch.device("cuda")
    attn.rotary_emb = attn.rotary_emb.to(device)
    latent = (
        torch.arange(tokens * (kv_lora_rank + qk_rope_head_dim), dtype=torch.float32)
        .view(tokens, kv_lora_rank + qk_rope_head_dim)
        .to(device=device, dtype=torch.bfloat16)
    )
    nope_before = latent[:, :kv_lora_rank].clone()
    rope_before = latent[:, kv_lora_rank:].clone()

    out = attn.apply_latent_rope(torch.arange(tokens, device=device), latent)

    assert out.shape == (tokens, kv_lora_rank + qk_rope_head_dim)
    # The compressed KV half is untouched; only the positional tail rotates.
    torch.testing.assert_close(out[:, :kv_lora_rank], nope_before)
    assert not torch.allclose(out[:, kv_lora_rank:], rope_before)


def test_latent_rope_is_a_noop_on_an_empty_batch() -> None:
    from tokenspeed.runtime.models.kimi_k3_dspark import K3DSparkAttention

    attn = K3DSparkAttention.__new__(K3DSparkAttention)
    torch.nn.Module.__init__(attn)
    attn.kv_lora_rank = 8
    attn.qk_rope_head_dim = 4
    attn.rotary_emb = None  # must not be reached
    empty = torch.zeros((0, 12))
    assert attn.apply_latent_rope(torch.zeros(0), empty).shape == (0, 12)


@pytest.mark.parametrize("windows", [((0, 4),), ((0, 2), (2, 4))])
def test_attnres_capture_runs_at_owner_entry_across_pipeline_boundaries(
    monkeypatch, windows
):
    from types import SimpleNamespace

    from tokenspeed.runtime.execution.dspark_context import DSparkContextProducer
    from tokenspeed.runtime.models import kimi_k3

    events = []
    writes = []

    class Layer:
        def __init__(self, index):
            self.index = index

        def capture_attnres(self, prefix, residual):
            events.append(("capture", self.index - 1))
            return prefix + 100 * (self.index + 1)

        def __call__(self, positions, prefix, ctx, residual):
            events.append(("layer", self.index))
            residual[self.index // 2].fill_(self.index)
            return prefix + self.index + 1, residual

    def mix(prefix, residual, proj, norm, num_blocks, **kwargs):
        if "out_norm" not in kwargs:
            events.append(("capture", 3))
        return prefix + 1000

    monkeypatch.setattr(kimi_k3, "_apply_attn_res", mix)
    inbound = None
    for stage, (start, end) in enumerate(windows):
        mapping = SimpleNamespace(
            is_first_pp_rank=stage == 0, is_last_pp_rank=stage == len(windows) - 1
        )
        projector = SimpleNamespace(
            hidden_size=3,
            mapping=mapping,
            project_target_tap=lambda index, rows: rows,
            finalize_target_projection=lambda rows: rows,
            write_context_kv=lambda rows, positions, locations, pool: writes.append(
                (rows.clone(), locations)
            ),
        )
        producer = DSparkContextProducer(
            projector, object() if mapping.is_last_pp_rank else None
        )
        model = SimpleNamespace(
            config=SimpleNamespace(num_hidden_layers=4, attn_res_block_size=2),
            mapping=mapping,
            pp_start_layer=start,
            pp_end_layer=end,
            layers=[Layer(i) if start <= i < end else None for i in range(4)],
            layers_to_capture=[1, 3],
            _dflash_capture_idx_map={1: 0, 3: 1},
            eagle3_layers_to_capture=(),
            dflash_aux_stream="attn_res",
            output_attn_res_proj=None,
            output_attn_res_norm=None,
            norm=None,
        )
        model._dspark_capture_stream = lambda index, prefix, residual: kimi_k3.KimiLinearModel._dspark_capture_stream(
            model, index, prefix, residual
        )
        locations = torch.tensor([9, 10])
        ctx = SimpleNamespace(
            dspark_context_producer=producer,
            target_capture_sink=None,
            num_extends=1,
            bs=1,
            input_num_tokens=2,
            attn_backend=SimpleNamespace(
                decode_window_locations=lambda: torch.empty(0, dtype=torch.int64),
                extend_span_locations=lambda: locations,
            ),
        )
        output, aux = kimi_k3.KimiLinearModel.forward(
            model,
            input_ids=torch.tensor([1, 2]),
            positions=torch.tensor([7, 8]),
            ctx=ctx,
            input_embeds=torch.ones(2, 3) if stage == 0 else None,
            pp_inbound=inbound,
        )
        assert aux is None
        assert len(writes) == int(mapping.is_last_pp_rank)
        if not mapping.is_last_pp_rank:
            inbound = output
    assert events == [
        ("layer", 0),
        ("layer", 1),
        ("capture", 1),
        ("layer", 2),
        ("layer", 3),
        ("capture", 3),
    ]
    assert len(writes) == 1
    torch.testing.assert_close(writes[0][0], torch.full((2, 3), 1315.0))
    torch.testing.assert_close(writes[0][1], torch.tensor([9, 10]))
