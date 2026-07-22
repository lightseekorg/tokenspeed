"""CPU-only tests for the EAGLE3 llama draft model variants.

Covers the torchspec checkpoint layout (e.g. Inferact/MiniMax-M3-EAGLE3):
per-aux-state ``fc_norm`` RMSNorms and the ``norm_output`` aux convention.
"""

from __future__ import annotations

import pytest
import torch
from transformers import LlamaConfig

from tokenspeed.runtime.distributed.mapping import Mapping
from tokenspeed.runtime.layers.layernorm import RMSNorm
from tokenspeed.runtime.models.llama_eagle3 import LlamaForCausalLMEagle3
from tokenspeed.runtime.utils.env import global_server_args_dict

_HIDDEN = 32
_INTERMEDIATE = 64
_VOCAB = 64


def _draft_config(**overrides) -> LlamaConfig:
    values = dict(
        vocab_size=_VOCAB,
        hidden_size=_HIDDEN,
        intermediate_size=_INTERMEDIATE,
        num_hidden_layers=1,
        num_attention_heads=4,
        num_key_value_heads=4,
        max_position_embeddings=256,
        rms_norm_eps=1e-6,
        tie_word_embeddings=False,
        draft_vocab_size=_VOCAB,
    )
    values.update(overrides)
    return LlamaConfig(**values)


def _tp1_mapping() -> Mapping:
    return Mapping(
        rank=0,
        world_size=1,
        attn_tp_size=1,
        attn_cp_size=1,
        attn_dp_size=1,
        dense_tp_size=1,
        dense_dp_size=1,
        moe_tp_size=1,
        moe_ep_size=1,
        moe_dp_size=1,
        nprocs_per_node=1,
        nnodes=1,
    )


def _build_model(
    monkeypatch: pytest.MonkeyPatch, config: LlamaConfig
) -> LlamaForCausalLMEagle3:
    mapping = _tp1_mapping()
    monkeypatch.setitem(global_server_args_dict, "ep_num_redundant_experts", 0)
    monkeypatch.setitem(global_server_args_dict, "max_model_len", 256)
    monkeypatch.setitem(global_server_args_dict, "mapping", mapping)
    monkeypatch.setitem(global_server_args_dict, "comm_fusion_max_num_tokens", 256)

    old_dtype = torch.get_default_dtype()
    torch.set_default_dtype(torch.bfloat16)
    try:
        with torch.device("meta"):
            return LlamaForCausalLMEagle3(config, mapping)
    finally:
        torch.set_default_dtype(old_dtype)


def _torchspec_checkpoint_weights() -> list[tuple[str, torch.Tensor]]:
    """Tensor names/shapes of a torchspec EAGLE3 checkpoint (tiny config)."""

    def meta(*shape: int) -> torch.Tensor:
        return torch.empty(*shape, dtype=torch.bfloat16, device="meta")

    return [
        ("embed_tokens.weight", meta(_VOCAB, _HIDDEN)),
        ("fc.weight", meta(_HIDDEN, 3 * _HIDDEN)),
        ("fc_norm.0.weight", meta(_HIDDEN)),
        ("fc_norm.1.weight", meta(_HIDDEN)),
        ("fc_norm.2.weight", meta(_HIDDEN)),
        ("layers.0.hidden_norm.weight", meta(_HIDDEN)),
        ("layers.0.input_layernorm.weight", meta(_HIDDEN)),
        ("layers.0.mlp.down_proj.weight", meta(_HIDDEN, _INTERMEDIATE)),
        ("layers.0.mlp.gate_proj.weight", meta(_INTERMEDIATE, _HIDDEN)),
        ("layers.0.mlp.up_proj.weight", meta(_INTERMEDIATE, _HIDDEN)),
        ("layers.0.post_attention_layernorm.weight", meta(_HIDDEN)),
        ("layers.0.self_attn.q_proj.weight", meta(_HIDDEN, 2 * _HIDDEN)),
        ("layers.0.self_attn.k_proj.weight", meta(_HIDDEN, 2 * _HIDDEN)),
        ("layers.0.self_attn.v_proj.weight", meta(_HIDDEN, 2 * _HIDDEN)),
        ("layers.0.self_attn.o_proj.weight", meta(_HIDDEN, _HIDDEN)),
        ("lm_head.weight", meta(_VOCAB, _HIDDEN)),
        ("norm.weight", meta(_HIDDEN)),
    ]


def test_eagle3_default_config_has_no_fc_norm(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model = _build_model(monkeypatch, _draft_config())

    assert model.model.fc_norm is None
    assert model.model.input_norm is None
    assert model.model.norm_output is False


def test_eagle3_fc_norm_and_norm_output_variant(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model = _build_model(monkeypatch, _draft_config(fc_norm=True, norm_output=True))

    assert model.model.norm_output is True
    assert model.model.input_norm is None
    fc_norm = model.model.fc_norm
    assert fc_norm is not None and len(fc_norm) == 3
    for norm in fc_norm:
        assert isinstance(norm, RMSNorm)
        assert norm.weight.shape == (_HIDDEN,)
    assert model.model.fc.weight.shape == (_HIDDEN, 3 * _HIDDEN)


def test_eagle3_torchspec_checkpoint_names_all_reach_a_loader(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model = _build_model(monkeypatch, _draft_config(fc_norm=True, norm_output=True))

    loaded: list[str] = []

    def _track(name: str):
        def wrapper(param, weight, *args, **kwargs):
            loaded.append(name)

        return wrapper

    for name, param in model.named_parameters():
        param.weight_loader = _track(name)

    weights = _torchspec_checkpoint_weights()
    model.load_weights(weights)

    assert len(loaded) == len(weights), (
        f"only {len(loaded)}/{len(weights)} checkpoint tensors reached a "
        f"weight loader; loaded params: {sorted(loaded)}"
    )
