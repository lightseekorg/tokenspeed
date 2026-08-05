"""Config validation and weight-contract coverage for the K3 DSpark draft.

The weight manifest below is the real one from `Inferact/Kimi-K3-DSpark
<https://huggingface.co/Inferact/Kimi-K3-DSpark>`_ (68 tensors, 3.56B
parameters). Building the model needs a distributed mapping and a GPU, so these
tests check the pieces that can be checked without one: the config contract and
the exact set of checkpoint keys the loader must route, skip, or reject.
"""

from __future__ import annotations

import copy
from types import SimpleNamespace

import pytest
import torch

from tokenspeed.runtime.configs.kimi_k3_dspark_config import (
    K3_DSPARK_SKIPPED_WEIGHT_PREFIXES,
    KimiK3DSparkConfig,
    k3_dspark_inactive_features,
    validate_k3_dspark_config,
)
from tokenspeed.runtime.models.kimi_k3_dspark import (
    K3DSparkConfidenceHead,
    K3DSparkModel,
)

# The published Inferact/Kimi-K3-DSpark config.json.
INFERACT_CONFIG = dict(
    hidden_size=7168,
    intermediate_size=14336,
    num_hidden_layers=5,
    num_attention_heads=64,
    num_key_value_heads=64,
    q_lora_rank=1536,
    kv_lora_rank=512,
    qk_nope_head_dim=128,
    qk_rope_head_dim=64,
    v_head_dim=128,
    mla_use_nope=False,
    mla_use_output_gate=False,
    vocab_size=163840,
    draft_vocab_size=163840,
    rms_norm_eps=1e-5,
    max_position_embeddings=1048576,
    rope_theta=50000.0,
    rope_parameters={
        "rope_type": "yarn",
        "factor": 32.0,
        "original_max_position_embeddings": 32768,
        "rope_theta": 50000.0,
        "beta_fast": 32,
        "beta_slow": 1,
        "mscale": 1.0,
        "mscale_all_dim": 1.0,
    },
    num_target_layers=5,
    target_hidden_size=7168,
    target_num_hidden_layers=93,
    target_layer_ids=[2, 23, 47, 71, 89],
    mask_token_id=163837,
    markov_rank=256,
    markov_head_type="vanilla",
    enable_confidence_head=True,
    confidence_head_with_markov=True,
)


def _per_layer_keys(layer: int) -> list[str]:
    return [
        f"layers.{layer}.input_layernorm.weight",
        f"layers.{layer}.post_attention_layernorm.weight",
        f"layers.{layer}.mlp.gate_proj.weight",
        f"layers.{layer}.mlp.up_proj.weight",
        f"layers.{layer}.mlp.down_proj.weight",
        f"layers.{layer}.self_attn.q_a_proj.weight",
        f"layers.{layer}.self_attn.q_a_layernorm.weight",
        f"layers.{layer}.self_attn.q_b_proj.weight",
        f"layers.{layer}.self_attn.kv_a_proj_with_mqa.weight",
        f"layers.{layer}.self_attn.kv_a_layernorm.weight",
        f"layers.{layer}.self_attn.kv_b_proj.weight",
        f"layers.{layer}.self_attn.o_proj.weight",
    ]


CHECKPOINT_KEYS = sorted(
    [
        "confidence_head.proj.bias",
        "confidence_head.proj.weight",
        "context_norm.weight",
        "context_proj.weight",
        "embed_tokens.weight",
        "final_norm.weight",
        "markov_head.markov_w1.weight",
        "markov_head.markov_w2.weight",
    ]
    + [key for layer in range(5) for key in _per_layer_keys(layer)]
)


def make_config(**overrides) -> KimiK3DSparkConfig:
    fields = copy.deepcopy(INFERACT_CONFIG)
    fields.update(overrides)
    return KimiK3DSparkConfig(**fields)


# --------------------------------------------------------------------------
# The published checkpoint must validate as-is
# --------------------------------------------------------------------------


def test_published_checkpoint_config_validates() -> None:
    validate_k3_dspark_config(make_config())


def test_manifest_has_the_published_tensor_count() -> None:
    assert len(CHECKPOINT_KEYS) == 68


def test_model_type_and_latent_geometry() -> None:
    config = make_config()
    assert config.model_type == "k3_dspark"
    # One cached row is [c_KV_norm:512 | k_PE_RoPE:64].
    assert config.kv_latent_dim == 576
    assert config.qk_head_dim == 192


def test_yarn_scaling_is_translated_for_get_rope() -> None:
    config = make_config()
    scaling = config.rope_scaling_dict()
    assert scaling["rope_type"] == "deepseek_yarn"
    assert scaling["factor"] == 32.0
    assert scaling["original_max_position_embeddings"] == 32768
    # rope_parameters.rope_theta is what the draft trained with.
    assert config.resolved_rope_theta() == 50000.0


def test_non_yarn_rope_yields_no_scaling() -> None:
    config = make_config(
        rope_parameters={"rope_type": "default", "rope_theta": 10000.0}
    )
    assert config.rope_scaling_dict() is None
    assert config.resolved_rope_theta() == 10000.0


# --------------------------------------------------------------------------
# Validation rejects the corruptions that would otherwise be silent
# --------------------------------------------------------------------------


def test_tap_count_must_match_num_target_layers() -> None:
    with pytest.raises(ValueError, match="context_proj expects"):
        validate_k3_dspark_config(make_config(target_layer_ids=[2, 23, 47]))


def test_taps_must_be_ascending() -> None:
    """Concat order is positional, so a permuted tap list is silent corruption."""
    with pytest.raises(ValueError, match="ascending"):
        validate_k3_dspark_config(make_config(target_layer_ids=[2, 47, 23, 71, 89]))


def test_taps_must_be_distinct() -> None:
    with pytest.raises(ValueError, match="distinct"):
        validate_k3_dspark_config(make_config(target_layer_ids=[2, 2, 47, 71, 89]))


def test_taps_must_be_in_range_for_the_target() -> None:
    with pytest.raises(ValueError, match="out of range"):
        validate_k3_dspark_config(make_config(target_layer_ids=[2, 23, 47, 71, 93]))


def test_mask_token_is_required() -> None:
    with pytest.raises(ValueError, match="mask_token_id"):
        validate_k3_dspark_config(make_config(mask_token_id=None))


def test_mask_token_must_be_inside_the_vocabulary() -> None:
    with pytest.raises(ValueError, match="outside"):
        validate_k3_dspark_config(make_config(mask_token_id=200000))


def test_markov_head_is_mandatory() -> None:
    with pytest.raises(ValueError, match="markov_rank > 0"):
        validate_k3_dspark_config(make_config(markov_rank=0))


def test_unsupported_markov_head_type_is_rejected() -> None:
    with pytest.raises(ValueError, match="markov_head_type"):
        validate_k3_dspark_config(make_config(markov_head_type="rnn"))


def test_nope_attention_is_rejected() -> None:
    """K3's target is NoPE MLA; this draft is not, and running it NoPE is wrong."""
    with pytest.raises(ValueError, match="RoPE MLA"):
        validate_k3_dspark_config(make_config(mla_use_nope=True))


def test_output_gate_is_rejected() -> None:
    with pytest.raises(ValueError, match="output gate"):
        validate_k3_dspark_config(make_config(mla_use_output_gate=True))


# --------------------------------------------------------------------------
# Cross-checks against the loaded target
# --------------------------------------------------------------------------


class _Target:
    def __init__(self, hidden_size=7168, num_hidden_layers=93, vocab_size=163840):
        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers
        self.vocab_size = vocab_size


def test_matching_target_passes() -> None:
    validate_k3_dspark_config(make_config(), target_config=_Target())


def test_target_hidden_size_mismatch_is_rejected() -> None:
    with pytest.raises(ValueError, match="target_hidden_size"):
        validate_k3_dspark_config(
            make_config(), target_config=_Target(hidden_size=4096)
        )


def test_target_depth_mismatch_is_rejected() -> None:
    with pytest.raises(ValueError, match="target_num_hidden_layers"):
        validate_k3_dspark_config(
            make_config(), target_config=_Target(num_hidden_layers=61)
        )


def test_vocab_mismatch_is_rejected() -> None:
    """The draft samples through the target's lm_head; vocabularies must agree."""
    with pytest.raises(ValueError, match="vocab_size"):
        validate_k3_dspark_config(
            make_config(), target_config=_Target(vocab_size=152064)
        )


# --------------------------------------------------------------------------
# Weight contract
# --------------------------------------------------------------------------


def test_skipped_prefixes_cover_exactly_the_shared_target_weights() -> None:
    skipped = [
        k for k in CHECKPOINT_KEYS if k.startswith(K3_DSPARK_SKIPPED_WEIGHT_PREFIXES)
    ]
    assert sorted(skipped) == ["embed_tokens.weight"]
    # No lm_head ships at all; the draft borrows the target's.
    assert not any(k.startswith("lm_head") for k in CHECKPOINT_KEYS)


def test_confidence_head_is_reported_loaded_but_not_scheduling() -> None:
    notes = k3_dspark_inactive_features(make_config())
    assert len(notes) == 1
    note = notes[0]
    assert "confidence_head" in note
    assert "loaded" in note
    assert "static" in note


def test_no_inactive_features_reported_without_a_confidence_head() -> None:
    assert k3_dspark_inactive_features(make_config(enable_confidence_head=False)) == []


def test_every_remaining_checkpoint_key_has_a_destination() -> None:
    """After skips and stacking, no checkpoint key is left unrouted.

    This mirrors the loader's routing table without constructing the model, so
    a renamed submodule shows up here rather than as a 7 GB load failure.
    """
    stacked = {".gate_proj.": ".gate_up_proj.", ".up_proj.": ".gate_up_proj."}
    fused = {".q_a_proj.", ".kv_a_proj_with_mqa."}
    expected_params = {
        "context_proj.weight",
        "context_norm.weight",
        "final_norm.weight",
        "confidence_head.proj.bias",
        "confidence_head.proj.weight",
        "markov_head.markov_w1.weight",
        "markov_head.markov_w2.weight",
    }
    for layer in range(5):
        expected_params.update(
            {
                f"layers.{layer}.input_layernorm.weight",
                f"layers.{layer}.post_attention_layernorm.weight",
                f"layers.{layer}.mlp.gate_up_proj.weight",
                f"layers.{layer}.mlp.down_proj.weight",
                f"layers.{layer}.self_attn.fused_qkv_a_proj_with_mqa.weight",
                f"layers.{layer}.self_attn.q_a_layernorm.weight",
                f"layers.{layer}.self_attn.q_b_proj.weight",
                f"layers.{layer}.self_attn.kv_a_layernorm.weight",
                f"layers.{layer}.self_attn.kv_b_proj.weight",
                f"layers.{layer}.self_attn.o_proj.weight",
            }
        )

    routed = set()
    for key in CHECKPOINT_KEYS:
        if key.startswith(K3_DSPARK_SKIPPED_WEIGHT_PREFIXES):
            continue
        target = key
        for src, dst in stacked.items():
            if src in key:
                target = key.replace(src, dst)
                break
        else:
            for src in fused:
                if src in key:
                    target = key.replace(src, ".fused_qkv_a_proj_with_mqa.")
                    break
        routed.add(target)

    assert routed == expected_params


def test_confidence_head_consumes_hidden_and_markov_features() -> None:
    head = K3DSparkConfidenceHead(hidden_size=6, markov_rank=2, with_markov=True)
    output = head(torch.randn(3, 6), torch.randn(3, 2))
    assert output.shape == (3,)
    with pytest.raises(ValueError, match="Markov latent"):
        head(torch.randn(3, 6), None)


class _FakePackedParameter:
    def __init__(self) -> None:
        self.loads: list[tuple[tuple, dict]] = []

    def weight_loader(self, param, loaded_weight, *args, **kwargs) -> None:
        assert param is self
        self.loads.append((args, kwargs))


class _FakeLoaderModel:
    def __init__(self, target: str) -> None:
        self.config = SimpleNamespace(q_lora_rank=1536)
        self.param = _FakePackedParameter()
        self.target = target
        self.post_load_called = False

    def named_parameters(self):
        return [(self.target, self.param)]

    def post_load_weights(self) -> None:
        self.post_load_called = True


@pytest.mark.parametrize(
    ("target", "present_source", "missing_source"),
    [
        (
            "layers.0.mlp.gate_up_proj.weight",
            "layers.0.mlp.gate_proj.weight",
            "up_proj",
        ),
        (
            "layers.0.self_attn.fused_qkv_a_proj_with_mqa.weight",
            "layers.0.self_attn.q_a_proj.weight",
            "kv_a_proj_with_mqa",
        ),
    ],
)
def test_packed_parameter_rejects_a_missing_source_shard(
    target: str, present_source: str, missing_source: str
) -> None:
    model = _FakeLoaderModel(target)
    with pytest.raises(ValueError, match=missing_source):
        K3DSparkModel.load_weights(
            model, [(present_source, torch.zeros(1, dtype=torch.bfloat16))]
        )
    assert not model.post_load_called


@pytest.mark.parametrize(
    ("target", "sources"),
    [
        (
            "layers.0.mlp.gate_up_proj.weight",
            ("layers.0.mlp.gate_proj.weight", "layers.0.mlp.up_proj.weight"),
        ),
        (
            "layers.0.self_attn.fused_qkv_a_proj_with_mqa.weight",
            (
                "layers.0.self_attn.q_a_proj.weight",
                "layers.0.self_attn.kv_a_proj_with_mqa.weight",
            ),
        ),
    ],
)
def test_packed_parameter_accepts_all_source_shards(
    target: str, sources: tuple[str, str]
) -> None:
    model = _FakeLoaderModel(target)
    K3DSparkModel.load_weights(
        model, [(source, torch.zeros(1, dtype=torch.bfloat16)) for source in sources]
    )
    assert len(model.param.loads) == 2
    assert model.post_load_called
