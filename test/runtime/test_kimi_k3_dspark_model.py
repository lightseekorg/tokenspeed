"""Config validation and weight-contract coverage for the K3 DSpark draft.

The weight manifest below is the real one from `Inferact/Kimi-K3-DSpark
<https://huggingface.co/Inferact/Kimi-K3-DSpark>`_ (68 tensors, 3.56B
parameters). Most tests check what needs no GPU: the config contract and the
exact set of checkpoint keys the loader must route, skip, or reject; the
context-write tests build a small draft on the GPU.
"""

from __future__ import annotations

import copy
import os
import sys
from types import SimpleNamespace
from unittest import mock

import pytest
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from ci_system.ci_register import register_cuda_ci

register_cuda_ci(est_time=30, suite="runtime-1gpu")

from tokenspeed.runtime.configs.kimi_k3_dspark_config import (
    K3_DSPARK_SKIPPED_WEIGHT_PREFIXES,
    KimiK3DSparkConfig,
    k3_dspark_inactive_features,
    validate_k3_dspark_config,
)
from tokenspeed.runtime.execution.forward_batch_info import ForwardMode
from tokenspeed.runtime.layers.attention.configs.base import block_draft_is_causal
from tokenspeed.runtime.layers.rotary_embedding import yarn_get_mscale
from tokenspeed.runtime.models import kimi_k3_dspark as dspark_model_module
from tokenspeed.runtime.models.kimi_k3_dspark import K3DSparkModel

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


# The published Inferact/Kimi-K3.1-Pro-DSpark-preview config.json.
INFERACT_K31_CONFIG = {
    "attention_bias": False,
    "attention_dropout": 0.0,
    "bos_token_id": 163584,
    "eos_token_id": 163586,
    "head_dim": 64,
    "hidden_act": "silu",
    "hidden_size": 7168,
    "initializer_range": 0.02,
    "intermediate_size": 14336,
    "kv_lora_rank": 512,
    "layer_types": [
        "full_attention",
        "full_attention",
        "full_attention",
        "full_attention",
        "full_attention",
    ],
    "num_attention_heads": 96,
    "num_hidden_layers": 5,
    "num_key_value_heads": 96,
    "pad_token_id": 163839,
    "q_lora_rank": 1536,
    "qk_head_dim": 192,
    "qk_nope_head_dim": 128,
    "qk_rope_head_dim": 64,
    "rms_norm_eps": 1e-05,
    "rope_interleave": True,
    "rope_parameters": {
        "rope_type": "yarn",
        "rope_theta": 1000000.0,
        "factor": 32.0,
        "original_max_position_embeddings": 32768,
        "beta_fast": 32,
        "beta_slow": 1,
        "mscale": 1.0,
        "mscale_all_dim": 0.0,
        "attn_factor": 0.7426255848312643,
    },
    "sliding_window": None,
    "tie_word_embeddings": False,
    "use_cache": True,
    "use_sliding_window": False,
    "v_head_dim": 128,
    "vocab_size": 163840,
    "architectures": ["K3DSparkModel"],
    "model_type": "k3_dspark",
    "max_position_embeddings": 1048576,
    "rope_theta": 1000000.0,
    "num_target_layers": 5,
    "target_hidden_size": 7168,
    "target_num_hidden_layers": 93,
    "target_layer_ids": [75, 79, 83, 87, 91],
    "mask_token_id": 163839,
    "markov_rank": 512,
    "markov_head_type": "vanilla",
    "enable_confidence_head": True,
    "confidence_head_with_markov": True,
    "sample_from_anchor": True,
    "block_size": 7,
    "sliding_window_non_causal": False,
    "full_attention_causal": True,
    "dflash_config": {"causal": True},
    "draft_vocab_size": 163840,
    "torch_dtype": "bfloat16",
}


def test_k31_published_config_validates() -> None:
    config = KimiK3DSparkConfig(**INFERACT_K31_CONFIG)
    validate_k3_dspark_config(config)
    assert config.qk_head_dim == 192
    assert config.kv_latent_dim == 576


def test_k31_block_attends_causally_and_k3_block_does_not() -> None:
    assert block_draft_is_causal(KimiK3DSparkConfig(**INFERACT_K31_CONFIG))
    assert not block_draft_is_causal(make_config())


def test_block_causality_is_declared_by_is_causal_then_dflash_config() -> None:
    config = SimpleNamespace(is_causal=False, dflash_config={"causal": True})
    assert not block_draft_is_causal(config)
    assert block_draft_is_causal(SimpleNamespace(dflash_config={"causal": True}))
    # Sliding layers bound the window, not the block's causality (DFlash2).
    sliding = SimpleNamespace(layer_types=["sliding_attention"] * 2)
    assert not block_draft_is_causal(sliding)


def test_k31_yarn_attn_factor_keeps_unit_rope_magnitude() -> None:
    """K3.1 pairs mscale_all_dim 0 with attn_factor 0.743, so the cos/sin cache
    is unscaled; dropping attn_factor would scale it by yarn_get_mscale(32)."""
    scaling = KimiK3DSparkConfig(**INFERACT_K31_CONFIG).rope_scaling_dict()
    assert scaling["attn_factor"] == pytest.approx(0.7426255848312643)
    magnitude = (
        yarn_get_mscale(scaling["factor"], scaling["mscale"])
        / yarn_get_mscale(scaling["factor"], scaling["mscale_all_dim"])
        * scaling["attn_factor"]
    )
    assert magnitude == pytest.approx(1.0, abs=1e-6)


def test_k31_qk_head_dim_must_be_the_sum_of_its_parts() -> None:
    with pytest.raises(ValueError, match="qk_head_dim=191"):
        KimiK3DSparkConfig(**{**INFERACT_K31_CONFIG, "qk_head_dim": 191})


def test_a_draft_rotating_half_split_rope_is_rejected() -> None:
    with pytest.raises(ValueError, match="rope_interleave"):
        KimiK3DSparkConfig(**{**INFERACT_K31_CONFIG, "rope_interleave": False})


def test_a_draft_not_sampling_from_its_anchor_is_rejected() -> None:
    with pytest.raises(ValueError, match="sample_from_anchor"):
        KimiK3DSparkConfig(**{**INFERACT_K31_CONFIG, "sample_from_anchor": False})


# --------------------------------------------------------------------------
# Validation rejects the corruptions that would otherwise be silent
# --------------------------------------------------------------------------


def test_tap_count_must_match_num_target_layers() -> None:
    with pytest.raises(
        ValueError, match="target_layer_ids has 3 entries but num_target_layers=5"
    ):
        validate_k3_dspark_config(
            make_config(target_layer_ids=[2, 23, 47]), target_config=None
        )


def test_taps_must_be_ascending() -> None:
    """Concat order is positional, so a permuted tap list is silent corruption."""
    with pytest.raises(ValueError, match="ascending"):
        validate_k3_dspark_config(make_config(target_layer_ids=[2, 47, 23, 71, 89]))


def test_nope_attention_is_rejected() -> None:
    """K3's target is NoPE MLA; this draft is not, and running it NoPE is wrong."""
    with pytest.raises(ValueError, match="RoPE MLA"):
        validate_k3_dspark_config(make_config(mla_use_nope=True))


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


def test_vocab_mismatch_is_rejected() -> None:
    """The draft samples through the target's lm_head; vocabularies must agree."""
    with pytest.raises(ValueError, match="vocab_size"):
        validate_k3_dspark_config(
            make_config(), target_config=_Target(vocab_size=152064)
        )


# --------------------------------------------------------------------------
# Weight contract
# --------------------------------------------------------------------------


def test_skipped_prefixes_cover_exactly_the_shared_and_training_only_weights() -> None:
    skipped = [
        k for k in CHECKPOINT_KEYS if k.startswith(K3_DSPARK_SKIPPED_WEIGHT_PREFIXES)
    ]
    assert sorted(skipped) == [
        "confidence_head.proj.bias",
        "confidence_head.proj.weight",
        "embed_tokens.weight",
    ]
    # No lm_head ships at all; the draft borrows the target's.
    assert not any(k.startswith("lm_head") for k in CHECKPOINT_KEYS)


def test_confidence_head_is_reported_inactive_rather_than_dropped() -> None:
    """Issue #879: unsupported optional scheduling must be stated, not silent."""
    notes = k3_dspark_inactive_features(make_config())
    assert len(notes) == 1
    note = notes[0]
    assert "confidence_head" in note
    # It names both what is ignored and what runs instead.
    assert "static" in note


def test_no_inactive_features_reported_without_a_confidence_head() -> None:
    assert k3_dspark_inactive_features(make_config(enable_confidence_head=False)) == []


def test_idle_forward_with_no_rows_skips_dense_draft_layers() -> None:
    model = K3DSparkModel.__new__(K3DSparkModel)
    torch.nn.Module.__init__(model)
    model.config = SimpleNamespace(hidden_size=4)
    model.context_norm = SimpleNamespace(weight=torch.empty(4))
    model.layers = [mock.Mock()]
    model.final_norm = mock.Mock()
    ctx = SimpleNamespace(forward_mode=ForwardMode.IDLE)
    empty = torch.empty((0,), dtype=torch.int32)

    output = model(ctx, input_ids=empty, positions=empty, out_cache_loc=empty)

    assert output.next_token_logits is None
    assert output.hidden_states.shape == (0, 4)
    model.layers[0].assert_not_called()
    model.final_norm.assert_not_called()


def test_final_norm_reduces_the_last_row_parallel_mlp_output() -> None:
    model = K3DSparkModel.__new__(K3DSparkModel)
    torch.nn.Module.__init__(model)
    tp_group = object()
    model.mapping = SimpleNamespace(dense=SimpleNamespace(tp_group=tp_group))

    class _FinalNorm(torch.nn.Module):
        def forward(self, hidden_states, residual):
            return hidden_states + residual, residual

    model.final_norm = _FinalNorm()
    local_hidden = torch.tensor([[1.0, 2.0]])
    residual = torch.tensor([[10.0, 20.0]])

    with mock.patch.object(
        dspark_model_module,
        "all_reduce",
        side_effect=lambda hidden, group: hidden * 8,
    ) as reduce:
        out = model._finalize_hidden(local_hidden, residual)

    reduce.assert_called_once_with(local_hidden, tp_group)
    torch.testing.assert_close(out, local_hidden * 8 + residual)


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


@pytest.mark.parametrize(
    "stream,layer,expected",
    [
        ("prefix", 2, 2),
        ("prefix", 7, 7),
        ("attn_res", 2, 3),
        ("attn_res", 7, 7),
    ],
)
def test_k3_tap_ownership_matches_the_stream_producer(stream, layer, expected):
    assert dspark_model_module._context_tap_owner_layer(layer, 8, stream) == expected


@pytest.mark.parametrize("capture_idx", [0, 2])
@pytest.mark.parametrize("with_norm", [False, True])
def test_k3_per_tap_projection_uses_local_checkpoint_columns(capture_idx, with_norm):
    model = K3DSparkModel.__new__(K3DSparkModel)
    torch.nn.Module.__init__(model)
    model.config = SimpleNamespace(target_hidden_size=3)
    model.local_tap_indices = (0, 2)
    model.context_proj = torch.nn.Linear(6, 4, bias=False)
    model.context_proj.weight.data.copy_(torch.arange(24).reshape(4, 6) / 24)
    model.fc_norm = (
        torch.nn.ModuleList([torch.nn.RMSNorm(3, eps=1e-5) for _ in range(2)])
        if with_norm
        else None
    )
    hidden = torch.tensor([[1.0, 2.0, 4.0], [3.0, -1.0, 2.0]])
    local_idx = (0, 2).index(capture_idx)
    reference = hidden
    if with_norm:
        for index, norm in enumerate(model.fc_norm):
            norm.weight.data.copy_(torch.arange(3) + index + 1)
        reference = torch.nn.functional.rms_norm(
            hidden, (3,), model.fc_norm[local_idx].weight, 1e-5
        )
    expected = torch.nn.functional.linear(
        reference, model.context_proj.weight[:, local_idx * 3 : (local_idx + 1) * 3]
    )
    torch.testing.assert_close(model.project_target_tap(capture_idx, hidden), expected)
    with pytest.raises(ValueError):
        model.project_target_tap(1, hidden)


def _tiny_draft(mapping) -> K3DSparkModel:
    """A small real draft in bf16 on the GPU, with K3's 512 + 64 latent rows."""
    config = make_config(
        hidden_size=256,
        intermediate_size=256,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=4,
        q_lora_rank=128,
        qk_nope_head_dim=64,
        v_head_dim=64,
        vocab_size=1024,
        draft_vocab_size=1024,
        max_position_embeddings=4096,
        rope_theta=10000.0,
        rope_parameters=None,
        num_target_layers=3,
        target_hidden_size=256,
        target_num_hidden_layers=8,
        target_layer_ids=[1, 4, 6],
        mask_token_id=1023,
        markov_rank=16,
    )
    default_dtype = torch.get_default_dtype()
    torch.set_default_dtype(torch.bfloat16)
    try:
        with torch.device("cuda"):
            model = K3DSparkModel(config, mapping)
    finally:
        torch.set_default_dtype(default_dtype)
    # A cached RoPE table may predate the device scope.
    model.cuda()
    generator = torch.Generator(device="cuda").manual_seed(0)
    for param in model.parameters():
        param.data.normal_(0.0, 0.05, generator=generator)
    for layer in model.layers:
        layer.self_attn.kv_a_layernorm.weight.data.add_(1.0)
    model.post_load_weights()
    return model


class _LatentPool:
    """The MLA pool's write target over plain latent planes."""

    def __init__(self, planes: list[torch.Tensor], sanitize: bool) -> None:
        self.planes = planes
        self.sanitize = sanitize

    def kv_write_target(self, layer_id, slots, write_mask):
        from tokenspeed_kernel.ops.attention.prologue import LatentKVCache

        return LatentKVCache(self.planes[layer_id], self.sanitize, slots, write_mask)


def _chain_write(model, rows, positions, slots, planes, placement, sanitize):
    """The per-layer context write the fused one replaced.

    It runs F.linear over the KV rows, kv_a_layernorm, RoPE, then the latent
    store.
    """
    from tokenspeed_kernel.ops.kvcache.triton import set_mla_kv_buffer_triton

    from tokenspeed.runtime.layers.attention.dcp.placement import resolve_cache_slots

    for layer, plane in zip(model.layers, planes, strict=True):
        attn = layer.self_attn
        rank = attn.kv_lora_rank
        weight = attn.fused_qkv_a_proj_with_mqa.weight
        latent = torch.nn.functional.linear(rows, weight[attn.q_lora_rank :])
        latent = torch.cat(
            (attn.kv_a_layernorm(latent[:, :rank].contiguous()), latent[:, rank:]),
            dim=-1,
        )
        latent = attn.apply_latent_rope(positions, latent)
        local, owned = resolve_cache_slots(slots, placement)
        set_mla_kv_buffer_triton(
            plane,
            local,
            latent[:, :rank].contiguous(),
            latent[:, rank:].contiguous(),
            sanitize=sanitize,
            write_mask=owned,
        )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA kernels")
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float8_e4m3fn])
@pytest.mark.parametrize("sanitize", [False, True])
@pytest.mark.parametrize("rows", [1, 8, 40])
def test_the_one_launch_context_write_stores_the_per_layer_chains_bytes(
    dtype, sanitize, rows
):
    from tokenspeed.runtime.distributed.mapping import Mapping

    model = _tiny_draft(Mapping(rank=0, world_size=1))
    generator = torch.Generator(device="cuda").manual_seed(rows)
    hidden = (torch.randn(rows, 256, device="cuda", generator=generator)).bfloat16()
    positions = torch.randint(4096, (rows,), device="cuda", generator=generator)
    slots = torch.randperm(127, device="cuda", generator=generator)[:rows] + 1

    def planes():
        return [
            torch.full((128, 1, 576), 3.0, device="cuda").to(dtype)
            for _ in model.layers
        ]

    expected, actual = planes(), planes()
    _chain_write(model, hidden, positions, slots, expected, None, sanitize)
    backend = SimpleNamespace(cache_placement=lambda layer: None)
    model.write_context_kv(
        hidden, positions, slots, _LatentPool(actual, sanitize), attn_backend=backend
    )
    for got, want in zip(actual, expected, strict=True):
        assert torch.equal(got.view(torch.uint8), want.view(torch.uint8))
        assert not torch.equal(got.float(), torch.full_like(got.float(), 3.0))


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA kernels")
def test_context_kv_write_respects_dcp_ownership():
    """Under KV parallelism each rank stores only the rows it owns.

    A rank writes the rows whose pages it owns, at its local slots; padding
    rows (slot -1) are stored nowhere.
    """
    from tokenspeed.runtime.distributed.mapping import Mapping
    from tokenspeed.runtime.layers.attention.dcp.placement import CachePlacement

    model = _tiny_draft(Mapping(rank=0, world_size=1))
    slots = torch.tensor([0, 128, 129, 256, 384, 512, 896, 1024, -1], device="cuda")
    hidden = torch.randn(slots.numel(), 256, device="cuda").bfloat16()
    positions = torch.arange(slots.numel(), device="cuda")
    owned_rows = 0
    for rank in range(4):
        placement = CachePlacement(128, 8, (0, 1, 2, 3), rank)
        expected = [
            torch.zeros((512, 1, 576), device="cuda").bfloat16() for _ in range(2)
        ]
        actual = [torch.zeros_like(plane) for plane in expected]
        _chain_write(model, hidden, positions, slots, expected, placement, True)
        model.write_context_kv(
            hidden,
            positions,
            slots,
            _LatentPool(actual, True),
            attn_backend=SimpleNamespace(cache_placement=lambda layer: placement),
        )
        for got, want in zip(actual, expected, strict=True):
            assert torch.equal(got.view(torch.int16), want.view(torch.int16))
        owned_rows += int(actual[0].abs().sum(dim=(1, 2)).gt(0).sum())
    # Rows 128..896 are owned by exactly one rank each; 0, 1024 and -1 by none.
    assert owned_rows == 6
