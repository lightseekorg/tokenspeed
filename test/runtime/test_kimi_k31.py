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

"""Kimi-K3.1: config parse, routing, cache layout, index-K pool, indexer, load."""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

# ``test/`` (for ``ci_system``) and the repo root (for ``test.runtime.*``).
_TEST_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _TEST_DIR)
sys.path.insert(0, os.path.dirname(_TEST_DIR))

from test.runtime.conftest import kimi_layout, kimi_recipe, kimi_tp8_layout

from ci_system.ci_register import register_cuda_ci

register_cuda_ci(est_time=60, suite="runtime-1gpu")

from tokenspeed.runtime.configs.kimi_k3_config import KimiK3Config, KimiLinearConfig
from tokenspeed.runtime.configs.model_config import (
    AttentionArch,
    ModelConfig,
)
from tokenspeed.runtime.layers.attention.configs.dsa import dsa_index_k_row_bytes
from tokenspeed.runtime.layers.attention.kv_cache.recipes.spec import FULL_ATTENTION
from tokenspeed.runtime.layers.attention.registry import (
    _LINEAR_ATTN_CLS,
    _resolve_attn_side,
    _resolve_cache_family,
    _resolve_full_attn_backend_name,
    _validate_mla_dcp_backend,
)
from tokenspeed.runtime.utils.hf_transformers_utils import get_config
from tokenspeed.runtime.utils.server_args import ServerArgs

requires_gpu = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="requires a GPU"
)

# 1-based MLA layers of the 93-layer checkpoint: every 4th, plus the last.
_FULL_ATTN_LAYERS = list(range(4, 93, 4)) + [93]
_KDA_LAYERS = [layer for layer in range(1, 94) if layer not in set(_FULL_ATTN_LAYERS)]
_DSA_CONFIG = {
    "dsa_layers": _FULL_ATTN_LAYERS,
    "index_topk": 2048,
    "index_head_dim": 128,
    "index_num_heads": 32,
    "index_rope_dim": 0,
    "index_k_norm_type": "layernorm",
    "index_topk_freq": 1,
    "index_skip_topk_offset": 1,
}
# The checkpoint config.json fields these tests read (shapes and layer maps).
_K31_CONFIG_JSON = {
    "architectures": ["KimiK31ForConditionalGeneration"],
    "model_type": "kimi_k31",
    "text_config": {
        "architectures": ["KimiLinearForCausalLM"],
        "model_type": "kimi_linear",
        "num_hidden_layers": 93,
        "hidden_size": 7168,
        "num_attention_heads": 96,
        "num_key_value_heads": 96,
        "q_lora_rank": 1536,
        "kv_lora_rank": 512,
        "qk_nope_head_dim": 128,
        "qk_rope_head_dim": 64,
        "v_head_dim": 128,
        "mla_use_nope": True,
        "mla_use_output_gate": True,
        "linear_attn_config": {
            "full_attn_layers": _FULL_ATTN_LAYERS,
            "kda_layers": _KDA_LAYERS,
            "gate_lower_bound": -5.0,
            "head_dim": 128,
            "num_heads": 96,
            "short_conv_kernel_size": 4,
            "use_full_rank_gate": True,
        },
        "dsa_config": _DSA_CONFIG,
    },
}


def _load_k31_config(tmp_path: Path):
    (tmp_path / "config.json").write_text(json.dumps(_K31_CONFIG_JSON))
    return get_config(str(tmp_path), trust_remote_code=False)


def _load_k31_model_config(tmp_path: Path) -> ModelConfig:
    (tmp_path / "config.json").write_text(json.dumps(_K31_CONFIG_JSON))
    return ModelConfig(
        str(tmp_path),
        trust_remote_code=False,
        model_override_args="{}",
        dtype="bfloat16",
        server_args=ServerArgs(model=str(tmp_path)),
    )


# ---------------------------------------------------------------- config


def test_config_parses_dsa_geometry_and_layer_map(tmp_path) -> None:
    config = _load_k31_config(tmp_path)
    text = config.text_config

    assert isinstance(config, KimiK3Config)
    assert config.architectures == ["KimiK31ForConditionalGeneration"]
    assert (text.index_topk, text.index_head_dim, text.index_n_heads) == (
        2048,
        128,
        32,
    )
    # Every MLA layer carries an indexer, 0-based ids.
    assert text.dsa_layer_ids == text.full_attention_layer_ids
    assert text.dsa_layer_ids[:3] == [3, 7, 11]
    assert text.dsa_layer_ids[-2:] == [91, 92]
    assert len(text.dsa_layer_ids) == 24
    assert text.is_dsa_layer(3) and not text.is_dsa_layer(0)


def test_k3_config_has_no_indexer() -> None:
    text = KimiLinearConfig()

    assert text.dsa_layer_ids == []
    assert (text.index_topk, text.index_head_dim, text.index_n_heads) == (
        None,
        None,
        None,
    )
    assert not any(text.is_dsa_layer(layer) for layer in range(93))


@pytest.mark.parametrize(
    "key, value, error",
    [
        ("index_rope_dim", 64, NotImplementedError),
        ("index_k_norm_type", "rmsnorm", NotImplementedError),
        ("indexer_use_fp8", True, NotImplementedError),
        ("index_topk_freq", 2, NotImplementedError),
        ("dsa_layers", _FULL_ATTN_LAYERS[:-1], ValueError),
        ("index_topk", 0, ValueError),
        ("index_unknown_knob", 1, ValueError),
    ],
)
def test_config_refuses_unimplemented_indexer_semantics(key, value, error) -> None:
    dsa_config = dict(_DSA_CONFIG, **{key: value})
    with pytest.raises(error):
        KimiLinearConfig(dsa_config=dsa_config)


def test_config_accepts_bf16_indexer_weights_declared_or_not() -> None:
    """The released checkpoint omits ``indexer_use_fp8``; an explicit false is the same."""
    for dsa_config in (_DSA_CONFIG, dict(_DSA_CONFIG, indexer_use_fp8=False)):
        assert KimiLinearConfig(dsa_config=dsa_config).dsa_config == dsa_config


def test_model_config_routes_dsa_on_the_kimi_k3_cache_family(tmp_path) -> None:
    model_config = _load_k31_model_config(tmp_path)

    assert model_config.attention_arch == AttentionArch.DSA
    assert (
        model_config.index_topk,
        model_config.index_head_dim,
        model_config.index_n_heads,
    ) == (2048, 128, 32)
    # BF16 projection weights do not require BF16 cache storage.
    assert model_config.index_k_format == "fp8_scaled"
    # All DSA ops sit in the one eager break that narrows a padded prefill-graph replay.
    assert model_config.dsa_padded_prefill_replay is True
    # The 64-wide unrotated slice stays part of the latent row.
    assert (model_config.kv_lora_rank, model_config.qk_rope_head_dim) == (512, 64)
    assert model_config.scaling == pytest.approx(192**-0.5, rel=1e-12)

    profile = _resolve_attn_side(model_config, requested_backend=None)
    assert profile.is_kda and profile.is_kimi_dsa_kda and not profile.is_dsa_kda
    assert profile.linear_attention == "kda"
    assert _resolve_full_attn_backend_name(profile, None, None) == "dsa"
    attn_config = kimi_recipe(
        dsa_index_k_format=model_config.index_k_format
    ).attn_config
    assert _resolve_cache_family(profile, attn_config) == "kimi_k3"
    assert "KimiK31ForConditionalGeneration" in _LINEAR_ATTN_CLS


def test_kvp_launch_passes_the_full_attention_dcp_check(tmp_path) -> None:
    """A KVP launch of the hybrid checks its full-attention leaf before cache
    allocation; the DSA leaf shards history (test_kimi_k31_dcp.py)."""
    # Importing the backends registers them, as the launch does.
    import tokenspeed.runtime.layers.attention.backends  # noqa: F401

    model_config = _load_k31_model_config(tmp_path)
    profile = _resolve_attn_side(model_config, requested_backend=None)

    _validate_mla_dcp_backend(
        _resolve_full_attn_backend_name(profile, None, None),
        model_config.attention_arch,
    )


@pytest.mark.parametrize(
    "launch, is_draft, admitted",
    [
        ({"disaggregation_mode": "prefill"}, False, True),
        ({"disaggregation_mode": "prefill"}, True, False),
        (
            {"disaggregation_mode": "prefill", "disable_prefill_graph": True},
            False,
            False,
        ),
        (
            {"disaggregation_mode": "prefill", "disaggregation_layerwise_interval": 1},
            False,
            False,
        ),
        (
            {"disaggregation_mode": "prefill", "pipeline_parallel_size": 2},
            False,
            False,
        ),
        # The PD decode role runs no prompt prefill, so it is refused in every case.
        ({"disaggregation_mode": "decode"}, False, False),
        (
            {"disaggregation_mode": "decode", "disaggregation_layerwise_interval": 1},
            False,
            False,
        ),
        (
            {
                "disaggregation_mode": "decode",
                "prefill_graph_capture_sizes": [256, 1024, 4096],
            },
            False,
            False,
        ),
        # Admitted by default, with or without named buckets or a cap.
        ({"disaggregation_mode": "null"}, False, True),
        (
            {"disaggregation_mode": "null", "prefill_graph_max_tokens": 4096},
            False,
            True,
        ),
        (
            {
                "disaggregation_mode": "null",
                "prefill_graph_capture_sizes": [256, 1024, 4096],
            },
            False,
            True,
        ),
        # No transfer step counter, so the interval cannot veto.
        (
            {"disaggregation_mode": "null", "disaggregation_layerwise_interval": 1},
            False,
            True,
        ),
        # A MIXED round replays with its decode rows in the narrowed break.
        ({"disaggregation_mode": "null", "enable_mixed_batch": True}, False, True),
        ({"disaggregation_mode": "null"}, True, False),
        # --disable-prefill-graph turns it off.
        ({"disaggregation_mode": "null", "disable_prefill_graph": True}, False, False),
        ({"disaggregation_mode": "null", "pipeline_parallel_size": 2}, False, False),
        ({"disaggregation_mode": "encode"}, False, False),
    ],
)
def test_dsa_padded_prefill_replay_is_admitted_on_the_prefill_and_aggregated_roles(
    tmp_path, launch, is_draft, admitted
) -> None:
    from tokenspeed.runtime.layers.attention.configs.dsa import (
        dsa_padded_prefill_replay,
    )

    model_config = _load_k31_model_config(tmp_path)
    server_args = ServerArgs(model=str(tmp_path))
    server_args.disaggregation_layerwise_interval = 0
    for name, value in launch.items():
        setattr(server_args, name, value)
    assert dsa_padded_prefill_replay(server_args, model_config, is_draft) is admitted
    # A model that does not declare the padded-replay contract never is.
    model_config.dsa_padded_prefill_replay = False
    assert dsa_padded_prefill_replay(server_args, model_config, is_draft) is False


def test_dsa_spec_carries_the_padded_prefill_replay_admission(tmp_path) -> None:
    from tokenspeed.runtime.layers.attention.configs.dsa import DSAConfig

    model_config = _load_k31_model_config(tmp_path)
    server_args = ServerArgs(model=str(tmp_path))
    # The launch defaults admit the aggregated role.
    assert server_args.disaggregation_mode == "null"
    assert server_args.disaggregation_layerwise_interval == 1
    assert server_args.prefill_graph_capture_sizes is None
    assert server_args.enable_mixed_batch is False
    spec = DSAConfig._spec_kwargs(server_args, model_config, False)
    assert spec["padded_prefill_replay"] is True
    # The disable flag turns it off.
    server_args.disable_prefill_graph = True
    spec = DSAConfig._spec_kwargs(server_args, model_config, False)
    assert spec["padded_prefill_replay"] is False
    server_args.disable_prefill_graph = False
    # The prefill role keeps its layerwise condition.
    server_args.disaggregation_mode = "prefill"
    spec = DSAConfig._spec_kwargs(server_args, model_config, False)
    assert spec["padded_prefill_replay"] is False
    server_args.disaggregation_layerwise_interval = 0
    spec = DSAConfig._spec_kwargs(server_args, model_config, False)
    assert spec["padded_prefill_replay"] is True
    server_args.disaggregation_mode = "decode"
    spec = DSAConfig._spec_kwargs(server_args, model_config, False)
    assert spec["padded_prefill_replay"] is False


# ---------------------------------------------------------------- cache plan


def _layer_fields(layout, layer_id: int) -> dict[str, object]:
    return {
        field.field_id.split(".", 2)[2]: field
        for field in layout.fields
        if field.field_id.startswith(f"layer.{layer_id}.")
    }


def test_k3_layout_keeps_one_plane_per_mla_layer() -> None:
    recipe, _, layout = kimi_tp8_layout()

    assert recipe.num_index_k_planes == 0
    assert len(layout.plane_bytes) == 24
    assert layout.lcm_block_bytes == 24 * 884_736
    assert not any(field.field_id.endswith(".index_k") for field in layout.fields)


@pytest.mark.parametrize("draft_layers", [0, 1])
def test_k31_layout_adds_one_index_k_plane_per_dsa_layer(draft_layers) -> None:
    recipe, _, layout = kimi_tp8_layout(
        dsa_index_k_format="fp8_scaled", draft_layers=draft_layers
    )
    k3, _, k3_layout = kimi_tp8_layout(draft_layers=draft_layers)
    row_bytes = dsa_index_k_row_bytes(128)
    mla_layers = [
        layer
        for layer, group in enumerate(recipe.target_group_ids)
        if group == FULL_ATTENTION
    ]

    # The index planes leave the K3 packing (12 MLA pages per parent) alone.
    assert dict(layout.group_packing) == dict(k3_layout.group_packing)
    assert recipe.num_index_k_planes == 24
    assert len(layout.plane_bytes) == 24 + draft_layers + 24
    index_k = [field for field in layout.fields if field.field_id.endswith(".index_k")]
    assert sorted(int(field.field_id.split(".")[1]) for field in index_k) == mla_layers
    for field in index_k:
        assert field.group_id == FULL_ATTENTION
        assert field.shape == (128, row_bytes)
        # 128-token page strides make the plane a contiguous [slots, row_bytes] matrix.
        assert field.page_stride_bytes == 128 * row_bytes
    assert dict(layout.plane_bytes)[f"layer.{mla_layers[0]}.index_k"] == (
        12 * 128 * row_bytes
    )
    assert layout.lcm_block_bytes == k3_layout.lcm_block_bytes + 24 * 12 * 128 * (
        row_bytes
    )
    if draft_layers:
        draft = _layer_fields(layout, recipe.num_target_layers)
        assert set(draft) == {"latent_kv"}


def test_k31_layout_plans_a_bf16_plane_in_its_own_dtype() -> None:
    _, _, layout = kimi_tp8_layout(dsa_index_k_format="bf16")
    field = _layer_fields(layout, 3)["index_k"]

    assert (field.shape, field.dtype) == ((128, 128), "bfloat16")


# ---------------------------------------------------------------- pool + top-k


def _k31_pool(
    device: str,
    *,
    num_lcm_blocks: int,
    draft_layers: int = 0,
    index_k_format: str = "fp8_scaled",
):
    from tokenspeed.runtime.layers.attention.kv_cache.factory import (
        create_cache_arena,
        create_cache_pool,
    )
    from tokenspeed.runtime.layers.attention.kv_cache.recipes.setup import (
        CachePoolSpec,
    )

    recipe = kimi_recipe(dsa_index_k_format=index_k_format, draft_layers=draft_layers)
    _, groups, layout = kimi_tp8_layout(
        dsa_index_k_format=index_k_format, draft_layers=draft_layers
    )
    plan = layout.bind(num_lcm_blocks)
    spec = CachePoolSpec(
        family="kimi_k3",
        memory_plan=plan,
        layer_types=recipe.layer_types,
        cache_group_specs=tuple(group for group, _ in groups),
        token_capacity=num_lcm_blocks * 12 * 128,
    )
    arena = create_cache_arena(spec, device=device, enable_memory_saver=False)
    target = create_cache_pool(
        spec.layer_view(first_layer=0, num_layers=recipe.num_target_layers),
        recipe.attn_config,
        arena,
        num_layers=recipe.num_target_layers,
        rank=0,
    )
    draft = None
    if draft_layers:
        draft = create_cache_pool(
            spec.layer_view(
                first_layer=recipe.num_target_layers,
                num_layers=draft_layers,
                family="mla",
            ),
            recipe.draft_attn_config,
            arena,
            num_layers=draft_layers,
            rank=0,
            field_layer_offset=recipe.num_target_layers,
        )
    return target, draft


def _slots(block_ids: list[int], num_tokens: int, device) -> torch.Tensor:
    """Cache slots of positions 0..num_tokens-1 over 128-token blocks."""
    positions = torch.arange(num_tokens, device=device)
    blocks = torch.tensor(block_ids, device=device)[positions // 128]
    return blocks * 128 + positions % 128


def _kernel_page_table(block_ids: list[int], device) -> torch.Tensor:
    """The router's table: each 128-token block is two 64-token kernel pages."""
    pages = [page for block in block_ids for page in (2 * block, 2 * block + 1)]
    return torch.tensor([pages], dtype=torch.int32, device=device)


@requires_gpu
def test_index_k_rows_round_trip_through_the_kernel_page_layout() -> None:
    from tokenspeed_kernel.ops.quantization import quantize_fp8

    from tokenspeed.runtime.layers.attention.kv_cache.dsa import split_index_k_rows
    from tokenspeed.runtime.layers.attention.kv_cache.hybrid_kda import (
        HybridKDADSATokenToKVPool,
    )
    from tokenspeed.runtime.layers.attention.kv_cache.mla import MLATokenToKVPool

    torch.manual_seed(0)
    pool, draft = _k31_pool("cuda", num_lcm_blocks=1, draft_layers=1)
    assert isinstance(pool, HybridKDADSATokenToKVPool)
    assert type(draft) is MLATokenToKVPool
    assert pool.index_k_page_size == 64
    with pytest.raises(ValueError):
        pool.get_index_k_buffer(0)  # a KDA layer

    slots = _slots([5, 2], 200, "cuda")
    keys = torch.randn(200, 128, device="cuda", dtype=torch.bfloat16)
    pool.set_index_k_buffer(3, slots, keys, write_mask=None)
    fp8, scale = split_index_k_rows(
        pool.gather_index_k_rows(3, slots, index_k_format="fp8_scaled"),
        index_head_dim=128,
        index_k_format="fp8_scaled",
    )
    expected_fp8, expected_scale = quantize_fp8(
        keys, granularity="token_group", group_size=128, scale_encoding="float32"
    )

    assert torch.equal(fp8, expected_fp8.view(torch.uint8))
    assert torch.equal(scale, expected_scale.reshape(200, 1))
    # The next DSA layer's plane is untouched.
    assert not pool.get_index_k_buffer(7).any()


@requires_gpu
def test_bf16_index_k_plane_keeps_the_keys_unquantized() -> None:
    torch.manual_seed(0)
    pool, _ = _k31_pool("cuda", num_lcm_blocks=1, index_k_format="bf16")
    slots = _slots([5, 2], 200, "cuda")
    keys = torch.randn(200, 128, device="cuda", dtype=torch.bfloat16)
    pool.set_index_k_buffer(3, slots, keys, write_mask=None)

    plane = pool.get_index_k_buffer(3)
    assert plane.dtype == torch.bfloat16 and plane.shape[1] == 128
    assert torch.equal(plane[slots.long()], keys)
    assert not pool.get_index_k_buffer(7).any()


# Row widths of the projection the index keys are columns of: the indexer's
# own [wk; weights_proj] GEMM, and the K3 decode chain's packed DSA projection.
_KEY_PROJECTION_WIDTHS = pytest.mark.parametrize(
    "width", [160, 3808], ids=["indexer", "decode_chain"]
)


@requires_gpu
@_KEY_PROJECTION_WIDTHS
@pytest.mark.parametrize("index_k_format", ["bf16", "fp8_scaled"])
def test_the_index_k_write_applies_the_deferred_key_norm(index_k_format, width) -> None:
    """A pre-norm key handed over with its LayerNorm lands in the plane as the
    normalized key would; owned rows only."""
    from tokenspeed_kernel.ops.kvcache.triton import index_k_block_split_scatter
    from tokenspeed_kernel.platform import current_platform

    from tokenspeed.runtime.layers.layernorm import LayerNorm

    torch.manual_seed(0)
    norm = LayerNorm(128, eps=1e-6).cuda()
    with torch.no_grad():
        norm.weight.uniform_(0.5, 1.5)
        norm.bias.normal_(0.0, 0.1)
    slots = _slots([5, 2], 200, "cuda")
    # A row-strided view, as the key and head-weight columns of a projection.
    wide = torch.randn(200, width, device="cuda") * 3 + 1
    keys = wide.to(torch.bfloat16)[:, width - 160 : width - 32]
    owned = torch.rand(200, device="cuda") < 0.75
    deferred, _ = _k31_pool("cuda", num_lcm_blocks=1, index_k_format=index_k_format)
    eager, _ = _k31_pool("cuda", num_lcm_blocks=1, index_k_format=index_k_format)

    deferred.set_index_k_buffer(3, slots, keys, write_mask=owned, key_norm=norm)
    if index_k_format == "fp8_scaled" and not current_platform().is_amd:
        # Mirror the fused FP8 rounding; NVIDIA quantize_fp8 may differ, AMD's matches.
        normed = norm(keys).float()
        scale = normed.abs().amax(dim=1, keepdim=True).clamp_min(1e-10) / 448.0
        index_k_block_split_scatter(
            eager.index_k_buffer[3],
            (normed / scale).clamp(-448.0, 448.0).to(torch.float8_e4m3fn),
            scale,
            slots,
            page_size=eager.index_k_page_size,
            head_dim=128,
            group_size=128,
            write_mask=owned,
        )
    else:
        eager.set_index_k_buffer(3, slots, norm(keys), write_mask=owned)

    written, expected = deferred.get_index_k_buffer(3), eager.get_index_k_buffer(3)
    if index_k_format == "fp8_scaled":
        # Page-split rows: compare the whole plane, foreign slots included.
        assert torch.equal(written, expected)
    else:
        reference = torch.nn.functional.layer_norm(
            keys.float(), (128,), norm.weight, norm.bias, 1e-6
        )
        torch.testing.assert_close(
            written[slots[owned].long()].float(),
            reference[owned],
            rtol=2**-7,
            atol=1e-5,
        )
        assert not written[slots[~owned].long()].any()
    assert not deferred.get_index_k_buffer(7).any()


@requires_gpu
@_KEY_PROJECTION_WIDTHS
def test_the_index_k_write_can_quantize_the_index_query_in_its_launch(width) -> None:
    """The fused write matches the separate write and query quantization.

    It leaves the plane as the separate deferred-norm write does, foreign slots
    included, and returns the separate quantization of the same rows' index
    queries.
    """
    from tokenspeed_kernel.ops.attention.dsa import triton as dsa_triton

    from tokenspeed.runtime.layers import layernorm

    torch.manual_seed(0)
    norm = layernorm.LayerNorm(128, eps=1e-6).cuda()
    with torch.no_grad():
        norm.weight.uniform_(0.5, 1.5)
        norm.bias.normal_(0.0, 0.1)
    slots = _slots([5, 2], 200, "cuda")
    # Row-strided views of the key and head-weight columns of one projection.
    keys, weights = (
        (torch.randn(200, width, device="cuda") * 3 + 1)
        .to(torch.bfloat16)[:, width - 160 :]
        .split((128, 32), dim=1)
    )
    magnitudes = torch.logspace(-3, 3, 32, device="cuda").unsqueeze(-1)
    query = (torch.randn(200, 32, 128, device="cuda") * magnitudes).to(torch.bfloat16)
    owned = torch.rand(200, device="cuda") < 0.75
    fused, _ = _k31_pool("cuda", num_lcm_blocks=1)
    separate, _ = _k31_pool("cuda", num_lcm_blocks=1)

    q_fp8, folded = fused.set_index_k_buffer_and_quantize_query(
        3,
        slots,
        keys,
        write_mask=owned,
        key_norm=norm,
        query=query,
        query_weights=weights,
        softmax_scale=128**-0.5 * 32**-0.5,
    )
    separate.set_index_k_buffer(3, slots, keys, write_mask=owned, key_norm=norm)
    expected_q, expected_w = dsa_triton.quantize_index_query_weights(
        query, weights, 128**-0.5 * 32**-0.5
    )

    assert torch.equal(fused.get_index_k_buffer(3), separate.get_index_k_buffer(3))
    assert torch.equal(q_fp8.view(torch.uint8), expected_q.view(torch.uint8))
    assert torch.equal(folded, expected_w)
    assert not fused.get_index_k_buffer(7).any()


def _indexer_attention(*, index_topk: int):
    """A real K3.1 sparse MLA layer (id 3) at small width on one rank."""
    from tokenspeed.runtime.distributed import Mapping
    from tokenspeed.runtime.models.kimi_k31 import KimiK31SparseMLAAttention

    config = KimiLinearConfig(
        hidden_size=256,
        num_hidden_layers=4,
        num_attention_heads=2,
        num_key_value_heads=2,
        q_lora_rank=128,
        kv_lora_rank=128,
        linear_attn_config={
            "kda_layers": [1, 2, 3],
            "full_attn_layers": [4],
            "num_heads": 2,
            "head_dim": 128,
            "short_conv_kernel_size": 4,
            "use_full_rank_gate": True,
            "gate_lower_bound": -5.0,
        },
        dsa_config=dict(_DSA_CONFIG, dsa_layers=[4], index_topk=index_topk),
    )
    default_dtype = torch.get_default_dtype()
    torch.set_default_dtype(torch.bfloat16)
    try:
        with torch.device("cuda"):
            attention = KimiK31SparseMLAAttention(
                config=config,
                mapping=Mapping(rank=0),
                hidden_size=config.hidden_size,
                num_heads=config.num_attention_heads,
                qk_nope_head_dim=config.qk_nope_head_dim,
                qk_rope_head_dim=config.qk_rope_head_dim,
                v_head_dim=config.v_head_dim,
                q_lora_rank=config.q_lora_rank,
                kv_lora_rank=config.kv_lora_rank,
                max_position_embeddings=4096,
                quant_config=None,
                layer_id=3,
                prefix="model.layers.3.self_attn",
                reduce_attn_results=False,
                alt_stream=None,
                qkv_parallel=None,
                output_parallel=None,
            )
    finally:
        torch.set_default_dtype(default_dtype)
    return attention


@requires_gpu
@pytest.mark.parametrize(
    "num_rows, slots, gather_dtype",
    [
        (2500, 6144, torch.int16),
        (2501, 6144, torch.int16),
        (2500, 65534, torch.int16),
        (2500, 65535, torch.int32),
    ],
)
def test_prefill_topk_split_over_tp_matches_every_rank_selecting(
    monkeypatch: pytest.MonkeyPatch,
    num_rows: int,
    slots: int,
    gather_dtype: torch.dtype,
) -> None:
    from tokenspeed.runtime.distributed import Mapping
    from tokenspeed.runtime.layers.attention.dsa.utils import (
        workspace_indices_to_kv_slots,
    )
    from tokenspeed.runtime.models import glm5

    torch.manual_seed(0)
    attention = _indexer_attention(index_topk=2048)
    # Rows end past 2**15 when the workspace allows, so int16 indices wrap.
    prefix = min(slots, 40000) - num_rows
    q = torch.randn(num_rows, 32, 128, device="cuda", dtype=torch.bfloat16)
    weights = torch.randn(num_rows, 32, device="cuda", dtype=torch.bfloat16)
    pool, _ = _k31_pool("cuda", num_lcm_blocks=(slots + 1535) // 1536)
    pool.set_index_k_buffer(
        3,
        torch.arange(slots, device="cuda", dtype=torch.int64),
        torch.randn(slots, 128, device="cuda", dtype=torch.bfloat16),
        write_mask=None,
    )
    plane = pool.get_index_k_buffer(3)
    row_ends_cpu = prefix + torch.arange(1, num_rows + 1, dtype=torch.int64)
    args = dict(
        q=q,
        weights=weights,
        kv_workspace_slots=torch.arange(slots, device="cuda", dtype=torch.int64),
        row_starts=torch.zeros(num_rows, device="cuda", dtype=torch.int32),
        row_ends=row_ends_cpu.to(device="cuda", dtype=torch.int32),
        candidate_lens_cpu=row_ends_cpu,
        index_k_cache=plane,
        page_size=64,
        max_logits_bytes=1 << 30,
        topk=2048,
    )
    attention.prefill_topk_tp_split = False
    expected, expected_lens = attention._select_prefill_topk_rows(**args)

    # The split projects the query lazily, each rank only its own rows.
    projected_rows = []

    def query_proj(rows):
        projected_rows.append(rows.shape[0])
        return rows

    args["query_proj"] = query_proj
    # Two TP ranks, no process group: rank 1 runs first, rank 0 gathers its inputs.
    attention.prefill_topk_tp_split = True
    sent = {0: [], 1: []}

    def all_gather_single(output, input, group):
        del group
        # NCCL process groups take no int16 (or other short integer) tensors.
        assert input.dtype in (torch.int32, torch.bfloat16)
        assert output.dtype == input.dtype
        sent[rank].append(input.clone())
        if rank == 0:
            output.copy_(torch.cat((input, sent[1][len(sent[0]) - 1])))

    monkeypatch.setattr(glm5, "all_gather_single", all_gather_single)
    for rank in (1, 0):
        attention.mapping = Mapping(rank=rank, world_size=2, attn_tp_size=2)
        selected, lens = attention._select_prefill_topk_rows(**args)

    assert selected.dtype == gather_dtype
    assert projected_rows == [num_rows - 1264, 1264]
    selected = workspace_indices_to_kv_slots(selected, None)
    assert torch.equal(lens, expected_lens)
    assert torch.equal(selected.sort(dim=1).values, expected.sort(dim=1).values)


@requires_gpu
@pytest.mark.parametrize("num_rows, split", [(383, False), (384, True), (848, True)])
def test_prefill_topk_splits_from_384_rows(
    monkeypatch: pytest.MonkeyPatch, num_rows: int, split: bool
) -> None:
    """A 50K prompt's 848-row tail chunk splits its top-k rows over the ranks;
    below 384 rows every rank selects every row, with no all-gather."""
    from tokenspeed.runtime.distributed import Mapping
    from tokenspeed.runtime.models import glm5

    torch.manual_seed(0)
    attention = _indexer_attention(index_topk=2048)
    attention.prefill_topk_tp_split = True
    slots = 6144
    pool, _ = _k31_pool("cuda", num_lcm_blocks=4)
    pool.set_index_k_buffer(
        3,
        torch.arange(slots, device="cuda", dtype=torch.int64),
        torch.randn(slots, 128, device="cuda", dtype=torch.bfloat16),
        write_mask=None,
    )
    row_ends_cpu = 3000 + torch.arange(1, num_rows + 1, dtype=torch.int64)
    projected_rows = []

    def query_proj(rows):
        projected_rows.append(rows.shape[0])
        return rows

    gathers = []
    monkeypatch.setattr(
        glm5, "all_gather_single", lambda output, input, group: gathers.append(1)
    )
    attention.mapping = Mapping(rank=0, world_size=8, attn_tp_size=8)
    attention._select_prefill_topk_rows(
        q=torch.randn(num_rows, 32, 128, device="cuda", dtype=torch.bfloat16),
        weights=torch.randn(num_rows, 32, device="cuda", dtype=torch.bfloat16),
        kv_workspace_slots=torch.arange(slots, device="cuda", dtype=torch.int64),
        row_starts=torch.zeros(num_rows, device="cuda", dtype=torch.int32),
        row_ends=row_ends_cpu.to(device="cuda", dtype=torch.int32),
        candidate_lens_cpu=row_ends_cpu,
        index_k_cache=pool.get_index_k_buffer(3),
        page_size=64,
        max_logits_bytes=1 << 30,
        topk=2048,
        query_proj=query_proj,
    )
    share = -(-num_rows // (8 * 16)) * 16
    assert projected_rows == ([share] if split else [num_rows])
    assert len(gathers) == (2 if split else 0)


@requires_gpu
def test_prefill_topk_rows_gather_through_the_k3_iris_state(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """With the attention group's K3 Iris state, the int16 selection and its
    lengths gather in one Iris launch; NCCL remains the fallback."""
    from types import SimpleNamespace

    from tokenspeed.runtime.distributed import Mapping
    from tokenspeed.runtime.models import glm5

    attention = _indexer_attention(index_topk=2048)
    attention.mapping = Mapping(rank=0, world_size=8, attn_tp_size=8)
    local = torch.randint(-1, 6000, (16, 2048), device="cuda", dtype=torch.int32)
    lens = torch.randint(0, 2049, (16,), device="cuda", dtype=torch.int32)
    gathers = []
    monkeypatch.setattr(
        glm5, "all_gather_single", lambda output, input, group: gathers.append(1)
    )
    gathered = (
        torch.empty((128, 2048), device="cuda", dtype=torch.int16),
        torch.empty((128,), device="cuda", dtype=torch.int32),
    )
    calls = []

    def gather_prefill_topk(selection, selection_lens):
        calls.append((selection, selection_lens))
        return reply

    attention.topk_iris_state = SimpleNamespace(gather_prefill_topk=gather_prefill_topk)
    reply = gathered
    assert attention._gather_prefill_topk_rows(local, lens, torch.int16) is gathered
    assert calls == [(local, lens)] and gathers == []

    reply = None
    rows, rows_lens = attention._gather_prefill_topk_rows(local, lens, torch.int16)
    assert len(calls) == 2 and len(gathers) == 2
    assert rows.shape == (128, 2048) and rows.dtype == torch.int16
    assert rows_lens.shape == (128,)

    # Workspaces of 0xFFFF or more slots gather int32 rows through NCCL.
    rows, _ = attention._gather_prefill_topk_rows(local, lens, torch.int32)
    assert len(calls) == 2 and len(gathers) == 4 and rows.dtype == torch.int32


@pytest.mark.parametrize(
    "fused, packed",
    [(True, False), (False, False), (True, True)],
    ids=["fused", "separate", "decode-chain"],
)
def test_a_k31_decode_step_reads_its_query_and_index_projections(fused, packed) -> None:
    """The decode weights in reading order.

    They come from the stacked query and index-key weights when those were fused
    after load, else from each projection; the decode chain's packed projection
    holds the index-key rows.
    """
    from tokenspeed.runtime.models import kimi_k31

    attention = object.__new__(kimi_k31.KimiK31SparseMLAAttention)
    torch.nn.Module.__init__(attention)
    attention.fused_qkv_a_proj_with_mqa = SimpleNamespace(weight="qkv_a")
    attention.q_b_proj = SimpleNamespace(weight="q_b")
    attention.o_proj = SimpleNamespace(weight="o_proj")
    attention.w_kc, attention.w_vc = "w_kc", "w_vc"
    attention.q_b_wq_b_weight = "q_b_wq_b" if fused else None
    attention.decode_projection = "qkv_a_wk_weights_proj" if packed else None
    attention.indexer = SimpleNamespace(
        wq_b=SimpleNamespace(weight="wq_b"),
        wk=SimpleNamespace(weight="wk"),
        weights_proj=SimpleNamespace(weight="weights_proj"),
        wk_weights_proj_weight="wk_weights_proj" if fused else None,
    )
    if packed:
        head = ("qkv_a_wk_weights_proj", "q_b_wq_b")
    elif fused:
        head = ("qkv_a", "q_b_wq_b", "wk_weights_proj")
    else:
        head = ("qkv_a", "q_b", "wq_b", "wk", "weights_proj")
    assert attention.decode_weights() == (*head, "w_kc", "w_vc", "o_proj")


def test_k31_layers_take_the_attention_iris_state(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from types import SimpleNamespace

    from tokenspeed.runtime.models import kimi_k3, kimi_k31

    monkeypatch.setattr(
        kimi_k3.KimiK3ForConditionalGeneration,
        "prepare_communication_runtime",
        lambda self, max_num_tokens: True,
    )
    attention = object.__new__(kimi_k31.KimiK31SparseMLAAttention)
    other = SimpleNamespace()
    state = object()
    model = object.__new__(kimi_k31.KimiK31ForConditionalGeneration)
    torch.nn.Module.__init__(model)
    model.language_model = SimpleNamespace(
        model=SimpleNamespace(
            layers=[
                SimpleNamespace(
                    self_attn=attention, k3_comm=SimpleNamespace(iris_state=state)
                ),
                SimpleNamespace(
                    self_attn=other, k3_comm=SimpleNamespace(iris_state=state)
                ),
            ]
        )
    )
    assert model.prepare_communication_runtime(8192) is True
    assert attention.topk_iris_state is state
    assert not hasattr(other, "topk_iris_state")


@requires_gpu
def test_prefill_topk_split_reuses_rank_zero_kernels_on_other_ranks(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from unittest.mock import patch

    from tokenspeed_triton.runtime.jit import JITFunction

    from tokenspeed.runtime.distributed import Mapping
    from tokenspeed.runtime.models import glm5

    torch.manual_seed(0)
    attention = _indexer_attention(index_topk=2048)
    attention.prefill_topk_tp_split = True
    # Over two ranks, rank 1's int32 row offset (1251 * 4 bytes) is not 16-byte aligned.
    num_rows, prefix, slots = 2502, 3000, 6144
    row_ends_cpu = prefix + torch.arange(1, num_rows + 1, dtype=torch.int64)
    pool, _ = _k31_pool("cuda", num_lcm_blocks=4)
    pool.set_index_k_buffer(
        3,
        torch.arange(slots, device="cuda", dtype=torch.int64),
        torch.randn(slots, 128, device="cuda", dtype=torch.bfloat16),
        write_mask=None,
    )
    args = dict(
        q=torch.randn(num_rows, 32, 128, device="cuda", dtype=torch.bfloat16),
        weights=torch.randn(num_rows, 32, device="cuda", dtype=torch.bfloat16),
        kv_workspace_slots=torch.arange(slots, device="cuda", dtype=torch.int64),
        row_starts=torch.zeros(num_rows, device="cuda", dtype=torch.int32),
        row_ends=row_ends_cpu.to(device="cuda", dtype=torch.int32),
        candidate_lens_cpu=row_ends_cpu,
        index_k_cache=pool.get_index_k_buffer(3),
        page_size=64,
        max_logits_bytes=1 << 30,
        topk=2048,
    )
    monkeypatch.setattr(glm5, "all_gather_single", lambda output, input, group: None)
    attention.mapping = Mapping(rank=0, world_size=2, attn_tp_size=2)
    attention._select_prefill_topk_rows(**args)
    torch.cuda.synchronize()

    attention.mapping = Mapping(rank=1, world_size=2, attn_tp_size=2)
    with patch.object(
        JITFunction, "_do_compile", autospec=True, side_effect=JITFunction._do_compile
    ) as compiles:
        attention._select_prefill_topk_rows(**args)
        torch.cuda.synchronize()
    assert compiles.call_count == 0


@requires_gpu
def test_prefill_topk_split_scores_each_ranks_own_index_queries(monkeypatch) -> None:
    """With the TP split each rank projects its own rows, and over an
    fp8_scaled plane quantizes exactly those projected rows."""
    from tokenspeed.runtime.distributed import Mapping
    from tokenspeed.runtime.models import glm5
    from tokenspeed.runtime.models.kimi_k31 import _scorers_take_quantized_index_queries

    torch.manual_seed(0)
    attention = _indexer_attention(index_topk=2048)
    indexer = attention.indexer
    with torch.no_grad():
        indexer.wq_b.weight.normal_(0.0, 0.05)
    pool, _ = _k31_pool("cuda", num_lcm_blocks=1)
    num_rows = 1100
    slots = _slots(list(range(9)), num_rows, "cuda")
    keys = torch.randn(num_rows, 128, device="cuda", dtype=torch.bfloat16)
    pool.set_index_k_buffer(3, slots, keys, write_mask=None)
    q_lora = torch.randn(num_rows, 128, device="cuda", dtype=torch.bfloat16)
    row_ends_cpu = torch.arange(1, num_rows + 1, dtype=torch.int64)
    calls = _record_scorer_queries(monkeypatch, "dsa_prefill_topk")
    monkeypatch.setattr(glm5, "all_gather_single", lambda output, input, group: None)

    for rank in (0, 1):
        attention.mapping = Mapping(rank=rank, world_size=2, attn_tp_size=2)
        attention._select_prefill_topk_rows(
            q=q_lora,
            weights=torch.randn(num_rows, 32, device="cuda", dtype=torch.bfloat16),
            kv_workspace_slots=slots,
            row_starts=torch.zeros(num_rows, device="cuda", dtype=torch.int32),
            row_ends=row_ends_cpu.to(device="cuda", dtype=torch.int32),
            candidate_lens_cpu=row_ends_cpu,
            index_k_cache=pool.get_index_k_buffer(3),
            page_size=64,
            max_logits_bytes=1 << 30,
            topk=2048,
            query_proj=indexer._project_query,
        )

    # Whole 16-row groups per rank: 560 rows, then the remaining 540.
    assert len(calls) == 2
    fp8_queries = _scorers_take_quantized_index_queries(indexer)
    for (q, q_scales), rows in zip(calls, (slice(0, 560), slice(560, num_rows))):
        reference = indexer._project_query(q_lora[rows])
        if fp8_queries:
            _assert_fp8_index_query(q, q_scales, reference)
        else:
            assert torch.equal(q, reference) and q_scales is None


@requires_gpu
def test_indexer_matches_an_fp32_reference() -> None:
    torch.manual_seed(0)
    attention = _indexer_attention(index_topk=2048)
    indexer = attention.indexer
    with torch.no_grad():
        for linear in (indexer.wq_b, indexer.wk, indexer.weights_proj):
            linear.weight.normal_(0.0, 0.05)
        indexer.k_norm.weight.uniform_(0.5, 1.5)
        indexer.k_norm.bias.normal_(0.0, 0.1)
    hidden = torch.randn(37, 256, device="cuda", dtype=torch.bfloat16)
    q_lora = torch.randn(37, 128, device="cuda", dtype=torch.bfloat16)

    out = indexer(hidden, q_lora, torch.arange(37, device="cuda"))

    query = (q_lora.float() @ indexer.wq_b.weight.float().T).view(37, 32, 128)
    raw_key = hidden.float() @ indexer.wk.weight.float().T
    key = torch.nn.functional.layer_norm(
        raw_key,
        (128,),
        indexer.k_norm.weight.float(),
        indexer.k_norm.bias.float(),
        1e-6,
    )
    weights = hidden.float() @ indexer.weights_proj.weight.float().T
    projected = out.query_rows(0, 37)
    assert projected.shape == (37, 32, 128) and out.key.shape == (37, 128)
    assert out.key_norm is indexer.k_norm
    torch.testing.assert_close(projected.float(), query, atol=2e-3, rtol=1.6e-2)
    torch.testing.assert_close(out.query_rows(5, 9), projected[5:9], rtol=0, atol=0)
    torch.testing.assert_close(out.key.float(), raw_key, atol=2e-3, rtol=1.6e-2)
    torch.testing.assert_close(
        out.key_norm(out.key).float(), key, atol=2e-2, rtol=1.6e-2
    )
    torch.testing.assert_close(out.weights.float(), weights, atol=2e-3, rtol=1.6e-2)
    assert indexer.weights_softmax_scale == pytest.approx(128**-0.5 * 32**-0.5)


@requires_gpu
def test_the_fused_key_and_head_weight_projection_matches_the_two_gemms(
    monkeypatch,
) -> None:
    """One GEMM for ``wk`` and ``weights_proj`` gives the two projections'
    outputs, keeps tracking in-place loads into either parameter, and is
    skipped under bitwise numerics."""
    from tokenspeed.runtime.utils.env import global_server_args_dict

    torch.manual_seed(0)
    indexer = _indexer_attention(index_topk=2048).indexer
    with torch.no_grad():
        for linear in (indexer.wq_b, indexer.wk, indexer.weights_proj):
            linear.weight.normal_(0.0, 0.05)
    hidden = torch.randn(37, 256, device="cuda", dtype=torch.bfloat16)
    q_lora = torch.randn(37, 128, device="cuda", dtype=torch.bfloat16)
    positions = torch.arange(37, device="cuda")
    planned = indexer(hidden, q_lora, positions)

    indexer.fuse_key_weight_projections()
    assert indexer.wk_weights_proj_weight.shape == (128 + 32, 256)
    fused = indexer(hidden, q_lora, positions)
    torch.testing.assert_close(
        fused.query_rows(0, 37), planned.query_rows(0, 37), rtol=0, atol=0
    )
    for got, want in ((fused.key, planned.key), (fused.weights, planned.weights)):
        torch.testing.assert_close(got.float(), want.float(), rtol=2**-7, atol=1e-5)

    with torch.no_grad():
        indexer.wk.weight.normal_(0.0, 0.05)
        indexer.weights_proj.weight.normal_(0.0, 0.05)
    reloaded = indexer(hidden, q_lora, positions)
    torch.testing.assert_close(
        reloaded.key.float(),
        hidden.float() @ indexer.wk.weight.float().T,
        atol=2e-3,
        rtol=1.6e-2,
    )
    torch.testing.assert_close(
        reloaded.weights.float(),
        hidden.float() @ indexer.weights_proj.weight.float().T,
        atol=2e-3,
        rtol=1.6e-2,
    )

    bitwise = _indexer_attention(index_topk=2048).indexer
    monkeypatch.setitem(global_server_args_dict, "numerics", "rl-bitwise")
    bitwise.fuse_key_weight_projections()
    assert bitwise.wk_weights_proj_weight is None
    # Bitwise numerics project every row in one GEMM up front.
    eager = bitwise(hidden, q_lora, positions)
    assert eager.query_proj is None and eager.query.shape == (37, 32, 128)


@requires_gpu
def test_the_decode_chain_projection_carries_the_index_keys_and_head_weights() -> None:
    """The decode chain's projection carries the index keys and head weights.

    ``pack_decode_projection`` stacks [q_a | kv_a | gate | wk | weights_proj]
    once, changing no value and no GEMM of the other paths: the parameters and
    the stacked [wk; weights_proj] become views of its rows, the indexer's own
    projection stays bit-identical, and in-place loads (and a weight update's
    repeated fusion) keep it current. The indexer takes the key and head-weight
    columns of one GEMM over it, matching its two GEMMs within FP32
    accumulation.
    """
    from tokenspeed_kernel.ops import gemm

    torch.manual_seed(0)
    attention = _indexer_attention(index_topk=2048)
    indexer = attention.indexer
    with torch.no_grad():
        for linear in (
            attention.fused_qkv_a_proj_with_mqa,
            indexer.wq_b,
            indexer.wk,
            indexer.weights_proj,
        ):
            linear.weight.normal_(0.0, 0.05)
    hidden = torch.randn(37, 256, device="cuda", dtype=torch.bfloat16)
    q_lora = torch.randn(37, 128, device="cuda", dtype=torch.bfloat16)
    positions = torch.arange(37, device="cuda")
    two_gemms = indexer(hidden, q_lora, positions)
    indexer.fuse_key_weight_projections()
    stacked = indexer(hidden, q_lora, positions)
    parts = [part.detach().clone() for part in attention.decode_projection_parts()]
    rows = parts[0].shape[0]

    packed = attention.pack_decode_projection()
    assert attention.pack_decode_projection() is packed
    assert attention.decode_projection is packed
    assert packed.shape == (rows + 160, 256) and packed.is_contiguous()
    assert torch.equal(packed, torch.cat(parts))
    views = (
        attention.fused_qkv_a_proj_with_mqa.weight,
        indexer.wk.weight,
        indexer.weights_proj.weight,
        indexer.wk_weights_proj_weight,
    )
    for view, start in zip(views, (0, rows, rows + 128, rows)):
        assert view.data_ptr() == packed[start].data_ptr()
        assert view.stride() == packed.stride()
    assert indexer.wk_weights_proj_weight.shape == (160, 256)
    ordinary = indexer(hidden, q_lora, positions)
    assert torch.equal(ordinary.key, stacked.key)
    assert torch.equal(ordinary.weights, stacked.weights)

    out = gemm.mm(hidden, packed)
    chain = indexer.from_key_weights(q_lora, out[:, rows:])
    assert chain.key.data_ptr() == out[:, rows].data_ptr()
    assert chain.weights.data_ptr() == out[:, rows + 128].data_ptr()
    assert chain.key.stride() == chain.weights.stride() == (rows + 160, 1)
    assert chain.key_norm is indexer.k_norm
    torch.testing.assert_close(
        chain.query_rows(0, 37), two_gemms.query_rows(0, 37), rtol=0, atol=0
    )
    for got, want in ((chain.key, two_gemms.key), (chain.weights, two_gemms.weights)):
        torch.testing.assert_close(got.float(), want.float(), rtol=2**-7, atol=1e-5)

    with torch.no_grad():
        attention.fused_qkv_a_proj_with_mqa.weight.normal_(0.0, 0.05)
        indexer.wk.weight.normal_(0.0, 0.05)
        indexer.weights_proj.weight.normal_(0.0, 0.05)
    indexer.fuse_key_weight_projections()
    assert indexer.wk_weights_proj_weight.data_ptr() == packed[rows].data_ptr()
    reloaded = (
        attention.fused_qkv_a_proj_with_mqa.weight,
        indexer.wk.weight,
        indexer.weights_proj.weight,
    )
    assert torch.equal(packed, torch.cat(reloaded))


@requires_gpu
def test_the_fused_query_projection_matches_the_two_gemms(monkeypatch) -> None:
    """One GEMM over ``q_norm`` for ``q_b_proj`` and the indexer's ``wq_b``
    gives the two projections, keeps tracking in-place loads into either
    parameter across a repeated fusion, and is skipped under bitwise numerics."""
    from tokenspeed_kernel.ops.gemm import mm

    from tokenspeed.runtime.utils.env import global_server_args_dict

    torch.manual_seed(0)
    attention = _indexer_attention(index_topk=2048)
    with torch.no_grad():
        for linear in (attention.q_b_proj, attention.indexer.wq_b):
            linear.weight.normal_(0.0, 0.05)
    q_norm = torch.randn(37, 128, device="cuda", dtype=torch.bfloat16)
    want_q = attention.q_b_proj(q_norm)[0].float()
    want_index = attention.indexer._project_query(q_norm).float()

    attention.fuse_query_projections()
    fused = attention.q_b_wq_b_weight
    rows = attention.q_b_proj.weight.shape[0]
    assert fused.shape == (rows + 32 * 128, 128)
    got_q, got_index = mm(q_norm, fused).split((rows, 32 * 128), dim=-1)
    torch.testing.assert_close(got_q.float(), want_q, rtol=2**-7, atol=1e-5)
    torch.testing.assert_close(
        got_index.float().view(37, 32, 128), want_index, rtol=2**-7, atol=1e-5
    )

    with torch.no_grad():
        attention.q_b_proj.weight.normal_(0.0, 0.05)
        attention.indexer.wq_b.weight.normal_(0.0, 0.05)
    got_q, got_index = mm(q_norm, fused).split((rows, 32 * 128), dim=-1)
    torch.testing.assert_close(
        got_q.float(),
        q_norm.float() @ attention.q_b_proj.weight.float().T,
        atol=2e-3,
        rtol=1.6e-2,
    )
    torch.testing.assert_close(
        got_index.float(),
        q_norm.float() @ attention.indexer.wq_b.weight.float().T,
        atol=2e-3,
        rtol=1.6e-2,
    )
    attention.fuse_query_projections()
    assert attention.q_b_wq_b_weight is fused
    assert attention.q_b_proj.weight.data_ptr() == fused.data_ptr()

    bitwise = _indexer_attention(index_topk=2048)
    monkeypatch.setitem(global_server_args_dict, "numerics", "rl-bitwise")
    bitwise.fuse_query_projections()
    assert bitwise.q_b_wq_b_weight is None


def _separable_keys(num_tokens: int, bad: torch.Tensor, device):
    """Index rows whose top-k is unambiguous: every head's query is the same
    direction ``u``; good keys point along it (positive score), bad keys
    against it (ReLU zeroes every head, score exactly 0)."""
    u = torch.nn.functional.normalize(torch.randn(128, device=device), dim=0)
    magnitude = 1.0 + torch.rand(num_tokens, 1, device=device)
    sign = torch.ones(num_tokens, 1, device=device)
    sign[bad] = -1.0
    keys = sign * magnitude * u + 0.01 * torch.randn(num_tokens, 128, device=device)
    return u, keys.to(torch.bfloat16)


def _indexer_output(u: torch.Tensor, num_rows: int):
    from tokenspeed.runtime.models.glm5 import GlmDsaIndexerOutput

    query = (4.0 * u).expand(num_rows, 32, 128).to(torch.bfloat16).contiguous()
    weights = torch.rand(num_rows, 32, device=u.device).add(0.1).to(torch.bfloat16)
    return GlmDsaIndexerOutput(query=query, key=torch.empty(0), weights=weights)


def _record_scorer_queries(monkeypatch, name: str) -> list:
    """Run glm5's top-k scorer ``name`` unchanged, keeping every call's index
    query and query scales."""
    from tokenspeed.runtime.models import glm5

    scorer = getattr(glm5, name)
    calls = []

    def record(q, *args, q_scales, **kwargs):
        calls.append((q.clone(), q_scales))
        return scorer(q, *args, q_scales=q_scales, **kwargs)

    monkeypatch.setattr(glm5, name, record)
    return calls


def _assert_fp8_index_query(q, q_scales, reference) -> None:
    """``q`` with ``q_scales`` is the BF16 ``reference`` in E4M3, one FP32
    scale ``amax / 448`` per (token, head): to 1 ulp (the quantizer's FP32
    division), and dequantized within half an E4M3 step."""
    reference = reference.float()
    assert q.dtype == torch.float8_e4m3fn and q.shape == reference.shape
    assert q_scales.dtype == torch.float32 and q_scales.shape == reference.shape[:2]
    assert q_scales.stride(-1) == 1
    amax = reference.abs().amax(dim=-1).clamp_min(1e-10)
    torch.testing.assert_close(q_scales, amax / 448, rtol=2**-23, atol=0)
    scales = q_scales.unsqueeze(-1)
    error = (q.float() * scales - reference).abs()
    # 2**-4 relative on normal codes, 2**-10 of the scale on subnormal ones.
    assert bool((error <= 2**-4 * reference.abs() + 2**-10 * scales).all())


@requires_gpu
@pytest.mark.parametrize("phase", ["prefill", "decode"])
def test_configured_index_k_quantization_and_topk(tmp_path, phase) -> None:
    from tokenspeed_kernel.ops.attention.dsa import (
        dsa_decode_topk,
        dsa_plan,
        dsa_prefill_topk,
    )

    from tokenspeed.runtime.layers.attention.kv_cache.dsa import split_index_k_rows
    from tokenspeed.runtime.layers.layernorm import LayerNorm

    torch.manual_seed(4)
    model_config = _load_k31_model_config(tmp_path)
    pool, _ = _k31_pool(
        "cuda", num_lcm_blocks=2, index_k_format=model_config.index_k_format
    )
    seq_len, topk = 2200, model_config.index_topk
    block_ids = list(range(23, 5, -1))
    slots = _slots(block_ids, seq_len, "cuda")
    bad = torch.randperm(seq_len, device="cuda")[: seq_len - topk]
    u, keys = _separable_keys(seq_len, bad, "cuda")
    norm = LayerNorm(128, eps=1e-6).cuda()
    pool.set_index_k_buffer(3, slots, keys, write_mask=None, key_norm=norm)
    plane = pool.get_index_k_buffer(3)

    # The real model config must reach the shared FP8 write and top-k paths.
    assert keys.dtype == torch.bfloat16
    assert plane.dtype == torch.uint8
    fp8, scales = split_index_k_rows(
        pool.gather_index_k_rows(3, slots, index_k_format=model_config.index_k_format),
        index_head_dim=128,
        index_k_format=model_config.index_k_format,
    )
    normalized = norm(keys).float()
    torch.testing.assert_close(
        scales, normalized.abs().amax(dim=1, keepdim=True) / 448, rtol=1e-6, atol=1e-9
    )
    # E4M3 has three mantissa bits; check the quantized values after scaling.
    torch.testing.assert_close(
        fp8.view(torch.float8_e4m3fn).float() * scales,
        normalized,
        rtol=1 / 16,
        atol=1e-5,
    )
    indexer_output = _indexer_output(u, 1)
    common = dict(
        q=indexer_output.query,
        weights=indexer_output.weights,
        index_k_cache=plane,
        page_size=pool.index_k_page_size,
        topk=topk,
        softmax_scale=128**-0.5 * 32**-0.5,
        batch_invariant=False,
        slot_order="selection",
    )
    ends = torch.tensor([seq_len], device="cuda", dtype=torch.int32)
    if phase == "prefill":
        selected, lens = dsa_prefill_topk(
            **common,
            kv_workspace_slots=slots,
            row_starts=torch.zeros_like(ends),
            row_ends=ends,
        )
        selected = slots[selected.long()]
    else:
        seq_lens_2d = ends.unsqueeze(1)
        selected, lens = dsa_decode_topk(
            **common,
            seq_lens=ends,
            block_table=_kernel_page_table(block_ids, "cuda"),
            seq_lens_2d=seq_lens_2d,
            plan=dsa_plan(seq_lens_2d=seq_lens_2d, page_size=pool.index_k_page_size),
        )
    good = torch.ones(seq_len, device="cuda", dtype=torch.bool)
    good[bad] = False
    assert lens.tolist() == [topk]
    assert sorted(selected[0].tolist()) == sorted(slots[good].tolist())


@requires_gpu
def test_k31_scores_an_fp8_scaled_plane_with_fp8_index_queries() -> None:
    """Over an fp8_scaled plane K3.1 quantizes its index query per (token,
    head), where the platform registers FP8-query scorers."""
    from tokenspeed.runtime.models.kimi_k31 import _scorers_take_quantized_index_queries

    torch.manual_seed(0)
    attention = _indexer_attention(index_topk=2048)
    plane = _k31_pool("cuda", num_lcm_blocks=1)[0].get_index_k_buffer(3)
    # Head magnitudes six decades apart: each head needs its own scale.
    magnitudes = torch.logspace(-3, 3, 32, device="cuda").unsqueeze(-1)
    q = (torch.randn(37, 32, 128, device="cuda") * magnitudes).to(torch.bfloat16)

    q_out, q_scales = attention._scorer_index_query(q, plane)

    if _scorers_take_quantized_index_queries(attention.indexer):
        _assert_fp8_index_query(q_out, q_scales, q)
    else:
        assert q_out is q and q_scales is None


@requires_gpu
@pytest.mark.parametrize("case", ["bf16_plane", "glm_dsa_host", "bitwise_numerics"])
def test_index_query_stays_bf16_outside_k31_fp8_scaled_planes(
    monkeypatch, case
) -> None:
    from tokenspeed.runtime.models.glm5 import GlmMoeDsaAttention
    from tokenspeed.runtime.utils.env import global_server_args_dict

    index_k_format = "bf16" if case == "bf16_plane" else "fp8_scaled"
    plane = _k31_pool("cuda", num_lcm_blocks=1, index_k_format=index_k_format)[
        0
    ].get_index_k_buffer(3)
    if case == "glm_dsa_host":
        host = GlmMoeDsaAttention.__new__(GlmMoeDsaAttention)
    else:
        host = _indexer_attention(index_topk=2048)
    if case == "bitwise_numerics":
        monkeypatch.setitem(global_server_args_dict, "numerics", "rl-bitwise")
    q = torch.randn(37, 32, 128, device="cuda", dtype=torch.bfloat16)

    q_out, q_scales = host._scorer_index_query(q, plane)

    assert q_out is q and q_scales is None


@requires_gpu
@pytest.mark.parametrize("index_k_format", ["fp8_scaled", "bf16"])
@pytest.mark.parametrize("q_len_per_req", [1, 4])
def test_decode_topk_selects_every_visible_key_below_topk(
    monkeypatch, q_len_per_req, index_k_format
) -> None:
    from tokenspeed_kernel.ops.attention.dsa import dsa_plan

    from tokenspeed.runtime.models.kimi_k31 import _scorers_take_quantized_index_queries

    if index_k_format == "bf16" and torch.version.hip is None:
        pytest.skip("BF16 index-K top-k kernels require AMD")

    torch.manual_seed(0)
    attention = _indexer_attention(index_topk=2048)
    pool, _ = _k31_pool("cuda", num_lcm_blocks=1, index_k_format=index_k_format)
    block_ids = [3, 7, 1]
    seq_len = 300
    slots = _slots(block_ids, seq_len, "cuda")
    u, keys = _separable_keys(seq_len, torch.tensor([], dtype=torch.long), "cuda")
    pool.set_index_k_buffer(3, slots, keys, write_mask=None)
    seq_lens = torch.tensor([seq_len], dtype=torch.int32, device="cuda")
    seq_lens_2d = seq_lens.repeat_interleave(q_len_per_req).unsqueeze(1).contiguous()
    # The runtime sizes the page table for the whole context, at least topk candidates.
    page_table = torch.zeros(1, 32, dtype=torch.int32, device="cuda")
    kernel_pages = _kernel_page_table(block_ids, "cuda")
    page_table[:, : kernel_pages.shape[1]] = kernel_pages
    metadata = SimpleNamespace(
        _dsa_seq_lens_2d=seq_lens_2d,
        _dsa_plan=dsa_plan(seq_lens_2d=seq_lens_2d, page_size=64),
    )
    # Only the leaf surface the selection reads; a real leaf needs the serving stack.
    ctx = SimpleNamespace(
        attn_backend=SimpleNamespace(
            forward_decode_metadata=metadata, cache_placement=lambda layer: None
        ),
        token_to_kv_pool=pool,
        num_extends=0,
    )
    indexer_output = _indexer_output(u, q_len_per_req)
    calls = _record_scorer_queries(monkeypatch, "dsa_decode_topk")

    selected = attention._compute_decode_topk_indices_portable(
        indexer_output=indexer_output,
        ctx=ctx,
        seq_lens=seq_lens,
        page_table=page_table,
        q_len_per_req=q_len_per_req,
        decode_start=0,
        num_tokens=q_len_per_req,
        num_decode_tokens=q_len_per_req,
        topk=2048,
    )

    [(q, q_scales)] = calls
    if index_k_format == "fp8_scaled" and _scorers_take_quantized_index_queries(
        attention.indexer
    ):
        _assert_fp8_index_query(q, q_scales, indexer_output.query)
    else:
        assert torch.equal(q, indexer_output.query) and q_scales is None
    for row in range(q_len_per_req):
        # Verify row j sees the history up to its own position.
        visible = seq_len - (q_len_per_req - 1) + row
        assert int(selected.topk_lens[row]) == visible
        chosen = selected.topk_indices[row, :visible]
        assert sorted(chosen.tolist()) == sorted(slots[:visible].tolist())
        assert bool((selected.topk_indices[row, visible:] == -1).all())


@requires_gpu
def test_decode_topk_keeps_the_best_topk_of_a_longer_history() -> None:
    from tokenspeed_kernel.ops.attention.dsa import dsa_plan

    torch.manual_seed(1)
    attention = _indexer_attention(index_topk=2048)
    pool, _ = _k31_pool("cuda", num_lcm_blocks=2)
    block_ids = list(range(23, 5, -1))  # 18 blocks, scattered order
    seq_len = 2200
    slots = _slots(block_ids, seq_len, "cuda")
    bad = torch.randperm(seq_len, device="cuda")[: seq_len - 2048]
    u, keys = _separable_keys(seq_len, bad, "cuda")
    pool.set_index_k_buffer(3, slots, keys, write_mask=None)
    seq_lens = torch.tensor([seq_len], dtype=torch.int32, device="cuda")
    metadata = SimpleNamespace(
        _dsa_seq_lens_2d=seq_lens.unsqueeze(1),
        _dsa_plan=dsa_plan(seq_lens_2d=seq_lens.unsqueeze(1), page_size=64),
    )
    ctx = SimpleNamespace(
        attn_backend=SimpleNamespace(
            forward_decode_metadata=metadata, cache_placement=lambda layer: None
        ),
        token_to_kv_pool=pool,
        num_extends=0,
    )

    selected = attention._compute_decode_topk_indices_portable(
        indexer_output=_indexer_output(u, 1),
        ctx=ctx,
        seq_lens=seq_lens,
        page_table=_kernel_page_table(block_ids, "cuda"),
        q_len_per_req=1,
        decode_start=0,
        num_tokens=1,
        num_decode_tokens=1,
        topk=2048,
    )

    good = torch.ones(seq_len, dtype=torch.bool, device="cuda")
    good[bad] = False
    assert int(selected.topk_lens[0]) == 2048
    assert sorted(selected.topk_indices[0].tolist()) == sorted(slots[good].tolist())
    if attention.decode_topk_solution is not None:
        # The pinned fused top-k emits the selection in ascending position.
        assert torch.equal(selected.topk_indices[0], slots[good].to(torch.int32))


@requires_gpu
def test_the_fused_decode_topk_is_pinned_only_where_every_round_fits(
    monkeypatch,
) -> None:
    """More requests per round than the fused top-k takes keep normal selection."""
    from tokenspeed_kernel.ops.attention.dsa import cute_dsl as dsa_cute_dsl

    from tokenspeed.runtime.utils.env import global_server_args_dict

    cap = dsa_cute_dsl.FUSED_DECODE_TOPK_MAX_REQUESTS
    monkeypatch.setitem(global_server_args_dict, "max_num_seqs", cap)
    if _indexer_attention(index_topk=2048).decode_topk_solution is None:
        pytest.skip("needs the fused decode top-k (SM100-SM103)")
    monkeypatch.setitem(global_server_args_dict, "max_num_seqs", cap + 1)
    attention = _indexer_attention(index_topk=2048)
    assert attention.decode_topk_solution is None
    assert not attention.index_k_write_quantizes_query


@requires_gpu
@pytest.mark.parametrize("q_len_per_req", [1, 8])
def test_decode_topk_scores_the_query_the_index_k_write_quantized(
    q_len_per_req,
) -> None:
    """The index-K write's E4M3 query selects what the BF16 query selects.

    Handed the E4M3 query and folded weights of the fused index-K write, the
    decode top-k selects exactly what it selects from the BF16 query.
    """
    import dataclasses

    from tokenspeed_kernel.ops.attention import dsa
    from tokenspeed_kernel.ops.attention.dsa import triton as dsa_triton

    from tokenspeed.runtime.models import glm5

    torch.manual_seed(2)
    attention = _indexer_attention(index_topk=2048)
    if not attention.index_k_write_quantizes_query:
        pytest.skip("needs the fused decode top-k (SM100-SM103)")
    pool, _ = _k31_pool("cuda", num_lcm_blocks=2)
    block_ids = list(range(23, 5, -1))
    seq_len = 2200
    keys = torch.randn(seq_len, 128, device="cuda", dtype=torch.bfloat16)
    pool.set_index_k_buffer(
        3, _slots(block_ids, seq_len, "cuda"), keys, write_mask=None
    )
    seq_lens = torch.tensor([seq_len], dtype=torch.int32, device="cuda")
    seq_lens_2d = seq_lens.repeat_interleave(q_len_per_req).unsqueeze(1).contiguous()
    metadata = SimpleNamespace(
        _dsa_seq_lens_2d=seq_lens_2d,
        _dsa_plan=dsa.dsa_plan(seq_lens_2d=seq_lens_2d, page_size=64),
    )
    ctx = SimpleNamespace(
        attn_backend=SimpleNamespace(
            forward_decode_metadata=metadata, cache_placement=lambda layer: None
        ),
        token_to_kv_pool=pool,
        num_extends=0,
    )
    query = torch.randn(q_len_per_req, 32, 128, device="cuda").to(torch.bfloat16)
    weights = torch.randn(q_len_per_req, 32, device="cuda").to(torch.bfloat16)
    plain = glm5.GlmDsaIndexerOutput(query=query, key=torch.empty(0), weights=weights)
    quantized = dataclasses.replace(
        plain,
        quantized_query=dsa_triton.quantize_index_query_weights(
            query, weights, attention.indexer.weights_softmax_scale
        ),
    )

    def select(indexer_output):
        selected = attention._compute_decode_topk_indices_portable(
            indexer_output=indexer_output,
            ctx=ctx,
            seq_lens=seq_lens,
            page_table=_kernel_page_table(block_ids, "cuda"),
            q_len_per_req=q_len_per_req,
            decode_start=0,
            num_tokens=q_len_per_req,
            num_decode_tokens=q_len_per_req,
            topk=2048,
        )
        return selected.topk_indices.clone(), selected.topk_lens.clone()

    expected_slots, expected_lens = select(plain)
    slots, lens = select(quantized)
    assert torch.equal(lens, expected_lens) and torch.equal(slots, expected_slots)


@requires_gpu
@pytest.mark.parametrize(
    "prefix_len, extend_len",
    [(0, 300), (200, 100), (0, 2100), (2100, 64), (4736, 115), (4000, 900)],
)
def test_prefill_topk_matches_the_causal_reference(prefix_len, extend_len) -> None:
    from tokenspeed.runtime.models.glm5 import workspace_topk_to_global_slots

    torch.manual_seed(2)
    topk = 2048
    attention = _indexer_attention(index_topk=topk)
    seq_len = prefix_len + extend_len
    num_blocks = (seq_len + 127) // 128
    # Twelve 128-token blocks per LCM block; blocks 0 and 1 stay unused.
    pool, _ = _k31_pool("cuda", num_lcm_blocks=max(2, (num_blocks + 2 + 11) // 12))
    block_ids = list(range(2, 2 + num_blocks))[::-1]
    slots = _slots(block_ids, seq_len, "cuda")
    # Bad keys lead: a row past topk by the bad count selects exactly the good keys.
    bad = torch.arange(max(seq_len - topk, 0), device="cuda")
    u, keys = _separable_keys(seq_len, bad, "cuda")
    pool.set_index_k_buffer(3, slots, keys, write_mask=None)
    page_table = torch.zeros(
        1, max(64, 2 * num_blocks), dtype=torch.int32, device="cuda"
    )
    kernel_pages = _kernel_page_table(block_ids, "cuda")
    page_table[:, : kernel_pages.shape[1]] = kernel_pages
    chunk_meta = SimpleNamespace(
        extend_prefix_lens_cpu=[prefix_len],
        extend_seq_lens_cpu=[extend_len],
        page_table=page_table,
    )
    ctx = SimpleNamespace(
        attn_backend=SimpleNamespace(
            sparse_topk=SimpleNamespace(prefill=None),
            chunked_prefill_metadata=chunk_meta,
            cache_placement=lambda layer: None,
        ),
        token_to_kv_pool=pool,
        num_extends=1,
    )

    attention._compute_prefill_topk_indices(
        _indexer_output(u, extend_len), ctx, extend_len
    )
    # Once compiled, selection must not sync: the host stays ahead of a busy device.
    torch.cuda._sleep(1 << 30)
    earlier_work = torch.cuda.Event()
    earlier_work.record()
    selected = attention._compute_prefill_topk_indices(
        _indexer_output(u, extend_len), ctx, extend_len
    )
    assert not earlier_work.query()
    chosen = workspace_topk_to_global_slots(
        workspace_indices=selected.workspace_indices,
        kv_workspace_slots=selected.kv_workspace_slots,
    )

    for row in range(extend_len):
        visible = prefix_len + row + 1
        if visible <= topk:
            assert int(selected.topk_lens[row]) == visible
            assert sorted(chosen[row, :visible].tolist()) == sorted(
                slots[:visible].tolist()
            )
        elif visible == seq_len:
            assert int(selected.topk_lens[row]) == topk
            assert sorted(chosen[row].tolist()) == sorted(slots[bad.numel() :].tolist())


# ---------------------------------------------------------------- loader


def _tiny_text_config(*, dsa: bool) -> KimiLinearConfig:
    return KimiLinearConfig(
        vocab_size=256,
        hidden_size=256,
        intermediate_size=256,
        num_hidden_layers=4,
        num_attention_heads=2,
        num_key_value_heads=2,
        q_lora_rank=128,
        kv_lora_rank=128,
        # All layers dense; the loader still builds its expert plan from num_experts.
        num_experts=4,
        first_k_dense_replace=4,
        attn_res_block_size=2,
        pad_token_id=0,
        bos_token_id=1,
        eos_token_id=2,
        linear_attn_config={
            "kda_layers": [1, 2, 3],
            "full_attn_layers": [4],
            "num_heads": 2,
            "head_dim": 128,
            "short_conv_kernel_size": 4,
            "use_full_rank_gate": True,
            "gate_lower_bound": -5.0,
        },
        dsa_config=(
            dict(_DSA_CONFIG, dsa_layers=[4], index_topk=64, index_num_heads=4)
            if dsa
            else None
        ),
    )


@pytest.fixture
def isolated_k3_comm(monkeypatch):
    """K3's attention communication keeps process-wide state sized by the
    first model built; restore it so the small models here do not leak."""
    from tokenspeed.runtime.models.kimi_k3_comm import K3AttnComm

    monkeypatch.setattr(K3AttnComm, "_prepared_hidden_size", None)
    for name in ("attn_ar_fusion_ok", "dummy_norm", "cute_ar"):
        monkeypatch.setattr(K3AttnComm, name, getattr(K3AttnComm, name))


def _tiny_model(*, dsa: bool):
    from tokenspeed.runtime.distributed import Mapping
    from tokenspeed.runtime.models.kimi_k3 import KimiLinearForCausalLM

    default_dtype = torch.get_default_dtype()
    torch.set_default_dtype(torch.bfloat16)
    try:
        with torch.device("cuda"):
            return KimiLinearForCausalLM(
                _tiny_text_config(dsa=dsa), mapping=Mapping(rank=0), quant_config=None
            )
    finally:
        torch.set_default_dtype(default_dtype)


def _indexer_checkpoint(model) -> dict[str, torch.Tensor]:
    return {
        name: torch.randn_like(param)
        for name, param in model.named_parameters()
        if ".indexer." in name
    }


@requires_gpu
@pytest.mark.usefixtures("isolated_k3_comm")
def test_loader_fills_every_indexer_parameter() -> None:
    from tokenspeed.runtime.models.kimi_k3 import KimiLinearMLAAttention
    from tokenspeed.runtime.models.kimi_k31 import KimiK31SparseMLAAttention

    model = _tiny_model(dsa=True)
    checkpoint = _indexer_checkpoint(model)
    assert sorted(name.rsplit(".indexer.", 1)[1] for name in checkpoint) == [
        "k_norm.bias",
        "k_norm.weight",
        "weights_proj.weight",
        "wk.weight",
        "wq_b.weight",
    ]

    model.load_weights(list(checkpoint.items()))

    params = dict(model.named_parameters())
    for name, tensor in checkpoint.items():
        torch.testing.assert_close(params[name].float(), tensor.float(), rtol=0, atol=0)
    assert type(model.model.layers[3].self_attn) is KimiK31SparseMLAAttention
    assert isinstance(model.model.layers[3].self_attn, KimiLinearMLAAttention)


@requires_gpu
@pytest.mark.usefixtures("isolated_k3_comm")
def test_loader_refuses_an_unconsumed_indexer_tensor() -> None:
    model = _tiny_model(dsa=True)
    checkpoint = _indexer_checkpoint(model)
    checkpoint["model.layers.3.self_attn.indexer.wq_a.weight"] = torch.zeros(4, 4)

    with pytest.raises(RuntimeError, match="unconsumed"):
        model.load_weights(list(checkpoint.items()))


@requires_gpu
@pytest.mark.usefixtures("isolated_k3_comm")
def test_loader_refuses_a_missing_indexer_tensor() -> None:
    model = _tiny_model(dsa=True)
    checkpoint = _indexer_checkpoint(model)
    del checkpoint["model.layers.3.self_attn.indexer.k_norm.bias"]

    with pytest.raises(RuntimeError, match="never loaded"):
        model.load_weights(list(checkpoint.items()))


@requires_gpu
@pytest.mark.usefixtures("isolated_k3_comm")
def test_k3_loader_still_skips_unknown_tensors() -> None:
    from tokenspeed.runtime.models.kimi_k3 import KimiLinearMLAAttention

    model = _tiny_model(dsa=False)
    before = {name for name, _ in model.named_parameters()}

    model.load_weights(
        [("model.layers.3.self_attn.indexer.wq_b.weight", torch.zeros(8, 8))]
    )

    assert type(model.model.layers[3].self_attn) is KimiLinearMLAAttention
    assert {name for name, _ in model.named_parameters()} == before


# ---------------------------------------------------------------- sparse cores


def _dsa_leaf(pool, *, q_len: int, heads: int, padded_prefill_replay: bool = False):
    """The GPU DSA leaf over the K3.1 pool, at K3's latent geometry."""
    from tokenspeed.runtime.layers.attention.backends.paged.dsa import DSABackend
    from tokenspeed.runtime.layers.attention.configs.base import AttnConfig
    from tokenspeed.runtime.layers.attention.configs.dsa import DSAConfig

    spec = DSAConfig(
        backend_name="dsa",
        num_attention_heads=heads,
        num_kv_heads=1,
        head_dim=192,
        attn_tp_size=1,
        kv_lora_rank=512,
        qk_nope_head_dim=128,
        qk_rope_head_dim=64,
        v_head_dim=128,
        scaling=192**-0.5,
        kv_cache_dim=576,
        index_topk=2048,
        index_head_dim=128,
        index_n_heads=32,
        index_k_format="fp8_scaled",
        padded_prefill_replay=padded_prefill_replay,
    )
    config = AttnConfig(
        device="cuda",
        dtype=torch.bfloat16,
        kv_cache_dtype=torch.float8_e4m3fn,
        kernel_page_size=None,
        context_len=4096,
        max_bs=4,
        kv_cache_quant_method=None,
        speculative_num_draft_tokens=q_len,
        components=(spec,),
    )
    leaf = DSABackend(
        config,
        spec,
        kernel_page_size=DSABackend.resolve_kernel_page_size(config, 128),
    )
    leaf.set_cache_pool(pool)
    leaf.init_cuda_graph_state(max_bs=4)
    return leaf


@requires_gpu
@pytest.mark.parametrize("padded_prefill_replay", [False, True])
def test_dsa_leaf_admits_prefill_graphs_only_under_padded_replay(
    padded_prefill_replay, caplog, monkeypatch
) -> None:
    import logging

    from tokenspeed.runtime.layers.attention.backends.paged.dsa import DSABackend
    from tokenspeed.runtime.layers.attention.backends.support import (
        resolve_cuda_graph_support,
    )

    admits = padded_prefill_replay
    dsa_logger = "tokenspeed.runtime.layers.attention.backends.paged.dsa"
    support_logger = "tokenspeed.runtime.layers.attention.backends.support"
    for name in (dsa_logger, support_logger):
        monkeypatch.setattr(logging.getLogger(name), "propagate", True)
    pool, _ = _k31_pool("cuda", num_lcm_blocks=1)
    caplog.clear()
    with (
        caplog.at_level(logging.INFO, logger=dsa_logger),
        caplog.at_level(logging.INFO, logger=support_logger),
    ):
        leaf = _dsa_leaf(
            pool, q_len=1, heads=16, padded_prefill_replay=padded_prefill_replay
        )
        support = resolve_cuda_graph_support(leaf)
    messages = [
        record.getMessage()
        for record in caplog.records
        if record.name in (dsa_logger, support_logger)
    ]

    # The opt-in is per instance; the class keeps other DSA models' prefill eager.
    assert not DSABackend.cuda_graph_support.prefill_graph
    assert leaf.cuda_graph_support.prefill_graph is admits
    assert support.prefill_graph is admits
    admitted = any("DSA prefill graph admitted" in line for line in messages)
    assert admitted is admits
    vetoed = any(
        "Prefill CUDA graphs disabled by DSABackend" in line for line in messages
    )
    assert vetoed is not admits


def _mla_layer(heads: int):
    return SimpleNamespace(
        layer_id=3,
        tp_q_head_num=heads,
        head_dim=576,
        v_head_dim=512,
        scaling=192**-0.5,
        logit_cap=0.0,
    )


def _assert_attention_close(got: torch.Tensor, reference: torch.Tensor) -> None:
    """Within the BF16 probability/value tiles of the sparse cores (FP32
    accumulation): 2% of the reference magnitude, worst element and mean. A
    wrong slot or page gives errors of the reference's own size."""
    error = (got - reference).abs()
    assert error.max() <= 2e-2 * reference.abs().max(), error.max()
    assert error.mean() <= 2e-2 * reference.abs().mean(), error.mean()


def _dense_reference(q, history, scaling, visible):
    """FP32 attention of ``q`` [rows, heads, 576] over ``history[:visible[r]]``."""
    scores = torch.einsum("rhd,ld->rhl", q.float(), history.float()) * scaling
    mask = torch.arange(history.shape[0], device=q.device)[None, :] >= visible[:, None]
    scores.masked_fill_(mask[:, None, :], float("-inf"))
    return torch.softmax(scores, dim=-1) @ history[:, :512].float()


@requires_gpu
@pytest.mark.parametrize("q_len", [1, 4])
def test_sparse_decode_over_k3_latents_matches_dense_attention(q_len) -> None:
    torch.manual_seed(3)
    heads = 16
    pool, _ = _k31_pool("cuda", num_lcm_blocks=1)
    layer = _mla_layer(heads)
    block_ids = [6, 2, 9]
    seq_len = 300
    slots = _slots(block_ids, seq_len, "cuda")
    latent = torch.randn(seq_len, 1, 576, device="cuda", dtype=torch.bfloat16)
    pool.set_mla_kv_buffer(
        layer, slots, latent[..., :512], latent[..., 512:], write_mask=None
    )
    leaf = _dsa_leaf(pool, q_len=q_len, heads=heads)
    page_table = torch.zeros(1, leaf.max_num_pages, dtype=torch.int32, device="cuda")
    kernel_pages = _kernel_page_table(block_ids, "cuda")
    page_table[:, : kernel_pages.shape[1]] = kernel_pages
    seq_lens = torch.tensor([seq_len], dtype=torch.int32, device="cuda")
    leaf.refresh_decode_metadata(1, 1, seq_lens, page_table)
    visible = seq_len - (q_len - 1) + torch.arange(q_len, device="cuda")
    topk = torch.full((q_len, 2048), -1, dtype=torch.int32, device="cuda")
    for row in range(q_len):
        topk[row, : visible[row]] = slots[: visible[row]].to(torch.int32)
    q = torch.randn(q_len, heads, 576, device="cuda", dtype=torch.bfloat16)

    out = leaf.forward_decode(
        q,
        None,
        None,
        layer,
        None,
        pool,
        1,
        topk_indices=topk,
        topk_lens=visible.to(torch.int32),
    )

    history = pool.get_key_buffer(3)[slots, 0].float()
    # The FP8 cache runs the query in FP8 too.
    reference = _dense_reference(
        q.to(torch.float8_e4m3fn), history, layer.scaling, visible
    )
    got = out.view(q_len, heads, 512).float()
    _assert_attention_close(got, reference)


@requires_gpu
def test_sparse_prefill_over_k3_latents_matches_dense_attention() -> None:
    torch.manual_seed(4)
    heads = 16
    pool, _ = _k31_pool("cuda", num_lcm_blocks=1)
    layer = _mla_layer(heads)
    block_ids = [4, 11, 1]
    seq_len = 260
    slots = _slots(block_ids, seq_len, "cuda")
    latent = torch.randn(seq_len, 1, 576, device="cuda", dtype=torch.bfloat16)
    pool.set_mla_kv_buffer(
        layer, slots, latent[..., :512], latent[..., 512:], write_mask=None
    )
    leaf = _dsa_leaf(pool, q_len=1, heads=heads)
    rows = torch.arange(seq_len - 40, seq_len, device="cuda")
    visible = rows + 1
    topk = torch.full((rows.numel(), 2048), -1, dtype=torch.int32, device="cuda")
    for index, count in enumerate(visible.tolist()):
        topk[index, :count] = slots[:count].to(torch.int32)
    q = torch.randn(rows.numel(), heads, 576, device="cuda", dtype=torch.bfloat16)

    out = leaf.forward_sparse_prefill(
        q=q,
        layer=layer,
        token_to_kv_pool=pool,
        kv_seq_lens=visible.to(torch.int32),
        topk_slots=topk,
        topk_lens=visible.to(torch.int32),
        max_seq_len=seq_len,
    )

    history = pool.get_key_buffer(3)[slots, 0].float()
    reference = _dense_reference(
        q.to(torch.float8_e4m3fn), history, layer.scaling, visible
    )
    got = out.view(rows.numel(), heads, 512).float()
    _assert_attention_close(got, reference)


# ---------------------------------------------------------------- attention


def _k31_attention_stack(
    *,
    kv_cache_dtype,
    spec_width: int,
    heads: int,
    index_topk: int,
    kvp_size: int,
    kvp_rank: int,
    full_blocks: int = 0,
    context_len: int = 4096,
    q_lora_rank: int = 128,
):
    """A real K3.1 sparse MLA layer (id 3) over the K3.1 pool and the DSA
    router, at TP8's 12 local heads and K3's latent geometry (a narrow q
    latent unless ``q_lora_rank`` asks for the checkpoint's 1536).

    With ``kvp_size > 1`` the layer is one rank of a KVP group that is also
    its attention-TP group: the pool holds the rank's share of the
    full-attention blocks, and the router attends ``heads`` of the group's
    ``heads * kvp_size`` heads over the history the group's ranks own.
    """
    import dataclasses

    from tokenspeed.runtime.distributed import Mapping
    from tokenspeed.runtime.distributed.comm_manager import CommManager
    from tokenspeed.runtime.layers.attention.configs.base import AttnConfig
    from tokenspeed.runtime.layers.attention.configs.dsa import (
        DSAConfig,
        dsa_kvp_history_prefill,
    )
    from tokenspeed.runtime.layers.attention.kv_cache.factory import (
        create_cache_arena,
        create_cache_pool,
    )
    from tokenspeed.runtime.layers.attention.kv_cache.recipes.setup import (
        CachePoolSpec,
    )
    from tokenspeed.runtime.layers.attention.registry import (
        AttentionArch,
        create_paged_router,
    )
    from tokenspeed.runtime.layers.layernorm import RMSNorm
    from tokenspeed.runtime.layers.paged_attention import bind_cache_groups
    from tokenspeed.runtime.models.deepseek_v3 import _prepare_mla_kv_b_proj_weights
    from tokenspeed.runtime.models.kimi_k31 import KimiK31SparseMLAAttention

    recipe = kimi_recipe(dsa_index_k_format="fp8_scaled", kv_cache_dtype=kv_cache_dtype)
    kvp_group = tuple(range(kvp_size))
    dsa = dataclasses.replace(
        recipe.attn_config.component(DSAConfig),
        backend_name="dsa",
        index_topk=index_topk,
    )
    recipe.attn_config = dataclasses.replace(
        recipe.attn_config,
        device="cuda",
        kvp_size=kvp_size,
        kvp_rank=kvp_rank,
        kvp_group=kvp_group,
        components=(dsa, *recipe.attn_config.components[1:]),
    )
    groups, layout = kimi_layout(recipe)
    # At least two LCM blocks: twelve full-attention ones in BF16, more for full_blocks.
    packing = dict(layout.group_packing)[FULL_ATTENTION]
    num_lcm_blocks = max(2, -(-(full_blocks + 1) // packing))
    spec = CachePoolSpec(
        family="kimi_k3",
        memory_plan=layout.bind(num_lcm_blocks),
        layer_types=recipe.layer_types,
        cache_group_specs=tuple(group for group, _ in groups),
        token_capacity=num_lcm_blocks * packing * 128,
    )
    arena = create_cache_arena(spec, device="cuda", enable_memory_saver=False)
    pool = create_cache_pool(
        spec.layer_view(first_layer=0, num_layers=93),
        recipe.attn_config,
        arena,
        num_layers=93,
        rank=0,
    )
    config = KimiLinearConfig(
        hidden_size=256,
        num_hidden_layers=4,
        num_attention_heads=heads,
        num_key_value_heads=heads,
        q_lora_rank=q_lora_rank,
        linear_attn_config={
            "kda_layers": [1, 2, 3],
            "full_attn_layers": [4],
            "num_heads": 2,
            "head_dim": 128,
            "short_conv_kernel_size": 4,
            "use_full_rank_gate": True,
            "gate_lower_bound": -5.0,
        },
        dsa_config=dict(_DSA_CONFIG, dsa_layers=[4], index_topk=index_topk),
    )
    default_dtype = torch.get_default_dtype()
    torch.set_default_dtype(torch.bfloat16)
    try:
        with torch.device("cuda"):
            attention = KimiK31SparseMLAAttention(
                config=config,
                mapping=Mapping(rank=0),
                hidden_size=256,
                num_heads=heads,
                qk_nope_head_dim=128,
                qk_rope_head_dim=64,
                v_head_dim=128,
                q_lora_rank=q_lora_rank,
                kv_lora_rank=512,
                max_position_embeddings=context_len,
                quant_config=None,
                layer_id=3,
                prefix="model.layers.3.self_attn",
                reduce_attn_results=False,
                alt_stream=None,
                qkv_parallel=None,
                output_parallel=None,
            )
            norm = RMSNorm(256)
    finally:
        torch.set_default_dtype(default_dtype)
    comm = CommManager(
        mapping=Mapping(rank=0),
        layer_id=3,
        is_moe=False,
        prev_is_moe=False,
        dense_batch_invariant=False,
        input_layernorm=norm,
        post_attn_layernorm=norm,
        query_sharded=False,
    )
    with torch.no_grad():
        for name, param in attention.named_parameters():
            if name.endswith("layernorm.weight") or name.endswith("k_norm.weight"):
                param.uniform_(0.8, 1.2)
            else:
                param.normal_(0.0, 0.05)
    attention.w_kc, attention.w_vc = _prepare_mla_kv_b_proj_weights(
        attention.kv_b_proj.weight, attention
    )
    bind_cache_groups(attention, pool)
    config = AttnConfig(
        device="cuda",
        dtype=torch.bfloat16,
        kv_cache_dtype=kv_cache_dtype,
        kv_cache_quant_method=None,
        max_bs=4,
        context_len=context_len,
        speculative_num_draft_tokens=spec_width,
        kvp_size=kvp_size,
        kvp_rank=kvp_rank,
        kvp_group=kvp_group,
        components=(
            dataclasses.replace(
                dsa,
                num_attention_heads=heads * kvp_size,
                num_kv_heads=heads * kvp_size,
                attn_tp_size=kvp_size,
            ),
        ),
    )
    router = create_paged_router(config, AttentionArch.DSA, backend_name="dsa")
    router.set_cache_pool(pool)
    router.configure_runtime()
    if dsa_kvp_history_prefill(config):
        router.preallocate_history_gather_workspace(context_len)
    router.init_cuda_graph_state(4, max_extend_tokens=256)
    return attention, comm, router, pool


def _dense_gated_mla(attention, hidden: torch.Tensor, heads: int) -> torch.Tensor:
    """FP32 K3 MLA (no rotation, sigmoid output gate) over every causal key:
    what sparse MLA must equal while the history fits in index_topk."""
    x = hidden.float()
    q_a, kv, gate = (x @ attention.fused_qkv_a_proj_with_mqa.weight.float().T).split(
        [128, 576, heads * 128], dim=-1
    )

    def rms(t, norm):
        scale = torch.rsqrt(t.pow(2).mean(-1, keepdim=True) + norm.variance_epsilon)
        return t * scale * norm.weight.float()

    c_q = rms(q_a, attention.q_a_layernorm)
    c_kv = rms(kv[:, :512], attention.kv_a_layernorm)
    k_pe = kv[:, 512:]
    q = (c_q @ attention.q_b_proj.weight.float().T).view(-1, heads, 192)
    kv_b = (c_kv @ attention.kv_b_proj.weight.float().T).view(-1, heads, 256)
    scores = (
        torch.einsum("thd,shd->hts", q[..., :128], kv_b[..., :128])
        + torch.einsum("thd,sd->hts", q[..., 128:], k_pe)
    ) * 192**-0.5
    length = x.shape[0]
    causal = torch.ones(length, length, device=x.device).triu(1).bool()
    scores = scores.masked_fill(causal, float("-inf"))
    out = torch.einsum("hts,shd->thd", scores.softmax(-1), kv_b[..., 128:])
    out = out.reshape(length, heads * 128) * torch.sigmoid(gate)
    return out @ attention.o_proj.weight.float().T


def _assert_close_to_reference(got, reference, fraction: float) -> None:
    error = (got.float() - reference).abs()
    assert error.max() <= fraction * reference.abs().max(), error.max()
    assert error.mean() <= fraction * reference.abs().mean(), error.mean()


def _block_table(blocks: list[int]) -> dict[str, torch.Tensor]:
    return {FULL_ATTENTION: torch.tensor([blocks], dtype=torch.int32, device="cuda")}


def _extend_forward(
    attention, comm, router, pool, hidden, *, prefix: int, end: int, blocks
):
    """One EXTEND forward of rows ``[prefix, end)`` of ``hidden``, the rows
    before ``prefix`` being the request's cached history; their outputs."""
    from tokenspeed.runtime.execution.context import ForwardContext
    from tokenspeed.runtime.execution.forward_batch_info import ForwardMode
    from tokenspeed.runtime.execution.output_layout import ForwardOutputLayout

    blocks = _block_table(blocks)
    rows = end - prefix

    def lengths(value):
        return torch.tensor([value], dtype=torch.int32, device="cuda")

    def host(value):
        return torch.tensor([value], dtype=torch.int32)

    router.init_forward_metadata(
        1,
        1,
        torch.tensor([0], dtype=torch.int32, device="cuda"),
        lengths(end),
        ForwardMode.EXTEND,
        block_tables=blocks,
        extend_seq_lens=lengths(rows),
        extend_seq_lens_cpu=host(rows),
        extend_prefix_lens=lengths(prefix),
        extend_prefix_lens_cpu=host(prefix),
        extend_replay_lens_cpu=host(0),
        extend_prompt_lens_cpu=host(end),
        extend_with_prefix=prefix > 0,
        query_shard=None,
        block_tables_cpu={FULL_ATTENTION: blocks[FULL_ATTENTION].cpu()},
    )
    ctx = ForwardContext(
        attn_backend=router,
        token_to_kv_pool=pool,
        bs=1,
        num_extends=1,
        input_num_tokens=rows,
        forward_mode=ForwardMode.EXTEND,
        output_layout=ForwardOutputLayout(1, 1, 0, 1),
    )
    with torch.no_grad():
        return attention(
            torch.arange(prefix, end, device="cuda"),
            hidden[prefix:end],
            ctx,
            comm,
            projection_out=None,
        )


def _prefill_then_decode_twice(
    attention, comm, router, pool, hidden, *, prompt: int, spec_width: int, blocks
):
    """Prefill ``hidden``'s first ``prompt`` rows, decode the next
    ``spec_width`` rows eagerly while capturing that step in a CUDA graph,
    then replay the graph on the last ``spec_width`` rows at the same
    addresses with refreshed lengths.

    ``blocks`` is the request's full-attention block table. Returns the
    prefill, eager-decode and replayed outputs and the eager step's decode
    selection (its global slots and per-row counts).
    """
    from tokenspeed_kernel.workspace import capture_session

    from tokenspeed.runtime.execution.context import ForwardContext
    from tokenspeed.runtime.execution.forward_batch_info import ForwardMode
    from tokenspeed.runtime.execution.output_layout import ForwardOutputLayout

    prefill = _extend_forward(
        attention, comm, router, pool, hidden, prefix=0, end=prompt, blocks=blocks
    )
    total = prompt + 2 * spec_width
    positions = torch.arange(total, device="cuda")
    blocks = _block_table(blocks)
    request = torch.tensor([0], dtype=torch.int32, device="cuda")

    def lengths(*values):
        return torch.tensor(values, dtype=torch.int32, device="cuda")

    decode_ctx = ForwardContext(
        attn_backend=router,
        token_to_kv_pool=pool,
        bs=1,
        num_extends=0,
        input_num_tokens=spec_width,
        forward_mode=ForwardMode.DECODE,
        output_layout=ForwardOutputLayout(0, 0, 1, spec_width),
    )
    rows = slice(prompt, prompt + spec_width)
    step_hidden = hidden[rows].clone()
    step_positions = positions[rows].clone()
    router.refresh_decode_metadata(
        1,
        1,
        request,
        lengths(prompt + spec_width),
        forward_mode=ForwardMode.DECODE,
        block_tables=blocks,
    )
    # Run eagerly on the capture stream, like graph warm-up: a core may plan per stream.
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.no_grad(), torch.cuda.stream(stream):
        eager = attention(
            step_positions, step_hidden, decode_ctx, comm, projection_out=None
        ).clone()
        selection = router.sparse_topk.decode
        selection = (selection.topk_indices.clone(), selection.topk_lens.clone())
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.no_grad(), capture_session(graph), torch.cuda.graph(
        graph, stream=stream
    ):
        captured = attention(
            step_positions, step_hidden, decode_ctx, comm, projection_out=None
        )
    # Replay the next step at stable addresses with refreshed lengths.
    rows = slice(prompt + spec_width, total)
    step_hidden.copy_(hidden[rows])
    step_positions.copy_(positions[rows])
    router.refresh_decode_metadata(
        1,
        1,
        request,
        lengths(total),
        forward_mode=ForwardMode.DECODE,
        block_tables=blocks,
        for_graph_replay=True,
    )
    graph.replay()
    torch.cuda.synchronize()
    return prefill, eager, captured, selection


@requires_gpu
@pytest.mark.parametrize("fused_query", [False, True])
@pytest.mark.parametrize(
    "kv_cache_dtype, spec_width, fraction",
    [
        # BF16 cache: the BF16 activations and probability tiles.
        (torch.bfloat16, 1, 1e-2),
        (torch.bfloat16, 4, 1e-2),
        # FP8 E4M3 cache (K3 serving recipe): unit-scale latent rows and FP8 query.
        (torch.float8_e4m3fn, 1, 8e-2),
    ],
)
def test_k31_attention_matches_dense_mla_and_replays_in_a_decode_graph(
    monkeypatch, kv_cache_dtype, spec_width, fraction, fused_query
) -> None:
    from tokenspeed.runtime.utils.env import global_server_args_dict

    # The dense delegate sizes its prefix chunks from the launch's chunk size.
    monkeypatch.setitem(global_server_args_dict, "chunked_prefill_size", 8192)
    torch.manual_seed(5)
    heads = 12
    attention, comm, router, pool = _k31_attention_stack(
        kv_cache_dtype=kv_cache_dtype,
        spec_width=spec_width,
        heads=heads,
        index_topk=2048,
        kvp_size=1,
        kvp_rank=0,
    )
    if fused_query:
        attention.fuse_query_projections()
        assert attention.q_b_wq_b_weight is not None
    prompt = 40
    hidden = torch.randn(
        prompt + 2 * spec_width, 256, device="cuda", dtype=torch.bfloat16
    )
    reference = _dense_gated_mla(attention, hidden, heads)
    prefill, eager, replayed, _ = _prefill_then_decode_twice(
        attention,
        comm,
        router,
        pool,
        hidden,
        prompt=prompt,
        spec_width=spec_width,
        blocks=[5],
    )
    _assert_close_to_reference(prefill, reference[:prompt], fraction)
    _assert_close_to_reference(eager, reference[prompt : prompt + spec_width], fraction)
    _assert_close_to_reference(replayed, reference[prompt + spec_width :], fraction)


@requires_gpu
@pytest.mark.parametrize("projection", ["forward_gemms", "packed_weight"])
@pytest.mark.parametrize(
    "kv_cache_dtype, fraction", [(torch.bfloat16, 1e-2), (torch.float8_e4m3fn, 8e-2)]
)
def test_the_decode_chain_entry_attends_like_forward(
    monkeypatch, kv_cache_dtype, fraction, projection
) -> None:
    """The K3 TP8 chain enters via ``o_proj_input`` and attends like ``forward``.

    It enters with the packed [q_a | kv_a | gate | wk | weights_proj] projection;
    a K3.1 layer still attends its top-k, its indexer taking the keys and head
    weights from those columns.

    Packed from the GEMMs ``forward`` runs, the output and the index-K plane
    equal ``forward``'s bit for bit. From one GEMM over
    ``pack_decode_projection``'s weight (which leaves ``forward`` bit for bit
    as it was), the keys and head weights match ``forward``'s within FP32
    accumulation, and the output and the written index-K row within their
    rounding.
    """
    from tokenspeed_kernel.ops import gemm

    from tokenspeed.runtime.execution.context import ForwardContext
    from tokenspeed.runtime.execution.forward_batch_info import ForwardMode
    from tokenspeed.runtime.execution.output_layout import ForwardOutputLayout
    from tokenspeed.runtime.layers.attention.kv_cache import dsa as kv_dsa
    from tokenspeed.runtime.utils.env import global_server_args_dict

    monkeypatch.setitem(global_server_args_dict, "chunked_prefill_size", 8192)
    torch.manual_seed(5)
    attention, comm, router, pool = _k31_attention_stack(
        kv_cache_dtype=kv_cache_dtype,
        spec_width=1,
        heads=12,
        index_topk=2048,
        kvp_size=1,
        kvp_rank=0,
    )
    indexer = attention.indexer
    prompt = 40
    hidden = torch.randn(prompt + 1, 256, device="cuda", dtype=torch.bfloat16)
    _extend_forward(
        attention, comm, router, pool, hidden, prefix=0, end=prompt, blocks=[5]
    )
    ctx = ForwardContext(
        attn_backend=router,
        token_to_kv_pool=pool,
        bs=1,
        num_extends=0,
        input_num_tokens=1,
        forward_mode=ForwardMode.DECODE,
        output_layout=ForwardOutputLayout(0, 0, 1, 1),
    )
    router.refresh_decode_metadata(
        1,
        1,
        torch.tensor([0], dtype=torch.int32, device="cuda"),
        torch.tensor([prompt + 1], dtype=torch.int32, device="cuda"),
        forward_mode=ForwardMode.DECODE,
        block_tables=_block_table([5]),
    )
    positions = torch.tensor([prompt], device="cuda")
    step = hidden[prompt:]
    plane = pool.get_index_k_buffer(3)
    history = plane.clone()
    # The keys and head weights the index-K write and the decode top-k read.
    read = []
    select = attention._compute_decode_topk_indices

    def record(indexer_output, ctx, **kwargs):
        read.append((indexer_output.key.clone(), indexer_output.weights.clone()))
        return select(indexer_output, ctx, **kwargs)

    monkeypatch.setattr(attention, "_compute_decode_topk_indices", record)

    def forward_step():
        plane.copy_(history)
        with torch.no_grad():
            return attention(positions, step, ctx, comm, projection_out=None).clone()

    if projection == "forward_gemms":

        def packed_projection(x):
            # The GEMMs ``forward`` runs, packed as the chain supplies them.
            out = gemm.kimi3_mla_qkv_gate_projection(
                x, attention.fused_qkv_a_proj_with_mqa.weight, attention._qkv_a_width
            )
            qkv_gate = (
                out.packed
                if out.packed is not None
                else torch.cat((out.qkv, out.gate), dim=-1)
            )
            keys = (indexer.wk(x)[0], indexer.weights_proj(x)[0])
            return torch.cat((qkv_gate, *keys), dim=-1)

    else:
        unpacked = forward_step()
        weight = attention.pack_decode_projection()

        def packed_projection(x):
            return gemm.mm(x, weight)

    expected = forward_step()
    written = plane.clone()
    plane.copy_(history)
    with torch.no_grad():
        local = attention.o_proj_input(
            positions, step, ctx, comm, projection=packed_projection
        )
        actual, _ = attention.o_proj(local, ctx=ctx)
    (want_key, want_weights), (key, weights) = read[-2:]
    if projection == "forward_gemms":
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        assert torch.equal(key, want_key) and torch.equal(weights, want_weights)
        assert torch.equal(plane, written)
        return

    assert torch.equal(expected, unpacked)
    for got, want in ((key, want_key), (weights, want_weights)):
        torch.testing.assert_close(got.float(), want.float(), rtol=2**-7, atol=1e-5)
    _assert_close_to_reference(actual, expected.float(), fraction)
    # Only the step's index-K row differs, within FP8 rounding of nearby keys.
    slot = torch.tensor([5 * 128 + prompt], device="cuda")
    dequantized = []
    for source in (written, plane.clone()):
        plane.copy_(source)
        codes, scale = kv_dsa.split_index_k_rows(
            pool.gather_index_k_rows(3, slot, index_k_format="fp8_scaled"),
            index_head_dim=128,
            index_k_format="fp8_scaled",
        )
        dequantized.append(codes.view(torch.float8_e4m3fn).float() * scale)
    want_row, row = dequantized
    assert (plane != written).sum() <= 128 + 4
    torch.testing.assert_close(
        row, want_row, rtol=2**-2, atol=2**-5 * want_row.abs().max().item()
    )


@requires_gpu
@pytest.mark.parametrize("spec_width", [1, 4])
@pytest.mark.parametrize("prompt", [40, 2300], ids=["all_keys", "selected"])
def test_a_decode_round_quantizes_its_query_in_the_index_k_write(
    monkeypatch, prompt, spec_width
) -> None:
    """A decode round's fused index-K write matches the separate launches bit for bit.

    The fused index-K write of a decode round leaves the plane, the selection
    and the attention output bit-identical to the separate write and query
    quantization.
    """
    from tokenspeed.runtime.execution import context, forward_batch_info, output_layout
    from tokenspeed.runtime.utils import env

    monkeypatch.setitem(env.global_server_args_dict, "chunked_prefill_size", 8192)
    torch.manual_seed(5)
    total = prompt + spec_width
    num_blocks = -(-total // 128)
    attention, comm, router, pool = _k31_attention_stack(
        kv_cache_dtype=torch.float8_e4m3fn,
        spec_width=spec_width,
        heads=12,
        index_topk=2048,
        kvp_size=1,
        kvp_rank=0,
        full_blocks=num_blocks,
        context_len=max(4096, 2 * total),
    )
    if not attention.index_k_write_quantizes_query:
        pytest.skip("needs the fused decode top-k (SM100-SM103)")
    hidden = torch.randn(total, 256, device="cuda", dtype=torch.bfloat16)
    order = torch.randperm(num_blocks, generator=torch.Generator().manual_seed(3))
    blocks = (order + 1).tolist()
    _extend_forward(
        attention, comm, router, pool, hidden, prefix=0, end=prompt, blocks=blocks
    )
    plane = pool.get_index_k_buffer(3)
    history = plane.clone()
    ctx = context.ForwardContext(
        attn_backend=router,
        token_to_kv_pool=pool,
        bs=1,
        num_extends=0,
        input_num_tokens=spec_width,
        forward_mode=forward_batch_info.ForwardMode.DECODE,
        output_layout=output_layout.ForwardOutputLayout(0, 0, 1, spec_width),
    )

    def decode(fused: bool):
        plane.copy_(history)
        attention.index_k_write_quantizes_query = fused
        router.refresh_decode_metadata(
            1,
            1,
            torch.tensor([0], dtype=torch.int32, device="cuda"),
            torch.tensor([total], dtype=torch.int32, device="cuda"),
            forward_mode=forward_batch_info.ForwardMode.DECODE,
            block_tables=_block_table(blocks),
        )
        with torch.no_grad():
            out = attention(
                torch.arange(prompt, total, device="cuda"),
                hidden[prompt:],
                ctx,
                comm,
                projection_out=None,
            ).clone()
        selection = router.sparse_topk.decode
        return (
            out,
            selection.topk_indices.clone(),
            selection.topk_lens.clone(),
            plane.clone(),
        )

    separate = decode(False)
    fused = decode(True)
    assert not torch.equal(separate[3], history)
    for name, expected, got in zip(
        ("output", "slots", "lens", "plane"), separate, fused
    ):
        assert torch.equal(got, expected), name


@pytest.mark.parametrize(
    "case",
    [
        "takes",
        "extend",
        "shared_topk",
        "no_front",
        "unstacked_query",
        "separate_write",
        "indexer_projects_keys",
        "breakable_capture",
        "placed",
        "unsupported_rows",
        "most_rows",
        "past_most_rows",
    ],
)
def test_only_a_decode_chain_round_takes_the_decode_front(monkeypatch, case) -> None:
    """Only decode-chain rounds take the front.

    It serves decode rounds whose keys and head weights come from the
    a-projection and whose index-K write quantizes the query, without KV
    placement or a breakable capture, over at most 32 rows it addresses.
    """
    from tokenspeed.runtime.models import kimi_k31

    attention = object.__new__(kimi_k31.KimiK31SparseMLAAttention)
    torch.nn.Module.__init__(attention)
    attention.skip_indexer_topk = case == "shared_topk"
    attention.decode_front = None if case == "no_front" else object()
    attention.q_b_wq_b_weight = None if case == "unstacked_query" else object()
    attention.index_k_write_quantizes_query = case != "separate_write"
    attention.indexer = SimpleNamespace(index_head_dim=128, index_n_heads=32)
    attention.attn_mqa = SimpleNamespace(layer_id=3)
    placement = object() if case == "placed" else None
    backend = SimpleNamespace(cache_placement=lambda layer: placement)
    ctx = SimpleNamespace(num_extends=int(case == "extend"), attn_backend=backend)
    rows = {"most_rows": 32, "past_most_rows": 33}.get(case, 4)
    q_a, kv_a = torch.zeros(rows, 1536), torch.zeros(rows, 512)
    keys = None if case == "indexer_projects_keys" else torch.zeros(rows, 160)
    supported = []

    def rows_supported(*tensors):
        supported.append(tuple(t.shape for t in tensors))
        return case != "unsupported_rows"

    monkeypatch.setattr(
        kimi_k31.dsa_cute_dsl, "decode_front_rows_supported", rows_supported
    )
    monkeypatch.setattr(
        kimi_k31, "is_breakable_capture_active", lambda: case == "breakable_capture"
    )
    takes = case in ("takes", "most_rows")
    assert attention._takes_decode_front(ctx, q_a, kv_a, keys) == takes
    if case in ("takes", "unsupported_rows", "most_rows"):
        assert supported == [((rows, 1536), (rows, 512), (rows, 128), (rows, 32))]


@pytest.mark.parametrize(
    "num_extends, quantizes, caller_keys",
    [(1, True, True), (0, False, True), (0, True, False)],
    ids=["extend", "separate", "indexer_keys"],
)
def test_dsa_core_refuses_a_written_index_query_it_cannot_take(
    num_extends, quantizes, caller_keys
) -> None:
    """A written index query is refused outside the rounds it serves.

    It serves decode rounds of a layer whose index-K write quantizes the
    query, over keys the caller projected (the writing launch scattered
    them); anything else fails before touching the cache.
    """
    from tokenspeed.runtime.models.glm5 import GlmDsaSparseAttentionMixin

    core = SimpleNamespace(index_k_write_quantizes_query=quantizes)
    rows = torch.zeros(1, 4)
    with pytest.raises(ValueError, match="written index query"):
        GlmDsaSparseAttentionMixin.forward_dsa_core(
            core,
            torch.zeros(1, dtype=torch.long),
            rows,
            rows,
            rows,
            SimpleNamespace(num_extends=num_extends),
            None,
            index_key_weights=rows if caller_keys else None,
            written_index_query=(rows, rows),
        )


def _bf16_step(x: torch.Tensor) -> torch.Tensor:
    """The BF16 spacing at |x| (8 significant bits; the subnormal spacing below)."""
    exponent = torch.floor(torch.log2(x.abs().double().clamp_min(2.0**-126)))
    return torch.exp2(exponent - 7)


@requires_gpu
@pytest.mark.parametrize("spec_width", [1, 8])
@pytest.mark.parametrize("prompt", [40, 2300], ids=["all_keys", "selected"])
def test_a_decode_chain_round_runs_the_fused_front(
    monkeypatch, prompt, spec_width
) -> None:
    """A decode chain round runs the fused front where it is registered.

    Against the separate launches (``decode_front`` cleared), the front leaves
    the index-K plane, the latent cache and ``q_norm`` bit-identical, hands
    the core the E4M3 query ``quantize_index_query_weights`` gives for its own
    projection, and its projection, selection and output agree within the one
    BF16 step its K order may move a projected value. A CUDA graph of the
    round replays the next step bit for bit.
    """
    from tokenspeed_kernel.ops import gemm
    from tokenspeed_kernel.ops.attention.dsa._triton import topk as dsa_topk
    from tokenspeed_kernel.workspace import capture_session

    from tokenspeed.runtime.execution import context, forward_batch_info, output_layout
    from tokenspeed.runtime.utils import env

    monkeypatch.setitem(env.global_server_args_dict, "chunked_prefill_size", 8192)
    torch.manual_seed(5)
    total = prompt + 2 * spec_width
    num_blocks = -(-total // 128)
    attention, comm, router, pool = _k31_attention_stack(
        kv_cache_dtype=torch.float8_e4m3fn,
        spec_width=spec_width,
        heads=12,
        index_topk=2048,
        kvp_size=1,
        kvp_rank=0,
        full_blocks=num_blocks,
        context_len=max(4096, 2 * total),
        q_lora_rank=1536,
    )
    front = attention.decode_front
    if front is None:
        pytest.skip("needs the fused decode front (SM100-SM103)")
    attention.indexer.fuse_key_weight_projections()
    attention.fuse_query_projections()
    weight = attention.pack_decode_projection()
    hidden = torch.randn(total, 256, device="cuda", dtype=torch.bfloat16)
    order = torch.randperm(num_blocks, generator=torch.Generator().manual_seed(3))
    blocks = (order + 1).tolist()
    _extend_forward(
        attention, comm, router, pool, hidden, prefix=0, end=prompt, blocks=blocks
    )
    plane, latent = pool.get_index_k_buffer(3), pool.get_key_buffer(3)
    history = (plane.clone(), latent.clone())
    core_inputs = []
    recording = [True]
    core = attention.forward_dsa_core

    def record(positions, hidden_states, q_norm, latent_cache, ctx, comm, **kwargs):
        if recording[0]:
            written = kwargs["written_index_query"]
            core_inputs.append(
                (
                    q_norm.clone(),
                    kwargs["projected_q"].clone(),
                    kwargs["index_query"].clone(),
                    None if written is None else tuple(t.clone() for t in written),
                )
            )
        return core(positions, hidden_states, q_norm, latent_cache, ctx, comm, **kwargs)

    monkeypatch.setattr(attention, "forward_dsa_core", record)
    decode_mode = forward_batch_info.ForwardMode.DECODE
    ctx = context.ForwardContext(
        attn_backend=router,
        token_to_kv_pool=pool,
        bs=1,
        num_extends=0,
        input_num_tokens=spec_width,
        forward_mode=decode_mode,
        output_layout=output_layout.ForwardOutputLayout(0, 0, 1, spec_width),
    )
    request = torch.tensor([0], dtype=torch.int32, device="cuda")

    def refresh(length, **kwargs):
        router.refresh_decode_metadata(
            1,
            1,
            request,
            torch.tensor([length], dtype=torch.int32, device="cuda"),
            forward_mode=decode_mode,
            block_tables=_block_table(blocks),
            **kwargs,
        )

    def projection(x):
        return gemm.mm(x, weight)

    step_hidden = hidden[prompt : prompt + spec_width].clone()
    step_positions = torch.arange(prompt, prompt + spec_width, device="cuda")

    def decode(with_front: bool):
        plane.copy_(history[0])
        latent.copy_(history[1])
        attention.decode_front = front if with_front else None
        refresh(prompt + spec_width)
        with torch.no_grad():
            out = attention.o_proj_input(
                step_positions, step_hidden, ctx, comm, projection=projection
            ).clone()
        selection = router.sparse_topk.decode
        return (
            out,
            selection.topk_indices[:spec_width].clone(),
            selection.topk_lens[:spec_width].clone(),
            plane.clone(),
            latent.clone(),
        )

    separate = decode(False)
    fused = decode(True)
    (q_norm, proj, index_query, written), (
        front_q_norm,
        front_proj,
        front_index_query,
        front_written,
    ) = core_inputs
    assert written is None and front_written is not None
    assert torch.equal(front_q_norm.view(torch.int16), q_norm.view(torch.int16))
    assert torch.equal(fused[3], separate[3]) and not torch.equal(fused[3], history[0])
    assert torch.equal(fused[4].view(torch.uint8), separate[4].view(torch.uint8))
    for got, want in ((front_proj, proj), (front_index_query, index_query)):
        got, want = got.double(), want.double()
        step = _bf16_step(torch.maximum(got.abs(), want.abs()))
        assert ((got - want).abs() <= step + 1e-4 * want.abs().max()).all()
    head_weights = projection(step_hidden)[:, -attention.indexer.index_n_heads :]
    want_q, want_w = dsa_topk.quantize_index_query_weights(
        front_index_query, head_weights, attention.indexer.weights_softmax_scale
    )
    assert torch.equal(front_written[0].view(torch.uint8), want_q.view(torch.uint8))
    assert torch.equal(front_written[1].view(torch.int32), want_w.view(torch.int32))
    assert torch.equal(fused[2], separate[2])
    if prompt < 2048:
        assert torch.equal(fused[1], separate[1])
    for got, want, count in zip(fused[1], separate[1], separate[2].tolist()):
        shared = set(got[:count].tolist()) & set(want[:count].tolist())
        assert len(shared) >= 0.98 * count
    _assert_close_to_reference(fused[0], separate[0].float(), 2e-2)

    # A CUDA graph of the fused round replays the next step bit for bit.
    recording[0] = False
    attention.decode_front = front
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    plane.copy_(history[0])
    latent.copy_(history[1])
    refresh(prompt + spec_width)
    with torch.no_grad(), torch.cuda.stream(stream):
        attention.o_proj_input(
            step_positions, step_hidden, ctx, comm, projection=projection
        )
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.no_grad(), capture_session(graph), torch.cuda.graph(
        graph, stream=stream
    ):
        captured = attention.o_proj_input(
            step_positions, step_hidden, ctx, comm, projection=projection
        )
    before = (plane.clone(), latent.clone())
    step_hidden.copy_(hidden[prompt + spec_width :])
    step_positions.copy_(torch.arange(prompt + spec_width, total, device="cuda"))
    refresh(total, for_graph_replay=True)
    graph.replay()
    torch.cuda.synchronize()
    replayed = (captured.clone(), plane.clone(), latent.clone())
    plane.copy_(before[0])
    latent.copy_(before[1])
    refresh(total)
    with torch.no_grad():
        eager = attention.o_proj_input(
            step_positions, step_hidden, ctx, comm, projection=projection
        )
    assert torch.equal(replayed[0].view(torch.int16), eager.view(torch.int16))
    assert torch.equal(replayed[1], plane)
    assert torch.equal(replayed[2].view(torch.uint8), latent.view(torch.uint8))


def _extend_over_a_cached_prefix(stack, twin, hidden, *, prefix, blocks):
    """Rows ``[prefix, len(hidden))`` extending over a cached ``[0, prefix)``
    on ``stack``, and the same rows of a one-shot prefill on ``twin``."""
    end = hidden.shape[0]
    _extend_forward(*stack, hidden, prefix=0, end=prefix, blocks=blocks)
    resumed = _extend_forward(*stack, hidden, prefix=prefix, end=end, blocks=blocks)
    whole = _extend_forward(*twin, hidden, prefix=0, end=end, blocks=blocks)
    return resumed, whole[prefix:]


@requires_gpu
@pytest.mark.parametrize(
    "kv_cache_dtype, fraction",
    # Row counts differ, so BF16 rounding does too; an unwritten prefix errs by ~80%.
    [(torch.bfloat16, 1e-2), (torch.float8_e4m3fn, 3e-2)],
)
@pytest.mark.parametrize(
    "history, prefix, index_topk, slack",
    [
        # Below the 1200-row history: the extend's rows choose among the cached keys.
        pytest.param(1200, 640, 512, 1, id="short"),
        # A 4851-token prompt resumed at its 4736-token checkpoint, K3.1's index_topk.
        # gfx950's FP8 index queries swap a few near-tied keys among 2048 of 4851.
        pytest.param(4851, 4736, 2048, 3, id="served"),
    ],
)
def test_k31_extend_over_a_cached_prefix_matches_one_prefill(
    monkeypatch, kv_cache_dtype, fraction, history, prefix, index_topk, slack
) -> None:
    """Prefix caching: the extend selects over the index keys and attends the
    latent rows an earlier forward cached, as a one-shot prefill does."""
    from tokenspeed.runtime.utils.env import global_server_args_dict

    monkeypatch.setitem(global_server_args_dict, "chunked_prefill_size", 8192)
    num_blocks = -(-history // 128)
    stacks = []
    for _ in range(2):
        torch.manual_seed(5)
        stacks.append(
            _k31_attention_stack(
                kv_cache_dtype=kv_cache_dtype,
                spec_width=1,
                heads=12,
                index_topk=index_topk,
                kvp_size=1,
                kvp_rank=0,
                full_blocks=num_blocks,
                context_len=max(4096, 2 * history),
            )
        )
    hidden = torch.randn(history, 256, device="cuda", dtype=torch.bfloat16)
    # Scattered blocks: the history's order is the table's, not the pool's.
    blocks = torch.randperm(num_blocks, generator=torch.Generator().manual_seed(3)) + 1
    resumed, expected = _extend_over_a_cached_prefix(
        *stacks, hidden, prefix=prefix, blocks=blocks.tolist()
    )
    _assert_close_to_reference(resumed, expected.float(), slack * fraction)


# ---------------------------------------------------------------- prefill graph


_LIVE_ROWS = 11
_BUCKET_ROWS = 16


def _recording_dsa_core(calls: list, *, write_rows: int, gather: bool):
    """``GlmDsaSparseAttentionMixin.forward_dsa_core`` with recording collaborators.

    Its collaborators record every call into ``calls``: the indexer (projecting,
    or over keys and head weights projected ahead), the index-K write, the
    prefill top-k, ``q_b_proj`` and the sparse prefill core. The router hands
    out ``write_rows`` live extend slots and no KVP placement; the
    pre-attention comm is the identity unless ``gather``.
    """
    from tokenspeed.runtime.models.glm5 import (
        GlmDsaIndexerOutput,
        GlmDsaSparseAttentionMixin,
    )

    def record(name, *values):
        calls.append(
            (
                name,
                tuple(
                    value.clone() if isinstance(value, torch.Tensor) else value
                    for value in values
                ),
            )
        )

    class Indexer:
        def __call__(self, hidden_states, q_norm, positions):
            record("indexer", hidden_states, q_norm, positions)
            return GlmDsaIndexerOutput(
                query=q_norm * 2,
                key=hidden_states[:, :8],
                weights=positions[:, None].float(),
            )

        def from_key_weights(self, q_norm, key_weights):
            record("indexer_ahead", q_norm, key_weights)
            return GlmDsaIndexerOutput(
                query=q_norm * 2, key=key_weights[:, :8], weights=key_weights[:, 8:]
            )

    selection = object()

    def compute_prefill_topk(indexer_output, ctx, num_prefill_tokens):
        record(
            "prefill_topk",
            indexer_output.query,
            indexer_output.key,
            indexer_output.weights,
            num_prefill_tokens,
        )
        return selection

    weight = torch.linspace(-1.0, 1.0, 48).view(6, 8)

    def q_b_proj(q_norm):
        record("q_b_proj", q_norm)
        return q_norm @ weight, None

    def sparse_prefill(positions, q, latent, ctx, locs, output, *, prefill_topk):
        assert prefill_topk is selection
        record("sparse_prefill", positions, q, latent, ctx.input_num_tokens, locs)
        output.copy_(q + latent[:, :8])

    def set_index_k_buffer(layer_id, slots, key, *, write_mask, key_norm):
        record("index_k_write", layer_id, slots, key, write_mask, key_norm)

    core = object.__new__(GlmDsaSparseAttentionMixin)
    core.attn_mqa = SimpleNamespace(layer_id=3)
    core.skip_indexer_topk = False
    core.is_nextn = False
    core.num_local_heads = 2
    core.v_head_dim = 4
    core.indexer = Indexer()
    core._compute_prefill_topk_indices = compute_prefill_topk
    core.q_b_proj = q_b_proj
    core.forward_dsa_sparse_prefill = sparse_prefill
    backend = SimpleNamespace(
        sparse_topk=SimpleNamespace(prefill=None, decode=None),
        write_locations=lambda layer, mode: torch.arange(100, 100 + write_rows),
        cache_placement=lambda layer: None,
    )
    pool = SimpleNamespace(set_index_k_buffer=set_index_k_buffer)
    comm = SimpleNamespace(
        pre_attn_comm=lambda hidden_states, ctx: hidden_states,
        needs_pre_attn_all_gather=lambda: gather,
    )
    return core, backend, pool, comm


def _run_dsa_core(
    calls: list,
    rows: int,
    positions: torch.Tensor,
    *,
    write_rows: int,
    gather: bool,
    valid_rows: int | None,
    ahead: bool,
) -> torch.Tensor:
    """One pure-extend forward of ``_LIVE_ROWS`` live rows through the DSA break.

    It is handed ``rows`` rows (filler past the live ones) with ``valid_rows``
    set as a padded replay sets it, and, when ``ahead``, the index keys and head
    weights projected ahead.
    """
    from tokenspeed.runtime.execution.breakable_cuda_graph import (
        _set_replay_valid_rows,
    )
    from tokenspeed.runtime.execution.context import ForwardContext
    from tokenspeed.runtime.execution.forward_batch_info import ForwardMode
    from tokenspeed.runtime.execution.output_layout import ForwardOutputLayout

    core, backend, pool, comm = _recording_dsa_core(
        calls, write_rows=write_rows, gather=gather
    )
    generator = torch.Generator().manual_seed(7)

    def padded(width: int) -> torch.Tensor:
        live = torch.randn(_LIVE_ROWS, width, generator=generator)
        # The filler a padded replay's earlier segments leave in the tail.
        return torch.cat((live, torch.full((rows - _LIVE_ROWS, width), 1e4)))

    hidden_states, q_norm, latent_cache = padded(16), padded(6), padded(10)
    index_key_weights = padded(9) if ahead else None
    ctx = ForwardContext(
        attn_backend=backend,
        token_to_kv_pool=pool,
        bs=1,
        num_extends=1,
        # A replay pins the bucket here; only valid_rows has the live count.
        input_num_tokens=rows,
        forward_mode=ForwardMode.EXTEND,
        output_layout=ForwardOutputLayout(1, 1, 0, 1),
    )
    _set_replay_valid_rows(valid_rows)
    try:
        return core.forward_dsa_core(
            positions,
            hidden_states,
            q_norm,
            latent_cache,
            ctx,
            comm,
            index_key_weights=index_key_weights,
        )
    finally:
        _set_replay_valid_rows(None)


@pytest.mark.parametrize("ahead", [False, True], ids=["indexer", "projected_ahead"])
def test_dsa_core_narrows_a_padded_replay_to_the_eager_inputs(ahead) -> None:
    """Under a padded prefill-graph replay the DSA break receives the bucket's rows.

    It hands the indexer (or, with index keys and head weights projected ahead,
    the indexer's ``from_key_weights``), the index-K write, the top-k,
    ``q_b_proj`` and the sparse core exactly what the eager forward of the live
    rows hands them, and returns the live rows only (the replay lands them in
    the bucket-shaped handoff and zeroes its tail).
    """
    positions = torch.arange(40, 40 + _LIVE_ROWS)
    # A replay resets the positions of the padded rows to 0.
    padded_positions = torch.cat(
        (positions, torch.zeros(_BUCKET_ROWS - _LIVE_ROWS, dtype=torch.int64))
    )
    eager_calls, replay_calls = [], []
    eager = _run_dsa_core(
        eager_calls,
        _LIVE_ROWS,
        positions,
        write_rows=_LIVE_ROWS,
        gather=False,
        valid_rows=None,
        ahead=ahead,
    )
    replay = _run_dsa_core(
        replay_calls,
        _BUCKET_ROWS,
        padded_positions,
        write_rows=_LIVE_ROWS,
        gather=False,
        valid_rows=_LIVE_ROWS,
        ahead=ahead,
    )

    assert [name for name, _ in eager_calls] == [
        "indexer_ahead" if ahead else "indexer",
        "index_k_write",
        "prefill_topk",
        "q_b_proj",
        "sparse_prefill",
    ]
    assert [name for name, _ in replay_calls] == [name for name, _ in eager_calls]
    for (name, eager_args), (_, replay_args) in zip(eager_calls, replay_calls):
        assert len(eager_args) == len(replay_args), name
        for eager_arg, replay_arg in zip(eager_args, replay_args):
            if isinstance(eager_arg, torch.Tensor):
                assert torch.equal(eager_arg, replay_arg), name
            else:
                assert eager_arg == replay_arg, name
    assert eager.shape == replay.shape == (_LIVE_ROWS, 8)
    assert torch.equal(eager, replay)


@pytest.mark.parametrize(
    "case, match",
    [
        ("positions_2d", "1-D positions"),
        ("gathered_rows", "identity pre-attention comm"),
        ("write_span", "write span 10 != live rows 11"),
    ],
)
def test_dsa_core_refuses_a_padded_replay_it_cannot_narrow(case, match) -> None:
    """A padded replay the break cannot narrow to the eager inputs raises
    before the first cache write."""
    positions = torch.zeros(_BUCKET_ROWS, dtype=torch.int64)
    if case == "positions_2d":
        positions = torch.zeros(3, _BUCKET_ROWS, dtype=torch.int64)
    calls = []
    with pytest.raises(RuntimeError, match=match):
        _run_dsa_core(
            calls,
            _BUCKET_ROWS,
            positions,
            write_rows=_LIVE_ROWS - 1 if case == "write_span" else _LIVE_ROWS,
            gather=case == "gathered_rows",
            valid_rows=_LIVE_ROWS,
            ahead=False,
        )
    assert calls == []


_PADDED_BUCKET = 128
# (slots, prefix lens, new lens, blocks); buckets end over a page past the live rows.
_PADDED_CASES = {
    "prefix_tail_mid_page": ([0], [40], [20], [5]),
    "two_requests": ([0, 1], [0, 0], [20, 15], [5, 6]),
    "prefix_and_fresh": ([0, 1], [40, 0], [20, 15], [5, 6]),
}


def _int32(values, device="cuda") -> torch.Tensor:
    return torch.tensor(values, dtype=torch.int32, device=device)


def _extend_ctx(router, pool, requests, prefix_lens, new_lens, blocks):
    """The router's live extend metadata and the forward context over it."""
    from tokenspeed.runtime.execution.context import ForwardContext
    from tokenspeed.runtime.execution.forward_batch_info import ForwardMode
    from tokenspeed.runtime.execution.output_layout import ForwardOutputLayout

    bs = len(requests)
    seq_lens = [prefix + new for prefix, new in zip(prefix_lens, new_lens)]
    tables = _int32([[block] for block in blocks])
    router.init_forward_metadata(
        bs,
        bs,
        _int32(requests),
        _int32(seq_lens),
        ForwardMode.EXTEND,
        block_tables={FULL_ATTENTION: tables},
        extend_seq_lens=_int32(new_lens),
        extend_seq_lens_cpu=_int32(new_lens, "cpu"),
        extend_prefix_lens=_int32(prefix_lens),
        extend_prefix_lens_cpu=_int32(prefix_lens, "cpu"),
        extend_replay_lens_cpu=_int32([0] * bs, "cpu"),
        extend_prompt_lens_cpu=_int32(seq_lens, "cpu"),
        extend_with_prefix=any(prefix_lens),
        query_shard=None,
        block_tables_cpu={FULL_ATTENTION: tables.cpu()},
    )
    return ForwardContext(
        attn_backend=router,
        token_to_kv_pool=pool,
        bs=bs,
        num_extends=bs,
        input_num_tokens=sum(new_lens),
        forward_mode=ForwardMode.EXTEND,
        output_layout=ForwardOutputLayout(bs, bs, 0, 1),
    )


def _decode_ctx(router, pool, requests, seq_lens, blocks):
    """One decode row per request, ``seq_lens`` counting it."""
    from tokenspeed.runtime.execution.context import ForwardContext
    from tokenspeed.runtime.execution.forward_batch_info import ForwardMode
    from tokenspeed.runtime.execution.output_layout import ForwardOutputLayout

    bs = len(requests)
    router.refresh_decode_metadata(
        bs,
        bs,
        _int32(requests),
        _int32(seq_lens),
        forward_mode=ForwardMode.DECODE,
        block_tables={FULL_ATTENTION: _int32([[block] for block in blocks])},
    )
    return ForwardContext(
        attn_backend=router,
        token_to_kv_pool=pool,
        bs=bs,
        num_extends=0,
        input_num_tokens=bs,
        forward_mode=ForwardMode.DECODE,
        output_layout=ForwardOutputLayout(0, 0, bs, 1),
    )


def _slot_rows(pool) -> tuple[torch.Tensor, torch.Tensor]:
    """Layer 3's latent and index-K bytes, one row per cache slot."""
    latent = pool.get_key_buffer(3)
    num_slots = latent.shape[0]
    index_k = pool.gather_index_k_rows(
        3,
        torch.arange(num_slots, device=latent.device),
        index_k_format="fp8_scaled",
    )
    return latent.view(torch.uint8).reshape(num_slots, -1).clone(), index_k.clone()


def _capture_padded_prefill(router, pool, forward):
    """``forward(positions, hidden, ctx)`` captured as a breakable prefill
    graph over ``_PADDED_BUCKET`` rows, the way ``PrefillGraph`` captures a
    bucket: a dummy one-request extend on block 1, warmed up on the capture
    stream, then captured with the dummy context ambient.

    Returns:
        The capture, its output, and the static position and hidden buffers
        a replay reads.
    """
    from tokenspeed.runtime.execution.breakable_cuda_graph import (
        BreakableCapture,
        active_forward,
    )

    positions = torch.arange(_PADDED_BUCKET, device="cuda")
    hidden = torch.randn(_PADDED_BUCKET, 256, device="cuda", dtype=torch.bfloat16)
    dummy = _extend_ctx(router, pool, [3], [0], [_PADDED_BUCKET], [1])
    capture = BreakableCapture()
    # Warm up where the capture runs: on the capture stream, behind this one's writes.
    capture.stream.wait_stream(torch.cuda.current_stream())
    with torch.no_grad():
        with torch.cuda.stream(capture.stream):
            forward(positions, hidden, dummy)
        torch.cuda.synchronize()
        with active_forward(dummy), capture:
            output = forward(positions, hidden, dummy)
    torch.cuda.synchronize()
    return capture, output, positions, hidden


def _replay_padded_prefill(capture, positions, hidden, ctx, rows_positions, rows):
    """Land the live rows in the static buffers, filler after them, and replay
    with ``ctx`` ambient and pinned to the bucket, as ``PrefillGraph.replay``
    does."""
    import dataclasses

    from tokenspeed.runtime.execution.breakable_cuda_graph import active_forward

    num_rows = rows.shape[0]
    hidden[:num_rows].copy_(rows)
    hidden[num_rows:].fill_(1e4)
    positions[:num_rows].copy_(rows_positions)
    positions[num_rows:].zero_()
    padded = dataclasses.replace(ctx, input_num_tokens=_PADDED_BUCKET)
    with torch.no_grad(), active_forward(padded):
        capture.replay(valid_rows=num_rows)
    torch.cuda.synchronize()


def _live_rows(hidden, prefix_lens, new_lens):
    """The batch's new rows and their positions, request after request."""
    rows = torch.cat(
        [
            states[prefix : prefix + new]
            for states, prefix, new in zip(hidden, prefix_lens, new_lens)
        ]
    )
    positions = torch.cat(
        [
            torch.arange(prefix, prefix + new, device="cuda")
            for prefix, new in zip(prefix_lens, new_lens)
        ]
    )
    return rows, positions


def _prefill_history(attention, comm, router, pool, hidden, prefix_lens, blocks):
    """Eagerly prefill each request's cached prefix into its block."""
    for request, (states, prefix, block) in enumerate(zip(hidden, prefix_lens, blocks)):
        if prefix == 0:
            continue
        ctx = _extend_ctx(router, pool, [request], [0], [prefix], [block])
        with torch.no_grad():
            attention(
                torch.arange(prefix, device="cuda"),
                states[:prefix],
                ctx,
                comm,
                projection_out=None,
            )


@requires_gpu
@pytest.mark.parametrize("case", ["prefix_tail_mid_page", "two_requests"])
@pytest.mark.parametrize(
    "kv_cache_dtype, fraction", [(torch.bfloat16, 1e-2), (torch.float8_e4m3fn, 8e-2)]
)
def test_k31_attention_replays_a_padded_breakable_prefill_capture(
    monkeypatch, case, kv_cache_dtype, fraction
) -> None:
    """The DSA layer captured as a breakable prefill graph (projections
    captured, the DSA break eager) and replayed over a bucket wider than the
    live rows: the live rows match dense MLA, the padded rows stay finite,
    no cache slot outside the live write span changes, and the next eager
    decode step reads the cache the replay wrote."""
    from tokenspeed.runtime.execution.forward_batch_info import ForwardMode
    from tokenspeed.runtime.utils.env import global_server_args_dict

    # The dense delegate sizes its prefix chunks from the launch's chunk size.
    monkeypatch.setitem(global_server_args_dict, "chunked_prefill_size", 8192)
    torch.manual_seed(6)
    heads = 12
    attention, comm, router, pool = _k31_attention_stack(
        kv_cache_dtype=kv_cache_dtype,
        spec_width=1,
        heads=heads,
        index_topk=2048,
        kvp_size=1,
        kvp_rank=0,
    )
    requests, prefix_lens, new_lens, blocks = _PADDED_CASES[case]
    # Each request's prompt plus the token its decode step feeds.
    hidden = [
        torch.randn(prefix + new + 1, 256, device="cuda", dtype=torch.bfloat16)
        for prefix, new in zip(prefix_lens, new_lens)
    ]
    references = [_dense_gated_mla(attention, states, heads) for states in hidden]

    capture, output, static_positions, static_hidden = _capture_padded_prefill(
        router,
        pool,
        lambda positions, states, ctx: attention(
            positions, states, ctx, comm, projection_out=None
        ),
    )
    assert capture.num_segments == 3
    _prefill_history(attention, comm, router, pool, hidden, prefix_lens, blocks)

    live = _extend_ctx(router, pool, requests, prefix_lens, new_lens, blocks)
    live_slots = router.write_locations(attention.attn_mqa, ForwardMode.EXTEND)
    rows, positions = _live_rows(hidden, prefix_lens, new_lens)
    assert live_slots.numel() == rows.shape[0] < _PADDED_BUCKET - 64
    latent_before, index_k_before = _slot_rows(pool)
    _replay_padded_prefill(
        capture, static_positions, static_hidden, live, positions, rows
    )

    assert torch.isfinite(output).all()
    got = output[: rows.shape[0]].split(new_lens)
    for request_rows, reference, prefix, new in zip(
        got, references, prefix_lens, new_lens
    ):
        _assert_close_to_reference(
            request_rows, reference[prefix : prefix + new], fraction
        )
    latent_after, index_k_after = _slot_rows(pool)
    untouched = torch.ones(
        latent_after.shape[0], dtype=torch.bool, device=latent_after.device
    )
    untouched[live_slots.long()] = False
    assert torch.equal(latent_after[untouched], latent_before[untouched])
    assert torch.equal(index_k_after[untouched], index_k_before[untouched])

    # The next decode step attends over the rows the replay wrote.
    seq_lens = [prefix + new + 1 for prefix, new in zip(prefix_lens, new_lens)]
    decode = _decode_ctx(router, pool, requests, seq_lens, blocks)
    with torch.no_grad():
        step = attention(
            _int32([length - 1 for length in seq_lens]).long(),
            torch.stack([states[-1] for states in hidden]),
            decode,
            comm,
            projection_out=None,
        )
    for request, reference in enumerate(references):
        _assert_close_to_reference(
            step[request : request + 1], reference[-1:], fraction
        )


@requires_gpu
@pytest.mark.parametrize(
    "kv_cache_dtype, spec_width, fraction",
    [
        (torch.bfloat16, 1, 1e-2),
        (torch.bfloat16, 4, 1e-2),
        (torch.float8_e4m3fn, 1, 8e-2),
    ],
)
def test_k31_decode_graph_replays_after_a_padded_prefill_capture_and_replay(
    monkeypatch, kv_cache_dtype, spec_width, fraction
) -> None:
    """The aggregated role's order on one DSA layer: a decode graph captured
    first, a breakable padded prefill graph captured after it into its own
    pool, a padded prefill replay of another request, then the decode graph
    replayed for the next verify step. The prefill capture and replay
    rewrite the router's extend metadata and the shared static buffers; none
    of that may reach what the decode graph recorded and the decode refresh
    does not rewrite, so the prefill rows and the decode step after them
    both match dense MLA."""
    from tokenspeed_kernel.workspace import capture_session

    from tokenspeed.runtime.execution.context import ForwardContext
    from tokenspeed.runtime.execution.forward_batch_info import ForwardMode
    from tokenspeed.runtime.execution.output_layout import ForwardOutputLayout
    from tokenspeed.runtime.utils.env import global_server_args_dict

    # The dense delegate sizes its prefix chunks from the launch's chunk size.
    monkeypatch.setitem(global_server_args_dict, "chunked_prefill_size", 8192)
    torch.manual_seed(8)
    heads = 12
    attention, comm, router, pool = _k31_attention_stack(
        kv_cache_dtype=kv_cache_dtype,
        spec_width=spec_width,
        heads=heads,
        index_topk=2048,
        kvp_size=1,
        kvp_rank=0,
    )

    def forward(positions, states, ctx):
        return attention(positions, states, ctx, comm, projection_out=None)

    # Request 0 on block 5 decodes: its prompt, then two verify steps.
    prompt = 40
    total = prompt + 2 * spec_width
    decoding = torch.randn(total, 256, device="cuda", dtype=torch.bfloat16)
    decoding_reference = _dense_gated_mla(attention, decoding, heads)
    _prefill_history(attention, comm, router, pool, [decoding], [prompt], [5])
    request = _int32([0])
    blocks = {FULL_ATTENTION: _int32([[5]])}
    decode = ForwardContext(
        attn_backend=router,
        token_to_kv_pool=pool,
        bs=1,
        num_extends=0,
        input_num_tokens=spec_width,
        forward_mode=ForwardMode.DECODE,
        output_layout=ForwardOutputLayout(0, 0, 1, spec_width),
    )
    rows = slice(prompt, prompt + spec_width)
    step_hidden = decoding[rows].clone()
    step_positions = torch.arange(prompt, prompt + spec_width, device="cuda")
    router.refresh_decode_metadata(
        1,
        1,
        request,
        _int32([prompt + spec_width]),
        forward_mode=ForwardMode.DECODE,
        block_tables=blocks,
    )
    # Run eagerly on the capture stream, like graph warm-up: a core may plan per stream.
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.no_grad(), torch.cuda.stream(stream):
        eager = forward(step_positions, step_hidden, decode)
    torch.cuda.synchronize()
    _assert_close_to_reference(eager, decoding_reference[rows], fraction)
    graph = torch.cuda.CUDAGraph()
    with torch.no_grad(), capture_session(graph), torch.cuda.graph(
        graph, stream=stream
    ):
        decoded = forward(step_positions, step_hidden, decode)

    # The prefill bucket, captured after the decode ladder into its own pool.
    capture, output, static_positions, static_hidden = _capture_padded_prefill(
        router, pool, forward
    )

    # Request 1 on block 6 prefills 20 rows behind a 40-row cached prefix.
    prefix, new = 40, 20
    prefilling = torch.randn(prefix + new, 256, device="cuda", dtype=torch.bfloat16)
    prefilling_reference = _dense_gated_mla(attention, prefilling, heads)
    _prefill_history(
        attention, comm, router, pool, [decoding, prefilling], [0, prefix], [5, 6]
    )
    live = _extend_ctx(router, pool, [1], [prefix], [new], [6])
    live_rows, live_positions = _live_rows([prefilling], [prefix], [new])
    _replay_padded_prefill(
        capture, static_positions, static_hidden, live, live_positions, live_rows
    )
    assert torch.isfinite(output).all()
    _assert_close_to_reference(output[:new], prefilling_reference[prefix:], fraction)

    # Request 0's next verify replays the graph at stable addresses after a refresh.
    rows = slice(prompt + spec_width, total)
    step_hidden.copy_(decoding[rows])
    step_positions.copy_(torch.arange(prompt + spec_width, total, device="cuda"))
    router.refresh_decode_metadata(
        1,
        1,
        request,
        _int32([total]),
        forward_mode=ForwardMode.DECODE,
        block_tables=blocks,
        for_graph_replay=True,
    )
    graph.replay()
    torch.cuda.synchronize()
    _assert_close_to_reference(decoded, decoding_reference[rows], fraction)


@requires_gpu
@pytest.mark.parametrize(
    "kv_cache_dtype, fraction", [(torch.bfloat16, 1e-2), (torch.float8_e4m3fn, 8e-2)]
)
def test_two_k31_dsa_breaks_in_one_padded_capture_match_eager(
    monkeypatch, kv_cache_dtype, fraction
) -> None:
    """Two DSA breaks of one capture, a captured token-shaped op between
    them, hand their equal-shape outputs through one shared handoff buffer.
    A padded replay matches the same composite run eagerly over the live
    rows on the same cache state."""
    from tokenspeed.runtime.utils.env import global_server_args_dict

    monkeypatch.setitem(global_server_args_dict, "chunked_prefill_size", 8192)
    torch.manual_seed(7)
    heads = 12
    attention, comm, router, pool = _k31_attention_stack(
        kv_cache_dtype=kv_cache_dtype,
        spec_width=1,
        heads=heads,
        index_topk=2048,
        kvp_size=1,
        kvp_rank=0,
    )
    requests, prefix_lens, new_lens, blocks = _PADDED_CASES["prefix_and_fresh"]
    hidden = [
        torch.randn(prefix + new, 256, device="cuda", dtype=torch.bfloat16)
        for prefix, new in zip(prefix_lens, new_lens)
    ]

    def composite(positions, states, ctx):
        first = attention(positions, states, ctx, comm, projection_out=None)
        return attention(positions, states + first, ctx, comm, projection_out=None)

    capture, output, static_positions, static_hidden = _capture_padded_prefill(
        router, pool, composite
    )
    # Two breaks, each between captured segments, sharing one handoff.
    assert capture.num_segments == 5
    assert len(capture._handoff) == 1
    _prefill_history(attention, comm, router, pool, hidden, prefix_lens, blocks)

    latent = pool.get_key_buffer(3).clone()
    index_k = pool.get_index_k_buffer(3).clone()
    live = _extend_ctx(router, pool, requests, prefix_lens, new_lens, blocks)
    rows, positions = _live_rows(hidden, prefix_lens, new_lens)
    _replay_padded_prefill(
        capture, static_positions, static_hidden, live, positions, rows
    )
    replayed = output[: rows.shape[0]].clone()
    assert torch.isfinite(output).all()

    pool.get_key_buffer(3).copy_(latent)
    pool.get_index_k_buffer(3).copy_(index_k)
    live = _extend_ctx(router, pool, requests, prefix_lens, new_lens, blocks)
    with torch.no_grad():
        eager = composite(positions, rows, live)
    _assert_close_to_reference(replayed, eager.float(), fraction)
