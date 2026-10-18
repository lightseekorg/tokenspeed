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

"""Side caches share decoder storage without changing their allocation groups."""

import sys
from dataclasses import replace
from pathlib import Path

import pytest
import torch
from tokenspeed_kernel.ops.attention.qsa import (
    qsa_block_scores,
    qsa_block_topk_select,
    qsa_block_topk_stream,
)
from tokenspeed_kernel.ops.attention.qsa.triton import (
    qwen4_exp_qsa_compress_and_store,
    qwen4_exp_qsa_recent_write,
)
from tokenspeed_kernel.ops.ple import ple_page_gather_pair, ple_page_scatter

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from ci_system.ci_register import register_cuda_ci

from tokenspeed.runtime.configs.model_config import ModelConfig
from tokenspeed.runtime.configs.qwen4_exp_config import Qwen4ExpTextConfig
from tokenspeed.runtime.layers.attention.configs.base import AttnConfig
from tokenspeed.runtime.layers.attention.configs.linear_attn import LinearAttnConfig
from tokenspeed.runtime.layers.attention.configs.mha import MHAConfig
from tokenspeed.runtime.layers.attention.kv_cache.arena import CacheArena
from tokenspeed.runtime.layers.attention.kv_cache.qwen4_exp import (
    QWEN4_EXP_PLE_CACHE_GROUP,
    QWEN4_EXP_QSA_CACHE_GROUP,
    QWEN4_EXP_QSA_RECENT_CACHE_GROUP,
    qsa_compressed_field,
    qsa_raw_key_field,
    qsa_rope_position_field,
    qwen4_exp_ple_context_field,
    qwen4_exp_ple_conv_field,
)
from tokenspeed.runtime.layers.attention.kv_cache.recipes.plan import pack
from tokenspeed.runtime.layers.attention.kv_cache.recipes.qwen4_exp import (
    Qwen4ExpRecipe,
)
from tokenspeed.runtime.layers.attention.kv_cache.recipes.spec import (
    FULL_ATTENTION,
    LINEAR_ATTENTION,
)
from tokenspeed.runtime.utils.server_args import ServerArgs

register_cuda_ci(est_time=30, suite="runtime-1gpu")


def _recipe(
    tmp_path: Path,
    *,
    tp: int,
    draft_layers: int,
    has_gdn: bool,
    kv_dtype: torch.dtype,
) -> Qwen4ExpRecipe:
    """Build a small synthetic checkpoint using real cache/config objects."""
    layer_types = (
        ((LINEAR_ATTENTION,) * 3 + (FULL_ATTENTION,)) * 4
        if has_gdn
        else (FULL_ATTENTION,) * 16
    )
    text_kwargs = dict(
        hidden_size=512,
        num_attention_heads=8,
        num_key_value_heads=2,
        head_dim=128,
        linear_num_key_heads=4,
        linear_num_value_heads=8,
        linear_key_head_dim=128,
        linear_value_head_dim=128,
        linear_conv_kernel_dim=4,
        ple_layer_ids=[2],
        ple_offload_embedding=False,
        indexer_n_heads=4,
        indexer_head_dim=128,
        indexer_compress_ratio=4,
    )
    text = Qwen4ExpTextConfig(
        architectures=["Qwen4ExpForCausalLM"],
        num_hidden_layers=len(layer_types),
        layer_types=list(layer_types),
        **text_kwargs,
    )
    model_path = tmp_path / "target"
    text.save_pretrained(model_path)
    width = 3 if draft_layers else 1
    server_args = ServerArgs(
        model=str(model_path),
        prefix_granularity=256,
        attn_tp_size=tp,
        speculative_num_draft_tokens=width,
        enable_replay_ssm=False,
    )
    model_config = ModelConfig(
        str(model_path),
        trust_remote_code=False,
        model_override_args="{}",
        dtype="bfloat16",
        server_args=server_args,
    )
    draft_model_config = None
    if draft_layers:
        draft_path = tmp_path / "draft"
        draft_text = Qwen4ExpTextConfig(
            architectures=["Qwen4ExpForCausalLMNextN"],
            num_hidden_layers=draft_layers,
            layer_types=[FULL_ATTENTION] * draft_layers,
            **(text_kwargs | {"ple_layer_ids": []}),
        )
        draft_text.save_pretrained(draft_path)
        draft_model_config = ModelConfig(
            str(draft_path),
            trust_remote_code=False,
            model_override_args="{}",
            dtype="bfloat16",
            server_args=server_args,
            is_draft_worker=True,
        )
    softmax = MHAConfig(
        backend_name="qsa",
        num_attention_heads=text.num_attention_heads,
        num_kv_heads=text.num_key_value_heads,
        head_dim=text.head_dim,
        attn_tp_size=tp,
        cache_layer_types=layer_types,
        sliding_window_tokens=None,
    )
    components = (softmax,)
    if has_gdn:
        components += (
            LinearAttnConfig(
                num_k_heads=text.linear_num_key_heads,
                num_v_heads=text.linear_num_value_heads,
                head_k_dim=text.linear_key_head_dim,
                head_v_dim=text.linear_value_head_dim,
                conv_kernel_size=text.linear_conv_kernel_dim,
                layer_ids=tuple(
                    i for i, kind in enumerate(layer_types) if kind == LINEAR_ATTENTION
                ),
                tp_size=tp,
                replay_ssm=False,
                draft_tree=False,
            ),
        )
    config = AttnConfig(
        device="cpu",
        dtype=torch.bfloat16,
        kv_cache_dtype=kv_dtype,
        kv_cache_quant_method="none",
        kv_cache_mxfp8=False,
        prefix_granularity=256,
        kernel_page_size=64,
        context_len=4096,
        max_bs=2,
        pd_disaggregation_enabled=False,
        speculative_num_steps=2 if draft_layers else 0,
        speculative_num_draft_tokens=width,
        is_draft=False,
        draft_block_decode=False,
        components=components,
    )
    return Qwen4ExpRecipe(
        server_args=server_args,
        model_config=model_config,
        attn_config=config,
        draft_model_config=draft_model_config,
        draft_attn_config=(
            replace(
                config,
                is_draft=True,
                components=(replace(softmax, cache_layer_types=(FULL_ATTENTION,)),),
            )
            if draft_layers
            else None
        ),
        cache_budget_bytes=1 << 30,
        probe_batch_rows=None,
        decode_input_tokens=1,
        overlap_schedule_depth=0,
    )


def _layout(recipe):
    groups = recipe.groups()
    layout = pack(
        groups,
        prefix_granularity=recipe.prefix_granularity,
        cache_blocks_per_lcm_block=recipe.packing(groups),
        alignment=recipe.alignment,
        max_padding_fraction=recipe.max_padding_fraction,
    )
    recipe.check_layout(layout)
    return groups, layout


@pytest.mark.parametrize("tp", [1, 4])
@pytest.mark.parametrize("ssm_dtype", ["bfloat16", "float32"])
@pytest.mark.parametrize("draft_layers", [0, 1])
@pytest.mark.parametrize("kv_dtype", [torch.bfloat16, torch.float8_e4m3fn])
def test_side_caches_reuse_decoder_planes(
    monkeypatch, tmp_path, tp, ssm_dtype, draft_layers, kv_dtype
) -> None:
    monkeypatch.setenv("TOKENSPEED_MAMBA_SSM_DTYPE", ssm_dtype)
    recipe = _recipe(
        tmp_path, tp=tp, draft_layers=draft_layers, has_gdn=True, kv_dtype=kv_dtype
    )
    groups, layout = _layout(recipe)
    plan = layout.bind(2)
    assert len(plan.planes) == 8 + 2 * draft_layers
    assert layout.lcm_block_bytes == sum(size for _, size in layout.plane_bytes)
    for _, fields in groups:
        for field in fields:
            bound = plan.field(field.field_id)
            if field.exact_page_stride:
                assert bound.page_stride_bytes == field.payload_bytes
    for layer in (3, 15, *((16,) if draft_layers else ())):
        key = plan.field(f"layer.{layer}.k")
        value = plan.field(f"layer.{layer}.v")
        compressed = plan.field(qsa_compressed_field(layer))
        raw = plan.field(qsa_raw_key_field(layer))
        positions = plan.field(qsa_rope_position_field(layer))
        assert compressed.group_id == QWEN4_EXP_QSA_CACHE_GROUP
        assert raw.group_id == positions.group_id == QWEN4_EXP_QSA_RECENT_CACHE_GROUP
        assert compressed.plane_id == raw.plane_id == key.plane_id
        assert positions.plane_id == value.plane_id
    conv = plan.field(qwen4_exp_ple_conv_field(1))
    context = plan.field(qwen4_exp_ple_context_field(1))
    assert conv.group_id == context.group_id == QWEN4_EXP_PLE_CACHE_GROUP
    assert conv.plane_id == plan.field("layer.1.ssm").plane_id
    assert context.plane_id == plan.field("layer.1.conv").plane_id


@pytest.mark.parametrize("tp", [1, 4])
@pytest.mark.parametrize("kv_dtype", [torch.bfloat16, torch.float8_e4m3fn])
def test_full_attention_planes_fit_one_ple_state(tmp_path, tp, kv_dtype) -> None:
    recipe = _recipe(tmp_path, tp=tp, draft_layers=0, has_gdn=False, kv_dtype=kv_dtype)
    _, layout = _layout(recipe)
    plan = layout.bind(2)
    assert len(plan.planes) == 32
    assert layout.lcm_block_bytes == sum(size for _, size in layout.plane_bytes)
    conv = plan.field(qwen4_exp_ple_conv_field(1))
    assert conv.page_stride_bytes >= conv.payload_bytes
    assert conv.plane_id == plan.field("layer.1.k").plane_id


@pytest.mark.parametrize(
    "target_dtype,draft_dtype",
    [(torch.bfloat16, torch.float8_e4m3fn), (torch.float8_e4m3fn, torch.bfloat16)],
)
def test_mtp_compressed_cache_can_use_another_decoder_plane(
    monkeypatch, tmp_path, target_dtype, draft_dtype
) -> None:
    monkeypatch.setenv("TOKENSPEED_MAMBA_SSM_DTYPE", "bfloat16")
    recipe = _recipe(
        tmp_path, tp=1, draft_layers=1, has_gdn=True, kv_dtype=target_dtype
    )
    recipe.draft_attn_config = replace(
        recipe.draft_attn_config, kv_cache_dtype=draft_dtype
    )
    _, layout = _layout(recipe)
    plan = layout.bind(2)
    assert len(plan.planes) == 10
    compressed = plan.field(qsa_compressed_field(16))
    assert compressed.plane_id != plan.field("layer.16.k").plane_id
    assert compressed.page_stride_bytes == compressed.payload_bytes


def test_incompatible_index_width_keeps_an_exact_stride_plane(
    monkeypatch, tmp_path
) -> None:
    monkeypatch.setenv("TOKENSPEED_MAMBA_SSM_DTYPE", "bfloat16")
    recipe = _recipe(
        tmp_path, tp=1, draft_layers=1, has_gdn=True, kv_dtype=torch.bfloat16
    )
    recipe.draft_model_config.hf_text_config.indexer_head_dim = 64
    _, layout = _layout(recipe)
    plan = layout.bind(2)
    assert len(plan.planes) == 11
    compressed = plan.field(qsa_compressed_field(16))
    assert compressed.shape == (64, 1, 64)
    assert compressed.page_stride_bytes == compressed.payload_bytes


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires a GPU")
@pytest.mark.parametrize("tp", [1, 4])
@pytest.mark.parametrize("enable_pdl", [False, True])
def test_shared_planes_preserve_live_pages_and_mask_null_rows(
    monkeypatch, tmp_path, tp, enable_pdl
) -> None:
    monkeypatch.setenv("TOKENSPEED_MAMBA_SSM_DTYPE", "bfloat16")
    recipe = _recipe(
        tmp_path, tp=tp, draft_layers=1, has_gdn=True, kv_dtype=torch.bfloat16
    )
    groups, layout = _layout(recipe)
    plan = layout.bind(5)
    arena = CacheArena(
        plan, "cuda", cache_group_specs=tuple(spec for spec, _ in groups)
    )

    def page(group_id, parent):
        return 1 + (parent - 1) * plan.group(group_id).cache_blocks_per_lcm_block

    # Distinct owners: parent 1 = KV, 2 = PLE, 3 = compressed keys,
    # 4 = recent keys, 5 = the first GDN group. Poison the shared null parent.
    ssm = arena.field("layer.0.ssm")
    ssm[0].fill_(float("nan"))
    ssm[5].fill_(0.75)
    key = arena.field("layer.3.k")
    value = arena.field("layer.3.v")
    key[256:512].fill_(0.25)
    value[256:512].fill_(0.5)

    context = arena.field(qwen4_exp_ple_context_field(1))
    conv = arena.field(qwen4_exp_ple_conv_field(1))
    ple_page = page(QWEN4_EXP_PLE_CACHE_GROUP, 2)
    pages = torch.tensor([ple_page, 0, -1], device="cuda", dtype=torch.int32)
    context_values = torch.arange(6, device="cuda", dtype=torch.int64).view(3, 2)
    conv_values = torch.randn((3, *conv.shape[1:]), device="cuda", dtype=conv.dtype)
    context_null, conv_null = context[0].clone(), conv[0].clone()
    ple_page_scatter(context, pages, context_values, context.stride(0))
    ple_page_scatter(conv, pages, conv_values, conv.stride(0))
    gathered_context, gathered_conv = ple_page_gather_pair(
        context, conv, pages, context.stride(0), conv.stride(0), 99
    )
    torch.testing.assert_close(gathered_context[0], context_values[0], atol=0, rtol=0)
    torch.testing.assert_close(gathered_conv[0], conv_values[0], atol=0, rtol=0)
    assert bool((gathered_context[1:] == 99).all())
    assert bool((gathered_conv[1:] == 0).all())
    torch.testing.assert_close(context[0], context_null, atol=0, rtol=0)
    torch.testing.assert_close(conv[0], conv_null, atol=0, rtol=0, equal_nan=True)

    raw = arena.field(qsa_raw_key_field(3))
    compressed = arena.field(qsa_compressed_field(3))
    positions_cache = arena.field(qsa_rope_position_field(3))
    # NaN padding must leave finite null sentinels untouched. A NaN sentinel
    # alone would hide an invalid store that also writes NaN.
    raw[0].fill_(7)
    compressed[0].fill_(5)
    reference = tuple(
        torch.zeros_like(field) for field in (raw, compressed, positions_cache)
    )
    reference[0][0].fill_(7)
    reference[1][0].fill_(5)
    raw_null, compressed_null = raw[0].clone(), compressed[0].clone()
    raw_page = page(QWEN4_EXP_QSA_RECENT_CACHE_GROUP, 4)
    compressed_page = page(QWEN4_EXP_QSA_CACHE_GROUP, 3)
    positions = torch.tensor([0, 1, 2, 3, 4, 5, 6, 7, 3], device="cuda")
    requests = torch.tensor([0] * 8 + [1], device="cuda")
    keys = torch.randn(9, 1, 128, device="cuda", dtype=torch.bfloat16)
    keys[-1].fill_(float("nan"))
    recent_locs = raw_page * 64 + positions.to(torch.int32)
    compressed_locs = compressed_page * 256 + positions.to(torch.int32)
    recent_locs[-1] = compressed_locs[-1] = 0
    rope_positions = positions[:, None].expand(-1, 3)
    norm_weight = torch.ones(128, device="cuda")
    cos_sin = torch.zeros(16, 128, device="cuda")
    cos_sin[:, :64] = 1
    for raw_field, compressed_field, position_field in (
        (raw, compressed, positions_cache),
        reference,
    ):
        # A split inside a compression group forces the second call to read
        # committed keys through the field's padded page stride.
        for start, end in ((0, 2), (2, 9)):
            qwen4_exp_qsa_compress_and_store(
                keys[start:end],
                positions[start:end],
                requests[start:end],
                recent_locs[start:end],
                raw_field,
                rope_positions[start:end],
                position_field,
                norm_weight,
                1e-6,
                cos_sin,
                compressed_locs[start:end],
                compressed_field,
                64,
                4,
                256,
                enable_pdl=enable_pdl,
                stage_verify_buffers=None,
                stage_draft=False,
            )
            qwen4_exp_qsa_recent_write(
                keys[start:end],
                positions[start:end],
                requests[start:end],
                recent_locs[start:end],
                rope_positions[start:end],
                raw_field,
                position_field,
                64,
                4,
                enable_pdl=enable_pdl,
            )
    torch.testing.assert_close(raw[raw_page], keys[4:8], atol=0, rtol=0)
    torch.testing.assert_close(
        compressed[compressed_page], reference[1][compressed_page], atol=0, rtol=0
    )
    torch.testing.assert_close(
        positions_cache[raw_page], reference[2][raw_page], atol=0, rtol=0
    )
    torch.testing.assert_close(raw[0], raw_null, atol=0, rtol=0)
    torch.testing.assert_close(compressed[0], compressed_null, atol=0, rtol=0)
    # Separately poison null keys for the score/selection readers.
    compressed[0].fill_(float("nan"))
    reference[1][0].fill_(float("nan"))
    query = torch.randn(2, 4, 128, device="cuda", dtype=torch.bfloat16)
    table = torch.tensor([[compressed_page], [0]], device="cuda", dtype=torch.int32)
    query_requests = torch.tensor([0, 1], device="cuda", dtype=torch.int64)
    complete = torch.tensor([2, 0], device="cuda", dtype=torch.int32)
    for algorithm in ("stream", "logits"):
        selected = []
        for field in (compressed, reference[1]):
            if algorithm == "stream":
                output = qsa_block_topk_stream(
                    query,
                    field.view(-1, 1, 128),
                    table,
                    query_requests,
                    complete,
                    page_size=64,
                    block_topk=64,
                    max_partial_bytes=1 << 20,
                    enable_pdl=enable_pdl,
                    override=None,
                    solution="triton",
                )
            else:
                scores = qsa_block_scores(
                    query,
                    field.view(-1, 1, 128),
                    table,
                    query_requests,
                    complete,
                    page_size=64,
                    queries_per_request=None,
                    enable_pdl=enable_pdl,
                    override=None,
                    solution="triton",
                )
                output = qsa_block_topk_select(
                    scores,
                    64,
                    lengths=complete,
                    enable_pdl=enable_pdl,
                    override=None,
                    solution="triton",
                )
            selected.append(output)
        torch.testing.assert_close(selected[0], selected[1], atol=0, rtol=0)
        assert bool((selected[0][1] == -1).all())
    assert bool((key[256:512] == 0.25).all())
    assert bool((value[256:512] == 0.5).all())
    assert bool((ssm[5] == 0.75).all())
    torch.testing.assert_close(conv[ple_page], conv_values[0], atol=0, rtol=0)
