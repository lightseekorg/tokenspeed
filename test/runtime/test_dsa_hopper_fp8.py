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

"""Hopper DSA config and arena use the packed cache consumed by FlashMLA."""

from dataclasses import replace
from test.ci_system.ci_register import register_cuda_ci

import pytest
import torch
from tokenspeed_kernel.platform import ArchVersion, current_platform
from transformers import GlmMoeDsaConfig

from tokenspeed.runtime.configs.model_config import ModelConfig
from tokenspeed.runtime.distributed.mapping import Mapping
from tokenspeed.runtime.execution.context import ForwardContext
from tokenspeed.runtime.execution.forward_batch_info import ForwardMode
from tokenspeed.runtime.execution.output_layout import ForwardOutputLayout
from tokenspeed.runtime.layers.attention.configs import dsa
from tokenspeed.runtime.layers.attention.kv_cache.arena import CacheArena
from tokenspeed.runtime.layers.attention.kv_cache.mla import MLATokenToKVPool
from tokenspeed.runtime.layers.attention.kv_cache.recipes.ordinary import OrdinaryRecipe
from tokenspeed.runtime.models.glm5 import GlmMoeDsaAttention
from tokenspeed.runtime.utils.env import global_server_args_dict
from tokenspeed.runtime.utils.server_args import ServerArgs

register_cuda_ci(est_time=5, suite="runtime-1gpu")


def _generate(
    tmp_path,
    monkeypatch,
    *,
    arch,
    rope_dim,
    rank,
    dtype,
    kv_dtype,
    quant_method,
):
    # A local config exercises the real model parser without downloading weights.
    hf = GlmMoeDsaConfig(
        num_hidden_layers=2,
        num_attention_heads=8,
        num_key_value_heads=1,
        kv_lora_rank=rank,
        qk_rope_head_dim=rope_dim,
        qk_nope_head_dim=128,
        qk_head_dim=128 + rope_dim,
        architectures=["GlmMoeDsaForCausalLM"],
    )
    hf.save_pretrained(tmp_path)
    args = ServerArgs(
        model=str(tmp_path),
        dtype=dtype,
        kv_cache_dtype=kv_dtype,
        kv_cache_quant_method=quant_method,
        max_num_seqs=4,
    )
    args.mapping = Mapping(rank=0, world_size=1)
    model = ModelConfig(
        str(tmp_path), dtype=dtype, model_override_args="{}", server_args=args
    )
    # Only the GPU identity is replaced; all model/config/layout classes are real.
    hardware = replace(
        current_platform(), vendor="nvidia", arch_version=ArchVersion(arch, 0)
    )
    monkeypatch.setattr(dsa, "current_platform", lambda: hardware)
    config = dsa.DSAConfig.generate(args, model)
    return args, model, config


def _pool(args, model, config, *, device):
    spec = config.component(dsa.DSAConfig)
    recipe = OrdinaryRecipe(
        family="dsa",
        server_args=args,
        model_config=model,
        attn_config=config,
        draft_model_config=None,
        draft_attn_config=None,
        cache_budget_bytes=1 << 20,
        probe_batch_rows=None,
        decode_input_tokens=1,
        overlap_schedule_depth=0,
    )
    setup = recipe.setup()
    arena = CacheArena(
        setup.spec.memory_plan,
        device=device,
        cache_group_specs=setup.spec.cache_group_specs,
    )
    pool = MLATokenToKVPool(
        arena,
        torch.bfloat16,
        config.kv_cache_dtype,
        config.kv_cache_quant_method,
        512,
        spec.qk_rope_head_dim,
        2,
        0,
    )
    return setup, pool


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA RMSNorm required")
@pytest.mark.parametrize("kv_dtype", ["bfloat16", "fp8"])
@torch.inference_mode()
def test_glm_only_defers_latent_norm_for_the_packed_cache(
    tmp_path, monkeypatch, kv_dtype
):
    args, model, config = _generate(
        tmp_path,
        monkeypatch,
        arch=9,
        rope_dim=64,
        rank=512,
        dtype="bfloat16",
        kv_dtype=kv_dtype,
        quant_method="none",
    )
    _, pool = _pool(args, model, config, device="cuda")
    monkeypatch.setitem(global_server_args_dict, "mapping", args.mapping)
    monkeypatch.setitem(global_server_args_dict, "attention_backend", "dsa")
    attention = GlmMoeDsaAttention(
        config=model.hf_text_config,
        mapping=args.mapping,
        hidden_size=16,
        num_heads=8,
        qk_nope_head_dim=128,
        qk_rope_head_dim=64,
        v_head_dim=128,
        q_lora_rank=64,
        kv_lora_rank=512,
        layer_id=0,
    ).to(device="cuda", dtype=torch.bfloat16)
    attention.kv_a_layernorm.weight.fill_(1.1875)
    q = torch.randn((3, 64), device="cuda", dtype=torch.bfloat16)
    latent = torch.randn((3, 576), device="cuda", dtype=torch.bfloat16)
    before = latent.clone()
    ctx = ForwardContext(
        attn_backend=None,
        token_to_kv_pool=pool,
        bs=3,
        num_extends=0,
        input_num_tokens=3,
        forward_mode=ForwardMode.DECODE,
        output_layout=ForwardOutputLayout(0, 0, 3, 1),
    )
    normalized_q = attention._norm_qkv(q, latent, ctx)
    q_float = q.float()
    expected_q = q_float * torch.rsqrt(q_float.square().mean(-1, keepdim=True) + 1e-6)
    torch.testing.assert_close(
        normalized_q, expected_q.bfloat16(), atol=1e-2, rtol=1e-2
    )
    if kv_dtype == "fp8":
        torch.testing.assert_close(latent, before, atol=0, rtol=0)
    else:
        kv = before[:, :512].float()
        expected_kv = (
            kv * torch.rsqrt(kv.square().mean(-1, keepdim=True) + 1e-6) * 1.1875
        )
        torch.testing.assert_close(
            latent[:, :512], expected_kv.bfloat16(), atol=1e-2, rtol=1e-2
        )
        torch.testing.assert_close(latent[:, 512:], before[:, 512:], atol=0, rtol=0)


@pytest.mark.parametrize("rope_dim", [0, 64])
def test_hopper_fp8_arena_matches_row_size_and_returns_raw_bytes(
    tmp_path, monkeypatch, rope_dim
):
    args, model, config = _generate(
        tmp_path,
        monkeypatch,
        arch=9,
        rope_dim=rope_dim,
        rank=512,
        dtype="bfloat16",
        kv_dtype="fp8",
        quant_method="none",
    )
    assert config.kv_cache_quant_method == "flashmla"
    spec = config.component(dsa.DSAConfig)
    assert spec.cache_cell_size(config) == 656 + dsa.index_k_row_bytes(
        spec.index_head_dim, spec.index_k_format
    )
    setup, pool = _pool(args, model, config, device="cpu")
    rows = pool.get_key_buffer(0)
    assert rows.dtype == torch.uint8 and rows.shape[-1] == 656
    assert rows.numel() == rows.shape[0] * 656
    field = next(
        f for f in setup.spec.memory_plan.fields if f.field_id == "layer.0.latent_kv"
    )
    assert field.shape == (64, 1, 656)
    assert field.payload_bytes == 64 * 656
    target = pool.kv_write_target(0, torch.tensor([1]), None)
    assert target.kv_cache.data_ptr() == rows.data_ptr()
    assert target.format.value == "fp8_flashmla"
    with pytest.raises(ValueError, match="requires sparse attention"):
        pool.get_value_buffer(0)


@pytest.mark.parametrize(
    "rope_dim,rank,dtype,kv_dtype,quant_method",
    [
        (128, 512, "bfloat16", "fp8", "none"),
        (64, 256, "bfloat16", "fp8", "none"),
        (64, 512, "float16", "fp8", "none"),
        (64, 512, "bfloat16", "fp8_e5m2", "none"),
        (64, 512, "bfloat16", "fp8", "per_token_head"),
    ],
)
def test_hopper_rejects_incompatible_formats(
    tmp_path, monkeypatch, rope_dim, rank, dtype, kv_dtype, quant_method
):
    message = (
        "Unsupported kv_cache_dtype"
        if kv_dtype == "fp8_e5m2"
        else "Hopper DSA FP8 requires"
    )
    with pytest.raises(ValueError, match=message):
        _generate(
            tmp_path,
            monkeypatch,
            arch=9,
            rope_dim=rope_dim,
            rank=rank,
            dtype=dtype,
            kv_dtype=kv_dtype,
            quant_method=quant_method,
        )


@pytest.mark.parametrize("arch,kv_dtype", [(9, "bfloat16"), (10, "fp8")])
def test_existing_cache_formats_keep_their_storage(
    tmp_path, monkeypatch, arch, kv_dtype
):
    _, _, config = _generate(
        tmp_path,
        monkeypatch,
        arch=arch,
        rope_dim=64,
        rank=512,
        dtype="bfloat16",
        kv_dtype=kv_dtype,
        quant_method="none",
    )
    spec = config.component(dsa.DSAConfig)
    assert config.kv_cache_quant_method == "none"
    assert spec.cache_storage_dim(config) == 576


@pytest.mark.skipif(not current_platform().is_hopper, reason="Hopper FlashMLA runtime")
@pytest.mark.parametrize("rope_dim", [0, 64])
def test_runtime_prefill_and_verify_read_the_prologue_packed_cache(
    tmp_path, monkeypatch, rope_dim
):
    from tokenspeed_kernel.ops.attention.prologue import mla_prologue

    from tokenspeed.runtime.layers.attention.backends.paged.dsa import DSABackend
    from tokenspeed.runtime.layers.paged_attention import PagedAttention
    from tokenspeed.runtime.utils.env import global_server_args_dict

    args, model, config = _generate(
        tmp_path,
        monkeypatch,
        arch=9,
        rope_dim=rope_dim,
        rank=512,
        dtype="bfloat16",
        kv_dtype="fp8",
        quant_method="none",
    )
    config = replace(config, context_len=128, speculative_num_draft_tokens=2)
    spec = config.component(dsa.DSAConfig)
    monkeypatch.setitem(global_server_args_dict, "numerics", args.numerics)
    monkeypatch.setitem(global_server_args_dict, "dsa_slot_order", args.dsa_slot_order)
    backend = DSABackend(config, spec, kernel_page_size=64)
    _, pool = _pool(args, model, config, device="cuda")
    layer = PagedAttention(
        num_heads=8,
        head_dim=512 + rope_dim,
        scaling=spec.scaling,
        num_kv_heads=1,
        layer_id=0,
        v_head_dim=512,
        rotary_emb=None,
        qk_norm=None,
    )
    slots = torch.tensor([1, 2], dtype=torch.int64, device="cuda")
    latent = torch.full((2, 512 + rope_dim), 2, dtype=torch.bfloat16, device="cuda")
    latent[1] = 3
    query = torch.randn((2, 8, 512 + rope_dim), dtype=torch.bfloat16, device="cuda")
    prepared = mla_prologue(
        query,
        query[..., 512:],
        latent,
        expanded=None,
        rotary=None,
        cache=pool.kv_write_target(0, slots, None),
        solution=None,
        override=None,
    )
    selected = torch.full((2, 2048), -1, dtype=torch.int32, device="cuda")
    selected[:, 0] = slots
    lengths = torch.ones(2, dtype=torch.int32, device="cuda")
    prefill = backend.forward_sparse_prefill(
        q=prepared.query,
        layer=layer,
        token_to_kv_pool=pool,
        topk_slots=selected,
        topk_lens=lengths,
        kv_seq_lens=None,
        max_seq_len=3,
    )
    expected = latent[:, :512].unsqueeze(1).expand(-1, 8, -1).reshape(2, -1)
    torch.testing.assert_close(prefill, expected, atol=0, rtol=0)
    backend.init_cuda_graph_state(2)
    backend.refresh_decode_metadata(
        1,
        1,
        torch.tensor([3], dtype=torch.int32, device="cuda"),
        torch.zeros((1, 2), dtype=torch.int32, device="cuda"),
        num_extends=0,
        for_graph_replay=False,
    )
    verify = backend.forward_sparse_decode(
        q=prepared.query,
        layer=layer,
        token_to_kv_pool=pool,
        bs=1,
        topk_indices=selected,
        topk_lens=lengths,
    )
    torch.testing.assert_close(verify, expected, atol=0, rtol=0)
