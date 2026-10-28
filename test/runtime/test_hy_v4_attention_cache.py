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

"""HY4 model forward applies latent normalization once before cache reads."""

from test.ci_system.ci_register import register_cuda_ci
from test.runtime.test_dsa_hopper_fp8 import _pool
from test.runtime.test_hy_v4 import _config, runtime_settings

import pytest
import torch
from tokenspeed_kernel.platform import current_platform

from tokenspeed.runtime.configs.model_config import AttentionArch, ModelConfig
from tokenspeed.runtime.distributed.comm_manager import CommManager
from tokenspeed.runtime.distributed.mapping import Mapping
from tokenspeed.runtime.execution.context import ForwardContext
from tokenspeed.runtime.execution.forward_batch_info import ForwardMode
from tokenspeed.runtime.execution.output_layout import ForwardOutputLayout
from tokenspeed.runtime.layers.attention.configs.dsa import DSAConfig
from tokenspeed.runtime.layers.attention.registry import create_paged_router
from tokenspeed.runtime.layers.paged_attention import bind_cache_groups
from tokenspeed.runtime.models.glm5 import GlmDsaDecodeTopK
from tokenspeed.runtime.models.hy_v4 import HYV4Attention
from tokenspeed.runtime.utils.env import global_server_args_dict
from tokenspeed.runtime.utils.server_args import ServerArgs

register_cuda_ci(est_time=10, suite="runtime-1gpu")


@pytest.mark.skipif(not current_platform().is_hopper, reason="Hopper FlashMLA required")
@pytest.mark.parametrize("kv_dtype", ["bfloat16", "fp8"])
@torch.inference_mode()
def test_attention_forward_normalizes_latent_once(
    tmp_path, monkeypatch, runtime_settings, kv_dtype
):
    config = _config(
        architectures=["HYV4ForCausalLM"],
        num_hidden_layers=2,
        num_attention_heads=8,
        num_key_value_heads=8,
        q_lora_rank=64,
        kv_lora_rank=512,
        qk_nope_head_dim=128,
        qk_rope_head_dim=64,
        v_head_dim=128,
        index_head_dim=128,
        index_topk=2048,
        max_position_embeddings=256,
        mlp_layer_types=["dense", "dense"],
        indexer_types=["full", "shared"],
        gated_mla=False,
        num_nextn_predict_layers=0,
    )
    config.save_pretrained(tmp_path)
    args = ServerArgs(
        model=str(tmp_path),
        dtype="bfloat16",
        kv_cache_dtype=kv_dtype,
        kv_cache_quant_method="none",
        max_num_seqs=4,
    )
    args.mapping = Mapping(rank=0, world_size=1)
    model = ModelConfig(
        str(tmp_path), dtype="bfloat16", model_override_args="{}", server_args=args
    )
    attn_config = DSAConfig.generate(args, model)
    for key, value in (
        ("mapping", args.mapping),
        ("dsa_slot_order", "selection"),
        ("layer_boundary_norm", "fused"),
        ("moe_combine_order", "auto"),
    ):
        monkeypatch.setitem(global_server_args_dict, key, value)
    setup, pool = _pool(args, model, attn_config, device="cuda")
    backend = create_paged_router(attn_config, AttentionArch.DSA, backend_name="dsa")
    backend.set_cache_pool(pool)
    backend.init_cuda_graph_state(3)
    positions = torch.zeros(3, dtype=torch.int64, device="cuda")
    pages = torch.tensor([[1], [2], [3]], dtype=torch.int32, device="cuda")
    backend.refresh_decode_metadata(
        3,
        3,
        torch.arange(3, dtype=torch.int32, device="cuda"),
        torch.ones(3, dtype=torch.int32, device="cuda"),
        forward_mode=ForwardMode.DECODE,
        block_tables={group_id: pages for group_id in pool.paged_group_ids},
    )
    group_id = pool.history_group_by_layer()[1]
    group = next(
        group for group in setup.spec.cache_group_specs if group.group_id == group_id
    )
    slots = pages[:, 0].long() * group.block_granularity
    selected = torch.full((3, 2048), -1, dtype=torch.int32, device="cuda")
    selected[:, 0] = slots
    backend.sparse_topk.decode = GlmDsaDecodeTopK(
        topk_indices=selected,
        topk_lens=torch.ones(3, dtype=torch.int32, device="cuda"),
        row_offset=0,
    )
    ctx = ForwardContext(
        attn_backend=backend,
        token_to_kv_pool=pool,
        bs=3,
        num_extends=0,
        input_num_tokens=3,
        forward_mode=ForwardMode.DECODE,
        output_layout=ForwardOutputLayout(0, 0, 3, 1),
    )
    with torch.device("cuda"):
        attention = HYV4Attention(config, args.mapping, layer_id=1, is_nextn=False)
    bind_cache_groups(attention, pool)
    communication = CommManager(
        args.mapping,
        layer_id=1,
        is_moe=False,
        prev_is_moe=False,
        dense_batch_invariant=False,
        query_sharded=False,
    )
    # One selected key and an identity value projection expose the stored
    # latent directly. Nonuniform learned weights distinguish a second norm.
    signs = (torch.arange(512, device="cuda") % 2 * 2 - 1).float()
    amplitudes = torch.tensor([1.0, 2.0, 0.5], device="cuda", dtype=torch.float32)
    raw_latent = amplitudes[:, None] * signs[None, :]
    weight = torch.tensor(
        [0.5, 1.0, 2.0, 4.0], device="cuda", dtype=torch.float32
    ).repeat(128)
    attention.fused_qkv_a_proj_with_mqa.weight.zero_()
    attention.fused_qkv_a_proj_with_mqa.weight[:64, :3] = 1
    attention.fused_qkv_a_proj_with_mqa.weight[64:576, :3] = raw_latent.T.bfloat16()
    attention.q_a_layernorm.weight.fill_(1)
    attention.kv_a_layernorm.weight.copy_(weight)
    attention.q_b_proj.weight.zero_()
    attention.w_kc = torch.zeros((8, 128, 512), dtype=torch.bfloat16, device="cuda")
    attention.w_vc = torch.zeros((8, 512, 128), dtype=torch.bfloat16, device="cuda")
    diagonal = torch.arange(128, device="cuda")
    attention.w_vc[:, diagonal, diagonal] = 1
    attention.learnable_sink_param.fill_(-torch.inf)
    attention.bind_learnable_sink()
    cache = pool.get_key_buffer(1)
    cache.zero_()
    hidden = torch.eye(3, 16, dtype=torch.bfloat16, device="cuda")
    output = attention.attend_token_shaped(positions, hidden, ctx, communication)

    expected = (
        raw_latent
        * torch.rsqrt(raw_latent.square().mean(-1, keepdim=True) + config.rms_norm_eps)
        * weight
    )
    if kv_dtype == "fp8":
        blocks = expected.reshape(3, 4, 128)
        scales = torch.exp2(
            torch.ceil(torch.log2((blocks.abs().amax(-1) / 448).clamp_min(1e-4)))
        )
        quantized = (blocks / scales[..., None]).to(torch.float8_e4m3fn)
        rows = cache[slots, 0].contiguous()
        torch.testing.assert_close(
            rows[:, :512], quantized.reshape(3, 512).view(torch.uint8), rtol=0, atol=0
        )
        torch.testing.assert_close(
            rows[:, 512:528].view(torch.float32), scales, rtol=0, atol=0
        )
        assert not rows[:, 528:].any()
        expected = (quantized.float() * scales[..., None]).reshape(3, 512)
    else:
        torch.testing.assert_close(
            cache[slots, 0, :512], expected.bfloat16(), rtol=0, atol=0
        )
        assert not cache[slots, 0, 512:].any()
    expected_output = (
        expected[:, None, :128].expand(3, 8, 128).reshape(3, -1).bfloat16()
    )
    torch.testing.assert_close(output, expected_output, rtol=0, atol=0)
    assert not cache[0].any()
