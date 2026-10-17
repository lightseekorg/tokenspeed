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

"""Dots-specific latent rescaling, routing precision and checkpoint loading."""

from types import SimpleNamespace

import pytest
import torch
import torch.nn.functional as F
from torch import nn

from tokenspeed.runtime.models.dots3_note import (
    Dots3NoteAttention,
    Dots3NoteRouter,
)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.parametrize("kv_rank", [512, 1024])
@torch.inference_mode()
def test_latents_rescale_before_projection_and_rope_norm(kv_rank):
    torch.manual_seed(42)
    attn = Dots3NoteAttention.__new__(Dots3NoteAttention)
    nn.Module.__init__(attn)
    attn.q_lora_rank = 1024
    attn.qk_rope_head_dim = 64
    attn.kv_lora_rank = kv_rank
    attn.q_lora_scale = (5120 / 1024) ** 0.5
    attn.kv_lora_scale = (5120 / kv_rank) ** 0.5
    from tokenspeed.runtime.layers.layernorm import RMSNorm

    attn.q_a_layernorm = RMSNorm(1024, eps=1e-5).cuda().bfloat16()
    attn.kv_a_layernorm = RMSNorm(kv_rank, eps=1e-5).cuda().bfloat16()
    attn.k_rope_only_layernorm = RMSNorm(64, eps=1e-5).cuda().bfloat16()
    for norm in (attn.q_a_layernorm, attn.kv_a_layernorm, attn.k_rope_only_layernorm):
        norm.weight.uniform_(0.25, 1.5)
    q = torch.randn(17, 1024, device="cuda", dtype=torch.bfloat16)
    kv = torch.randn(17, kv_rank + 64, device="cuda", dtype=torch.bfloat16)
    q_out, latent = attn.normalize_latents(q, kv)

    def norm(x, weight):
        value = x.float()
        return (value * torch.rsqrt(value.square().mean(-1, keepdim=True) + 1e-5)).to(
            x.dtype
        ) * weight

    torch.testing.assert_close(
        q_out,
        norm(q, attn.q_a_layernorm.weight) * attn.q_lora_scale,
        atol=0.03125,
        rtol=0.008,
    )
    torch.testing.assert_close(
        latent[:, :kv_rank],
        norm(kv[:, :kv_rank], attn.kv_a_layernorm.weight) * attn.kv_lora_scale,
        atol=0.0625,
        rtol=0.008,
    )
    torch.testing.assert_close(
        latent[:, kv_rank:],
        norm(kv[:, kv_rank:], attn.k_rope_only_layernorm.weight),
        atol=0.015625,
        rtol=0.008,
    )


def test_router_keeps_correction_bias_and_float32_logits():
    torch.manual_seed(42)
    gate = nn.Module()
    gate.weight = nn.Parameter(torch.randn(16, 256).bfloat16())
    gate.e_score_correction_bias = nn.Parameter(torch.randn(16).float())
    router = Dots3NoteRouter(gate)
    x = torch.randn(9, 256).bfloat16()
    assert router.e_score_correction_bias is gate.e_score_correction_bias
    logits = router(x)
    assert logits.dtype == torch.float32
    torch.testing.assert_close(
        logits, F.linear(x.float(), gate.weight.float()), atol=0, rtol=0
    )


from test_dots3_note_cache import dots_runtime
from test_dots3_note_config import checkpoint


@pytest.mark.parametrize("layer_id", [0, 2])
@torch.inference_mode()
def test_attention_prefix_and_decode_match_cold_prefill(
    dots_runtime, checkpoint, monkeypatch, layer_id
):
    from tokenspeed.runtime.configs.dots3_note_config import Dots3NoteConfig
    from tokenspeed.runtime.distributed import Mapping
    from tokenspeed.runtime.execution.context import ForwardContext
    from tokenspeed.runtime.execution.forward_batch_info import ForwardMode
    from tokenspeed.runtime.layers.paged_attention import bind_cache_groups
    from tokenspeed.runtime.models.deepseek_v3 import _prepare_mla_kv_b_proj_weights
    from tokenspeed.runtime.utils.env import global_server_args_dict

    monkeypatch.setitem(global_server_args_dict, "dense_gemm_backend", "auto")
    monkeypatch.setitem(
        global_server_args_dict, "deepseek_v4_indexer_prefill_max_logits_mb", 32
    )
    config = Dots3NoteConfig.from_pretrained(checkpoint)
    torch.manual_seed(42)
    previous_dtype = torch.get_default_dtype()
    torch.set_default_dtype(torch.bfloat16)
    try:
        attn = Dots3NoteAttention(
            config, Mapping(rank=0, world_size=8), layer_id
        ).cuda()
    finally:
        torch.set_default_dtype(previous_dtype)
    for name, param in attn.named_parameters():
        if "layernorm" in name or "k_norm.weight" in name:
            param.fill_(1)
        elif name.endswith("bias"):
            param.zero_()
        else:
            param.normal_(std=0.015)
    attn.w_kc, attn.w_vc = _prepare_mla_kv_b_proj_weights(attn.kv_b_proj.weight, attn)
    router, pool = dots_runtime
    bind_cache_groups(attn, pool)
    router.init_cuda_graph_state(3)
    tables = {
        gid: torch.arange(1, 21, device="cuda", dtype=torch.int32)[None]
        for gid in router.group_ids
    }
    host_tables = {gid: table.cpu() for gid, table in tables.items()}
    request = torch.zeros(1, device="cuda", dtype=torch.int64)
    x = torch.randn(578, config.hidden_size, device="cuda", dtype=torch.bfloat16)
    positions = torch.arange(578, device="cuda", dtype=torch.int64)
    comm = SimpleNamespace(pre_attn_comm=lambda value, ctx: value)

    def run(start, end, decode=False):
        mode = ForwardMode.DECODE if decode else ForwardMode.EXTEND
        length_cpu = torch.tensor([end - start], dtype=torch.int32)
        prefix_cpu = torch.tensor([start], dtype=torch.int32)
        seq = torch.tensor([end], device="cuda", dtype=torch.int32)
        if decode:
            router.refresh_decode_metadata(
                1, 1, request, seq, forward_mode=mode, block_tables=tables
            )
        else:
            router.init_forward_metadata(
                1,
                1,
                request,
                seq,
                mode,
                block_tables=tables,
                extend_seq_lens=length_cpu.cuda(),
                extend_seq_lens_cpu=length_cpu,
                extend_prefix_lens=prefix_cpu.cuda(),
                extend_prefix_lens_cpu=prefix_cpu,
                extend_replay_lens_cpu=torch.zeros_like(prefix_cpu),
                extend_prompt_lens_cpu=seq.cpu(),
                extend_with_prefix=start > 0,
                query_shard=None,
                block_tables_cpu=host_tables,
            )
        ctx = ForwardContext(
            attn_backend=router,
            token_to_kv_pool=pool,
            bs=1,
            num_extends=0 if decode else 1,
            input_num_tokens=end - start,
            forward_mode=mode,
            output_layout=None,
        )
        result = attn(positions[start:end], x[start:end], ctx, comm)
        assert torch.isfinite(result).all()
        return result

    cold = run(0, 578)
    reference = _attention_reference(attn, config, x, positions)
    error = (
        (cold.float() - reference.float()).square().mean()
        / reference.float().square().mean()
    ).sqrt()
    print("attention_reference", layer_id, float(error))
    assert error < 0.03, (layer_id, float(error))
    run(0, 512)
    warm = run(512, 577)
    decode = run(577, 578, decode=True)
    for actual, expected in ((warm, cold[512:577]), (decode, cold[577:578])):
        error = (
            (actual.float() - expected.float()).square().mean()
            / expected.float().square().mean()
        ).sqrt()
        # FP8 expanded prefill and absorbed decode round at different projections.
        print("attention_roundtrip", layer_id, tuple(actual.shape), float(error))
        assert error < (0.08 if layer_id == 2 else 0.01), (layer_id, float(error))

    # The two requests share the same prefix but take different forward paths.
    mixed_tables = {gid: table.repeat(2, 1) for gid, table in tables.items()}
    lengths = torch.tensor([65, 1], dtype=torch.int32)
    prefixes = torch.tensor([512, 577], dtype=torch.int32)
    seq = (lengths + prefixes).cuda()
    router.init_forward_metadata(
        2,
        1,
        torch.arange(2, device="cuda"),
        seq,
        ForwardMode.MIXED,
        block_tables=mixed_tables,
        extend_seq_lens=lengths.cuda(),
        extend_seq_lens_cpu=lengths,
        extend_prefix_lens=prefixes.cuda(),
        extend_prefix_lens_cpu=prefixes,
        extend_replay_lens_cpu=torch.zeros_like(prefixes),
        extend_prompt_lens_cpu=seq.cpu(),
        extend_with_prefix=True,
        query_shard=None,
        block_tables_cpu={gid: table.cpu() for gid, table in mixed_tables.items()},
    )
    mixed_ctx = ForwardContext(
        attn_backend=router,
        token_to_kv_pool=pool,
        bs=2,
        num_extends=1,
        input_num_tokens=66,
        forward_mode=ForwardMode.MIXED,
        output_layout=None,
    )
    mixed = attn(positions[512:], x[512:], mixed_ctx, comm)
    expected = torch.cat((warm, decode))
    torch.testing.assert_close(mixed, expected, atol=0.004, rtol=0.02)


def _attention_reference(attn, config, x, positions):
    """Torch equations independent of the paged prologue and attention kernels."""

    def norm(value, weight):
        normalized = value.float() * torch.rsqrt(
            value.float().square().mean(-1, keepdim=True) + config.rms_norm_eps
        )
        return normalized.to(value.dtype) * weight

    def rope(value):
        theta = config.swa_rope_theta if attn.is_swa else config.rope_theta
        inv = theta ** (
            -torch.arange(0, 64, 2, device=x.device, dtype=torch.float32) / 64
        )
        angles = positions.float()[:, None] * inv[None]
        cos, sin = angles.cos()[:, None], angles.sin()[:, None]
        value = value.float()
        even, odd = value[..., ::2], value[..., 1::2]
        return (
            torch.stack((even * cos - odd * sin, odd * cos + even * sin), dim=-1)
            .flatten(-2)
            .to(x.dtype)
        )

    q_latent = (
        norm(F.linear(x, attn.q_a_proj.weight), attn.q_a_layernorm.weight)
        * attn.q_lora_scale
    )
    kv = F.linear(x, attn.kv_a_proj_with_mqa.weight)
    latent = (
        norm(kv[:, : attn.kv_lora_rank], attn.kv_a_layernorm.weight)
        * attn.kv_lora_scale
    )
    key_rope = rope(
        norm(kv[:, attn.kv_lora_rank :], attn.k_rope_only_layernorm.weight)[:, None]
    )
    q = F.linear(q_latent, attn.q_b_proj.weight).view(
        -1, attn.num_local_heads, attn.qk_head_dim
    )
    q_rope = rope(q[..., attn.qk_nope_head_dim :])
    if attn.is_swa:
        kv = F.linear(latent, attn.kv_b_proj.weight).view(
            -1, attn.num_local_heads, attn.qk_nope_head_dim + attn.v_head_dim
        )
        key = torch.cat(
            (
                kv[..., : attn.qk_nope_head_dim],
                key_rope.expand(-1, attn.num_local_heads, -1),
            ),
            -1,
        )
        value = kv[..., attn.qk_nope_head_dim :]
        query = torch.cat((q[..., : attn.qk_nope_head_dim], q_rope), -1)
    else:
        q_nope = torch.einsum(
            "thd,hdl->thl", q[..., : attn.qk_nope_head_dim], attn.w_kc
        )
        query = torch.cat((q_nope, q_rope), -1)
        key = torch.cat((latent[:, None], key_rope), -1).expand(
            -1, attn.num_local_heads, -1
        )
        value = latent[:, None].expand(-1, attn.num_local_heads, -1)
    query, key, value = (v.to(torch.float8_e4m3fn).float() for v in (query, key, value))
    scores = torch.einsum("thd,shd->hts", query, key) * attn.scaling
    distance = positions[:, None] - positions[None, :]
    mask = distance >= 0
    if attn.is_swa:
        mask &= distance <= 512
    probabilities = scores.masked_fill(~mask[None], float("-inf")).softmax(-1)
    attended = torch.einsum("hts,shd->thd", probabilities, value).to(x.dtype)
    if not attn.is_swa:
        attended = torch.einsum("thl,hlv->thv", attended, attn.w_vc)
    gate = F.linear(x, attn.g_proj.weight)[
        :, attn.head_start : attn.head_start + attn.num_local_heads
    ]
    attended = attended * gate.sigmoid().unsqueeze(-1)
    return F.linear(attended.flatten(1), attn.o_proj.weight)


def test_model_entry_accepts_text_only_loader_kwargs(monkeypatch):
    from tokenspeed.runtime.models.deepseek_v3 import DeepseekV3ForCausalLM
    from tokenspeed.runtime.models.dots3_note import Dots3NoteForCausalLM
    from tokenspeed.runtime.utils.env import global_server_args_dict

    monkeypatch.setitem(global_server_args_dict, "tp_batch_invariant", "none")
    calls = []
    monkeypatch.setattr(
        DeepseekV3ForCausalLM,
        "__init__",
        lambda self, *args, **kwargs: calls.append((args, kwargs)),
    )
    config = SimpleNamespace(
        apply_mla_qkv_lora_rescale=True,
        attention_gate_type="headwise",
        swa_attention_gate_type="headwise",
        attention_bias=False,
        rope_scaling=None,
    )
    mapping = SimpleNamespace(attn=SimpleNamespace(has_head_tp=False))
    Dots3NoteForCausalLM(
        config, mapping, is_multimodal_active=False, mm_attention_backend=None
    )
    assert len(calls) == 1 and calls[0][0] == (config, mapping)
    with pytest.raises(NotImplementedError, match="language-model-only"):
        Dots3NoteForCausalLM(
            config, mapping, is_multimodal_active=True, mm_attention_backend=None
        )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.parametrize("layer_id", [0, 1])
def test_dense_and_moe_decoder_construct(layer_id, monkeypatch):
    from tokenspeed.runtime.configs.dots3_note_config import Dots3NoteConfig
    from tokenspeed.runtime.distributed import Mapping
    from tokenspeed.runtime.models.dots3_note import Dots3NoteDecoderLayer
    from tokenspeed.runtime.utils.env import global_server_args_dict

    monkeypatch.setitem(global_server_args_dict, "tp_batch_invariant", "none")
    monkeypatch.setitem(global_server_args_dict, "numerics", "auto")
    config = Dots3NoteConfig(
        hidden_size=256,
        num_attention_heads=4,
        q_lora_rank=128,
        kv_lora_rank=128,
        qk_nope_head_dim=64,
        qk_rope_head_dim=64,
        v_head_dim=128,
        layer_types=["full_attention", "full_attention"],
        index_topk=8,
        index_head_dim=128,
        index_n_heads=4,
        rope_theta=80000000.0,
        max_position_embeddings=128,
        rms_norm_eps=1e-5,
        n_routed_experts=4,
        n_shared_experts=1,
        num_experts_per_tok=2,
        first_k_dense_replace=1,
        moe_layer_freq=1,
        hidden_act="silu",
        intermediate_size=512,
        moe_intermediate_size=128,
        topk_method="noaux_tc",
        scoring_func="sigmoid",
        routed_scaling_factor=1.0,
        norm_topk_prob=True,
    )
    with torch.device("cuda"):
        layer = Dots3NoteDecoderLayer(config, layer_id, Mapping(rank=0, world_size=1))
    assert layer.comm_manager is not None
    assert layer.is_moe_layer == (layer_id == 1)
    if layer_id == 1:
        assert isinstance(layer.mlp.gate, Dots3NoteRouter)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@torch.inference_mode()
def test_indexer_uses_leading_interleaved_rope_and_affine_layernorm():
    from tokenspeed.runtime.layers.rotary_embedding import get_rope
    from tokenspeed.runtime.models.dots3_note import Dots3NoteIndexer

    config = SimpleNamespace(
        hidden_size=256,
        q_lora_rank=128,
        index_n_heads=4,
        index_head_dim=128,
        qk_rope_head_dim=64,
    )
    previous_dtype = torch.get_default_dtype()
    torch.set_default_dtype(torch.bfloat16)
    try:
        rotary = get_rope(
            64, rotary_dim=64, max_position=8192, base=80000000.0, is_neox_style=False
        )
        indexer = Dots3NoteIndexer(config, rotary, None, "indexer").cuda()
    finally:
        torch.set_default_dtype(previous_dtype)
    torch.manual_seed(17)
    for param in indexer.parameters():
        param.normal_(std=0.1)
    hidden = torch.randn(5, 256, device="cuda", dtype=torch.bfloat16)
    latent = (
        torch.randn(5, 128, device="cuda", dtype=torch.bfloat16) * (256 / 128) ** 0.5
    )
    positions = torch.tensor([0, 1, 127, 513, 4097], device="cuda")
    q, k, weights = indexer(hidden, latent, positions)
    expected_q = F.linear(latent, indexer.wq_b.weight).view(5, 4, 128)
    expected_k = F.layer_norm(
        F.linear(hidden, indexer.wk.weight).float(),
        (128,),
        indexer.k_norm.weight.float(),
        indexer.k_norm.bias.float(),
        eps=1e-6,
    ).to(hidden.dtype)
    angles = positions.float()[:, None] * 80000000.0 ** (
        -torch.arange(0, 64, 2, device="cuda").float() / 64
    )
    for actual, expected in ((q, expected_q), (k[:, None], expected_k[:, None])):
        rot = expected[..., :64].float()
        cos, sin = angles.cos()[:, None], angles.sin()[:, None]
        even, odd = rot[..., ::2], rot[..., 1::2]
        rotated = (
            torch.stack((even * cos - odd * sin, odd * cos + even * sin), -1)
            .flatten(-2)
            .to(expected.dtype)
        )
        torch.testing.assert_close(actual[..., :64], rotated, atol=0.016, rtol=0.016)
        torch.testing.assert_close(
            actual[..., 64:], expected[..., 64:], atol=0.004, rtol=0.016
        )
    torch.testing.assert_close(
        weights, F.linear(hidden, indexer.weights_proj.weight), atol=0, rtol=0
    )
    assert indexer.scaling == (4 * 128) ** -0.5


def test_text_loader_skips_encoder_and_both_mtp_namespaces(monkeypatch):
    from tokenspeed.runtime.models.dots3_note import Dots3NoteForCausalLM

    model = Dots3NoteForCausalLM.__new__(Dots3NoteForCausalLM)
    nn.Module.__init__(model)
    model.config = SimpleNamespace(num_hidden_layers=2, n_routed_experts=4)
    model.mapping = SimpleNamespace(moe=SimpleNamespace(ep_rank=0, ep_size=1))
    model.model = nn.Module()
    model.model.embed_tokens = nn.Embedding(4, 2)
    monkeypatch.setattr(model, "post_load_weights", lambda: None)
    target = torch.arange(8).view(4, 2).float()
    loaded = model.load_weights(
        iter(
            [
                ("vision_encoder.weight", torch.zeros(1)),
                ("audio_encoder.weight", torch.zeros(1)),
                ("model.mtp.embed_tokens.weight", torch.zeros(1)),
                ("model.layers.2.enorm.weight", torch.zeros(1)),
                ("model.embed_tokens.weight", target),
            ]
        )
    )
    assert loaded == {"model.embed_tokens.weight"}
    torch.testing.assert_close(model.model.embed_tokens.weight, target)
    for name in (
        "model.unrecognized.weight",
        "model.layers.1.mlp.experts.0.unrecognized.weight",
    ):
        with pytest.raises(ValueError, match="Unexpected dots3-note text weight"):
            model.load_weights(iter([(name, torch.zeros(1))]))
