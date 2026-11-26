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

"""Numerical correctness unit tests for Hy4-flash core operators and modules."""

from __future__ import annotations

import os
import sys
from types import SimpleNamespace

import pytest
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from ci_system.ci_register import register_cuda_ci  # noqa: E402

register_cuda_ci(est_time=30, suite="runtime-1gpu")

from tokenspeed.runtime.configs.hy4_flash_config import (  # noqa: E402
    HYV4FlashDFlyConfig,
    HYV4FlashTextConfig,
    HYV4FlashVisionConfig,
)
from tokenspeed.runtime.distributed.mapping import Mapping  # noqa: E402
from tokenspeed.runtime.execution.drafter.dfly import DFly  # noqa: E402
from tokenspeed.runtime.execution.request_token_history import (  # noqa: E402
    RequestTokenHistoryView,
)
from tokenspeed.runtime.layers.vocab_parallel_embedding import (  # noqa: E402
    ParallelLMHead,
    VocabParallelEmbedding,
)
from tokenspeed.runtime.models.hy4_flash import (  # noqa: E402
    HYV4FlashForCausalLM,
    _apply_hy4_rmsnorm,
    _linear_cpu_or_cuda,
)
from tokenspeed.runtime.models.hy4_flash_dfly import (  # noqa: E402
    HYV4FlashDFlyModel,
    HYV4FlashForCausalLMDFly,
    _normalize_block_positions_3d,
)
from tokenspeed.runtime.models.hy4_flash_engram import (  # noqa: E402
    HYV4FlashEngramLayer,
    build_hy4_engram_inputs,
)
from tokenspeed.runtime.models.hy4_flash_vision import (  # noqa: E402
    HYV4FlashVisionTower,
)
from tokenspeed.runtime.utils.env import global_server_args_dict  # noqa: E402


def _dfly_eager_reference(
    dfly_backbone: HYV4FlashDFlyModel,
    *,
    anchor_tokens: torch.Tensor,
    prev_hidden: torch.Tensor | None,
    draft_positions: torch.Tensor,
    context_latents: list[torch.Tensor],
    keep_masks: list[torch.Tensor],
    block_size: int,
) -> torch.Tensor:
    """Single-softmax eager reference over explicit ``context_latent ++ draft_latent``."""
    hidden = dfly_backbone.build_block_inputs(anchor_tokens, prev_hidden, block_size)
    bsz = hidden.shape[0]
    positions_3d = _normalize_block_positions_3d(
        draft_positions, batch_size=bsz, block_size=block_size
    )
    for idx, layer in enumerate(dfly_backbone.layers):
        attn = layer.self_attn
        normed = _apply_hy4_rmsnorm(hidden, layer.input_layernorm)
        q_nope, q_pe, latent_draft = attn._project_draft_q_and_latent(
            normed, positions_3d
        )
        query = torch.cat((q_nope, q_pe), dim=-1).transpose(1, 2)
        full_latent = torch.cat(
            (context_latents[idx].to(latent_draft.dtype), latent_draft), dim=1
        )
        kv_len = full_latent.shape[1]
        k_pass, k_pe = full_latent.split(
            [attn.kv_lora_rank, attn.qk_rope_head_dim], dim=-1
        )
        kv_exp = _linear_cpu_or_cuda(
            attn.kv_b_proj, k_pass.reshape(bsz * kv_len, attn.kv_lora_rank)
        ).view(
            bsz, kv_len, attn.num_local_heads, attn.qk_nope_head_dim + attn.v_head_dim
        )
        k_nope, v_states = kv_exp.split(
            [attn.qk_nope_head_dim, attn.v_head_dim], dim=-1
        )
        k_pe_exp = k_pe.unsqueeze(2).expand(-1, -1, attn.num_local_heads, -1)
        key = torch.cat((k_nope, k_pe_exp), dim=-1).transpose(1, 2)
        value = v_states.transpose(1, 2)

        logits = (
            torch.matmul(query.float(), key.float().transpose(-1, -2)) * attn.scaling
        )
        keep_mask = keep_masks[idx]
        logits = logits.masked_fill(~keep_mask, torch.finfo(torch.float32).min)
        if attn.learnable_sink_param is not None:
            sink = (
                attn.learnable_sink_param.float()
                .view(1, -1, 1, 1)
                .expand(bsz, attn.num_local_heads, block_size, 1)
            )
            probs = torch.softmax(torch.cat((logits, sink), dim=-1), dim=-1)[..., :-1]
        else:
            probs = torch.softmax(logits, dim=-1)

        attn_out = (
            torch.matmul(probs, value.float())
            .to(normed.dtype)
            .transpose(1, 2)
            .contiguous()
        )
        attn_out = attn_out * keep_mask[:, 0].any(dim=-1, keepdim=True).unsqueeze(
            -1
        ).to(attn_out.dtype)
        num_tokens = bsz * block_size
        attn_flat = attn_out.reshape(num_tokens, attn.num_local_heads, attn.v_head_dim)
        if attn.linear_gate is not None:
            gate = _linear_cpu_or_cuda(
                attn.linear_gate, normed.reshape(num_tokens, normed.shape[-1])
            ).view(num_tokens, attn.num_local_heads, attn.v_head_dim)
            attn_flat = attn_flat * torch.sigmoid(gate)
        projected = _linear_cpu_or_cuda(
            attn.o_proj,
            attn_flat.reshape(num_tokens, attn.num_local_heads * attn.v_head_dim),
        )
        if attn.post_proj_norm is not None:
            projected = _apply_hy4_rmsnorm(projected, attn.post_proj_norm)

        hidden = hidden + projected.view(bsz, block_size, -1)
        hidden = hidden + layer.mlp(
            _apply_hy4_rmsnorm(hidden, layer.post_attention_layernorm)
        )
    return _apply_hy4_rmsnorm(hidden, dfly_backbone.final_norm)


def test_engram_host_table_and_incremental_request_token_history_parity():
    torch.manual_seed(3)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    prev_dtype = torch.get_default_dtype()
    torch.set_default_dtype(torch.bfloat16)
    cfg = HYV4FlashTextConfig(
        vocab_size=128,
        hidden_size=64,
        hc_mult=4,
        engram_layer_ids=[2],
        engram_max_ngram_size=3,
        engram_n_head_per_ngram=4,
        engram_n_embed_per_ngram=32,
        engram_layer_head_vocab_sizes={"2": [[17, 19, 23, 29], [31, 37, 41, 43]]},
        engram_pad_id=0,
        engram_fp8_embedding=False,
        engram_seed=0,
        engram_lookup_table_size=128,
    )
    mapping = Mapping(rank=0, world_size=1)

    prev_flag = global_server_args_dict.get("engram_host_table", False)
    try:
        global_server_args_dict["engram_host_table"] = False
        gpu_engram = (
            HYV4FlashEngramLayer(config=cfg, layer_id=2, mapping=mapping)
            .to(device=device)
            .eval()
        )
        with torch.no_grad():
            for p in gpu_engram.parameters():
                torch.nn.init.normal_(p, mean=0.0, std=0.05)
            gpu_engram.lookup_table.copy_(
                torch.randint(0, 16, (128,), dtype=torch.int32, device=device)
            )
            gpu_engram.refresh_hash_multipliers()

        global_server_args_dict["engram_host_table"] = True
        host_engram = (
            HYV4FlashEngramLayer(config=cfg, layer_id=2, mapping=mapping)
            .to(device=device)
            .eval()
        )
        host_engram.load_state_dict(
            {k: v.cpu() for k, v in gpu_engram.state_dict().items()}
        )
        host_engram.to(device=device)
        assert host_engram.multi_head_embedding.embedding.weight.device.type == "cpu"
        host_engram.refresh_hash_multipliers()
    finally:
        global_server_args_dict["engram_host_table"] = prev_flag
        torch.set_default_dtype(prev_dtype)

    # Full sequence vs chunked prefill + step-by-step decode through RequestTokenHistoryView
    seq = torch.tensor([5, 11, 22, 33, 44, 55, 66], dtype=torch.int32, device=device)
    streams = torch.randn(
        len(seq), cfg.hc_mult, cfg.hidden_size, dtype=torch.bfloat16, device=device
    )
    full_ngrams, full_mask = build_hy4_engram_inputs(seq, None, config=cfg)

    with torch.no_grad():
        gpu_out = gpu_engram(streams, full_ngrams, full_mask)
        host_out = host_engram(streams, full_ngrams, full_mask)
    assert torch.equal(gpu_out, host_out)

    # Incremental history replay: prefill 4 tokens, then decode 3 tokens one by one
    history_buf = torch.zeros((4, 16), dtype=torch.int32, device=device)
    committed = torch.zeros((4,), dtype=torch.int32, device=device)
    req_slots = torch.tensor([1], dtype=torch.int64, device=device)
    active = torch.tensor([True], device=device)

    view_prefill = RequestTokenHistoryView(
        history_token_ids=history_buf,
        committed_lengths=committed,
        req_pool_indices=req_slots,
        input_start_offsets=torch.tensor([0, 4], dtype=torch.int32, device=device),
        active_request_mask=active,
    )
    ng_p, mk_p = build_hy4_engram_inputs(
        seq[:4], None, config=cfg, request_token_history=view_prefill
    )
    committed[1] = 4
    step_ngrams = [ng_p]
    step_masks = [mk_p]
    for pos_idx in range(4, len(seq)):
        view_dec = RequestTokenHistoryView(
            history_token_ids=history_buf,
            committed_lengths=committed,
            req_pool_indices=req_slots,
            input_start_offsets=torch.tensor([0, 1], dtype=torch.int32, device=device),
            active_request_mask=active,
        )
        ng_d, mk_d = build_hy4_engram_inputs(
            seq[pos_idx : pos_idx + 1],
            None,
            config=cfg,
            request_token_history=view_dec,
        )
        committed[1] += 1
        step_ngrams.append(ng_d)
        step_masks.append(mk_d)

    assert torch.equal(torch.cat(step_ngrams, dim=0), full_ngrams)
    assert torch.equal(torch.cat(step_masks, dim=0), full_mask)


def test_vision_tower_image_and_video_shapes_and_finite_outputs():
    torch.manual_seed(4)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.bfloat16 if device.type == "cuda" else torch.float32
    prev_dtype = torch.get_default_dtype()
    torch.set_default_dtype(dtype)
    try:
        vit_cfg = HYV4FlashVisionConfig(
            hidden_size=64,
            intermediate_size=128,
            num_hidden_layers=2,
            num_attention_heads=4,
            patch_size=8,
            spatial_merge_size=2,
            out_hidden_size=96,
            num_channels=3,
            max_image_size=64,
        )
        vit = HYV4FlashVisionTower(vit_cfg).to(device=device, dtype=dtype).eval()
    finally:
        torch.set_default_dtype(prev_dtype)
    with torch.no_grad():
        for p in vit.parameters():
            torch.nn.init.normal_(p, mean=0.0, std=0.04)

    # One image (1x4x4 patches) + one 2-frame video (2x2x4 patches)
    grid_thw = torch.tensor([[1, 4, 4], [2, 2, 4]], dtype=torch.int64, device=device)
    total_patches = int((grid_thw[:, 0] * grid_thw[:, 1] * grid_thw[:, 2]).sum().item())
    pixel_values = torch.randn(total_patches, 3 * 8 * 8, device=device, dtype=dtype)

    with torch.no_grad():
        out = vit(pixel_values, grid_thw)
    # Image: 1 * (4/2) * (4/2 + 1) = 6 tokens; Video: 2 * (2/2) * (4/2 + 1) = 6 tokens -> 12 total
    assert out.shape == (12, 96)
    assert torch.isfinite(out).all()


def test_backbone_and_dfly2_target_kv_exact_equivalence_and_chain_walk():
    torch.manual_seed(5)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.bfloat16 if device.type == "cuda" else torch.float32
    prev_dtype = torch.get_default_dtype()
    torch.set_default_dtype(dtype)
    mapping = Mapping(rank=0, world_size=1)

    try:
        # End-to-end 4-layer backbone (dense L0 + MoE/Engram/SWA/Compressed-DSA + FP32 LM head)
        text_cfg = HYV4FlashTextConfig(
            vocab_size=128,
            hidden_size=64,
            intermediate_size=96,
            moe_intermediate_size=32,
            routed_expert_hidden_size=32,
            num_hidden_layers=4,
            num_attention_heads=2,
            num_key_value_heads=2,
            swa_num_attention_heads=4,
            n_routed_experts=4,
            num_experts_per_tok=2,
            n_shared_experts=1,
            q_lora_rank=32,
            kv_lora_rank=32,
            qk_nope_head_dim=16,
            qk_rope_head_dim=64,
            v_head_dim=16,
            index_n_heads=2,
            index_head_dim=64,
            index_topk=8,
            index_kpool=4,
            sliding_window=8,
            layer_types=[
                "sliding_attention",
                "sliding_attention",
                "indexed_attention",
                "sliding_attention",
            ],
            mlp_layer_types=["dense", "sparse", "sparse", "sparse"],
            hc_mult=4,
            engram_layer_ids=[2],
            engram_max_ngram_size=3,
            engram_n_head_per_ngram=2,
            engram_n_embed_per_ngram=32,
            engram_layer_head_vocab_sizes={"2": [[17, 19], [23, 29]]},
            engram_pad_id=0,
            engram_lookup_table_size=128,
            enable_lm_head_fp32=True,
        )
        target = (
            HYV4FlashForCausalLM(config=text_cfg, mapping=mapping, quant_config=None)
            .to(device=device)
            .eval()
        )
        with torch.no_grad():
            for p in target.parameters():
                torch.nn.init.normal_(p, mean=0.0, std=0.04)
        target.post_load_weights()

        ids = torch.randint(1, 120, (16,), dtype=torch.long, device=device)
        pos_1d = torch.arange(16, dtype=torch.long, device=device)
        pos_3d = torch.stack([pos_1d, pos_1d // 2, pos_1d % 3], dim=0)
        with torch.no_grad():
            hidden, _ = target.model(ids, pos_3d, ctx=None)
        assert hidden.shape == (16, 64)
        assert torch.isfinite(hidden).all()

        # DFly2 target-KV path vs single-softmax eager reference + empty/padded slots + chain walk
        dfly_cfg = HYV4FlashDFlyConfig(
            vocab_size=128,
            hidden_size=64,
            intermediate_size=96,
            num_hidden_layers=2,
            num_attention_heads=4,
            num_key_value_heads=4,
            q_lora_rank=32,
            kv_lora_rank=32,
            qk_nope_head_dim=16,
            qk_rope_head_dim=64,
            v_head_dim=16,
            block_size=5,
            target_kv_layers=[0, 2],
            target_index_layer=[0, 2],
            target_index_topk=8,
            mtp_fusion="slot0",
            enable_hidden_correction=True,
            enable_confidence_head=True,
        )
        dfly_model = (
            HYV4FlashForCausalLMDFly(
                config=dfly_cfg, mapping=mapping, quant_config=None
            )
            .to(device=device)
            .eval()
        )
        with torch.no_grad():
            for p in dfly_model.parameters():
                torch.nn.init.normal_(p, mean=0.0, std=0.04)
        dfly_model.post_load_weights()
        dfly_model.set_embed_and_head_module(
            target.model.embed_tokens.weight, target.lm_head
        )
    finally:
        torch.set_default_dtype(prev_dtype)

    bsz, block_size, ctx_len = 2, 5, 6
    anchor_tokens = torch.tensor([7, 19], dtype=torch.int32, device=device)
    prev_hidden = torch.randn(bsz, 64, device=device, dtype=dtype)
    draft_pos = (
        torch.arange(block_size, dtype=torch.int64, device=device)
        .unsqueeze(0)
        .expand(bsz, -1)
    )

    # Use non-contiguous 4D paged arena views matching HybridHy4FlashTokenToKVPool.get_key_buffer
    arena = torch.randn(4, 2, 8, 1, 32 + 64, device=device, dtype=dtype) * 0.2
    kv0 = arena[:, 0]  # [num_pages=4, page_size=8, 1, D], strided across pages
    kv2 = arena[:, 1]
    assert not kv0.is_contiguous()
    # Row 0 has empty context (all -1); row 1 has 4 valid scattered slots + 2 padded -1
    ctx_slots = torch.tensor(
        [[-1, -1, -1, -1, -1, -1], [3, 8, 12, 19, -1, -1]],
        dtype=torch.int32,
        device=device,
    )
    pool = SimpleNamespace(get_key_buffer=lambda lid: kv0 if lid == 0 else kv2)

    with torch.no_grad():
        out_kv = dfly_model.model.forward_with_target_kv(
            anchor_tokens=anchor_tokens,
            prev_hidden=prev_hidden,
            draft_positions=draft_pos,
            target_kv_pool=pool,
            ctx_slots_per_layer=[ctx_slots, ctx_slots],
            ctx_lens_per_layer=[torch.tensor([0, 4], dtype=torch.int32, device=device)]
            * 2,
            block_size=block_size,
        )
        # Matching explicit reference
        lat0 = torch.zeros(bsz, ctx_len, 32 + 64, device=device, dtype=dtype)
        lat2 = torch.zeros(bsz, ctx_len, 32 + 64, device=device, dtype=dtype)
        for j, s in enumerate([3, 8, 12, 19]):
            lat0[1, j] = kv0[s // 8, s % 8, 0]
            lat2[1, j] = kv2[s // 8, s % 8, 0]
        q_idx = torch.arange(block_size, device=device).unsqueeze(1)
        k_idx = torch.arange(block_size, device=device).unsqueeze(0)
        draft_keep = (k_idx <= q_idx).unsqueeze(0).unsqueeze(0).expand(bsz, 1, -1, -1)
        ctx_keep = (
            (ctx_slots >= 0)
            .unsqueeze(1)
            .unsqueeze(2)
            .expand(bsz, 1, block_size, ctx_len)
        )
        keep_mask = torch.cat([ctx_keep, draft_keep], dim=-1)
        out_ref = _dfly_eager_reference(
            dfly_model.model,
            anchor_tokens=anchor_tokens,
            prev_hidden=prev_hidden,
            draft_positions=draft_pos,
            context_latents=[lat0, lat2],
            keep_masks=[keep_mask, keep_mask],
            block_size=block_size,
        )
    atol = 2e-2 if dtype == torch.bfloat16 else 1e-5
    assert torch.allclose(out_kv, out_ref, atol=atol)

    # Verify DFly.propose_chain matches step-by-step hidden_correction + target FP32 lm_head
    drafter = DFly(
        spec_num_tokens=block_size,
        spec_num_steps=block_size - 1,
        draft_model_runner=SimpleNamespace(model=dfly_model, device=device),
        input_buffers=SimpleNamespace(max_bs=4),
        vocab_size=128,
    )
    drafter.lm_head = target.lm_head
    drafter.logits_processor = None
    with torch.no_grad():
        proposals = drafter.propose_chain(out_kv, anchor_tokens)
        expected = torch.empty((bsz, block_size), dtype=torch.int32, device=device)
        expected[:, 0] = anchor_tokens
        prev_tok = anchor_tokens.clone()
        for slot in range(1, block_size):
            corr = dfly_model.model.correct_hidden(out_kv[:, slot, :], prev_tok)
            logits = F.linear(
                corr.to(target.lm_head.weight.dtype), target.lm_head.weight
            )
            prev_tok = logits.argmax(dim=-1).to(torch.int32)
            expected[:, slot] = prev_tok
    assert torch.equal(proposals, expected)


def _make_vocab_heads(
    weights: torch.Tensor,
    base_vocab: int,
    head_type: type[VocabParallelEmbedding] = ParallelLMHead,
) -> list[VocabParallelEmbedding]:
    heads: list[VocabParallelEmbedding] = []
    for rank in range(2):
        head = head_type(
            weights.shape[0],
            weights.shape[1],
            org_num_embeddings=base_vocab,
            tp_rank=rank,
            tp_size=2,
            tp_group=(0, 1),
            params_dtype=weights.dtype,
        )
        shard = head.shard_indices
        with torch.no_grad():
            head.weight.fill_(99)  # Padding must never win a selection.
            head.weight[: shard.num_org_elements].copy_(
                weights[shard.org_vocab_start_index : shard.org_vocab_end_index]
            )
            first = shard.num_org_elements_padded
            head.weight[first : first + shard.num_added_elements].copy_(
                weights[shard.added_vocab_start_index : shard.added_vocab_end_index]
            )
        heads.append(head)
    return heads


@pytest.mark.parametrize(
    ("base_vocab", "added_vocab", "valid_vocab", "mode"),
    [
        (128, 0, 128, "prepared_argmax"),
        (17, 0, 17, "prepared_argmax"),  # Rank 1 contains only padding.
        (106, 0, 106, "greedy"),
        (106, 0, 103, "sample"),
        (67, 39, 103, "no_processor"),
    ],
)
@torch.no_grad()
def test_dfly_propose_chain_tp_vocab(
    monkeypatch, base_vocab, added_vocab, valid_vocab, mode
):
    """DFly selects from valid global columns across TP shards."""
    from tokenspeed.runtime.execution.drafter import dfly
    from tokenspeed.runtime.layers import logits_processor

    vocab = base_vocab + added_vocab
    cfg = HYV4FlashDFlyConfig(
        vocab_size=vocab,
        hidden_size=128,
        num_hidden_layers=0,
        mtp_fusion=None,
        enable_hidden_correction=False,
        enable_confidence_head=False,
        block_size=3,
    )
    mapping = Mapping(rank=1, world_size=2, attn_tp_size=2)
    model = HYV4FlashForCausalLMDFly(cfg, mapping=mapping)
    drafter = DFly(
        spec_num_tokens=3,
        spec_num_steps=2,
        draft_model_runner=SimpleNamespace(model=model, device="cpu"),
        vocab_size=valid_vocab,
    )
    weights = torch.zeros(vocab, cfg.hidden_size)
    weights[:, 0] = torch.arange(vocab) / vocab
    weights[valid_vocab - 2, :2] = 3
    weights[2, 1] = 3  # Cross-rank tie with a smaller token ID.
    weights[valid_vocab:] = 50
    heads = _make_vocab_heads(weights, base_vocab)
    peer, drafter.lm_head = heads
    hidden = torch.zeros(2, 3, cfg.hidden_size)
    hidden[0, :, 0] = hidden[1, :, 1] = 1
    peer_logits = hidden[:, 1] @ peer.weight.T
    expected_logits = hidden[:, 1] @ weights[:valid_vocab].T

    # Only the missing peer's logits and the CUDA proposal sampler are emulated.
    if mode == "no_processor":
        drafter.logits_processor = None
        monkeypatch.setattr(
            dfly,
            "all_gather",
            lambda local, group, dim: torch.cat((peer_logits, local), dim),
        )
    else:

        def gather(output, local, group):
            output.copy_(torch.cat((peer_logits, local)))

        monkeypatch.setattr(logits_processor, "all_gather_single", gather)
        if mode == "prepared_argmax":
            drafter.logits_processor.fuse_argmax()
            drafter.logits_processor.prepare_communication(drafter.lm_head)

    anchors = torch.tensor([5, 8], dtype=torch.int32)
    expected = torch.stack(
        (anchors, expected_logits.argmax(1), expected_logits.argmax(1)), dim=1
    )
    if mode == "sample":

        def sample(logits, *, step):
            assert logits.is_contiguous()  # Required by the CUDA softmax kernel.
            torch.testing.assert_close(logits, expected_logits)
            generator = torch.Generator().manual_seed(step)
            return torch.multinomial(
                logits.softmax(1), 1, generator=generator
            ).flatten()

        drafter.draft_sampler = SimpleNamespace(propose=sample)
        for step in range(2):
            expected[:, step + 1] = sample(expected_logits, step=step)
    torch.testing.assert_close(
        drafter.propose_chain(hidden, anchors), expected.to(torch.int32)
    )
