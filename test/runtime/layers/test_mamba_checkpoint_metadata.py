from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

from tokenspeed.runtime.execution.forward_batch_info import ForwardMode
from tokenspeed.runtime.layers.attention.backends import hybrid_linear_attn
from tokenspeed.runtime.layers.attention.backends.hybrid_linear_attn import (
    MambaAttnBackend,
    MambaForwardMetadata,
    SimpleMambaPool,
)
from tokenspeed.runtime.layers.attention.linear.mamba_state_scatter_triton import (
    fused_mamba_state_copy,
)


def _new_backend(page_size: int = 64) -> MambaAttnBackend:
    pool = SimpleMambaPool(
        size=8,
        num_mamba_layers=1,
        conv_state_shape=(4,),
        temporal_state_shape=(2, 2),
        conv_dtype=torch.float32,
        ssm_dtype=torch.float32,
        mamba_layer_ids=[0],
        device="cpu",
        page_size=page_size,
    )

    backend = object.__new__(MambaAttnBackend)
    backend.pool = pool
    backend.device = "cpu"
    backend.is_draft = False
    backend.spec_num_tokens = 1
    backend.speculative_num_draft_tokens = 0
    backend.flat_state_active = False
    return backend


def test_simple_mamba_pool_current_input_map_uses_rank_local_req_pool_range():
    pool = SimpleMambaPool(
        size=48,
        num_mamba_layers=1,
        conv_state_shape=(4,),
        temporal_state_shape=(2, 2),
        conv_dtype=torch.float32,
        ssm_dtype=torch.float32,
        mamba_layer_ids=[0],
        device="cpu",
        page_size=64,
        speculative_num_draft_tokens=4,
        max_req_pool_size=21,
    )

    assert pool.current_input_size == 22
    assert pool.current_input_indices.shape[0] == 22
    # MTP draft slots are addressed by rank-local req_pool_idx, plus one
    # graph-padding sink row after the scheduler-owned 1-based range.
    assert pool.total_size == 48 + 22 * 3


def test_extend_tracks_final_page_boundary_when_branch_checkpoint_is_inside():
    backend = _new_backend(page_size=64)

    backend.init_forward_metadata(
        bs=1,
        req_pool_indices=torch.tensor([0], dtype=torch.int32),
        seq_lens=torch.tensor([320], dtype=torch.int32),
        forward_mode=ForwardMode.EXTEND,
        mamba_pool_indices=torch.tensor([2], dtype=torch.int32),
        mamba_branching_seqlens=torch.tensor([256], dtype=torch.int32),
        extend_prefix_lens=torch.tensor([192], dtype=torch.int32),
        mamba_track_pool_indices=torch.tensor([5], dtype=torch.int32),
    )

    metadata = backend.forward_metadata

    assert metadata.track_ssm_h_dst is None
    assert metadata.track_ssm_final_src.tolist() == [2]
    assert metadata.track_ssm_final_dst.tolist() == [5]


def test_extend_tracks_only_branch_boundary_when_final_boundary_is_not_aligned():
    backend = _new_backend(page_size=64)

    backend.init_forward_metadata(
        bs=1,
        req_pool_indices=torch.tensor([0], dtype=torch.int32),
        seq_lens=torch.tensor([319], dtype=torch.int32),
        forward_mode=ForwardMode.EXTEND,
        mamba_pool_indices=torch.tensor([2], dtype=torch.int32),
        mamba_branching_seqlens=torch.tensor([256], dtype=torch.int32),
        extend_prefix_lens=torch.tensor([192], dtype=torch.int32),
        mamba_track_pool_indices=torch.tensor([5], dtype=torch.int32),
    )

    metadata = backend.forward_metadata

    assert metadata.track_ssm_h_dst.tolist() == [5]
    assert metadata.track_ssm_final_dst is None


def test_extend_tracks_last_inserted_page_boundary_when_branch_is_earlier():
    backend = _new_backend(page_size=64)

    backend.init_forward_metadata(
        bs=1,
        req_pool_indices=torch.tensor([0], dtype=torch.int32),
        seq_lens=torch.tensor([350], dtype=torch.int32),
        forward_mode=ForwardMode.EXTEND,
        mamba_pool_indices=torch.tensor([2], dtype=torch.int32),
        mamba_branching_seqlens=torch.tensor([256], dtype=torch.int32),
        extend_prefix_lens=torch.tensor([192], dtype=torch.int32),
        mamba_track_pool_indices=torch.tensor([5], dtype=torch.int32),
    )

    metadata = backend.forward_metadata

    assert metadata.track_ssm_h_src.tolist() == [1]
    assert metadata.track_ssm_h_src_fla.tolist() == [2]
    assert metadata.track_ssm_h_dst.tolist() == [5]
    assert metadata.track_ssm_final_dst is None


def test_extend_tracks_last_inserted_page_boundary_without_branch_hint():
    backend = _new_backend(page_size=64)

    backend.init_forward_metadata(
        bs=1,
        req_pool_indices=torch.tensor([0], dtype=torch.int32),
        seq_lens=torch.tensor([350], dtype=torch.int32),
        forward_mode=ForwardMode.EXTEND,
        mamba_pool_indices=torch.tensor([2], dtype=torch.int32),
        mamba_branching_seqlens=torch.tensor([-1], dtype=torch.int32),
        extend_prefix_lens=torch.tensor([256], dtype=torch.int32),
        mamba_track_pool_indices=torch.tensor([5], dtype=torch.int32),
    )

    metadata = backend.forward_metadata

    assert metadata.track_ssm_h_src.tolist() == [0]
    assert metadata.track_ssm_h_src_fla.tolist() == [1]


def test_extend_skips_unaligned_inserted_page_boundary():
    backend = _new_backend(page_size=64)

    backend.init_forward_metadata(
        bs=1,
        req_pool_indices=torch.tensor([0], dtype=torch.int32),
        seq_lens=torch.tensor([66], dtype=torch.int32),
        forward_mode=ForwardMode.EXTEND,
        mamba_pool_indices=torch.tensor([2], dtype=torch.int32),
        mamba_branching_seqlens=torch.tensor([-1], dtype=torch.int32),
        extend_prefix_lens=torch.tensor([41], dtype=torch.int32),
        mamba_track_pool_indices=torch.tensor([5], dtype=torch.int32),
    )

    metadata = backend.forward_metadata

    assert metadata.track_ssm_h_dst is None
    assert metadata.track_ssm_final_dst is None


def test_gdn_extend_trims_prefill_graph_bucket_to_live_rows(monkeypatch):
    bucket_tokens = 1536
    live_tokens = 1478
    num_heads = 2
    head_dim = 2
    mixed_width = num_heads * head_dim * 3

    backend = object.__new__(MambaAttnBackend)
    backend.is_draft = False
    backend.speculative_num_draft_tokens = 4
    backend.flat_state_active = False

    conv_states = torch.zeros(2, mixed_width)
    ssm_states = torch.zeros(2, num_heads, head_dim, head_dim)
    backend.pool = SimpleNamespace(
        get_mamba_params=lambda _layer_id: (conv_states, ssm_states)
    )
    backend.forward_metadata = MambaForwardMetadata(
        query_start_loc=torch.tensor([0, live_tokens], dtype=torch.int32),
        mamba_cache_indices=torch.tensor([0], dtype=torch.int64),
        extend_prefix_lens=torch.tensor([60_222], dtype=torch.int32),
        extend_seq_lens_cpu=torch.tensor([live_tokens], dtype=torch.int32),
        track_ssm_h_src=torch.tensor([0], dtype=torch.int64),
        track_ssm_h_dst=torch.tensor([1], dtype=torch.int64),
        track_conv_indices=torch.tensor([live_tokens - 1], dtype=torch.int64),
    )

    seen = {}

    def fake_causal_conv(x, _weights, _bias, **kwargs):
        seen["conv_rows"] = x.shape[1]
        seen["conv_seq_lens_cpu"] = kwargs["seq_lens_cpu"]
        return x

    def fake_qkv_split(x, **_kwargs):
        seen["split_rows"] = x.shape[0]
        shape = (1, x.shape[0], num_heads, head_dim)
        return torch.zeros(shape), torch.zeros(shape), torch.zeros(shape)

    def fake_gating(_a_log, a, _dt_bias):
        seen["gating_rows"] = a.shape[0]
        return torch.zeros_like(a)

    def fake_gdn_prefill(query, key, value, g, beta, **kwargs):
        seen["gdn_shapes"] = tuple(
            tensor.shape for tensor in (query, key, value, g, beta)
        )
        seen["gdn_cu_seqlens"] = kwargs["cu_seqlens"]
        seen["gdn_output_h"] = kwargs["output_h"]
        return SimpleNamespace(
            out=query,
            final_state=kwargs["initial_state"].clone(),
            h=torch.zeros(1, num_heads, head_dim, head_dim),
            h_layout=hybrid_linear_attn.GdnCheckpointLayout.FLASHINFER,
        )

    monkeypatch.setattr(hybrid_linear_attn, "causal_conv1d_fn", fake_causal_conv)
    monkeypatch.setattr(
        hybrid_linear_attn, "fused_qkv_split_gdn_prefill", fake_qkv_split
    )
    monkeypatch.setattr(hybrid_linear_attn, "fused_gdn_gating", fake_gating)
    monkeypatch.setattr(hybrid_linear_attn, "gdn_chunk_prefill", fake_gdn_prefill)

    mixed_qkv = torch.arange(bucket_tokens * mixed_width, dtype=torch.float32).view(
        bucket_tokens, mixed_width
    )
    a = torch.zeros(bucket_tokens, num_heads)
    b = torch.zeros(bucket_tokens, num_heads)
    output = backend.forward_extend(
        q=None,
        k=None,
        v=None,
        layer=SimpleNamespace(),
        out_cache_loc=None,
        token_to_kv_pool=None,
        bs=1,
        forward_mode=ForwardMode.EXTEND,
        mixed_qkv=mixed_qkv,
        conv_weights=torch.empty(0),
        bias=None,
        activation="silu",
        key_dim=num_heads * head_dim,
        value_dim=num_heads * head_dim,
        attention_tp_size=1,
        head_k_dim=head_dim,
        head_v_dim=head_dim,
        a=a,
        b=b,
        A_log=torch.zeros(num_heads),
        dt_bias=torch.zeros(num_heads),
        layer_id=0,
        seq_len=bucket_tokens,
    )

    assert seen["conv_rows"] == live_tokens
    assert seen["split_rows"] == live_tokens
    assert seen["gating_rows"] == live_tokens
    assert [shape[1] for shape in seen["gdn_shapes"]] == [live_tokens] * 5
    assert seen["conv_seq_lens_cpu"] is backend.forward_metadata.extend_seq_lens_cpu
    assert seen["gdn_cu_seqlens"] is backend.forward_metadata.query_start_loc
    assert seen["gdn_output_h"] is True
    assert output.shape[1] == live_tokens


@pytest.mark.skipif(
    not torch.cuda.is_available(), reason="CUDA is required for Triton copy kernel"
)
def test_mamba_state_copy_single_layer_rank3_interprets_first_dim_as_slot():
    pool = (
        torch.arange(6 * 2 * 3, device="cuda", dtype=torch.float32)
        .reshape(6, 2, 3)
        .clone()
    )
    original = pool.clone()
    src = torch.tensor([0, 3], device="cuda", dtype=torch.int32)
    dst = torch.tensor([1, 4], device="cuda", dtype=torch.int32)

    fused_mamba_state_copy(pool, src, dst, single_layer=True)
    torch.cuda.synchronize()

    assert torch.equal(pool[1], original[0])
    assert torch.equal(pool[4], original[3])
    assert torch.equal(pool[0], original[0])
    assert torch.equal(pool[3], original[3])


@pytest.mark.skipif(
    not torch.cuda.is_available(), reason="CUDA is required for Triton copy kernel"
)
def test_mamba_state_copy_full_pool_rank4_keeps_default_layer_slot_layout():
    pool = (
        torch.arange(2 * 6 * 2 * 3, device="cuda", dtype=torch.float32)
        .reshape(2, 6, 2, 3)
        .clone()
    )
    original = pool.clone()
    src = torch.tensor([0, 3], device="cuda", dtype=torch.int32)
    dst = torch.tensor([1, 4], device="cuda", dtype=torch.int32)

    fused_mamba_state_copy(pool, src, dst)
    torch.cuda.synchronize()

    assert torch.equal(pool[:, 1], original[:, 0])
    assert torch.equal(pool[:, 4], original[:, 3])
    assert torch.equal(pool[:, 0], original[:, 0])
    assert torch.equal(pool[:, 3], original[:, 3])
