from __future__ import annotations

import pytest
import torch

from tokenspeed.runtime.execution.forward_batch_info import ForwardMode
from tokenspeed.runtime.layers.attention.backends.hybrid_linear_attn import (
    MambaAttnBackend,
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
