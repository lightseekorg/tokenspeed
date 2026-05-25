from __future__ import annotations

import torch

from tokenspeed.runtime.execution.forward_batch_info import ForwardMode
from tokenspeed.runtime.layers.attention.backends.hybrid_linear_attn import (
    MambaAttnBackend,
    SimpleMambaPool,
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

    assert metadata.track_ssm_h_src.tolist() == [2]
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

    assert metadata.track_ssm_h_src.tolist() == [1]


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
