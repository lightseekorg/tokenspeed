from __future__ import annotations

import torch

from tokenspeed.runtime.execution.forward_batch_info import ForwardMode
from tokenspeed.runtime.layers.attention.backends.hybrid_linear_attn import (
    MambaAttnBackend,
    SimpleMambaPool,
)
from tokenspeed.runtime.layers.attention.linear.mamba2 import _composed_weight_loader


def _new_backend(
    page_size: int = 64,
    *,
    mamba_cache_chunk_size: int = 64,
    uses_mamba2_tracking: bool = False,
) -> MambaAttnBackend:
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
    backend.mamba_cache_chunk_size = mamba_cache_chunk_size
    backend.uses_mamba2_tracking = uses_mamba2_tracking
    return backend


def test_composed_weight_loader_transforms_loaded_parameter():
    observed = []

    def loader(param: torch.Tensor, loaded_weight: torch.Tensor) -> None:
        param.data.copy_(loaded_weight + 1)

    def transform(param: torch.Tensor) -> torch.Tensor:
        observed.append(param.detach().clone())
        return param * 2

    param = torch.empty(2)
    loaded_weight = torch.tensor([3.0, 5.0])

    _composed_weight_loader(loader, transform)(param, loaded_weight)

    torch.testing.assert_close(observed[0], torch.tensor([4.0, 6.0]))
    torch.testing.assert_close(param, torch.tensor([8.0, 12.0]))


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


def test_mamba2_tracking_skips_before_model_chunk_boundary():
    backend = _new_backend(
        page_size=64,
        mamba_cache_chunk_size=256,
        uses_mamba2_tracking=True,
    )

    backend.init_forward_metadata(
        bs=1,
        req_pool_indices=torch.tensor([0], dtype=torch.int32),
        seq_lens=torch.tensor([128], dtype=torch.int32),
        forward_mode=ForwardMode.EXTEND,
        mamba_pool_indices=torch.tensor([2], dtype=torch.int32),
        extend_prefix_lens=torch.tensor([0], dtype=torch.int32),
        mamba_track_pool_indices=torch.tensor([5], dtype=torch.int32),
        mamba_cache_chunk_size=256,
    )

    metadata = backend.forward_metadata

    assert metadata.track_ssm_h_dst is None
    assert metadata.track_ssm_final_dst is None
    assert metadata.mamba2_metadata.mixed_metadata.chunk_size == 256


def test_mamba2_tracking_uses_final_state_on_model_chunk_boundary():
    backend = _new_backend(
        page_size=64,
        mamba_cache_chunk_size=256,
        uses_mamba2_tracking=True,
    )

    backend.init_forward_metadata(
        bs=1,
        req_pool_indices=torch.tensor([0], dtype=torch.int32),
        seq_lens=torch.tensor([256], dtype=torch.int32),
        forward_mode=ForwardMode.EXTEND,
        mamba_pool_indices=torch.tensor([2], dtype=torch.int32),
        extend_prefix_lens=torch.tensor([0], dtype=torch.int32),
        mamba_track_pool_indices=torch.tensor([5], dtype=torch.int32),
        mamba_cache_chunk_size=256,
    )

    metadata = backend.forward_metadata

    assert metadata.track_ssm_h_dst is None
    assert metadata.track_ssm_final_src.tolist() == [2]
    assert metadata.track_ssm_final_dst.tolist() == [5]


def test_mamba2_tracking_uses_intermediate_state_after_model_chunk_boundary():
    backend = _new_backend(
        page_size=64,
        mamba_cache_chunk_size=256,
        uses_mamba2_tracking=True,
    )

    backend.init_forward_metadata(
        bs=1,
        req_pool_indices=torch.tensor([0], dtype=torch.int32),
        seq_lens=torch.tensor([300], dtype=torch.int32),
        forward_mode=ForwardMode.EXTEND,
        mamba_pool_indices=torch.tensor([2], dtype=torch.int32),
        extend_prefix_lens=torch.tensor([0], dtype=torch.int32),
        mamba_track_pool_indices=torch.tensor([5], dtype=torch.int32),
        mamba_cache_chunk_size=256,
    )

    metadata = backend.forward_metadata

    assert metadata.track_ssm_h_src.tolist() == [1]
    assert metadata.track_ssm_h_dst.tolist() == [5]
    assert metadata.track_ssm_final_dst is None


def test_mamba2_mixed_metadata_separates_prefill_and_decode_rows():
    backend = _new_backend(page_size=64, mamba_cache_chunk_size=4)

    backend.init_forward_metadata(
        bs=3,
        num_extends=1,
        req_pool_indices=torch.tensor([0, 1, 2], dtype=torch.int32),
        seq_lens=torch.tensor([7, 1, 1], dtype=torch.int32),
        forward_mode=ForwardMode.MIXED,
        mamba_pool_indices=torch.tensor([2, 3, 4], dtype=torch.int32),
        extend_prefix_lens=torch.tensor([3], dtype=torch.int32),
        mamba_track_pool_indices=torch.tensor([5, 6, 7], dtype=torch.int32),
    )

    metadata = backend.forward_metadata.mamba2_metadata

    assert metadata.num_prefills == 1
    assert metadata.num_prefill_tokens == 4
    assert metadata.num_decodes == 2
    assert metadata.query_start_loc.tolist() == [0, 4]
    assert metadata.mamba_cache_indices.tolist() == [2, 3, 4]
    assert metadata.mixed_metadata.extend_seq_lens_cpu.tolist() == [4]
    assert metadata.mixed_metadata.has_initial_states.tolist() == [True]
    assert metadata.mixed_metadata.seq_idx.tolist() == [[0, 0, 0, 0]]
    assert metadata.track_ssm_final_src.tolist() == [2]
    assert metadata.track_ssm_final_dst.tolist() == [5]
