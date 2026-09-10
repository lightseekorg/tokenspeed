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

from __future__ import annotations

from dataclasses import replace
from functools import partial
from types import SimpleNamespace

import pytest
import torch

from tokenspeed.runtime.configs.model_config import AttentionArch
from tokenspeed.runtime.execution.forward_batch_info import ForwardMode
from tokenspeed.runtime.execution.graph_ptr_guard import snapshot_graph_metadata
from tokenspeed.runtime.layers.attention.backends.specific.qsa_indexer import (
    QSAIndexerBackend,
)
from tokenspeed.runtime.layers.attention.backends.specific.qwen4_exp import (
    Qwen4ExpBackend,
)
from tokenspeed.runtime.layers.attention.configs.base import AttnConfig
from tokenspeed.runtime.layers.attention.configs.mha import MHAConfig
from tokenspeed.runtime.layers.attention.kv_cache.base import CachePool
from tokenspeed.runtime.layers.attention.kv_cache.qwen4_exp import (
    QWEN4_EXP_QSA_CACHE_GROUP,
    QWEN4_EXP_QSA_RECENT_CACHE_GROUP,
    qsa_compressed_field,
    qsa_raw_key_field,
    qsa_rope_position_field,
)
from tokenspeed.runtime.layers.attention.kv_cache.recipes.spec import FULL_ATTENTION
from tokenspeed.runtime.layers.attention.qsa.metadata import qsa_forward_layout
from tokenspeed.runtime.layers.attention.qsa.verify_state import QSAVerifyState
from tokenspeed.runtime.layers.attention.registry import (
    create_paged_router,
)


def _qsa_config(*, max_bs: int, is_draft: bool, device: str) -> AttnConfig:
    spec = MHAConfig(
        backend_name="mha",
        num_attention_heads=1,
        num_kv_heads=1,
        head_dim=2,
        attn_tp_size=1,
        cache_layer_types=(),
        sliding_window_tokens=None,
    )
    config = AttnConfig(
        device=device,
        dtype=torch.bfloat16,
        kv_cache_dtype=torch.bfloat16,
        kv_cache_quant_method="none",
        kv_cache_mxfp8=False,
        prefix_granularity=64,
        kernel_page_size=64,
        context_len=1024,
        max_bs=max_bs,
        pd_disaggregation_enabled=False,
        speculative_num_steps=0,
        speculative_num_draft_tokens=4,
        is_draft=is_draft,
        draft_block_decode=False,
        components=(spec,),
    )
    return config


def _qsa_pool(*, device: str, layer_offset: int) -> SimpleNamespace:
    groups = {
        FULL_ATTENTION: 256,
        QWEN4_EXP_QSA_CACHE_GROUP: 256,
        QWEN4_EXP_QSA_RECENT_CACHE_GROUP: 64,
    }
    tensors = {}
    # The adjacent draft layer must not get target staging or commits.
    for local_layer in (1, 3, 4):
        layer_id = layer_offset + local_layer
        tensors[qsa_raw_key_field(layer_id)] = torch.zeros(
            4, 4, 1, 8, dtype=torch.bfloat16, device=device
        )
        tensors[qsa_rope_position_field(layer_id)] = torch.zeros(
            4, 3, dtype=torch.int64, device=device
        )
        tensors[qsa_compressed_field(layer_id)] = torch.zeros(
            4, 64, 1, 8, dtype=torch.bfloat16, device=device
        )
    pool = SimpleNamespace(
        arena=SimpleNamespace(
            plan=SimpleNamespace(
                fields=[
                    SimpleNamespace(
                        group_id=(
                            QWEN4_EXP_QSA_CACHE_GROUP
                            if name.endswith(".qsa.compressed_key")
                            else QWEN4_EXP_QSA_RECENT_CACHE_GROUP
                        ),
                        field_id=name,
                    )
                    for name in tensors
                ]
            ),
            field=tensors.__getitem__,
            cache_group_specs=tuple(
                SimpleNamespace(
                    group_id=gid,
                    block_granularity=granularity,
                    family="history",
                    retention="full_history",
                )
                for gid, granularity in groups.items()
            ),
        ),
        paged_group_ids=tuple(groups),
        layer_num=4,
        _field_layer_offset=layer_offset,
        field_layer_range=range(layer_offset, layer_offset + 4),
    )
    pool._field_layer_id = partial(CachePool._field_layer_id, pool)
    return pool


@pytest.fixture
def state() -> QSAVerifyState:
    state = QSAVerifyState(
        _qsa_config(max_bs=8, is_draft=False, device="cpu"),
        _qsa_pool(device="cpu", layer_offset=0),
    )
    state.preallocate_verify_workspace(8, 4)
    return state


def _root_with_indexer(config, pool):
    router = create_paged_router(config, AttentionArch.MHA, backend_name="qsa")
    indexer = QSAIndexerBackend(config, router)
    root = Qwen4ExpBackend(config, router, None, indexer)
    root.set_cache_pool(pool)
    return root, indexer


@pytest.fixture
def commit_calls(monkeypatch: pytest.MonkeyPatch) -> list[tuple[tuple, dict]]:
    import tokenspeed.runtime.layers.attention.qsa.verify_state as module

    calls: list[tuple[tuple, dict]] = []

    def commit(*args, **kwargs):
        calls.append((args, kwargs))

    monkeypatch.setattr(module, "qwen4_exp_qsa_commit_verify_layers", commit)
    return calls


def test_qsa_preallocation_preserves_workspace_and_budget(state) -> None:
    workspace = state._verify_workspace
    expected_bytes = 2 * 8 * 4 * 8 * 2 + 8 * 4 * 36 + 2 * 2 * 8
    assert state.preallocate_verify_workspace(8, 4) == expected_bytes
    assert state._verify_workspace is workspace


def test_qsa_indexer_workspace_cannot_be_rebound(state) -> None:
    _, backend = _root_with_indexer(
        _qsa_config(max_bs=8, is_draft=False, device="cpu"), state.cache_pool
    )
    backend.preallocate_verify_workspace(8, 4)
    workspace = backend._verify_state._verify_workspace
    backend.set_cache_pool(state.cache_pool)
    with pytest.raises(RuntimeError, match="cannot be rebound"):
        backend.set_cache_pool(_qsa_pool(device="cpu", layer_offset=0))
    assert backend._verify_state._verify_workspace is workspace


@pytest.mark.parametrize("layer_offset", [0, 5])
def test_qsa_commit_uses_only_owned_layers_once(commit_calls, layer_offset) -> None:
    pool = _qsa_pool(device="cpu", layer_offset=layer_offset)
    state = QSAVerifyState(_qsa_config(max_bs=8, is_draft=False, device="cpu"), pool)
    state.preallocate_verify_workspace(8, 4)
    workspace = state._verify_workspace
    # Both eager batches and graph buckets use the same capacity-sized tensors.
    for bs in (8, 2):
        views = [state.verify_staging_buffers(layer, bs) for layer in (1, 3)]
        assert views[0][0].data_ptr() == workspace.token_k[0].data_ptr()
        assert views[1][0].data_ptr() == workspace.token_k[1].data_ptr()
        assert views[0][1].data_ptr() == views[1][1].data_ptr()
        assert views[0][0].shape == (bs, 4, 1, 8)
        state.commit_after_mtp_verify(
            torch.tensor([9] + [3] * bs, dtype=torch.int32), num_extends=1
        )
    assert len(commit_calls) == 2
    for args, kwargs in commit_calls:
        assert args[0].tolist() == [
            pool.arena.field(qsa_raw_key_field(layer_offset + layer)).data_ptr()
            for layer in (1, 3)
        ]
        assert args[1].tolist() == [
            pool.arena.field(qsa_rope_position_field(layer_offset + layer)).data_ptr()
            for layer in (1, 3)
        ]
        assert args[2] is workspace.token_k
        assert kwargs == {"verify_width": 4}
    assert commit_calls[1][0][3].shape == (8,)
    assert commit_calls[1][0][5].shape == (8, 3)
    assert commit_calls[1][0][6].tolist() == [3, 3]
    assert commit_calls[0][0][0] is commit_calls[1][0][0]


@pytest.mark.parametrize("bs", [0, 9])
def test_qsa_staging_rejects_invalid_capacity(state, bs) -> None:
    with pytest.raises(RuntimeError, match="preallocated shape"):
        state.verify_staging_buffers(1, bs)


def test_qsa_preallocation_rejects_a_different_verify_width(state) -> None:
    with pytest.raises(ValueError, match="workspace width"):
        state.preallocate_verify_workspace(8, 3)


def test_qsa_staging_requires_preallocation() -> None:
    state = QSAVerifyState(
        _qsa_config(max_bs=2, is_draft=False, device="cpu"),
        _qsa_pool(device="cpu", layer_offset=0),
    )
    with pytest.raises(RuntimeError, match="must be preallocated"):
        state.verify_staging_buffers(1, 2)


@pytest.mark.parametrize("layer_id, error", [(0, KeyError), (4, ValueError)])
def test_qsa_staging_rejects_layers_outside_its_fields(state, layer_id, error) -> None:
    with pytest.raises(error):
        state.verify_staging_buffers(layer_id, 2)


def test_qsa_commit_without_target_staging_is_silent(commit_calls) -> None:
    state = QSAVerifyState(
        _qsa_config(max_bs=2, is_draft=False, device="cpu"),
        _qsa_pool(device="cpu", layer_offset=0),
    )
    state.preallocate_verify_workspace(2, 4)
    state.commit_after_mtp_verify(torch.tensor([2], dtype=torch.int32), num_extends=0)
    assert commit_calls == []


@pytest.mark.parametrize("num_extends", [-1, 3])
def test_qsa_commit_rejects_invalid_extend_prefix(state, num_extends) -> None:
    with pytest.raises(ValueError, match="invalid extend prefix"):
        state.commit_after_mtp_verify(
            torch.tensor([1, 2], dtype=torch.int32), num_extends=num_extends
        )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
@pytest.mark.parametrize("use_graph", [False, True])
def test_qsa_state_refreshes_layout_and_commits_live_verify_rows(
    use_graph: bool,
) -> None:
    pool = _qsa_pool(device="cuda", layer_offset=0)
    backend, indexer = _root_with_indexer(
        _qsa_config(max_bs=4, is_draft=False, device="cuda"), pool
    )
    indexer.preallocate_verify_workspace(4, 4)
    state = indexer._verify_state
    backend.init_cuda_graph_state(4)
    raws = [pool.arena.field(qsa_raw_key_field(layer)) for layer in (1, 3)]
    positions = [pool.arena.field(qsa_rope_position_field(layer)) for layer in (1, 3)]
    workspace = state._verify_workspace
    source = torch.arange(2 * 8 * 8, dtype=torch.float32, device="cuda").reshape(
        2, 8, 1, 8
    )
    source = source.to(torch.bfloat16)
    seq_lens = torch.tensor([4, 12], dtype=torch.int32, device="cuda")
    slots = torch.tensor([1, 2], dtype=torch.int32, device="cuda")
    tables = {
        gid: torch.tensor([[1], [2]], dtype=torch.int32, device="cuda")
        for gid in pool.paged_group_ids
    }
    ctx = SimpleNamespace(bs=2, forward_mode=ForwardMode.DECODE, attn_backend=backend)

    def refresh(actual_bs: int) -> None:
        backend.sparse_topk.qsa_metadata = object()
        backend.refresh_decode_metadata(
            2,
            actual_bs,
            slots,
            seq_lens,
            forward_mode=ForwardMode.DECODE,
            block_tables=tables,
            num_extends=0,
            for_graph_replay=use_graph,
        )
        assert backend.sparse_topk.qsa_metadata is None

    def stage() -> None:
        layout = qsa_forward_layout(
            ctx,
            8,
            compressed_token_page_size=256,
            recent_page_size=64,
            compress_ratio=4,
            reset_draft_tags=None,
        )
        positions = layout.logical_positions[:, None].expand(-1, 3)
        for slot, layer in enumerate((1, 3)):
            destinations = indexer.verify_staging_buffers(layer, 2)
            destinations[0].copy_(source[slot].view(2, 4, 1, 8))
            destinations[1].copy_(positions.view(2, 4, 3))
            destinations[2].copy_(layout.logical_positions.view(2, 4))
            destinations[3].copy_(layout.recent_locs.view(2, 4))

    for _ in range(2):
        refresh(2)
        stage()
    torch.cuda.synchronize()
    graph = None
    if use_graph:
        refresh(2)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            stage()

    expected_keys = [raw.cpu().clone() for raw in raws]
    expected_positions = [position.cpu().clone() for position in positions]
    for lengths, accepted, actual_bs in [([6, 14], [1, 3], 2), ([10, 18], [4, 0], 1)]:
        seq_lens.copy_(torch.tensor(lengths, dtype=torch.int32, device="cuda"))
        source.add_(1)
        refresh(actual_bs)
        if graph is None:
            stage()
        else:
            # Replay fills staging on-device; its Python accessor is not called.
            # Both rounds must commit even without another host-side staging call.
            graph.replay()
        backend.commit_speculative_state_after_verify(
            torch.tensor(accepted, dtype=torch.int32, device="cuda"), num_extends=0
        )
        assert state._verify_workspace is workspace
        source_cpu = source.cpu()
        for layer, (raw, position_cache) in enumerate(
            zip(raws, positions, strict=True)
        ):
            for request, count in enumerate(accepted):
                for step in range(count):
                    position = lengths[request] - 4 + step
                    expected_keys[layer][request + 1, position % 4] = source_cpu[
                        layer, request * 4 + step
                    ]
                    if position % 4 == 0:
                        expected_positions[layer][request + 1].fill_(position)
            torch.testing.assert_close(raw.cpu(), expected_keys[layer], atol=0, rtol=0)
            torch.testing.assert_close(
                position_cache.cpu(), expected_positions[layer], atol=0, rtol=0
            )
    assert torch.count_nonzero(pool.arena.field(qsa_raw_key_field(4))) == 0
    assert torch.count_nonzero(pool.arena.field(qsa_rope_position_field(4))) == 0


@pytest.mark.parametrize("is_draft,width", [(False, 1), (True, 4)])
def test_qsa_only_target_verification_creates_state(
    is_draft, width, commit_calls
) -> None:
    config = replace(
        _qsa_config(max_bs=8, is_draft=is_draft, device="cpu"),
        speculative_num_draft_tokens=width,
    )
    root, backend = _root_with_indexer(config, _qsa_pool(device="cpu", layer_offset=0))
    assert backend._verify_state is None
    with pytest.raises(ValueError, match="speculative target"):
        QSAVerifyState(config, _qsa_pool(device="cpu", layer_offset=0))
    assert backend.preallocate_verify_workspace(8, width) == 0
    with pytest.raises(RuntimeError, match="speculative target"):
        backend.verify_staging_buffers(1, 2)
    root.commit_speculative_state_after_verify(
        torch.tensor([3, 1], dtype=torch.int32), num_extends=0
    )
    assert commit_calls == []


def test_qsa_raw_tables_refresh_in_place_and_clear_padding() -> None:
    root, indexer = _root_with_indexer(
        _qsa_config(max_bs=4, is_draft=False, device="cpu"),
        _qsa_pool(device="cpu", layer_offset=0),
    )
    root.init_cuda_graph_state(4)
    tables = {
        gid: torch.tensor([[2, -1, 3], [4, 5, -1]], dtype=torch.int32)
        for gid in (
            FULL_ATTENTION,
            QWEN4_EXP_QSA_CACHE_GROUP,
            QWEN4_EXP_QSA_RECENT_CACHE_GROUP,
        )
    }
    slots = torch.arange(3, dtype=torch.int32)
    root.refresh_decode_metadata(
        3,
        2,
        slots,
        torch.tensor([1, 9, 1], dtype=torch.int32),
        forward_mode=ForwardMode.DECODE,
        block_tables=tables,
    )
    metadata = indexer.forward_decode_metadata
    assert metadata.seq_lens.tolist() == [4, 9, 4]
    assert metadata.qsa_block_table.tolist() == [[2, 0, 3, 0], [4, 5, 0, 0], [0] * 4]
    assert metadata.recent_block_table[:, :3].tolist() == [
        [2, 0, 3],
        [4, 5, 0],
        [0, 0, 0],
    ]
    snapshot = snapshot_graph_metadata(indexer)
    assert any("qsa_block_table" in name for name in snapshot)
    assert any("recent_block_table" in name for name in snapshot)
    root.refresh_decode_metadata(
        3,
        1,
        slots,
        torch.tensor([7, 1, 1], dtype=torch.int32),
        forward_mode=ForwardMode.DECODE,
        block_tables={gid: torch.tensor([[6]], dtype=torch.int32) for gid in tables},
    )
    assert snapshot_graph_metadata(indexer) == snapshot
    assert indexer.forward_decode_metadata is metadata
    assert metadata.qsa_block_table.tolist() == [[6, 0, 0, 0], [0] * 4, [0] * 4]
    assert torch.count_nonzero(metadata.recent_block_table[:, 1:]) == 0
    root.refresh_decode_metadata(
        3,
        0,
        slots,
        torch.ones(3, dtype=torch.int32),
        forward_mode=ForwardMode.IDLE,
        block_tables={},
    )
    assert torch.count_nonzero(metadata.qsa_block_table) == 0
    assert torch.count_nonzero(metadata.recent_block_table) == 0
    assert snapshot_graph_metadata(indexer) == snapshot


def test_qsa_indexer_requires_both_live_tables() -> None:
    root, indexer = _root_with_indexer(
        _qsa_config(max_bs=4, is_draft=False, device="cpu"),
        _qsa_pool(device="cpu", layer_offset=0),
    )
    root.init_cuda_graph_state(4)
    with pytest.raises(RuntimeError, match="missing cache groups"):
        indexer.refresh_decode_metadata(
            1,
            1,
            torch.tensor([0]),
            torch.tensor([4], dtype=torch.int32),
            forward_mode=ForwardMode.DECODE,
            block_tables={
                QWEN4_EXP_QSA_CACHE_GROUP: torch.ones((1, 1), dtype=torch.int32)
            },
        )


def test_qsa_draft_narrowing_preserves_layout_and_updates_the_frontier(
    monkeypatch,
) -> None:
    import tokenspeed.runtime.layers.attention.qsa.metadata as metadata_module
    from tokenspeed.runtime.layers.attention.qsa.indexer import QSAIndexer

    root, indexer = _root_with_indexer(
        _qsa_config(max_bs=4, is_draft=True, device="cpu"),
        _qsa_pool(device="cpu", layer_offset=0),
    )
    root.init_cuda_graph_state(4)
    tables = {
        gid: torch.tensor([[1], [2]], dtype=torch.int32)
        for gid in (QWEN4_EXP_QSA_CACHE_GROUP, QWEN4_EXP_QSA_RECENT_CACHE_GROUP)
    }
    seq_lens = torch.tensor([8, 12], dtype=torch.int32)
    indexer.init_forward_metadata(
        2,
        1,
        torch.arange(2),
        seq_lens,
        ForwardMode.MIXED,
        block_tables=tables,
        extend_seq_lens=torch.tensor([3], dtype=torch.int32),
        extend_seq_lens_cpu=torch.tensor([3], dtype=torch.int32),
        extend_prefix_lens=torch.tensor([5], dtype=torch.int32),
        extend_prefix_lens_cpu=torch.tensor([5], dtype=torch.int32),
        extend_with_prefix=True,
    )
    extend = indexer.forward_extend_metadata
    indexer.refresh_decode_metadata(
        2,
        2,
        torch.arange(2),
        seq_lens,
        forward_mode=ForwardMode.DECODE,
        block_tables=tables,
    )
    assert indexer.forward_extend_metadata is extend
    assert extend.extend_seq_lens.tolist() == [3, 4]
    launches = []

    def prepare(*args, **kwargs):
        launches.append((args, kwargs))
        if len(launches) == 1:
            assert args[0].tolist() == [8, 12]
            assert args[1].tolist() == [3, 4]
            logical = torch.tensor([5, 6, 7, 8, 9, 10, 11])
            requests = torch.tensor([0, 0, 0, 1, 1, 1, 1])
        else:
            assert args[0].tolist() == [8, 10]
            assert args[1] == 1
            logical = torch.tensor([7, 9])
            requests = torch.tensor([0, 1])
        slots = torch.ones(logical.shape, dtype=torch.int32)
        return logical, requests, slots, slots, slots

    monkeypatch.setattr(metadata_module, "qwen4_exp_qsa_prepare_metadata", prepare)
    ctx = SimpleNamespace(
        attn_backend=root, bs=2, num_extends=1, forward_mode=ForwardMode.MIXED
    )
    kwargs = dict(
        compressed_token_page_size=256,
        recent_page_size=64,
        compress_ratio=4,
        reset_draft_tags=None,
    )
    layout = qsa_forward_layout(ctx, 7, **kwargs)
    root.advance_draft_forward_metadata(torch.tensor([8, 10], dtype=torch.int32))
    assert layout.seq_lens.tolist() == [8, 10]
    assert layout.logical_positions.tolist() == [5, 6, 7, 8, 9, 10, 11]
    mask = QSAIndexer._draft_accepted_write_mask(
        ctx,
        layout.seq_lens,
        layout.logical_positions,
        layout.request_indices,
        layout.recent_locs,
    )
    assert mask.tolist() == [True, True, True, True, True, False, False]
    assert qsa_forward_layout(ctx, 7, **kwargs) is layout
    assert len(launches) == 1
    # Eagle starts each model invocation with fresh row geometry while carrying
    # forward the selected slots. The narrowed decode has one query per request.
    topk = torch.tensor([[1], [2]])
    root.sparse_topk.decode = topk
    root.sparse_topk.qsa_metadata = None
    ctx.forward_mode = ForwardMode.DECODE
    next_layout = qsa_forward_layout(ctx, 2, **kwargs)
    assert next_layout.logical_positions.tolist() == [7, 9]
    assert root.sparse_topk.decode is topk
    root.update_draft_forward_metadata(torch.tensor([9, 11], dtype=torch.int32))
    assert next_layout.seq_lens.tolist() == [9, 11]
    root.fill_block_decode_seq_lens(2, torch.tensor([2, 9999], dtype=torch.int32))
    assert next_layout.seq_lens.tolist() == [4, 1024]
