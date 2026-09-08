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

from functools import partial
from types import SimpleNamespace

import pytest
import torch

from tokenspeed.runtime.configs.model_config import AttentionArch
from tokenspeed.runtime.execution.forward_batch_info import ForwardMode
from tokenspeed.runtime.layers.attention.backends.paged.qsa import QSAAttnBackend
from tokenspeed.runtime.layers.attention.backends.paged.router import CacheGroupRouter
from tokenspeed.runtime.layers.attention.configs.base import AttnConfig
from tokenspeed.runtime.layers.attention.configs.mha import MHAConfig
from tokenspeed.runtime.layers.attention.kv_cache.base import CachePool
from tokenspeed.runtime.layers.attention.kv_cache.qwen4_exp import (
    QWEN4_EXP_QSA_CACHE_GROUP,
    QWEN4_EXP_QSA_RECENT_CACHE_GROUP,
    qsa_raw_key_field,
    qsa_rope_position_field,
)
from tokenspeed.runtime.layers.attention.kv_cache.recipes.spec import FULL_ATTENTION
from tokenspeed.runtime.layers.attention.qsa.metadata import qsa_forward_layout
from tokenspeed.runtime.layers.attention.qsa.runtime import QSAIndexerRuntime
from tokenspeed.runtime.layers.attention.registry import (
    _create_indexer_runtime,
    _prepare_verify_workspace,
    create_paged_router,
)


def _qsa_config(*, max_bs: int, is_draft: bool, device: str) -> AttnConfig:
    spec = MHAConfig(
        backend_name="mha",
        num_attention_heads=1,
        num_kv_heads=1,
        head_dim=2,
        attn_tp_size=1,
        layer_types=(),
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


def _make_qsa_backend(*, max_bs: int, is_draft: bool, device: str) -> CacheGroupRouter:
    return create_paged_router(
        _qsa_config(max_bs=max_bs, is_draft=is_draft, device=device),
        AttentionArch.MHA,
        backend_name="qsa",
    )


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
    pool = SimpleNamespace(
        arena=SimpleNamespace(
            plan=SimpleNamespace(
                fields=[
                    SimpleNamespace(
                        group_id=QWEN4_EXP_QSA_RECENT_CACHE_GROUP,
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
def runtime() -> QSAIndexerRuntime:
    runtime = _create_indexer_runtime(
        _qsa_config(max_bs=8, is_draft=False, device="cpu"),
        _qsa_pool(device="cpu", layer_offset=0),
    )
    _prepare_verify_workspace(
        server_args=SimpleNamespace(speculative_num_draft_tokens=4),
        config=SimpleNamespace(max_bs=8),
        backend=None,
        draft_backend=None,
        speculative_states=(runtime,),
        uses_paged_state_verify=False,
        is_inkling=False,
        expected_bytes=2 * 8 * 4 * 8 * 2 + 8 * 4 * 36 + 2 * 2 * 8,
    )
    return runtime


@pytest.fixture
def commit_calls(monkeypatch: pytest.MonkeyPatch) -> list[tuple[tuple, dict]]:
    import tokenspeed.runtime.layers.attention.qsa.runtime as module

    calls: list[tuple[tuple, dict]] = []

    def commit(*args, **kwargs):
        calls.append((args, kwargs))

    monkeypatch.setattr(module, "qwen4_exp_qsa_commit_verify_layers", commit)
    return calls


def test_qsa_registry_preallocates_from_the_cache_plan(runtime) -> None:
    router = _make_qsa_backend(max_bs=8, is_draft=False, device="cpu")
    router.set_cache_pool(runtime.cache_pool)
    assert not hasattr(router, "runtime")
    assert not hasattr(router, "commit_speculative_state_after_verify")
    assert all(isinstance(leaf, QSAAttnBackend) for leaf in router.leaves.values())
    workspace = runtime._verify_workspace
    assert workspace.token_k.shape == (2, 8, 4, 1, 8)
    assert workspace.position_values.shape == (8, 4, 3)
    assert workspace.logical_positions.shape == workspace.recent_locs.shape == (8, 4)
    expected_bytes = 2 * 8 * 4 * 8 * 2 + 8 * 4 * 36 + 2 * 2 * 8
    assert runtime.preallocate_verify_workspace(8, 4) == expected_bytes
    assert runtime._verify_workspace is workspace


@pytest.mark.parametrize("layer_offset", [0, 5])
def test_qsa_commit_uses_only_owned_layers_once(commit_calls, layer_offset) -> None:
    pool = _qsa_pool(device="cpu", layer_offset=layer_offset)
    runtime = QSAIndexerRuntime(
        _qsa_config(max_bs=8, is_draft=False, device="cpu"), pool
    )
    runtime.preallocate_verify_workspace(8, 4)
    workspace = runtime._verify_workspace
    # Both eager batches and graph buckets use the same capacity-sized tensors.
    for bs in (8, 2):
        views = [runtime.verify_staging_buffers(layer, bs) for layer in (1, 3)]
        assert views[0][0].data_ptr() == workspace.token_k[0].data_ptr()
        assert views[1][0].data_ptr() == workspace.token_k[1].data_ptr()
        assert views[0][1].data_ptr() == views[1][1].data_ptr()
        assert views[0][0].shape == (bs, 4, 1, 8)
        runtime.commit_after_verify(
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
def test_qsa_staging_rejects_invalid_capacity(runtime, bs) -> None:
    with pytest.raises(RuntimeError, match="preallocated shape"):
        runtime.verify_staging_buffers(1, bs)


def test_qsa_staging_rejects_changed_verify_width(runtime) -> None:
    runtime.spec_num_tokens = 3
    with pytest.raises(RuntimeError, match="preallocated shape"):
        runtime.verify_staging_buffers(1, 2)


def test_qsa_staging_requires_preallocation() -> None:
    runtime = QSAIndexerRuntime(
        _qsa_config(max_bs=2, is_draft=False, device="cpu"),
        _qsa_pool(device="cpu", layer_offset=0),
    )
    with pytest.raises(RuntimeError, match="must be preallocated"):
        runtime.verify_staging_buffers(1, 2)


@pytest.mark.parametrize("layer_id, error", [(0, KeyError), (4, ValueError)])
def test_qsa_staging_rejects_layers_outside_its_fields(
    runtime, layer_id, error
) -> None:
    with pytest.raises(error):
        runtime.verify_staging_buffers(layer_id, 2)


@pytest.mark.parametrize("is_draft", [False, True])
def test_qsa_commit_without_target_staging_is_silent(commit_calls, is_draft) -> None:
    runtime = QSAIndexerRuntime(
        _qsa_config(max_bs=2, is_draft=is_draft, device="cpu"),
        _qsa_pool(device="cpu", layer_offset=0),
    )
    nbytes = runtime.preallocate_verify_workspace(2, 4)
    assert (nbytes == 0) == is_draft
    runtime.commit_after_verify(torch.tensor([2], dtype=torch.int32), num_extends=0)
    assert commit_calls == []


@pytest.mark.parametrize("num_extends", [-1, 3])
def test_qsa_commit_rejects_invalid_extend_prefix(runtime, num_extends) -> None:
    with pytest.raises(ValueError, match="invalid extend prefix"):
        runtime.commit_after_verify(
            torch.tensor([1, 2], dtype=torch.int32), num_extends=num_extends
        )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
@pytest.mark.parametrize("use_graph", [False, True])
def test_qsa_runtime_refreshes_layout_and_commits_live_verify_rows(
    use_graph: bool,
) -> None:
    router = _make_qsa_backend(max_bs=4, is_draft=False, device="cuda")
    pool = _qsa_pool(device="cuda", layer_offset=0)
    router.set_cache_pool(pool)
    runtime = QSAIndexerRuntime(
        _qsa_config(max_bs=4, is_draft=False, device="cuda"), pool
    )
    runtime.preallocate_verify_workspace(4, 4)
    router.init_cuda_graph_state(4)
    raws = [pool.arena.field(qsa_raw_key_field(layer)) for layer in (1, 3)]
    positions = [pool.arena.field(qsa_rope_position_field(layer)) for layer in (1, 3)]
    workspace = runtime._verify_workspace
    source = torch.arange(2 * 8 * 8, dtype=torch.float32, device="cuda").reshape(
        2, 8, 1, 8
    )
    source = source.to(torch.bfloat16)
    seq_lens = torch.tensor([4, 12], dtype=torch.int32, device="cuda")
    slots = torch.tensor([1, 2], dtype=torch.int32, device="cuda")
    tables = {
        gid: torch.tensor([[1], [2]], dtype=torch.int32, device="cuda")
        for gid in router.group_ids
    }
    ctx = SimpleNamespace(bs=2, forward_mode=ForwardMode.DECODE, attn_backend=router)

    def refresh(actual_bs: int) -> None:
        router.refresh_decode_metadata(
            2,
            actual_bs,
            slots,
            seq_lens,
            forward_mode=ForwardMode.DECODE,
            block_tables=tables,
            num_extends=0,
            for_graph_replay=use_graph,
        )

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
            destinations = runtime.verify_staging_buffers(layer, 2)
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
            graph.replay()
        runtime.commit_after_verify(
            torch.tensor(accepted, dtype=torch.int32, device="cuda"), num_extends=0
        )
        assert runtime._verify_workspace is workspace
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


def test_qsa_without_speculation_needs_no_verify_workspace() -> None:
    import dataclasses

    config = dataclasses.replace(
        _qsa_config(max_bs=8, is_draft=False, device="cpu"),
        speculative_num_draft_tokens=1,
    )
    runtime = _create_indexer_runtime(config, _qsa_pool(device="cpu", layer_offset=0))
    _prepare_verify_workspace(
        server_args=SimpleNamespace(speculative_num_draft_tokens=None),
        config=config,
        backend=None,
        draft_backend=None,
        speculative_states=(runtime,),
        uses_paged_state_verify=False,
        is_inkling=False,
        expected_bytes=0,
    )
    assert runtime._verify_workspace is None


def test_qsa_runtime_requires_fields_in_the_side_view() -> None:
    config = _qsa_config(max_bs=8, is_draft=False, device="cpu")
    pool = _qsa_pool(device="cpu", layer_offset=0)
    pool.paged_group_ids = (FULL_ATTENTION,)
    assert _create_indexer_runtime(config, pool) is None
    assert _create_indexer_runtime(None, None) is None


def test_qsa_workspace_budget_mismatch_fails_before_forward(runtime) -> None:
    with pytest.raises(RuntimeError, match="planned execution verify workspace"):
        _prepare_verify_workspace(
            server_args=SimpleNamespace(speculative_num_draft_tokens=4),
            config=SimpleNamespace(max_bs=8),
            backend=None,
            draft_backend=None,
            speculative_states=(runtime,),
            uses_paged_state_verify=False,
            is_inkling=False,
            expected_bytes=runtime._verify_workspace.nbytes - 1,
        )
