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

from types import SimpleNamespace

import pytest
import torch

import tokenspeed.runtime.layers.attention.backends.specific.qwen4_exp_ple as ple_module
from tokenspeed.runtime.execution.forward_batch_info import ForwardMode
from tokenspeed.runtime.layers.attention.backends.specific.qwen4_exp_ple import (
    Qwen4ExpPLEBackend,
)
from tokenspeed.runtime.layers.attention.backends.state.mamba import MambaAttnBackend
from tokenspeed.runtime.layers.attention.kv_cache.qwen4_exp import (
    QWEN4_EXP_PLE_CACHE_GROUP,
    qwen4_exp_ple_context_field,
    qwen4_exp_ple_conv_field,
)


def _make_backend(width: int, is_draft: bool, device: str):
    fields = {
        qwen4_exp_ple_context_field(0): torch.zeros(16, 2, dtype=torch.int64),
        qwen4_exp_ple_conv_field(0): torch.zeros(16, 4, 3, dtype=torch.bfloat16),
        qwen4_exp_ple_conv_field(2): torch.zeros(16, 4, 3, dtype=torch.bfloat16),
        # The merged arena can hold another view's layers.
        qwen4_exp_ple_conv_field(4): torch.zeros(16, 4, 3, dtype=torch.bfloat16),
    }
    fields = {field_id: field.to(device) for field_id, field in fields.items()}
    config = SimpleNamespace(
        device=torch.device(device),
        dtype=torch.bfloat16,
        is_draft=is_draft,
        speculative_num_draft_tokens=width,
    )
    spec = SimpleNamespace(
        num_attention_heads=1,
        num_kv_heads=1,
        attn_tp_size=1,
        head_dim=8,
    )
    result = Qwen4ExpPLEBackend(config, spec)
    result.set_cache_pool(
        SimpleNamespace(
            field_layer_range=range(3),
            _field_layer_id=lambda layer: layer,
            arena=SimpleNamespace(
                plan=SimpleNamespace(
                    fields=tuple(
                        SimpleNamespace(
                            group_id=QWEN4_EXP_PLE_CACHE_GROUP, field_id=field_id
                        )
                        for field_id in fields
                    )
                ),
                field=fields.__getitem__,
                runtime_contract=SimpleNamespace(
                    group_specs=(
                        SimpleNamespace(
                            group_id=QWEN4_EXP_PLE_CACHE_GROUP,
                            family="state",
                            checkpoint_granularity=4,
                        ),
                    )
                ),
            ),
        )
    )
    result.preallocate_verify_workspace(4, width)
    result.init_cuda_graph_state(4)
    return result


@pytest.fixture
def backend():
    return _make_backend(3, False, "cpu")


def test_ple_rebind_rejection_preserves_verify_workspace(backend):
    pool = backend.cache_pool
    scratch = backend._ple_verify_scratch
    replacement = _make_backend(3, False, "cpu").cache_pool

    backend.set_cache_pool(pool)
    with pytest.raises(RuntimeError, match="cannot be rebound"):
        backend.set_cache_pool(replacement)

    assert backend.cache_pool is pool
    assert backend._ple_verify_scratch is scratch


def test_ple_invalid_fields_do_not_publish_a_pool(backend):
    pool = backend.cache_pool
    pool.arena.plan.fields = tuple(
        field
        for field in pool.arena.plan.fields
        if not field.field_id.endswith(".ple.context")
    )
    unbound = Qwen4ExpPLEBackend.__new__(Qwen4ExpPLEBackend)
    unbound._init_pool_binding()
    unbound.is_draft = False

    with pytest.raises(RuntimeError, match="one shared context field"):
        unbound.set_cache_pool(pool)

    assert unbound.cache_pool is None


def _verify(backend):
    backend.refresh_decode_metadata(
        4,
        2,
        torch.arange(4, device=backend.device),
        torch.tensor([6, 7, 1, 1], dtype=torch.int32, device=backend.device),
        forward_mode=ForwardMode.DECODE,
        block_tables={
            QWEN4_EXP_PLE_CACHE_GROUP: torch.tensor(
                [[1, 2, 3], [4, 5, 6]], dtype=torch.int32, device=backend.device
            )
        },
    )


def test_ple_capture_and_verify_use_own_stable_metadata(backend, monkeypatch):
    backend.init_forward_metadata_capture_cuda_graph(
        4,
        torch.arange(4),
        torch.ones(4, dtype=torch.int32),
        ForwardMode.DECODE,
        block_tables={},
    )
    captured = backend.forward_metadata
    assert captured.input_blocks.tolist() == [-1] * 4
    assert captured.output_blocks.tolist() == [-1] * 4
    assert captured.query_lengths == [3] * 4
    assert captured.verify_width == 3
    assert backend._verify_commit_ctx is None
    _verify(backend)
    assert backend.forward_metadata is captured
    assert captured.input_blocks.tolist() == [1, 4, -1, -1]
    assert captured.output_blocks.tolist() == [-1] * 4
    rows_calls = []
    monkeypatch.setattr(
        ple_module,
        "state_verify_commit_rows",
        lambda *a, **k: rows_calls.append((a, k)),
    )
    monkeypatch.setattr(ple_module, "copy_state_rows", lambda *a, **k: None)
    accepted = torch.tensor([0, 9], dtype=torch.int32)
    backend.commit_verified_state(accepted)
    backend.commit_verified_state(accepted)
    assert len(rows_calls) == 1
    steps, pages, _, _ = rows_calls[0][0]
    assert steps.tolist() == [1, 3]
    assert pages.tolist() == [1, 5]
    assert rows_calls[0][1] == {
        "verify_width": 3,
        "num_layers": 2,
        "group_indices": None,
    }
    assert backend._verify_commit_ctx is None


@pytest.mark.parametrize(
    "mode", [ForwardMode.EXTEND, ForwardMode.MIXED, ForwardMode.IDLE]
)
def test_ple_non_verify_metadata_disarms_previous_round(backend, mode, monkeypatch):
    _verify(backend)
    prefix = torch.tensor([4, 6], dtype=torch.int32)
    lengths = torch.tensor([3, 3], dtype=torch.int32)
    backend.init_forward_metadata(
        2,
        2 if mode.is_extend() else 1,
        torch.arange(2),
        torch.tensor([7, 9], dtype=torch.int32),
        mode,
        block_tables={
            QWEN4_EXP_PLE_CACHE_GROUP: torch.tensor(
                [[1, 2, 3], [4, 5, 6]], dtype=torch.int32
            )
        },
        extend_seq_lens=lengths,
        extend_seq_lens_cpu=lengths,
        extend_prefix_lens=prefix,
        extend_prefix_lens_cpu=prefix,
        extend_replay_lens_cpu=torch.zeros_like(prefix),
        extend_prompt_lens_cpu=prefix + lengths,
        extend_with_prefix=True,
    )
    if mode != ForwardMode.IDLE:
        metadata = backend.forward_metadata
        assert metadata.input_blocks.tolist() == [1, 5]
        assert metadata.output_blocks.tolist() == [2, 6]
        assert metadata.query_lengths == [3, 3]
        assert metadata.verify_width is None
    assert backend._verify_commit_ctx is None
    monkeypatch.setattr(
        ple_module,
        "state_verify_commit_rows",
        lambda *a, **k: pytest.fail("stale PLE commit"),
    )
    backend.commit_verified_state(torch.tensor([2, 3], dtype=torch.int32))


def test_ple_idle_refresh_and_failed_refresh_disarm_verify(backend):
    _verify(backend)
    backend.refresh_decode_metadata(
        4,
        0,
        torch.arange(4),
        torch.ones(4, dtype=torch.int32),
        forward_mode=ForwardMode.IDLE,
        block_tables={},
    )
    assert backend._verify_commit_ctx is None
    assert backend.forward_metadata.input_blocks.tolist() == [-1] * 4
    _verify(backend)
    with pytest.raises(RuntimeError, match="missing the PLE cache group"):
        backend.refresh_decode_metadata(
            4,
            2,
            torch.arange(4),
            torch.ones(4, dtype=torch.int32),
            forward_mode=ForwardMode.DECODE,
            block_tables={},
        )
    assert backend._verify_commit_ctx is None
    assert backend.forward_metadata is None


def test_ple_preallocation_preserves_workspace_and_budget(backend):
    scratch = backend.ple_verify_scratch(qwen4_exp_ple_context_field(0), 0, 4)
    allocated = backend.preallocate_verify_workspace(4, 3)
    assert allocated == 16 * (16 + 2 * 24) + 2 * 4 * 2 * 8
    assert backend.preallocate_verify_workspace(2, 3) == allocated
    for bs in (1, 2, 4):
        views = backend.ple_verify_scratch(qwen4_exp_ple_context_field(0), 0, bs)
        for view, full in zip(views, scratch, strict=True):
            assert view.shape == (bs * 4, *full.shape[1:])
            assert view.data_ptr() == full.data_ptr()
    with pytest.raises(RuntimeError, match="preallocated capacity"):
        backend.ple_verify_scratch(qwen4_exp_ple_context_field(0), 0, 5)
    with pytest.raises(RuntimeError, match="preallocated capacity"):
        backend.preallocate_verify_workspace(5, 3)
    with pytest.raises(ValueError, match="width differs"):
        backend.preallocate_verify_workspace(4, 2)


def test_mamba_only_claims_its_recurrent_groups():
    backend = object.__new__(MambaAttnBackend)
    backend._init_pool_binding()
    backend.set_kv_pool(
        SimpleNamespace(
            state_group_by_layer={1: "gdn"},
            get_component=lambda *args: torch.zeros(4, 2),
            arena=SimpleNamespace(
                runtime_contract=SimpleNamespace(
                    prefix_granularity=16,
                    group_specs=(
                        SimpleNamespace(
                            group_id="ple", family="state", checkpoint_granularity=4
                        ),
                        SimpleNamespace(
                            group_id="gdn", family="state", checkpoint_granularity=8
                        ),
                    ),
                )
            ),
        )
    )
    assert backend._state_group_ids == ("gdn",)
    assert backend._checkpoint_granularity == 8


def test_ple_non_spec_decode_uses_checkpoint_blocks_without_workspace():
    backend = _make_backend(1, False, "cpu")
    assert backend._ple_verify_scratch == {}
    backend.refresh_decode_metadata(
        4,
        2,
        torch.arange(4),
        torch.tensor([5, 6, 1, 1], dtype=torch.int32),
        forward_mode=ForwardMode.DECODE,
        block_tables={
            QWEN4_EXP_PLE_CACHE_GROUP: torch.tensor([[1, 2], [4, 5]], dtype=torch.int32)
        },
    )
    metadata = backend.forward_metadata
    assert metadata.input_blocks.tolist() == [1, 5, -1, -1]
    assert metadata.output_blocks.tolist() == [2, 5, -1, -1]
    assert metadata.query_lengths == [1] * 4
    assert metadata.verify_width is None
    assert backend._verify_commit_ctx is None


def test_draft_view_cannot_claim_target_ple_fields():
    backend = _make_backend(3, True, "cpu")
    assert backend._conv_field_ids == ()
    assert backend._ple_verify_scratch == {}
    assert backend.preallocate_verify_workspace(4, 3) == 0


@pytest.mark.skipif(
    not torch.cuda.is_available(), reason="PLE batched copy requires CUDA"
)
@pytest.mark.parametrize("accepted_dtype", [torch.int32, torch.int64])
@pytest.mark.parametrize(
    "seq_lens, accepted_lengths, block_rows, copies",
    [
        ([6, 7], [2, 3], [[1, 2, 3], [4, 5, 6]], [(2, 2), (5, 7)]),
        ([6, 7], [0, 9], [[1, 2, 3], [4, 5, 6]], [(1, 1), (5, 7)]),
        ([3, 30], [-3, 9], [[1, 2, 3], [4, 5, 6]], [(1, 1), (6, 7)]),
        ([6, 7], [0, 9], [[-1, 2, 3], [4, 0, 6]], []),
    ],
    ids=["checkpoint-crossing", "acceptance-clamp", "slot-clamp", "invalid-pages"],
)
def test_ple_commit_copies_only_accepted_checkpoints(
    accepted_dtype, seq_lens, accepted_lengths, block_rows, copies
):
    backend = _make_backend(3, False, "cuda")
    other_view = backend.cache_pool.arena.field(qwen4_exp_ple_conv_field(4))
    other_view.fill_(-77)
    for index, (field_id, scratch) in enumerate(backend._ple_verify_scratch.items()):
        scratch.copy_(
            torch.arange(scratch.numel(), device="cuda").reshape(scratch.shape)
            + 100 * index
        )
        backend.cache_pool.arena.field(field_id).fill_(-77)
    backend.refresh_decode_metadata(
        4,
        2,
        torch.arange(4, device="cuda"),
        torch.tensor([*seq_lens, 1, 1], dtype=torch.int32, device="cuda"),
        forward_mode=ForwardMode.DECODE,
        block_tables={
            QWEN4_EXP_PLE_CACHE_GROUP: torch.tensor(
                block_rows, dtype=torch.int32, device="cuda"
            )
        },
    )
    backend.commit_verified_state(
        torch.tensor(accepted_lengths, dtype=accepted_dtype, device="cuda")
    )
    for field_id, scratch in backend._ple_verify_scratch.items():
        field = backend.cache_pool.arena.field(field_id)
        expected = torch.full_like(field, -77)
        for dst, src in copies:
            expected[dst] = scratch[src]
        torch.testing.assert_close(field, expected, rtol=0, atol=0)
    assert torch.all(other_view == -77)
    assert backend._verify_commit_ctx is None


@pytest.mark.skipif(
    not torch.cuda.is_available(), reason="PLE batched copy requires CUDA"
)
def test_ple_commit_uses_fused_kernels_and_replays_live_inputs():
    backend = _make_backend(3, False, "cuda")
    for index, scratch in enumerate(backend._ple_verify_scratch.values()):
        scratch.copy_(
            torch.arange(scratch.numel(), device="cuda").reshape(scratch.shape)
            + 100 * index
        )
    _verify(backend)
    context = backend._verify_commit_ctx
    block_rows, _, _ = context
    accepted = torch.tensor([0, 9], dtype=torch.int32, device="cuda")
    # Compile before profiling or capture, as a warmed serving loop does.
    backend.commit_verified_state(accepted)
    backend._verify_commit_ctx = context
    backend._ple_commit_rows.fill_(-99)
    with torch.profiler.profile(
        activities=[
            torch.profiler.ProfilerActivity.CPU,
            torch.profiler.ProfilerActivity.CUDA,
        ]
    ) as profile:
        backend.commit_verified_state(accepted)
        torch.cuda.synchronize()
    gpu_kernels = [
        event.name
        for event in profile.events()
        if event.device_type == torch.autograd.DeviceType.CUDA
    ]
    assert len(gpu_kernels) == 4
    for kernel, count in (
        ("_commit_state_pages_kernel", 1),
        ("_state_verify_commit_rows_kernel", 1),
        ("_copy_state_rows_kernel", 2),
    ):
        assert sum(kernel in name for name in gpu_kernels) == count
    assert torch.all(backend._ple_commit_rows[:, 4:] == -99)

    backend._verify_commit_ctx = context
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        backend.commit_verified_state(accepted)
    for lengths, pages, sources in (
        ([2, 1], [2, 5], [2, 5]),
        ([3, 2], [7, 8], [3, 6]),
    ):
        accepted.copy_(torch.tensor(lengths, dtype=torch.int32, device="cuda"))
        block_rows[:, 1].copy_(torch.tensor(pages, dtype=torch.int32, device="cuda"))
        for field_id in backend._ple_verify_scratch:
            backend.cache_pool.arena.field(field_id).fill_(-77)
        graph.replay()
        for field_id, scratch in backend._ple_verify_scratch.items():
            field = backend.cache_pool.arena.field(field_id)
            expected = torch.full_like(field, -77)
            for dst, src in zip(pages, sources, strict=True):
                expected[dst] = scratch[src]
            torch.testing.assert_close(field, expected, rtol=0, atol=0)
        assert torch.all(backend._ple_commit_rows[:, 4:] == -99)
    assert backend._verify_commit_ctx is None
