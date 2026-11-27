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

"""A block drafter's prefill-round graphs.

A DFlash drafter over an MLA draft records its prefill round's block draft per
request count up to 16. These tests pin the plan the CUDA-graph memory
projection prices, the capture order the probe pairs its readings with, which
rounds replay, and that a replay drafts from the round's staged inputs. The
drafter carries only the state these paths read; the draft model's forward is
a stand-in, since the decode graph already records the real one.
"""

from __future__ import annotations

import os
import sys
from test.runtime.drafter_prefill_graph_test_utils import RecordingObserver
from types import SimpleNamespace

import pytest
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from ci_system.ci_register import register_cuda_ci  # noqa: E402

from tokenspeed.runtime.execution.breakable_cuda_graph import (  # noqa: E402
    current_forward_ctx,
)
from tokenspeed.runtime.execution.cudagraph_memory import (  # noqa: E402
    ladder_from_plan,
)
from tokenspeed.runtime.execution.drafter import dflash  # noqa: E402
from tokenspeed.runtime.execution.drafter.base import (  # noqa: E402
    EXTEND_DECODE_SERIES,
    DrafterExtendCapture,
)
from tokenspeed.runtime.execution.drafter.dspark import DSpark  # noqa: E402
from tokenspeed.runtime.execution.forward_batch_info import ForwardMode  # noqa: E402

register_cuda_ci(est_time=10, suite="runtime-1gpu")

WIDTH = 4
MASK = 7


class _RecordingBackend:
    """The draft attention calls the block draft makes around its forward."""

    def __init__(self) -> None:
        self.refreshes: list[dict] = []

    def refresh_decode_metadata(self, bs, actual_bs, req_pool_indices, seq_lens, **kw):
        self.refreshes.append(
            dict(bs=bs, actual_bs=actual_bs, seq_lens=seq_lens.clone(), **kw)
        )

    def fill_block_decode_seq_lens(self, bs, seq_lens) -> None:
        pass

    def publish_draft_step_locations(self, *, cache_start, num_tokens) -> None:
        pass


def _drafter(
    *,
    max_bs: int = 4,
    attention_kind: str = "kimi_mla",
    dp_size: int = 1,
    pp_size: int = 1,
    device: str = "cpu",
) -> DSpark:
    """A DSpark carrying only the state the block graphs read."""
    drafter = object.__new__(DSpark)
    drafter.draft_sampler = None
    drafter.attention_kind = attention_kind
    drafter.draft_model_runner = SimpleNamespace(
        mapping=SimpleNamespace(attn=SimpleNamespace(dp_size=dp_size), pp_size=pp_size)
    )
    drafter.input_buffers = SimpleNamespace(
        max_bs=max_bs,
        req_pool_indices_buf=torch.arange(max_bs, device=device),
    )
    drafter.spec_num_tokens = WIDTH
    drafter.draft_query_width = WIDTH
    drafter.draft_seq_lens_buf = torch.zeros(max_bs, dtype=torch.int32, device=device)
    drafter.draft_block_end_lens_buf = torch.zeros_like(drafter.draft_seq_lens_buf)
    drafter.block_offsets = torch.arange(WIDTH, dtype=torch.int64, device=device)
    drafter.block_ids_buf = torch.full(
        (max_bs, WIDTH), MASK, dtype=torch.int32, device=device
    )
    drafter.block_positions_buf = torch.zeros(
        (max_bs, WIDTH), dtype=torch.int64, device=device
    )
    drafter.next_tokens_buf = torch.zeros(
        (max_bs, WIDTH), dtype=torch.int32, device=device
    )
    drafter.attn_backend = _RecordingBackend()
    drafter.token_to_kv_pool = None
    drafter.round_block_tables = {}
    drafter._block_graphs = {}
    drafter._block_graph_pool = None
    drafter._block_graph_handoffs = {}
    drafter._block_graph_engaged = False
    return drafter


def _capture(drafter, plan, *, entries, num_warmup=1, stream=None, prepare=None):
    return DrafterExtendCapture(
        plan=plan,
        ladders={
            series: ladder_from_plan(graphs, entries) for series, graphs in plan.items()
        },
        stream=stream,
        num_warmup=num_warmup,
        target_hidden_width=0,
        target_hidden_dtype=torch.bfloat16,
        prepare_batch=prepare or (lambda tokens, bs: None),
    )


# ----------------------------------------------------------------------
# The plan and the capture.
# ----------------------------------------------------------------------


def test_the_plan_takes_the_block_draft_at_every_request_count() -> None:
    assert _drafter(max_bs=4).extend_graph_plan([16, 32]) == {
        EXTEND_DECODE_SERIES: [(4, 4), (3, 3), (2, 2), (1, 1)]
    }
    plan = _drafter(max_bs=40).extend_graph_plan([16])
    assert plan[EXTEND_DECODE_SERIES] == [(bs, bs) for bs in range(16, 0, -1)]


@pytest.mark.parametrize(
    "setup",
    [
        dict(attention_kind="qwen_mha"),
        dict(dp_size=2),
        dict(pp_size=2),
    ],
)
def test_configurations_the_block_graphs_cannot_bake_record_nothing(setup) -> None:
    assert _drafter(**setup).extend_graph_plan([16]) == {}


def test_the_capture_records_the_sampled_counts_in_ladder_order(monkeypatch) -> None:
    drafter = _drafter()
    prepared, captured = [], []

    def capture_block(self, bs, capture, window):
        with window:
            captured.append(bs)
        return dflash._BlockGraph(capture=None, ctx=None)

    monkeypatch.setattr(DSpark, "_capture_block", capture_block)
    observer = RecordingObserver()
    plan = drafter.extend_graph_plan([16])
    drafter.capture_extend_graphs(
        _capture(
            drafter,
            plan,
            entries=3,
            prepare=lambda tokens, bs: prepared.append((tokens, bs)),
        ),
        observer=observer,
    )

    # A probe of three entries over four counts takes the widest three, each
    # after its own dummy pure extend.
    assert prepared == [(4, 4), (3, 3), (2, 2)]
    assert captured == [4, 3, 2]
    assert observer.order == [EXTEND_DECODE_SERIES] * 3
    assert sorted(drafter._block_graphs) == [2, 3, 4]


def test_a_capture_without_warmup_is_refused() -> None:
    drafter = _drafter()
    plan = drafter.extend_graph_plan([16])
    with pytest.raises(ValueError, match="warmup"):
        drafter.capture_extend_graphs(
            _capture(drafter, plan, entries=None, num_warmup=0),
            observer=RecordingObserver(),
        )


# ----------------------------------------------------------------------
# Which rounds replay.
# ----------------------------------------------------------------------


def _round(bs, num_extends, *, global_num_tokens=None, shard_size=None):
    return SimpleNamespace(
        bs=bs,
        num_extends=num_extends,
        input_num_tokens=30,
        global_num_tokens=global_num_tokens,
        query_shard=None if shard_size is None else SimpleNamespace(size=shard_size),
        target_capture_sink=None,
    )


@pytest.mark.parametrize(
    "ctx, replays",
    [
        (_round(2, 2), True),
        # A count the plan did not record, or one past it.
        (_round(3, 3), False),
        # A mixed round carries decode requests.
        (_round(2, 1), False),
        (_round(2, 2, global_num_tokens=[30]), False),
        (_round(2, 2, shard_size=2), False),
    ],
)
def test_only_a_pure_extend_at_a_recorded_count_replays(ctx, replays) -> None:
    drafter = _drafter()
    drafter._block_graphs[2] = dflash._BlockGraph(capture=None, ctx=None)
    assert drafter._replays_block(ctx) is replays


class _ReplayRecorder:
    def __init__(self, events) -> None:
        self.events = events

    def replay(self) -> None:
        self.events.append(("replay", current_forward_ctx()))


@pytest.mark.parametrize("num_extends", [2, 1])
def test_a_replayed_round_stages_its_block_then_replays(
    monkeypatch, num_extends
) -> None:
    """A pure extend stages and replays; a mixed round drafts eagerly."""
    drafter = _drafter()
    drafter.target_model = object()
    drafter._context_written = False
    marker = object()
    events = []
    drafter._block_graphs[2] = dflash._BlockGraph(
        capture=_ReplayRecorder(events), ctx=marker
    )
    monkeypatch.setattr(
        DSpark, "_update_native_cache_from_target", lambda self, *args: None
    )
    monkeypatch.setattr(
        DSpark, "_stage_block", lambda self, bs: events.append(("stage", bs))
    )
    monkeypatch.setattr(
        DSpark, "draft", lambda self, current: events.append(("eager", current))
    )
    output_tokens = torch.tensor([11, 12, 13, 14, 15, 16], dtype=torch.int32)

    result = drafter.run(
        _round(2, num_extends),
        None,
        output_tokens,
        torch.zeros(2, dtype=torch.int32),
    )

    if num_extends == 2:
        assert events == [("stage", 2), ("replay", marker)]
        assert result.data_ptr() == drafter.next_tokens_buf.data_ptr()
        assert drafter.block_ids_buf[:2, 0].tolist() == [11, 12]
    else:
        assert [event for event, _ in events] == ["eager"]


# ----------------------------------------------------------------------
# A replay on a GPU.
# ----------------------------------------------------------------------


def _toy_draft(self, current_tokens, prepared=False):
    """A block draft reading what a round stages: positions and the anchors."""
    bs = current_tokens.shape[0]
    if not prepared:
        torch.add(
            self.draft_seq_lens_buf[:bs].unsqueeze(1),
            self.block_offsets,
            out=self.block_positions_buf[:bs],
        )
    out = self.next_tokens_buf[:bs]
    out.copy_(self.block_positions_buf[:bs] * 10 + self.block_ids_buf[:bs, :1])
    return out


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires a CUDA device")
def test_a_replay_drafts_from_the_rounds_staged_inputs(monkeypatch) -> None:
    drafter = _drafter(max_bs=2, device="cuda")
    monkeypatch.setattr(DSpark, "_draft_native", _toy_draft)

    def prepare(tokens, bs):
        # The dummy extend's lengths, which the round below replaces.
        drafter.draft_seq_lens_buf[:bs].fill_(3)

    plan = {EXTEND_DECODE_SERIES: [(2, 2)]}
    drafter.capture_extend_graphs(
        _capture(
            drafter, plan, entries=None, stream=torch.cuda.Stream(), prepare=prepare
        ),
        observer=RecordingObserver(),
    )
    drafter.attn_backend.refreshes.clear()

    lens = torch.tensor([100, 7], dtype=torch.int32, device="cuda")
    anchors = torch.tensor([5, 9], dtype=torch.int32, device="cuda")
    drafter.draft_seq_lens_buf.copy_(lens)
    drafter.block_ids_buf[:, 0] = anchors
    drafter._stage_block(2)
    graph = drafter._block_graphs[2]
    graph.capture.replay()
    torch.cuda.synchronize()

    positions = lens.long().unsqueeze(1) + torch.arange(WIDTH, device="cuda")
    expected = (positions * 10 + anchors.long().unsqueeze(1)).int()
    assert torch.equal(drafter.next_tokens_buf, expected)
    (refresh,) = drafter.attn_backend.refreshes
    assert torch.equal(refresh["seq_lens"], lens + WIDTH)
    assert refresh["num_extends"] == 0 and refresh["for_graph_replay"]
    assert refresh["forward_mode"] == ForwardMode.DECODE


# ----------------------------------------------------------------------
# Fused MLA prefill KV write + DFlash2 prefill graph integration.
# ----------------------------------------------------------------------


def _mla_dflash2_model(dtype: torch.dtype = torch.bfloat16):
    from transformers import PretrainedConfig

    from tokenspeed.runtime.distributed.mapping import Mapping
    from tokenspeed.runtime.models.dflash2 import DFlash2DraftModel

    config = PretrainedConfig(
        vocab_size=64,
        hidden_size=128,
        intermediate_size=64,
        num_hidden_layers=2,
        num_attention_heads=2,
        num_key_value_heads=2,
        head_dim=64,
        q_lora_rank=32,
        kv_lora_rank=64,
        qk_nope_head_dim=32,
        qk_rope_head_dim=16,
        v_head_dim=32,
        rms_norm_eps=1e-6,
        max_position_embeddings=256,
        rope_theta=10000.0,
        layer_types=["sliding_attention", "full_attention"],
        sliding_window=128,
        decode_monokernel=False,
        dflash_config={
            "attention_mode": "mla",
            "block_size": 16,
            "conv_group_size": 16,
            "conv_kernel_size": 2,
            "selector_rank": 8,
            "selector_top_k": 4,
            "target_layer_ids": [0],
        },
    )
    torch.manual_seed(0)
    model = DFlash2DraftModel(config, Mapping(rank=0, world_size=1)).to(dtype=dtype)
    for param in model.parameters():
        torch.nn.init.normal_(param, std=0.05)
    # Run RMSNorm via pure PyTorch on CPU where Triton norm kernels are absent.
    if not torch.cuda.is_available():
        hidden_norm = model.hidden_norm
        hidden_norm.forward = lambda x: torch.nn.functional.rms_norm(
            x,
            (hidden_norm.weight.shape[0],),
            hidden_norm.weight,
            hidden_norm.variance_epsilon,
        )
    return model


def _cpu_mla_pool(n_layers: int, slots: int, width: int, dtype: torch.dtype):
    from tokenspeed.runtime.layers.attention.kv_cache.mla import MLATokenToKVPool

    pool = object.__new__(MLATokenToKVPool)
    pool.quant_method = "none"
    pool.dtype = dtype
    pool.store_dtype = dtype
    pool.latent_write_sanitizes = False
    pool.layerwise_load_tracker = None
    pool.kv_buffer = [
        torch.zeros(slots, 1, width, dtype=dtype) for _ in range(n_layers)
    ]
    return pool


def test_fused_mla_helper_declines_sharded_placement_and_chunks_prefill(
    monkeypatch,
) -> None:
    """Prefill context writes use chunked fused MLA when cache_placement is None."""
    from tokenspeed.runtime.distributed.mapping import Mapping
    from tokenspeed.runtime.execution.drafter.dflash2 import DFlash2

    model = _mla_dflash2_model(dtype=torch.bfloat16)
    kv_width = 64 + 16
    pool = _cpu_mla_pool(
        len(model.layers), slots=16, width=kv_width, dtype=torch.bfloat16
    )

    drafter = object.__new__(DFlash2)
    drafter.model = model
    drafter.attention_kind = model.attention_kind
    drafter.device = torch.device("cpu")
    drafter.spec_num_tokens = 1
    drafter.input_buffers = SimpleNamespace(max_bs=2)
    drafter.draft_model_runner = SimpleNamespace(
        model=model, mapping=Mapping(rank=0, world_size=1)
    )
    drafter.token_to_kv_pool = pool
    drafter._fused_kv_enabled = False
    drafter._fused_kv_is_mla = False
    drafter._fused_kv_proj_workspace = None
    drafter._fused_kv_workspace_capacity = 0
    drafter._fused_kv_workspace_dtype = None

    # Sharded placement must decline fused scatter cleanly.
    drafter.attn_backend = SimpleNamespace(cache_placement=lambda _mqa: object())
    drafter._init_fused_kv_helper_mla(model.layers)
    assert not drafter._fused_kv_enabled

    # Unsharded DRAFT_SLIDING_ATTENTION (placement None) enables it cleanly.
    drafter.attn_backend = SimpleNamespace(cache_placement=lambda _mqa: None)
    drafter._init_fused_kv_helper_mla(model.layers)
    assert drafter._fused_kv_enabled and drafter._fused_kv_is_mla

    ws_before = drafter._fused_kv_proj_workspace
    ws_ptr_before = ws_before.data_ptr()
    assert drafter._fused_kv_workspace_capacity == 2

    buf_by_ptr = {buf.data_ptr(): buf for buf in pool.kv_buffer}
    calls: list[tuple[int, int, tuple[int, ...], tuple[int, ...]]] = []

    def fake_scatter(
        latent,
        norm_weight,
        eps,
        cos_sin_cache,
        positions,
        loc,
        planes,
        *,
        write_mask=None,
        is_neox=True,
        sanitize=False,
    ) -> None:
        del norm_weight, eps, cos_sin_cache, write_mask, is_neox, sanitize
        assert positions.stride(0) == 1
        assert loc.stride(0) == 1
        calls.append(
            (
                int(latent.shape[0]),
                int(latent.shape[1]),
                tuple(int(x) for x in positions.tolist()),
                tuple(int(x) for x in loc.tolist()),
            )
        )
        for layer_idx, ptr in enumerate(planes.ptrs.tolist()):
            buf_by_ptr[ptr][loc.long(), 0, :] = latent[:, layer_idx, :]

    monkeypatch.setattr(dflash, "mla_latent_norm_rope_scatter", fake_scatter)
    monkeypatch.setattr(drafter, "_PREFILL_FUSED_KV_CHUNK_ROWS", 3)

    # Empty write is a safe no-op.
    drafter._write_native_cache(
        torch.empty(0, model.context_in_features, dtype=torch.bfloat16),
        torch.empty(0, dtype=torch.int32),
        torch.empty(0, dtype=torch.int64),
        decode_only=False,
    )
    assert calls == []

    # Prefill write of 7 rows chunks cleanly into 3 + 3 + 1 without growing workspace.
    hidden = torch.randn(7, model.context_in_features, dtype=torch.bfloat16)
    positions = torch.arange(10, 17, dtype=torch.int32)
    cache_locs = torch.tensor([2, 5, 1, 9, 4, 11, 7], dtype=torch.int64)
    drafter._write_native_cache(hidden, positions, cache_locs, decode_only=False)

    assert [c[0] for c in calls] == [3, 3, 1]
    assert drafter._fused_kv_proj_workspace is ws_before
    assert drafter._fused_kv_proj_workspace.data_ptr() == ws_ptr_before

    with torch.inference_mode():
        ctx_hidden = model.project_target_hidden(hidden)
    for layer_idx, layer in enumerate(model.layers):
        attn = layer.self_attn
        expected_latents = torch.matmul(
            ctx_hidden,
            attn.fused_qkv_a_proj_with_mqa.weight[
                attn.q_lora_rank : attn.q_lora_rank + kv_width
            ].t(),
        )
        torch.testing.assert_close(
            pool.kv_buffer[layer_idx][cache_locs, 0, :], expected_latents
        )

    # Over-capacity calls under stream capture still fail loudly instead of allocating.
    monkeypatch.setattr(dflash, "is_current_stream_capturing", lambda _dev: True)
    with pytest.raises(RuntimeError, match="workspace too small"):
        drafter._write_native_cache(hidden, positions, cache_locs, decode_only=False)


def test_dflash2_draft_native_uses_zero_extends_for_mla() -> None:
    """DFlash2 with MLA passes num_extends=0 to attention metadata refresh."""
    from tokenspeed.runtime.execution.drafter.dflash2 import DFlash2

    model = _mla_dflash2_model()
    drafter = _drafter(attention_kind=model.attention_kind)
    hidden_size = int(model.config.hidden_size)
    dflash2_drafter = object.__new__(DFlash2)
    dflash2_drafter.__dict__.update(drafter.__dict__)
    dflash2_drafter.model = model
    dflash2_drafter.attention_kind = model.attention_kind
    dflash2_drafter.hidden_size = hidden_size
    dflash2_drafter._first_layer_input = "full"
    dflash2_drafter.mask_embedding_partial = None
    dflash2_drafter.embed_tokens = lambda flat_ids, reduce_results=True: torch.zeros(
        (flat_ids.shape[0], hidden_size), dtype=torch.bfloat16
    )
    dflash2_drafter.draft_model_runner = SimpleNamespace(
        mapping=drafter.draft_model_runner.mapping,
        forward=lambda **kw: SimpleNamespace(
            hidden_states=torch.zeros(
                (kw["positions"].shape[0], hidden_size), dtype=torch.bfloat16
            )
        ),
    )
    dflash2_drafter._sample_block = (
        lambda draft_hidden, block_ids, next_tokens: next_tokens.zero_()
    )

    anchors = torch.tensor([11, 12], dtype=torch.int32)
    out = dflash2_drafter._draft_native(anchors)

    assert out.shape == (2, WIDTH)
    assert len(dflash2_drafter.attn_backend.refreshes) == 1
    refresh = dflash2_drafter.attn_backend.refreshes[0]
    assert refresh["num_extends"] == 0
    assert refresh["forward_mode"] == ForwardMode.DECODE


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, *sys.argv[1:]]))
