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

"""Shared prefix identity: shifted rows, committed lookahead, host keys and PD."""

import hashlib
import struct

import pytest
import tokenspeed_scheduler as ts
from conftest import _advance, _finish, _spec

ROLE = ts.SchedulerConfig.Role
G = 64


def _config(lookahead: int, role) -> ts.SchedulerConfig:
    cfg = ts.SchedulerConfig()
    cfg.prefix_hash_lookahead_tokens = lookahead
    cfg.prefix_granularity = G
    cfg.num_device_pages = 128
    cfg.num_host_pages = 0
    cfg.max_scheduled_tokens = G
    cfg.max_batch_size = 8
    cfg.decode_input_tokens = 4
    cfg.disable_l2_cache = True
    cfg.enable_kv_cache_events = True
    cfg.role = role
    cfg.cache_groups = [
        ts.CacheGroupConfig(
            group_id="full",
            block_granularity=G,
            total_pages=128,
            transfer_policy=ts.CacheTransferPolicy.FullSuffix,
        ),
        ts.CacheGroupConfig(
            group_id="draft",
            block_granularity=G // 2,
            total_pages=128,
            retention=ts.CacheRetention.SlidingWindow,
            sliding_window_tokens=1024,
            transfer_policy=ts.CacheTransferPolicy.FullSuffix,
        ),
    ]
    return cfg


def _event(scheduler, event):
    scheduler.advance(ts.ExecutionEvent().add_event(event))


def _abort(scheduler, rid):
    event = ts.ForwardEvent.Abort()
    event.request_id = rid
    _event(scheduler, event)


def _prefill(scheduler, tokens, rid):
    scheduler.submit_requests([_spec(rid, tokens)])
    for _ in range(len(tokens) // G + 1):
        op = scheduler.next_execution_plan().forward[0]
        end = op.extend_prefix_lens[0] + op.input_lengths[0]
        if end == len(tokens):
            return op
        _advance(scheduler, rid, [])
    raise AssertionError("prefill did not reach prompt end")


@pytest.mark.parametrize("lookahead", [-2, -1, 2])
def test_config_requires_explicit_supported_lookahead(lookahead):
    assert ts.SchedulerConfig().prefix_hash_lookahead_tokens == -1
    with pytest.raises(ValueError, match="prefix_hash_lookahead_tokens"):
        ts.Scheduler(_config(lookahead, ROLE.Fused))


def test_zero_lookahead_keeps_old_hash_bytes_and_one_has_separate_identity():
    tokens = list(range(3 * G + 1))
    ordinary = ts.Scheduler(_config(0, ROLE.Fused))
    shifted = ts.Scheduler(_config(1, ROLE.Fused))
    expected = []
    prior = b""
    for start in range(0, 3 * G, G):
        page = tokens[start : start + G]
        prior = hashlib.sha256(
            struct.pack("<I", len(prior))
            + prior
            + struct.pack("<I", G)
            + struct.pack(f"<{G}i", *page)
        ).digest()
        expected.append(prior.hex())
    assert ordinary.prefix_hashes_for_tokens(tokens) == expected
    shifted_hashes = shifted.prefix_hashes_for_tokens(tokens)
    assert all(a != b for a, b in zip(expected, shifted_hashes, strict=True))
    assert shifted.prefix_hashes_for_tokens(tokens[:G]) == []
    assert len(shifted.prefix_hashes_for_tokens(tokens[: G + 1])) == 1
    wider = _config(0, ROLE.Fused)
    wider.prefix_granularity = G + 1
    wider.cache_groups = [
        ts.CacheGroupConfig(group_id="full", block_granularity=1, total_pages=128)
    ]
    assert shifted_hashes[0] != ts.Scheduler(wider).prefix_hashes_for_tokens(tokens)[0]


@pytest.mark.parametrize("lookahead", [0, 1])
@pytest.mark.parametrize("divergence", [G, 2 * G, 2 * G + 1])
def test_chunked_partial_hit_keeps_only_pages_with_matching_continuation(
    lookahead, divergence
):
    scheduler = ts.Scheduler(_config(lookahead, ROLE.Fused))
    tokens = list(range(2 * G + 2))
    original = _prefill(scheduler, tokens, "parent")
    rows = dict(original.block_tables)
    _advance(scheduler, "parent", [700])
    _finish(scheduler, "parent")
    scheduler.next_execution_plan()

    branch = tokens.copy()
    branch[divergence] = 800
    scheduler.submit_requests([_spec("child", branch)])
    assert scheduler.waiting_prefix_hashes() == scheduler.prefix_hashes_for_tokens(
        branch
    )
    op = scheduler.next_execution_plan().forward[0]
    hit = ((divergence - lookahead) // G) * G
    assert op.extend_prefix_lens == [hit]
    for group, grain in (("full", G), ("draft", G // 2)):
        row = dict(op.block_tables)[group][0]
        assert row[: hit // grain] == rows[group][0][: hit // grain]
        assert row[hit // grain] != rows[group][0][hit // grain]


@pytest.mark.parametrize("finish", [False, True])
def test_generated_boundary_uses_landed_sample_not_unaccepted_candidates(finish):
    scheduler = ts.Scheduler(_config(1, ROLE.Fused))
    _prefill(scheduler, list(range(G - 2)), "parent")
    _advance(scheduler, "parent", [G - 2])
    scheduler.next_execution_plan()
    result = ts.ForwardEvent.ExtendResult()
    result.request_id = "parent"
    result.tokens = [G - 1, G]  # accepted row 63 plus the sampled continuation
    result.spec_candidate_ids = [900, 901, 902]
    _event(scheduler, result)
    if finish:
        _finish(scheduler, "parent")
    else:
        scheduler.next_execution_plan()  # publishes row 63 on decode admission
        _abort(scheduler, "parent")
    scheduler.next_execution_plan()
    for continuation, hit in ((G, G), (900, 0)):
        scheduler.submit_requests([_spec("child", list(range(G)) + [continuation])])
        op = scheduler.next_execution_plan().forward[0]
        assert op.extend_prefix_lens == [hit]
        _abort(scheduler, "child")
        scheduler.next_execution_plan()


@pytest.mark.parametrize("role", [ROLE.P, ROLE.D])
def test_pd_publishes_boundary_after_bootstrap_with_same_hashes_as_fused(role):
    scheduler = ts.Scheduler(_config(1, role))
    fused = ts.Scheduler(_config(1, ROLE.Fused))
    tokens = list(range(2 * G))
    continuation = tokens + [2 * G]
    assert scheduler.prefix_hashes_for_tokens(
        continuation
    ) == fused.prefix_hashes_for_tokens(continuation)
    scheduler.submit_requests([_spec("parent", tokens)])
    _event(scheduler, ts.PD.BootstrappedEvent("parent"))
    first = scheduler.next_execution_plan()
    if role == ROLE.P:
        assert first.forward[0].input_lengths == [G]
        _advance(scheduler, "parent", [])
        scheduler.next_execution_plan()
        assert scheduler.next_execution_plan().remote_decode is None
        _advance(scheduler, "parent", [2 * G])
        handoff = scheduler.next_execution_plan()
        assert handoff.remote_decode.decode_input_ids == [2 * G]
        _event(scheduler, ts.PD.SucceededEvent("parent"))
    else:
        assert first.remote_prefill.extend_prefix_lens == [0]
        assert scheduler.drain_kv_events() == []
        _event(scheduler, ts.PD.RemotePrefillDoneEvent("parent", 2 * G))
        scheduler.next_execution_plan()
        _finish(scheduler, "parent")
    stored = [e for e in scheduler.drain_kv_events() if e.kind == "BlockStored"]
    assert [e.token_ids for e in stored] == [tokens[:G], tokens[G:]]
    assert all(e.block_size == G for e in stored)
    scheduler.next_execution_plan()
    for next_token, hit in ((2 * G, 2 * G), (999, G)):
        scheduler.submit_requests([_spec("child", tokens + [next_token])])
        _event(scheduler, ts.PD.BootstrappedEvent("child"))
        plan = scheduler.next_execution_plan()
        op = plan.forward[0] if role == ROLE.P else plan.remote_prefill
        assert op.extend_prefix_lens == [hit]
        _abort(scheduler, "child")
        scheduler.next_execution_plan()


@pytest.mark.parametrize("stored_lookahead", [0, 1])
def test_l3_keys_and_host_shortage_retry_use_configured_identity(stored_lookahead):
    cfg = _config(1, ROLE.Fused)
    cfg.cache_groups = cfg.cache_groups[:1]
    cfg.num_host_pages = 3  # two pages, less than the three-page L3 hit
    cfg.disable_l2_cache = False
    cfg.enable_l3_storage = True
    cfg.enable_kv_cache_events = False
    scheduler = ts.Scheduler(cfg)
    source = ts.Scheduler(_config(stored_lookahead, ROLE.Fused))
    tokens = list(range(3 * G + 1))
    hashes = source.prefix_hashes_for_tokens(tokens)
    scheduler.register_storage_keys(*scheduler.expand_prefix_keys(hashes))
    scheduler.submit_requests([_spec("child", tokens)])
    assert scheduler.waiting_prefix_hashes() == scheduler.prefix_hashes_for_tokens(
        tokens
    )
    plan = scheduler.next_execution_plan()
    expected_hit = 2 * G if stored_lookahead == 1 else 0
    assert plan.forward[0].extend_prefix_lens == [expected_hit]
    loads = [op for op in plan.cache if isinstance(op, ts.Cache.LoadBackOp)]
    if stored_lookahead == 0:
        assert loads == []
    else:
        assert len(loads) == 1
        assert {h for row in loads[0].content_hashes for h in row} == set(hashes[:2])
        for op_id in loads[0].op_ids:
            _event(scheduler, ts.Cache.LoadBackDoneEvent(op_id, True))


def test_host_publication_keys_include_sampled_continuation():
    cfg = _config(1, ROLE.Fused)
    cfg.num_host_pages = 128
    cfg.disable_l2_cache = False
    cfg.max_scheduled_tokens = 256
    scheduler = ts.Scheduler(cfg)
    tokens = list(range(2 * G))
    _prefill(scheduler, tokens, "parent")
    _advance(scheduler, "parent", [2 * G])
    _finish(scheduler, "parent")
    plan = scheduler.next_execution_plan()
    stores = [op for op in plan.cache if isinstance(op, ts.Cache.WriteBackOp)]
    expected = scheduler.prefix_hashes_for_tokens(tokens + [2 * G])
    assert stores
    assert {h for op in stores for row in op.content_hashes for h in row} == set(
        expected
    )
    for op in stores:
        for op_id in op.op_ids:
            done = ts.Cache.WriteBackDoneEvent()
            done.op_id = op_id
            _event(scheduler, done)
    assert scheduler.clear_l1_cache()
    for next_token, hit in ((2 * G, 2 * G), (999, G)):
        scheduler.submit_requests([_spec("child", tokens + [next_token])])
        plan = scheduler.next_execution_plan()
        assert plan.forward[0].extend_prefix_lens == [hit]
        loads = [op for op in plan.cache if isinstance(op, ts.Cache.LoadBackOp)]
        assert loads
        for op in loads:
            for op_id in op.op_ids:
                _event(scheduler, ts.Cache.LoadBackDoneEvent(op_id, True))
        _abort(scheduler, "child")
        scheduler.next_execution_plan()
        assert scheduler.clear_l1_cache()


def test_readmission_rehashes_prompt_and_committed_output_consistently():
    scheduler = ts.Scheduler(_config(1, ROLE.Fused))
    tokens = list(range(2 * G))
    _prefill(scheduler, tokens, "parent")
    _advance(scheduler, "parent", [2 * G])
    scheduler.next_execution_plan()  # publishes the final prompt page
    event = ts.ForwardEvent.Retract()
    event.request_id = "parent"
    _event(scheduler, event)
    assert scheduler.waiting_prefix_hashes() == scheduler.prefix_hashes_for_tokens(
        tokens + [2 * G]
    )
    op = scheduler.next_execution_plan().forward[0]
    assert op.extend_prefix_lens == [2 * G]
    assert op.input_ids == [2 * G]
