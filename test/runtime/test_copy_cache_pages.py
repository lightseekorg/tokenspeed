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

"""The plan's Device-to-Device block copies (``ExecutionPlan.pages_to_copy``).

``ModelExecutor.copy_cache_pages`` translates the scheduler's pairs through the
same owner filter zeroing uses, orders itself behind the execution stream and
hands the arena whole-block copies; ``CacheArena.copy_blocks`` expands them
over the group's field table byte for byte. Under Host write-back a copy's
source or destination, or a restore's destination, can be a page the same
plan evicts and writes back; the store still reads it first.
"""

from __future__ import annotations

import os
import sys
import types
from unittest.mock import patch

import numpy as np
import pytest
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from test.runtime.cache_pool_test_utils import make_arena, one_group

from ci_system.ci_register import register_cuda_ci

from tokenspeed.runtime.layers.attention.kv_cache.recipes.plan import (
    CacheFieldSpec,
    pack,
)
from tokenspeed.runtime.layers.attention.kv_cache.virtual_blocks import (
    local_page_pairs_by_group,
    local_pages_by_group,
)

register_cuda_ci(est_time=15, suite="runtime-1gpu")


def _arena(device: str):
    plan = pack(
        (
            one_group(
                "history",
                CacheFieldSpec("history.k", "plane.a", (4,), "uint8"),
                CacheFieldSpec("history.v", "plane.c", (4,), "uint8"),
                rows_per_page=2,
            ),
            one_group(
                "state",
                CacheFieldSpec("state.ssm", "plane.b", (16,), "uint8"),
                rows_per_page=4,
            ),
        ),
        prefix_granularity=4,
        cache_blocks_per_lcm_block={"history": 2, "state": 1},
        max_padding_fraction=1.0,
    ).bind(3)
    return make_arena(plan, device)


class _DefaultStream:
    def __init__(self, trace):
        self._trace = trace

    def wait_stream(self, stream):
        self._trace.append(("wait", stream))


def _contract(shard_counts: dict[str, int], virtual_block_count: int):
    return types.SimpleNamespace(
        group_specs=tuple(
            types.SimpleNamespace(group_id=group_id, shard_count=shard_count)
            for group_id, shard_count in shard_counts.items()
        ),
        virtual_block_counts=dict.fromkeys(shard_counts, virtual_block_count),
    )


class TestPairTranslationTest:
    def test_replicated_group_keeps_every_pair_minus_the_null_block_offset(self):
        pairs = local_page_pairs_by_group(
            {"kv": np.asarray([[3, 8], [5, 9]])},
            contract=_contract({"kv": 1}, 16),
            rank=0,
        )
        np.testing.assert_array_equal(pairs["kv"], [[3, 8], [5, 9]])

    def test_sharded_group_keeps_the_pairs_this_rank_owns(self):
        # Virtual ids deal cyclically to two owners: 3 and 9 are rank 0's
        # (local 2 and 5), 4 and 10 are rank 1's (local 2 and 5).
        virtual = {"kv": np.asarray([[3, 9], [4, 10]])}
        contract = _contract({"kv": 2}, 32)
        np.testing.assert_array_equal(
            local_page_pairs_by_group(virtual, contract=contract, rank=0)["kv"],
            [[2, 5]],
        )
        np.testing.assert_array_equal(
            local_page_pairs_by_group(virtual, contract=contract, rank=1)["kv"],
            [[2, 5]],
        )

    def test_a_pair_across_owners_is_rejected(self):
        with pytest.raises((ValueError), match=("different owners")):
            local_page_pairs_by_group(
                {"kv": np.asarray([[3, 4]])}, contract=_contract({"kv": 2}, 32), rank=0
            )


class TestCopyCachePagesContractTest:
    def _fake(self, trace, *, shard_counts, rank):
        arena = types.SimpleNamespace(
            copy_blocks=lambda pairs: trace.append(
                ("copy", {gid: arr.tolist() for gid, arr in pairs.items()})
            )
        )
        return types.SimpleNamespace(
            token_to_kv_pool=types.SimpleNamespace(arena=arena),
            default_stream=_DefaultStream(trace),
            execution_stream="execution-stream",
            _cache_runtime_contract=_contract(shard_counts, 64),
            _cache_kvp_rank=rank,
        )

    def test_copies_wait_on_the_execution_stream_then_reach_the_arena(self):
        from tokenspeed.runtime.execution.model_executor import ModelExecutor

        trace = []
        fake = self._fake(trace, shard_counts={"kv": 1, "state": 1}, rank=0)
        ModelExecutor.copy_cache_pages(
            fake, {"kv": np.asarray([[3, 8], [5, 9]], dtype=np.int32), "state": []}
        )
        assert (trace) == (
            [("wait", "execution-stream"), ("copy", {"kv": [[3, 8], [5, 9]]})]
        )

    def test_a_rank_owning_no_pair_places_no_fence_and_copies_nothing(self):
        from tokenspeed.runtime.execution.model_executor import ModelExecutor

        trace = []
        fake = self._fake(trace, shard_counts={"kv": 2}, rank=1)
        ModelExecutor.copy_cache_pages(fake, {"kv": np.asarray([[3, 9]])})
        assert (trace) == ([])


class TestCopyWavesTest:
    """A block is read by every pair naming it as source before the pair
    that overwrites it runs; independent pairs share the first wave."""

    def _waves(self, pairs):
        from tokenspeed.runtime.layers.attention.kv_cache.arena import _copy_waves

        return [
            wave["cache"].tolist()
            for wave in _copy_waves(
                {"cache": np.asarray(pairs)}, {"cache": ((0, 1, 1),)}
            )
        ]

    def test_independent_pairs_form_one_wave(self):
        assert (self._waves([[1, 3], [2, 4], [1, 5]])) == ([[[1, 3], [2, 4], [1, 5]]])

    def test_a_chain_reads_before_it_overwrites(self):
        # (2, 1) overwrites block 1, which (1, 3) still reads; (0, 2)
        # overwrites block 2, which (2, 1) reads.
        assert (self._waves([[2, 1], [1, 3], [0, 2]])) == (
            [[[1, 3]], [[2, 1]], [[0, 2]]]
        )

    def test_a_cycle_is_rejected(self):
        with pytest.raises((ValueError), match=("cycle")):
            self._waves([[1, 2], [2, 1]])

    def test_a_cycle_between_differently_packed_groups_is_rejected(self):
        from tokenspeed.runtime.layers.attention.kv_cache.arena import _copy_waves

        # History reads [8, 16) and writes [16, 24); recent reads [16, 20)
        # and writes [8, 12). Neither copy may overwrite the other's source.
        with pytest.raises(ValueError, match="cycle"):
            _copy_waves(
                {
                    "history": np.asarray([[1, 2]]),
                    "recent": np.asarray([[3, 1]]),
                },
                {"history": ((0, 8, 8),), "recent": ((4, 4, 4),)},
            )

    def test_no_pairs_no_waves(self):
        assert (self._waves(np.zeros((0, 2), dtype=np.int32))) == ([])


@pytest.mark.skipif(not (torch.cuda.is_available()), reason=("requires CUDA"))
class TestFinishTailHitCopyCudaTest:
    """The hitter's private copy of a published partial page, end to end.

    The real scheduler publishes a finished request's partial page as a tail,
    a later prompt hits inside it, and the plan's copy pair -- translated by
    the owner filter and executed by the arena -- reproduces every field of
    every layer of the group (k and v of target and draft layers, the fp8
    scales) on the hitter's block, byte for byte, before any forward reads it.
    """

    PAGE = 16
    LAYERS = 3  # two target layers and a draft continuation layer

    def _arena(self):
        fields = []
        for layer in range(self.LAYERS):
            fields += [
                CacheFieldSpec(f"kv.k.{layer}", f"k.{layer}", (4, 8), "bfloat16"),
                CacheFieldSpec(f"kv.v.{layer}", f"v.{layer}", (4, 8), "bfloat16"),
                CacheFieldSpec(
                    f"kv.k_scale.{layer}", f"ks.{layer}", (4,), "float8_e4m3fn"
                ),
                CacheFieldSpec(f"kv.v_scale.{layer}", f"vs.{layer}", (1,), "float32"),
            ]
        plan = pack(
            (one_group("kv", *fields, rows_per_page=self.PAGE),),
            prefix_granularity=self.PAGE,
            cache_blocks_per_lcm_block={"kv": 1},
            max_padding_fraction=1.0,
        ).bind(16)
        return make_arena(plan, "cuda")

    def test_the_hitters_block_holds_the_published_partial_page_byte_for_byte(self):
        ts = pytest.importorskip("tokenspeed_scheduler")
        from test.runtime.scheduler_driver import (
            advance_scheduler,
            make_extend_result_event,
        )

        from tokenspeed.runtime.engine.scheduler_utils import make_config

        arena = self._arena()
        contract = arena.runtime_contract
        page_count = arena.plan.group("kv").page_count
        scheduler = ts.Scheduler(
            make_config(
                num_device_pages=page_count,
                max_scheduled_tokens=256,
                max_batch_size=4,
                prefix_granularity=self.PAGE,
                num_host_pages=0,
                disable_l2_cache=True,
                enable_l3_storage=False,
                role="fused",
                num_snapshot_pages=1,
                host_group_packing=(),
                max_retracted_requests=0,
                host_write_policy="write_through",
                storage_config=None,
                cache_groups=[
                    ts.CacheGroupConfig(
                        group_id="kv",
                        block_granularity=self.PAGE,
                        total_pages=page_count,
                        attn_kind=ts.AttnKind.Full,
                    )
                ],
            )
        )

        def forwards(plan):
            return [op for op in plan.forward if op.request_ids]

        prompt = list(range(41))  # two full pages and a 9-row tail
        spec = ts.RequestSpec()
        spec.request_id = "a"
        spec.tokens = prompt
        spec.max_new_tokens = 1
        scheduler.submit_requests([spec])
        plan = scheduler.next_execution_plan()
        (prefill,) = forwards(plan)
        tail_block = int(prefill.block_tables_arrays()["kv"][0][2])
        # The engine lands the one token and finishes the request; the finish
        # publishes the partial page (rows 32..40) in place.
        advance_scheduler(scheduler, [make_extend_result_event("a", [1000])])
        finish = ts.ForwardEvent.Finish()
        finish.request_id = "a"
        advance_scheduler(scheduler, [finish])
        assert (scheduler.tail_entries()) == (1)

        # Random bytes stand in for the KV the forward wrote.
        generator = torch.Generator(device="cpu").manual_seed(41)
        arena.buffer.copy_(
            torch.randint(
                0, 256, (arena.buffer.numel(),), dtype=torch.uint8, generator=generator
            )
        )
        before = arena.buffer.clone()

        spec = ts.RequestSpec()
        spec.request_id = "b"
        spec.tokens = prompt + [7, 8, 9]
        spec.max_new_tokens = 1
        scheduler.submit_requests([spec])
        plan = scheduler.next_execution_plan()
        (hit,) = forwards(plan)
        assert (list(hit.extend_prefix_lens)) == ([41])
        copies = plan.pages_to_copy_arrays()["kv"]
        assert (copies.shape) == ((1, 2))
        source, destination = (int(value) for value in copies[0])
        assert (source) == (tail_block)
        assert (int(hit.block_tables_arrays()["kv"][0][2])) == (destination)
        assert (destination) not in (plan.pages_to_zero["kv"])

        local = local_page_pairs_by_group({"kv": copies}, contract=contract, rank=0)
        arena.copy_blocks(local)
        torch.cuda.synchronize()

        local_source, local_destination = (int(value) for value in local["kv"][0])
        src_segments = arena.block_byte_segments("kv", [local_source])
        dst_segments = arena.block_byte_segments("kv", [local_destination])
        # Every declared field of every layer, in the plan's own byte layout.
        assert (len(src_segments)) == (4 * self.LAYERS)
        expected = before.clone()
        for (src_off, size), (dst_off, dst_size) in zip(
            src_segments, dst_segments, strict=True
        ):
            assert (size) == (dst_size)
            expected[dst_off : dst_off + size] = before[src_off : src_off + size]
        torch.testing.assert_close(arena.buffer, expected, rtol=0, atol=0)
        for src_off, size in src_segments:
            assert torch.equal(
                arena.buffer[src_off : src_off + size],
                before[src_off : src_off + size],
            )


@pytest.mark.skipif(not (torch.cuda.is_available()), reason=("requires CUDA"))
class TestWriteBackOfPagesThePlanReusesCudaTest:
    """Under Host write-back an admission's or a restore's evictions reach the
    Host in the plan that re-grants their pages, end to end through
    ``DeviceHandle.execute``.

    On a full Device two finished prompts, A and E, each hold two full pages
    and a tail. B resends A and hits inside A's tail; C resends E. B's
    admission evicts E's tail and takes its page for the private copy of A's
    tail; C's evicts A's tail, the page B's copy reads, and gets it zeroed.
    One stream-ordered store writes both tails back. The handle submits the
    round -- the store, then the copy, then the zeroing -- while the store's
    copy is held back on the write stream, and the copy and the zeroing still
    wait for it. The Host then gives both tails back as the prompts left them
    through the scheduler's own load-backs: D resends A once B and C are
    done, and its plan evicts B's and C's tails and loads A's into B's page;
    F resends E once D is done, and its plan loads E's into C's former page.

    At overlap depth 1 the loop hands a round over while the forward of the
    round before may still be running on the execution stream. Two variants
    stand that forward in, held back, as the last writer of both tails: the
    store, not held back, still reads what it writes; held back longer than
    that forward, the store still goes ahead of the round's copy and zeroing,
    which that forward alone would have released.
    """

    PAGE = 16

    @pytest.mark.parametrize("held", ["store", "forward", "forward_and_store"])
    def test_pages_a_plan_reuses_reach_the_host_as_the_prompts_left_them(self, held):
        ts = pytest.importorskip("tokenspeed_scheduler")
        from test.runtime.cache_pool_test_utils import (
            make_mha_memory_plan,
            make_pool,
            specs_for_layers,
        )
        from test.runtime.scheduler_driver import (
            UNBOUNDED_CACHED_PREFIX_TOKENS,
            advance_scheduler,
            make_extend_result_event,
            make_spec,
        )

        from tokenspeed.runtime.cache.l2.executor import HostCacheExecutor
        from tokenspeed.runtime.cache.l2.sizing import RetractionPoolRequest
        from tokenspeed.runtime.engine.scheduler_utils import (
            cache_ops_from_plan,
            make_config,
            pool_to_cache_groups,
            scheduler_cache_geometry_from_pool,
        )
        from tokenspeed.runtime.execution.device import DeviceHandle
        from tokenspeed.runtime.execution.model_executor import ModelExecutor
        from tokenspeed.runtime.layers.attention.kv_cache.hybrid_mha import (
            HybridMHATokenToKVPool,
        )

        # Six usable pages: the two prompts fill the Device. The hybrid MHA
        # pool over full-attention layers alone zeroes the pages a plan
        # grants, as every hybrid pool does (the MLA + KDA one too); a plain
        # MHA or MLA pool zeroes nothing.
        group = "full_attention"
        layer_types = ("full_attention",) * 2
        arena, pool = make_pool(
            HybridMHATokenToKVPool,
            make_mha_memory_plan(
                size=6 * self.PAGE,
                prefix_granularity=self.PAGE,
                layer_num=2,
                kv_heads=1,
                head_dim=8,
                dtype=torch.float16,
            ),
            device="cuda",
            cache_group_specs=specs_for_layers(
                layer_types=layer_types,
                group_ids=(group,) * 2,
                prefix_granularity=self.PAGE,
                pd_disaggregation_enabled=True,
            ),
            dtype=torch.float16,
            head_num=1,
            head_dim=8,
            layer_num=2,
            rank=0,
            layer_types=layer_types,
        )
        executor = HostCacheExecutor(
            pool,
            draft_pool=None,
            l2_tier=True,
            host_ratio=1.0,
            host_size_gb=0,
            snapshot_pool=RetractionPoolRequest(
                host_gb=0.0,
                ratio=0.0,
                max_retracted_requests=0,
                tail_lcm_blocks_per_request=0,
            ),
            slot_state_exporters=None,
            io_backend="kernel",
            attn_tp_rank=0,
            kvp_rank=0,
            nprocs_per_node=1,
            world_cpu_group=None,
        )
        # A model executor without a model (none is loaded here): the fields
        # its page copy and page zeroing read, and no drafter for the handle
        # to settle before a round's work.
        model_executor = ModelExecutor.__new__(ModelExecutor)
        model_executor.device = "cuda"
        model_executor.device_module = torch.cuda
        model_executor.default_stream = torch.cuda.default_stream()
        model_executor.execution_stream = torch.cuda.Stream()
        model_executor.token_to_kv_pool = pool
        model_executor.draft_token_to_kv_pool = None
        model_executor._cache_runtime_contract = arena.runtime_contract
        model_executor._cache_kvp_rank = 0
        model_executor.drafter = None
        handle = DeviceHandle(model_executor, host_cache_executor=executor)
        # This thread is the forward thread: the handle's work runs here.
        handle.bind_forward_thread()
        try:
            scheduler = ts.Scheduler(
                make_config(
                    num_device_pages=scheduler_cache_geometry_from_pool(
                        pool
                    ).num_device_pages,
                    max_scheduled_tokens=256,
                    max_batch_size=4,
                    prefix_granularity=self.PAGE,
                    num_host_pages=executor.num_host_pages,
                    disable_l2_cache=False,
                    enable_l3_storage=False,
                    role="prefill",
                    num_snapshot_pages=1,
                    host_group_packing=executor.host_group_packing,
                    max_retracted_requests=0,
                    storage_config=None,
                    host_write_policy="write_back",
                    cache_groups=pool_to_cache_groups(pool),
                )
            )

            def forwards(plan):
                return [op for op in plan.forward if op.request_ids]

            def submit(prompts):
                scheduler.submit_requests(
                    [
                        make_spec(
                            rid,
                            tokens,
                            max_cached_prefix_tokens=UNBOUNDED_CACHED_PREFIX_TOKENS,
                        )
                        for rid, tokens in prompts.items()
                    ]
                )
                advance_scheduler(
                    scheduler, [ts.PD.BootstrappedEvent(rid) for rid in prompts]
                )
                return scheduler.next_execution_plan()

            def finish(rids, token):
                # The prompts' token lands and the decode peer takes their KV
                # (no Device work here); the transfer's success caches them.
                advance_scheduler(
                    scheduler, [make_extend_result_event(rid, [token]) for rid in rids]
                )
                plan = scheduler.next_execution_plan()
                assert list(plan.remote_decode.request_ids) == list(rids)
                assert not cache_ops_from_plan(plan).device
                assert not any(len(pages) for pages in plan.pages_to_zero.values())
                advance_scheduler(
                    scheduler, [ts.PD.SucceededEvent(rid) for rid in rids]
                )

            def execute(plan):
                # The round's Device work, all of it enqueued by the handle
                # when it returns.
                cache_ops = cache_ops_from_plan(plan).device
                assert handle.execute(plan, None, cache_ops=cache_ops) is None
                return cache_ops

            def land():
                torch.cuda.synchronize()
                acks = handle.poll_cache_results()
                advance_scheduler(scheduler, acks)
                return [int(ack.op_id) for ack in acks]

            contract = arena.runtime_contract

            def segments(page):
                (local,) = local_pages_by_group(
                    {group: [page]}, contract=contract, rank=0
                )[group]
                return arena.block_byte_segments(group, [int(local)])

            # A and E: two full pages and a 9-token tail each, cached once
            # the decode peer has them.
            prompts = {"a": list(range(41)), "e": list(range(200, 241))}
            (prefill,) = forwards(submit(prompts))
            tables = prefill.block_tables_arrays()[group]
            tail_a, tail_e = int(tables[0][2]), int(tables[1][2])
            finish(prompts, 1000)
            assert scheduler.tail_entries() == 2

            # Warm both store rows, plus copying and zeroing, so the round
            # below loads no kernel and grows no staging buffer: either can
            # wait for the device, the store's held-back copy included.
            arena.copy_blocks({group: np.asarray([[tail_a, tail_e]])})
            arena.zero_blocks({group: np.asarray([tail_a])})
            executor._start_writing(  # pylint: disable=protected-access
                [999],
                [(0, tail_a, 1), (0, tail_e, 2)],
                lane=executor._ordered_write_lane,  # pylint: disable=protected-access
                prerequisite_stream=torch.cuda.current_stream(),
            )
            torch.cuda.synchronize()
            assert [int(ack.op_id) for ack in handle.poll_cache_results()] == [999]
            # Random bytes stand in for the KV the forwards wrote.
            generator = torch.Generator(device="cpu").manual_seed(29)
            arena.buffer.copy_(
                torch.randint(
                    0,
                    256,
                    (arena.buffer.numel(),),
                    dtype=torch.uint8,
                    generator=generator,
                )
            )
            before = arena.buffer.clone()

            plan = submit(
                {"b": prompts["a"] + [7, 8, 9], "c": prompts["e"] + [4, 5, 6]}
            )
            hit_b, hit_c = forwards(plan)[0].extend_prefix_lens
            assert (hit_b, hit_c) == (41, 32)
            assert plan.remote_decode is None
            (store,) = cache_ops_from_plan(plan).device
            assert list(store.source_pinned) == [False]
            assert [list(rows) for rows in store.src_pages] == [[tail_a, tail_e]]
            host_a, host_e = (int(page) for page in store.dst_pages[0])
            # B copies A's tail into E's; C's new page, A's tail, is zeroed.
            assert plan.pages_to_copy_arrays()[group].tolist() == [[tail_a, tail_e]]
            assert list(plan.pages_to_zero[group]) == [tail_a]

            if held != "store":
                # The round before's forward, held back, writes both tails
                # last: they leave the Device as it leaves them.
                written = torch.randint(
                    0,
                    256,
                    (arena.buffer.numel(),),
                    dtype=torch.uint8,
                    generator=generator,
                ).cuda()
                for page in (tail_a, tail_e):
                    for offset, size in segments(page):
                        before[offset : offset + size] = written[offset : offset + size]
                torch.cuda.synchronize()
                with torch.cuda.stream(model_executor.execution_stream):
                    torch.cuda._sleep(200_000_000)
                    for page in (tail_a, tail_e):
                        for offset, size in segments(page):
                            arena.buffer[offset : offset + size].copy_(
                                written[offset : offset + size]
                            )
            # The round, with the store's copy held back unless only the
            # forward is: the handle has submitted the copy and the zeroing
            # behind the store while it waits.
            if held != "forward":
                with torch.cuda.stream(executor.write_stream):
                    torch.cuda._sleep(
                        400_000_000 if held == "forward_and_store" else 200_000_000
                    )
            execute(plan)
            assert not executor.write_stream.query()
            if held != "store":
                assert not model_executor.execution_stream.query()
            assert land() == list(store.op_ids)
            assert scheduler.host_pool_cached_blocks() == 2
            expected = before.clone()
            for (src_off, size), (dst_off, _) in zip(
                segments(tail_a), segments(tail_e), strict=True
            ):
                expected[dst_off : dst_off + size] = before[src_off : src_off + size]
                expected[src_off : src_off + size] = 0
            torch.testing.assert_close(arena.buffer, expected, rtol=0, atol=0)

            # B's and C's forwards write their pages; once they are done, D
            # resends A. Its plan evicts B's and C's tails and loads A's from
            # the Host into B's page.
            arena.buffer.copy_(
                torch.randint(
                    0,
                    256,
                    (arena.buffer.numel(),),
                    dtype=torch.uint8,
                    generator=generator,
                )
            )
            expected = arena.buffer.clone()
            finish(("b", "c"), 2000)
            plan = submit({"d": prompts["a"] + [11, 12, 13]})
            assert list(forwards(plan)[0].extend_prefix_lens) == [41]
            store, load = execute(plan)
            assert [list(rows) for rows in store.src_pages] == [[tail_a, tail_e]]
            assert [list(rows) for rows in load.src_pages] == [[host_a]]
            assert [list(rows) for rows in load.dst_pages] == [[tail_e]]
            assert sorted(land()) == sorted([*load.op_ids, *store.op_ids])
            for (src_off, size), (dst_off, _) in zip(
                segments(tail_a), segments(tail_e), strict=True
            ):
                expected[dst_off : dst_off + size] = before[src_off : src_off + size]
            torch.testing.assert_close(arena.buffer, expected, rtol=0, atol=0)

            # Once D is done, F resends E: its plan evicts D's tail and loads
            # E's from the Host into C's former page.
            finish(("d",), 3000)
            plan = submit({"f": prompts["e"] + [14, 15, 16]})
            assert list(forwards(plan)[0].extend_prefix_lens) == [41]
            store, load = execute(plan)
            assert [list(rows) for rows in store.src_pages] == [[tail_e]]
            assert [list(rows) for rows in load.src_pages] == [[host_e]]
            assert [list(rows) for rows in load.dst_pages] == [[tail_a]]
            assert sorted(land()) == sorted([*load.op_ids, *store.op_ids])
            for (src_off, size), (dst_off, _) in zip(
                segments(tail_e), segments(tail_a), strict=True
            ):
                expected[dst_off : dst_off + size] = before[src_off : src_off + size]
            torch.testing.assert_close(arena.buffer, expected, rtol=0, atol=0)
        finally:
            handle.shutdown_cache()

    def test_a_restore_overwrites_a_page_after_its_plan_writes_the_page_back(self):
        """A restore that evicts Device cache under Host write-back writes it
        back in its own plan, and its rows can overwrite a page that store
        reads. The handle submits the store first: with the store's copy held
        back, the restore waits for it, the Host gets the bytes the page held
        and the page ends with the restored ones.

        A retracted request's image of page Z lies in the snapshot pool. One
        plan writes page X back to Host block 1 and restores the image into
        X; Host block 1 then comes back into page Y.
        """
        ts = pytest.importorskip("tokenspeed_scheduler")
        from test.runtime.cache_pool_test_utils import (
            make_mha_memory_plan,
            make_pool,
            specs_for_layers,
        )

        from tokenspeed.runtime.cache.l2.executor import HostCacheExecutor
        from tokenspeed.runtime.cache.l2.sizing import RetractionPoolRequest
        from tokenspeed.runtime.cache.transfer.ops import (
            CacheTransfer,
            HostTier,
            RestoreOp,
            SnapshotOp,
        )
        from tokenspeed.runtime.execution.device import DeviceHandle
        from tokenspeed.runtime.execution.model_executor import ModelExecutor
        from tokenspeed.runtime.layers.attention.kv_cache.hybrid_mha import (
            HybridMHATokenToKVPool,
        )

        group = "full_attention"
        layer_types = ("full_attention",) * 2
        arena, pool = make_pool(
            HybridMHATokenToKVPool,
            make_mha_memory_plan(
                size=6 * self.PAGE,
                prefix_granularity=self.PAGE,
                layer_num=2,
                kv_heads=1,
                head_dim=8,
                dtype=torch.float16,
            ),
            device="cuda",
            cache_group_specs=specs_for_layers(
                layer_types=layer_types,
                group_ids=(group,) * 2,
                prefix_granularity=self.PAGE,
            ),
            dtype=torch.float16,
            head_num=1,
            head_dim=8,
            layer_num=2,
            rank=0,
            layer_types=layer_types,
        )
        executor = HostCacheExecutor(
            pool,
            draft_pool=None,
            l2_tier=True,
            host_ratio=1.0,
            host_size_gb=0,
            # A pool as large as the Device; no slot state beside the pages.
            snapshot_pool=RetractionPoolRequest(
                host_gb=0.0,
                ratio=1.0,
                max_retracted_requests=1,
                tail_lcm_blocks_per_request=0,
            ),
            slot_state_exporters=(),
            io_backend="kernel",
            attn_tp_rank=0,
            kvp_rank=0,
            nprocs_per_node=1,
            world_cpu_group=None,
        )
        # A model executor without a model, as above; the plans here copy and
        # zero no page.
        model_executor = ModelExecutor.__new__(ModelExecutor)
        model_executor.device = "cuda"
        model_executor.device_module = torch.cuda
        model_executor.default_stream = torch.cuda.default_stream()
        model_executor.execution_stream = torch.cuda.Stream()
        model_executor.token_to_kv_pool = pool
        model_executor.draft_token_to_kv_pool = None
        model_executor._cache_runtime_contract = arena.runtime_contract
        model_executor._cache_kvp_rank = 0
        model_executor.drafter = None
        handle = DeviceHandle(model_executor, host_cache_executor=executor)
        handle.bind_forward_thread()

        class WriteBackOp:
            # Cache.WriteBackOp has no Python constructor; stream-ordered.
            def __init__(self, op_id, src_page, dst_page):
                self.op_ids = [op_id]
                self.group_ids = [[0]]
                self.src_pages = [[src_page]]
                self.dst_pages = [[dst_page]]
                self.source_pinned = [False]

        x, y, z = 1, 2, 3
        x_segments, y_segments, z_segments = (
            arena.block_byte_segments(group, [int(local)])
            for local in local_pages_by_group(
                {group: [x, y, z]}, contract=arena.runtime_contract, rank=0
            )[group]
        )
        # Nothing to copy or zero: the rounds below are their cache ops.
        plan = ts.ExecutionPlan()
        try:
            # Random bytes stand in for the KV the forwards wrote.
            generator = torch.Generator(device="cpu").manual_seed(31)
            arena.buffer.copy_(
                torch.randint(
                    0,
                    256,
                    (arena.buffer.numel(),),
                    dtype=torch.uint8,
                    generator=generator,
                )
            )
            before = arena.buffer.clone()
            with patch.object(ts.Cache, "WriteBackOp", WriteBackOp):
                # Z's image into pool block 1, and a store and a restore on
                # the lanes the round below takes, so that round loads no
                # kernel and pins no staging memory: either would wait for the
                # device, the held-back store included.
                handle.execute(
                    plan,
                    None,
                    cache_ops=[
                        SnapshotOp(
                            op_id=1,
                            request_id="r",
                            request_pool_index=1,
                            snapshot_slot=1,
                            transfers=(CacheTransfer(0, z, 1, "", 0),),
                        ),
                        WriteBackOp(2, z, 2),
                    ],
                )
                torch.cuda.synchronize()
                handle.execute(
                    plan,
                    None,
                    cache_ops=[
                        RestoreOp(
                            op_id=3,
                            request_id="r",
                            request_pool_index=1,
                            snapshot_slot=1,
                            transfers=(CacheTransfer(0, 1, y, "", 0),),
                            source_tier=(HostTier.SNAPSHOT_POOL,),
                        )
                    ],
                )
                torch.cuda.synchronize()
                assert sorted(
                    (type(ack).__name__, int(ack.op_id))
                    for ack in handle.poll_cache_results()
                ) == [
                    ("RestoreDoneEvent", 3),
                    ("SnapshotDoneEvent", 1),
                    ("WriteBackDoneEvent", 2),
                ]

                # The round, with the store's copy held back: the restore
                # waits behind it.
                with torch.cuda.stream(executor.write_stream):
                    torch.cuda._sleep(200_000_000)
                handle.execute(
                    plan,
                    None,
                    cache_ops=[
                        WriteBackOp(4, x, 1),
                        RestoreOp(
                            op_id=5,
                            request_id="r",
                            request_pool_index=1,
                            snapshot_slot=1,
                            transfers=(CacheTransfer(0, 1, x, "", 0),),
                            source_tier=(HostTier.SNAPSHOT_POOL,),
                        ),
                    ],
                )
                assert not executor.write_stream.query()
                assert not executor.load_stream.query()
                torch.cuda.synchronize()
                assert sorted(
                    (type(ack).__name__, int(ack.op_id))
                    for ack in handle.poll_cache_results()
                ) == [("RestoreDoneEvent", 5), ("WriteBackDoneEvent", 4)]

                # Host block 1 back into Y, as a restore reads a retracted
                # request's Host L2 leg.
                handle.execute(
                    plan,
                    None,
                    cache_ops=[
                        RestoreOp(
                            op_id=6,
                            request_id="r",
                            request_pool_index=1,
                            snapshot_slot=1,
                            transfers=(CacheTransfer(0, 1, y, "", 0),),
                            source_tier=(HostTier.L2,),
                        )
                    ],
                )
                torch.cuda.synchronize()
                assert [int(ack.op_id) for ack in handle.poll_cache_results()] == [6]
            # X holds Z's restored bytes, and the Host got X's from before.
            expected = before.clone()
            for (x_off, size), (y_off, _), (z_off, _) in zip(
                x_segments, y_segments, z_segments, strict=True
            ):
                expected[x_off : x_off + size] = before[z_off : z_off + size]
                expected[y_off : y_off + size] = before[x_off : x_off + size]
            torch.testing.assert_close(arena.buffer, expected, rtol=0, atol=0)
        finally:
            handle.shutdown_cache()


@pytest.mark.skipif(not (torch.cuda.is_available()), reason=("requires CUDA"))
class TestCacheArenaCopyBlocksCudaTest:
    @pytest.mark.parametrize("reverse_groups", [False, True])
    def test_cross_group_reuse_reads_the_window_before_overwriting_its_parent(
        self, reverse_groups
    ):
        # A history block fills a parent; two recent-window blocks share one.
        # The history copy overwrites the two window sources, so both window
        # copies must run first regardless of the groups' declaration order.
        declarations = (
            one_group(
                "history",
                CacheFieldSpec("history.k", "shared", (8,), "uint8"),
                rows_per_page=2,
            ),
            one_group(
                "recent",
                CacheFieldSpec("recent.k", "shared", (4,), "uint8"),
                attn_kind="sliding_window",
                sliding_window_tokens=1,
                rows_per_page=1,
            ),
        )
        arena = make_arena(
            pack(
                declarations,
                prefix_granularity=2,
                cache_blocks_per_lcm_block={"history": 1, "recent": 2},
            ).bind(3),
            "cuda",
            cache_group_specs=tuple(spec for spec, _ in declarations),
        )
        initial = torch.arange(arena.buffer.numel(), dtype=torch.uint8)
        arena.buffer.copy_(initial)
        pairs = {
            "history": np.asarray([[1, 2]], dtype=np.int32),
            "recent": np.asarray([[3, 5], [4, 6]], dtype=np.int32),
        }
        if reverse_groups:
            pairs = dict(reversed(tuple(pairs.items())))

        arena.copy_blocks(pairs)
        torch.cuda.synchronize()

        # Bytes 16:24 were the recent group's two source blocks; the full
        # history copy now owns them. Bytes 24:32 must retain their old data.
        expected = initial.clone()
        expected[16:24] = initial[8:16]
        expected[24:32] = initial[16:24]
        torch.testing.assert_close(arena.buffer.cpu(), expected, rtol=0, atol=0)

    def test_disjoint_child_copies_between_two_parents_need_no_temporary_storage(self):
        # Both groups use the same plane and packing. The parents exchange
        # ownership, but the individual source and destination byte ranges
        # are disjoint; ordering by whole parent would invent a cycle.
        plan = pack(
            tuple(
                one_group(
                    group,
                    CacheFieldSpec(f"{group}.k", "shared", (4,), "uint8"),
                    rows_per_page=1,
                )
                for group in ("history", "recent")
            ),
            prefix_granularity=2,
            cache_blocks_per_lcm_block={"history": 2, "recent": 2},
        ).bind(2)
        arena = make_arena(plan, "cuda")
        initial = torch.arange(arena.buffer.numel(), dtype=torch.uint8)
        arena.buffer.copy_(initial)

        arena.copy_blocks(
            {
                "history": np.asarray([[1, 3]], dtype=np.int32),
                "recent": np.asarray([[4, 2]], dtype=np.int32),
            }
        )
        torch.cuda.synchronize()

        expected = initial.clone()
        expected[16:20] = initial[8:12]
        expected[12:16] = initial[20:24]
        torch.testing.assert_close(arena.buffer.cpu(), expected, rtol=0, atol=0)

    def test_copies_every_field_of_the_source_block_byte_for_byte(self):
        arena = _arena("cuda")
        generator = torch.Generator(device="cpu").manual_seed(0)
        initial = torch.randint(
            0, 256, (arena.buffer.numel(),), dtype=torch.uint8, generator=generator
        )
        arena.buffer.copy_(initial)
        history_pages = arena.plan.group("history").page_count
        pairs = {
            "history": np.asarray([[1, 3], [2, history_pages - 1]], dtype=np.int32),
            "state": np.asarray([[1, 2]], dtype=np.int64),
        }
        expected = initial.clone()
        for group_id, group_pairs in pairs.items():
            for source, destination in group_pairs.tolist():
                src_segments = arena.block_byte_segments(group_id, [source])
                dst_segments = arena.block_byte_segments(group_id, [destination])
                for (src_off, size), (dst_off, dst_size) in zip(
                    src_segments, dst_segments, strict=True
                ):
                    assert (size) == (dst_size)
                    expected[dst_off : dst_off + size] = initial[
                        src_off : src_off + size
                    ]

        arena.copy_blocks(pairs)
        torch.cuda.synchronize()
        torch.testing.assert_close(arena.buffer.cpu(), expected, rtol=0, atol=0)

    def test_a_block_the_plan_reads_and_overwrites_is_read_first(self):
        """A published tail some hitter copies may be evicted and re-granted
        as another request's copy-in destination in the same plan: ``(1, 3)``
        reads block 1, ``(2, 1)`` overwrites it. Block 3 must receive block
        1's content as it was, not block 2's; a longer chain orders the same
        way, and two chains of one wave stay parallel."""
        arena = _arena("cuda")
        generator = torch.Generator(device="cpu").manual_seed(1)
        initial = torch.randint(
            0, 256, (arena.buffer.numel(),), dtype=torch.uint8, generator=generator
        )
        arena.buffer.copy_(initial)
        pairs = {
            "history": np.asarray([[2, 1], [1, 3], [0, 2]], dtype=np.int32),
            "state": np.asarray([[1, 2], [0, 1]], dtype=np.int64),
        }
        expected = initial.clone()
        for group_id, group_pairs in pairs.items():
            for source, destination in group_pairs.tolist():
                src_segments = arena.block_byte_segments(group_id, [source])
                dst_segments = arena.block_byte_segments(group_id, [destination])
                for (src_off, size), (dst_off, _) in zip(
                    src_segments, dst_segments, strict=True
                ):
                    expected[dst_off : dst_off + size] = initial[
                        src_off : src_off + size
                    ]

        arena.copy_blocks(pairs)
        torch.cuda.synchronize()
        torch.testing.assert_close(arena.buffer.cpu(), expected, rtol=0, atol=0)

    def test_validates_every_group_before_touching_a_byte(self):
        arena = _arena("cuda")
        arena.buffer.fill_(7)
        state_pages = arena.plan.group("state").page_count
        with pytest.raises((IndexError)):
            arena.copy_blocks(
                {
                    "history": np.asarray([[1, 2]]),
                    "state": np.asarray([[1, state_pages]]),
                }
            )
        with pytest.raises((ValueError), match=("cycle")):
            arena.copy_blocks({"history": np.asarray([[1, 2], [2, 1]])})
        with pytest.raises((ValueError), match=("onto itself")):
            arena.copy_blocks({"history": np.asarray([[2, 2]])})
        with pytest.raises((ValueError), match=("twice")):
            arena.copy_blocks({"history": np.asarray([[1, 3], [2, 3]])})
        with pytest.raises((TypeError)):
            arena.copy_blocks({"history": np.asarray([1, 2])})
        torch.cuda.synchronize()
        assert bool((arena.buffer == 7).all())
        # Nothing to copy is a no-op, not a launch.
        arena.copy_blocks({"history": np.zeros((0, 2), dtype=np.int32)})
        arena.copy_blocks({})


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, *sys.argv[1:]]))
