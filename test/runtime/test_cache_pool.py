from __future__ import annotations

import os
import sys
import unittest

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from cache_pool_test_utils import MinimalCacheView, make_arena, one_group
from ci_system.ci_register import register_cuda_ci

from tokenspeed.runtime.layers.attention.kv_cache.arena import CacheArena
from tokenspeed.runtime.layers.attention.kv_cache.base import CachePool
from tokenspeed.runtime.layers.attention.kv_cache.recipes.plan import (
    CacheFieldSpec,
    pack,
)

register_cuda_ci(est_time=10, suite="runtime-1gpu")


def _plan():
    return pack(
        (
            one_group(
                "history",
                CacheFieldSpec("history.k", "plane.a", (4,), "uint8"),
                rows_per_page=2,
            ),
            one_group(
                "state",
                CacheFieldSpec("state.ssm", "plane.b", (8,), "uint8"),
                rows_per_page=4,
            ),
        ),
        prefix_granularity=4,
        cache_blocks_per_lcm_block={"history": 2, "state": 1},
        max_padding_fraction=1.0,
    ).bind(2)


def _arena(device: str = "cpu") -> CacheArena:
    return make_arena(_plan(), device)


class CacheArenaContractTest(unittest.TestCase):
    def test_allocates_the_planned_arena_on_construction(self):
        arena = _arena()

        # Allocation is eager and owned here, so no compute view can observe
        # an unbound arena or race to allocate it.
        self.assertEqual(arena.buffer.dtype, torch.uint8)
        self.assertEqual(arena.buffer.numel(), arena.plan.arena_bytes)

    def test_materializes_every_planned_field_on_construction(self):
        arena = _arena()

        # The plan names every field and its dtype, so nothing is bound
        # later and no consumer can observe a half-built arena.
        self.assertEqual(
            arena.field_ids(), {field.field_id for field in arena.plan.fields}
        )

    def test_field_views_share_the_one_buffer_and_are_stable(self):
        arena = _arena()

        history = arena.field("history.k")

        self.assertIs(history, arena.field("history.k"))
        self.assertEqual(history.dtype, torch.uint8)
        self.assertEqual(
            history.untyped_storage().data_ptr(),
            arena.buffer.untyped_storage().data_ptr(),
        )

    def test_field_takes_its_dtype_from_the_plan(self):
        plan = pack(
            (
                one_group(
                    "history",
                    CacheFieldSpec("history.k", "plane.a", (4,), "bfloat16"),
                    rows_per_page=4,
                ),
            ),
            prefix_granularity=4,
            cache_blocks_per_lcm_block={"history": 1},
            max_padding_fraction=1.0,
        ).bind(2)

        self.assertEqual(
            make_arena(plan, "cpu").field("history.k").dtype, torch.bfloat16
        )

    def test_field_rejects_an_unplanned_field(self):
        arena = _arena()

        with self.assertRaisesRegex(ValueError, "not planned"):
            arena.field("history.absent")

    def test_size_matches_the_tightest_packed_group(self):
        arena = _arena()

        # 2 parents * 2 history blocks per parent * P=4.
        self.assertEqual(arena.size, 16)

    def test_block_segments_match_the_canonical_plan_geometry(self):
        arena = _arena()

        expected = [
            (
                arena.plan.field_page_byte_offset(field.field_id, block_id),
                field.payload_bytes,
            )
            for block_id in (0, 2)
            for field in arena.plan.fields
            if field.group_id == "history"
        ]

        self.assertEqual(arena.block_byte_segments("history", [0, 2]), expected)

    def test_block_segments_reject_invalid_ids_before_expansion(self):
        arena = _arena()

        for invalid in (-1, True, arena.plan.group("history").page_count):
            with self.subTest(invalid=invalid), self.assertRaises(IndexError):
                arena.block_byte_segments("history", [invalid])


class CachePoolViewTest(unittest.TestCase):
    def test_a_pool_reports_the_arena_it_views(self):
        arena = _arena()
        pool = MinimalCacheView(arena, torch.uint8, rank=0)

        # A view owns no memory and no geometry: it names the arena, and
        # callers ask the arena. Nothing is mirrored onto the view.
        self.assertIs(pool.arena, arena)
        for owned_by_arena in (
            "plan",
            "buffer",
            "size",
            "prefix_granularity",
            "kv_page_size",
            "runtime_contract",
            "cache_group_specs",
        ):
            self.assertFalse(
                hasattr(pool, owned_by_arena),
                f"{owned_by_arena} must be read off the arena, not the view",
            )

    def test_two_views_of_one_arena_share_field_views(self):
        arena = _arena()
        target = MinimalCacheView(arena, torch.uint8, rank=0)
        draft = MinimalCacheView(arena, torch.uint8, rank=0)

        self.assertIs(target.arena, draft.arena)
        self.assertIs(target.arena.field("history.k"), draft.arena.field("history.k"))

    def test_views_of_one_arena_may_read_it_as_different_dtypes(self):
        arena = _arena()

        # A heterogeneous draft (bf16 head over an fp8 target) is two views
        # over one allocation, so dtype is per-view, never arena-wide.
        self.assertEqual(
            MinimalCacheView(arena, torch.uint8, rank=0).dtype, torch.uint8
        )
        self.assertEqual(
            MinimalCacheView(arena, torch.bfloat16, rank=0).dtype, torch.bfloat16
        )

    def test_a_view_must_implement_every_kernel_accessor(self):
        """The four accessors are abstract, so a pool that forgets one cannot
        be constructed -- the failure lands at wiring time, not at the first
        write into a half-implemented view."""
        self.assertEqual(
            sorted(CachePool.__abstractmethods__),
            ["get_key_buffer", "get_kv_buffer", "get_value_buffer", "set_kv_buffer"],
        )

        class _Forgetful(CachePool):
            def get_key_buffer(self, layer_id: int):
                raise AssertionError("not exercised")

        with self.assertRaisesRegex(TypeError, "abstract"):
            _Forgetful(_arena(), torch.uint8, rank=0)

    def test_rejects_a_negative_field_layer_offset(self):
        arena = _arena()

        with self.assertRaisesRegex(ValueError, "non-negative"):
            MinimalCacheView(arena, torch.uint8, rank=0, field_layer_offset=-1)


@unittest.skipUnless(torch.cuda.is_available(), "requires CUDA")
class CacheArenaCudaTest(unittest.TestCase):
    def test_zero_blocks_clears_only_the_selected_group_blocks(self):
        arena = _arena("cuda")
        history = arena.field("history.k")
        state = arena.field("state.ssm")
        history.fill_(7)
        state.fill_(9)

        arena.zero_blocks({"history": np.asarray([1])})
        torch.cuda.synchronize()

        # history.k is a per-token field (planned shape leads with P=4), so
        # its view is token-indexed: block 1 covers tokens [4, 8).
        self.assertTrue(bool((history[:4] == 7).all()))
        self.assertTrue(bool((history[4:8] == 0).all()))
        self.assertTrue(bool((history[8:] == 7).all()))
        self.assertTrue(bool((state == 9).all()))

    def test_zero_blocks_rejects_out_of_range_block(self):
        arena = _arena("cuda")

        with self.assertRaises(IndexError):
            arena.zero_blocks(
                {"state": np.asarray([arena.plan.group("state").page_count])}
            )
        # Every group is validated before any block is touched.
        arena.buffer.fill_(7)
        with self.assertRaises(IndexError):
            arena.zero_blocks({"history": np.asarray([1]), "state": np.asarray([-1])})
        torch.cuda.synchronize()
        self.assertTrue(bool((arena.buffer == 7).all()))

    def test_zero_blocks_many_pages_across_groups_match_the_segment_geometry(self):
        arena = _arena("cuda")
        history_pages = arena.plan.group("history").page_count
        state_pages = arena.plan.group("state").page_count
        # Unsorted with duplicates and uneven span lengths, so the staging
        # layout has to align each group's span on its own.
        # int32 read-only views are what the scheduler export hands over.
        requests = {
            "history": np.asarray([3, 0, 3, history_pages - 1, 1], dtype=np.int32),
            "state": np.arange(
                state_pages - 1, max(state_pages - 4, -1), -1, dtype=np.int64
            ),
        }
        for ids in requests.values():
            ids.setflags(write=False)
        arena.buffer.fill_(7)
        expected = torch.full_like(arena.buffer, 7, device="cpu")
        for group_id, block_ids in requests.items():
            for offset, size in arena.block_byte_segments(group_id, block_ids.tolist()):
                expected[offset : offset + size] = 0

        arena.zero_blocks(requests)
        torch.cuda.synchronize()

        torch.testing.assert_close(arena.buffer.cpu(), expected, rtol=0, atol=0)

        # Empty requests are a no-op, not a launch; lists are not accepted.
        arena.zero_blocks(
            {
                "history": np.asarray([], dtype=np.int32),
                "state": np.asarray([], dtype=np.int64),
            }
        )
        arena.zero_blocks({})
        with self.assertRaises(TypeError):
            arena.zero_blocks({"history": [1]})
        torch.cuda.synchronize()
        torch.testing.assert_close(arena.buffer.cpu(), expected, rtol=0, atol=0)

    def test_zero_blocks_skips_groups_this_stage_holds_no_fields_of(self):
        # Pipeline parallelism narrows the plan to a stage's layers but keeps
        # every group, so a stage may own a group with no fields at all. The
        # scheduler still hands that group's fresh pages to every rank.
        plan = pack(
            (
                one_group(
                    "history",
                    CacheFieldSpec("layer.0.k", "plane.a", (4,), "uint8"),
                    rows_per_page=2,
                ),
                one_group(
                    "state",
                    CacheFieldSpec("layer.1.ssm", "plane.b", (8,), "uint8"),
                    rows_per_page=4,
                ),
            ),
            prefix_granularity=4,
            cache_blocks_per_lcm_block={"history": 2, "state": 1},
            max_padding_fraction=1.0,
        ).bind(2)
        stage0 = make_arena(plan.narrow_to_layers(0, 1), "cuda")
        self.assertEqual(stage0.field_ids(), {"layer.0.k"})
        history = stage0.field("layer.0.k")
        history.fill_(7)
        state_pages = stage0.plan.group("state").page_count

        stage0.zero_blocks(
            {"state": np.asarray([1, state_pages - 1]), "history": np.asarray([1])}
        )
        torch.cuda.synchronize()
        self.assertTrue(bool((history[4:8] == 0).all()))
        self.assertTrue(bool((history[:4] == 7).all()))
        # A fieldless group's ids are still validated.
        with self.assertRaises(IndexError):
            stage0.zero_blocks({"state": np.asarray([state_pages])})
        # Nothing at all to zero on this stage is a plain no-op.
        stage0.zero_blocks({"state": np.asarray([1])})
        torch.cuda.synchronize()

    def test_clear_zeros_the_whole_arena_once(self):
        arena = _arena("cuda")
        history = arena.field("history.k")
        state = arena.field("state.ssm")
        history.fill_(7)
        state.fill_(9)

        # Both views name the same owner, so a fan-out over target+draft is
        # idempotent rather than double work on half the arena.
        MinimalCacheView(arena, torch.uint8, rank=0).clear_kv_buffers()
        MinimalCacheView(arena, torch.uint8, rank=0).clear_kv_buffers()
        torch.cuda.synchronize()

        self.assertTrue(bool((history == 0).all()))
        self.assertTrue(bool((state == 0).all()))


if __name__ == "__main__":
    unittest.main()
