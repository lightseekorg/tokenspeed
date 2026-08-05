from __future__ import annotations

import os
import sys
import unittest

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from ci_system.ci_register import register_cuda_ci

from tokenspeed.runtime.layers.attention.kv_cache.base import CachePool
from tokenspeed.runtime.layers.attention.kv_cache.plan import (
    CacheFieldSpec,
    plan_cache_fields,
)

register_cuda_ci(est_time=10, suite="runtime-1gpu")


def _plan():
    return plan_cache_fields(
        (
            CacheFieldSpec("history", "history.k", "plane.a", (4,), 1),
            CacheFieldSpec("state", "state.ssm", "plane.b", (8,), 1),
        ),
        logical_block_tokens=4,
        num_lcm_blocks=2,
        cache_blocks_per_lcm_block={"history": 2, "state": 1},
        max_padding_fraction=1.0,
    )


class CachePoolContractTest(unittest.TestCase):
    def test_requires_a_prepared_memory_plan(self):
        with self.assertRaisesRegex(TypeError, "memory_plan"):
            CachePool(
                size=16,
                dtype=torch.uint8,
                device="cpu",
                page_size=4,
                rank=0,
            )

    def test_owns_buffer_and_creates_typed_field_views(self):
        pool = CachePool(
            size=16,
            dtype=torch.uint8,
            device="cpu",
            page_size=4,
            rank=0,
            memory_plan=_plan(),
        )

        history = pool.field("history.k", torch.uint8)

        self.assertIs(history, pool.field("history.k", torch.uint8))
        self.assertEqual(
            history.untyped_storage().data_ptr(),
            pool.buffer.untyped_storage().data_ptr(),
        )

    def test_expands_scheduler_blocks_for_a_kernel_view(self):
        pool = CachePool(
            size=16,
            dtype=torch.uint8,
            device="cpu",
            page_size=4,
            rank=0,
            memory_plan=_plan(),
        )
        scheduler_table = torch.tensor([[1, 0, -1]], dtype=torch.int32)

        actual = pool.expand_block_table(
            "history",
            scheduler_table,
            kernel_block_tokens=2,
        )

        expected = torch.tensor([[2, 3, 0, 1, 0, 1]], dtype=torch.int32)
        self.assertTrue(torch.equal(actual, expected))


@unittest.skipUnless(torch.cuda.is_available(), "requires CUDA")
class CachePoolCudaTest(unittest.TestCase):
    def test_fields_share_one_buffer_and_reuse_the_same_view(self):
        pool = CachePool(
            size=16,
            dtype=torch.uint8,
            device="cuda",
            page_size=4,
            rank=0,
            memory_plan=_plan(),
        )

        history = pool.field("history.k", torch.uint8)

        self.assertIs(history, pool.field("history.k", torch.uint8))
        self.assertEqual(
            history.untyped_storage().data_ptr(),
            pool.buffer.untyped_storage().data_ptr(),
        )

    def test_field_rejects_wrong_dtype_width(self):
        pool = CachePool(
            size=16,
            dtype=torch.uint8,
            device="cuda",
            page_size=4,
            rank=0,
            memory_plan=_plan(),
        )

        with self.assertRaisesRegex(ValueError, "dtype itemsize"):
            pool.field("history.k", torch.float32)

    def test_zero_blocks_clears_only_the_selected_group_blocks(self):
        pool = CachePool(
            size=16,
            dtype=torch.uint8,
            device="cuda",
            page_size=4,
            rank=0,
            memory_plan=_plan(),
        )
        history = pool.field("history.k", torch.uint8)
        state = pool.field("state.ssm", torch.uint8)
        history.fill_(7)
        state.fill_(9)

        pool.zero_blocks({"history": [1]})
        torch.cuda.synchronize()

        self.assertTrue(bool((history[0] == 7).all()))
        self.assertTrue(bool((history[1] == 0).all()))
        self.assertTrue(bool((history[2:] == 7).all()))
        self.assertTrue(bool((state == 9).all()))

    def test_zero_blocks_rejects_out_of_range_block(self):
        pool = CachePool(
            size=16,
            dtype=torch.uint8,
            device="cuda",
            page_size=4,
            rank=0,
            memory_plan=_plan(),
        )

        with self.assertRaises(IndexError):
            pool.zero_blocks({"state": [pool.plan.group("state").page_count]})


if __name__ == "__main__":
    unittest.main()
