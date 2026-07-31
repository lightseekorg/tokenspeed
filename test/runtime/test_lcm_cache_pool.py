from __future__ import annotations

import os
import sys
import unittest

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from ci_system.ci_register import register_cuda_ci

from tokenspeed.runtime.configs.lcm_memory_plan import (
    LcmFieldSpec,
    plan_lcm_fields,
)
from tokenspeed.runtime.layers.attention.kv_cache.lcm import LcmCachePool

register_cuda_ci(est_time=10, suite="runtime-1gpu")


def _plan():
    return plan_lcm_fields(
        (
            LcmFieldSpec("history", "history.k", "plane.a", (4,), 1),
            LcmFieldSpec("state", "state.ssm", "plane.b", (8,), 1),
        ),
        logical_block_tokens=4,
        num_lcm_blocks=2,
        cache_blocks_per_lcm_block={"history": 2, "state": 1},
        max_padding_fraction=1.0,
    )


@unittest.skipUnless(torch.cuda.is_available(), "requires CUDA")
class LcmCachePoolTest(unittest.TestCase):
    def test_fields_share_one_backing_and_reuse_the_same_view(self):
        pool = LcmCachePool(_plan(), "cuda")

        history = pool.field("history.k", torch.uint8)

        self.assertIs(history, pool.field("history.k", torch.uint8))
        self.assertEqual(
            history.untyped_storage().data_ptr(),
            pool.backing.untyped_storage().data_ptr(),
        )

    def test_field_rejects_wrong_dtype_width(self):
        pool = LcmCachePool(_plan(), "cuda")

        with self.assertRaisesRegex(ValueError, "dtype itemsize"):
            pool.field("history.k", torch.float32)

    def test_zero_pages_clears_only_the_selected_group_pages(self):
        pool = LcmCachePool(_plan(), "cuda")
        history = pool.field("history.k", torch.uint8)
        state = pool.field("state.ssm", torch.uint8)
        history.fill_(7)
        state.fill_(9)

        pool.zero_pages({"history": [1]})
        torch.cuda.synchronize()

        self.assertTrue(bool((history[0] == 7).all()))
        self.assertTrue(bool((history[1] == 0).all()))
        self.assertTrue(bool((history[2:] == 7).all()))
        self.assertTrue(bool((state == 9).all()))

    def test_zero_pages_rejects_out_of_range_page(self):
        pool = LcmCachePool(_plan(), "cuda")

        with self.assertRaises(IndexError):
            pool.zero_pages({"state": [pool.plan.group("state").page_count]})


if __name__ == "__main__":
    unittest.main()
