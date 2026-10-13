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
over the group's field table byte for byte.
"""

from __future__ import annotations

import os
import sys
import types
import unittest

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from cache_pool_test_utils import make_arena, one_group
from ci_system.ci_register import register_cuda_ci

from tokenspeed.runtime.layers.attention.kv_cache.recipes.plan import (
    CacheFieldSpec,
    pack,
)
from tokenspeed.runtime.layers.attention.kv_cache.virtual_blocks import (
    local_page_pairs_by_group,
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


class PairTranslationTest(unittest.TestCase):
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
        with self.assertRaisesRegex(ValueError, "different owners"):
            local_page_pairs_by_group(
                {"kv": np.asarray([[3, 4]])}, contract=_contract({"kv": 2}, 32), rank=0
            )


class CopyCachePagesContractTest(unittest.TestCase):
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
        self.assertEqual(
            trace,
            [("wait", "execution-stream"), ("copy", {"kv": [[3, 8], [5, 9]]})],
        )

    def test_a_rank_owning_no_pair_places_no_fence_and_copies_nothing(self):
        from tokenspeed.runtime.execution.model_executor import ModelExecutor

        trace = []
        fake = self._fake(trace, shard_counts={"kv": 2}, rank=1)
        ModelExecutor.copy_cache_pages(fake, {"kv": np.asarray([[3, 9]])})
        self.assertEqual(trace, [])


@unittest.skipUnless(torch.cuda.is_available(), "requires CUDA")
class CacheArenaCopyBlocksCudaTest(unittest.TestCase):
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
                    self.assertEqual(size, dst_size)
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
        with self.assertRaises(IndexError):
            arena.copy_blocks(
                {
                    "history": np.asarray([[1, 2]]),
                    "state": np.asarray([[1, state_pages]]),
                }
            )
        with self.assertRaisesRegex(ValueError, "chain"):
            arena.copy_blocks({"history": np.asarray([[1, 2], [2, 3]])})
        with self.assertRaisesRegex(ValueError, "chain"):
            arena.copy_blocks({"history": np.asarray([[2, 2]])})
        with self.assertRaisesRegex(ValueError, "twice"):
            arena.copy_blocks({"history": np.asarray([[1, 3], [2, 3]])})
        with self.assertRaises(TypeError):
            arena.copy_blocks({"history": np.asarray([1, 2])})
        torch.cuda.synchronize()
        self.assertTrue(bool((arena.buffer == 7).all()))
        # Nothing to copy is a no-op, not a launch.
        arena.copy_blocks({"history": np.zeros((0, 2), dtype=np.int32)})
        arena.copy_blocks({})


if __name__ == "__main__":
    unittest.main()
