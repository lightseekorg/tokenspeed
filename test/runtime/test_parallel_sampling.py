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

"""Tests for per-replica seeds in parallel sampling (n > 1)."""

import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from ci_system.ci_register import register_cuda_ci  # noqa: E402

register_cuda_ci(est_time=10, suite="runtime-1gpu")

from tokenspeed.runtime.engine.async_llm import AsyncLLM  # noqa: E402
from tokenspeed.runtime.engine.io_struct import (  # noqa: E402
    GenerateReqInput,
    TokenizedGenerateReqInput,
)
from tokenspeed.runtime.engine.parallel_sampling import (  # noqa: E402
    prepare_parallel_sampling_replica,
)
from tokenspeed.runtime.sampling.sampling_params import SamplingParams  # noqa: E402


def _tokenize(obj: GenerateReqInput) -> TokenizedGenerateReqInput:
    """Materialize a request the way ``InputProcessor`` does for the seed."""
    sampling_params = SamplingParams(**obj.sampling_params)
    sampling_params.resolve_seed(obj.rid)
    return TokenizedGenerateReqInput(
        rid=obj.rid, input_ids=list(obj.input_ids), sampling_params=sampling_params
    )


def _replica_seeds(sampling_params: dict, n: int) -> tuple[int, list[int]]:
    parent = GenerateReqInput(input_ids=[1, 2, 3], sampling_params=sampling_params)
    parent.regenerate_rid()
    tokenized = _tokenize(parent)
    seeds = [
        prepare_parallel_sampling_replica(
            parent, tokenized, replica_index
        ).sampling_params.seed
        for replica_index in range(n)
    ]
    return tokenized.sampling_params.seed, seeds


class TestReplicaSeeds(unittest.TestCase):
    def test_sampled_replicas_get_distinct_seeds(self) -> None:
        base, seeds = _replica_seeds({"temperature": 1.0}, n=8)
        self.assertEqual(seeds, [base + i for i in range(8)])

    def test_caller_seed_is_offset_per_replica(self) -> None:
        base, seeds = _replica_seeds({"temperature": 1.0, "seed": 7}, n=4)
        self.assertEqual(base, 7)
        self.assertEqual(seeds, [7, 8, 9, 10])

    def test_greedy_replicas_keep_pinned_seed(self) -> None:
        _, seeds = _replica_seeds({"temperature": 0.0}, n=4)
        self.assertEqual(seeds, [0, 0, 0, 0])

    def test_replicas_do_not_share_parent_sampling_params(self) -> None:
        parent = GenerateReqInput(input_ids=[1, 2, 3], sampling_params={})
        parent.regenerate_rid()
        tokenized = _tokenize(parent)
        parent_seed = tokenized.sampling_params.seed
        replica = prepare_parallel_sampling_replica(parent, tokenized, 3)
        self.assertIsNot(replica.sampling_params, tokenized.sampling_params)
        self.assertEqual(tokenized.sampling_params.seed, parent_seed)

    def test_unresolved_seed_is_rejected(self) -> None:
        with self.assertRaises(ValueError):
            SamplingParams().for_parallel_sample(1)


class _StubInputProcessor:
    async def tokenize_batch(self, objs):
        return [_tokenize(obj) for obj in objs]


class _StubAsyncLLM(AsyncLLM):
    """Bypass ZMQ / ModelConfig / tokenizer bring-up; capture sent requests."""

    def __init__(self) -> None:
        self.input_processor = _StubInputProcessor()
        self.sent: list[TokenizedGenerateReqInput] = []

    def _send_one_request(self, obj, tokenized_obj, created_time=None) -> None:
        self.sent.append(tokenized_obj)

    async def _wait_one_response(self, obj):
        yield {"meta_info": {"id": obj.rid}}


class TestBatchFanOut(unittest.IsolatedAsyncioTestCase):
    async def test_fan_out_sends_one_seed_per_replica(self) -> None:
        mgr = _StubAsyncLLM()
        obj = GenerateReqInput(
            input_ids=[1, 2, 3], sampling_params={"temperature": 1.0, "n": 4}
        )
        obj.normalize_batch_and_arguments()

        async for _ in mgr._handle_batch_request(obj):
            pass

        warmup, *replicas = mgr.sent
        self.assertEqual(warmup.sampling_params.max_new_tokens, 0)
        self.assertEqual(len(replicas), 4)
        self.assertEqual(len({r.rid for r in replicas}), 4)
        self.assertEqual(len({r.sampling_params.seed for r in replicas}), 4)


if __name__ == "__main__":
    unittest.main()
