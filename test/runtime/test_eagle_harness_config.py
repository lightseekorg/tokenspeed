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

"""EAGLE-Harness server arguments: the trees and chains it serves and the
settings it refuses."""

import os
import sys
import tempfile

import pytest
import torch
import torch.distributed as dist
from transformers import PretrainedConfig

# CI Registration (parsed via AST, runtime no-op)
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from ci_system.ci_register import register_cuda_ci

register_cuda_ci(est_time=2, suite="runtime-1gpu")

from tokenspeed.runtime.execution.drafter.eagle_harness.config import (
    _routes_experts,
    chain_table_width,
    chain_width_table,
    decode_graph_floors,
    first_rank_value,
)
from tokenspeed.runtime.utils.server_args import ServerArgs


class TestEagleHarnessConfig:
    def _resolve_tree(self, algorithm, topk, steps, nodes, **overrides):
        return ServerArgs(
            model="test/model",
            speculative_algorithm=algorithm,
            speculative_eagle_topk=topk,
            speculative_num_steps=steps,
            speculative_num_draft_tokens=nodes,
            **overrides,
        )

    def test_eagle_harness_trees(self, subtests):
        harness = {
            "speculative_exec_mode": 0,
            "speculative_data_path": "/cost/data",
            "max_num_seqs": 4,
        }
        chain = {
            **harness,
            "speculative_compute_intensity": "chain_intensity",
            "max_num_seqs": 64,
        }
        sampled = {
            "enable_speculative_sampling": True,
            "sampling_backend": "flashinfer",
        }
        # Lanes draft past the verify budget: (12 - 1) * 4 = 44 lane slots for 16 nodes.
        sa = self._resolve_tree("EAGLE4", 4, 12, 16, **harness)
        assert sa.eagle_harness and sa.speculative_draft_tree
        # EAGLE-4's cost measurement shape at TP > 1: lane masks of two words.
        sa = self._resolve_tree("EAGLE4", 8, 13, 64, **harness)
        assert sa.eagle_harness
        # A harness chain takes the draft-tree path with one child per node.
        sa = self._resolve_tree("EAGLE4", 1, 15, 16, **chain)
        assert sa.eagle_harness and sa.speculative_draft_tree
        # Exec mode 1 benchmarks chains like exec mode 0: the batch cut is a tree's.
        sa = self._resolve_tree(
            "EAGLE4", 1, 15, 16, **{**chain, "speculative_exec_mode": 1}
        )
        assert sa.eagle_harness
        sa = self._resolve_tree("EAGLE4", 4, 12, 16, **harness, **sampled)
        assert sa.eagle_harness and sa.enable_speculative_sampling
        # A sampled chain draws each node's one child, as a tree its children.
        sa = self._resolve_tree("EAGLE4", 1, 15, 16, **chain, **sampled)
        assert sa.eagle_harness and sa.enable_speculative_sampling
        tree = ("EAGLE4", 4, 12, 16)
        # Only exec mode 2 (up to 64 steps) meets the lane limit below the node limit.
        measure = {**harness, "speculative_exec_mode": 2}
        cases = [
            (("EAGLE4", 4, 34, 64), measure, "128 lane slots"),
            (("EAGLE4", 8, 17, 64), harness, "1024 drafted nodes"),
            (("EAGLE4", 2, 65, 64), harness, "2..64 steps"),
            (("EAGLE4", 2, 33, 64), harness, "for at most 32 steps"),
            (("EAGLE4", 3, 6, 16), harness, "1, 2, 4 or 8"),
            (tree, {**harness, "max_num_seqs": 80}, "serve at most 32"),
            (("EAGLE4", 1, 16, 16), chain, "at most 15 steps"),
            (
                ("EAGLE4", 1, 15, 16),
                {**chain, **sampled, "sampling_backend": "triton"},
                "flashinfer",
            ),
            (tree, {**harness, **sampled, "sampling_backend": "triton"}, "flashinfer"),
            (("EAGLE3", 4, 5, 16), sampled, "only the topk=1 chain"),
        ]
        for (algorithm, topk, steps, nodes), overrides, message in cases:
            with (
                subtests.test(algorithm=algorithm, topk=topk, overrides=overrides),
                pytest.raises(ValueError, match=message),
            ):
                self._resolve_tree(algorithm, topk, steps, nodes, **overrides)

    def test_eagle_harness_chain_width_table(self, subtests):
        chain = {
            "speculative_exec_mode": 0,
            "speculative_data_path": "/cost/data",
            "speculative_compute_intensity": "chain_intensity",
            "max_num_seqs": 64,
        }

        def table(steps=15, nodes=16, **overrides):
            sa = self._resolve_tree("EAGLE4", 1, steps, nodes, **{**chain, **overrides})
            return [(e.min_bs, e.width) for e in chain_width_table(sa)]

        # Off unless given: the confidence rule decides every round.
        assert table() == []
        assert table(speculative_chain_width_table="64:4,384:3") == [(64, 4), (384, 3)]
        # Sorted by batch size, widths capped at the chain's; a round takes the largest entry it reaches.
        width_table = {"speculative_chain_width_table": " 16:3 , 4:9 "}
        sa = self._resolve_tree("EAGLE4", 1, 7, 8, **{**chain, **width_table})
        entries = chain_width_table(sa)
        assert [(e.min_bs, e.width) for e in entries] == [(4, 8), (16, 3)]
        widths = {bs: chain_table_width(entries, bs) for bs in (3, 4, 15, 16, 64)}
        assert widths == {3: None, 4: 8, 15: 8, 16: 3, 64: 3}
        assert table(speculative_chain_width_table="off") == []
        # Trees and the cost measurement run no table.
        tree = {**chain, "speculative_compute_intensity": "high_intensity"}
        sa = self._resolve_tree("EAGLE4", 4, 12, 16, **{**tree, "max_num_seqs": 4})
        assert chain_width_table(sa) == ()
        assert table(speculative_exec_mode=2) == []
        for text in ("64", "64:4,", "a:4", "64:4:1", "0:4", "64:1", "8:4,8:3"):
            with (
                subtests.test(text=text),
                pytest.raises(ValueError, match="chain-width-table"),
            ):
                table(speculative_chain_width_table=text)

        # Chains serve 512 requests, the confidence rule at most 256 of them.
        wide = table(max_num_seqs=512, speculative_chain_width_table="257:4")
        assert wide == [(257, 4)]
        cases = [
            ({"max_num_seqs": 513}, "serve at most 512"),
            ({"max_num_seqs": 257}, "most 256"),
            ({"speculative_chain_width_table": "258:4", "max_num_seqs": 512}, "257"),
            ({"speculative_exec_mode": 2, "max_num_seqs": 257}, "in exec mode 2"),
        ]
        for overrides, message in cases:
            with (
                subtests.test(overrides=overrides),
                pytest.raises(ValueError, match=message),
            ):
                table(**overrides)

        # Cost data measured up to 8 requests: the confidence rule must not decide past it.
        with tempfile.TemporaryDirectory() as data_path:
            for bs in (1, 2, 4, 8):
                os.makedirs(os.path.join(data_path, "cost", f"bs_{bs}"))
            measured = {"speculative_data_path": data_path}
            with pytest.raises(ValueError, match="measured up to 8 requests"):
                table(**measured)
            assert table(**measured, speculative_chain_width_table="9:4") == [(9, 4)]
            assert table(**measured, max_num_seqs=8) == []

    def test_decode_graph_floors(self):
        """Each decode graph's floor: the live batches replaying it are those
        above it, up to the graph's padded batch."""
        chain = {
            "speculative_exec_mode": 0,
            "speculative_data_path": "/cost/data",
            "speculative_compute_intensity": "chain_intensity",
            "max_num_seqs": 64,
        }

        def floors(**overrides):
            sa = self._resolve_tree("EAGLE4", 1, 15, 16, **{**chain, **overrides})
            return dict(decode_graph_floors(sa))

        # Batches pad up to the next entry: a graph serves those above the entry before.
        assert floors(cudagraph_capture_sizes=[1, 4, 8, 16]) == {
            1: 0,
            4: 1,
            8: 4,
            16: 8,
        }
        # Without padding a graph serves its own batch size only.
        no_padding = floors(
            cudagraph_capture_sizes=[1, 4, 8], disable_cuda_graph_padding=True
        )
        assert no_padding == {1: 0, 4: 3, 8: 7}
        # A ladder above --max-num-seqs ends at it.
        ladder = floors(cudagraph_capture_sizes=[1, 32, 128])
        assert ladder == {1: 0, 32: 1, 63: 32, 64: 63}

    def test_moe_targets(self):
        # Dense variants of some configs set the expert counts to None.
        assert not _routes_experts(PretrainedConfig(num_experts=None))
        assert _routes_experts(PretrainedConfig(num_experts=None, n_routed_experts=512))
        assert _routes_experts(PretrainedConfig(architectures=["Qwen3MoeForCausalLM"]))


def _first_rank_worker(rank: int, init_file: str) -> None:
    dist.init_process_group(
        "gloo", init_method=f"file://{init_file}", rank=rank, world_size=2
    )
    try:
        group = dist.new_group([0, 1], backend="gloo")
        config = first_rank_value(lambda: {"batch_size": 24 + rank}, group)
        assert config == {"batch_size": 24}

        def missing() -> dict:
            raise FileNotFoundError(f"rank {rank} has no benchmark config")

        with pytest.raises(FileNotFoundError, match="rank 0 has no"):
            first_rank_value(missing, group)
    finally:
        dist.destroy_process_group()


def test_every_rank_takes_the_first_ranks_read(tmp_path):
    """Whatever a rank's own files hold, it takes the first rank's read: its
    value, or the exception it raised."""
    torch.multiprocessing.spawn(
        _first_rank_worker, args=(str(tmp_path / "init"),), nprocs=2
    )


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, *sys.argv[1:]]))
