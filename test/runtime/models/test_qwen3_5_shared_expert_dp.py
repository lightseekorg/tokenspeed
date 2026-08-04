"""The DeepEP shared expert must be data parallel, not tensor parallel.

``_forward_deepep`` runs on this rank's token shard (``post_attn_comm``
reduce-scatters whenever ``attn_tp_size != moe.tp_ep_size``). A tensor-parallel
shared expert there would reduce partial products belonging to *different*
tokens, so its weights have to be replicated instead.
"""

from __future__ import annotations

import unittest

from tokenspeed.runtime.distributed.mapping import Mapping
from tokenspeed.runtime.layers.linear import (
    MergedColumnParallelLinear,
    ReplicatedLinear,
    RowParallelLinear,
)
from tokenspeed.runtime.models.qwen3_5_moe import Qwen3_5MoeMLP

HIDDEN = 256
INTERMEDIATE = 512


def _mlp(world_size: int, replicate: bool) -> Qwen3_5MoeMLP:
    return Qwen3_5MoeMLP(
        hidden_size=HIDDEN,
        intermediate_size=INTERMEDIATE,
        hidden_act="silu",
        mapping=Mapping(rank=0, world_size=world_size),
        quant_config=None,
        reduce_results=False,
        replicate_weights=replicate,
    )


class TestSharedExpertReplication(unittest.TestCase):
    def test_tensor_parallel_by_default(self):
        mlp = _mlp(world_size=4, replicate=False)
        self.assertIsInstance(mlp.gate_up_proj, MergedColumnParallelLinear)
        self.assertIsInstance(mlp.down_proj, RowParallelLinear)
        # Sharded: each rank holds 1/4 of the intermediate dimension.
        self.assertEqual(mlp.gate_up_proj.weight.shape, (2 * INTERMEDIATE // 4, HIDDEN))
        self.assertEqual(mlp.down_proj.weight.shape, (HIDDEN, INTERMEDIATE // 4))

    def test_replicated_keeps_full_weights(self):
        mlp = _mlp(world_size=4, replicate=True)
        self.assertIsInstance(mlp.gate_up_proj, ReplicatedLinear)
        self.assertIsInstance(mlp.down_proj, ReplicatedLinear)
        self.assertEqual(mlp.gate_up_proj.weight.shape, (2 * INTERMEDIATE, HIDDEN))
        self.assertEqual(mlp.down_proj.weight.shape, (HIDDEN, INTERMEDIATE))

    def test_replication_lifts_the_block_quant_shard_floor(self):
        """Replicated weights make the dense TP degree irrelevant.

        Block-quantized FP8 requires every shard to stay >= the 128-wide
        quantization block, which caps dense TP at ``INTERMEDIATE / 128`` when
        the shared expert is sharded. Replication removes that cap, so degrees
        past it (TP8 for a 512 intermediate) stay loadable.
        """
        for world_size in (4, 8, 16):
            mlp = _mlp(world_size=world_size, replicate=True)
            self.assertEqual(mlp.gate_up_proj.weight.shape, (2 * INTERMEDIATE, HIDDEN))


if __name__ == "__main__":
    unittest.main()
