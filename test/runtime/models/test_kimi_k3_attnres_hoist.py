"""Slot invariants for the hoisted AttnRes mlp-side partial.

Layer L's mlp-side partial is computed on layer L-1's aux sweep, so it lands a
layer before the all-reduce that reads it. That is safe only while a layer's
own slot differs from the one its aux branch writes for the next layer, and
only where the previous layer sweeps the same block range.

Usage:
    cd test/runtime
    python3 -m unittest models.test_kimi_k3_attnres_hoist -v
"""

import unittest

import torch

from tokenspeed.runtime.models.kimi_k3 import _attnres_mlp_slot, _attnres_scratch

LAYERS = 93
BLOCK = 12  # config.attn_res_block_size for Kimi-K3


def _mlp_blocks(layer_id):
    """Blocks layer ``layer_id``'s mlp-side partial must cover."""
    return -(-layer_id // BLOCK) + (1 if layer_id % BLOCK == 0 else 0)


class TestAttnResMlpHoist(unittest.TestCase):
    def test_adjacent_layers_use_different_mlp_slots(self):
        """A layer must not read the slot its own aux branch is writing."""
        for i in range(LAYERS - 1):
            self.assertNotEqual(_attnres_mlp_slot(i), _attnres_mlp_slot(i + 1))

    def test_mlp_slots_never_take_the_attn_side_slot(self):
        """Slot 1 belongs to the attn-side mix."""
        for i in range(LAYERS):
            self.assertNotEqual(_attnres_mlp_slot(i), 1)

    def test_hoist_keys_on_block_write_exactly_where_the_range_matches(self):
        """The wiring hoists iff not is_block_write_layer; that must coincide
        with the previous layer sweeping the same block range."""
        for i in range(1, LAYERS):
            same_range = _mlp_blocks(i) == _mlp_blocks(i - 1)
            self.assertEqual(i % BLOCK != 0, same_range, f"layer {i}")

    def test_scratch_pool_serves_every_slot_without_aliasing(self):
        """Every slot in use must be a distinct buffer."""
        like = torch.zeros(1, 8)
        used = {1} | {_attnres_mlp_slot(i) for i in range(LAYERS)}
        ptrs = {s: _attnres_scratch(like, slot=s)[2].data_ptr() for s in sorted(used)}
        self.assertEqual(len(set(ptrs.values())), len(ptrs), ptrs)


if __name__ == "__main__":
    unittest.main()
