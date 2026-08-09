"""Scratch-slot and block-range invariants for the hoisted AttnRes mlp partial.

Layer L's mlp-side partial is computed on layer L-1's aux stream, riding the
sweep already there, so it lands a full layer before the all-reduce that reads
it. Two things have to hold for that to be safe, and both are cheap to check
without a GPU or a checkpoint:

1. **Range** -- layer L-1 sweeps ``block_residual[:mlp(L-1)]``, so hoisting is
   only legal where ``mlp(L) == mlp(L-1)``. The block-write layers fold in the
   block they snapshot themselves and must keep the in-layer partial.
2. **Lifetime** -- layer L's all-reduce reads its mlp slot while layer L's own
   aux branch writes the *next* layer's. Those must be different buffers, or
   the hoist reintroduces exactly the write/read race it removes.

Usage:
    cd test/runtime
    python3 -m unittest models.test_kimi_k3_attnres_hoist -v
"""

import unittest

from tokenspeed.runtime.models.kimi_k3 import (
    attnres_mlp_is_hoistable,
    attnres_mlp_slot,
)

# (num_layers, attn_res_block_size); the first is the shipping Kimi-K3 shape.
SHAPES = ((93, 12), (93, 1), (64, 8), (13, 12), (2, 12), (48, 16))


def _ceil_div(a, b):
    return -(-a // b)


def _mlp_blocks(layer_id, block):
    """Blocks layer ``layer_id``'s mlp-side partial must cover."""
    return _ceil_div(layer_id, block) + (1 if layer_id % block == 0 else 0)


class TestAttnResMlpHoist(unittest.TestCase):
    def test_hoistable_exactly_when_block_range_matches(self):
        """Hoisting is legal precisely where the previous layer's sweep covers us."""
        for n, block in SHAPES:
            for layer_id in range(1, n):
                same_range = _mlp_blocks(layer_id, block) == _mlp_blocks(
                    layer_id - 1, block
                )
                self.assertEqual(
                    attnres_mlp_is_hoistable(layer_id, block),
                    same_range,
                    f"layers={n} block={block} layer={layer_id}: hoistable disagrees "
                    f"with whether the previous layer sweeps the same blocks",
                )

    def test_layer_zero_is_never_hoistable(self):
        """Layer 0 has no predecessor to ride, whatever the block size."""
        for _, block in SHAPES:
            self.assertFalse(attnres_mlp_is_hoistable(0, block))

    def test_block_write_layers_are_the_only_exclusions(self):
        """The non-hoistable set is exactly {0} plus the block-write layers."""
        for n, block in SHAPES:
            excluded = {
                layer_id
                for layer_id in range(n)
                if not attnres_mlp_is_hoistable(layer_id, block)
            }
            expected = {layer_id for layer_id in range(n) if layer_id % block == 0}
            self.assertEqual(excluded, expected, f"layers={n} block={block}")

    def test_adjacent_layers_use_different_mlp_slots(self):
        """A layer's read must not alias the slot its own aux branch writes."""
        for n, _ in SHAPES:
            for layer_id in range(n - 1):
                self.assertNotEqual(
                    attnres_mlp_slot(layer_id),
                    attnres_mlp_slot(layer_id + 1),
                    f"layer {layer_id} reads the slot it concurrently writes",
                )

    def test_mlp_slots_never_collide_with_the_attn_side_slot(self):
        """Slot 1 belongs to the attn-side mix and must stay untouched."""
        for n, _ in SHAPES:
            for layer_id in range(n):
                self.assertNotEqual(attnres_mlp_slot(layer_id), 1)

    def test_scratch_pool_serves_every_slot_the_wiring_asks_for(self):
        """The pool must actually allocate each slot in use, not just index into it."""
        import torch

        from tokenspeed.runtime.models.kimi_k3 import _attnres_scratch

        like = torch.zeros(1, 8)
        used = {1} | {attnres_mlp_slot(i) for n, _ in SHAPES for i in range(n)}
        seen = {}
        for slot in sorted(used):
            m, s, acc = _attnres_scratch(like, slot=slot)
            self.assertEqual(acc.shape[1], like.shape[-1])
            seen[slot] = acc.data_ptr()
        self.assertEqual(
            len(set(seen.values())),
            len(seen),
            f"slots {sorted(used)} alias the same buffer: {seen}",
        )


class TestCrossLayerRefsAreNotSubmodules(unittest.TestCase):
    """A layer may not hold a bare reference to a sibling layer.

    ``nn.Module.__setattr__`` registers any ``nn.Module`` value as a SUBMODULE.
    Assigning the next layer directly would therefore alias every one of its
    parameters under a second name, which breaks checkpoint loading -- observed
    as ``MoECheckpointLoadError: ... the target parameter was not found``. The
    existing ``_next_attn_mix`` avoids this by storing a tuple; anything that
    points at a sibling layer must do the same. Checked against the source so
    the guard holds without building a 93-layer model.
    """

    CROSS_LAYER_ATTRS = ("_next_attn_mix", "_next_mlp_mix")

    def test_sibling_layer_refs_are_wrapped(self):
        import ast
        import inspect

        from tokenspeed.runtime.models import kimi_k3

        tree = ast.parse(inspect.getsource(kimi_k3))
        offenders = []
        for node in ast.walk(tree):
            if not isinstance(node, ast.Assign):
                continue
            for target in node.targets:
                if (
                    isinstance(target, ast.Attribute)
                    and target.attr in self.CROSS_LAYER_ATTRS
                    and not isinstance(node.value, (ast.Tuple, ast.Constant))
                ):
                    offenders.append(
                        f"line {node.lineno}: {target.attr} = "
                        f"{ast.dump(node.value)[:60]}"
                    )
        self.assertEqual(
            offenders,
            [],
            "cross-layer reference assigned unwrapped; a bare nn.Module here "
            "registers the sibling as a submodule and corrupts weight loading",
        )


if __name__ == "__main__":
    unittest.main()
