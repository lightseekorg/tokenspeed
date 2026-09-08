"""Cache-group binding: the plan decides storage, the layer declares visibility.

``PagedAttention`` carries no group id at construction. ``bind_cache_groups``
stamps each layer from the pool's plan and enforces the one relation between
the two contracts -- a group must retain every token its layers can see. The
pool side is exercised over a real (tiny, CPU) MHA plan so the layer -> group
mapping is read back from planned fields, not restated.
"""

from __future__ import annotations

import os
import sys
import unittest
from types import SimpleNamespace

import torch
from torch import nn

# CI Registration (parsed via AST, runtime no-op)
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from ci_system.ci_register import register_cuda_ci

register_cuda_ci(est_time=5, suite="runtime-1gpu")

from tokenspeed.runtime.layers.paged_attention import (  # noqa: E402
    PagedAttention,
    bind_cache_groups,
    check_block_drafter_storage,
    hf_sliding_window_to_window_left,
)

FULL = "full_attention"
SWA = "sliding_attention"


def _spec(group_id, retention, sliding_window_tokens):
    return SimpleNamespace(
        group_id=group_id,
        retention=retention,
        sliding_window_tokens=sliding_window_tokens,
        family="history",
    )


def _pool(group_by_layer, specs):
    return SimpleNamespace(
        arena=SimpleNamespace(cache_group_specs=tuple(specs)),
        history_group_by_layer=lambda: dict(group_by_layer),
    )


def _layer(layer_id, sliding_window_size):
    return PagedAttention(
        num_heads=1,
        head_dim=4,
        scaling=1.0,
        num_kv_heads=1,
        layer_id=layer_id,
        sliding_window_size=sliding_window_size,
    )


class _Model(nn.Module):
    def __init__(self, *layers):
        super().__init__()
        self.attns = nn.ModuleList(layers)


class PagedAttentionStorageStateTest(unittest.TestCase):
    def test_group_id_is_unbound_until_bound(self):
        layer = _layer(0, sliding_window_size=-1)
        with self.assertRaisesRegex(RuntimeError, "no cache group bound"):
            layer.group_id
        layer.bind_cache_group(FULL)
        self.assertEqual(layer.group_id, FULL)

    def test_rebinding_to_another_group_is_a_bug(self):
        layer = _layer(0, sliding_window_size=-1)
        layer.bind_cache_group(FULL)
        layer.bind_cache_group(FULL)  # idempotent
        with self.assertRaisesRegex(ValueError, "cannot rebind"):
            layer.bind_cache_group(SWA)
        with self.assertRaisesRegex(ValueError, "nonempty"):
            _layer(1, sliding_window_size=-1).bind_cache_group("")

    def test_hf_window_counts_the_current_token(self):
        self.assertEqual(hf_sliding_window_to_window_left(1024), 1023)

    def test_zero_window_left_is_a_window_not_full_attention(self):
        # HF sliding_window=1 sees the current token only; the layer must keep
        # that 0 rather than fold it into -1 by truthiness.
        layer = _layer(0, sliding_window_size=hf_sliding_window_to_window_left(1))
        self.assertEqual(layer.sliding_window_size, 0)
        bind_cache_groups(
            _Model(layer), _pool({0: SWA}, (_spec(SWA, "sliding_window", 1),))
        )
        self.assertEqual(layer.group_id, SWA)

    def test_window_left_must_be_an_int_at_or_above_minus_one(self):
        for bad in (None, -2):
            with self.assertRaisesRegex(ValueError, "sliding_window_size"):
                _layer(0, sliding_window_size=bad)


class BindCacheGroupsTest(unittest.TestCase):
    def test_binds_each_layer_from_the_plan(self):
        model = _Model(
            _layer(0, sliding_window_size=-1), _layer(1, sliding_window_size=127)
        )
        bind_cache_groups(
            model,
            _pool(
                {0: FULL, 1: SWA},
                (_spec(FULL, "full_history", None), _spec(SWA, "sliding_window", 128)),
            ),
        )
        self.assertEqual([m.group_id for m in model.attns], [FULL, SWA])

    def test_layer_missing_from_the_plan_raises(self):
        model = _Model(
            _layer(0, sliding_window_size=-1), _layer(3, sliding_window_size=-1)
        )
        with self.assertRaisesRegex(ValueError, r"layer_id=3.*no history-family"):
            bind_cache_groups(
                model, _pool({0: FULL}, (_spec(FULL, "full_history", None),))
            )

    def test_full_visibility_cannot_ride_a_sliding_group(self):
        model = _Model(_layer(0, sliding_window_size=-1))
        with self.assertRaisesRegex(ValueError, "sees the full history"):
            bind_cache_groups(
                model, _pool({0: SWA}, (_spec(SWA, "sliding_window", 128),))
            )

    def test_sliding_mask_must_fit_the_retention_window(self):
        # window_left 128 needs 129 retained tokens; the group keeps 128.
        model = _Model(_layer(0, sliding_window_size=128))
        with self.assertRaisesRegex(ValueError, "retains only a 128-token window"):
            bind_cache_groups(
                model, _pool({0: SWA}, (_spec(SWA, "sliding_window", 128),))
            )

    def test_sliding_mask_on_a_full_group_is_fine(self):
        # A block drafter: sliding compute mask, full-history storage.
        model = _Model(_layer(0, sliding_window_size=1023))
        bind_cache_groups(model, _pool({0: FULL}, (_spec(FULL, "full_history", None),)))
        self.assertEqual(model.attns[0].group_id, FULL)


class BlockDrafterStorageTest(unittest.TestCase):
    def _bound_draft(self, group_id):
        model = _Model(_layer(0, sliding_window_size=1023))
        model.attns[0].bind_cache_group(group_id)
        return model

    def test_shares_the_targets_full_history_group(self):
        target = _pool(
            {0: FULL, 1: SWA},
            (_spec(FULL, "full_history", None), _spec(SWA, "sliding_window", 128)),
        )
        check_block_drafter_storage(self._bound_draft(FULL), target)

    def test_target_without_a_full_history_group_is_rejected(self):
        # The plan gave the draft a full_attention group of its own; no
        # target layer lives there, so there is nothing to borrow.
        target = _pool(
            {0: SWA},
            (_spec(SWA, "sliding_window", 128), _spec(FULL, "full_history", None)),
        )
        with self.assertRaisesRegex(ValueError, "must share a full-history group"):
            check_block_drafter_storage(self._bound_draft(FULL), target)

    def test_sliding_target_group_is_rejected(self):
        target = _pool({0: SWA}, (_spec(SWA, "sliding_window", 128),))
        with self.assertRaisesRegex(ValueError, "must share a full-history group"):
            check_block_drafter_storage(self._bound_draft(SWA), target)


class HistoryGroupByLayerOverRealPlanTest(unittest.TestCase):
    """The pool reads layer -> group back from the planned KV fields."""

    def _pool(self, *, layer_types, sliding_window_tokens, num_draft_layers):
        from cache_pool_test_utils import (
            make_layer_group_ids,
            make_mha_memory_plan,
            make_pool,
            specs_for_layers,
        )

        from tokenspeed.runtime.layers.attention.kv_cache.mha import (
            MHATokenToKVPool,
        )

        layer_num = len(layer_types)
        plan = make_mha_memory_plan(
            size=64,
            prefix_granularity=16,
            layer_num=layer_num,
            kv_heads=1,
            head_dim=4,
            dtype=torch.float16,
            layer_types=layer_types,
            sliding_window_tokens=sliding_window_tokens,
        )
        group_ids = make_layer_group_ids(
            layer_num=layer_num,
            layer_types=layer_types,
            sliding_window_tokens=sliding_window_tokens,
        )
        specs = specs_for_layers(
            layer_types=layer_types,
            group_ids=group_ids,
            sliding_window_tokens=sliding_window_tokens,
            prefix_granularity=16,
        )
        arena, target = make_pool(
            MHATokenToKVPool,
            plan,
            device="cpu",
            cache_group_specs=specs,
            dtype=torch.float16,
            head_num=1,
            head_dim=4,
            layer_num=layer_num - num_draft_layers,
            rank=0,
        )
        draft = None
        if num_draft_layers:
            draft = MHATokenToKVPool(
                arena=arena,
                dtype=torch.float16,
                head_num=1,
                head_dim=4,
                layer_num=num_draft_layers,
                rank=0,
                field_layer_offset=layer_num - num_draft_layers,
            )
        return target, draft

    def test_hybrid_target_maps_each_layer_to_its_label_group(self):
        target, _ = self._pool(
            layer_types=(FULL, SWA, FULL),
            sliding_window_tokens=32,
            num_draft_layers=0,
        )
        self.assertEqual(target.history_group_by_layer(), {0: FULL, 1: SWA, 2: FULL})

    def test_draft_view_maps_local_ids_onto_continuation_layers(self):
        target, draft = self._pool(
            layer_types=(FULL, SWA, FULL),
            sliding_window_tokens=32,
            num_draft_layers=1,
        )
        self.assertEqual(target.history_group_by_layer(), {0: FULL, 1: SWA})
        self.assertEqual(draft.history_group_by_layer(), {0: FULL})

    def test_binding_end_to_end_over_the_real_plan(self):
        target, draft = self._pool(
            layer_types=(FULL, SWA, FULL),
            sliding_window_tokens=32,
            num_draft_layers=1,
        )
        target_model = _Model(
            _layer(0, sliding_window_size=-1), _layer(1, sliding_window_size=31)
        )
        bind_cache_groups(target_model, target)
        self.assertEqual([m.group_id for m in target_model.attns], [FULL, SWA])
        # The block drafter masks to its own window but rides the target's
        # full-history group -- the plan said so, the model never did.
        draft_model = _Model(_layer(0, sliding_window_size=7))
        bind_cache_groups(draft_model, draft)
        self.assertEqual(draft_model.attns[0].group_id, FULL)
        check_block_drafter_storage(draft_model, target)


if __name__ == "__main__":
    unittest.main()
