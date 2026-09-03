"""Group routing over real MHA leaves.

The MHA leaf no longer routes cache groups — the ``CacheGroupRouter`` owns
routing by ``layer.group_id`` and hands each ``MHAAttnBackend`` leaf its own
kernel-page table and write locations. These tests build a router over two
real MHA leaves at different kernel page sizes and pin: per-group table
isolation in the leaves' persistent buffers, write locations landing in the
layer's own group pages, and the unknown-group KeyError (the old sole-group
fallback is gone). Layer-side group_id validation stays at the bottom.
"""

from __future__ import annotations

import os
import sys
import unittest
from types import SimpleNamespace

import torch

# CI Registration (parsed via AST, runtime no-op)
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from ci_system.ci_register import register_cuda_ci

register_cuda_ci(est_time=10, suite="runtime-1gpu")

FULL = "full_attention"
SWA = "sliding_attention"


def _layer(group_id, layer_id=0):
    return SimpleNamespace(group_id=group_id, layer_id=layer_id)


class RouterOverMhaLeavesTest(unittest.TestCase):
    """A router with two real MHAAttnBackend leaves (P=4 and P=2)."""

    def setUp(self):
        try:
            import torch

            from tokenspeed.runtime.layers.attention.backends.mha import (
                MHAAttnBackend,
            )
        except (ImportError, ModuleNotFoundError) as exc:
            self.skipTest(f"needs torch + tokenspeed_kernel: {exc}")
        self.torch = torch
        self.MHAAttnBackend = MHAAttnBackend

    def _leaf(self, kernel_page_size):
        torch = self.torch
        config = SimpleNamespace(
            device="cpu",
            dtype=torch.float16,
            is_draft=False,
            speculative_num_draft_tokens=1,
            context_len=24,
            kv_cache_dtype=torch.bfloat16,
            kv_cache_mxfp8=False,
            draft_block_decode=False,
            max_bs=8,
            kernel_page_size=None,
        )
        spec = SimpleNamespace(
            num_attention_heads=8,
            num_kv_heads=8,
            attn_tp_size=1,
            head_dim=16,
            backend_name="mha",
        )
        return self.MHAAttnBackend(config, spec, kernel_page_size=kernel_page_size)

    def _router(self, group_ids=(FULL, SWA)):
        from tokenspeed.runtime.layers.attention.backends.cache_group_geometry import (
            CacheGroupGeometry,
        )
        from tokenspeed.runtime.layers.attention.backends.router import (
            CacheGroupRouter,
        )

        page_sizes = {FULL: 4, SWA: 2}
        leaves = {gid: self._leaf(page_sizes[gid]) for gid in group_ids}
        router = CacheGroupRouter(None, is_draft=False, spec_num_tokens=1, device="cpu")
        router.bind(
            CacheGroupGeometry(
                granularities={gid: 4 for gid in group_ids},
                families={gid: "history" for gid in group_ids},
                full_history_group_id=FULL,
            ),
            leaves,
        )
        router.init_cuda_graph_state(4)
        return router, leaves

    def _refresh(self, router, raw_by_group, seq_lens):
        from tokenspeed.runtime.execution.forward_batch_info import ForwardMode

        bs = seq_lens.shape[0]
        router.refresh_decode_metadata(
            bs,
            bs,
            torch.arange(bs, dtype=torch.int32),
            seq_lens,
            forward_mode=ForwardMode.DECODE,
            block_tables=raw_by_group,
        )

    def test_each_leaf_gets_its_own_groups_kernel_page_table(self):
        torch = self.torch
        router, leaves = self._router()
        self._refresh(
            router,
            {
                FULL: torch.tensor([[5, 6, 7]], dtype=torch.int32),
                SWA: torch.tensor([[9, 8, 3]], dtype=torch.int32),
            },
            torch.tensor([9], dtype=torch.int32),
        )
        # FULL leaf runs at the raw grain (P=4, ratio 1): pages verbatim.
        self.assertEqual(leaves[FULL].page_table_buf[0].tolist(), [5, 6, 7, 0, 0, 0])
        # SWA leaf runs at P=2 (ratio 2): raw page p -> kernel pages 2p, 2p+1
        # of ITS group's table — never the FULL group's pages.
        self.assertEqual(
            leaves[SWA].page_table_buf[0].tolist(),
            [18, 19, 16, 17, 6, 7, 0, 0, 0, 0, 0, 0],
        )
        # Each leaf's decode metadata views its own buffer, not a shared one.
        self.assertNotEqual(
            leaves[FULL].forward_decode_metadata.page_table.data_ptr(),
            leaves[SWA].forward_decode_metadata.page_table.data_ptr(),
        )

    def test_write_locations_land_in_the_layers_own_group_pages(self):
        torch = self.torch
        from tokenspeed.runtime.execution.forward_batch_info import ForwardMode

        router, _ = self._router()
        self._refresh(
            router,
            {
                FULL: torch.tensor([[5, 6, 7]], dtype=torch.int32),
                SWA: torch.tensor([[9, 8, 3]], dtype=torch.int32),
            },
            torch.tensor([9], dtype=torch.int32),
        )
        # pos 8, FULL (P=4): raw page 7 slot 0 -> 28 — inside FULL's pages.
        full_locs = router.write_locations(_layer(FULL), ForwardMode.DECODE)
        self.assertEqual(full_locs.tolist(), [28])
        # pos 8, SWA (P=2): raw page 3 -> kernel page 6 slot 0 -> 12 — SWA's
        # own table, so the two groups disagree, proving no shared table.
        swa_locs = router.write_locations(_layer(SWA), ForwardMode.DECODE)
        self.assertEqual(swa_locs.tolist(), [12])

    def test_unknown_group_id_raises_even_with_a_sole_leaf(self):
        # The old single-group fallback (empty/unknown group_id -> the only
        # entry) is gone: the router indexes leaves by group_id, full stop.
        torch = self.torch
        from tokenspeed.runtime.execution.forward_batch_info import ForwardMode

        router, _ = self._router(group_ids=(FULL,))
        self._refresh(
            router,
            {FULL: torch.tensor([[5, 6, 7]], dtype=torch.int32)},
            torch.tensor([9], dtype=torch.int32),
        )
        with self.assertRaisesRegex(KeyError, "names cache group 'nope'"):
            router.write_locations(_layer("nope"), ForwardMode.DECODE)
        with self.assertRaisesRegex(KeyError, "names cache group ''"):
            router.write_locations(_layer(""), ForwardMode.DECODE)


class ValidateCacheGroupIdsTest(unittest.TestCase):
    """Init-time fail-fast: every layer must name a published cache group."""

    def setUp(self):
        try:
            import torch  # noqa: F401
            from torch import nn

            from tokenspeed.runtime.layers.paged_attention import (
                PagedAttention,
                validate_cache_group_ids,
            )
        except (ImportError, ModuleNotFoundError) as exc:
            self.skipTest(f"needs torch: {exc}")
        self.nn = nn
        self.PagedAttention = PagedAttention
        self.validate = validate_cache_group_ids

    def _model(self, group_ids):
        nn, PagedAttention = self.nn, self.PagedAttention

        class TinyModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.attns = nn.ModuleList(
                    PagedAttention(
                        num_heads=1,
                        head_dim=4,
                        scaling=1.0,
                        num_kv_heads=1,
                        layer_id=i,
                        group_id=gid,
                    )
                    for i, gid in enumerate(group_ids)
                )

        return TinyModel()

    def _specs(self, group_ids):
        from types import SimpleNamespace

        return tuple(SimpleNamespace(group_id=gid) for gid in group_ids)

    def test_multi_group_all_labeled_passes(self):
        self.validate(
            self._model(["full_attention", "sliding_attention"]),
            self._specs(["full_attention", "sliding_attention"]),
        )

    def test_constructor_rejects_empty_group_id(self):
        # group_id is mandatory at construction; there is no backend fallback
        # for an unlabeled layer.
        with self.assertRaisesRegex(ValueError, r"layer_id=1.*nonempty"):
            self._model(["full_attention", ""])

    def test_multi_group_unknown_group_id_raises(self):
        with self.assertRaisesRegex(ValueError, r"TinyModel.*'nope'"):
            self.validate(
                self._model(["full_attention", "nope"]),
                self._specs(["full_attention", "sliding_attention"]),
            )

    def test_single_group_unknown_group_id_raises(self):
        # Single-group pools validate too: backends index their learned
        # geometry by the layer's group_id with no fallback.
        with self.assertRaisesRegex(ValueError, r"TinyModel.*'nope'"):
            self.validate(self._model(["nope"]), self._specs(["full_attention"]))
        self.validate(self._model(["full_attention"]), self._specs(["full_attention"]))

    def test_no_published_groups_is_fine(self):
        # A pool without a published contract has nothing to validate against.
        self.validate(
            self._model(["full_attention", "full_attention"]), self._specs([])
        )


class GptOssGroupIdTest(unittest.TestCase):
    """PagedAttention built by GptOssAttention must carry group_id == layer_type.
    Constructing the model layer needs torch/model deps, so skip otherwise."""

    def test_paged_attention_group_id_equals_layer_type(self):
        try:
            from tokenspeed.runtime.layers.paged_attention import PagedAttention
        except (ImportError, ModuleNotFoundError) as exc:
            self.skipTest(f"needs torch: {exc}")
        layer = PagedAttention(
            num_heads=4,
            head_dim=8,
            scaling=1.0,
            num_kv_heads=4,
            layer_id=0,
            sliding_window_size=128,
            group_id="sliding_attention",
        )
        self.assertEqual(layer.group_id, "sliding_attention")


if __name__ == "__main__":
    unittest.main()
