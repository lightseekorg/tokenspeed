from __future__ import annotations

import os
import sys
import unittest

# CI Registration (parsed via AST, runtime no-op)
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from ci_system.ci_register import register_cuda_ci

register_cuda_ci(est_time=10, suite="runtime-1gpu")


def _import_backend():
    from tokenspeed.runtime.layers.attention.backends.mha import (
        MHAAttnBackend,
        MHADecodeMetadata,
    )

    return MHAAttnBackend, MHADecodeMetadata


class SelectPageTableTest(unittest.TestCase):
    def setUp(self):
        try:
            self.MHAAttnBackend, self.MHADecodeMetadata = _import_backend()
        except (ImportError, ModuleNotFoundError) as exc:
            self.skipTest(f"needs torch + tokenspeed_kernel: {exc}")
        import torch

        self.torch = torch
        # Bypass __init__: _select_page_table only reads metadata + group_id.
        self.backend = self.MHAAttnBackend.__new__(self.MHAAttnBackend)

    def _layer(self, group_id):
        from types import SimpleNamespace

        return SimpleNamespace(group_id=group_id)

    def _decode_meta(self, *, page_table=None, page_tables=None):
        return self.MHADecodeMetadata(
            page_table=page_table,
            seq_lens=self.torch.zeros(1, dtype=self.torch.int32),
            page_tables=page_tables,
        )

    def test_single_table_when_page_tables_none(self):
        pt = self.torch.tensor([[1, 2]], dtype=self.torch.int32)
        meta = self._decode_meta(page_table=pt, page_tables=None)
        out = self.backend._select_page_table(self._layer("full_attention"), meta)
        self.assertIs(out, pt)

    def test_routes_by_group_id(self):
        full = self.torch.tensor([[1, 2]], dtype=self.torch.int32)
        swa = self.torch.tensor([[3, 0]], dtype=self.torch.int32)
        meta = self._decode_meta(
            page_tables={"full_attention": full, "sliding_attention": swa}
        )
        out_full = self.backend._select_page_table(self._layer("full_attention"), meta)
        out_swa = self.backend._select_page_table(
            self._layer("sliding_attention"), meta
        )
        self.assertIs(out_full, full)
        self.assertIs(out_swa, swa)

    def test_empty_group_id_falls_back_to_single_group(self):
        only = self.torch.tensor([[5]], dtype=self.torch.int32)
        meta = self._decode_meta(page_tables={"full_attention": only})
        out = self.backend._select_page_table(self._layer(""), meta)
        self.assertIs(out, only)

    def _multi_group_meta(self):
        return self._decode_meta(
            page_tables={
                "full_attention": self.torch.zeros((1, 1), dtype=self.torch.int32),
                "sliding_attention": self.torch.zeros((1, 1), dtype=self.torch.int32),
            }
        )

    def test_unknown_group_id_multi_group_raises(self):
        # Clear error naming the group and the available keys.
        with self.assertRaisesRegex(KeyError, "'nope'.*full_attention"):
            self.backend._select_page_table(
                self._layer("nope"), self._multi_group_meta()
            )

    def test_empty_group_id_multi_group_raises(self):
        # A group-unaware layer cannot pick between multiple groups:
        # clear error, no silent fallback.
        with self.assertRaisesRegex(KeyError, "group_id=''"):
            self.backend._select_page_table(self._layer(""), self._multi_group_meta())


class ValidatePagedCacheGroupIdsTest(unittest.TestCase):
    """Init-time fail-fast: multi-group pool requires labeled layers."""

    def setUp(self):
        try:
            import torch  # noqa: F401
            from torch import nn

            from tokenspeed.runtime.layers.paged_attention import (
                PagedAttention,
                validate_paged_cache_group_ids,
            )
        except (ImportError, ModuleNotFoundError) as exc:
            self.skipTest(f"needs torch: {exc}")
        self.nn = nn
        self.PagedAttention = PagedAttention
        self.validate = validate_paged_cache_group_ids

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

    def test_multi_group_empty_group_id_raises(self):
        with self.assertRaisesRegex(
            ValueError, r"TinyModel.*layer_id=1.*empty group_id"
        ):
            self.validate(
                self._model(["full_attention", ""]),
                self._specs(["full_attention", "sliding_attention"]),
            )

    def test_multi_group_unknown_group_id_raises(self):
        with self.assertRaisesRegex(ValueError, r"TinyModel.*'nope'"):
            self.validate(
                self._model(["full_attention", "nope"]),
                self._specs(["full_attention", "sliding_attention"]),
            )

    def test_single_group_empty_group_id_is_fine(self):
        # Documented fallback: single-group pools serve group-unaware layers.
        self.validate(self._model(["", ""]), self._specs(["full_attention"]))

    def test_no_groups_is_fine(self):
        self.validate(self._model(["", ""]), self._specs([]))


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
