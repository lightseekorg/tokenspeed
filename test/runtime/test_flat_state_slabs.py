"""Unit tests for FlatStateSlabs (extracted from MHATokenToKVPool).

Covers the flat-GDN gate, get_state_buffers happy/error paths,
state_layer_ids exposure, and the inactive-when-no-shapes case. The
equalization pre-check is exercised too (same trigger, same ValueError).
"""

from __future__ import annotations

import os
import sys
import unittest
from unittest import mock

# CI Registration (parsed via AST, runtime no-op)
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from ci_system.ci_register import register_cuda_ci

register_cuda_ci(est_time=10, suite="runtime-1gpu")

_FLAT_GATE = "tokenspeed.runtime.configs.paged_cache_spec.scheduler_ext_flat_kvcache"


class FlatStateSlabsTest(unittest.TestCase):
    def setUp(self):
        try:
            import torch

            from tokenspeed.runtime.layers.attention.kv_cache.flat_state_slabs import (  # noqa: E501
                FlatStateSlabs,
            )
        except (ImportError, ModuleNotFoundError) as exc:
            self.skipTest(f"needs torch + tokenspeed_kernel: {exc}")
        self.torch = torch
        self.cls = FlatStateSlabs
        # 4 layers, 2 of them GDN state layers (occurrence order 0, 1).
        self.layer_types = (
            "full_attention",
            "linear_attention",
            "full_attention",
            "linear_attention",
        )
        self.conv_shape = (3, 2)
        # Non-square K-last [Hv, V, K] shape catches accidental V/K swaps.
        self.ssm_shape = (4, 3, 2)

    def _make(self, *, layer_types=None, conv=None, ssm=None, page_size=1, size=8):
        return self.cls(
            layer_types=self.layer_types if layer_types is None else layer_types,
            conv_state_shape=self.conv_shape if conv is None else conv,
            temporal_state_shape=self.ssm_shape if ssm is None else ssm,
            conv_dtype=self.torch.float32,
            ssm_dtype=self.torch.float32,
            default_dtype=self.torch.float16,
            page_size=page_size,
            size=size,
            kv_bytes_per_slot=4096,
        )

    def test_active_when_flat_gate_on(self):
        with mock.patch(_FLAT_GATE, return_value=True):
            s = self._make()
        self.assertTrue(s.is_active)
        self.assertEqual(s.state_layer_ids, frozenset({1, 3}))
        self.assertTrue(s.is_state_layer(1))
        self.assertFalse(s.is_state_layer(0))

    def test_inactive_when_flat_gate_off(self):
        with mock.patch(_FLAT_GATE, return_value=False):
            s = self._make()
            s.allocate("cpu")
        self.assertFalse(s.is_active)
        self.assertEqual(s.state_layer_ids, frozenset())
        self.assertFalse(s.is_state_layer(1))
        self.assertEqual(s.state_slabs, [])
        # get_state_buffers on a state layer with no slabs raises.
        with self.assertRaises(ValueError):
            s.get_state_buffers(1)

    def test_inactive_when_no_shapes(self):
        with mock.patch(_FLAT_GATE, return_value=True):
            s = self._make(conv=None, ssm=None)  # will be passed None below
        self.assertTrue(s.is_active)  # sanity: shapes present -> active
        # Now with shapes missing the gate must be off regardless of ext.
        with mock.patch(_FLAT_GATE, return_value=True):
            s2 = self.cls(
                layer_types=self.layer_types,
                conv_state_shape=None,
                temporal_state_shape=None,
                conv_dtype=None,
                ssm_dtype=None,
                default_dtype=self.torch.float16,
                page_size=1,
                size=8,
                kv_bytes_per_slot=4096,
            )
            s2.allocate("cpu")
        self.assertFalse(s2.is_active)
        self.assertEqual(s2.state_slabs, [])
        self.assertEqual(s2.state_layer_ids, frozenset())

    def test_inactive_when_no_state_layers(self):
        with mock.patch(_FLAT_GATE, return_value=True):
            s = self._make(layer_types=("full_attention", "full_attention"))
            s.allocate("cpu")
        self.assertFalse(s.is_active)
        self.assertEqual(s.state_slabs, [])

    def test_allocate_shapes_and_null_page(self):
        with mock.patch(_FLAT_GATE, return_value=True):
            s = self._make(page_size=2, size=8)
            s.allocate("cpu")
        # 2 state layers -> 2 (conv, ssm) pairs.
        self.assertEqual(len(s.state_slabs), 2)
        # num_pages_with_null = size // page_size + 1 = 5.
        self.assertEqual(s.num_pages_with_null, 5)
        for conv, ssm in s.state_slabs:
            self.assertEqual(tuple(conv.shape), (5, *self.conv_shape))
            self.assertEqual(tuple(ssm.shape), (5, *self.ssm_shape))
            self.assertEqual(conv.dtype, self.torch.float32)
            self.assertEqual(ssm.dtype, self.torch.float32)
            # Row 0 is the never-written null page: zero-initialized.
            self.assertTrue(bool((conv[0] == 0).all()))
            self.assertTrue(bool((ssm[0] == 0).all()))

    def test_get_state_buffers_binding_and_errors(self):
        with mock.patch(_FLAT_GATE, return_value=True):
            s = self._make(page_size=2, size=8)
            s.allocate("cpu")
        # n-th state layer binds pair n: layer 1 -> pair 0, layer 3 -> pair 1.
        conv1, ssm1 = s.get_state_buffers(1)
        conv3, ssm3 = s.get_state_buffers(3)
        self.assertIs(conv1, s.state_slabs[0][0])
        self.assertIs(ssm1, s.state_slabs[0][1])
        self.assertIs(conv3, s.state_slabs[1][0])
        self.assertIs(ssm3, s.state_slabs[1][1])
        # Non-state layer raises.
        with self.assertRaises(ValueError):
            s.get_state_buffers(0)
        with self.assertRaises(ValueError):
            s.get_state_buffers(2)

    def test_equalization_precheck_raises(self):
        # A tiny page_size that cannot cover the constant state rows must
        # raise ValueError in __init__ (before any allocation).
        big_conv = (1024, 1024)
        with self.assertRaises(ValueError) as ctx:
            self._make(conv=big_conv, page_size=1, size=8)
        self.assertIn("pre-equalized", str(ctx.exception))


if __name__ == "__main__":
    unittest.main()
