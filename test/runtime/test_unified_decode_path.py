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

"""Unified decode path invariants (docs/design/unified_path.md).

CPU-only checks of the single-decode-path contract over the router + leaf
architecture:

* eager refresh and padded graph refresh write the SAME persistent leaf
  buffers and produce identical live rows (eager is "refresh + forward",
  replay is "refresh + graph.replay()");
* per-bs metadata views are pointer-stable and lazily built for a bs never
  captured (the above-ladder decode path);
* the persistent buffers are sized by max decode bs, not the capture ladder;
* every leaf class carries the uniform leaf signature, every runner-facing
  node the runner signature (conformance).
"""

from __future__ import annotations

import os
import sys
import unittest
from types import SimpleNamespace

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from ci_system.ci_register import register_cuda_ci

register_cuda_ci(est_time=10, suite="runtime-1gpu")

from tokenspeed.runtime.execution.forward_batch_info import ForwardMode
from tokenspeed.runtime.layers.attention.backends.cache_group_geometry import (
    CacheGroupGeometry,
)

MAX_DECODE_BS = 8
LADDER_BS = 4  # capture ladder max, deliberately below MAX_DECODE_BS
MAX_NUM_PAGES = 6
FULL = "full_attention"


class _TorchCase(unittest.TestCase):
    def setUp(self):
        try:
            import torch
        except (ImportError, ModuleNotFoundError) as exc:
            self.skipTest(f"needs torch: {exc}")
        self.torch = torch


class _RouterCase(_TorchCase):
    """A single-group router over a real MHA leaf, buffers sized by
    MAX_DECODE_BS (the wrapper passes max_decode_bs to
    init_cuda_graph_state, never the capture-ladder max)."""

    def setUp(self):
        super().setUp()
        try:
            from tokenspeed.runtime.layers.attention.backends.mha import (
                MHAAttnBackend,
            )
        except (ImportError, ModuleNotFoundError) as exc:
            self.skipTest(f"needs tokenspeed_kernel: {exc}")
        from tokenspeed.runtime.layers.attention.backends.router import (
            CacheGroupRouter,
        )

        torch = self.torch
        leaf = MHAAttnBackend.__new__(MHAAttnBackend)
        leaf.spec_num_tokens = 1
        leaf.is_draft = False
        leaf.draft_block_decode = False
        leaf.max_num_pages = MAX_NUM_PAGES
        leaf.max_context_len = MAX_NUM_PAGES * 2
        leaf.kernel_page_size = 2
        leaf.device = "cpu"
        router = CacheGroupRouter(None, is_draft=False, spec_num_tokens=1, device="cpu")
        router.bind(
            CacheGroupGeometry(
                granularities={FULL: 2},
                families={FULL: "history"},
                full_history_group_id=FULL,
                history_block_granularity=2,
            ),
            {FULL: leaf},
        )
        router.init_cuda_graph_state(MAX_DECODE_BS)
        self.router = router
        self.leaf = leaf
        del torch

    def _tables(self, rows):
        torch = self.torch
        return {FULL: torch.arange(1, rows * 2 + 1, dtype=torch.int32).reshape(rows, 2)}

    def _refresh(self, bs, actual_bs, seq_lens, replay):
        torch = self.torch
        self.router.refresh_decode_metadata(
            bs,
            actual_bs,
            torch.arange(bs, dtype=torch.int64),
            seq_lens,
            forward_mode=ForwardMode.DECODE,
            block_tables=self._tables(actual_bs) if actual_bs else self._tables(bs),
            for_graph_replay=replay,
        )
        return self.leaf.forward_decode_metadata


class EagerMatchesPaddedReplayTest(_RouterCase):
    def test_live_rows_identical_across_paths(self):
        torch = self.torch
        seq = torch.tensor([5, 4], dtype=torch.int32)

        # Padded graph refresh: 2 live rows in a 4-row graph batch.
        padded_seq = torch.cat([seq, torch.ones(2, dtype=torch.int32)])
        self._refresh(LADDER_BS, 2, padded_seq, replay=True)
        replay_table = self.leaf.page_table_buf[:2].clone()
        replay_locs = self.router.write_locations(
            SimpleNamespace(group_id=FULL, layer_id=0), ForwardMode.DECODE
        )[:2].clone()
        # Padded rows landed on the null page.
        self.assertTrue((self.leaf.page_table_buf[2:LADDER_BS] == 0).all())

        # Eager refresh at the true bs (unpadded) over the same buffers.
        md = self._refresh(2, 2, seq, replay=False)
        torch.testing.assert_close(self.leaf.page_table_buf[:2], replay_table)
        torch.testing.assert_close(
            self.router.write_locations(
                SimpleNamespace(group_id=FULL, layer_id=0), ForwardMode.DECODE
            )[:2],
            replay_locs,
        )
        # Metadata views the SAME persistent storage on both paths.
        self.assertEqual(md.page_table.data_ptr(), self.leaf.page_table_buf.data_ptr())


class AboveLadderDecodeTest(_RouterCase):
    def test_refresh_serves_bs_above_capture_ladder(self):
        torch = self.torch
        bs = MAX_DECODE_BS  # above LADDER_BS: no graph exists, same refresh
        seq = torch.arange(3, 3 + bs, dtype=torch.int32)
        md = self._refresh(bs, bs, seq, replay=False)
        self.assertEqual(md.seq_lens.shape[0], bs)
        torch.testing.assert_close(self.leaf.seq_lens_buf[:bs], seq)
        # Lazy per-bs views: built once, pointer-stable on the next refresh.
        md2 = self._refresh(bs, bs, seq + 1, replay=False)
        self.assertIs(md2, md)


class DefaultCaptureTest(_RouterCase):
    """The default capture (idle fill + leaf idle refresh) over a real
    router: it binds the same cached per-bs views a refresh binds, seeds
    the same buffers, and is idempotent across the capture-time re-init
    sequence (4 warmups + 2 re-inits per capture)."""

    def _default_capture(self, bs, seq_lens):
        torch = self.torch
        placeholder = {FULL: torch.zeros((bs, 1), dtype=torch.int32)}
        self.router.init_forward_metadata_capture_cuda_graph(
            bs,
            torch.arange(bs, dtype=torch.int64),
            seq_lens,
            ForwardMode.DECODE,
            block_tables=placeholder,
        )
        return self.leaf.forward_decode_metadata

    def test_capture_binds_cached_views_and_seeds_buffers(self):
        torch = self.torch
        seq = torch.full((LADDER_BS,), 7, dtype=torch.int32)
        md = self._default_capture(LADDER_BS, seq)
        self.assertIs(md, self.leaf._decode_views_by_bs[LADDER_BS])
        torch.testing.assert_close(self.leaf.seq_lens_buf[:LADDER_BS], seq)
        # Idle fill: no live tables were read, leaf buffers stay null.
        self.assertTrue((self.leaf.page_table_buf[:LADDER_BS] == 0).all())

    def test_capture_is_idempotent_and_pointer_stable(self):
        from tokenspeed.runtime.execution.graph_ptr_guard import (
            snapshot_graph_metadata,
        )

        torch = self.torch
        seq = torch.full((LADDER_BS,), 7, dtype=torch.int32)
        md = self._default_capture(LADDER_BS, seq)
        snap = snapshot_graph_metadata(self.router)
        for _ in range(5):
            self.assertIs(self._default_capture(LADDER_BS, seq), md)
        self.assertEqual(snapshot_graph_metadata(self.router), snap)

    def test_replay_refresh_rebinds_the_captured_views(self):
        torch = self.torch
        seq = torch.full((LADDER_BS,), 7, dtype=torch.int32)
        md = self._default_capture(LADDER_BS, seq)
        padded_seq = torch.tensor([5, 4, 1, 1], dtype=torch.int32)
        self.assertIs(self._refresh(LADDER_BS, 2, padded_seq, replay=True), md)


class LeafSignatureConformanceTest(_TorchCase):
    """Every paged leaf class carries the uniform leaf contract.

    The router calls every leaf with the same positional/keyword shape; a
    narrower bespoke signature TypeErrors on exactly the configs that need
    it. Runner-facing nodes (router, composites, V4, state) are checked
    against the runner kwarg set separately below.
    """

    _LEAF_CLASSES = (
        ("tokenspeed.runtime.layers.attention.backends.mha", "MHAAttnBackend"),
        ("tokenspeed.runtime.layers.attention.backends.msa", "MSAAttnBackend"),
        ("tokenspeed.runtime.layers.attention.backends.trtllm", "TRTLLMMHAAttnBackend"),
        ("tokenspeed.runtime.layers.attention.backends.mla", "MLAAttnBackend"),
        ("tokenspeed.runtime.layers.attention.backends.trtllm_mla", "TRTLLMMLABackend"),
        (
            "tokenspeed.runtime.layers.attention.backends.tokenspeed_mla",
            "CuteDSLMLABackend",
        ),
        ("tokenspeed.runtime.layers.attention.backends.flashmla", "FlashMLABackend"),
        ("tokenspeed.runtime.layers.attention.backends.dsa", "DSABackend"),
    )

    def _classes(self):
        import importlib

        for module_name, cls_name in self._LEAF_CLASSES:
            try:
                yield getattr(importlib.import_module(module_name), cls_name)
            except (ImportError, ModuleNotFoundError) as exc:
                self.skipTest(f"needs optional deps for {cls_name}: {exc}")

    def test_every_leaf_is_a_paged_attention_backend(self):
        from tokenspeed.runtime.layers.attention.backends.paged import (
            PagedAttentionBackend,
        )

        for cls in self._classes():
            with self.subTest(backend=cls.__name__):
                self.assertTrue(issubclass(cls, PagedAttentionBackend))

    def test_constructor_takes_injected_kernel_page_size(self):
        import inspect

        for cls in self._classes():
            with self.subTest(backend=cls.__name__):
                sig = inspect.signature(cls.__init__)
                param = sig.parameters.get("kernel_page_size")
                self.assertIsNotNone(param, f"{cls.__name__} lacks kernel_page_size")
                self.assertEqual(param.kind, param.KEYWORD_ONLY)

    def test_refresh_signature_binds_the_leaf_call_shape(self):
        import inspect

        for cls in self._classes():
            with self.subTest(backend=cls.__name__):
                sig = inspect.signature(cls.refresh_decode_metadata)
                try:
                    # (self, bs, actual_bs, seq_lens, page_table, ...)
                    sig.bind(
                        None, 8, 8, None, None, num_extends=0, for_graph_replay=True
                    )
                except TypeError as exc:
                    self.fail(f"{cls.__name__}.refresh_decode_metadata: {exc}")

    def test_init_forward_metadata_binds_the_leaf_call_shape(self):
        import inspect

        for cls in self._classes():
            with self.subTest(backend=cls.__name__):
                sig = inspect.signature(cls.init_forward_metadata)
                try:
                    # (self, bs, num_extends, seq_lens, page_table, mode, ...)
                    sig.bind(
                        None,
                        8,
                        8,
                        None,
                        None,
                        None,
                        extend_seq_lens=None,
                        extend_seq_lens_cpu=None,
                        extend_prefix_lens=None,
                        extend_prefix_lens_cpu=None,
                        extend_with_prefix=False,
                    )
                except TypeError as exc:
                    self.fail(f"{cls.__name__}.init_forward_metadata: {exc}")


class RunnerSignatureConformanceTest(_TorchCase):
    """Every runner-facing node accepts the runner's kwarg set."""

    _RUNNER_CLASSES = (
        ("tokenspeed.runtime.layers.attention.backends.router", "CacheGroupRouter"),
        ("tokenspeed.runtime.layers.attention.backends.msa", "MSAHybridAttnBackend"),
        (
            "tokenspeed.runtime.layers.attention.backends.deepseek_v4",
            "DeepseekV4AttentionBackend",
        ),
        (
            "tokenspeed.runtime.layers.attention.backends.mamba",
            "MambaAttnBackend",
        ),
        (
            "tokenspeed.runtime.layers.attention.backends.hybrid",
            "HybridLinearAttnBackend",
        ),
        ("tokenspeed.runtime.layers.attention.backends.kda", "KdaAttnBackend"),
        ("tokenspeed.runtime.layers.attention.backends.inkling", "InklingAttnBackend"),
        (
            "tokenspeed.runtime.layers.attention.backends.qwen4_exp",
            "Qwen4ExpMambaAttnBackend",
        ),
    )

    def test_init_forward_metadata_binds_the_runner_call_shape(self):
        """The runner's extend call: five positionals, then block_tables and
        the five extend fields as required keywords (no defaults anywhere),
        plus the model-side extras a node may ignore."""
        import importlib
        import inspect

        for module_name, cls_name in self._RUNNER_CLASSES:
            try:
                cls = getattr(importlib.import_module(module_name), cls_name)
            except (ImportError, ModuleNotFoundError) as exc:
                self.skipTest(f"needs optional deps for {cls_name}: {exc}")
            sig = inspect.signature(cls.init_forward_metadata)
            with self.subTest(backend=cls_name):
                try:
                    # (self, bs, num_extends, req_pool_indices, seq_lens, mode, ...)
                    sig.bind(
                        None,
                        8,
                        8,
                        None,
                        None,
                        None,
                        block_tables={},
                        extend_seq_lens=None,
                        extend_seq_lens_cpu=None,
                        extend_prefix_lens=None,
                        extend_prefix_lens_cpu=None,
                        extend_with_prefix=False,
                        positions=None,
                        global_num_tokens=None,
                        all_decode_or_idle=False,
                        capture_hidden_mode=None,
                        num_tokens=8,
                    )
                except TypeError as exc:
                    self.fail(f"{cls_name}.init_forward_metadata: {exc}")
                for name in (
                    "extend_seq_lens",
                    "extend_seq_lens_cpu",
                    "extend_prefix_lens",
                    "extend_prefix_lens_cpu",
                    "extend_with_prefix",
                ):
                    param = sig.parameters.get(name)
                    if param is None:
                        # Composites forward ``*args, **kwargs`` to children
                        # that declare the field themselves.
                        self.assertTrue(
                            any(
                                p.kind is inspect.Parameter.VAR_KEYWORD
                                for p in sig.parameters.values()
                            ),
                            f"{cls_name}.init_forward_metadata lacks {name}",
                        )
                        continue
                    self.assertIs(
                        param.default,
                        inspect.Parameter.empty,
                        f"{cls_name}.init_forward_metadata gives {name} a default",
                    )

    def test_capture_signatures_bind_the_runner_kwarg_set(self):
        import importlib
        import inspect

        for module_name, cls_name in self._RUNNER_CLASSES:
            try:
                cls = getattr(importlib.import_module(module_name), cls_name)
            except (ImportError, ModuleNotFoundError) as exc:
                self.skipTest(f"needs optional deps for {cls_name}: {exc}")
            sig = inspect.signature(cls.init_forward_metadata_capture_cuda_graph)
            with self.subTest(backend=cls_name):
                try:
                    # (self, bs, req_pool_indices, seq_lens, forward_mode, ...)
                    sig.bind(None, 8, None, None, None, block_tables={}, num_tokens=8)
                except TypeError as exc:
                    self.fail(
                        f"{cls_name}.init_forward_metadata_capture_cuda_graph "
                        f"rejects a runner kwarg: {exc}"
                    )

    def test_init_cuda_graph_state_signatures_bind_the_runner_kwarg_set(self):
        import importlib
        import inspect

        for module_name, cls_name in self._RUNNER_CLASSES:
            try:
                cls = getattr(importlib.import_module(module_name), cls_name)
            except (ImportError, ModuleNotFoundError) as exc:
                self.skipTest(f"needs optional deps for {cls_name}: {exc}")
            sig = inspect.signature(cls.init_cuda_graph_state)
            with self.subTest(backend=cls_name):
                try:
                    sig.bind(
                        None,
                        8,
                        cache_group_specs=(),
                        cache_group_page_counts=None,
                        max_tokens_per_req=2,
                        overlap_schedule_depth=1,
                    )
                except TypeError as exc:
                    self.fail(
                        f"{cls_name}.init_cuda_graph_state rejects a runner "
                        f"kwarg: {exc}"
                    )


class GraphPtrGuardWalkTest(_TorchCase):
    """Walk semantics of graph_ptr_guard on a stub backend (no kernels)."""

    def setUp(self):
        super().setUp()
        from tokenspeed.runtime.execution import graph_ptr_guard

        self.guard = graph_ptr_guard
        torch = self.torch

        class _StubBackend:
            graph_unstable_metadata_fields = frozenset({"volatile"})

            def __init__(self):
                self.forward_decode_metadata = SimpleNamespace(
                    stable=torch.zeros(4),
                    volatile=torch.zeros(2),
                    nested={"a": torch.ones(3)},
                    scalar=7,
                )

            def child_backends(self):
                return ()

        self.stub_cls = _StubBackend
        self.stub = _StubBackend()

    def test_snapshot_records_tensors_and_skips_unstable_fields(self):
        snap = self.guard.snapshot_graph_metadata(self.stub)
        paths = set(snap)
        self.assertIn("_StubBackend.forward_decode_metadata.stable", paths)
        self.assertIn("_StubBackend.forward_decode_metadata.nested['a']", paths)
        self.assertNotIn("_StubBackend.forward_decode_metadata.volatile", paths)

    def test_unstable_field_may_mutate_between_replays(self):
        snap = self.guard.snapshot_graph_metadata(self.stub)
        self.stub.forward_decode_metadata.volatile = self.torch.zeros(2)
        self.guard.verify_graph_metadata(self.stub, snap, context="test")

    def test_rebound_tensor_is_reported_with_its_path(self):
        snap = self.guard.snapshot_graph_metadata(self.stub)
        self.stub.forward_decode_metadata.stable = self.torch.zeros(4)
        with self.assertRaisesRegex(RuntimeError, r"forward_decode_metadata\.stable"):
            self.guard.verify_graph_metadata(self.stub, snap, context="test")

    def test_child_backends_are_walked(self):
        stub = self.stub

        class _Wrapper:
            def child_backends(self):
                return (stub,)

        snap = self.guard.snapshot_graph_metadata(_Wrapper())
        self.assertIn("_Wrapper._StubBackend.forward_decode_metadata.stable", set(snap))


class RouterPtrGuardTest(_RouterCase):
    """The guard over a real router refresh: the positive arm pins the
    per-bs view cache (leaf metadata + router write-loc views), the
    negative arm is the address-freezing bug class (buffer reallocated,
    views lazily rebuilt over fresh storage)."""

    def setUp(self):
        super().setUp()
        from tokenspeed.runtime.execution.graph_ptr_guard import (
            snapshot_graph_metadata,
            verify_graph_metadata,
        )

        self.snapshot = snapshot_graph_metadata
        self.verify = verify_graph_metadata

    def _padded_seq(self):
        torch = self.torch
        return torch.cat(
            [
                torch.tensor([5, 4], dtype=torch.int32),
                torch.ones(LADDER_BS - 2, dtype=torch.int32),
            ]
        )

    def test_refresh_keeps_captured_identities(self):
        self._refresh(LADDER_BS, 2, self._padded_seq(), replay=True)
        snap = self.snapshot(self.router)
        # The router's write-loc views are part of the walked surface.
        self.assertTrue(
            any("decode_write_locations" in path for path in snap),
            sorted(snap),
        )
        # Fresh lengths, same buffers: replay refresh must not move anything.
        self._refresh(LADDER_BS, 2, self._padded_seq() + 1, replay=True)
        self.verify(self.router, snap, context="test")

    def test_reallocated_buffer_breaches_the_guard(self):
        torch = self.torch
        self._refresh(LADDER_BS, 2, self._padded_seq(), replay=True)
        snap = self.snapshot(self.router)
        # The bug class the guard exists for: a persistent buffer is
        # reallocated and the per-bs views are lazily rebuilt over the new
        # storage — the captured graph keeps reading the old addresses.
        self.leaf.seq_lens_buf = torch.ones(MAX_DECODE_BS, dtype=torch.int32)
        self.leaf._decode_views_by_bs.clear()
        self._refresh(LADDER_BS, 2, self._padded_seq(), replay=True)
        with self.assertRaisesRegex(RuntimeError, r"seq_lens"):
            self.verify(self.router, snap, context="test")


if __name__ == "__main__":
    unittest.main()
