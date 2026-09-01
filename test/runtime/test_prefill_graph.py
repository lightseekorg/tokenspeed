"""Paged KV-cache and prefill CUDA-graph seams.

Prefill-graph replay pads q/k/v rows to the bucket while flat per-group
write locs cover only the real (leading) tokens; the mha KV write must trim
the padded tail or the store kernel walks past the loc array (IAE on the
first padded replay -- reproduced on gpt-oss + flat + default prefill graph).
Capture must also exercise the cache metadata branch via dummy block tables
so capture and replay take the same code path.
"""

from __future__ import annotations

import os
import sys
import unittest
from types import SimpleNamespace

# CI Registration (parsed via AST, runtime no-op)
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from ci_system.ci_register import register_cuda_ci

register_cuda_ci(est_time=10, suite="runtime-1gpu")


def _spec(
    group_id: str,
    *,
    family: str = "history",
    block_granularity: int = 64,
    retention: str = "full_history",
    sliding_window_tokens: int | None = None,
):
    """A published group spec.

    The real dataclass, not a namespace: it derives ``block_granularity`` from
    the geometry the way production does, refuses ``page_size`` on a
    checkpoint-state group, and carries ``retention`` -- which the width rule
    deliberately ignores, and which a namespace double cannot express at all,
    so ``test_sliding_window_group_still_spans_the_whole_extent`` could not be
    written against one.
    """
    from tokenspeed.runtime.layers.attention.kv_cache.recipes.spec import (
        CacheGroupSpec,
    )

    if family == "state":
        return CacheGroupSpec(
            group_id=group_id,
            retention=retention,
            family="state",
            checkpoint_granularity=block_granularity,
            sliding_window_tokens=sliding_window_tokens,
        )
    return CacheGroupSpec(
        group_id=group_id,
        retention=retention,
        family=family,
        rows_per_page=block_granularity,
        entry_stride_tokens=1,
        sliding_window_tokens=sliding_window_tokens,
    )


def _fake_pool(*, specs=(), **arena_attrs) -> SimpleNamespace:
    """A cache-view double: the arena publishes, the view just names it."""
    return SimpleNamespace(
        arena=SimpleNamespace(cache_group_specs=tuple(specs), **arena_attrs)
    )


def _backend(**attrs) -> SimpleNamespace:
    """An attention-backend double whose ``consumes_cache_metadata`` the
    production property computes, so these doubles cannot drift from it."""
    from tokenspeed.runtime.layers.attention.backends.base import AttentionBackend

    attrs.setdefault("uses_cache_groups", False)
    attrs.setdefault("_cache_contract_bound", False)
    ns = SimpleNamespace(**attrs)
    ns.consumes_cache_metadata = AttentionBackend.consumes_cache_metadata.fget(ns)
    return ns


class SliceMhaExtendInputsTest(unittest.TestCase):
    """MHA kernels see exactly the rows covered by live cu-seqlens."""

    def setUp(self):
        try:
            import torch

            from tokenspeed.runtime.layers.attention.backends import mha
        except (ImportError, ModuleNotFoundError) as exc:
            self.skipTest(f"needs torch + tokenspeed_kernel: {exc}")
        self.torch = torch
        self.slice_inputs = mha._slice_extend_inputs

    def test_padded_tail_is_not_passed_to_kernel(self):
        metadata = SimpleNamespace(cu_extend_seq_lens_cpu=[0, 3])
        q = self.torch.zeros(4, 2, 8)
        k = self.torch.zeros(4, 2, 8)
        v = self.torch.zeros(4, 2, 8)

        q, k, v = self.slice_inputs(metadata, q, k, v)

        self.assertEqual((q.shape[0], k.shape[0], v.shape[0]), (3, 3, 3))

    def test_unpadded_inputs_are_unchanged(self):
        metadata = SimpleNamespace(cu_extend_seq_lens_cpu=[0, 4])
        q = self.torch.zeros(4, 2, 8)
        self.assertIs(self.slice_inputs(metadata, q, None, None)[0], q)


class TrimKvToLocsTest(unittest.TestCase):
    """_trim_kv_to_locs slices padded k/v tails to the write-loc count --
    the shared fix point every flat-capable backend's KV write calls.
    Trimming (not loc-padding) keeps the null page 0 all-zero: trtllm does
    not scrub padded tail rows before saving KV."""

    def setUp(self):
        try:
            import torch

            from tokenspeed.runtime.layers.attention.backends.cache_groups import (
                CacheGroupsMixin,
            )
        except (ImportError, ModuleNotFoundError) as exc:
            self.skipTest(f"needs torch + tokenspeed_kernel: {exc}")
        self.torch = torch
        self.trim = CacheGroupsMixin._trim_kv_to_locs

    def test_padded_tail_trimmed(self):
        k = self.torch.zeros(16, 2, 8)
        v = self.torch.zeros(16, 2, 8)
        locs = self.torch.zeros(5, dtype=self.torch.int32)
        k2, v2 = self.trim(locs, k, v)
        self.assertEqual((k2.shape[0], v2.shape[0]), (5, 5))

    def test_equal_rows_identity(self):
        k = self.torch.zeros(16, 2, 8)
        v = self.torch.zeros(16, 2, 8)
        locs = self.torch.zeros(16, dtype=self.torch.int32)
        k2, v2 = self.trim(locs, k, v)
        self.assertIs(k2, k)
        self.assertIs(v2, v)

    def test_none_kv_passthrough(self):
        locs = self.torch.zeros(4, dtype=self.torch.int32)
        self.assertEqual(self.trim(locs, None, None), (None, None))


class _StopAfterSpy(Exception):
    """Raised by the spy to stop make_dummy_batch at the seam under test."""


class DummyGroupTablesTest(unittest.TestCase):
    """Capture-time dummy tables: every group gets a real, writable block;
    none get the reserved null block 0."""

    def setUp(self):
        try:
            import torch  # noqa: F401

            from tokenspeed.runtime.execution.prefill_graph import PrefillGraph
        except (ImportError, ModuleNotFoundError) as exc:
            self.skipTest(f"needs torch + runtime deps: {exc}")
        self.PrefillGraph = PrefillGraph

    def _bare(self, backend, pool):
        pg = self.PrefillGraph.__new__(self.PrefillGraph)
        pg.attn_backend = backend
        pg.token_to_kv_pool = pool
        pg.config = SimpleNamespace(
            device="cpu",
            physical_context_len=1000,
            spec_num_tokens=None,
            overlap_schedule_depth=0,
        )
        return pg

    def test_every_group_gets_a_real_block(self):
        backend = _backend(
            uses_cache_groups=True,
        )
        pool = _fake_pool(
            specs=(
                _spec("full_attention", block_granularity=64),
                _spec("sliding_attention", block_granularity=64),
                # state: included
                _spec("linear_attention", family="state", block_granularity=128),
            )
        )
        tables = self._bare(backend, pool)._dummy_group_tables(1)
        self.assertEqual(
            set(tables),
            {"full_attention", "sliding_attention", "linear_attention"},
        )
        # Each group in its own grain, rounded UP: 1000 is a multiple of
        # neither, so a floor would give 15 and 7 and fail here.
        self.assertEqual(tables["full_attention"].shape, (1, 16))
        self.assertEqual(tables["linear_attention"].shape, (1, 8))
        for group_id, table in tables.items():
            self.assertGreater(
                int(table.min()),
                0,
                f"{group_id}: capture writes KV, so no group may get the "
                "reserved null block",
            )

    def test_width_spans_the_extent_at_a_realistic_magnitude(self):
        """Width is ceil(physical extent / grain), at a size where a silent
        cap would hide -- every other fixture here is a few hundred columns."""
        bare = self._bare(
            _backend(uses_cache_groups=True),
            _fake_pool(specs=(_spec("full_attention", block_granularity=64),)),
        )
        bare.config.physical_context_len = 262144
        self.assertEqual(bare._dummy_group_tables(1)["full_attention"].shape[1], 4096)

    def test_sliding_window_group_still_spans_the_whole_extent(self):
        """A sliding group must NOT be narrowed to its window.

        The decode capture helper bounds a sliding row by the window, because
        a decode row describes live cache history. Capture fabricates one
        extend over the whole extent and derives a write column for every
        position in it, so the window bound underflows the table. Measured on
        Inkling: a window-sized ``sliding_attention_0`` row got 6 columns and
        the extend needed 63 -- "extend write locations out of table bounds",
        boot dead. This asserts the width that survives.
        """
        pool = _fake_pool(
            specs=(
                _spec("full_attention", block_granularity=128),
                _spec(
                    "sliding_attention",
                    block_granularity=128,
                    retention="sliding_window",
                    sliding_window_tokens=128,
                ),
            )
        )
        bare = self._bare(_backend(uses_cache_groups=True), pool)
        bare.config.physical_context_len = 8192

        tables = bare._dummy_group_tables(1)
        self.assertEqual(tables["full_attention"].shape[1], 64)
        self.assertEqual(
            tables["sliding_attention"].shape[1],
            64,
            "retention must not narrow a capture row; the window bound is a "
            "decode-side answer to a different question",
        )
        # The bound the runtime actually enforces, restated here so the number
        # above is tied to it: ceil-1 of the largest position in the extend.
        self.assertLess((8192 - 1) // 128, tables["sliding_attention"].shape[1])

    def test_width_follows_each_groups_own_geometry(self):
        # DeepSeek-V4 shape: sibling groups with very different grains. Each
        # gets ceil(extent / its own granularity) -- one rule, no width flag.
        backend = _backend(uses_cache_groups=True)
        pool = _fake_pool(
            specs=(
                _spec("fine", block_granularity=4),
                _spec("coarse", block_granularity=256),
            )
        )

        # 1000 % 256 != 0, so a floor would give 3 for the coarse group.
        tables = self._bare(backend, pool)._dummy_group_tables(1)

        self.assertEqual(tables["fine"].shape, (1, 250))  # ceil(1000/4)
        self.assertEqual(tables["coarse"].shape, (1, 4))  # ceil(1000/256)
        self.assertGreater(int(tables["fine"].min()), 0)
        self.assertGreater(int(tables["coarse"].min()), 0)

    def test_hybrid_wrapper_needs_no_child_descent(self):
        # A hybrid wrapper carries no kernel geometry of its own; the one
        # width rule needs none, so the wrapper alone must produce tables
        # for every group, state included.
        wrapper = _backend(uses_cache_groups=True)
        pool = _fake_pool(
            specs=(
                _spec("full_attention", block_granularity=128),
                _spec("linear_attention", family="state", block_granularity=128),
            )
        )
        tables = self._bare(wrapper, pool)._dummy_group_tables(1)
        self.assertEqual(set(tables), {"full_attention", "linear_attention"})
        self.assertEqual(tables["full_attention"].shape, (1, 8))  # ceil(1000/128)

    def test_each_capture_row_gets_its_own_block(self):
        """A state group needs one working block per request: two rows sharing
        one silently clobber each other. The runtime check is gated on
        TOKENSPEED_CACHE_DEBUG, so a regression would be silent and this test
        is the guard. Reachable at bs>1, which ``_autotune`` produces whenever
        the chunk budget exceeds the model context -- and ``_autotune`` runs
        even with the prefill graph disabled."""
        import torch

        from tokenspeed.runtime.layers.attention.backends.hybrid_linear_attn import (
            compute_state_block_indices,
        )

        pool = _fake_pool(
            specs=(
                _spec("full_attention", block_granularity=64),
                _spec("linear_attention", family="state", block_granularity=128),
            )
        )
        bare = self._bare(_backend(uses_cache_groups=True), pool)
        bare.config.physical_context_len = 1024
        tables = bare._dummy_group_tables(3)
        for group_id, table in tables.items():
            self.assertEqual(table.shape[0], 3, group_id)
            self.assertGreater(int(table.min()), 0, group_id)
            # The state path gathers at (seq_len - 1) // grain -- the LAST live
            # column, never column 0. Asserting only on column 0 admits a table
            # that aliases everywhere the runtime actually reads.
            last_column = table[:, -1].tolist()
            self.assertEqual(
                len(set(last_column)),
                3,
                f"{group_id}: rows alias at the column the state path reads, "
                f"got {last_column}",
            )
        # Strongest form: hand the shipped table to the production helper and
        # let its own uniqueness rule be the assertion.
        compute_state_block_indices(
            tables["linear_attention"],
            128,
            torch.zeros(3, dtype=torch.int32),
            torch.full((3,), 1024, dtype=torch.int32),
            validate=True,
        )

    def test_expanded_row_reaches_the_kernels_full_width(self):
        """The width is stated in block granularity, but a stride-deriving
        kernel (trtllm) indexes the whole row. The safety step is the mapping
        point: ``_kernel_page_tables`` re-expands to ``max_num_pages``. Pin
        that, or the contract the deleted width flag protected has no test."""
        from tokenspeed.runtime.layers.attention.backends.cache_groups import (
            CacheGroupsMixin,
        )

        spec = _spec("full_attention", block_granularity=128)
        bare = self._bare(_backend(uses_cache_groups=True), _fake_pool(specs=(spec,)))
        bare.config.physical_context_len = 8192
        tables = bare._dummy_group_tables(1)

        host = CacheGroupsMixin.__new__(CacheGroupsMixin)
        host.kernel_page_size = 64
        host.max_num_pages = -(-8192 // 64)
        host.group_block_granularities = {"full_attention": 128}
        expanded = CacheGroupsMixin._kernel_page_tables(host, dict(tables))

        self.assertEqual(
            expanded["full_attention"].shape[1],
            host.max_num_pages,
            "the expanded row must span the width the kernel derives from "
            "max_kv_len",
        )
        self.assertGreater(int(expanded["full_attention"].min()), 0)

        # Padded max_num_pages: TRTLLM-MLA rounds its width up to a block
        # constraint, so the expansion tail is zero-filled past the live
        # range. The contract is "no null block INSIDE the live range"; a
        # blanket min() > 0 passes above only because 8192/128*2 lands exactly
        # on 128, an arithmetic accident this case removes.
        host.max_num_pages = 130
        padded = CacheGroupsMixin._kernel_page_tables(host, dict(tables))
        live = -(-8192 // host.kernel_page_size)
        self.assertEqual(padded["full_attention"].shape[1], 130)
        self.assertGreater(
            int(padded["full_attention"][:, :live].min()),
            0,
            "the live prefix must never contain the reserved null block",
        )

    def _dummy_batch_probe(
        self,
        *,
        num_tokens,
        context_len,
        physical,
        specs,
        needs_group_block_tables=False,
        consumes_cache_metadata=True,
        arena_blocks=64,
    ):
        """Drive make_dummy_batch to the backend hand-off and record it.

        Stops at ``init_forward_metadata`` -- one statement past everything
        the capture path builds -- so the real ``CacheBatchMetadata`` and the
        real ``block_tables_from_forward_op`` run on the way. Those enforce
        int32, row count against the batch, non-zero width, contract group
        order, and every entry inside ``group_page_counts - 1``; a capture
        they refuse kills the boot, so they are the assertion.
        """
        from unittest import mock

        import torch

        from tokenspeed.runtime.layers.attention.kv_cache.recipes.cache_runtime import (
            CacheRuntimeContract,
        )

        # The contract requires group_page_counts == num_lcm_blocks * packing
        # + 1; the +1 is the reserved null block every table can point at.
        num_lcm_blocks = arena_blocks
        contract = CacheRuntimeContract(
            prefix_granularity=64,
            num_lcm_blocks=num_lcm_blocks,
            token_capacity=num_lcm_blocks * 64,
            group_specs=tuple(specs),
            group_page_counts={str(sp.group_id): num_lcm_blocks + 1 for sp in specs},
            group_packing={str(sp.group_id): 1 for sp in specs},
        )
        pg = self.PrefillGraph.__new__(self.PrefillGraph)
        pg.attn_backend = _backend(
            uses_cache_groups=consumes_cache_metadata,
            needs_group_block_tables=needs_group_block_tables,
        )
        pg.token_to_kv_pool = _fake_pool(specs=tuple(specs), runtime_contract=contract)
        pg.config = SimpleNamespace(
            device="cpu",
            context_len=context_len,
            physical_context_len=physical,
            world_size=1,
        )
        pg.dp_size = 1
        pg.drafter = None
        buf = lambda n, dt: torch.zeros(n, dtype=dt)  # noqa: E731
        pg.input_buffers = SimpleNamespace(
            dummy_kv_slot=0,
            input_ids_buf=buf(4096, torch.int32),
            out_cache_loc_buf=buf(4096, torch.int32),
            positions_buf=buf(4096, torch.int64),
            req_pool_indices_buf=buf(16, torch.int32),
            seq_lens_buf=buf(16, torch.int32),
            extend_seq_lens_buf=buf(16, torch.int32),
            extend_seq_lens_cpu=buf(16, torch.int32),
            extend_prefix_lens_buf=buf(16, torch.int32),
            extend_prefix_lens_cpu=buf(16, torch.int32),
        )
        pg.page_table = torch.zeros(16, 64, dtype=torch.int32)

        seen = {}

        def _record(**kwargs):
            seen.update(kwargs)
            # The row-constant table is legal only for a prefix-free extend:
            # with history the state gather resolves in == out and refuses.
            seen["max_prefix"] = int(pg.input_buffers.extend_prefix_lens_cpu.max())
            raise _StopAfterSpy

        pg.attn_backend.init_forward_metadata = _record
        with mock.patch(
            "tokenspeed.runtime.execution.prefill_graph.ForwardContext",
            lambda **kw: SimpleNamespace(**kw),
        ):
            with self.assertRaises(_StopAfterSpy):
                pg.make_dummy_batch(num_tokens)
        return seen

    def test_make_dummy_batch_tables_survive_the_cache_contract(self):
        """The real validator sees the tables, and rows track the batch."""
        spec = _spec("full_attention", block_granularity=64)
        # 2048 tokens over a 960 context is three fabricated requests, so a
        # rule that collapsed rows to one would be visible here.
        seen = self._dummy_batch_probe(
            num_tokens=2048, context_len=960, physical=1024, specs=(spec,)
        )
        tables = seen["block_tables"]
        table = tables["full_attention"]
        self.assertEqual(table.shape[0], 3, "one row per fabricated request")
        # physical (1024), not the user-facing context_len (960): 16 vs 15.
        self.assertEqual(table.shape[1], 16)
        self.assertGreater(int(table.min()), 0)
        self.assertIsNotNone(seen.get("cache_metadata"))
        # The tables are built on the host; the packer's output is what the
        # backend gets, and it must land on the configured device. Nothing
        # else re-places them, so this assertion is the whole device contract.
        self.assertEqual(table.device.type, "cpu")
        self.assertEqual(seen["max_prefix"], 0, "capture fabricates no prefix")

    def test_group_table_backend_gets_positions_alongside_its_tables(self):
        """A backend that asks for group tables is told how many tokens the
        batch carries and handed the live positions slice, and it still gets
        the metadata-derived tables. The decode wrapper is not consulted: it
        used to supply a set here that the metadata tables always replaced."""
        spec = _spec("full_attention", block_granularity=64)
        seen = self._dummy_batch_probe(
            num_tokens=128,
            context_len=960,
            physical=1024,
            specs=(spec,),
            needs_group_block_tables=True,
        )
        self.assertEqual(seen["num_tokens"], 128)
        self.assertEqual(seen["positions"].shape[0], 128)
        self.assertIn("full_attention", seen["block_tables"])

    def test_block_ids_are_checked_against_the_groups_real_block_count(self):
        """The probe's default arena has ~20x slack, so an id error would not
        reach the packer. Shrink it until the bound is tight and confirm the
        packer -- not this test -- is what rejects an out-of-range id."""
        spec = _spec("full_attention", block_granularity=64)
        # 2 blocks + the reserved null one: bs=2 fits, bs=3 does not.
        self._dummy_batch_probe(
            num_tokens=1920,
            context_len=960,
            physical=1024,
            specs=(spec,),
            arena_blocks=2,
        )
        with self.assertRaises(ValueError) as caught:
            self._dummy_batch_probe(
                num_tokens=2880,
                context_len=960,
                physical=1024,
                specs=(spec,),
                arena_blocks=2,
            )
        self.assertIn("page ID outside", str(caught.exception))

    def test_ceiling_is_exact_at_the_residue_that_distinguishes_it(self):
        """Pin the rounding at the only residue where it shows.

        Every other fixture here uses an extent where ceil(N/g) == ceil((N-1)/g),
        so an off-by-one in the extent is invisible. At physical % grain == 1 the
        two differ, and one column short is the "extend write locations out of
        table bounds" dead boot.
        """
        spec = _spec("full_attention", block_granularity=64)
        bare = self._bare(_backend(uses_cache_groups=True), _fake_pool(specs=(spec,)))
        bare.config.physical_context_len = 4097

        cols = bare._dummy_group_tables(1)["full_attention"].shape[1]
        self.assertEqual(cols, 65)  # ceil(4097/64); extent-1 would give 64
        # Restate the bound the runtime enforces so the constant is tied to it.
        self.assertGreater(cols, (4097 - 1) // 64)

    def test_degenerate_extent_still_yields_one_column(self):
        """A non-positive extent is not reachable through ServerArgs, but the
        clamp is what keeps a zero-width table -- which the contract packer
        rejects outright -- from being the failure mode."""
        spec = _spec("full_attention", block_granularity=64)
        bare = self._bare(_backend(uses_cache_groups=True), _fake_pool(specs=(spec,)))
        for extent in (0, -8):
            bare.config.physical_context_len = extent
            self.assertEqual(
                bare._dummy_group_tables(1)["full_attention"].shape[1], 1, extent
            )

    def test_group_tables_are_required_when_the_backend_asks_for_them(self):
        """``needs_group_block_tables`` without published metadata must fail
        loudly. A bare assert would vanish under -O, which is exactly when a
        silent empty dict would be worst."""
        spec = _spec("full_attention", block_granularity=64)
        with self.assertRaises(RuntimeError) as caught:
            self._dummy_batch_probe(
                num_tokens=128,
                context_len=960,
                physical=1024,
                specs=(spec,),
                needs_group_block_tables=True,
                consumes_cache_metadata=False,
            )
        self.assertIn("needs_group_block_tables", str(caught.exception))

    def test_backend_without_cache_groups_is_empty(self):
        backend = _backend(uses_cache_groups=False)
        pool = _fake_pool(specs=())
        self.assertEqual(self._bare(backend, pool)._dummy_group_tables(1), {})

    def test_unbound_contract_is_still_empty(self):
        """An MLA backend the registry never bound keeps the legacy page_table
        path and must not be handed capture metadata."""
        backend = _backend(uses_cache_groups=False, _cache_contract_bound=False)
        pool = _fake_pool(specs=(_spec("full_attention", block_granularity=64),))
        self.assertEqual(self._bare(backend, pool)._dummy_group_tables(1), {})

    def test_contract_bound_mla_target_gets_tables(self):
        """A plain MLA target leaves uses_cache_groups False by design, so
        gating capture on it handed the backend a dummy batch with no metadata,
        which it refuses -- Kimi-K2.5 ran every eval on eager prefill."""
        backend = _backend(
            uses_cache_groups=False,
            _cache_contract_bound=True,
        )
        pool = _fake_pool(specs=(_spec("full_attention", block_granularity=128),))
        tables = self._bare(backend, pool)._dummy_group_tables(2)
        self.assertEqual(set(tables), {"full_attention"})
        # Scheduler-table columns span block_granularity: ceil(1000/128).
        self.assertEqual(tables["full_attention"].shape, (2, 8))
        self.assertGreater(
            int(tables["full_attention"].min()),
            0,
            "MLA rejects the null block in live metadata, so capture needs a "
            "real writable block",
        )

    def test_delegating_wrapper_answers_for_its_child(self):
        """A wrapper that forwards mark_cache_contract must forward the capture
        predicates too, or capture hands its bound child a batch the child
        declares invalid. Asserting the values, not their presence: a property
        that returns the wrong answer is the failure being guarded against."""
        from tokenspeed.runtime.layers.attention.backends.dsa import DSABackend

        wrapper = DSABackend.__new__(DSABackend)
        for answer in (True, False):
            wrapper._dense_backend = SimpleNamespace(consumes_cache_metadata=answer)
            self.assertIs(wrapper.consumes_cache_metadata, answer)

    def test_inkling_wrapper_answers_for_its_inner(self):
        """Inkling answers the capture gate for its inner backend. The
        mirror is explicit rather than inherited so that a class-level default
        on the base cannot shadow it -- but note nothing outside this wrapper
        reads its geometry, so only the gate is pinned here."""
        from tokenspeed.runtime.layers.attention.backends.inkling import (
            InklingAttnBackend,
        )

        wrapper = InklingAttnBackend.__new__(InklingAttnBackend)
        for answer in (True, False):
            wrapper.inner = SimpleNamespace(
                consumes_cache_metadata=answer,
                uses_cache_groups=answer,
            )
            self.assertIs(wrapper.consumes_cache_metadata, answer)

    def test_runtime_contract_pool_is_eligible_for_capture(self):
        from unittest import mock

        inner_model = SimpleNamespace(embed_tokens=object())
        model_runner = SimpleNamespace(
            model=SimpleNamespace(model=inner_model),
            is_generation=True,
            is_multimodal=False,
        )
        config = SimpleNamespace(
            enforce_eager=False,
            disable_prefill_graph=False,
            data_parallel_size=1,
        )
        pool = _fake_pool(runtime_contract=object())
        with (
            mock.patch(
                "tokenspeed.runtime.execution.prefill_graph.get_prefill_token_buckets",
                return_value=[64],
            ),
            mock.patch.object(self.PrefillGraph, "capture") as capture,
        ):
            graph = self.PrefillGraph(
                model_runner=model_runner,
                attn_backend=object(),
                token_to_kv_pool=pool,
                input_buffers=object(),
                config=config,
                page_table=object(),
            )

        self.assertFalse(graph.disable)
        capture.assert_not_called()


class CaptureFailureIsLoudTest(unittest.TestCase):
    """A capture the dummy-batch machinery cannot serve must stop the boot.

    Degrading here is what let a whole model family run eager prefill with a
    warning nobody read: the warning was indistinguishable from the families
    that are deliberately eager.
    """

    def setUp(self):
        try:
            import torch

            from tokenspeed.runtime.execution.prefill_graph import PrefillGraph
        except (ImportError, ModuleNotFoundError) as exc:
            self.skipTest(f"needs torch + runtime deps: {exc}")
        self.torch = torch
        self.PrefillGraph = PrefillGraph

    def _bare(self, raises=None):
        pg = self.PrefillGraph.__new__(self.PrefillGraph)
        pg.disable = False
        pg.capture_buckets = [4]
        pg.attn_backend = SimpleNamespace()
        pg.config = SimpleNamespace(device="cpu", world_group=None, world_size=1)
        pg._embed_tokens = SimpleNamespace(
            weight=self.torch.zeros(2, 8, dtype=self.torch.float32)
        )

        def _capture_all_buckets(_decode_wrapper):
            if raises is not None:
                raise raises

        pg._capture_all_buckets = _capture_all_buckets
        return pg

    def test_capture_failure_propagates_untouched(self):
        """Nothing between the backend and the operator: same exception object,
        same traceback. ``capture`` has no handler at all, so a partial ladder
        cannot be left behind -- the boot dies with it."""
        cause = RuntimeError("backend refused the dummy batch")
        pg = self._bare(raises=cause)
        with self.assertRaises(RuntimeError) as caught:
            pg.capture(None)
        self.assertIs(caught.exception, cause)

    def test_successful_capture_does_not_raise(self):
        self._bare().capture(None)

    def test_oom_propagates(self):
        """OOM keeps its own type and message. The capture pool not fitting is
        an operator-visible sizing failure, not something to recover from."""
        pg = self._bare(raises=self.torch.cuda.OutOfMemoryError("no room"))
        with self.assertRaises(self.torch.cuda.OutOfMemoryError):
            pg.capture(None)


class PrefillGraphOutputStorageTest(unittest.TestCase):
    def setUp(self):
        try:
            import torch

            from tokenspeed.runtime.execution.prefill_graph import (
                CapturedForward,
                PrefillGraph,
            )
        except (ImportError, ModuleNotFoundError) as exc:
            self.skipTest(f"needs torch + runtime deps: {exc}")
        self.torch = torch
        self.CapturedForward = CapturedForward
        self.graph = PrefillGraph.__new__(PrefillGraph)
        self.graph.capture_buckets = [2, 4]
        self.graph._hidden_output_buf = None
        self.graph._aux_output_bufs = None
        self.graph._outputs = {}

    def _store(self, bucket, value):
        hidden = self.torch.full((bucket, 3), value)
        aux = [self.torch.full((bucket, 3), value + 10)]
        outputs = self.CapturedForward(hidden, aux)
        if self.graph._hidden_output_buf is None:
            self.graph._allocate_output_buffers(outputs)
        self.graph._outputs[bucket] = self.graph._store_outputs(bucket, outputs)
        return self.graph._outputs[bucket]

    def test_buckets_retain_views_of_the_same_max_bucket_storage(self):
        largest = self._store(4, 1)
        smaller = self._store(2, 2)

        self.assertEqual(
            largest.hidden_states.untyped_storage().data_ptr(),
            smaller.hidden_states.untyped_storage().data_ptr(),
        )
        self.assertEqual(
            largest.aux_hidden_states[0].untyped_storage().data_ptr(),
            smaller.aux_hidden_states[0].untyped_storage().data_ptr(),
        )

    def test_null_capture_mode_keeps_padding_out_of_the_slice(self):
        """No aux taps: hidden still lands in the shared buffer and slices."""
        hidden = self.torch.tensor([[3, 3, 3], [3, 3, 3], [99, 99, 99], [99, 99, 99]])
        outputs = self.CapturedForward(hidden, None)
        self.graph._allocate_output_buffers(outputs)
        captured = self.graph._store_outputs(4, outputs)
        rows, aux = captured.sliced(2)

        self.assertIsNone(aux)
        self.assertEqual(rows.shape, (2, 3))
        self.assertTrue(self.torch.equal(rows, self.torch.full((2, 3), 3)))


class TrtllmPrefillGraphSeamsTest(unittest.TestCase):
    """trtllm under the prefill graph: the extend prewrite must not bake
    capture-time write locs into the graph, and the break's KV write must
    trim padded tails like mha."""

    def setUp(self):
        try:
            import torch

            from tokenspeed.runtime.layers.attention.backends import trtllm
        except (ImportError, ModuleNotFoundError) as exc:
            self.skipTest(f"needs torch + tokenspeed_kernel: {exc}")
        self.torch = torch
        self.mod = trtllm

    def _bare_backend(self):
        b = self.mod.TRTLLMMHAAttnBackend.__new__(self.mod.TRTLLMMHAAttnBackend)
        b.kv_cache_dtype = self.torch.bfloat16
        return b

    def test_prewrite_disabled_during_breakable_capture(self):
        from unittest import mock

        b = self._bare_backend()
        self.assertTrue(b.support_kv_cache_prewrite(None))
        with mock.patch.object(
            self.mod, "is_breakable_capture_active", return_value=True
        ):
            self.assertFalse(b.support_kv_cache_prewrite(None))

    def test_declares_history_contract_family(self):
        self.assertEqual(
            self.mod.TRTLLMMHAAttnBackend.cache_consumer_families,
            frozenset({"history"}),
        )


if __name__ == "__main__":
    unittest.main()
