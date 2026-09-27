"""Paged KV-cache and prefill CUDA-graph seams.

Prefill-graph replay pads q/k/v rows to the bucket while flat per-group
write locs cover only the real (leading) tokens; the mha KV write must trim
the padded tail or the store kernel walks past the loc array (IAE on the
first padded replay -- reproduced on gpt-oss + flat + default prefill graph).
Capture must also exercise the cache metadata branch via dummy block tables
so capture and replay take the same code path.
"""

from __future__ import annotations

import argparse
import contextlib
import io
import os
import sys
import unittest
from types import SimpleNamespace

# CI Registration (parsed via AST, runtime no-op)
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from ci_system.ci_register import register_cuda_ci

from tokenspeed.runtime.execution.memory_delta import NULL_MEMORY_DELTA_OBSERVER

register_cuda_ci(est_time=10, suite="runtime-1gpu")


class PrefillCaptureArgsTest(unittest.TestCase):
    def setUp(self):
        from tokenspeed.runtime.utils.server_args import ServerArgs

        self.parser = argparse.ArgumentParser()
        ServerArgs.add_cli_args(self.parser)

    def test_token_aliases_share_config_and_capture_selection(self):
        from tokenspeed.cli._argsplit import split_argv
        from tokenspeed.runtime.execution.prefill_graph import (
            PrefillGraph,
            get_prefill_token_buckets,
            resolve_prefill_capture_batch_sizes,
        )

        configurations = []
        for flag in (
            "--prefill-graph-capture-token-sizes",
            "--prefill-graph-capture-sizes",
        ):
            argv = [
                "--model",
                "test",
                flag,
                "1024",
                "2048",
                "4096",
                "--prefill-graph-capture-batch-sizes",
                "2",
                "1",
                "2",
            ]
            direct = self.parser.parse_args(argv)
            routed = self.parser.parse_args(split_argv(argv).engine)
            self.assertEqual(vars(direct), vars(routed))
            self.assertFalse(hasattr(direct, "prefill_graph_capture_token_sizes"))
            self.assertEqual(direct.prefill_graph_capture_sizes, [1024, 2048, 4096])
            config = SimpleNamespace(**vars(direct))
            config.prefill_graph_max_tokens = config.chunked_prefill_size = 4096
            config.context_len = 4096
            config.max_num_seqs = 8
            config.data_parallel_size = 1
            buckets = get_prefill_token_buckets(config)
            combinations = [
                (bucket, bs)
                for bucket in buckets
                for bs in resolve_prefill_capture_batch_sizes(config, bucket)
            ]
            self.assertEqual(
                combinations,
                [
                    (1024, 1),
                    (1024, 2),
                    (2048, 1),
                    (2048, 2),
                    (4096, 1),
                    (4096, 2),
                ],
            )
            owner = PrefillGraph.__new__(PrefillGraph)
            owner.capture_buckets = buckets
            self.assertEqual((owner._padded_bucket(868 + 869), 2), (2048, 2))
            self.assertIsNone(owner._padded_bucket(4097))
            configurations.append((vars(direct), combinations))
        self.assertEqual(*configurations)

    def test_token_aliases_are_mutually_exclusive(self):
        from tokenspeed.cli._argsplit import split_argv

        flags = ("--prefill-graph-capture-token-sizes", "--prefill-graph-capture-sizes")
        for first, second in (flags, flags[::-1]):
            for value in ("1024", "2048"):
                for inline_value in (False, True):
                    args = (
                        [first + "=1024", second + "=" + value]
                        if inline_value
                        else [first, "1024", second, value]
                    )
                    argv = ["--model", "test", *args]
                    for routed in (argv, split_argv(argv).engine):
                        with self.subTest(argv=routed):
                            with contextlib.redirect_stderr(io.StringIO()) as error:
                                with self.assertRaises(SystemExit) as raised:
                                    self.parser.parse_args(routed)
                            self.assertEqual(raised.exception.code, 2)
                            self.assertIn("not allowed with argument", error.getvalue())
                            self.assertIn(first, error.getvalue())
                            self.assertIn(second, error.getvalue())

    def test_defaults_and_help_keep_token_and_request_units_separate(self):
        args = self.parser.parse_args(["--model", "test"])
        self.assertIsNone(args.prefill_graph_capture_sizes)
        self.assertIsNone(args.prefill_graph_capture_batch_sizes)
        help_text = " ".join(self.parser.format_help().split())
        self.assertIn("Total input-token capacities per forward", help_text)
        self.assertIn("not per-request sequence lengths", help_text)
        self.assertIn(
            "Request capacities for inline prefill attention capture", help_text
        )
        self.assertIn("smallest fitting captured batch size", help_text)
        self.assertIn("Compatibility alias", help_text)

    def test_executor_requires_explicit_capture_batch_sizes(self):
        from tokenspeed.runtime.execution.model_executor import ModelExecutorConfig
        from tokenspeed.runtime.execution.prefill_graph import (
            resolve_prefill_capture_batch_sizes,
        )

        config_args = dict(
            max_req_pool_size=5,
            output_length=1,
            enforce_eager=False,
            prefix_granularity=128,
            max_num_seqs=4,
            chunked_prefill_size=4096,
            vocab_size=32,
            context_len=4096,
            physical_context_len=4096,
            device="cpu",
            gpu_id=0,
            global_rank=0,
            cudagraph_capture_sizes=[1, 2, 4],
            disable_cuda_graph_padding=False,
            max_cudagraph_capture_size=4,
            model_is_mrope=False,
            prefill_only=False,
        )
        with self.assertRaisesRegex(TypeError, "prefill_graph_capture_batch_sizes"):
            ModelExecutorConfig(**config_args)

        for sizes, expected in ((None, [1]), ([1, 2, 4], [1, 2, 4])):
            with self.subTest(capture_batch_sizes=sizes):
                config = ModelExecutorConfig(
                    **config_args, prefill_graph_capture_batch_sizes=sizes
                )
                self.assertIs(config.prefill_graph_capture_batch_sizes, sizes)
                self.assertEqual(
                    resolve_prefill_capture_batch_sizes(config, 1024), expected
                )


class KdaPrefillFallbackTest(unittest.TestCase):
    def test_outer_attention_break_does_not_capture_kda_graphs(self):
        from unittest.mock import patch

        import torch

        from tokenspeed.runtime.execution.breakable_cuda_graph import BreakableCapture
        from tokenspeed.runtime.execution.forward_batch_info import ForwardMode
        from tokenspeed.runtime.layers.attention.backends.state.kda import (
            KdaAttnBackend,
        )
        from tokenspeed.runtime.layers.attention.backends.state.mamba import (
            MambaAttnBackend,
        )

        backend = object.__new__(KdaAttnBackend)
        backend._prefill_graph_enabled = True
        backend.kda_backend = "cutedsl_kda"
        capture = object.__new__(BreakableCapture)
        output = torch.ones(1)
        results = []

        # Exercise the real replay boundary with CPU tensors and a mocked scan.
        # Even repeated fallback shapes must not warm or capture a private graph.
        with (
            patch.object(MambaAttnBackend, "forward_extend", autospec=True) as scan,
            patch.object(torch.cuda, "is_current_stream_capturing", return_value=False),
            patch.object(torch.cuda, "CUDAGraph") as graph,
            patch.object(
                torch.cuda,
                "current_stream",
                side_effect=AssertionError("fallback attempted graph preparation"),
            ),
        ):
            scan.return_value = output
            for bs, bucket in ((1, 128), (2, 2048), (4, 4096), (1, 8192)):
                for checkpoint in (None, object()):
                    live = SimpleNamespace(prefill_checkpoint_batch=checkpoint)
                    backend.forward_metadata = live

                    def forward():
                        results.append(
                            backend.forward_extend(
                                None,
                                None,
                                None,
                                None,
                                None,
                                bs,
                                ForwardMode.EXTEND,
                                save_kv_cache=True,
                                layer_id=0,
                                seq_len=bucket,
                            )
                        )

                    capture.segments = [forward]
                    for iteration in range(3):
                        with self.subTest(bs=bs, bucket=bucket, iteration=iteration):
                            scan.reset_mock()
                            capture.replay(valid_rows=bucket - 1)
                            scan.assert_called_once_with(
                                backend,
                                None,
                                None,
                                None,
                                None,
                                None,
                                bs,
                                ForwardMode.EXTEND,
                                save_kv_cache=True,
                                layer_id=0,
                                seq_len=bucket,
                            )
                            self.assertIs(results[-1], output)
                            self.assertIs(backend.forward_metadata, live)
                            self.assertFalse(backend.prefill_metadata_is_capture_ready)
            graph.assert_not_called()
        self.assertEqual(len(results), 24)


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
            replayable=False,
        )
    return CacheGroupSpec(
        group_id=group_id,
        retention=retention,
        family=family,
        rows_per_page=block_granularity,
        entry_stride_tokens=1,
        sliding_window_tokens=sliding_window_tokens,
        replayable=False,
    )


def _fake_pool(*, specs=(), **arena_attrs) -> SimpleNamespace:
    """A cache-view double: the arena publishes, the view just names it."""
    return SimpleNamespace(
        arena=SimpleNamespace(cache_group_specs=tuple(specs), **arena_attrs)
    )


def _backend(**attrs) -> SimpleNamespace:
    """An attention-backend double. Interface attributes the production code
    reads directly (no getattr probes) must be declared explicitly."""
    return SimpleNamespace(**attrs)


class SliceMhaExtendInputsTest(unittest.TestCase):
    """MHA kernels see exactly the rows covered by live cu-seqlens."""

    def setUp(self):
        try:
            import torch

            from tokenspeed.runtime.layers.attention.backends.paged import mha
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
    """mha.trim_kv_to_locs slices padded k/v tails to the write-loc count --
    the shared fix point every leaf's KV write calls (mha, msa, trtllm).
    Trimming (not loc-padding) keeps the null page 0 all-zero: trtllm does
    not scrub padded tail rows before saving KV."""

    def setUp(self):
        try:
            import torch

            from tokenspeed.runtime.layers.attention.backends.paged.mha import (
                trim_kv_to_locs,
            )
        except (ImportError, ModuleNotFoundError) as exc:
            self.skipTest(f"needs torch + tokenspeed_kernel: {exc}")
        self.torch = torch
        self.trim = trim_kv_to_locs

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
        backend = _backend()
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
            _backend(),
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
        bare = self._bare(_backend(), pool)
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
        backend = _backend()
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
        wrapper = _backend()
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
        is the guard. Reachable at bs>1, which ``autotune`` produces whenever
        the chunk budget exceeds the model context -- and ``autotune`` runs
        even with the prefill graph disabled."""
        import torch

        from tokenspeed.runtime.layers.attention.backends.state.checkpoint import (
            compute_state_block_indices,
        )

        pool = _fake_pool(
            specs=(
                _spec("full_attention", block_granularity=64),
                _spec("linear_attention", family="state", block_granularity=128),
            )
        )
        bare = self._bare(_backend(), pool)
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
            group_id="linear_attention",
        )

    def test_expanded_row_reaches_the_kernels_full_width(self):
        """The width is stated in block granularity, but a stride-deriving
        kernel (trtllm) indexes the whole row. The safety step is the one
        mapping point: the router's ``GroupTableStacks`` expands the raw row
        to the leaf's ``max_num_pages``. Pin that, or the contract the
        deleted width flag protected has no test."""
        from tokenspeed.runtime.layers.attention.backends.paged.group_tables import (
            GroupTableSpec,
            GroupTableStacks,
        )

        spec = _spec("full_attention", block_granularity=128)
        bare = self._bare(_backend(), _fake_pool(specs=(spec,)))
        bare.config.physical_context_len = 8192
        tables = bare._dummy_group_tables(1)

        max_num_pages = -(-8192 // 64)
        stacks = GroupTableStacks(
            [
                GroupTableSpec(
                    "full_attention",
                    block_granularity=128,
                    kernel_page_size=64,
                    max_num_pages=max_num_pages,
                )
            ],
            max_bs=1,
            max_tokens_per_req=1,
            device="cpu",
        )
        stacks.fill(1, 1, dict(tables))
        expanded = stacks.table("full_attention", 1)

        self.assertEqual(
            expanded.shape[1],
            max_num_pages,
            "the expanded row must span the width the kernel derives from "
            "max_kv_len",
        )
        self.assertGreater(int(expanded.min()), 0)

        # Padded max_num_pages: TRTLLM-MLA rounds its width up to a block
        # constraint, so the expansion tail is zero-filled past the live
        # range. The contract is "no null block INSIDE the live range"; a
        # blanket min() > 0 passes above only because 8192/128*2 lands exactly
        # on 128, an arithmetic accident this case removes.
        stacks = GroupTableStacks(
            [
                GroupTableSpec(
                    "full_attention",
                    block_granularity=128,
                    kernel_page_size=64,
                    max_num_pages=130,
                )
            ],
            max_bs=1,
            max_tokens_per_req=1,
            device="cpu",
        )
        stacks.fill(1, 1, dict(tables))
        padded = stacks.table("full_attention", 1)
        live = -(-8192 // 64)
        self.assertEqual(padded.shape[1], 130)
        self.assertGreater(
            int(padded[:, :live].min()),
            0,
            "the live prefix must never contain the reserved null block",
        )
        self.assertEqual(
            int(padded[:, live:].abs().sum()),
            0,
            "the tail past the live range is the zero-filled null page",
        )

    def _dummy_batch_probe(
        self,
        *,
        num_tokens,
        context_len,
        physical,
        specs,
        capture_bs,
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

        from tokenspeed.runtime.execution.input_buffer import InputBuffers
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
        pg.attn_backend = _backend()
        pg.token_to_kv_pool = _fake_pool(specs=tuple(specs), runtime_contract=contract)
        pg.config = SimpleNamespace(
            device="cpu",
            context_len=context_len,
            physical_context_len=physical,
            world_size=1,
        )
        pg.dp_size = 1
        pg.drafter = None
        pg.input_buffers = InputBuffers(
            max_bs=16,
            max_num_tokens=4096,
            state_write_padding_pool_index=0,
            device="cpu",
        )
        pg.block_table = torch.zeros(16, 64, dtype=torch.int32)

        seen = {}

        def _record(**kwargs):
            seen.update(kwargs)
            # The row-constant table is legal only for a prefix-free extend:
            # with history the state gather resolves in == out and refuses.
            seen["max_prefix"] = int(pg.input_buffers.extend_prefix_lens_cpu.max())

        pg.attn_backend.init_forward_metadata = _record
        ctx = pg.make_dummy_batch(
            num_tokens,
            -(-num_tokens // context_len) if capture_bs is None else capture_bs,
        )
        bs = ctx.bs
        ib = pg.input_buffers
        self.assertEqual(
            ib.request_token_history_input_lengths_buf[:bs].tolist(),
            ib.extend_seq_lens_cpu[:bs].tolist(),
        )
        self.assertEqual(ib.input_start_offsets_buf[0].item(), 0)
        self.assertEqual(ib.input_start_offsets_buf[bs].item(), num_tokens)
        self.assertTrue(ib.active_request_mask_buf[:bs].all().item())
        self.assertIs(ctx.attn_backend, pg.attn_backend)
        self.assertIs(ctx.token_to_kv_pool, pg.token_to_kv_pool)
        return seen

    def test_make_dummy_batch_tables_survive_the_cache_contract(self):
        """The real validator sees the tables, and rows track the batch."""
        spec = _spec("full_attention", block_granularity=64)
        # 2048 tokens over a 960 context is three fabricated requests, so a
        # rule that collapsed rows to one would be visible here.
        seen = self._dummy_batch_probe(
            num_tokens=2048,
            context_len=960,
            physical=1024,
            specs=(spec,),
            capture_bs=None,
        )
        tables = seen["block_tables"]
        table = tables["full_attention"]
        self.assertEqual(table.shape[0], 3, "one row per fabricated request")
        # physical (1024), not the user-facing context_len (960): 16 vs 15.
        self.assertEqual(table.shape[1], 16)
        self.assertGreater(int(table.min()), 0)
        # The dummy tables travel bridge-packed (one storage, contract order);
        # the cache-metadata object itself no longer rides to backends.
        self.assertNotIn("cache_metadata", seen)
        # The tables are built on the host; the packer's output is what the
        # backend gets, and it must land on the configured device. Nothing
        # else re-places them, so this assertion is the whole device contract.
        self.assertEqual(table.device.type, "cpu")
        self.assertEqual(seen["max_prefix"], 0, "capture fabricates no prefix")

    def test_explicit_request_count_uses_balanced_nonempty_placeholder_rows(self):
        spec = _spec("full_attention", block_granularity=64)
        seen = self._dummy_batch_probe(
            num_tokens=1737,
            context_len=2048,
            physical=2048,
            specs=(spec,),
            capture_bs=2,
        )
        self.assertEqual(seen["extend_seq_lens_cpu"].tolist(), [869, 868])
        self.assertEqual(seen["block_tables"]["full_attention"].shape[0], 2)
        for tokens, bs in [(1, 2), (1737, 0), (1737, 17), (4096, 1)]:
            with self.subTest(tokens=tokens, bs=bs):
                with self.assertRaisesRegex(ValueError, "token/context capacity"):
                    self._dummy_batch_probe(
                        num_tokens=tokens,
                        context_len=2048,
                        physical=2048,
                        specs=(spec,),
                        capture_bs=bs,
                    )

    def test_real_active_page_backend_gets_positions_alongside_its_tables(self):
        """A backend that validates live-page geometry (V4) is told how many
        tokens the batch carries and handed the live positions slice, and it
        still gets the metadata-derived tables. The decode wrapper is not
        consulted: capture builds its own distinct-block tables."""
        spec = _spec("full_attention", block_granularity=64)
        seen = self._dummy_batch_probe(
            num_tokens=128,
            context_len=960,
            physical=1024,
            specs=(spec,),
            capture_bs=None,
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
            capture_bs=None,
            arena_blocks=2,
        )
        with self.assertRaises(ValueError) as caught:
            self._dummy_batch_probe(
                num_tokens=2880,
                context_len=960,
                physical=1024,
                specs=(spec,),
                capture_bs=None,
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
        bare = self._bare(_backend(), _fake_pool(specs=(spec,)))
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
        bare = self._bare(_backend(), _fake_pool(specs=(spec,)))
        for extent in (0, -8):
            bare.config.physical_context_len = extent
            self.assertEqual(
                bare._dummy_group_tables(1)["full_attention"].shape[1], 1, extent
            )

    def test_pool_without_groups_is_empty(self):
        backend = _backend()
        pool = _fake_pool(specs=())
        self.assertEqual(self._bare(backend, pool)._dummy_group_tables(1), {})

    def test_mla_target_gets_tables_from_the_pool_alone(self):
        """No backend gate: the pool's published specs are the only key.
        Gating capture metadata on a backend flag handed MLA targets a dummy
        batch with no metadata, which they refuse -- Kimi-K2.5 ran every eval
        on eager prefill before the gate was removed."""
        backend = _backend()
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

    def test_runtime_contract_pool_is_eligible_for_capture(self):
        from unittest import mock

        inner_model = SimpleNamespace(embed_tokens=object())
        model_runner = SimpleNamespace(
            model=SimpleNamespace(model=inner_model),
            model_config=SimpleNamespace(requires_request_token_history=False),
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
            )

        self.assertFalse(graph.disable)
        capture.assert_not_called()

        model_runner.model_config.requires_request_token_history = True
        with (
            mock.patch(
                "tokenspeed.runtime.execution.prefill_graph.get_prefill_token_buckets",
                return_value=[64],
            ),
            mock.patch.object(self.PrefillGraph, "capture"),
        ):
            graph = self.PrefillGraph(
                model_runner=model_runner,
                attn_backend=object(),
                token_to_kv_pool=pool,
                input_buffers=object(),
                config=config,
            )
        self.assertTrue(graph.disable)


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
        pg._captures = {}
        pg._encoders = {}
        pg._decoders = {}
        pg._narrowing = None
        pg.attn_backend = SimpleNamespace(
            init_prefill_graph_state=lambda **kwargs: None
        )
        pg.block_table = self.torch.zeros(4, 4, dtype=self.torch.int32)
        pg.config = SimpleNamespace(
            device="cpu",
            world_group=None,
            world_size=1,
            max_num_seqs=4,
            data_parallel_size=1,
        )
        pg._embed_tokens = SimpleNamespace(
            weight=self.torch.zeros(2, 8, dtype=self.torch.float32)
        )

        def _capture_all_buckets(_decode_wrapper, _entries, _observer):
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
            pg.capture(None, entries=None, observer=NULL_MEMORY_DELTA_OBSERVER)
        self.assertIs(caught.exception, cause)

    def test_successful_capture_does_not_raise(self):
        self._bare().capture(None, entries=None, observer=NULL_MEMORY_DELTA_OBSERVER)

    def test_oom_propagates(self):
        """OOM keeps its own type and message. The capture pool not fitting is
        an operator-visible sizing failure, not something to recover from."""
        pg = self._bare(raises=self.torch.cuda.OutOfMemoryError("no room"))
        with self.assertRaises(self.torch.cuda.OutOfMemoryError):
            pg.capture(None, entries=None, observer=NULL_MEMORY_DELTA_OBSERVER)


class NarrowingPrefillGraphTest(unittest.TestCase):
    """A NarrowingPrefillModel is captured as encoder graphs per token bucket
    plus decoder graphs per decoder-row bucket around its eager narrowing
    stage; replay sequences the three and falls back to an eager decoder
    stage above the largest decoder bucket."""

    def setUp(self):
        try:
            import torch

            from tokenspeed.runtime.execution import prefill_graph
        except (ImportError, ModuleNotFoundError) as exc:
            self.skipTest(f"needs torch + runtime deps: {exc}")
        self.torch = torch
        self.mod = prefill_graph

    def test_decoder_row_buckets_clip_the_token_ladder_to_the_row_cap(self):
        buckets = self.mod.get_decoder_row_buckets
        ladder = [16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192]
        # 128 rows per request x 32 requests caps the decoder at 4096 rows.
        self.assertEqual(buckets(ladder, 128, 32), ladder[:-1])
        # A cap between rungs becomes the top rung itself.
        self.assertEqual(buckets(ladder, 128, 3), [16, 32, 64, 128, 256, 384])
        # The token ladder bounds the cap: narrowing never adds rows.
        self.assertEqual(buckets([16, 48], 128, 32), [16, 48])
        self.assertEqual(buckets([], 128, 32), [])

    def _model(self, rows, calls):
        torch = self.torch

        class State:
            def __init__(self, rows):
                self.rows = rows

            def land_into(self, dst):
                calls.append(("land", self.rows, dst.rows))

        class Model:
            max_decoder_rows_per_request = 128

            def encoder_forward(self, input_ids, positions, ctx, **model_kwargs):
                return State(input_ids.shape[0])

            def narrowing_forward(self, state, ctx):
                calls.append(("narrow", state.rows, ctx.input_num_tokens))
                return State(rows)

            def decoder_forward(self, state, ctx):
                calls.append(("decoder eager", state.rows, ctx.input_num_tokens))
                return torch.zeros(state.rows, 2), []

            def finish_forward(self, hidden, captured, ctx):
                calls.append(("finish", hidden.shape[0], ctx.input_num_tokens))
                return hidden, None

            def decoder_rows(self, ctx):
                return rows

            def allocate_decoder_state(self, rows):
                return State(rows)

        return Model(), State

    def test_protocol_detection_sizes_the_decoder_ladder(self):
        from unittest import mock

        model, _ = self._model(128, [])
        self.assertIsInstance(model, self.mod.NarrowingPrefillModel)
        self.assertNotIsInstance(
            SimpleNamespace(embed_tokens=object()), self.mod.NarrowingPrefillModel
        )
        for inner, expected_buckets in (
            (model, [64, 256, 384]),
            (SimpleNamespace(embed_tokens=object()), []),
        ):
            inner.embed_tokens = object()
            model_runner = SimpleNamespace(
                model=SimpleNamespace(model=inner),
                model_config=SimpleNamespace(requires_request_token_history=False),
                is_generation=True,
                is_multimodal=False,
            )
            config = SimpleNamespace(
                enforce_eager=False,
                disable_prefill_graph=False,
                data_parallel_size=1,
                max_num_seqs=3,
            )
            with (
                mock.patch.object(
                    self.mod, "get_prefill_token_buckets", return_value=[64, 256, 1024]
                ),
                mock.patch.object(self.mod.PrefillGraph, "capture"),
            ):
                graph = self.mod.PrefillGraph(
                    model_runner=model_runner,
                    attn_backend=object(),
                    token_to_kv_pool=_fake_pool(runtime_contract=object()),
                    input_buffers=object(),
                    config=config,
                )
            self.assertFalse(graph.disable)
            self.assertIs(graph._narrowing, inner if expected_buckets else None)
            self.assertEqual(graph.decoder_buckets, expected_buckets)

    def test_narrowing_model_stays_eager_under_attention_dp(self):
        """The narrowed row count is rank-local, so decoder buckets (and the
        collective shapes their graphs bake) could differ across DP ranks:
        the split graph is off under DP; an ordinary model keeps its graph."""
        from unittest import mock

        model, _ = self._model(128, [])
        for inner, expected_disable in (
            (model, True),
            (SimpleNamespace(embed_tokens=object()), False),
        ):
            inner.embed_tokens = object()
            model_runner = SimpleNamespace(
                model=SimpleNamespace(model=inner),
                model_config=SimpleNamespace(requires_request_token_history=False),
                is_generation=True,
                is_multimodal=False,
            )
            config = SimpleNamespace(
                enforce_eager=False,
                disable_prefill_graph=False,
                data_parallel_size=2,
                max_num_seqs=8,
            )
            with (
                mock.patch.object(
                    self.mod, "get_prefill_token_buckets", return_value=[64, 256]
                ),
                mock.patch.object(self.mod.PrefillGraph, "capture"),
            ):
                graph = self.mod.PrefillGraph(
                    model_runner=model_runner,
                    attn_backend=object(),
                    token_to_kv_pool=_fake_pool(runtime_contract=object()),
                    input_buffers=object(),
                    config=config,
                )
            self.assertEqual(graph.disable, expected_disable)
            self.assertEqual(graph.decoder_buckets, [])

    def _bare(self, model, State, decoder_buckets, calls):
        pg = self.mod.PrefillGraph.__new__(self.mod.PrefillGraph)
        pg._narrowing = model
        pg.decoder_buckets = decoder_buckets
        pg.dp_size = 1
        pg._engaged_logged = set()
        pg.config = SimpleNamespace(world_size=1)
        pg.attn_backend = SimpleNamespace(
            step_counter=None,
            prepare_prefill_metadata=lambda *args, **kwargs: False,
        )
        pg._encoders = {
            256: self.mod.CapturedEncoder(
                SimpleNamespace(
                    replay=lambda valid_rows: calls.append(("encoder", valid_rows))
                ),
                State(256),
            )
        }
        pg._decoders = {
            rows: self.mod.CapturedDecoder(
                SimpleNamespace(
                    replay=lambda valid_rows, rows=rows: calls.append(
                        ("decoder graph", rows, valid_rows)
                    )
                ),
                State(rows),
                self.mod.CapturedForward(self.torch.zeros(rows, 2), []),
            )
            for rows in decoder_buckets
        }
        pg._captures = {}
        return pg

    def _ctx(self, num_tokens):
        from tokenspeed.runtime.execution.context import ForwardContext
        from tokenspeed.runtime.execution.forward_batch_info import ForwardMode

        return ForwardContext(
            attn_backend=None,
            token_to_kv_pool=None,
            bs=1,
            num_extends=1,
            input_num_tokens=num_tokens,
            forward_mode=ForwardMode.EXTEND,
        )

    def test_replay_sequences_encoder_narrowing_and_decoder_graph(self):
        calls = []
        model, State = self._model(100, calls)
        pg = self._bare(model, State, [64, 192], calls)
        ctx = self._ctx(200)
        hidden, aux = pg._replay_narrowed(256, ctx, 200)
        self.assertEqual(
            calls,
            [
                # The encoder replays over the padded bucket with the real
                # token count as its valid rows; the narrowing stage runs
                # eager under the bucket-pinned ambient ctx on the encoder's
                # padded output; the decoder graph replays over the smallest
                # fitting bucket with the narrowed row count as its valid
                # rows; the finish stage sees the pin lifted.
                ("encoder", 200),
                ("narrow", 256, 256),
                ("land", 100, 192),
                ("decoder graph", 192, 100),
                ("finish", 100, 200),
            ],
        )
        self.assertEqual(hidden.shape, (100, 2))
        self.assertIsNone(aux)
        self.assertEqual(ctx.input_num_tokens, 200, "the pin is restored")
        self.assertTrue(pg._has_bucket(256) and not pg._has_bucket(64))

    def test_rows_above_the_decoder_ladder_run_the_decoder_eager(self):
        calls = []
        model, State = self._model(300, calls)
        pg = self._bare(model, State, [64, 192], calls)
        self.assertIsNone(pg._decoder_bucket(300))
        self.assertEqual(pg._decoder_bucket(64), 64)
        self.assertEqual(pg._decoder_bucket(65), 192)
        hidden, _ = pg._replay_narrowed(256, self._ctx(256), 256)
        self.assertEqual(
            calls,
            [
                ("encoder", 256),
                ("narrow", 256, 256),
                ("decoder eager", 300, 256),
                ("finish", 300, 256),
            ],
        )
        self.assertEqual(hidden.shape, (300, 2))

    def test_narrowing_disagreeing_with_the_metadata_is_fatal(self):
        calls = []
        model, State = self._model(100, calls)
        model.decoder_rows = lambda ctx: 99
        pg = self._bare(model, State, [192], calls)
        with self.assertRaisesRegex(RuntimeError, "narrowing yielded 100 rows"):
            pg._replay_narrowed(256, self._ctx(200), 200)


class TrtllmPrefillGraphSeamsTest(unittest.TestCase):
    """trtllm under the prefill graph: the extend prewrite must not bake
    capture-time write locs into the graph, and the break's KV write must
    trim padded tails like mha."""

    def setUp(self):
        try:
            import torch

            from tokenspeed.runtime.layers.attention.backends.paged import trtllm
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

    def test_router_declares_history_contract_family(self):
        # The family claim moved off the leaves: the runner-facing node in
        # front of every trtllm leaf is the CacheGroupRouter, whose base
        # declaration is the history family.
        from tokenspeed.runtime.layers.attention.backends.paged.router import (
            CacheGroupRouter,
        )

        self.assertEqual(
            CacheGroupRouter.cache_consumer_families, frozenset({"history"})
        )


class PrefillRoleGraphsTest(unittest.TestCase):
    """The PD prefill role never runs a decode step, so the decode graph has
    nothing to capture there; the prefill graph keeps its ordinary gating
    instead of the role forcing eager execution."""

    def setUp(self):
        try:
            import torch  # noqa: F401

            from tokenspeed.runtime.execution import forward_step, prefill_graph
            from tokenspeed.runtime.execution.model_executor import (
                ModelExecutorConfig,
            )
        except (ImportError, ModuleNotFoundError) as exc:
            self.skipTest(f"needs torch + runtime deps: {exc}")
        self.forward_step = forward_step
        self.prefill_graph = prefill_graph
        self.ModelExecutorConfig = ModelExecutorConfig

    def _config(self, *, prefill_only: bool, enforce_eager: bool = False):
        return self.ModelExecutorConfig(
            max_req_pool_size=5,
            output_length=1,
            enforce_eager=enforce_eager,
            prefix_granularity=128,
            max_num_seqs=4,
            chunked_prefill_size=4096,
            vocab_size=32,
            context_len=4096,
            physical_context_len=4096,
            device="cpu",
            gpu_id=0,
            global_rank=0,
            cudagraph_capture_sizes=[1, 2, 4],
            disable_cuda_graph_padding=False,
            max_cudagraph_capture_size=4,
            model_is_mrope=False,
            prefill_only=prefill_only,
            prefill_graph_capture_batch_sizes=None,
            prefill_graph_max_tokens=256,
        )

    def _decode_runner(self, config):
        class Backend:
            def init_cuda_graph_state(self, *args, **kwargs):
                pass

        return self.forward_step.ForwardStepRunner(
            forward_func=lambda *args, **kwargs: None,
            attn_backend=Backend(),
            token_to_kv_pool=_fake_pool(cache_group_page_counts={}),
            input_buffers=object(),
            config=config,
        )

    def _prefill_owner(self, config):
        from unittest import mock

        inner = SimpleNamespace(embed_tokens=object())
        model_runner = SimpleNamespace(
            model=SimpleNamespace(model=inner),
            model_config=SimpleNamespace(requires_request_token_history=False),
            is_generation=True,
            is_multimodal=False,
        )
        with mock.patch.object(self.prefill_graph.PrefillGraph, "capture"):
            return self.prefill_graph.PrefillGraph(
                model_runner=model_runner,
                attn_backend=object(),
                token_to_kv_pool=_fake_pool(runtime_contract=object()),
                input_buffers=object(),
                config=config,
            )

    def test_prefill_role_skips_the_decode_graph_and_keeps_the_prefill_graph(
        self,
    ):
        config = self._config(prefill_only=True)
        self.assertTrue(self._decode_runner(config).disable)
        self.assertFalse(self._prefill_owner(config).disable)

    def test_a_serving_node_keeps_both_graphs(self):
        config = self._config(prefill_only=False)
        self.assertFalse(self._decode_runner(config).disable)
        self.assertFalse(self._prefill_owner(config).disable)

    def test_explicit_eager_still_disables_the_prefill_graph_on_the_role(self):
        config = self._config(prefill_only=True, enforce_eager=True)
        self.assertTrue(self._decode_runner(config).disable)
        self.assertTrue(self._prefill_owner(config).disable)


if __name__ == "__main__":
    unittest.main()
