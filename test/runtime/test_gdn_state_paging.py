"""GDN dual-index state paging.

compute_state_block_indices maps per-request (seq_len_before, seq_len_after)
to (in, out) state page ids over the "linear_attention" block table;
the GPU test drives MambaAttnBackend (prefill + decodes over
paged state slabs) against the FLA chunk_gated_delta_rule oracle run once
over the full contiguous sequence.
"""

from __future__ import annotations

import os
import sys
import unittest
from types import SimpleNamespace

# CI Registration (parsed via AST, runtime no-op)
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from ci_system.ci_register import register_cuda_ci

register_cuda_ci(est_time=90, suite="runtime-1gpu")


def test_state_helpers_use_shared_implementations():
    """Backend-local definitions must not shadow the shared state helpers."""
    from tokenspeed_kernel.ops.attention.gdn.triton import prepare_prefill_state_inputs

    from tokenspeed.runtime.layers.attention.backends.state import checkpoint, mamba

    assert mamba._prepare_cache_prefill_state_inputs is prepare_prefill_state_inputs
    assert (
        mamba._compute_state_block_index_plan
        is checkpoint._compute_state_block_index_plan
    )
    assert mamba._gather_state_block_indices is checkpoint._gather_state_block_indices


class _ContractPool:
    def __init__(self, page_size, components):
        # The arena publishes the contract; a view only names its arena.
        contract = SimpleNamespace(
            prefix_granularity=page_size,
            group_specs=tuple(
                SimpleNamespace(
                    group_id=group_id,
                    family="state",
                    checkpoint_granularity=page_size,
                )
                for group_id in dict.fromkeys(
                    group_id for group_id, _, _ in components.values()
                )
            ),
        )
        self.arena = SimpleNamespace(runtime_contract=contract)
        self._components = components
        self.state_group_by_layer = {
            layer_id: group_id for layer_id, (group_id, _, _) in components.items()
        }

    def get_component(self, layer_id, name):
        _, conv_state, recurrent_state = self._components[layer_id]
        return conv_state if name == "conv_state" else recurrent_state


def _mamba_config_pair(
    torch,
    *,
    heads,
    head_dim,
    spec_tokens=1,
    max_bs=8,
    device="cpu",
    replay_ssm=False,
):
    """(AttnConfig, softmax spec) for MambaAttnBackend: model-wide facts live on
    the config, softmax geometry on the softmax spec, and the GDN geometry plus
    replay_ssm on the LinearAttnConfig component."""
    from tokenspeed.runtime.layers.attention.configs.base import AttnConfig
    from tokenspeed.runtime.layers.attention.configs.linear_attn import (
        LinearAttnConfig,
    )
    from tokenspeed.runtime.layers.attention.configs.mha import MHAConfig

    spec = MHAConfig(
        num_attention_heads=heads,
        num_kv_heads=heads,
        head_dim=head_dim,
        attn_tp_size=1,
    )
    linear = LinearAttnConfig(
        num_k_heads=heads,
        num_v_heads=heads,
        head_k_dim=head_dim,
        head_v_dim=head_dim,
        conv_kernel_size=4,
        layer_ids=(0,),
        tp_size=1,
        replay_ssm=replay_ssm,
    )
    config = AttnConfig(
        device=device,
        dtype=torch.bfloat16,
        kv_cache_dtype=torch.bfloat16,
        kv_cache_quant_method="none",
        prefix_granularity=64,
        context_len=4096,
        max_bs=max_bs,
        speculative_num_draft_tokens=spec_tokens,
        components=(spec, linear),
    )
    return config, spec


def _extend_kwargs(torch, extend_seq_lens_cpu, extend_prefix_lens_cpu, device):
    """The ``init_forward_metadata`` extend bundle from its host mirrors."""
    return dict(
        extend_seq_lens=extend_seq_lens_cpu.to(device),
        extend_seq_lens_cpu=extend_seq_lens_cpu,
        extend_prefix_lens=extend_prefix_lens_cpu.to(device),
        extend_prefix_lens_cpu=extend_prefix_lens_cpu,
        extend_with_prefix=bool(extend_prefix_lens_cpu.any()),
    )


def _no_extends(torch, device):
    """The extend bundle of a decode-mode call: no extend rows."""
    empty = torch.zeros(0, dtype=torch.int32)
    return _extend_kwargs(torch, empty, empty, device)


class ComputeStatePageIndicesTest(unittest.TestCase):
    """CPU-only contract tests for the pure dual-index helper."""

    def setUp(self):
        try:
            import torch

            from tokenspeed.runtime.layers.attention.backends.state.checkpoint import (  # noqa: E501
                compute_state_block_indices,
            )
        except (ImportError, ModuleNotFoundError) as exc:
            self.skipTest(f"needs torch + tokenspeed_kernel: {exc}")
        self.torch = torch
        self.fn = compute_state_block_indices

    def _run(self, rows, before, after, page_size=4):
        torch = self.torch
        return self.fn(
            torch.tensor(rows, dtype=torch.int32),
            page_size,
            torch.tensor(before, dtype=torch.int32),
            torch.tensor(after, dtype=torch.int32),
            validate=True,
            group_id="linear_attention",
        )

    def test_across_boundary(self):
        state_in, state_out = self._run([[7, 9, 12]], [4], [5])
        self.assertEqual(state_in.tolist(), [7])
        self.assertEqual(state_out.tolist(), [9])

    def test_within_page(self):
        state_in, state_out = self._run([[7, 9, 12]], [5], [6])
        self.assertEqual(state_in.tolist(), [9])
        self.assertEqual(state_out.tolist(), [9])

    def test_first_step_null_in_page(self):
        state_in, state_out = self._run([[7, 9, 12]], [0], [3])
        self.assertEqual(state_in.tolist(), [0])
        self.assertEqual(state_out.tolist(), [7])

    def test_resume_from_prefix_hit(self):
        state_in, state_out = self._run([[3, 5, 8]], [8], [9])
        self.assertEqual(state_in.tolist(), [5])
        self.assertEqual(state_out.tolist(), [8])

    def test_sparse_prefill_ignores_intermediate_holes(self):
        state_in, state_out = self._run([[7, 0, 0, 0, 9]], [4], [20])
        self.assertEqual(state_in.tolist(), [7])
        self.assertEqual(state_out.tolist(), [9])

    def test_batch_mixed(self):
        # Distinct rows per request: out pages are exclusive per batch (the scheduler
        # invariant the validate path enforces).
        rows = [
            [7, 9, 12],
            [21, 22, 23],
            [31, 33, 35],
            [3, 5, 8],
        ]
        state_in, state_out = self._run(rows, [4, 5, 0, 8], [5, 6, 3, 9])
        self.assertEqual(state_in.tolist(), [7, 22, 0, 5])
        self.assertEqual(state_out.tolist(), [9, 22, 31, 8])

    def test_out_slot_hole_raises(self):
        with self.assertRaises(ValueError):
            self._run([[7, 0, 12]], [4], [5])

    def test_index_plan_preserves_int32_inputs(self):
        torch = self.torch
        from tokenspeed.runtime.layers.attention.backends.state.checkpoint import (
            _compute_state_block_index_plan,
        )

        plan = _compute_state_block_index_plan(
            4,
            torch.tensor([4, 7], dtype=torch.int32),
            torch.tensor([5, 8], dtype=torch.int32),
        )

        self.assertEqual(plan.before.dtype, torch.int32)
        self.assertEqual(plan.after.dtype, torch.int32)
        self.assertEqual(plan.in_slots.dtype, torch.int32)
        self.assertEqual(plan.out_slots.dtype, torch.int32)
        self.assertEqual(plan.in_slots.tolist(), [0, 1])
        self.assertEqual(plan.out_slots.tolist(), [1, 1])

    def test_out_slot_pad_raises(self):
        with self.assertRaises(ValueError):
            self._run([[7, -1, 12]], [4], [5])

    def test_out_slot_past_table_raises(self):
        with self.assertRaises(ValueError):
            self._run([[7, 9]], [8], [9])

    def test_in_slot_hole_raises(self):
        # before=5 -> in slot 1 is a hole (0): a silent zero-state resume
        # must fail loud like the out-page case.
        with self.assertRaises(ValueError):
            self._run([[7, 0, 12]], [5], [6])

    def test_in_slot_pad_raises(self):
        with self.assertRaises(ValueError):
            self._run([[7, -1, 12]], [5], [6])

    def test_duplicate_out_pages_raise(self):
        # req0: before=4 after=5 -> out slot 1 -> page 9; req1: before=0
        # after=1 -> out slot 0 -> page 9. All other guards pass (pages
        # positive, in-page valid/no history), so only the batch-uniqueness
        # invariant fires: two requests writing the same working state page
        # would silently clobber each other.
        with self.assertRaisesRegex(ValueError, "unique"):
            self._run([[7, 9, 12], [9, 22, 23]], [4, 0], [5, 1])

    def test_no_history_null_in_page_passes(self):
        # before=0 legitimately reads the null page 0 (see
        # test_first_step_null_in_page); the in-page guard must not fire.
        state_in, state_out = self._run([[7, 9, 12]], [0], [1])
        self.assertEqual(state_in.tolist(), [0])
        self.assertEqual(state_out.tolist(), [7])


class PrefillCheckpointPageTest(unittest.TestCase):
    """A finishing off-page prefill owns two distinct state outputs."""

    def setUp(self):
        try:
            import torch

            from tokenspeed.runtime.layers.attention.backends.state.checkpoint import (
                compute_state_block_indices,
            )
            from tokenspeed.runtime.layers.attention.backends.state.mamba import (
                MambaAttnBackend,
            )
        except (ImportError, ModuleNotFoundError) as exc:
            self.skipTest(f"needs torch + tokenspeed_kernel: {exc}")
        self.torch = torch
        self.backend = object.__new__(MambaAttnBackend)
        self.fn = compute_state_block_indices
        self.backend._checkpoint_granularity = 4
        self.backend._state_group_ids = ("linear_attention",)
        self.backend.pad_slot_id = -1

    def test_off_page_endpoint_selects_aligned_and_final_output_pages(self):
        torch = self.torch
        from tokenspeed.runtime.layers.attention.backends.state.mamba import (
            _build_prefill_checkpoint_batch,
        )

        before = torch.tensor([4], dtype=torch.int32)
        after = torch.tensor([11], dtype=torch.int32)
        state_in, state_out, checkpoint = self.backend._cache_contract_state_blocks(
            before,
            after,
            {"linear_attention": torch.tensor([[7, 8, 9]], dtype=torch.int32)},
            validate=True,
            checkpoint_batch=_build_prefill_checkpoint_batch(
                after - before, before, 1, 4, "cpu"
            ),
        )

        self.assertEqual(state_in["linear_attention"].tolist(), [7])
        self.assertEqual(checkpoint["linear_attention"].tolist(), [8])
        self.assertEqual(state_out["linear_attention"].tolist(), [9])

    def test_aligned_endpoint_has_no_extra_checkpoint_page(self):
        torch = self.torch
        _, state_out, checkpoint = self.backend._cache_contract_state_blocks(
            torch.tensor([8], dtype=torch.int32),
            torch.tensor([12], dtype=torch.int32),
            {"linear_attention": torch.tensor([[7, 8, 9]], dtype=torch.int32)},
            validate=True,
            checkpoint_batch=None,
        )

        self.assertIsNone(checkpoint)
        self.assertEqual(state_out["linear_attention"].tolist(), [9])

    def test_prefix_boundary_is_independent_of_state_block_span(self):
        from tokenspeed.runtime.layers.attention.backends.state.mamba import (
            _build_prefill_checkpoint_batch,
        )

        torch = self.torch
        before = torch.tensor([0], dtype=torch.int32)
        after = torch.tensor([7], dtype=torch.int32)
        self.backend._checkpoint_granularity = 2
        batch = _build_prefill_checkpoint_batch(after - before, before, 1, 4, "cpu")
        self.assertEqual(batch.checkpoint_positions.tolist(), [4])
        _, state_out, checkpoint = self.backend._cache_contract_state_blocks(
            before,
            after,
            {"linear_attention": torch.tensor([[0, 7, 8, 9]], dtype=torch.int32)},
            validate=True,
            checkpoint_batch=batch,
        )
        # Publish token 4 (slot 1), not token 6 (slot 2). The endpoint is token 7.
        self.assertEqual(checkpoint["linear_attention"].tolist(), [7])
        self.assertEqual(state_out["linear_attention"].tolist(), [9])

    def test_decode_never_builds_checkpoint_tensors(self):
        from unittest.mock import patch

        torch = self.torch
        with patch.object(
            torch, "full_like", side_effect=AssertionError("checkpoint work in decode")
        ):
            _, _, checkpoint = self.backend._cache_contract_state_blocks(
                torch.tensor([8], dtype=torch.int32),
                torch.tensor([9], dtype=torch.int32),
                {"linear_attention": torch.tensor([[0, 7, 9]], dtype=torch.int32)},
                validate=False,
                checkpoint_batch=None,
            )
        self.assertIsNone(checkpoint)

    def test_validate_off_masks_guards(self):
        torch = self.torch
        state_in, state_out = self.fn(
            torch.tensor([[0, 0, 0]], dtype=torch.int32),
            4,
            torch.tensor([0], dtype=torch.int32),
            torch.tensor([1], dtype=torch.int32),
            validate=False,
            group_id="linear_attention",
        )
        self.assertEqual(state_in.tolist(), [0])
        self.assertEqual(state_out.tolist(), [0])


class PrefillCheckpointBatchTest(unittest.TestCase):
    """Cover batch planning, upload, and execution as separate contracts.

    Related metadata cases share subtests; CUDA upload and single-request
    execution stay separate because they exercise different paths.
    """

    def setUp(self):
        try:
            import torch

            from tokenspeed.runtime.layers.attention.backends.state.mamba import (
                MambaAttnBackend,
                _build_prefill_checkpoint_batch,
            )
        except (ImportError, ModuleNotFoundError) as exc:
            self.skipTest(f"needs torch + tokenspeed_kernel: {exc}")
        self.torch = torch
        self.backend = object.__new__(MambaAttnBackend)
        self.build_batch = _build_prefill_checkpoint_batch
        self.plan = _build_prefill_checkpoint_batch(
            torch.tensor([5, 4, 6], dtype=torch.int32),
            torch.tensor([3, 2, 0], dtype=torch.int32),
            3,
            4,
            "cpu",
        )
        assert self.plan is not None

    def test_builds_packed_body_and_tail_batches(self):
        from tokenspeed.runtime.layers.attention.backends.state.kda import (
            KdaAttnBackend,
        )

        torch = self.torch
        single = self.build_batch(
            torch.tensor([868], dtype=torch.int32),
            torch.tensor([50_432], dtype=torch.int32),
            1,
            128,
            "cpu",
        )
        for name, plan, expected in (
            (
                "mixed_endpoints",
                self.plan,
                dict(
                    rows=[1, 2],
                    sequence_starts=[5, 9],
                    checkpoint_seq_lens=[2, 4],
                    checkpoint_positions=[4, 4],
                    body_rows=[0, 1, 2],
                    body_seq_lens_cpu=[5, 2, 4],
                    body_token_indices=[0, 1, 2, 3, 4, 5, 6, 9, 10, 11, 12],
                    body_query_start_loc=[0, 5, 7, 11],
                    tail_seq_lens_cpu=[2, 2],
                    tail_token_indices=[7, 8, 13, 14],
                    tail_query_start_loc=[0, 2, 4],
                ),
            ),
            (
                "868_tokens",
                single,
                dict(
                    rows=[0],
                    sequence_starts=[0],
                    checkpoint_seq_lens=[768],
                    checkpoint_positions=[51_200],
                    body_rows=[0],
                    body_seq_lens_cpu=[768],
                    body_token_indices=list(range(768)),
                    body_query_start_loc=[0, 768],
                    tail_seq_lens_cpu=[100],
                    tail_token_indices=list(range(768, 868)),
                    tail_query_start_loc=[0, 100],
                ),
            ),
        ):
            with self.subTest(case=name):
                self.assertIsNotNone(plan)
                for field, values in expected.items():
                    with self.subTest(field=field):
                        self.assertEqual(getattr(plan, field).tolist(), values)
                # Device metadata fields are views of one upload buffer.
                parts = [
                    getattr(plan, field)
                    for field in expected
                    if not field.endswith("_cpu")
                ]
                self.assertEqual(
                    len({part.untyped_storage().data_ptr() for part in parts}), 1
                )
                self.assertEqual(plan.body_query_start_loc.dtype, torch.int32)
                converted = KdaAttnBackend._prepare_prefill_scan_query_start_loc(
                    object(), plan.body_query_start_loc
                )
                self.assertEqual(converted.dtype, torch.int64)
                self.assertEqual(converted.tolist(), expected["body_query_start_loc"])
                self.assertIs(
                    KdaAttnBackend._prepare_prefill_scan_query_start_loc(
                        object(), converted
                    ),
                    converted,
                )

    def test_only_extend_rows_may_create_internal_checkpoints(self):
        batch = self.build_batch(
            self.torch.tensor([7, 7], dtype=self.torch.int32),
            self.torch.tensor([0, 0], dtype=self.torch.int32),
            1,
            4,
            "cpu",
        )
        assert batch is not None
        self.assertEqual(batch.rows.tolist(), [0])
        self.assertEqual(batch.body_seq_lens_cpu.tolist(), [4, 7])
        self.assertEqual(batch.tail_seq_lens_cpu.tolist(), [3])

    def test_cuda_metadata_upload_does_not_synchronize(self):
        torch = self.torch
        if not torch.cuda.is_available():
            self.skipTest("CUDA required")
        torch.cuda.synchronize()
        previous = torch.cuda.get_sync_debug_mode()
        try:
            torch.cuda.set_sync_debug_mode("error")
            batch = self.build_batch(
                torch.tensor([5, 4, 6], dtype=torch.int32),
                torch.tensor([3, 2, 0], dtype=torch.int32),
                3,
                4,
                "cuda",
            )
        finally:
            torch.cuda.set_sync_debug_mode(previous)
        self.assertEqual(batch.checkpoint_positions.cpu().tolist(), [4, 4])
        self.assertEqual(
            batch.body_token_indices.cpu().tolist(),
            self.plan.body_token_indices.tolist(),
        )

    def test_batches_without_internal_checkpoints(self):
        for name, prefixes, lengths in (
            ("aligned_body", [50_432], [768]),
            ("short_tail", [51_200], [100]),
            ("empty", [], []),
        ):
            with self.subTest(case=name):
                self.assertIsNone(
                    self.build_batch(
                        self.torch.tensor(lengths, dtype=self.torch.int32),
                        self.torch.tensor(prefixes, dtype=self.torch.int32),
                        len(lengths),
                        128,
                        "cpu",
                    )
                )

    def test_invalid_batch_dimensions_raise(self):
        for lengths, prefixes, rows, error in (
            ([7, 7], [0], 2, "same shape"),
            ([7], [0], 2, "checkpoint row count"),
        ):
            with self.subTest(error=error), self.assertRaisesRegex(ValueError, error):
                self.build_batch(
                    self.torch.tensor(lengths),
                    self.torch.tensor(prefixes),
                    rows,
                    4,
                    "cpu",
                )

    def test_conv_checkpoints_are_written_as_one_batch(self):
        from tokenspeed_kernel.ops.attention._triton.prefill_state_checkpoints import (
            write_prefill_conv_checkpoints,
        )

        torch = self.torch
        raw = torch.arange(30, dtype=torch.float32).view(15, 2)
        states = torch.arange(60, dtype=torch.float32).view(10, 2, 3)
        source_page_two = states[2].clone()

        write_prefill_conv_checkpoints(
            raw,
            states,
            torch.tensor([1, 2, 3], dtype=torch.int32),
            torch.tensor([4, 5, 6], dtype=torch.int32),
            torch.tensor([-1, 7, 8], dtype=torch.int32),
            self.plan.rows,
            self.plan.sequence_starts,
            self.plan.checkpoint_seq_lens,
        )

        expected_short = torch.stack((source_page_two[:, -1], raw[5], raw[6]), dim=1)
        expected_long = raw[[10, 11, 12]].transpose(0, 1)
        self.assertTrue(torch.equal(states[7], expected_short))
        self.assertTrue(torch.equal(states[8], expected_long))

    def test_recurrent_prefill_scans_with_and_without_checkpoints(self):
        from unittest.mock import patch

        from tokenspeed.runtime.layers.attention.backends.state import mamba

        torch = self.torch
        calls = []

        def fake_scan(query, key, value, recurrent_state, query_start_loc, **kwargs):
            calls.append((query, key, value, recurrent_state, query_start_loc, kwargs))
            # KDA removes the leading scan batch and returns [T, H, D].
            return query.squeeze(0), recurrent_state + 100

        self.backend._prefill_scan = fake_scan
        tokens = torch.arange(15, dtype=torch.float32).view(1, 15, 1, 1)
        per_token = tokens.view(15, 1)
        recurrent = torch.arange(3, dtype=torch.float32).view(3, 1, 1, 1)
        slab = torch.zeros(10, 1, 1, 1)

        scan_kwargs = dict(
            seq_len=15,
            num_real_tokens=15,
            A_log=torch.empty(1),
            dt_bias=torch.empty(1),
            a=per_token,
            b=per_token,
            g_raw=per_token,
            f_a_out=per_token,
            f_b_weight=torch.empty(1),
            beta_raw=per_token,
            lower_bound=-5.0,
        )
        checkpoint_blocks = torch.tensor([-1, 7, 8], dtype=torch.int32)
        output, final_state = self.backend._run_prefill_recurrent(
            tokens,
            tokens,
            tokens,
            recurrent,
            slab,
            checkpoint_blocks,
            self.plan,
            **scan_kwargs,
        )

        self.assertEqual(len(calls), 2)
        body_query, _, _, body_initial, body_boundaries, body_kwargs = calls[0]
        self.assertEqual(
            body_query.flatten().tolist(), [0, 1, 2, 3, 4, 5, 6, 9, 10, 11, 12]
        )
        self.assertEqual(body_initial.flatten().tolist(), [0, 1, 2])
        self.assertEqual(body_boundaries.tolist(), [0, 5, 7, 11])
        self.assertEqual(
            body_kwargs["a"].flatten().tolist(),
            [0, 1, 2, 3, 4, 5, 6, 9, 10, 11, 12],
        )
        self.assertEqual(body_kwargs["seq_len"], 11)

        tail_query, _, _, tail_initial, tail_boundaries, tail_kwargs = calls[1]
        self.assertEqual(tail_query.flatten().tolist(), [7, 8, 13, 14])
        self.assertEqual(tail_initial.flatten().tolist(), [101, 102])
        self.assertEqual(tail_boundaries.tolist(), [0, 2, 4])
        self.assertEqual(tail_kwargs["a"].flatten().tolist(), [7, 8, 13, 14])
        self.assertEqual(tail_kwargs["seq_len"], 4)
        self.assertEqual(output.shape, (15, 1, 1))
        self.assertEqual(output.flatten().tolist(), list(range(15)))
        self.assertEqual(final_state.flatten().tolist(), [100, 201, 202])
        self.assertEqual(slab[[7, 8]].flatten().tolist(), [101, 102])

        # The same entry point also handles a complete scan, including padding,
        # without writing checkpoint or continuation slots in the state pool.
        boundaries = torch.tensor([0, 5, 9, 15], dtype=torch.int64)
        padded_tokens = torch.arange(16, dtype=torch.float32).view(1, 16, 1, 1)
        fallback_kwargs = dict(scan_kwargs, seq_len=16)
        for blocks, plan in (
            (None, None),
            (None, self.plan),
            (checkpoint_blocks, None),
        ):
            for lengths in (None, torch.tensor([5, 4, 6], dtype=torch.int32)):
                with self.subTest(
                    blocks=blocks is not None,
                    plan=plan is not None,
                    has_lengths=lengths is not None,
                ), patch.object(mamba, "set_total_chunks_hint") as hint:
                    calls.clear()
                    self.backend.forward_metadata = SimpleNamespace(
                        scan_query_start_loc=boundaries,
                        extend_seq_lens_cpu=lengths,
                        cu_extend_seq_lens_cpu=boundaries,
                    )
                    saved_slab = slab.clone()
                    output, final_state = self.backend._run_prefill_recurrent(
                        padded_tokens,
                        padded_tokens,
                        padded_tokens,
                        recurrent,
                        slab,
                        blocks,
                        plan,
                        **fallback_kwargs,
                    )
                    self.assertEqual(len(calls), 1)
                    query, _, _, initial, scan_boundaries, kwargs = calls[0]
                    self.assertIs(query, padded_tokens)
                    self.assertIs(initial, recurrent)
                    self.assertIs(scan_boundaries, boundaries)
                    self.assertIs(kwargs["cu_seqlens_cpu"], boundaries)
                    self.assertEqual(kwargs["seq_len"], 16)
                    self.assertEqual(kwargs["num_real_tokens"], 15)
                    self.assertTrue(torch.equal(output, padded_tokens.squeeze(0)))
                    self.assertTrue(torch.equal(final_state, recurrent + 100))
                    self.assertTrue(torch.equal(slab, saved_slab))
                    if lengths is None:
                        hint.assert_not_called()
                    else:
                        hint.assert_called_once_with(lengths, boundaries)

    def test_single_checkpoint_uses_the_same_pack_and_write_contract(self):
        from tokenspeed_kernel.ops.attention._triton.prefill_state_checkpoints import (
            write_prefill_conv_checkpoints,
        )

        torch = self.torch
        from tokenspeed.runtime.layers.attention.backends.state.mamba import (
            _build_prefill_checkpoint_batch,
        )

        plan = _build_prefill_checkpoint_batch(
            torch.tensor([5], dtype=torch.int32),
            torch.tensor([2], dtype=torch.int32),
            1,
            4,
            "cpu",
        )
        assert plan is not None
        raw = torch.arange(10, dtype=torch.float32).view(5, 2)
        conv_states = torch.arange(36, dtype=torch.float32).view(6, 2, 3)
        source = conv_states[1].clone()
        write_prefill_conv_checkpoints(
            raw,
            conv_states,
            torch.tensor([1], dtype=torch.int32),
            torch.tensor([2], dtype=torch.int32),
            torch.tensor([4], dtype=torch.int32),
            plan.rows,
            plan.sequence_starts,
            plan.checkpoint_seq_lens,
        )
        expected_conv = torch.stack((source[:, -1], raw[0], raw[1]), dim=1)
        self.assertTrue(torch.equal(conv_states[4], expected_conv))

        calls = []

        def fake_scan(query, key, value, recurrent_state, query_start_loc, **kwargs):
            calls.append((query, recurrent_state, query_start_loc, kwargs))
            return query, recurrent_state + 10

        self.backend._prefill_scan = fake_scan
        tokens = torch.arange(5, dtype=torch.float32).view(1, 5, 1, 1)
        per_token = tokens.view(5, 1)
        recurrent = torch.tensor([[[[3.0]]]])
        slab = torch.zeros(6, 1, 1, 1)
        output, final_state = self.backend._run_prefill_recurrent(
            tokens,
            tokens,
            tokens,
            recurrent,
            slab,
            torch.tensor([4], dtype=torch.int32),
            plan,
            seq_len=5,
            num_real_tokens=5,
            A_log=torch.empty(1),
            dt_bias=torch.empty(1),
            a=per_token,
            b=per_token,
            g_raw=per_token,
            f_a_out=per_token,
            f_b_weight=torch.empty(1),
            beta_raw=per_token,
            lower_bound=-5.0,
        )
        self.assertEqual(len(calls), 2)
        query, initial, boundaries, kwargs = calls[0]
        self.assertEqual(query.flatten().tolist(), [0, 1])
        self.assertEqual(initial.flatten().tolist(), [3])
        self.assertEqual(boundaries.tolist(), [0, 2])
        self.assertEqual(kwargs["a"].flatten().tolist(), [0, 1])
        query, initial, boundaries, kwargs = calls[1]
        self.assertEqual(query.flatten().tolist(), [2, 3, 4])
        self.assertEqual(initial.flatten().tolist(), [13])
        self.assertEqual(boundaries.tolist(), [0, 3])
        self.assertEqual(kwargs["a"].flatten().tolist(), [2, 3, 4])
        self.assertEqual(output.flatten().tolist(), [0, 1, 2, 3, 4])
        self.assertEqual(final_state.item(), 23)
        self.assertEqual(slab[4].item(), 13)


class CacheContractMetadataTest(unittest.TestCase):
    """Every metadata entry point resolves state through the cache contract."""

    P = 4  # state page size (tokens)

    def setUp(self):
        try:
            import torch

            from tokenspeed.runtime.execution.forward_batch_info import (
                ForwardMode,
            )
            from tokenspeed.runtime.layers.attention.backends.state.mamba import (  # noqa: E501
                MambaAttnBackend,
            )
        except (ImportError, ModuleNotFoundError) as exc:
            self.skipTest(f"needs torch + tokenspeed_kernel: {exc}")
        self.torch = torch
        self.ForwardMode = ForwardMode
        backend = MambaAttnBackend(
            *_mamba_config_pair(torch, heads=16, head_dim=128, spec_tokens=1)
        )
        stub_pool = _ContractPool(
            self.P,
            {0: ("linear_attention", torch.zeros(2, 3), torch.zeros(2, 5))},
        )
        backend.set_kv_pool(stub_pool)
        self.assertTrue(backend.state_paging_active)
        self.backend = backend

    def test_decode_metadata(self):
        torch = self.torch
        backend = self.backend
        req_pool_indices = torch.tensor([0], dtype=torch.int32)
        seq_lens = torch.tensor([9], dtype=torch.int32)
        block_tables = {
            "linear_attention": torch.tensor([[1, 2, 3]], dtype=torch.int32)
        }
        # Decode metadata is the refresh's alone; a DECODE init is a contract
        # violation on every node, the state backend included.
        with self.assertRaisesRegex(RuntimeError, "refresh_decode_metadata"):
            backend.init_forward_metadata(
                bs=1,
                num_extends=0,
                req_pool_indices=req_pool_indices,
                seq_lens=seq_lens,
                forward_mode=self.ForwardMode.DECODE,
                block_tables=block_tables,
                **_no_extends(torch, "cpu"),
            )
        backend.init_cuda_graph_state(max_bs=1)
        backend.refresh_decode_metadata(
            1,
            1,
            req_pool_indices,
            seq_lens,
            forward_mode=self.ForwardMode.DECODE,
            block_tables=block_tables,
        )
        md = backend.forward_metadata
        # before = 8 -> page slot 1 (row 2); after = 9 -> page slot 2 (row 3).
        self.assertEqual(md.state_in_blocks_by_group["linear_attention"].tolist(), [2])
        self.assertEqual(md.state_out_blocks_by_group["linear_attention"].tolist(), [3])
        # Decode builds no host boundary tuple; only extend batches carry one.
        self.assertIsNone(md.cu_extend_seq_lens_cpu)
        self.assertIsNone(md.query_start_loc_int64)
        self.assertIsNone(md.conv_prefill_metadata)

    def test_extend_metadata(self):
        torch = self.torch
        backend = self.backend
        backend.init_forward_metadata(
            bs=1,
            num_extends=1,
            req_pool_indices=torch.tensor([0], dtype=torch.int32),
            seq_lens=torch.tensor([8], dtype=torch.int32),
            forward_mode=self.ForwardMode.EXTEND,
            block_tables={
                "linear_attention": torch.tensor([[1, 2]], dtype=torch.int32)
            },
            **_extend_kwargs(
                torch,
                torch.tensor([8], dtype=torch.int32),
                torch.zeros(1, dtype=torch.int32),
                "cpu",
            ),
        )
        md = backend.forward_metadata
        self.assertEqual(md.state_in_blocks_by_group["linear_attention"].tolist(), [0])
        self.assertEqual(md.state_out_blocks_by_group["linear_attention"].tolist(), [2])
        # The metadata builds the host boundary tensor once, next to
        # query_start_loc, and keeps the raw lengths for the conv kernel.
        self.assertEqual(md.cu_extend_seq_lens_cpu.tolist(), [0, 8])
        self.assertEqual(md.cu_extend_seq_lens_cpu.dtype, self.torch.int64)
        self.assertFalse(md.cu_extend_seq_lens_cpu.is_cuda)
        self.assertEqual(md.extend_seq_lens_cpu.tolist(), [8])
        self.assertEqual(md.query_start_loc.tolist(), [0, 8])
        self.assertEqual(md.query_start_loc.dtype, torch.int32)
        self.assertEqual(md.query_start_loc_int64.dtype, torch.int64)
        self.assertEqual(md.query_start_loc_int64.tolist(), [0, 8])
        self.assertEqual(md.conv_prefill_metadata.batch_indices.tolist(), [0])
        self.assertEqual(md.conv_prefill_metadata.chunk_offsets.tolist(), [0])

    def test_mixed_metadata_pads_decode_rows(self):
        torch = self.torch
        backend = self.backend
        backend.init_forward_metadata(
            bs=2,
            num_extends=1,
            req_pool_indices=torch.tensor([0, 1], dtype=torch.int32),
            seq_lens=torch.tensor([5, 9], dtype=torch.int32),
            forward_mode=self.ForwardMode.MIXED,
            block_tables={
                "linear_attention": torch.tensor(
                    [[1, 2, 0], [0, 3, 4]], dtype=torch.int32
                )
            },
            **_extend_kwargs(
                torch,
                torch.tensor([5], dtype=torch.int32),
                torch.zeros(1, dtype=torch.int32),
                "cpu",
            ),
        )
        md = backend.forward_metadata
        # One extend row (5 tokens) plus one decode row padded to
        # spec_num_tokens (= 1): boundaries and the raw cat agree.
        self.assertEqual(md.cu_extend_seq_lens_cpu.tolist(), [0, 5, 6])
        self.assertEqual(md.extend_seq_lens_cpu.tolist(), [5, 1])
        self.assertEqual(md.query_start_loc.tolist(), [0, 5, 6])
        self.assertEqual(md.prefill_checkpoint_batch.rows.tolist(), [0])
        self.assertEqual(
            md.state_checkpoint_blocks_by_group["linear_attention"].tolist(), [1, -1]
        )
        self.assertEqual(md.query_start_loc_int64.tolist(), [0, 5, 6])
        self.assertEqual(md.conv_prefill_metadata.batch_indices.tolist(), [0, 1])
        self.assertEqual(md.conv_prefill_metadata.chunk_offsets.tolist(), [0, 0])

    def test_conv_metadata_is_owned_by_each_forward(self):
        torch = self.torch
        saved = []
        boundaries = []
        for lengths in ([9, 7], [7, 9]):
            lens = torch.tensor(lengths, dtype=torch.int32)
            self.backend.init_forward_metadata(
                bs=2,
                num_extends=2,
                req_pool_indices=torch.tensor([0, 1], dtype=torch.int32),
                seq_lens=lens,
                forward_mode=self.ForwardMode.EXTEND,
                block_tables={
                    "linear_attention": torch.tensor(
                        [[1, 2, 3], [4, 5, 6]], dtype=torch.int32
                    )
                },
                **_extend_kwargs(torch, lens, torch.zeros(2, dtype=torch.int32), "cpu"),
            )
            saved.append(self.backend.forward_metadata.conv_prefill_metadata)
            boundaries.append(self.backend.forward_metadata.query_start_loc_int64)
        # Same total tokens/programs, different request partition. Keep the
        # first forward alive while preparing the next, as overlap can do.
        self.assertIsNot(saved[0], saved[1])
        self.assertNotEqual(
            saved[0].batch_indices.data_ptr(), saved[1].batch_indices.data_ptr()
        )
        self.assertEqual(saved[0].batch_indices.tolist(), [0, 0, 1])
        self.assertEqual(saved[1].batch_indices.tolist(), [0, 1, 1])
        self.assertNotEqual(boundaries[0].data_ptr(), boundaries[1].data_ptr())
        self.assertEqual(boundaries[0].tolist(), [0, 9, 16])
        self.assertEqual(boundaries[1].tolist(), [0, 7, 16])

    def test_capture_replay_metadata(self):
        torch = self.torch
        backend = self.backend
        backend.init_cuda_graph_state(max_bs=2)
        backend.init_forward_metadata_capture_cuda_graph(
            bs=1,
            req_pool_indices=torch.tensor([0], dtype=torch.int32),
            seq_lens=torch.tensor([1], dtype=torch.int32),
            forward_mode=self.ForwardMode.DECODE,
        )
        md = backend.forward_metadata
        # Capture binds the persistent pad-filled buffers.
        self.assertEqual(md.state_in_blocks_by_group["linear_attention"].tolist(), [-1])
        self.assertEqual(
            md.state_out_blocks_by_group["linear_attention"].tolist(), [-1]
        )

        backend.refresh_decode_metadata(
            1,
            1,
            torch.tensor([0], dtype=torch.int32),
            torch.tensor([9], dtype=torch.int32),
            forward_mode=self.ForwardMode.DECODE,
            for_graph_replay=True,
            block_tables={
                "linear_attention": torch.tensor([[1, 2, 3]], dtype=torch.int32)
            },
        )
        md = backend.forward_metadata
        self.assertEqual(md.state_in_blocks_by_group["linear_attention"].tolist(), [2])
        self.assertEqual(md.state_out_blocks_by_group["linear_attention"].tolist(), [3])


class VerifyMetadataTest(unittest.TestCase):
    """Qwen's state groups use per-layer verify scratch."""

    def setUp(self):
        try:
            import torch

            from tokenspeed.runtime.execution.forward_batch_info import (
                ForwardMode,
            )
            from tokenspeed.runtime.layers.attention.backends.state.mamba import (  # noqa: E501
                MambaAttnBackend,
            )
        except (ImportError, ModuleNotFoundError) as exc:
            self.skipTest(f"needs torch + tokenspeed_kernel: {exc}")
        self.torch = torch
        self.ForwardMode = ForwardMode
        self.backend = MambaAttnBackend(
            *_mamba_config_pair(torch, heads=2, head_dim=2, spec_tokens=4)
        )
        self.state_buffers = {
            layer_id: (
                torch.zeros((8, 2, 3), dtype=torch.bfloat16),
                torch.zeros((8, 1, 2, 2), dtype=torch.float32),
            )
            for layer_id in range(2)
        }
        stub_pool = _ContractPool(
            4,
            {
                layer_id: (
                    f"linear_attention_{layer_id}",
                    *self.state_buffers[layer_id],
                )
                for layer_id in self.state_buffers
            },
        )
        self.backend.set_kv_pool(stub_pool)
        self.backend.init_cuda_graph_state(max_bs=2)

    def test_target_verify_uses_per_layer_scratch(self):
        torch = self.torch
        self.backend.refresh_decode_metadata(
            1,
            1,
            torch.tensor([1], dtype=torch.int32),
            torch.tensor([8], dtype=torch.int32),
            forward_mode=self.ForwardMode.DECODE,
            block_tables={
                "linear_attention_0": torch.tensor([[3, 4]], dtype=torch.int32),
                "linear_attention_1": torch.tensor([[5, 6]], dtype=torch.int32),
            },
        )

        metadata = self.backend.forward_metadata
        self.assertEqual(metadata.mamba_output_indices.tolist(), [[1, 2, 3, 4]])
        self.assertEqual(metadata.mamba_output_indices.dtype, torch.int32)
        self.assertEqual(
            metadata.state_in_blocks_by_group["linear_attention_0"].dtype,
            torch.int32,
        )
        self.assertEqual(
            metadata.state_in_blocks_by_group["linear_attention_0"].tolist(),
            [3],
        )
        self.assertEqual(
            metadata.state_in_blocks_by_group["linear_attention_1"].tolist(),
            [5],
        )
        self.assertEqual(set(self.backend._verify_scratch), {0, 1})
        for conv_scratch, state_scratch in self.backend._verify_scratch.values():
            self.assertEqual(conv_scratch.shape[0], 10)
            self.assertEqual(state_scratch.shape[0], 10)
        self.assertEqual(
            self.backend.preallocate_verify_workspace(2, 4),
            560,
        )

    def test_target_verify_reuses_graph_stable_scratch_base_rows(self):
        rows = self.backend._verify_scratch_base_rows(3, 4)
        grid = self.backend._verify_scratch_grid(3, 4)

        self.assertIs(rows, self.backend._verify_scratch_base_rows(3, 4))
        self.assertEqual(rows.dtype, self.torch.int32)
        self.assertEqual(rows.tolist(), [0, 5, 10])
        self.assertEqual(
            grid.tolist(),
            [[1, 2, 3, 4], [6, 7, 8, 9], [11, 12, 13, 14]],
        )


class GDNStatePagingGPUTest(unittest.TestCase):
    """MambaAttnBackend state paging vs the
    FLA chunk_gated_delta_rule oracle over the full contiguous sequence."""

    # Smallest fastpath parametrization: Hk = Hv = 16, D = 128 (sm100 GDN).
    H = 16
    D = 128
    P = 4  # state page size (tokens)
    PREFILL = 8
    DECODES = 3
    WIDTH = 4  # conv kernel width; state_len = WIDTH - 1

    def setUp(self):
        try:
            import torch
            from tokenspeed_kernel.ops.attention.gdn import flashinfer as gdn
            from tokenspeed_kernel.ops.attention.gdn import (
                gdn_replay_commit_supported,
            )

            from tokenspeed.runtime.execution.forward_batch_info import (
                ForwardMode,
            )
            from tokenspeed.runtime.layers.attention.backends.state.mamba import (  # noqa: E501
                MambaAttnBackend,
            )
        except (ImportError, ModuleNotFoundError) as exc:
            self.skipTest(f"needs torch + tokenspeed_kernel: {exc}")
        if not torch.cuda.is_available():
            self.skipTest("needs a CUDA device")
        self.torch = torch
        self.gdn = gdn
        self.ForwardMode = ForwardMode
        self.MambaAttnBackend = MambaAttnBackend
        self.gdn_replay_commit_supported = gdn_replay_commit_supported
        torch.manual_seed(0)

    def _make_backend(self, pool, spec_num_tokens, *, replay_ssm):
        """Bind the final pool geometry before initializing graph state."""
        torch = self.torch
        backend = self.MambaAttnBackend(
            *_mamba_config_pair(
                torch,
                heads=self.H,
                head_dim=self.D,
                spec_tokens=spec_num_tokens,
                device="cuda",
                replay_ssm=replay_ssm,
            )
        )
        backend.set_kv_pool(pool)
        self.assertTrue(backend.state_paging_active)
        backend.init_cuda_graph_state(max_bs=2)
        return backend

    def test_unaligned_prefill_checkpoint_matches_split_and_can_resume(self):
        """Use the published prefix with P != g, not just its metadata indices."""
        if not self.gdn.is_available():
            self.skipTest("sm100 GDN kernel unavailable")
        torch = self.torch
        h, d, prefix, total = self.H, self.D, 4, 7
        channels = 3 * h * d
        raw = torch.randn(total, channels, device="cuda", dtype=torch.bfloat16)
        a = torch.randn(total, h, device="cuda", dtype=torch.float32)
        b = torch.randn_like(a)
        common = dict(
            conv_weights=torch.randn(
                channels, self.WIDTH, device="cuda", dtype=torch.bfloat16
            )
            * 0.1,
            bias=torch.randn(channels, device="cuda", dtype=torch.bfloat16) * 0.1,
            activation="silu",
            key_dim=h * d,
            value_dim=h * d,
            attention_tp_size=1,
            head_k_dim=d,
            head_v_dim=d,
            A_log=torch.randn(h, device="cuda", dtype=torch.float32) * 0.1,
            dt_bias=torch.randn(h, device="cuda", dtype=torch.float32) * 0.1,
            layer_id=0,
        )

        def prefill(backend, before, after, row):
            backend.init_forward_metadata(
                bs=1,
                num_extends=1,
                req_pool_indices=torch.tensor([1], dtype=torch.int32, device="cuda"),
                seq_lens=torch.tensor([after], dtype=torch.int32, device="cuda"),
                forward_mode=self.ForwardMode.EXTEND,
                block_tables={
                    "linear_attention": torch.tensor(
                        [row], dtype=torch.int32, device="cuda"
                    )
                },
                **_extend_kwargs(
                    torch,
                    torch.tensor([after - before], dtype=torch.int32),
                    torch.tensor([before], dtype=torch.int32),
                    "cuda",
                ),
            )
            return backend.forward_extend(
                None,
                None,
                None,
                layer=None,
                out_cache_loc=None,
                token_to_kv_pool=backend.kv_pool,
                bs=1,
                forward_mode=self.ForwardMode.EXTEND,
                mixed_qkv=raw[before:after],
                a=a[before:after],
                b=b[before:after],
                seq_len=after - before,
                **common,
            )

        for granularity in (1, 2, 4):
            with self.subTest(state_granularity=granularity):
                states = []
                backends = []
                for _ in range(2):
                    conv = torch.zeros(
                        5, channels, self.WIDTH - 1, device="cuda", dtype=torch.bfloat16
                    )
                    recurrent = torch.zeros(
                        5, h, d, d, device="cuda", dtype=torch.float32
                    )
                    pool = _ContractPool(
                        granularity, {0: ("linear_attention", conv, recurrent)}
                    )
                    pool.arena.runtime_contract.prefix_granularity = prefix
                    backend = self._make_backend(
                        pool, spec_num_tokens=1, replay_ssm=False
                    )
                    states.append((conv, recurrent))
                    backends.append(backend)
                candidate, reference = backends
                row = [0] * ((total - 1) // granularity + 1)
                checkpoint_slot = (prefix - 1) // granularity
                row[checkpoint_slot] = 1
                row[-1] = 3
                full_output = prefill(candidate, 0, total, row)
                prefill(reference, 0, prefix, row)
                for actual, expected in zip(states[0], states[1]):
                    torch.testing.assert_close(
                        actual[1], expected[1], atol=1e-3, rtol=1e-3
                    )
                    self.assertGreater(actual[1].abs().max().item(), 0.0)
                    self.assertEqual(actual[0].abs().max().item(), 0.0)

                # Reuse the published checkpoint, writing a distinct continuation.
                saved = tuple(state[1].clone() for state in states[0])
                row[-1] = 4
                resumed = prefill(candidate, prefix, total, row)
                difference = (resumed.float() - full_output[:, prefix:].float()).abs()
                self.assertLess(difference.mean().item(), 1e-3)
                torch.testing.assert_close(
                    resumed, full_output[:, prefix:], atol=1e-1, rtol=1e-2
                )
                for state, snapshot in zip(states[0], saved):
                    torch.testing.assert_close(state[4], state[3], atol=1e-3, rtol=1e-2)
                    self.assertTrue(torch.equal(state[1], snapshot))

    def test_verify_scratch_seeds_conv_but_omits_replayed_ssm_state(self):
        torch = self.torch
        conv_dim = 3 * self.H * self.D
        conv_slab = torch.zeros(
            7, conv_dim, self.WIDTH - 1, device="cuda", dtype=torch.bfloat16
        )
        ssm_slab = torch.zeros(
            7, self.H, self.D, self.D, device="cuda", dtype=torch.float32
        )
        conv_slab[3].fill_(3)
        conv_slab[5].fill_(5)
        ssm_slab[3].fill_(3)
        ssm_slab[5].fill_(5)
        if not self.gdn_replay_commit_supported(torch.bfloat16):
            self.skipTest("GDN ReplaySSM kernel unavailable")
        pool = _ContractPool(self.P, {0: ("linear_attention", conv_slab, ssm_slab)})
        backend = self._make_backend(pool, spec_num_tokens=4, replay_ssm=True)
        backend.refresh_decode_metadata(
            2,
            2,
            torch.tensor([0, 1], dtype=torch.int32, device="cuda"),
            torch.tensor([8, 8], dtype=torch.int32, device="cuda"),
            forward_mode=self.ForwardMode.DECODE,
            block_tables={
                "linear_attention": torch.tensor(
                    [[3, 4], [5, 6]], dtype=torch.int32, device="cuda"
                )
            },
        )

        backend._seed_verify_scratch_batched(2, 4)
        conv_scratch, ssm_scratch = backend._verify_scratch[0]
        torch.cuda.synchronize()

        self.assertTrue(torch.equal(conv_scratch[0], conv_slab[3]))
        self.assertTrue(torch.equal(conv_scratch[5], conv_slab[5]))
        self.assertIsNone(ssm_scratch)

    def test_paged_states_match_fla_oracle(self):
        if not self.gdn.is_available():
            self.skipTest("sm100 GDN kernel unavailable")
        torch = self.torch
        ForwardMode = self.ForwardMode
        from tokenspeed_kernel.ops.attention.gdn._triton.chunk import (
            chunk_gated_delta_rule,
        )
        from tokenspeed_kernel.ops.attention.gdn.triton import (
            CAUSAL_CONV1D_BLOCK_M,
            build_causal_conv1d_prefill_metadata,
        )

        from tokenspeed.runtime.layers.attention.linear.causal_conv1d import (
            causal_conv1d_fn,
        )
        from tokenspeed.runtime.layers.attention.linear.gdn import fused_gdn_gating

        H, D, P = self.H, self.D, self.P
        total = self.PREFILL + self.DECODES  # 11 tokens
        key_dim = H * D
        value_dim = H * D
        conv_dim = 2 * key_dim + value_dim

        mixed_full = torch.randn(total, conv_dim, device="cuda", dtype=torch.bfloat16)
        conv_weights = (
            torch.randn(conv_dim, self.WIDTH, device="cuda", dtype=torch.bfloat16) * 0.1
        )
        bias = torch.randn(conv_dim, device="cuda", dtype=torch.bfloat16) * 0.1
        A_log = torch.randn(H, device="cuda", dtype=torch.float32) * 0.1
        dt_bias = torch.randn(H, device="cuda", dtype=torch.float32) * 0.1
        a_full = torch.randn(total, H, device="cuda", dtype=torch.float32)
        b_full = torch.randn(total, H, device="cuda", dtype=torch.float32)

        # ---- Oracle: one contiguous pass over all 11 tokens ----
        ref_conv_state = torch.zeros(
            1, conv_dim, self.WIDTH - 1, device="cuda", dtype=torch.bfloat16
        )
        query_start_loc = torch.tensor([0, total], dtype=torch.int32, device="cuda")
        conv_out = causal_conv1d_fn(
            mixed_full.transpose(0, 1),
            conv_weights,
            bias,
            activation="silu",
            conv_states=ref_conv_state,
            has_initial_state=torch.zeros(1, dtype=torch.bool, device="cuda"),
            cache_indices=torch.zeros(1, dtype=torch.int32, device="cuda"),
            query_start_loc=query_start_loc,
            prefill_metadata=build_causal_conv1d_prefill_metadata(
                query_start_loc,
                torch.tensor([total], dtype=torch.int32),
                CAUSAL_CONV1D_BLOCK_M,
            ),
        ).transpose(0, 1)[:total]
        q_ref, k_ref, v_ref = torch.split(
            conv_out, [key_dim, key_dim, value_dim], dim=-1
        )
        q_ref = q_ref.view(1, total, H, D)
        k_ref = k_ref.view(1, total, H, D)
        v_ref = v_ref.view(1, total, H, D)
        g_ref = fused_gdn_gating(A_log, a_full, dt_bias).view(1, total, H)
        beta_ref = b_full.sigmoid().to(torch.bfloat16).view(1, total, H)
        o_ref, st_ref = chunk_gated_delta_rule(
            q=q_ref,
            k=k_ref,
            v=v_ref,
            g=g_ref,
            beta=beta_ref,
            initial_state=torch.zeros(1, H, D, D, device="cuda", dtype=torch.float32),
            output_final_state=True,
            cu_seqlens=torch.tensor([0, total], device="cuda").long(),
            head_first=False,
            use_qk_l2norm_in_kernel=True,
        )

        # Page 0 is null; pages 1..N fill as the sequence grows.
        num_pages = total // P + 2  # null + pages 1..3
        conv_slab = torch.zeros(
            num_pages, conv_dim, self.WIDTH - 1, device="cuda", dtype=torch.bfloat16
        )
        ssm_slab = torch.zeros(num_pages, H, D, D, device="cuda", dtype=torch.float32)
        pool = _ContractPool(self.P, {0: ("linear_attention", conv_slab, ssm_slab)})
        backend = self._make_backend(pool, spec_num_tokens=1, replay_ssm=False)

        req_pool_indices = torch.tensor([1], dtype=torch.int32, device="cuda")
        common = dict(
            conv_weights=conv_weights,
            bias=bias,
            activation="silu",
            key_dim=key_dim,
            value_dim=value_dim,
            attention_tp_size=1,
            head_k_dim=D,
            head_v_dim=D,
            A_log=A_log,
            dt_bias=dt_bias,
            layer_id=0,
        )
        stub = backend.kv_pool

        # Prefill 8 tokens: in = null page 0, out = page 2 (slot 1).
        backend.init_forward_metadata(
            bs=1,
            num_extends=1,
            req_pool_indices=req_pool_indices,
            seq_lens=torch.tensor([self.PREFILL], dtype=torch.int32, device="cuda"),
            forward_mode=ForwardMode.EXTEND,
            block_tables={
                "linear_attention": torch.tensor(
                    [[1, 2]], dtype=torch.int32, device="cuda"
                )
            },
            **_extend_kwargs(
                torch,
                torch.tensor([self.PREFILL], dtype=torch.int32),
                torch.zeros(1, dtype=torch.int32),
                "cuda",
            ),
        )
        self.assertEqual(
            backend.forward_metadata.state_in_blocks_by_group[
                "linear_attention"
            ].tolist(),
            [0],
        )
        self.assertEqual(
            backend.forward_metadata.state_out_blocks_by_group[
                "linear_attention"
            ].tolist(),
            [2],
        )
        outputs = [
            backend.forward_extend(
                None,
                None,
                None,
                layer=None,
                out_cache_loc=None,
                token_to_kv_pool=stub,
                bs=1,
                forward_mode=ForwardMode.EXTEND,
                mixed_qkv=mixed_full[: self.PREFILL],
                a=a_full[: self.PREFILL],
                b=b_full[: self.PREFILL],
                seq_len=self.PREFILL,
                **common,
            )
        ]

        conv_page2_after_prefill = conv_slab[2].clone()
        ssm_page2_after_prefill = ssm_slab[2].clone()

        # 3 decode steps: page ids (in, out) = (2, 3), (3, 3), (3, 3).
        rows = torch.tensor([[1, 2, 3]], dtype=torch.int32, device="cuda")
        expected_pages = [(2, 3), (3, 3), (3, 3)]
        for i in range(self.DECODES):
            pos = self.PREFILL + i
            backend.refresh_decode_metadata(
                1,
                1,
                req_pool_indices,
                torch.tensor([pos + 1], dtype=torch.int32, device="cuda"),
                forward_mode=ForwardMode.DECODE,
                block_tables={"linear_attention": rows},
            )
            self.assertEqual(
                backend.forward_metadata.state_in_blocks_by_group[
                    "linear_attention"
                ].tolist(),
                [expected_pages[i][0]],
            )
            self.assertEqual(
                backend.forward_metadata.state_out_blocks_by_group[
                    "linear_attention"
                ].tolist(),
                [expected_pages[i][1]],
            )
            outputs.append(
                backend.forward_decode(
                    None,
                    None,
                    None,
                    layer=None,
                    out_cache_loc=None,
                    token_to_kv_pool=stub,
                    bs=1,
                    mixed_qkv=mixed_full[pos : pos + 1],
                    a=a_full[pos : pos + 1],
                    b=b_full[pos : pos + 1],
                    **common,
                )
            )

        paged_output = torch.cat(outputs, dim=1)
        self.assertEqual(tuple(paged_output.shape), tuple(o_ref.shape))

        # Fastpath-test tolerances: mean diff is the real bar, loose max.
        out_diff = (paged_output.float() - o_ref.float()).abs()
        self.assertLess(out_diff.mean().item(), 1e-3)
        self.assertTrue(
            torch.allclose(paged_output.float(), o_ref.float(), atol=1e-1, rtol=1e-2)
        )
        st_diff = (ssm_slab[3] - st_ref[0].float().transpose(-1, -2)).abs()
        self.assertLess(st_diff.mean().item(), 1e-3)

        # Null page 0 must never be written; page 2 (prefill's out page)
        # keeps the shared snapshot untouched by the boundary-crossing decode.
        self.assertEqual(conv_slab[0].abs().max().item(), 0.0)
        self.assertEqual(ssm_slab[0].abs().max().item(), 0.0)
        self.assertTrue(torch.equal(conv_slab[2], conv_page2_after_prefill))
        self.assertTrue(torch.equal(ssm_slab[2], ssm_page2_after_prefill))
        self.assertGreater(ssm_slab[2].abs().max().item(), 0.0)
        self.assertGreater(ssm_slab[3].abs().max().item(), 0.0)


class TritonCheckpointContinuationTest(unittest.TestCase):
    def test_batched_transposed_body_state_matches_full_scan(self):
        import torch

        if not torch.cuda.is_available():
            self.skipTest("GPU required")
        from unittest.mock import patch

        from tokenspeed_kernel.ops.attention.gdn.triton import (
            triton_gdn_chunk_prefill,
        )

        import tokenspeed.runtime.layers.attention.backends.state.mamba as mamba

        torch.manual_seed(7)
        h, d = 2, 128
        backend = object.__new__(mamba.MambaAttnBackend)
        for lengths in ([7, 7], [7, 3]):
            with self.subTest(lengths=lengths), patch.object(
                mamba, "gdn_chunk_prefill", triton_gdn_chunk_prefill
            ):
                n = sum(lengths)
                q, k, v = (
                    torch.randn(1, n, h, d, device="cuda", dtype=torch.bfloat16)
                    for _ in range(3)
                )
                initial = torch.zeros(2, h, d, d, device="cuda")
                slab = torch.zeros(6, h, d, d, device="cuda")
                kwargs = dict(
                    A_log=torch.zeros(h, device="cuda"),
                    dt_bias=torch.zeros(h, device="cuda"),
                    a=torch.randn(n, h, device="cuda", dtype=torch.bfloat16),
                    b=torch.randn(n, h, device="cuda", dtype=torch.bfloat16),
                    g_raw=None,
                    f_a_out=None,
                    f_b_weight=None,
                    beta_raw=None,
                    lower_bound=None,
                )
                bounds_cpu = torch.tensor([0, lengths[0], n], dtype=torch.int64)
                expected_out, expected_state = backend._prefill_scan(
                    q,
                    k,
                    v,
                    initial,
                    bounds_cpu.to(device="cuda", dtype=torch.int32),
                    seq_len=n,
                    num_real_tokens=n,
                    cu_seqlens_cpu=bounds_cpu,
                    **kwargs,
                )
                self.assertFalse(expected_state[0].is_contiguous())
                plan = mamba._build_prefill_checkpoint_batch(
                    torch.tensor(lengths, dtype=torch.int32),
                    torch.zeros(2, dtype=torch.int32),
                    2,
                    4,
                    "cuda",
                )
                checkpoint_blocks = torch.tensor(
                    [3, 4], device="cuda", dtype=torch.int32
                )
                actual_out, actual_state = backend._run_prefill_recurrent(
                    q,
                    k,
                    v,
                    initial,
                    slab,
                    checkpoint_blocks,
                    plan,
                    seq_len=n,
                    num_real_tokens=n,
                    **kwargs,
                )
                torch.testing.assert_close(
                    actual_out, expected_out, atol=2e-3, rtol=1e-2
                )
                torch.testing.assert_close(
                    actual_state, expected_state, atol=2e-3, rtol=1e-2
                )
                # The saved body state must also be reusable independently.
                for row in plan.rows.cpu().tolist():
                    start = int(bounds_cpu[row])
                    body_kwargs = dict(kwargs)
                    for name in ("a", "b"):
                        body_kwargs[name] = kwargs[name][start : start + 4]
                    prefix_bounds = torch.tensor([0, 4], dtype=torch.int64)
                    _, prefix_state = backend._prefill_scan(
                        q[:, start : start + 4],
                        k[:, start : start + 4],
                        v[:, start : start + 4],
                        initial[row : row + 1],
                        prefix_bounds.to(device="cuda", dtype=torch.int32),
                        seq_len=4,
                        num_real_tokens=4,
                        cu_seqlens_cpu=prefix_bounds,
                        **body_kwargs,
                    )
                    torch.testing.assert_close(
                        slab[3 + row], prefix_state[0], atol=2e-3, rtol=1e-2
                    )


if __name__ == "__main__":
    unittest.main()
