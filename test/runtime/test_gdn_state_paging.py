"""GDN two-slot state paging.

compute_state_block_indices resolves each request's (in, out) state blocks
from the "linear_attention" two-slot block table; the GPU test drives
MambaAttnBackend (chunked prefill + decodes over paged state slabs) against
the FLA chunk_gated_delta_rule oracle run once over the full contiguous
sequence.
"""

from __future__ import annotations

import os
import sys
from types import SimpleNamespace
from unittest.mock import patch

import pytest

# CI Registration (parsed via AST, runtime no-op)
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from ci_system.ci_register import register_cuda_ci

register_cuda_ci(est_time=90, suite="runtime-1gpu")


def test_state_helpers_use_shared_implementations():
    """Backend-local definitions must not shadow the shared state helpers."""
    from tokenspeed_kernel.ops.attention.gdn.triton import prepare_prefill_state_inputs

    from tokenspeed.runtime.layers.attention.backends.state import checkpoint, mamba

    assert mamba._prepare_cache_prefill_state_inputs is prepare_prefill_state_inputs
    assert mamba.compute_state_block_indices is checkpoint.compute_state_block_indices


class _ContractPool:
    def __init__(self, components):
        # The arena publishes the contract; a view only names its arena.
        contract = SimpleNamespace(
            prefix_granularity=64,
            group_specs=tuple(
                SimpleNamespace(group_id=group_id, attn_kind="state")
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
    draft_tree=False,
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
        draft_tree=draft_tree,
    )
    config = AttnConfig(
        device=device,
        dtype=torch.bfloat16,
        kv_cache_dtype=torch.bfloat16,
        kv_cache_quant_method="none",
        context_len=4096,
        max_bs=max_bs,
        max_resident_requests=max_bs,
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
        extend_replay_lens_cpu=torch.zeros_like(extend_prefix_lens_cpu),
        extend_prompt_lens_cpu=extend_prefix_lens_cpu + extend_seq_lens_cpu,
        extend_with_prefix=bool(extend_prefix_lens_cpu.any()),
        query_shard=None,
    )


def _no_extends(torch, device):
    """The extend bundle of a decode-mode call: no extend rows."""
    empty = torch.zeros(0, dtype=torch.int32)
    return _extend_kwargs(torch, empty, empty, device)


class TestComputeStatePageIndicesTest:
    """CPU-only contract tests for the two-slot resolve."""

    def setup_method(self):
        try:
            import torch

            from tokenspeed.runtime.layers.attention.backends.state.checkpoint import (  # noqa: E501
                compute_state_block_indices,
            )
        except (ImportError, ModuleNotFoundError) as exc:
            pytest.skip(f"needs torch + tokenspeed_kernel: {exc}")
        self.torch = torch
        self.fn = compute_state_block_indices

    def _run(self, rows, before, validate=True):
        torch = self.torch
        return self.fn(
            torch.tensor(rows, dtype=torch.int32),
            torch.tensor(before, dtype=torch.int32),
            validate=validate,
            group_id="linear_attention",
        )

    def test_prefill_reads_the_input_slot_and_writes_the_live_slot(self):
        # A hit (or a rotated previous chunk) in slot 0, a fresh slot 1.
        state_in, state_out = self._run([[7, 9]], [4])
        assert (state_in.tolist()) == ([7])
        assert (state_out.tolist()) == ([9])

    def test_cold_prefill_reads_the_zero_state(self):
        state_in, state_out = self._run([[0, 7]], [0])
        assert (state_in.tolist()) == ([0])
        assert (state_out.tolist()) == ([7])

    def test_first_decode_reads_the_rotated_prompt_end_state(self):
        state_in, state_out = self._run([[5, 8]], [8])
        assert (state_in.tolist()) == ([5])
        assert (state_out.tolist()) == ([8])

    def test_later_decodes_evolve_the_live_slot_in_place(self):
        state_in, state_out = self._run([[0, 8]], [9])
        assert (state_in.tolist()) == ([8])
        assert (state_out.tolist()) == ([8])

    def test_wider_rows_are_read_at_the_two_slots(self):
        # The wire pads every group's rows to the batch's column count.
        state_in, state_out = self._run([[0, 8, -1, -1]], [9])
        assert (state_in.tolist()) == ([8])
        assert (state_out.tolist()) == ([8])

    def test_batch_mixed(self):
        rows = [[7, 9], [0, 22], [0, 31], [3, 5]]
        state_in, state_out = self._run(rows, [4, 5, 0, 8])
        assert (state_in.tolist()) == ([7, 22, 0, 3])
        assert (state_out.tolist()) == ([9, 22, 31, 5])

    def test_live_slot_hole_raises(self):
        with pytest.raises((ValueError)):
            self._run([[7, 0]], [4])

    def test_live_slot_pad_raises(self):
        with pytest.raises((ValueError)):
            self._run([[7, -1]], [4])

    def test_in_place_pad_with_history_raises(self):
        # slot 0 null, slot 1 a pad: a request with history has no block to
        # read, and a silent zero-state resume must fail loud.
        with pytest.raises((ValueError)):
            self._run([[0, -1]], [5])

    def test_padding_rows_read_and_write_nothing(self):
        # Both slots -1: a graph padding row or a request the device stopped
        # behind an in-flight forward. Kernels skip negative blocks.
        state_in, state_out = self._run([[-1, -1], [0, 8], [-1, -1]], [9, 9, 4])
        assert (state_in.tolist()) == ([-1, 8, -1])
        assert (state_out.tolist()) == ([-1, 8, -1])

    def test_claimed_checkpoint_equal_to_live_block_raises(self):
        with pytest.raises((ValueError), match=("same block")):
            self._run([[7, 7]], [4])

    def test_duplicate_out_pages_raise(self):
        with pytest.raises((ValueError), match=("unique")):
            self._run([[7, 9], [0, 9]], [4, 0])

    def test_validate_off_masks_guards(self):
        state_in, state_out = self._run([[0, 0]], [0], validate=False)
        assert (state_in.tolist()) == ([0])
        assert (state_out.tolist()) == ([0])

    @pytest.mark.parametrize(
        ("rows", "num_blocks", "expected_fragment"),
        [
            ([[10, 4]], 10, "row 0, slot 0 is 10"),
            ([[0, 10]], 10, "row 0, slot 1 is 10"),
            ([[0, 4], [2, 13]], 10, "row 1, slot 1 is 13"),
            ([[0, 4, 10]], 10, "row 0, slot 2 is 10"),
            ([[-2, 4]], 10, "row 0, slot 0 is -2"),
            ([[0, 4, -3]], 10, "row 0, slot 2 is -3"),
        ],
    )
    def test_num_blocks_bound_rejects_out_of_range_entries(
        self, rows, num_blocks, expected_fragment
    ):
        torch = self.torch
        with pytest.raises(ValueError, match=expected_fragment):
            self.fn(
                torch.tensor(rows, dtype=torch.int32),
                torch.tensor([4] * len(rows), dtype=torch.int32),
                validate=True,
                group_id="linear_attention",
                num_blocks=num_blocks,
            )

    def test_num_blocks_bound_accepts_valid_pad_null_and_in_range_entries(self):
        torch = self.torch
        state_in, state_out = self.fn(
            torch.tensor(
                [[-1, -1, -1], [-1, -1, -1], [0, 9, -1], [3, 8, 9]], dtype=torch.int32
            ),
            torch.tensor([0, 4, 0, 64], dtype=torch.int32),
            validate=True,
            group_id="linear_attention",
            num_blocks=10,
        )
        assert state_in.tolist() == [0, -1, 0, 3]
        assert state_out.tolist() == [-1, -1, 9, 8]

    def test_validate_state_block_table_bounds_rejects_invalid_shapes_and_sizes(self):
        from tokenspeed.runtime.layers.attention.backends.state.checkpoint import (
            validate_state_block_table_bounds,
        )

        torch = self.torch
        with pytest.raises(ValueError, match="must be positive"):
            validate_state_block_table_bounds(
                torch.tensor([[0, 1]], dtype=torch.int32),
                num_blocks=0,
                group_id="linear_attention",
            )
        with pytest.raises(ValueError, match="must be 2D"):
            validate_state_block_table_bounds(
                torch.tensor([0, 1], dtype=torch.int32),
                num_blocks=4,
                group_id="linear_attention",
            )


class TestCacheContractStateBlocksTest:
    """The backend resolves every declared group once per batch."""

    def setup_method(self):
        try:
            import torch

            from tokenspeed.runtime.layers.attention.backends.state.mamba import (
                MambaAttnBackend,
            )
        except (ImportError, ModuleNotFoundError) as exc:
            pytest.skip(f"needs torch + tokenspeed_kernel: {exc}")
        self.torch = torch
        self.backend = object.__new__(MambaAttnBackend)
        self.backend._state_group_ids = ("linear_attention_0", "linear_attention_1")
        self.backend.pad_slot_id = -1

    def test_every_group_is_resolved_from_its_own_table(self):
        torch = self.torch
        state_in, state_out = self.backend._cache_contract_state_blocks(
            torch.tensor([4, 0], dtype=torch.int32),
            {
                "linear_attention_0": torch.tensor([[7, 8], [0, 9]], dtype=torch.int32),
                "linear_attention_1": torch.tensor(
                    [[17, 18], [0, 19]], dtype=torch.int32
                ),
            },
            validate=True,
        )
        assert (state_in["linear_attention_0"].tolist()) == ([7, 0])
        assert (state_out["linear_attention_0"].tolist()) == ([8, 9])
        assert (state_in["linear_attention_1"].tolist()) == ([17, 0])
        assert (state_out["linear_attention_1"].tolist()) == ([18, 19])

    def test_a_missing_group_table_is_a_delivery_bug(self):
        torch = self.torch
        with pytest.raises((RuntimeError), match=("missing state group")):
            self.backend._cache_contract_state_blocks(
                torch.tensor([4], dtype=torch.int32),
                {"linear_attention_0": torch.tensor([[7, 8]], dtype=torch.int32)},
                validate=True,
            )


class TestCacheContractMetadataTest:
    """Every metadata entry point resolves state through the cache contract."""

    def setup_method(self):
        try:
            import torch

            from tokenspeed.runtime.execution.forward_batch_info import (
                ForwardMode,
            )
            from tokenspeed.runtime.layers.attention.backends.state.mamba import (  # noqa: E501
                MambaAttnBackend,
            )
        except (ImportError, ModuleNotFoundError) as exc:
            pytest.skip(f"needs torch + tokenspeed_kernel: {exc}")
        self.torch = torch
        self.ForwardMode = ForwardMode
        backend = MambaAttnBackend(
            *_mamba_config_pair(torch, heads=16, head_dim=128, spec_tokens=1)
        )
        stub_pool = _ContractPool(
            {0: ("linear_attention", torch.zeros(8, 3), torch.zeros(8, 5))},
        )
        backend.set_kv_pool(stub_pool)
        assert backend.state_paging_active
        self.backend = backend

    def test_decode_metadata(self):
        torch = self.torch
        backend = self.backend
        req_pool_indices = torch.tensor([0], dtype=torch.int32)
        seq_lens = torch.tensor([9], dtype=torch.int32)
        # The first decode after prompt end: the prompt-end state was rotated
        # into slot 0, slot 1 is fresh.
        block_tables = {"linear_attention": torch.tensor([[2, 3]], dtype=torch.int32)}
        # Decode metadata is the refresh's alone; a DECODE init is a contract
        # violation on every node, the state backend included.
        with pytest.raises((RuntimeError), match=("refresh_decode_metadata")):
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
        assert (md.state_in_blocks_by_group["linear_attention"].tolist()) == ([2])
        assert (md.state_out_blocks_by_group["linear_attention"].tolist()) == ([3])
        # Decode builds no host boundary tuple; only extend batches carry one.
        assert (md.cu_extend_seq_lens_cpu) is None
        assert (md.query_start_loc_int64) is None
        assert (md.conv_prefill_metadata) is None

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
                "linear_attention": torch.tensor([[0, 2]], dtype=torch.int32)
            },
            **_extend_kwargs(
                torch,
                torch.tensor([8], dtype=torch.int32),
                torch.zeros(1, dtype=torch.int32),
                "cpu",
            ),
        )
        md = backend.forward_metadata
        assert (md.state_in_blocks_by_group["linear_attention"].tolist()) == ([0])
        assert (md.state_out_blocks_by_group["linear_attention"].tolist()) == ([2])
        # The metadata builds the host boundary tensor once, next to
        # query_start_loc, and keeps the raw lengths for the conv kernel.
        assert (md.cu_extend_seq_lens_cpu.tolist()) == ([0, 8])
        assert (md.cu_extend_seq_lens_cpu.dtype) == (self.torch.int64)
        assert not (md.cu_extend_seq_lens_cpu.is_cuda)
        assert (md.extend_seq_lens_cpu.tolist()) == ([8])
        assert (md.query_start_loc.tolist()) == ([0, 8])
        assert (md.query_start_loc.dtype) == (torch.int32)
        assert (md.query_start_loc_int64.dtype) == (torch.int64)
        assert (md.query_start_loc_int64.tolist()) == ([0, 8])
        assert (md.conv_prefill_metadata.batch_indices.tolist()) == ([0])
        assert (md.conv_prefill_metadata.chunk_offsets.tolist()) == ([0])

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
                "linear_attention": torch.tensor([[0, 1], [0, 4]], dtype=torch.int32)
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
        assert (md.cu_extend_seq_lens_cpu.tolist()) == ([0, 5, 6])
        assert (md.extend_seq_lens_cpu.tolist()) == ([5, 1])
        assert (md.query_start_loc.tolist()) == ([0, 5, 6])
        # The cold extend row reads the zero state; the decode row runs in place.
        assert (md.state_in_blocks_by_group["linear_attention"].tolist()) == ([0, 4])
        assert (md.state_out_blocks_by_group["linear_attention"].tolist()) == ([1, 4])
        assert (md.query_start_loc_int64.tolist()) == ([0, 5, 6])
        assert (md.conv_prefill_metadata.batch_indices.tolist()) == ([0, 1])
        assert (md.conv_prefill_metadata.chunk_offsets.tolist()) == ([0, 0])

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
                        [[0, 3], [0, 6]], dtype=torch.int32
                    )
                },
                **_extend_kwargs(torch, lens, torch.zeros(2, dtype=torch.int32), "cpu"),
            )
            saved.append(self.backend.forward_metadata.conv_prefill_metadata)
            boundaries.append(self.backend.forward_metadata.query_start_loc_int64)
        # Same total tokens/programs, different request partition. Keep the
        # first forward alive while preparing the next, as overlap can do.
        assert (saved[0]) is not (saved[1])
        assert (saved[0].batch_indices.data_ptr()) != (
            saved[1].batch_indices.data_ptr()
        )
        assert (saved[0].batch_indices.tolist()) == ([0, 0, 1])
        assert (saved[1].batch_indices.tolist()) == ([0, 1, 1])
        assert (boundaries[0].data_ptr()) != (boundaries[1].data_ptr())
        assert (boundaries[0].tolist()) == ([0, 9, 16])
        assert (boundaries[1].tolist()) == ([0, 7, 16])

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
        assert (md.state_in_blocks_by_group["linear_attention"].tolist()) == ([-1])
        assert (md.state_out_blocks_by_group["linear_attention"].tolist()) == ([-1])

        backend.refresh_decode_metadata(
            1,
            1,
            torch.tensor([0], dtype=torch.int32),
            torch.tensor([9], dtype=torch.int32),
            forward_mode=self.ForwardMode.DECODE,
            for_graph_replay=True,
            block_tables={
                "linear_attention": torch.tensor([[0, 3]], dtype=torch.int32)
            },
        )
        md = backend.forward_metadata
        assert (md.state_in_blocks_by_group["linear_attention"].tolist()) == ([3])
        assert (md.state_out_blocks_by_group["linear_attention"].tolist()) == ([3])


class TestVerifyMetadataTest:
    """Qwen's state groups use per-layer verify scratch."""

    def setup_method(self):
        try:
            import torch

            from tokenspeed.runtime.execution.forward_batch_info import (
                ForwardMode,
            )
            from tokenspeed.runtime.layers.attention.backends.state.mamba import (  # noqa: E501
                MambaAttnBackend,
            )
        except (ImportError, ModuleNotFoundError) as exc:
            pytest.skip(f"needs torch + tokenspeed_kernel: {exc}")
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
        # The first verify after prompt end reads the rotated checkpoint in slot 0.
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
        assert (metadata.mamba_output_indices.tolist()) == ([[1, 2, 3, 4]])
        assert (metadata.mamba_output_indices.dtype) == (torch.int32)
        assert (metadata.state_in_blocks_by_group["linear_attention_0"].dtype) == (
            torch.int32
        )
        assert (metadata.state_in_blocks_by_group["linear_attention_0"].tolist()) == (
            [3]
        )
        assert (metadata.state_in_blocks_by_group["linear_attention_1"].tolist()) == (
            [5]
        )
        assert (set(self.backend._verify_scratch)) == ({0, 1})
        for conv_scratch, state_scratch in self.backend._verify_scratch.values():
            assert (conv_scratch.shape[0]) == (10)
            assert (state_scratch.shape[0]) == (10)
        assert (self.backend.preallocate_verify_workspace(2, 4)) == (560)

    def test_target_verify_reuses_graph_stable_scratch_base_rows(self):
        rows = self.backend._verify_scratch_base_rows(3, 4)
        grid = self.backend._verify_scratch_grid(3, 4)

        assert (rows) is (self.backend._verify_scratch_base_rows(3, 4))
        assert (rows.dtype) == (self.torch.int32)
        assert (rows.tolist()) == ([0, 5, 10])
        assert (grid.tolist()) == ([[1, 2, 3, 4], [6, 7, 8, 9], [11, 12, 13, 14]])

    def test_verify_commit_resolves_pages_with_fused_group_kernel(self):
        torch = self.torch
        from tokenspeed.runtime.layers.attention.backends.state import (
            mamba as mamba_module,
        )

        # Row 0 is a first verify (slot 0 holds the checkpoint), row 1 a later
        # one evolving slot 1 in place; both commit into slot 1.
        block_tables = {
            "linear_attention_0": torch.tensor([[12, 11], [0, 23]], dtype=torch.int32),
            "linear_attention_1": torch.tensor([[32, 31], [0, 43]], dtype=torch.int32),
        }
        self.backend.refresh_decode_metadata(
            2,
            2,
            torch.tensor([0, 1], dtype=torch.int32),
            torch.tensor([7, 10], dtype=torch.int32),
            forward_mode=self.ForwardMode.DECODE,
            block_tables=block_tables,
        )
        accepted = torch.tensor([0, 5], dtype=torch.int32)
        resolve_calls = []
        copy_calls = []
        original_commit = mamba_module.commit_state_pages

        def counted_commit(*args, **kwargs):
            resolve_calls.append((args, kwargs))
            return original_commit(*args, **kwargs)

        def recorded_copy(*args, **kwargs):
            copy_calls.append((args, kwargs))

        def reference_rows(
            steps, pages, src, dst, *, verify_width, num_layers, group_indices
        ):
            base = torch.arange(steps.numel(), dtype=torch.int32) * (verify_width + 1)
            src.copy_((base + steps.clamp(1, verify_width)).repeat(num_layers))
            selected = pages.index_select(0, group_indices).reshape(-1)
            dst.copy_(torch.where(selected > 0, selected, -1))

        with (
            patch.object(mamba_module, "commit_state_pages", counted_commit),
            patch.object(mamba_module, "copy_state_rows", recorded_copy),
            patch.object(mamba_module, "state_verify_commit_rows", reference_rows),
        ):
            self.backend.commit_verified_state(accepted, accepted_path=None)

        assert (len(resolve_calls)) == (2)
        assert (
            [
                kwargs["pages_out"][kwargs["out_row"]].tolist()
                for _, kwargs in resolve_calls
            ]
        ) == ([[11, 23], [31, 43]])
        for _, kwargs in resolve_calls:
            assert (kwargs["batch_size"]) == (2)
            assert (kwargs["draft_tokens"]) == (4)
            assert (kwargs["pages_out"].dtype) == (torch.int32)
            assert (kwargs["steps_out"].tolist()) == ([1, 4])

        assert (len(copy_calls)) == (2)
        for args, _ in copy_calls:
            assert (args[2].dtype) == (torch.int32)
            assert (args[3].dtype) == (torch.int32)
            assert (args[2].tolist()) == ([1, 9, 1, 9])
            assert (args[3].tolist()) == ([11, 23, 31, 43])
        assert (self.backend._verify_commit_ctx) is None


class TestVerifyCommitGPUTest:
    def test_grouped_commit_copies_and_replay_share_fused_rows(self, subtests):
        import torch

        if not torch.cuda.is_available():
            pytest.skip("needs a CUDA device")
        from tokenspeed_kernel.ops.kvcache.triton import (
            _state_verify_commit_rows_kernel,
        )

        from tokenspeed.runtime.execution.forward_batch_info import ForwardMode
        from tokenspeed.runtime.layers.attention.backends.state import mamba

        width, capacity = 3, 4
        # Sorted layers select groups [1, 0, 1], exercising both reordering
        # and a group shared by multiple non-adjacent layers.
        layer_groups = {0: "group0", 2: "group1", 5: "group0"}
        # Row 0 verifies right after prompt end (slot 0 checkpoint, fresh slot
        # 1); rows 1 and 2 evolve slot 1 in place. Row 1 of group1 has no live
        # block, so its commit is masked.
        tables = {
            "group0": torch.tensor(
                [[2, 1], [0, 5], [0, 9]], dtype=torch.int32, device="cuda"
            ),
            "group1": torch.tensor(
                [[12, 11], [0, -1], [0, 15]], dtype=torch.int32, device="cuda"
            ),
        }
        seq_lens = torch.tensor([6, 7, 11, 3], dtype=torch.int32, device="cuda")
        accepted = torch.tensor([0, 2, 9], dtype=torch.int32, device="cuda")
        expected_pages = {"group0": [1, 5, 9], "group1": [11, -1, 15]}
        for replay_ssm in (False, True):
            for live_bs in (1, 3):
                with subtests.test(replay_ssm=replay_ssm, live_bs=live_bs):
                    components = {
                        layer: (
                            group,
                            torch.full(
                                (16, 6, 3), -7, dtype=torch.bfloat16, device="cuda"
                            ),
                            torch.full(
                                (16, 1, 2, 2), -7, dtype=torch.float32, device="cuda"
                            ),
                        )
                        for layer, group in layer_groups.items()
                    }
                    pool = _ContractPool(components)
                    pool.arena.runtime_contract.group_specs = tuple(
                        reversed(pool.arena.runtime_contract.group_specs)
                    )
                    backend = mamba.MambaAttnBackend(
                        *_mamba_config_pair(
                            torch,
                            heads=1,
                            head_dim=2,
                            spec_tokens=width,
                            max_bs=capacity,
                            device="cuda",
                            replay_ssm=replay_ssm,
                        )
                    )
                    backend.set_kv_pool(pool)
                    backend.init_cuda_graph_state(capacity)
                    backend.refresh_decode_metadata(
                        capacity,
                        live_bs,
                        torch.arange(capacity, dtype=torch.int32, device="cuda"),
                        seq_lens,
                        forward_mode=ForwardMode.DECODE,
                        block_tables=tables,
                    )
                    read_pages = backend._verify_commit_ctx[2]
                    for layer, scratches in backend._verify_scratch.items():
                        for tensor in scratches:
                            if tensor is not None:
                                values = torch.arange(
                                    tensor.shape[0], device="cuda", dtype=tensor.dtype
                                ) + 100 * (layer + 1)
                                tensor.copy_(
                                    values.view(
                                        -1, *([1] * (tensor.ndim - 1))
                                    ).expand_as(tensor)
                                )
                    # Warm pointer tables outside profiling, as forward seeding does.
                    backend._verify_copy_tables_get()
                    # Count real launches directly: ROCm profiler traces can omit events.
                    with (
                        patch.object(mamba, "gdn_replay_commit") as replay,
                        patch.object(
                            _state_verify_commit_rows_kernel,
                            "run",
                            wraps=_state_verify_commit_rows_kernel.run,
                        ) as commit_rows_launch,
                    ):
                        with torch.profiler.profile(
                            activities=[
                                torch.profiler.ProfilerActivity.CPU,
                                torch.profiler.ProfilerActivity.CUDA,
                            ]
                        ) as profile:
                            backend.commit_verified_state(
                                accepted[:live_bs], accepted_path=None
                            )
                            torch.cuda.synchronize()
                    events = profile.events()
                    gpu_kernels = [
                        e.name
                        for e in events
                        if e.device_type == torch.autograd.DeviceType.CUDA
                    ]
                    commit_rows_launch.assert_called_once()
                    # Replay still assembles its separate read indices; the
                    # scratch-copy path needs no PyTorch index arithmetic.
                    if not replay_ssm:
                        assert not (
                            {
                                "aten::index_select",
                                "aten::add",
                                "aten::repeat",
                                "aten::stack",
                            }.intersection(e.name for e in events)
                        )
                        assert not any("elementwise" in name for name in gpu_kernels)
                    source_rows = [1, 6, 11][:live_bs]
                    for layer, group in layer_groups.items():
                        for kind, name in enumerate(("conv_state", "recurrent_state")):
                            actual = pool.get_component(layer, name)
                            expected = torch.full_like(actual, -7)
                            if kind == 0 or not replay_ssm:
                                for src, dst in zip(
                                    source_rows,
                                    expected_pages[group][:live_bs],
                                    strict=True,
                                ):
                                    if dst > 0:
                                        expected[dst] = backend._verify_scratch[layer][
                                            kind
                                        ][src]
                            torch.testing.assert_close(actual, expected, rtol=0, atol=0)
                    if replay_ssm:
                        replay.assert_called_once()
                        kwargs = replay.call_args.kwargs
                        torch.testing.assert_close(
                            kwargs["accepted_length"],
                            accepted[:live_bs].clamp(1, width),
                        )
                        expected_write = torch.tensor(
                            [
                                expected_pages[g][:live_bs]
                                for g in layer_groups.values()
                            ],
                            dtype=torch.int32,
                            device="cuda",
                        )
                        torch.testing.assert_close(
                            kwargs["write_indices"], expected_write
                        )
                        torch.testing.assert_close(
                            kwargs["read_indices"],
                            torch.stack(
                                [read_pages[g][:live_bs] for g in layer_groups.values()]
                            ),
                        )
                    else:
                        replay.assert_not_called()
                    assert backend._verify_commit_ctx is None


class TestGDNStatePagingGPUTest:
    """MambaAttnBackend state paging vs the
    FLA chunk_gated_delta_rule oracle over the full contiguous sequence."""

    # Smallest fastpath parametrization: Hk = Hv = 16, D = 128 (sm100 GDN).
    H = 16
    D = 128
    PREFILL = 8
    DECODES = 3
    WIDTH = 4  # conv kernel width; state_len = WIDTH - 1

    def setup_method(self):
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
            pytest.skip(f"needs torch + tokenspeed_kernel: {exc}")
        if not torch.cuda.is_available():
            pytest.skip("needs a CUDA device")
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
        assert backend.state_paging_active
        backend.init_cuda_graph_state(max_bs=2)
        return backend

    def test_chunked_prefill_resumes_from_the_live_block(self, subtests):
        """A chunk's final state in slot 1 is the next chunk's input in slot 0.

        The scheduler rotates the live block into slot 0 and grants a fresh
        slot 1 between chunks; the published block is never written again.
        The chunked delta rule rounds its intra-chunk intermediates to bf16,
        so a boundary inside a kernel chunk regroups the reduction: outputs
        agree to the oracle test's tolerance and the state to a mean bound,
        not bitwise.
        """
        from tokenspeed_kernel.ops.attention.gdn.triton import (
            triton_gdn_chunk_prefill,
        )

        import tokenspeed.runtime.layers.attention.backends.state.mamba as mamba

        torch = self.torch
        h, d, total = self.H, self.D, 11
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

        # Boundaries off any kernel-chunk multiple: 7, 4 and 10 of 11 tokens.
        for boundary in (7, 4, 10):
            with (
                subtests.test(boundary=boundary),
                patch.object(mamba, "gdn_chunk_prefill", triton_gdn_chunk_prefill),
            ):
                slabs = []
                backends = []
                for _ in range(2):
                    conv = torch.zeros(
                        5, channels, self.WIDTH - 1, device="cuda", dtype=torch.bfloat16
                    )
                    recurrent = torch.zeros(
                        5, h, d, d, device="cuda", dtype=torch.float32
                    )
                    pool = _ContractPool({0: ("linear_attention", conv, recurrent)})
                    backends.append(
                        self._make_backend(pool, spec_num_tokens=1, replay_ssm=False)
                    )
                    slabs.append((conv, recurrent))
                candidate, reference = backends
                full_output = prefill(reference, 0, total, [0, 3])

                first = prefill(candidate, 0, boundary, [0, 1])
                checkpoint = tuple(slab[1].clone() for slab in slabs[0])
                assert (checkpoint[1].abs().max().item()) > (0.0)
                resumed = prefill(candidate, boundary, total, [1, 3])

                torch.testing.assert_close(
                    torch.cat((first, resumed), dim=1),
                    full_output,
                    atol=1e-1,
                    rtol=1e-2,
                )
                difference = (resumed.float() - full_output[:, boundary:].float()).abs()
                assert (difference.mean().item()) < (1e-3)
                for candidate_slab, reference_slab in zip(slabs[0], slabs[1]):
                    state_difference = (
                        candidate_slab[3].float() - reference_slab[3].float()
                    ).abs()
                    assert (state_difference.mean().item()) < (1e-3)
                    assert (state_difference.max().item()) < (5e-2)
                # The checkpoint read by the second chunk is immutable, and the
                # null block is never written.
                for slab, saved in zip(slabs[0], checkpoint):
                    assert torch.equal(slab[1], saved)
                    assert (slab[0].abs().max().item()) == (0.0)

    def test_prefill_conv_writing_qkv_matches_conv_then_split(self):
        """Two prefill chunks give the same outputs and states bit for bit
        whether the convolution writes q, k and v or the split pass does."""
        from tokenspeed_kernel.platform import current_platform

        import tokenspeed.runtime.layers.attention.backends.state.mamba as mamba

        torch = self.torch
        h, d, total, boundary = self.H, self.D, 45, 19
        channels = 3 * h * d
        # Every platform but Hopper takes the direct write in serving.
        assert mamba._prefill_conv_writes_qkv(h * d, h * d) is not (
            current_platform().is_hopper
        )
        raw = torch.randn(total, channels, device="cuda", dtype=torch.bfloat16)
        a = torch.randn(total, h, device="cuda", dtype=torch.float32)
        b = torch.randn_like(a)
        weights = torch.randn(channels, self.WIDTH, device="cuda") * 0.1
        common = dict(
            conv_weights=weights.to(torch.bfloat16),
            bias=None,
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

        def run(direct):
            conv = torch.zeros(
                5, channels, self.WIDTH - 1, device="cuda", dtype=torch.bfloat16
            )
            recurrent = torch.zeros(5, h, d, d, device="cuda", dtype=torch.float32)
            pool = _ContractPool({0: ("linear_attention", conv, recurrent)})
            backend = self._make_backend(pool, spec_num_tokens=1, replay_ssm=False)
            outputs = []
            for before, after, row in (
                (0, boundary, [0, 1]),
                (boundary, total, [1, 3]),
            ):
                backend.init_forward_metadata(
                    bs=1,
                    num_extends=1,
                    req_pool_indices=torch.tensor(
                        [1], dtype=torch.int32, device="cuda"
                    ),
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
                with patch.object(mamba, "_prefill_conv_writes_qkv", lambda *_: direct):
                    outputs.append(
                        backend.forward_extend(
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
                    )
            return outputs, conv, recurrent

        (direct, direct_conv, direct_state), (split, split_conv, split_state) = (
            run(True),
            run(False),
        )
        for actual, expected in zip(direct, split, strict=True):
            assert torch.equal(actual, expected)
        assert torch.equal(direct_conv, split_conv)
        assert torch.equal(direct_state, split_state)

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
            pytest.skip("GDN ReplaySSM kernel unavailable")
        pool = _ContractPool({0: ("linear_attention", conv_slab, ssm_slab)})
        backend = self._make_backend(pool, spec_num_tokens=4, replay_ssm=True)
        backend.refresh_decode_metadata(
            2,
            2,
            torch.tensor([0, 1], dtype=torch.int32, device="cuda"),
            torch.tensor([8, 8], dtype=torch.int32, device="cuda"),
            forward_mode=self.ForwardMode.DECODE,
            block_tables={
                "linear_attention": torch.tensor(
                    [[3, 4], [0, 5]], dtype=torch.int32, device="cuda"
                )
            },
        )

        backend._seed_verify_scratch_batched(2, 4)
        conv_scratch, ssm_scratch = backend._verify_scratch[0]
        torch.cuda.synchronize()

        assert torch.equal(conv_scratch[0], conv_slab[3])
        assert torch.equal(conv_scratch[5], conv_slab[5])
        assert (ssm_scratch) is None

    def test_gating_reads_a_row_strided_decay_view(self):
        """The prefill scan gates over a row-strided view of the projection."""
        torch = self.torch
        from tokenspeed.runtime.layers.attention.linear.gdn import fused_gdn_gating

        heads = 12
        projection = torch.randn(37, 4 * heads, device="cuda", dtype=torch.bfloat16)
        a = projection[:, 3 * heads :]
        A_log = torch.randn(heads, device="cuda")
        dt_bias = torch.randn(heads, device="cuda")
        expected = fused_gdn_gating(A_log, a.contiguous(), dt_bias)
        torch.testing.assert_close(
            fused_gdn_gating(A_log, a, dt_bias), expected, rtol=0, atol=0
        )

    def test_paged_states_match_fla_oracle(self):
        if not self.gdn.is_available():
            pytest.skip("sm100 GDN kernel unavailable")
        torch = self.torch
        ForwardMode = self.ForwardMode
        from tokenspeed_kernel.ops.attention.gdn._triton.chunk import (
            chunk_gated_delta_rule,
        )
        from tokenspeed_kernel.ops.attention.gdn.triton import (
            CAUSAL_CONV1D_BLOCK_M,
            build_causal_conv1d_prefill_metadata,
        )
        from tokenspeed_kernel.ops.conv import (
            causal_conv1d_fn,
        )

        from tokenspeed.runtime.layers.attention.linear.gdn import fused_gdn_gating

        H, D = self.H, self.D
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

        # Block 0 is null; block 2 holds the prompt-end state, block 3 the
        # live decode state.
        num_pages = 4
        conv_slab = torch.zeros(
            num_pages, conv_dim, self.WIDTH - 1, device="cuda", dtype=torch.bfloat16
        )
        ssm_slab = torch.zeros(num_pages, H, D, D, device="cuda", dtype=torch.float32)
        pool = _ContractPool({0: ("linear_attention", conv_slab, ssm_slab)})
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

        # Prefill 8 tokens from the zero state (slot 0 null) into block 2.
        backend.init_forward_metadata(
            bs=1,
            num_extends=1,
            req_pool_indices=req_pool_indices,
            seq_lens=torch.tensor([self.PREFILL], dtype=torch.int32, device="cuda"),
            forward_mode=ForwardMode.EXTEND,
            block_tables={
                "linear_attention": torch.tensor(
                    [[0, 2]], dtype=torch.int32, device="cuda"
                )
            },
            **_extend_kwargs(
                torch,
                torch.tensor([self.PREFILL], dtype=torch.int32),
                torch.zeros(1, dtype=torch.int32),
                "cuda",
            ),
        )
        assert (
            backend.forward_metadata.state_in_blocks_by_group[
                "linear_attention"
            ].tolist()
        ) == ([0])
        assert (
            backend.forward_metadata.state_out_blocks_by_group[
                "linear_attention"
            ].tolist()
        ) == ([2])
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

        # 3 decode steps: the first reads the rotated prompt-end state (slot 0
        # = block 2) and writes the fresh block 3; the scheduler then releases
        # slot 0 and every later step evolves block 3 in place.
        tables = [
            torch.tensor([[2, 3]], dtype=torch.int32, device="cuda"),
            torch.tensor([[0, 3]], dtype=torch.int32, device="cuda"),
            torch.tensor([[0, 3]], dtype=torch.int32, device="cuda"),
        ]
        expected_pages = [(2, 3), (3, 3), (3, 3)]
        for i in range(self.DECODES):
            pos = self.PREFILL + i
            backend.refresh_decode_metadata(
                1,
                1,
                req_pool_indices,
                torch.tensor([pos + 1], dtype=torch.int32, device="cuda"),
                forward_mode=ForwardMode.DECODE,
                block_tables={"linear_attention": tables[i]},
            )
            assert (
                backend.forward_metadata.state_in_blocks_by_group[
                    "linear_attention"
                ].tolist()
            ) == ([expected_pages[i][0]])
            assert (
                backend.forward_metadata.state_out_blocks_by_group[
                    "linear_attention"
                ].tolist()
            ) == ([expected_pages[i][1]])
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
        assert (tuple(paged_output.shape)) == (tuple(o_ref.shape))

        # Fastpath-test tolerances: mean diff is the real bar, loose max.
        out_diff = (paged_output.float() - o_ref.float()).abs()
        assert (out_diff.mean().item()) < (1e-3)
        assert torch.allclose(paged_output.float(), o_ref.float(), atol=1e-1, rtol=1e-2)
        st_diff = (ssm_slab[3] - st_ref[0].float().transpose(-1, -2)).abs()
        assert (st_diff.mean().item()) < (1e-3)

        # Null block 0 must never be written; block 2 (the published
        # prompt-end checkpoint) is untouched by the first decode that read it.
        assert (conv_slab[0].abs().max().item()) == (0.0)
        assert (ssm_slab[0].abs().max().item()) == (0.0)
        assert torch.equal(conv_slab[2], conv_page2_after_prefill)
        assert torch.equal(ssm_slab[2], ssm_page2_after_prefill)
        assert (ssm_slab[2].abs().max().item()) > (0.0)
        assert (ssm_slab[3].abs().max().item()) > (0.0)


def _poison(torch, tensor):
    """Fill ``tensor`` with a NaN of distinctive payload: a stray read leaks NaN
    into the outputs and a stray write changes the bytes."""
    view_dtype, bits = {
        torch.bfloat16: (torch.int16, 0x7FA5),
        torch.float32: (torch.int32, 0x7FA5A5A5),
    }[tensor.dtype]
    tensor.view(view_dtype).fill_(bits)


def _same_bytes(torch, actual, expected):
    """Bitwise equality, which also tells NaN payloads and signed zeros apart."""
    return actual.dtype == expected.dtype and torch.equal(
        actual.contiguous().view(torch.uint8), expected.contiguous().view(torch.uint8)
    )


@pytest.mark.parametrize("family", ["gdn", "kda-fla", "kda-cutedsl"])
class TestPrefillStateGuardPagesGPUTest:
    """GDN and KDA prefill over state slabs fenced by NaN guard blocks.

    Each conv and recurrent slab is a view into a larger backing. The blocks
    around the slab and the slab's null block 0 hold NaNs with a distinctive
    payload, and every out-of-range block id under test (``-7`` to ``-1`` and
    ``PAGES`` to ``PAGES + 3``) lands in that guard memory. A row whose source
    lies outside ``(0, PAGES)`` starts from the zero state, and a destination
    outside it is never written. Every row with a live block must match, bit
    for bit, the same batch on a plain slab whose rows carry what the runtime
    delivers instead: source 0 for a row without history, and -1 for a block
    a row lacks.

    The convolution writes the live block itself and skips only -1, so an
    out-of-range block other than -1 is exercised as an in-forward checkpoint
    block, which only the state-row store writes.
    """

    PAGES = 10
    LEAD = 7  # guard blocks before the slab: ids -7 to -1
    TRAIL = 4  # guard blocks after the slab: ids PAGES to PAGES + 3
    SNAPSHOTS = (2, 3)  # published input states; every other block is stale
    GROUP = "linear_attention"
    RESUMED_PREFIX = 64

    def setup_method(self):
        from importlib.util import find_spec

        # Only the optional packages may skip; a broken runtime import fails.
        missing = [
            name for name in ("torch", "tokenspeed_kernel") if not find_spec(name)
        ]
        if missing:
            pytest.skip(f"needs {' + '.join(missing)}")
        import torch

        if not torch.cuda.is_available():
            pytest.skip("needs a CUDA device")
        from tokenspeed.runtime.execution.forward_batch_info import ForwardMode

        self.torch = torch
        self.ForwardMode = ForwardMode
        torch.manual_seed(0)

    def _gdn(self):
        """GDN: (heads, make_backend(conv, ssm), per-token inputs, kwargs)."""
        from tokenspeed_kernel.ops.attention.gdn import flashinfer as flashinfer_gdn
        from tokenspeed_kernel.platform import ArchVersion, current_platform

        from tokenspeed.runtime.layers.attention.backends.state.gdn import (
            GdnAttnBackend,
        )

        torch = self.torch
        heads = 16
        # Of the GDN chunk-prefill kernels only FlashInfer's (NVIDIA SM90 to
        # SM10.7 with CUDA 13) stores in-forward checkpoints; the Triton scan
        # that runs elsewhere, e.g. on ROCm, does not.
        platform = current_platform()
        checkpoint_kernel_available = (
            platform.is_nvidia
            and ArchVersion(9, 0) <= platform.arch_version <= ArchVersion(10, 7)
            and flashinfer_gdn.is_supported(128, torch.bfloat16, heads, heads)
        )

        def make(conv, ssm):
            backend = GdnAttnBackend(
                *_mamba_config_pair(torch, heads=heads, head_dim=128, device="cuda"),
                writes_state_checkpoints_in_forward=True,
                defer_recurrent_state=False,
            )
            backend.set_kv_pool(_ContractPool({0: (self.GROUP, conv, ssm)}))
            backend.init_cuda_graph_state(max_bs=8)
            return backend

        def tokens(total):
            a = torch.randn(total, heads, device="cuda")
            return dict(
                mixed_qkv=torch.randn(
                    total, 3 * heads * 128, device="cuda", dtype=torch.bfloat16
                ),
                a=a,
                b=torch.randn_like(a),
            )

        return (
            heads,
            make,
            tokens,
            dict(dt_bias=torch.randn(heads, device="cuda") * 0.1),
            checkpoint_kernel_available,
        )

    def _kda(self, solution):
        """KDA: (heads, make_backend(conv, ssm), per-token inputs, kwargs)."""
        if solution == "fla":
            from importlib.util import find_spec

            if find_spec("fla") is None:
                pytest.skip("flash-linear-attention (fla) required")
        else:
            from tokenspeed_kernel.ops.attention.kda.cute_dsl import (
                cutedsl_kda_supported,
            )

            if not cutedsl_kda_supported():
                pytest.skip("requires the native CuteDSL KDA prefill package")
        from tokenspeed.runtime.execution.request_slots import RequestSlotLayout
        from tokenspeed.runtime.layers.attention.backends.state.kda import (
            KdaAttnBackend,
        )
        from tokenspeed.runtime.layers.attention.configs.base import AttnConfig
        from tokenspeed.runtime.layers.attention.configs.mha import MHAConfig

        torch = self.torch
        heads = 4
        # Both KDA solutions store in-forward checkpoints once their
        # prerequisites above pass.
        checkpoint_kernel_available = True
        spec = MHAConfig(
            num_attention_heads=heads, num_kv_heads=heads, head_dim=128, attn_tp_size=1
        )
        config = AttnConfig(
            device="cuda",
            dtype=torch.bfloat16,
            kv_cache_dtype=torch.bfloat16,
            kv_cache_quant_method="none",
            context_len=4096,
            max_bs=8,
            max_resident_requests=8,
            is_draft=False,
            speculative_num_draft_tokens=1,
            components=(spec,),
        )

        def make(conv, ssm):
            backend = KdaAttnBackend(
                config, spec, enable_prefill_graph=False, kda_backend=solution
            )
            backend.set_kv_pool(_ContractPool({0: (self.GROUP, conv, ssm)}))
            backend.bind_request_stop_flags(
                torch.zeros(
                    RequestSlotLayout(8).capacity, dtype=torch.bool, device="cuda"
                ),
                torch.zeros(8, dtype=torch.int32, device="cuda"),
            )
            backend.init_cuda_graph_state(max_bs=8)
            # Serving allocates the request-slot records up front, so the
            # prefill also folds them into the staged inputs.
            backend.preallocate_verify_workspace(8, 1)
            return backend

        def tokens(total):
            return dict(
                mixed_qkv=torch.randn(
                    total, 3 * heads * 128, device="cuda", dtype=torch.bfloat16
                ),
                g_raw=torch.randn(
                    total, heads * 128, device="cuda", dtype=torch.bfloat16
                )
                * 0.5,
                beta_raw=torch.randn(total, heads, device="cuda", dtype=torch.bfloat16),
            )

        kwargs = dict(
            dt_bias=torch.randn(heads * 128, device="cuda") * 0.1,
            a=None,
            b=None,
            lower_bound=-5.0,
        )
        return heads, make, tokens, kwargs, checkpoint_kernel_available

    def _family(self, family):
        """(heads, make_backend(conv, ssm), per-token inputs, fixed kwargs, ckpt_avail)."""
        torch = self.torch
        heads, make, tokens, fixed, checkpoint_kernel_available = (
            self._gdn()
            if family == "gdn"
            else self._kda({"kda-fla": "fla", "kda-cutedsl": "cutedsl_kda"}[family])
        )
        channels = 3 * heads * 128
        fixed.update(
            conv_weights=torch.randn(channels, 4, device="cuda", dtype=torch.bfloat16)
            * 0.1,
            bias=torch.randn(channels, device="cuda", dtype=torch.bfloat16) * 0.1,
            activation="silu",
            key_dim=heads * 128,
            value_dim=heads * 128,
            attention_tp_size=1,
            head_k_dim=128,
            head_v_dim=128,
            A_log=torch.randn(heads, device="cuda") * 0.1,
            layer_id=0,
        )
        return heads, make, tokens, fixed, checkpoint_kernel_available

    def _prefill(
        self,
        make,
        slabs,
        rows,
        prefix,
        lens,
        checkpoint,
        inputs,
        fixed,
        checkpoint_kernel_available,
    ):
        """One EXTEND forward of ``rows`` (source, destination[, checkpoint]).

        Returns the ``[tokens, -1]`` output and the recurrent inputs and
        history flags the backend staged for the scan.
        """
        torch = self.torch
        backend = make(*slabs)
        # A platform without a kernel that stores in-forward checkpoints for
        # this family has nothing to check. Where one runs, a backend that
        # stops advertising them fails.
        if (
            checkpoint
            and not checkpoint_kernel_available
            and not backend.stores_in_forward_state_checkpoints
        ):
            pytest.skip("no chunk-prefill kernel stores in-forward checkpoints here")
        assert not checkpoint or backend.stores_in_forward_state_checkpoints
        bs = len(rows)
        lens_cpu = torch.tensor(lens, dtype=torch.int32)
        prefix_cpu = torch.tensor(prefix, dtype=torch.int32)
        # Intentionally omit block_tables_cpu and disable TOKENSPEED_CACHE_DEBUG
        # so these tests directly exercise the kernel-side arena guard.
        with patch.dict(os.environ, {"TOKENSPEED_CACHE_DEBUG": "0"}):
            backend.init_forward_metadata(
                bs=bs,
                num_extends=bs,
                req_pool_indices=torch.arange(
                    1, bs + 1, dtype=torch.int32, device="cuda"
                ),
                seq_lens=(prefix_cpu + lens_cpu).cuda(),
                forward_mode=self.ForwardMode.EXTEND,
                block_tables={
                    self.GROUP: torch.tensor(rows, dtype=torch.int32, device="cuda")
                },
                extend_state_checkpoint_lens_cpu=(
                    torch.full((bs,), checkpoint, dtype=torch.int32)
                    if checkpoint
                    else None
                ),
                **_extend_kwargs(torch, lens_cpu, prefix_cpu, "cuda"),
            )
        # Pin the resolved blocks to the ids under test: the two-slot resolve
        # falls back to the live block when the input slot is not positive.
        metadata = backend.forward_metadata
        for resolved, column in (
            (metadata.state_in_blocks_by_group, 0),
            (metadata.state_out_blocks_by_group, 1),
        ):
            resolved[self.GROUP].copy_(
                torch.tensor([row[column] for row in rows], dtype=torch.int32)
            )
        staged = []
        prepare = backend._prepare_prefill_state_inputs

        def record_staging(*args, **kwargs):
            result = prepare(*args, **kwargs)
            staged.append(tuple(tensor.clone() for tensor in result))
            return result

        with patch.object(backend, "_prepare_prefill_state_inputs", record_staging):
            out = backend.forward_extend(
                None,
                None,
                None,
                layer=None,
                out_cache_loc=None,
                token_to_kv_pool=backend.kv_pool,
                bs=bs,
                forward_mode=self.ForwardMode.EXTEND,
                seq_len=sum(lens),
                **{name: tensor.clone() for name, tensor in inputs.items()},
                **fixed,
            )
        torch.cuda.synchronize()
        ((recurrent, has_initial_state),) = staged
        return out.reshape(sum(lens), -1), recurrent, has_initial_state

    def _check(self, family, rows, lens, checkpoint=None):
        torch = self.torch
        pages, lead = self.PAGES, self.LEAD
        heads, make, tokens, fixed, checkpoint_kernel_available = self._family(family)

        def valid(block):
            return 0 < block < pages

        backings, plain = [], []
        for shape, dtype in (
            ((3 * heads * 128, 3), torch.bfloat16),
            ((heads, 128, 128), torch.float32),
        ):
            backing = torch.empty(
                lead + pages + self.TRAIL, *shape, device="cuda", dtype=dtype
            )
            _poison(torch, backing)
            slab = torch.zeros(pages, *shape, device="cuda", dtype=dtype)
            for block in self.SNAPSHOTS:
                slab[block].normal_(0, 0.1)
                backing[lead + block].copy_(slab[block])
            backings.append(backing)
            plain.append(slab)
        before = [backing.clone() for backing in backings]
        # What the runtime delivers instead: a row without history reads
        # source 0 and a row without a live block is a padding row.
        delivered = [
            ((src if valid(src) else 0, dst) if valid(dst) else (-1, -1))
            + tuple(block if valid(block) else -1 for block in ckpt)
            for src, dst, *ckpt in rows
        ]
        prefix = [self.RESUMED_PREFIX if row[0] else 0 for row in rows]
        inputs = tokens(sum(lens))
        out, recurrent, has_initial_state = self._prefill(
            make,
            [backing[lead : lead + pages] for backing in backings],
            rows,
            prefix,
            lens,
            checkpoint,
            inputs,
            fixed,
            checkpoint_kernel_available,
        )
        expected_out, _, _ = self._prefill(
            make,
            plain,
            delivered,
            prefix,
            lens,
            checkpoint,
            inputs,
            fixed,
            checkpoint_kernel_available,
        )

        # Only an in-range source is read; any other starts from exact zeros.
        assert has_initial_state.tolist() == [valid(row[0]) for row in rows]
        for i, (src, *_) in enumerate(rows):
            staged = (
                before[1][lead + src].to(recurrent.dtype)
                if valid(src)
                else torch.zeros_like(recurrent[i])
            )
            assert _same_bytes(torch, recurrent[i], staged), f"row {i} input state"

        bounds = [0, *torch.tensor(lens).cumsum(0).tolist()]
        for i, (_, dst, *_) in enumerate(rows):
            if valid(dst):
                rows_out = slice(bounds[i], bounds[i + 1])
                assert torch.isfinite(expected_out[rows_out].float()).all()
                assert _same_bytes(
                    torch, out[rows_out], expected_out[rows_out]
                ), f"row {i} output"

        # Only in-range destinations change, to exactly what the plain slab
        # holds; the guards, the null block and every other block keep
        # their bytes.
        written = {block for _, *blocks in rows for block in blocks if valid(block)}
        for backing, start, slab in zip(backings, before, plain, strict=True):
            expected = start.clone()
            for block in written:
                expected[lead + block] = slab[block]
            changed = (
                (backing.view(torch.uint8) != expected.view(torch.uint8))
                .flatten(1)
                .any(1)
            )
            assert (
                not changed.any()
            ), f"blocks {(changed.nonzero().flatten() - lead).tolist()} differ"

    def test_resumed_row_without_a_live_block_reads_its_source_only(self, family):
        # A resumed request whose group has no live block this forward
        # (source 3, destination -1), beside a cold and a resumed row.
        self._check(family, rows=[(0, 4), (2, 5), (3, -1)], lens=[37, 29, 21])

    def test_out_of_range_sources_start_from_the_zero_state(self, family):
        pages = self.PAGES
        self._check(
            family,
            rows=[(0, 4), (2, 5), (pages, 6), (pages + 3, 7), (-7, 8)],
            lens=[37, 29, 21, 18, 26],
        )

    def test_out_of_range_checkpoint_blocks_store_nothing(self, family):
        # Each row stores an in-forward checkpoint 64 tokens into its chunk;
        # only the first row's checkpoint block is a page of the slab.
        pages = self.PAGES
        self._check(
            family,
            rows=[(0, 4, 9), (2, 5, 0), (0, 6, pages), (3, 7, pages + 3), (0, 8, -7)],
            lens=[96, 80, 72, 100, 70],
            checkpoint=64,
        )


class TestReplayStateTapeGPUTest:
    """Decode replay refresh over many state groups and more rows than one tape block."""

    def _backend(self, num_groups, bs):
        import torch

        from tokenspeed.runtime.layers.attention.backends.state.mamba import (
            MambaAttnBackend,
        )

        backend = MambaAttnBackend(
            *_mamba_config_pair(torch, heads=2, head_dim=2, max_bs=bs, device="cuda")
        )
        backend.set_kv_pool(
            _ContractPool(
                {
                    layer_id: (
                        f"state_{layer_id}",
                        torch.zeros(2, 3, device="cuda"),
                        torch.zeros(2, 5, device="cuda"),
                    )
                    for layer_id in range(num_groups)
                },
            )
        )
        backend.init_cuda_graph_state(max_bs=bs)
        return backend

    def _refresh(self, backend, num_groups, bs, real_bs):
        import torch

        from tokenspeed.runtime.execution.forward_batch_info import ForwardMode

        seq_lens = torch.randint(2, 4096, (bs,), dtype=torch.int32, device="cuda")
        tables = {}
        for g in range(num_groups):
            # Padded to a wider column count, as the wire pads every group's
            # rows; a third of the rows are first decodes (slot 0 held).
            rows = torch.full((real_bs, 5), -1, dtype=torch.int32, device="cuda")
            rows[:, :2] = torch.randint(
                1, 1000, (real_bs, 2), dtype=torch.int32, device="cuda"
            )
            rows[1::3, 0] = 0
            tables[f"state_{g}"] = rows
        slots = torch.arange(bs, dtype=torch.int32, device="cuda")
        previous = torch.cuda.get_sync_debug_mode()
        try:
            torch.cuda.set_sync_debug_mode("error")
            backend.refresh_decode_metadata(
                bs,
                real_bs,
                slots,
                seq_lens,
                forward_mode=ForwardMode.DECODE,
                for_graph_replay=True,
                block_tables=tables,
            )
        finally:
            torch.cuda.set_sync_debug_mode(previous)
        torch.cuda.synchronize()
        md = backend.forward_metadata
        for gid, rows in tables.items():
            ref_in = torch.where(rows[:, 0] > 0, rows[:, 0], rows[:, 1])
            state_in = md.state_in_blocks_by_group[gid]
            state_out = md.state_out_blocks_by_group[gid]
            _pytest_expr_1 = torch.equal(state_in[:real_bs], ref_in)
            _pytest_msg_2 = gid
            assert _pytest_expr_1, _pytest_msg_2
            _pytest_expr_3 = torch.equal(state_out[:real_bs], rows[:, 1])
            _pytest_msg_4 = gid
            assert _pytest_expr_3, _pytest_msg_4
            _pytest_expr_5 = (state_in[real_bs:] == -1).all()
            _pytest_msg_6 = gid
            assert _pytest_expr_5, _pytest_msg_6
            _pytest_expr_7 = (state_out[real_bs:] == -1).all()
            _pytest_msg_8 = gid
            assert _pytest_expr_7, _pytest_msg_8

    def test_tape_and_eager_fallback_match_the_two_slot_reference(self):
        import torch

        if not torch.cuda.is_available():
            pytest.skip("GPU required")
        torch.manual_seed(0)
        for num_groups, taped in ((1, True), (5, True), (8, True), (9, False)):
            backend = self._backend(num_groups, bs=160)
            self._refresh(backend, num_groups, bs=160, real_bs=150)
            _pytest_first_9 = bool(backend._replay_state_tapes)
            _pytest_second_10 = taped
            _pytest_msg_11 = num_groups
            assert _pytest_first_9 == _pytest_second_10, _pytest_msg_11

    def test_rebuilt_graph_state_drops_tapes_bound_to_the_old_buffers(self):
        import torch

        if not torch.cuda.is_available():
            pytest.skip("GPU required")
        torch.manual_seed(1)
        backend = self._backend(5, bs=8)
        self._refresh(backend, 5, bs=8, real_bs=6)
        previous_buffers = [
            values[7]
            for buffers in (backend.state_in_by_group, backend.state_out_by_group)
            for values in buffers.values()
        ]
        backend.init_cuda_graph_state(max_bs=8)
        for buffer in previous_buffers:
            buffer.fill_(-77)
        # The refresh must write the rebuilt buffers a recaptured graph reads.
        self._refresh(backend, 5, bs=8, real_bs=6)
        assert all(bool((buffer == -77).all()) for buffer in previous_buffers)

    def test_first_refresh_at_every_supported_size_is_already_prepared(self):
        import torch

        if not torch.cuda.is_available():
            pytest.skip("GPU required")
        backend = self._backend(5, bs=8)
        # A graph ladder may warm only 1/2/4. The first eager or replay refresh
        # at any other supported size must not upload a tape description.
        for bs in (1, 2, 3, 4, 5, 6, 7, 8):
            self._refresh(backend, 5, bs=bs, real_bs=max(0, bs - 1))


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, *sys.argv[1:]]))


class TestHostStateBlockBoundsTest:
    """Always-on zero-GPU-sync host bounds check in MambaAttnBackend.init_forward_metadata."""

    GROUP = "linear_attention"

    def setup_method(self):
        from importlib.util import find_spec

        missing = [
            name for name in ("torch", "tokenspeed_kernel") if not find_spec(name)
        ]
        if missing:
            pytest.skip(f"needs {' + '.join(missing)}")
        import torch

        from tokenspeed.runtime.execution.forward_batch_info import ForwardMode
        from tokenspeed.runtime.layers.attention.backends.state.mamba import (
            MambaAttnBackend,
        )

        self.torch = torch
        self.ForwardMode = ForwardMode
        self.MambaAttnBackend = MambaAttnBackend

    def _make_backend(self, *, device="cpu", num_blocks=10, ssm_blocks=None):
        torch = self.torch
        ssm_blocks = num_blocks if ssm_blocks is None else ssm_blocks
        backend = self.MambaAttnBackend(
            *_mamba_config_pair(torch, heads=2, head_dim=4, max_bs=8, device=device)
        )
        backend.set_kv_pool(
            _ContractPool(
                {
                    0: (
                        self.GROUP,
                        torch.zeros(num_blocks, 24, 3, device=device),
                        torch.zeros(ssm_blocks, 2, 4, 4, device=device),
                    )
                }
            )
        )
        return backend

    def _init_extend(
        self,
        backend,
        rows,
        *,
        device="cpu",
        pass_cpu=True,
        checkpoint=None,
        debug=False,
    ):
        torch = self.torch
        bs = len(rows)
        lens_cpu = torch.full((bs,), 64, dtype=torch.int32)
        prefix_cpu = torch.tensor(
            [64 if row[0] > 0 else 0 for row in rows], dtype=torch.int32
        )
        table_cpu = torch.tensor(rows, dtype=torch.int32)
        table_dev = table_cpu.to(device=device)
        kwargs = _extend_kwargs(torch, lens_cpu, prefix_cpu, device)
        if pass_cpu:
            kwargs["block_tables_cpu"] = {self.GROUP: table_cpu}
        with patch.dict(os.environ, {"TOKENSPEED_CACHE_DEBUG": "1" if debug else "0"}):
            backend.init_forward_metadata(
                bs=bs,
                num_extends=bs,
                req_pool_indices=torch.arange(
                    1, bs + 1, dtype=torch.int32, device=device
                ),
                seq_lens=(prefix_cpu + lens_cpu).to(device=device),
                forward_mode=self.ForwardMode.EXTEND,
                block_tables={self.GROUP: table_dev},
                extend_state_checkpoint_lens_cpu=(
                    torch.full((bs,), checkpoint, dtype=torch.int32)
                    if checkpoint
                    else None
                ),
                **kwargs,
            )

    @pytest.mark.parametrize(
        ("rows", "expected_fragment"),
        [
            ([[10, 4]], "row 0, slot 0 is 10"),
            ([[0, 10]], "row 0, slot 1 is 10"),
            ([[0, 4], [2, 15]], "row 1, slot 1 is 15"),
            ([[0, 4, 10]], "row 0, slot 2 is 10"),
            ([[-2, 4]], "row 0, slot 0 is -2"),
            ([[0, -5]], "row 0, slot 1 is -5"),
        ],
    )
    def test_host_table_oob_fails_fast_without_cache_debug(
        self, rows, expected_fragment
    ):
        backend = self._make_backend(device="cpu", num_blocks=10)
        with pytest.raises(ValueError, match=expected_fragment):
            self._init_extend(backend, rows, device="cpu", pass_cpu=True, debug=False)

    def test_host_table_uses_min_component_capacity(self):
        backend = self._make_backend(device="cpu", num_blocks=12, ssm_blocks=8)
        with pytest.raises(ValueError, match=r"slot 1 is 8.*pool has 8 blocks"):
            self._init_extend(
                backend, [[0, 8]], device="cpu", pass_cpu=True, debug=False
            )

    def test_host_table_valid_pad_null_and_in_range_pass(self):
        backend = self._make_backend(device="cpu", num_blocks=10)
        self._init_extend(
            backend,
            [[-1, -1], [0, 4], [3, 9]],
            device="cpu",
            pass_cpu=True,
            debug=False,
        )
        md = backend.forward_metadata
        assert md.state_in_blocks_by_group[self.GROUP].tolist() == [0, 0, 3]
        assert md.state_out_blocks_by_group[self.GROUP].tolist() == [-1, 4, 9]

    def test_cache_debug_without_host_table_also_rejects_oob(self):
        backend = self._make_backend(device="cpu", num_blocks=10)
        with pytest.raises(ValueError, match="row 0, slot 1 is 10"):
            self._init_extend(
                backend, [[0, 10]], device="cpu", pass_cpu=False, debug=True
            )

    def test_host_table_rejects_short_rows(self):
        torch = self.torch
        backend = self._make_backend(device="cpu", num_blocks=10)
        lens_cpu = torch.tensor([64, 64], dtype=torch.int32)
        prefix_cpu = torch.tensor([0, 0], dtype=torch.int32)
        with patch.dict(os.environ, {"TOKENSPEED_CACHE_DEBUG": "0"}):
            with pytest.raises(ValueError, match="at least 2 rows"):
                backend.init_forward_metadata(
                    bs=2,
                    num_extends=2,
                    req_pool_indices=torch.tensor([1, 2], dtype=torch.int32),
                    seq_lens=lens_cpu,
                    forward_mode=self.ForwardMode.EXTEND,
                    block_tables={
                        self.GROUP: torch.tensor([[0, 4], [0, 5]], dtype=torch.int32)
                    },
                    block_tables_cpu={
                        self.GROUP: torch.tensor([[0, 4]], dtype=torch.int32)
                    },
                    **_extend_kwargs(torch, lens_cpu, prefix_cpu, "cpu"),
                )

    def test_host_table_rejects_shape_and_dtype_mismatch(self):
        torch = self.torch
        backend = self._make_backend(device="cpu", num_blocks=10)
        lens_cpu = torch.tensor([64], dtype=torch.int32)
        prefix_cpu = torch.tensor([0], dtype=torch.int32)
        dev_table = torch.tensor([[0, 4, 5]], dtype=torch.int32)
        with patch.dict(os.environ, {"TOKENSPEED_CACHE_DEBUG": "0"}):
            with pytest.raises(ValueError, match="disagrees with device table shape"):
                backend.init_forward_metadata(
                    bs=1,
                    num_extends=1,
                    req_pool_indices=torch.tensor([1], dtype=torch.int32),
                    seq_lens=lens_cpu,
                    forward_mode=self.ForwardMode.EXTEND,
                    block_tables={self.GROUP: dev_table},
                    block_tables_cpu={
                        self.GROUP: torch.tensor([[0, 4]], dtype=torch.int32)
                    },
                    **_extend_kwargs(torch, lens_cpu, prefix_cpu, "cpu"),
                )
            with pytest.raises(ValueError, match="must use int32"):
                backend.init_forward_metadata(
                    bs=1,
                    num_extends=1,
                    req_pool_indices=torch.tensor([1], dtype=torch.int32),
                    seq_lens=lens_cpu,
                    forward_mode=self.ForwardMode.EXTEND,
                    block_tables={self.GROUP: dev_table},
                    block_tables_cpu={
                        self.GROUP: torch.tensor([[0, 4, 5]], dtype=torch.int64)
                    },
                    **_extend_kwargs(torch, lens_cpu, prefix_cpu, "cpu"),
                )

    def test_cuda_host_table_check_has_zero_gpu_sync_and_rejects_gpu_tensor(self):
        torch = self.torch
        if not torch.cuda.is_available():
            pytest.skip("needs a CUDA device")
        backend = self._make_backend(device="cuda", num_blocks=10)
        # Warm up any lazy CUDA allocations outside the sync guard.
        self._init_extend(
            backend, [[0, 4], [2, 5]], device="cuda", pass_cpu=True, debug=False
        )

        def _prepare(rows):
            bs = len(rows)
            lens_cpu = torch.full((bs,), 64, dtype=torch.int32)
            prefix_cpu = torch.tensor(
                [64 if row[0] > 0 else 0 for row in rows], dtype=torch.int32
            )
            table_cpu = torch.tensor(rows, dtype=torch.int32)
            return dict(
                bs=bs,
                num_extends=bs,
                req_pool_indices=torch.arange(
                    1, bs + 1, dtype=torch.int32, device="cuda"
                ),
                seq_lens=(prefix_cpu + lens_cpu).cuda(),
                forward_mode=self.ForwardMode.EXTEND,
                block_tables={self.GROUP: table_cpu.cuda()},
                block_tables_cpu={self.GROUP: table_cpu},
                **_extend_kwargs(torch, lens_cpu, prefix_cpu, "cuda"),
            )

        valid_call = _prepare([[-1, -1], [0, 4], [2, 9]])
        oob_call = _prepare([[0, 4], [2, 10]])
        torch.cuda.synchronize()
        previous = torch.cuda.get_sync_debug_mode()
        try:
            with patch.dict(os.environ, {"TOKENSPEED_CACHE_DEBUG": "0"}):
                torch.cuda.set_sync_debug_mode("error")
                backend.init_forward_metadata(**valid_call)
                with pytest.raises(ValueError, match="row 1, slot 1 is 10"):
                    backend.init_forward_metadata(**oob_call)
        finally:
            torch.cuda.set_sync_debug_mode(previous)

        lens_cpu = torch.tensor([64], dtype=torch.int32)
        prefix_cpu = torch.tensor([0], dtype=torch.int32)
        gpu_table = torch.tensor([[0, 4]], dtype=torch.int32, device="cuda")
        with patch.dict(os.environ, {"TOKENSPEED_CACHE_DEBUG": "0"}):
            with pytest.raises(ValueError, match="must be a CPU tensor"):
                backend.init_forward_metadata(
                    bs=1,
                    num_extends=1,
                    req_pool_indices=torch.tensor(
                        [1], dtype=torch.int32, device="cuda"
                    ),
                    seq_lens=lens_cpu.cuda(),
                    forward_mode=self.ForwardMode.EXTEND,
                    block_tables={self.GROUP: gpu_table},
                    block_tables_cpu={self.GROUP: gpu_table},
                    **_extend_kwargs(torch, lens_cpu, prefix_cpu, "cuda"),
                )
