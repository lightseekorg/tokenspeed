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


class _CacheMetadata:
    def __init__(self, tables):
        self.tables = tables

    def require_table(self, group_id, active_forward_op=None):
        return self.tables[group_id]


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


class ComputeStatePageIndicesTest(unittest.TestCase):
    """CPU-only contract tests for the pure dual-index helper."""

    def setUp(self):
        try:
            import torch

            from tokenspeed.runtime.layers.attention.backends.hybrid_linear_attn import (  # noqa: E501
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

    def test_validate_off_masks_guards(self):
        torch = self.torch
        state_in, state_out = self.fn(
            torch.tensor([[0, 0, 0]], dtype=torch.int32),
            4,
            torch.tensor([0], dtype=torch.int32),
            torch.tensor([1], dtype=torch.int32),
            validate=False,
        )
        self.assertEqual(state_in.tolist(), [0])
        self.assertEqual(state_out.tolist(), [0])


class CacheContractMetadataTest(unittest.TestCase):
    """Every metadata entry point resolves state through the cache contract."""

    P = 4  # state page size (tokens)

    def setUp(self):
        try:
            import torch

            from tokenspeed.runtime.execution.forward_batch_info import (
                ForwardMode,
            )
            from tokenspeed.runtime.layers.attention.backends.hybrid_linear_attn import (  # noqa: E501
                MambaAttnBackend,
            )
        except (ImportError, ModuleNotFoundError) as exc:
            self.skipTest(f"needs torch + tokenspeed_kernel: {exc}")
        self.torch = torch
        self.ForwardMode = ForwardMode
        config = SimpleNamespace(
            device="cpu",
            num_attention_heads=16,
            num_kv_heads=16,
            attn_tp_size=1,
            dtype=torch.bfloat16,
            head_dim=128,
            is_draft=False,
            speculative_num_draft_tokens=1,
            max_bs=8,
        )
        backend = MambaAttnBackend(config)
        stub_pool = _ContractPool(
            self.P,
            {0: ("linear_attention", object(), object())},
        )
        backend.set_kv_pool(stub_pool)
        self.assertTrue(backend.state_paging_active)
        self.backend = backend

    def test_decode_metadata(self):
        torch = self.torch
        backend = self.backend
        backend.init_forward_metadata(
            bs=1,
            req_pool_indices=torch.tensor([0], dtype=torch.int32),
            seq_lens=torch.tensor([9], dtype=torch.int32),
            forward_mode=self.ForwardMode.DECODE,
            cache_metadata=_CacheMetadata(
                {"linear_attention": torch.tensor([[1, 2, 3]], dtype=torch.int32)}
            ),
        )
        md = backend.forward_metadata
        # before = 8 -> page slot 1 (row 2); after = 9 -> page slot 2 (row 3).
        self.assertEqual(md.state_in_blocks_by_group["linear_attention"].tolist(), [2])
        self.assertEqual(md.state_out_blocks_by_group["linear_attention"].tolist(), [3])
        # Decode builds no host boundary tuple; only extend batches carry one.
        self.assertIsNone(md.cu_extend_seq_lens_cpu)

    def test_extend_metadata(self):
        torch = self.torch
        backend = self.backend
        backend.init_forward_metadata(
            bs=1,
            req_pool_indices=torch.tensor([0], dtype=torch.int32),
            seq_lens=torch.tensor([8], dtype=torch.int32),
            forward_mode=self.ForwardMode.EXTEND,
            extend_prefix_lens=torch.zeros(1, dtype=torch.int32),
            extend_seq_lens_cpu=torch.tensor([8], dtype=torch.int32),
            cache_metadata=_CacheMetadata(
                {"linear_attention": torch.tensor([[1, 2]], dtype=torch.int32)}
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

    def test_extend_metadata_requires_host_lens(self):
        torch = self.torch
        backend = self.backend
        with self.assertRaisesRegex(RuntimeError, "host extend lengths"):
            backend.init_forward_metadata(
                bs=1,
                req_pool_indices=torch.tensor([0], dtype=torch.int32),
                seq_lens=torch.tensor([8], dtype=torch.int32),
                forward_mode=self.ForwardMode.EXTEND,
                extend_prefix_lens=torch.zeros(1, dtype=torch.int32),
                cache_metadata=_CacheMetadata(
                    {"linear_attention": torch.tensor([[1, 2]], dtype=torch.int32)}
                ),
            )

    def test_mixed_metadata_pads_decode_rows(self):
        torch = self.torch
        backend = self.backend
        backend.init_forward_metadata(
            bs=2,
            num_extends=1,
            req_pool_indices=torch.tensor([0, 1], dtype=torch.int32),
            seq_lens=torch.tensor([8, 9], dtype=torch.int32),
            forward_mode=self.ForwardMode.MIXED,
            extend_seq_lens=torch.tensor([5, 1], dtype=torch.int32),
            extend_seq_lens_cpu=torch.tensor([5], dtype=torch.int32),
            cache_metadata=_CacheMetadata(
                {"linear_attention": torch.tensor([[1, 2], [3, 4]], dtype=torch.int32)}
            ),
        )
        md = backend.forward_metadata
        # One extend row (5 tokens) plus one decode row padded to
        # spec_num_tokens (= 1): boundaries and the raw cat agree.
        self.assertEqual(md.cu_extend_seq_lens_cpu.tolist(), [0, 5, 6])
        self.assertEqual(md.extend_seq_lens_cpu.tolist(), [5, 1])
        self.assertEqual(md.query_start_loc.tolist(), [0, 5, 6])

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
            cache_metadata=_CacheMetadata(
                {"linear_attention": torch.tensor([[1, 2, 3]], dtype=torch.int32)}
            ),
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
            from tokenspeed.runtime.layers.attention.backends.hybrid_linear_attn import (  # noqa: E501
                MambaAttnBackend,
            )
        except (ImportError, ModuleNotFoundError) as exc:
            self.skipTest(f"needs torch + tokenspeed_kernel: {exc}")
        self.torch = torch
        self.ForwardMode = ForwardMode
        config = SimpleNamespace(
            device="cpu",
            num_attention_heads=2,
            num_kv_heads=2,
            attn_tp_size=1,
            dtype=torch.bfloat16,
            head_dim=2,
            is_draft=False,
            speculative_num_draft_tokens=4,
            max_bs=8,
        )
        self.backend = MambaAttnBackend(config)
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
        self.backend.init_forward_metadata(
            bs=1,
            req_pool_indices=torch.tensor([1], dtype=torch.int32),
            seq_lens=torch.tensor([8], dtype=torch.int32),
            forward_mode=self.ForwardMode.DECODE,
            tokens_per_req=4,
            cache_metadata=_CacheMetadata(
                {
                    "linear_attention_0": torch.tensor([[3, 4]], dtype=torch.int32),
                    "linear_attention_1": torch.tensor([[5, 6]], dtype=torch.int32),
                }
            ),
        )

        metadata = self.backend.forward_metadata
        self.assertEqual(metadata.mamba_output_indices.tolist(), [[1, 2, 3, 4]])
        self.assertEqual(metadata.mamba_output_indices.dtype, torch.int32)
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
            from tokenspeed_kernel.ops.attention import gdn_replay_commit_supported
            from tokenspeed_kernel.ops.attention.flashinfer import (
                gated_delta_rule as gdn,
            )

            from tokenspeed.runtime.execution.forward_batch_info import (
                ForwardMode,
            )
            from tokenspeed.runtime.layers.attention.backends.hybrid_linear_attn import (  # noqa: E501
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

    def _make_backend(
        self, conv_slab, ssm_slab, spec_num_tokens=1, *, replay_ssm=False
    ):
        torch = self.torch
        config = SimpleNamespace(
            device="cuda",
            num_attention_heads=self.H,
            num_kv_heads=self.H,
            attn_tp_size=1,
            dtype=torch.bfloat16,
            head_dim=self.D,
            is_draft=False,
            speculative_num_draft_tokens=spec_num_tokens,
            replay_ssm=replay_ssm,
            max_bs=8,
        )
        backend = self.MambaAttnBackend(config)
        stub_pool = _ContractPool(
            self.P,
            {0: ("linear_attention", conv_slab, ssm_slab)},
        )
        backend.set_kv_pool(stub_pool)
        self.assertTrue(backend.state_paging_active)
        return backend

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
        backend = self._make_backend(
            conv_slab, ssm_slab, spec_num_tokens=4, replay_ssm=True
        )
        backend.init_forward_metadata(
            bs=2,
            req_pool_indices=torch.tensor([0, 1], dtype=torch.int32, device="cuda"),
            seq_lens=torch.tensor([8, 8], dtype=torch.int32, device="cuda"),
            forward_mode=self.ForwardMode.DECODE,
            tokens_per_req=4,
            cache_metadata=_CacheMetadata(
                {
                    "linear_attention": torch.tensor(
                        [[3, 4], [5, 6]], dtype=torch.int32, device="cuda"
                    )
                }
            ),
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
        from tokenspeed_kernel.ops.attention.triton.linear.chunk import (
            chunk_gated_delta_rule,
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
        conv_out = causal_conv1d_fn(
            mixed_full.transpose(0, 1),
            conv_weights,
            bias,
            activation="silu",
            conv_states=ref_conv_state,
            has_initial_state=torch.zeros(1, dtype=torch.bool, device="cuda"),
            cache_indices=torch.zeros(1, dtype=torch.int32, device="cuda"),
            query_start_loc=torch.tensor([0, total], dtype=torch.int32, device="cuda"),
            seq_lens_cpu=torch.tensor([total], dtype=torch.int32),
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
        backend = self._make_backend(conv_slab, ssm_slab)

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
            req_pool_indices=req_pool_indices,
            seq_lens=torch.tensor([self.PREFILL], dtype=torch.int32, device="cuda"),
            forward_mode=ForwardMode.EXTEND,
            extend_prefix_lens=torch.zeros(1, dtype=torch.int32, device="cuda"),
            extend_seq_lens_cpu=torch.tensor([self.PREFILL], dtype=torch.int32),
            cache_metadata=_CacheMetadata(
                {
                    "linear_attention": torch.tensor(
                        [[1, 2]], dtype=torch.int32, device="cuda"
                    )
                }
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
            backend.init_forward_metadata(
                bs=1,
                req_pool_indices=req_pool_indices,
                seq_lens=torch.tensor([pos + 1], dtype=torch.int32, device="cuda"),
                forward_mode=ForwardMode.DECODE,
                cache_metadata=_CacheMetadata({"linear_attention": rows}),
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


if __name__ == "__main__":
    unittest.main()
