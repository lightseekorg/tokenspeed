"""GDN dual-index state paging on the flat path (M17).

compute_state_page_indices maps per-request (seq_len_before, seq_len_after)
to (in, out) state page ids over the flat "linear_attention" block table;
the GPU test drives MambaAttnBackend in flat mode (prefill + decodes over
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


class ComputeStatePageIndicesTest(unittest.TestCase):
    """CPU-only contract tests for the pure dual-index helper."""

    def setUp(self):
        try:
            import torch

            from tokenspeed.runtime.layers.attention.backends.hybrid_linear_attn import (  # noqa: E501
                compute_state_page_indices,
            )
        except (ImportError, ModuleNotFoundError) as exc:
            self.skipTest(f"needs torch + tokenspeed_kernel: {exc}")
        self.torch = torch
        self.fn = compute_state_page_indices

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

    def test_batch_mixed(self):
        rows = [
            [7, 9, 12],
            [7, 9, 12],
            [7, 9, 12],
            [3, 5, 8],
        ]
        state_in, state_out = self._run(rows, [4, 5, 0, 8], [5, 6, 3, 9])
        self.assertEqual(state_in.tolist(), [7, 9, 0, 5])
        self.assertEqual(state_out.tolist(), [9, 9, 7, 8])

    def test_out_slot_hole_raises(self):
        with self.assertRaises(ValueError):
            self._run([[7, 0, 12]], [4], [5])

    def test_out_slot_pad_raises(self):
        with self.assertRaises(ValueError):
            self._run([[7, -1, 12]], [4], [5])

    def test_out_slot_past_table_raises(self):
        with self.assertRaises(ValueError):
            self._run([[7, 9]], [8], [9])

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


class PoollessFlatMetadataTest(unittest.TestCase):
    """Flat mode runs without a SimpleMambaPool (the runner no longer creates
    one), so every metadata entry point must tolerate ``pool is None``.
    CPU-only: pure index math, no kernels."""

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
        )
        backend = MambaAttnBackend(config)
        stub_pool = SimpleNamespace(
            state_slabs=[(object(), object())],
            paged_cache_group_specs=(
                SimpleNamespace(group_id="linear_attention"),
            ),
            page_size=self.P,
        )
        # set_pool is intentionally never called: flat mode has no
        # SimpleMambaPool.
        backend.set_kv_pool(stub_pool)
        self.assertTrue(backend.flat_state_active)
        self.assertIsNone(backend.pool)
        self.backend = backend

    def test_decode_metadata_without_pool(self):
        torch = self.torch
        backend = self.backend
        backend.init_forward_metadata(
            bs=1,
            req_pool_indices=torch.tensor([0], dtype=torch.int32),
            seq_lens=torch.tensor([9], dtype=torch.int32),
            forward_mode=self.ForwardMode.DECODE,
            flat_block_tables={
                "linear_attention": torch.tensor([[1, 2, 3]], dtype=torch.int32)
            },
        )
        md = backend.forward_metadata
        # before = 8 -> page slot 1 (row 2); after = 9 -> page slot 2 (row 3).
        self.assertEqual(md.state_in_pages.tolist(), [2])
        self.assertEqual(md.state_out_pages.tolist(), [3])

    def test_extend_metadata_without_pool(self):
        torch = self.torch
        backend = self.backend
        backend.init_forward_metadata(
            bs=1,
            req_pool_indices=torch.tensor([0], dtype=torch.int32),
            seq_lens=torch.tensor([8], dtype=torch.int32),
            forward_mode=self.ForwardMode.EXTEND,
            extend_prefix_lens=torch.zeros(1, dtype=torch.int32),
            flat_block_tables={
                "linear_attention": torch.tensor([[1, 2]], dtype=torch.int32)
            },
        )
        md = backend.forward_metadata
        self.assertEqual(md.state_in_pages.tolist(), [0])
        self.assertEqual(md.state_out_pages.tolist(), [2])

    def test_capture_replay_metadata_without_pool(self):
        torch = self.torch
        backend = self.backend
        backend.init_cuda_graph_state(max_num_tokens=2)
        backend.init_forward_metadata_capture_cuda_graph(
            bs=1,
            req_pool_indices=torch.tensor([0], dtype=torch.int32),
            seq_lens=torch.tensor([1], dtype=torch.int32),
            forward_mode=self.ForwardMode.DECODE,
            flat_cache_group_ids=("linear_attention",),
        )
        md = backend.forward_metadata
        # Capture binds the persistent pad-filled buffers.
        self.assertEqual(md.state_in_pages.tolist(), [-1])
        self.assertEqual(md.state_out_pages.tolist(), [-1])

        backend.init_forward_metadata_replay_cuda_graph(
            bs=1,
            req_pool_indices=torch.tensor([0], dtype=torch.int32),
            seq_lens=torch.tensor([9], dtype=torch.int32),
            forward_mode=self.ForwardMode.DECODE,
            flat_block_tables={
                "linear_attention": torch.tensor([[1, 2, 3]], dtype=torch.int32)
            },
        )
        md = backend.forward_metadata
        self.assertEqual(md.state_in_pages.tolist(), [2])
        self.assertEqual(md.state_out_pages.tolist(), [3])


class GDNFlatStatePagingGPUTest(unittest.TestCase):
    """MambaAttnBackend in flat mode (paged state slabs, dual-index) vs the
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
            from tokenspeed_kernel.ops.attention.flashinfer import (
                gated_delta_rule as gdn,
            )

            from tokenspeed.runtime.execution.forward_batch_info import (
                ForwardMode,
            )
            from tokenspeed.runtime.layers.attention.backends.hybrid_linear_attn import (  # noqa: E501
                MambaAttnBackend,
                SimpleMambaPool,
            )
        except (ImportError, ModuleNotFoundError) as exc:
            self.skipTest(f"needs torch + tokenspeed_kernel: {exc}")
        if not torch.cuda.is_available():
            self.skipTest("needs a CUDA device")
        if not gdn.is_available():
            self.skipTest("sm100 GDN kernel unavailable")
        self.torch = torch
        self.ForwardMode = ForwardMode
        self.MambaAttnBackend = MambaAttnBackend
        self.SimpleMambaPool = SimpleMambaPool
        torch.manual_seed(0)

    def _make_backend(self, conv_slab, ssm_slab):
        torch = self.torch
        config = SimpleNamespace(
            device="cuda",
            num_attention_heads=self.H,
            num_kv_heads=self.H,
            attn_tp_size=1,
            dtype=torch.bfloat16,
            head_dim=self.D,
            is_draft=False,
            speculative_num_draft_tokens=1,
        )
        backend = self.MambaAttnBackend(config)
        conv_dim = conv_slab.shape[1]
        backend.set_pool(
            self.SimpleMambaPool(
                size=4,
                num_mamba_layers=1,
                conv_state_shape=(conv_dim, self.WIDTH - 1),
                temporal_state_shape=(self.H, self.D, self.D),
                conv_dtype=torch.bfloat16,
                ssm_dtype=torch.float32,
                mamba_layer_ids=[0],
                device="cuda",
                page_size=self.P,
                max_req_pool_size=2,
            )
        )
        stub_pool = SimpleNamespace(
            state_slabs=[(conv_slab, ssm_slab)],
            paged_cache_group_specs=(SimpleNamespace(group_id="linear_attention"),),
            page_size=self.P,
            get_state_buffers=lambda layer_id: (conv_slab, ssm_slab),
        )
        backend.set_kv_pool(stub_pool)
        self.assertTrue(backend.flat_state_active)
        return backend

    def test_flat_paged_states_match_fla_oracle(self):
        torch = self.torch
        ForwardMode = self.ForwardMode
        from tokenspeed.runtime.layers.attention.linear.causal_conv1d import (
            causal_conv1d_fn,
        )
        from tokenspeed.runtime.layers.attention.linear.chunk import (
            chunk_gated_delta_rule,
        )
        from tokenspeed.runtime.layers.attention.linear.gdn import fused_gdn_gating

        H, D, P = self.H, self.D, self.P
        total = self.PREFILL + self.DECODES  # 11 tokens
        key_dim = H * D
        value_dim = H * D
        conv_dim = 2 * key_dim + value_dim

        mixed_full = torch.randn(total, conv_dim, device="cuda", dtype=torch.bfloat16)
        conv_weights = (
            torch.randn(conv_dim, self.WIDTH, device="cuda", dtype=torch.bfloat16)
            * 0.1
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
            initial_state=torch.zeros(
                1, H, D, D, device="cuda", dtype=torch.float32
            ),
            output_final_state=True,
            cu_seqlens=torch.tensor([0, total], device="cuda").long(),
            head_first=False,
            use_qk_l2norm_in_kernel=True,
        )

        # ---- Flat path: page 0 = null, pages fill as the sequence grows ----
        num_pages = total // P + 2  # null + pages 1..3
        conv_slab = torch.zeros(
            num_pages, conv_dim, self.WIDTH - 1, device="cuda", dtype=torch.bfloat16
        )
        ssm_slab = torch.zeros(
            num_pages, H, D, D, device="cuda", dtype=torch.float32
        )
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
            flat_block_tables={
                "linear_attention": torch.tensor(
                    [[1, 2]], dtype=torch.int32, device="cuda"
                )
            },
        )
        self.assertEqual(
            backend.forward_metadata.state_in_pages.tolist(), [0]
        )
        self.assertEqual(
            backend.forward_metadata.state_out_pages.tolist(), [2]
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
                flat_block_tables={"linear_attention": rows},
            )
            self.assertEqual(
                backend.forward_metadata.state_in_pages.tolist(),
                [expected_pages[i][0]],
            )
            self.assertEqual(
                backend.forward_metadata.state_out_pages.tolist(),
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

        o_flat = torch.cat(outputs, dim=1)
        self.assertEqual(tuple(o_flat.shape), tuple(o_ref.shape))

        # Fastpath-test tolerances: mean diff is the real bar, loose max.
        out_diff = (o_flat.float() - o_ref.float()).abs()
        self.assertLess(out_diff.mean().item(), 1e-3)
        self.assertTrue(
            torch.allclose(o_flat.float(), o_ref.float(), atol=1e-1, rtol=1e-2)
        )
        st_diff = (ssm_slab[3] - st_ref[0].float()).abs()
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
