"""Inkling sconv state under speculative decoding — unit tests.

Validates the window-blend math (`InklingAttnBackend._write_window_at`) that
both the target-verify rollback and the draft catch-up use: the working
window must equal a from-scratch recompute over ``[old window || accepted
chunk prefix]`` for every accept length, including ``accept < W-1`` (borrow
from the old window). Also checks the verify stash + post-verify select
path end-to-end at the backend level (no attention, CPU-friendly but run on
GPU to match the pool's device usage).
"""

import os
import sys
import unittest

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class TestInklingCacheContract(unittest.TestCase):
    def test_wrapper_consumes_history_and_checkpoint_state(self):
        from tokenspeed.runtime.layers.attention.backends.inkling import (
            InklingAttnBackend,
        )

        class HistoryBackend:
            cache_consumer_families = frozenset({"history"})

        backend = InklingAttnBackend.__new__(InklingAttnBackend)
        backend.inner = HistoryBackend()

        self.assertEqual(
            backend.cache_consumer_families,
            frozenset({"history", "state"}),
        )


@unittest.skipUnless(torch.cuda.is_available(), "needs a CUDA device")
class TestInklingConvSpecState(unittest.TestCase):
    W = 4  # sconv kernel size (window W-1 = 3)
    DIM = 8
    BS = 5
    K = 4  # spec_num_tokens (draft tokens per verify round)
    LAYERS = 3

    def _make_pool(self):
        from tokenspeed.runtime.layers.attention.backends.inkling import (
            InklingConvStatePool,
        )

        pool = InklingConvStatePool(
            num_layers=self.LAYERS,
            num_slots=self.BS + 2,
            conv_dim=self.DIM,
            kernel_size=self.W,
            dtype=torch.float32,
            device="cuda",
        )
        torch.manual_seed(7)
        pool.conv_state.copy_(torch.randn_like(pool.conv_state))
        return pool

    def _reference_window(self, old, chunk, accept):
        """Last W-1 rows of [old || chunk[:accept]] (per request)."""
        stream = torch.cat([old, chunk[:accept]], dim=0)
        return stream[-(self.W - 1) :]

    def test_checkpoint_stream_registration(self):
        """Instance-level registration API: idempotent re-register with the
        same buffers, error on changed storage. (Regression: an orphaned
        @staticmethod once unbound this method and broke server startup.)"""
        from tokenspeed.runtime.layers.attention.backends.inkling import (
            InklingAttnBackend,
        )

        backend = InklingAttnBackend.__new__(InklingAttnBackend)
        backend._checkpoint_streams = {}
        buf = torch.zeros(4, self.W - 1, self.DIM, device="cuda")
        for _ in range(2):  # re-registering the same view is a no-op
            backend.register_shortconv_checkpoint_stream(
                layer_id=0,
                channel_offset=0,
                dim=self.DIM,
                group_id="state",
                buffers=(buf,),
            )
        self.assertEqual(len(backend._checkpoint_streams), 1)
        with self.assertRaises(RuntimeError):
            backend.register_shortconv_checkpoint_stream(
                layer_id=0,
                channel_offset=0,
                dim=self.DIM,
                group_id="state",
                buffers=(buf.clone(),),
            )

    def test_write_window_mixed_accepts(self):
        # accepts [1, 2, 3, 4, 2] span every accept length 1..K in one call
        # (the implementation is vectorized per request, no cross-request
        # coupling), so this is the full accept-length sweep.
        from tokenspeed.runtime.layers.attention.backends.inkling import (
            InklingAttnBackend,
        )

        pool = self._make_pool()
        state = pool.layer_state_wd(1)
        cache_indices = torch.arange(1, self.BS + 1, dtype=torch.int32).cuda()
        chunk = torch.randn(self.BS * self.K, self.DIM).cuda()
        old = state[cache_indices.long()].clone()
        accepts = [1, 2, 3, 4, 2]
        accept = torch.tensor(accepts, dtype=torch.int32).cuda()

        InklingAttnBackend._write_window_at(state, chunk, cache_indices, self.K, accept)
        for i, a in enumerate(accepts):
            expect = self._reference_window(
                old[i], chunk.view(self.BS, self.K, self.DIM)[i], a
            )
            self.assertTrue(torch.equal(state[cache_indices[i].long()], expect))

    def test_verify_stash_then_select(self):
        """Target-verify flow: stash per-layer chunks, working state untouched
        until the post-verify hook, then every layer's window equals the
        recompute at its request's accept length."""
        from tokenspeed.runtime.layers.attention.backends.inkling import (
            InklingAttnBackend,
            InklingConvMetadata,
        )

        pool = self._make_pool()
        backend = InklingAttnBackend.__new__(InklingAttnBackend)
        backend.conv_pool = pool
        backend.conv_spec_num_tokens = self.K
        backend.conv_is_draft = False
        backend._verify_stash = None
        backend._stash_pinned = False
        backend._ensure_verify_stash(self.BS * self.K, "cuda")

        cache_indices = torch.arange(1, self.BS + 1, dtype=torch.int32).cuda()
        md = InklingConvMetadata(
            query_start_loc=torch.arange(
                0, self.BS * self.K + 1, self.K, dtype=torch.int32
            ).cuda(),
            cache_indices=cache_indices,
            has_initial_state=torch.ones(self.BS, dtype=torch.bool).cuda(),
            is_decode=False,
            update_mode="stash",
            tokens_per_req=self.K,
        )
        backend.conv_metadata = md

        pre = pool.conv_state.clone()
        chunks = {}
        for layer in range(self.LAYERS):
            x = torch.randn(self.BS * self.K, self.DIM).cuda()
            chunks[layer] = x
            # The unified compute kernel writes the scratch in-kernel; the
            # backend hook is a no-op for stash mode. Emulate the kernel.
            backend._verify_stash[layer, : self.BS * self.K].copy_(x)
            backend.apply_conv_state_update(
                x, pool.layer_state_wd(layer), md, layer, 0, self.DIM
            )
        # Stash mode must not touch the pool.
        self.assertTrue(torch.equal(pool.conv_state, pre))

        accepts = [2, 1, 4, 3, 1]
        accept = torch.tensor(accepts, dtype=torch.int32).cuda()
        backend.update_mamba_state_after_mtp_verify(accept, None)

        for layer in range(self.LAYERS):
            state = pool.layer_state_wd(layer)
            for i, a in enumerate(accepts):
                expect = self._reference_window(
                    pre[layer, cache_indices[i].long()],
                    chunks[layer].view(self.BS, self.K, self.DIM)[i],
                    a,
                )
                self.assertTrue(
                    torch.equal(state[cache_indices[i].long()], expect),
                    f"layer {layer} req {i} accept {a}",
                )

    def test_checkpoint_metadata_keeps_only_chunk_endpoint(self):
        from tokenspeed.runtime.layers.attention.backends.inkling import (
            InklingAttnBackend,
        )

        pool = self._make_pool()
        backend = InklingAttnBackend.__new__(InklingAttnBackend)
        backend.conv_pool = pool
        backend.conv_columns = {
            "mode": "checkpoint",
            "block_tokens": 4,
            "group_block_tokens": {"state": 4},
        }
        metadata = backend._new_checkpoint_metadata(
            size=2,
            groups=("state",),
            device=torch.device("cuda"),
            include_prefill_rows=True,
        )
        pointers = tuple(
            tensor.data_ptr()
            for tensor in (
                metadata.restore_pages["state"],
                metadata.write_pages["state"],
                metadata.packed_rows,
            )
        )
        table = torch.tensor([[11, 12, 13], [21, 22, 23]], device="cuda")
        backend._fill_checkpoint_metadata(
            metadata,
            before=torch.tensor([0, 7], device="cuda"),
            after=torch.tensor([8, 8], device="cuda"),
            query_start_loc=torch.tensor([0, 8, 9], device="cuda"),
            col_page_table={"state": table},
            write_endpoint=True,
        )

        self.assertEqual(metadata.restore_pages["state"].tolist(), [0, 0])
        # Request 0 crosses two boundaries, but only its published endpoint is
        # materialized. Request 1 borrows two rows from its prior window.
        self.assertEqual(metadata.write_pages["state"].tolist(), [12, 22])
        self.assertEqual(
            metadata.packed_row_mask.tolist(), [[True] * 3, [False, False, True]]
        )
        self.assertEqual(metadata.packed_rows.tolist(), [[5, 6, 7], [6, 7, 8]])
        self.assertEqual(metadata.prior_state_rows.tolist(), [[2, 2, 2], [1, 2, 2]])
        self.assertEqual(
            pointers,
            tuple(
                tensor.data_ptr()
                for tensor in (
                    metadata.restore_pages["state"],
                    metadata.write_pages["state"],
                    metadata.packed_rows,
                )
            ),
        )

    def test_verify_select_padded_batch_oversized_stash(self):
        """Post-verify select with graph-padded shapes: the stash is larger
        than the round's n*k rows (sliced view), accept_lengths covers fewer
        requests than the padded metadata batch, and a padded row carries
        PAD_SLOT_ID (-1) — its write must land in reserved slot 0."""
        from tokenspeed.runtime.layers.attention.backends.inkling import (
            InklingAttnBackend,
            InklingConvMetadata,
        )

        pool = self._make_pool()
        backend = InklingAttnBackend.__new__(InklingAttnBackend)
        backend.conv_pool = pool
        backend.conv_spec_num_tokens = self.K
        backend.conv_is_draft = False
        backend._verify_stash = None
        backend._stash_pinned = False
        # Stash sized for the full padded capacity, round uses fewer rows.
        backend._ensure_verify_stash((self.BS + 2) * self.K, "cuda")

        n_real = 3
        cache_indices = torch.tensor(
            [2, 4, -1, 1, 3], dtype=torch.int32
        ).cuda()  # row 2 is a padded slot
        backend.conv_metadata = InklingConvMetadata(
            query_start_loc=torch.arange(
                0, self.BS * self.K + 1, self.K, dtype=torch.int32
            ).cuda(),
            cache_indices=cache_indices,
            has_initial_state=torch.ones(self.BS, dtype=torch.bool).cuda(),
            is_decode=False,
            update_mode="stash",
            tokens_per_req=self.K,
        )

        pre = pool.conv_state.clone()
        stash = torch.randn(self.LAYERS, n_real * self.K, self.DIM, device="cuda")
        backend._verify_stash[:, : n_real * self.K].copy_(stash)

        accepts = [3, 1, 2]  # covers only the leading n_real requests
        backend.update_mamba_state_after_mtp_verify(
            torch.tensor(accepts, dtype=torch.int32).cuda(), None
        )

        for layer in range(self.LAYERS):
            state = pool.layer_state_wd(layer)
            for i, a in enumerate(accepts):
                slot = int(cache_indices[i].clamp_min(0))
                expect = self._reference_window(
                    pre[layer, slot],
                    stash[layer].view(n_real, self.K, self.DIM)[i],
                    a,
                )
                self.assertTrue(
                    torch.equal(state[slot], expect), f"layer {layer} req {i}"
                )
            # Requests beyond accept_lengths (rows 3, 4) stay untouched.
            for slot in (1, 3):
                self.assertTrue(torch.equal(state[slot], pre[layer, slot]))

    def test_channel_slice_update(self):
        """valid_len write through a channel-offset slice only touches that
        slice (the fused K+V call updates a sub-range of conv_dim)."""
        from tokenspeed.runtime.layers.attention.backends.inkling import (
            InklingAttnBackend,
        )

        pool = self._make_pool()
        off, dim = 2, 4
        full = pool.layer_state_wd(2)
        state = full[:, :, off : off + dim]
        pre = pool.conv_state.clone()
        cache_indices = torch.arange(1, self.BS + 1, dtype=torch.int32).cuda()
        chunk = torch.randn(self.BS * self.K, dim).cuda()
        accept = torch.tensor([1, 2, 3, 4, 2], dtype=torch.int32).cuda()

        InklingAttnBackend._write_window_at(state, chunk, cache_indices, self.K, accept)

        # Outside the channel slice: unchanged.
        self.assertTrue(torch.equal(full[:, :, :off], pre[2][:, :, :off]))
        self.assertTrue(torch.equal(full[:, :, off + dim :], pre[2][:, :, off + dim :]))
        # Inside: matches the recompute.
        for i, a in enumerate(accept.tolist()):
            expect = self._reference_window(
                pre[2, cache_indices[i].long(), :, off : off + dim],
                chunk.view(self.BS, self.K, dim)[i],
                a,
            )
            self.assertTrue(torch.equal(state[cache_indices[i].long()], expect))


@unittest.skipUnless(torch.cuda.is_available(), "needs a CUDA device")
class TestSconvUnifiedKernel(unittest.TestCase):
    """The single sconv compute kernel: decode = T=1 case, in-kernel scratch
    write and speculative boundary-checkpoint publish."""

    W = 4
    DIM = 8
    K = 4

    def _weight(self):
        torch.manual_seed(11)
        return torch.randn(self.DIM, self.W, device="cuda")

    def _ref_conv(self, x_req, prefix, weight, use_residual=True):
        """Per-request reference: causal conv over [prefix || x]."""
        ext = torch.cat([prefix, x_req], dim=0)
        y = torch.zeros_like(x_req)
        for t in range(x_req.shape[0]):
            window = ext[t : t + self.W]  # W rows ending at token t
            y[t] = (window * weight.t()).sum(0)
        if use_residual:
            y = y + x_req
        return y

    def _ref_window(self, prefix, x_req, upto):
        """Conv window (last W-1 input rows) at position `upto` (1-based in
        the chunk): rows of [prefix || x[:upto]]."""
        return torch.cat([prefix, x_req[:upto]], dim=0)[-(self.W - 1) :]

    def _state(self, num_slots=8):
        torch.manual_seed(5)
        return torch.randn(num_slots, self.W - 1, self.DIM, device="cuda")

    def test_decode_is_t1_case(self):
        """Decode = the unified kernel with T=1 rows: y matches the reference
        conv over [state || x_t]; the state is untouched (no fused shift);
        sconv_cache_update then persists exactly the shifted window."""
        from tokenspeed_kernel.ops.conv import inkling_sconv, sconv_cache_update

        weight = self._weight()
        state = self._state()
        pre = state.clone()
        B = 3
        cache_indices = torch.tensor([1, 2, -1], dtype=torch.int32, device="cuda")
        x = torch.randn(B, self.DIM, device="cuda")

        qsl = torch.arange(B + 1, dtype=torch.int32, device="cuda")
        y = inkling_sconv(
            x,
            weight,
            state,
            qsl,
            qsl[:B],
            cache_indices,
            torch.ones(B, dtype=torch.bool, device="cuda"),
        )

        self.assertTrue(torch.equal(state, pre))  # read-only now
        for b, slot in enumerate([1, 2, None]):
            prefix = (
                pre[slot]
                if slot is not None
                else torch.zeros(self.W - 1, self.DIM, device="cuda")
            )
            expect = self._ref_conv(x[b : b + 1], prefix, weight)
            self.assertTrue(torch.allclose(y[b : b + 1], expect, atol=1e-4), f"req {b}")

        sconv_cache_update(
            x,
            state,
            qsl,
            cache_indices,
            torch.ones(B, dtype=torch.bool, device="cuda"),
        )
        for b, slot in enumerate([1, 2]):
            expect = torch.cat([pre[slot], x[b : b + 1]])[-(self.W - 1) :]
            self.assertTrue(torch.equal(state[slot], expect), f"slot {slot}")

    def test_scratch_write(self):
        """WRITE_SCRATCH copies every chunk row, padded requests included."""
        from tokenspeed_kernel.ops.conv import inkling_sconv, seq_idx_from_cu_seqlens

        weight = self._weight()
        state = self._state()
        B, k = 3, self.K
        qsl = torch.arange(0, B * k + 1, k, dtype=torch.int32, device="cuda")
        x = torch.randn(B * k, self.DIM, device="cuda")
        scratch = torch.zeros(B * k + 2, self.DIM, device="cuda")
        cache_indices = torch.tensor([1, -1, 2], dtype=torch.int32, device="cuda")

        inkling_sconv(
            x,
            weight,
            state,
            qsl,
            seq_idx_from_cu_seqlens(qsl, B * k),
            cache_indices,
            torch.ones(B, dtype=torch.bool, device="cuda"),
            scratch=scratch,
        )
        self.assertTrue(torch.equal(scratch[: B * k], x))

    def _publish_setup(self, col_seq_lens, cache_indices, page_size=8, pages=40):
        from tokenspeed_kernel.ops.conv import seq_idx_from_cu_seqlens

        B = cache_indices.shape[0]
        k = self.K
        qsl = torch.arange(0, B * k + 1, k, dtype=torch.int32, device="cuda")
        seq_idx = seq_idx_from_cu_seqlens(qsl, B * k)
        table = torch.arange(11, 11 + B * 2, dtype=torch.int32, device="cuda").reshape(
            B, 2
        )
        checkpoint = torch.full((pages, self.W - 1, self.DIM), -7.0, device="cuda")
        return qsl, seq_idx, table, checkpoint

    def test_publish_verify_boundaries(self):
        """Verify-shaped chunks (uniform K): covered boundaries publish the
        window (borrowing state rows when the boundary falls early in the
        chunk), uncovered/padded requests and untouched pages stay clean —
        all independent of any accept decision."""
        from tokenspeed_kernel.ops.conv import inkling_sconv

        weight = self._weight()
        state = self._state()
        # S0 = [4, 2, 6, 5] -> boundary L=8 covered for reqs 0 (p*=4) and
        # 2 (p*=2, borrows one state row); req1 uncovered; req3 padded.
        cache_indices = torch.tensor([1, 2, 3, -1], dtype=torch.int32).cuda()
        col_seq_lens = torch.tensor([8, 6, 10, 9], dtype=torch.int32).cuda()
        qsl, seq_idx, table, checkpoint = self._publish_setup(
            col_seq_lens, cache_indices
        )
        x = torch.randn(4 * self.K, self.DIM, device="cuda")

        inkling_sconv(
            x,
            weight,
            state,
            qsl,
            seq_idx,
            cache_indices,
            torch.ones(4, dtype=torch.bool, device="cuda"),
            publish=(table, col_seq_lens, checkpoint, None, 8),
        )

        # req0: p*=4 -> page table[0,0]=11
        self.assertTrue(
            torch.equal(
                checkpoint[11],
                self._ref_window(state[1], x[0 : self.K], 4),
            )
        )
        # req2: p*=2 -> page table[2,0]=15, borrows state row
        self.assertTrue(
            torch.equal(
                checkpoint[15],
                self._ref_window(state[3], x[2 * self.K : 3 * self.K], 2),
            )
        )
        touched = {11, 15}
        for page in range(checkpoint.shape[0]):
            if page not in touched:
                self.assertTrue(bool((checkpoint[page] == -7).all()), f"page {page}")

    def test_publish_prefill_interior_boundaries(self):
        """A prefill chunk spanning several pages publishes EVERY interior
        boundary (today's python path published only chunk endpoints)."""
        from tokenspeed_kernel.ops.conv import inkling_sconv, seq_idx_from_cu_seqlens

        weight = self._weight()
        state = self._state()
        T, page_size = 16, 4
        qsl = torch.tensor([0, T], dtype=torch.int32, device="cuda")
        seq_idx = seq_idx_from_cu_seqlens(qsl, T)
        cache_indices = torch.tensor([1], dtype=torch.int32, device="cuda")
        # Fresh prefill from length 0: boundaries at 4, 8, 12, 16.
        col_seq_lens = torch.tensor([T], dtype=torch.int32, device="cuda")
        table = torch.arange(21, 21 + 4, dtype=torch.int32, device="cuda").reshape(1, 4)
        checkpoint = torch.full((40, self.W - 1, self.DIM), -7.0, device="cuda")
        x = torch.randn(T, self.DIM, device="cuda")

        inkling_sconv(
            x,
            weight,
            state,
            qsl,
            seq_idx,
            cache_indices,
            torch.zeros(1, dtype=torch.bool, device="cuda"),  # fresh: no borrow
            publish=(table, col_seq_lens, checkpoint, None, page_size),
        )

        zeros = torch.zeros(self.W - 1, self.DIM, device="cuda")
        for i, boundary in enumerate([4, 8, 12, 16]):
            expect = self._ref_window(zeros, x, boundary)
            self.assertTrue(
                torch.equal(checkpoint[21 + i], expect), f"boundary {boundary}"
            )

    def test_publish_two_field_split_and_fp8(self):
        """Fused K+V split across two fields, and an fp8 destination: the
        kernel's store-side casts must match torch's."""
        from tokenspeed_kernel.ops.conv import inkling_sconv

        weight = self._weight()
        state = self._state()
        cache_indices = torch.tensor([1], dtype=torch.int32).cuda()
        col_seq_lens = torch.tensor([8], dtype=torch.int32).cuda()  # p*=4
        qsl, seq_idx, table, _ = self._publish_setup(col_seq_lens, cache_indices)
        field_a = torch.zeros(40, self.W - 1, 2, dtype=torch.bfloat16, device="cuda")
        field_b = torch.zeros(40, self.W - 1, 6, dtype=torch.float8_e5m2, device="cuda")
        x = torch.randn(self.K, self.DIM, device="cuda")

        inkling_sconv(
            x,
            weight,
            state,
            qsl,
            seq_idx,
            cache_indices,
            torch.ones(1, dtype=torch.bool, device="cuda"),
            publish=(table, col_seq_lens, field_a, field_b, 8),
        )

        window = self._ref_window(state[1], x, 4)
        self.assertTrue(torch.equal(field_a[11], window[:, :2].to(torch.bfloat16)))
        self.assertTrue(
            torch.equal(
                field_b[11].view(torch.uint8),
                window[:, 2:].to(torch.float8_e5m2).view(torch.uint8),
            )
        )

    def test_publish_overwrites_rejected_round(self):
        """Round 1 publishes candidate rows past its accepted length; round 2
        covering the same boundary overwrites with the committed rows."""
        from tokenspeed_kernel.ops.conv import inkling_sconv

        weight = self._weight()
        state = self._state()
        old = state[1].clone()
        cache_indices = torch.tensor([1], dtype=torch.int32).cuda()
        col_seq_lens = torch.tensor([10], dtype=torch.int32).cuda()  # S0=6, p*=2
        qsl, seq_idx, table, checkpoint = self._publish_setup(
            col_seq_lens, cache_indices
        )
        ones = torch.ones(1, dtype=torch.bool, device="cuda")
        x1 = torch.randn(self.K, self.DIM, device="cuda")

        inkling_sconv(
            x1,
            weight,
            state,
            qsl,
            seq_idx,
            cache_indices,
            ones,
            publish=(table, col_seq_lens, checkpoint, None, 8),
        )
        self.assertTrue(torch.equal(checkpoint[11], self._ref_window(old, x1, 2)))

        # accept=1: window advances by one committed row; S0=7 -> p*=1.
        state[1] = torch.cat([old, x1[:1]])[-(self.W - 1) :]
        col_seq_lens.fill_(11)
        x2 = torch.randn(self.K, self.DIM, device="cuda")
        inkling_sconv(
            x2,
            weight,
            state,
            qsl,
            seq_idx,
            cache_indices,
            ones,
            publish=(table, col_seq_lens, checkpoint, None, 8),
        )
        expect = torch.stack([old[-1], x1[0], x2[0]])
        self.assertTrue(torch.equal(checkpoint[11], expect))

    def test_publish_cuda_graph_replay(self):
        """All inputs are stable buffers: replays after in-place updates
        reproduce the eager result, including overwriting a prior replay's
        speculative write."""
        from tokenspeed_kernel.ops.conv import inkling_sconv

        weight = self._weight()
        state = self._state()
        cache_indices = torch.tensor([1, 2], dtype=torch.int32).cuda()
        col_seq_lens = torch.tensor([10, 12], dtype=torch.int32).cuda()
        qsl, seq_idx, table, checkpoint = self._publish_setup(
            col_seq_lens, cache_indices
        )
        ones = torch.ones(2, dtype=torch.bool, device="cuda")
        x = torch.randn(2 * self.K, self.DIM, device="cuda")
        scratch = torch.zeros(2 * self.K, self.DIM, device="cuda")

        def run():
            inkling_sconv(
                x,
                weight,
                state,
                qsl,
                seq_idx,
                cache_indices,
                ones,
                scratch=scratch,
                publish=(table, col_seq_lens, checkpoint, None, 8),
            )

        run()  # warmup compiles outside capture
        torch.cuda.synchronize()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            run()

        for round_lens in ([10, 12], [11, 16]):
            checkpoint.fill_(-7)
            col_seq_lens.copy_(
                torch.tensor(round_lens, dtype=torch.int32, device="cuda")
            )
            x.copy_(torch.randn_like(x))
            state.copy_(torch.randn_like(state))
            graph.replay()
            torch.cuda.synchronize()
            self.assertTrue(torch.equal(scratch, x))
            for req in range(2):
                base = round_lens[req] - self.K
                p = 8 - base % 8
                if p > self.K:
                    continue
                page = 11 + req * 2 + (base + p) // 8 - 1
                slot = int(cache_indices[req])
                expect = self._ref_window(
                    state[slot], x[req * self.K : (req + 1) * self.K], p
                )
                self.assertTrue(
                    torch.equal(checkpoint[page], expect),
                    f"req {req} lens {round_lens}",
                )


if __name__ == "__main__":
    unittest.main()
