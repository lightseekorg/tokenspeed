"""Inkling sconv ring state under speculative decoding — unit tests.

The working state is a per-slot ring of the last ``R`` input rows: ring row
of absolute position ``p`` is ``p % R``, positions derive from the
through-chunk ``seq_lens``. Verify rounds write all K candidate rows
speculatively; acceptance only decides which positions the next round reads,
and rejected rows are overwritten when their positions recur. These tests
validate the ring addressing at the backend level (accept sweeps, padded
batches, channel slices, checkpoint restore) and the unified compute
kernel's decode/publish behavior (no attention; run on GPU to match the
pool's device usage).
"""

import os
import sys
import unittest

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def _inert_publish(bs, dim, rows=3):
    """Publish plumbing that never fires: a hole-only table."""
    table = torch.zeros(bs, 64, dtype=torch.int32, device="cuda")
    ckpt = torch.zeros(1, rows, dim, device="cuda")
    return table, ckpt


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
class TestInklingConvRingState(unittest.TestCase):
    W = 4  # sconv kernel size (W-1 = 3 history taps)
    R = 9  # ring rows: (W-1) + K + lookback(K-2)
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
            ring_size=self.R,
            dtype=torch.float32,
            device="cuda",
        )
        torch.manual_seed(7)
        pool.conv_state.copy_(torch.randn_like(pool.conv_state))
        return pool

    def _ring_rows(self, state, slot, positions):
        """Rows of ``state[slot]`` at the given absolute positions."""
        return torch.stack([state[slot, p % self.R] for p in positions])

    def _weight(self):
        torch.manual_seed(11)
        return torch.randn(self.DIM, self.W, device="cuda")

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

    def test_ring_holds_window_for_every_accept(self):
        """One verify round writes all K candidate rows; for EVERY accept
        length the ring rows at the accepted frontier equal the recompute
        over [committed history || accepted chunk prefix]."""
        from tokenspeed_kernel.ops.conv import (
            inkling_ring_sconv,
            seq_idx_from_cu_seqlens,
        )

        pool = self._make_pool()
        weight = self._weight()
        state = pool.layer_state_wd(1)
        pre = state.clone()
        cache_indices = torch.arange(1, self.BS + 1, dtype=torch.int32).cuda()
        L0 = 20  # committed length per request
        seq_lens = torch.full((self.BS,), L0 + self.K, dtype=torch.int32).cuda()
        qsl = torch.arange(0, self.BS * self.K + 1, self.K, dtype=torch.int32).cuda()
        chunk = torch.randn(self.BS * self.K, self.DIM).cuda()

        inkling_ring_sconv(
            chunk,
            weight,
            state,
            qsl,
            seq_idx_from_cu_seqlens(qsl, self.BS * self.K),
            cache_indices,
            torch.ones(self.BS, dtype=torch.bool).cuda(),
            seq_lens,
            *_inert_publish(self.BS, self.DIM),
            None,
            num_extends=0,
            page_size=128,
        )

        for i in range(self.BS):
            slot = int(cache_indices[i])
            old = self._ring_rows(pre, slot, range(L0 - (self.W - 1), L0))
            chunk_i = chunk.view(self.BS, self.K, self.DIM)[i]
            for accept in range(1, self.K + 1):
                window = self._ring_rows(
                    state, slot, range(L0 + accept - (self.W - 1), L0 + accept)
                )
                expect = torch.cat([old, chunk_i[:accept]], dim=0)[-(self.W - 1) :]
                self.assertTrue(torch.equal(window, expect), f"req {i} accept {accept}")

    def test_verify_padded_batch_writes_nothing_for_pad_rows(self):
        """PAD rows (cache index -1) must leave every slot untouched, and
        non-padded requests behind them are unaffected."""
        from tokenspeed_kernel.ops.conv import (
            inkling_ring_sconv,
            seq_idx_from_cu_seqlens,
        )

        pool = self._make_pool()
        weight = self._weight()
        state = pool.layer_state_wd(0)
        pre = state.clone()
        cache_indices = torch.tensor([2, 4, -1, 1, 3], dtype=torch.int32).cuda()
        seq_lens = torch.tensor([24, 31, 999, 17, 40], dtype=torch.int32).cuda()
        qsl = torch.arange(0, self.BS * self.K + 1, self.K, dtype=torch.int32).cuda()
        chunk = torch.randn(self.BS * self.K, self.DIM).cuda()

        inkling_ring_sconv(
            chunk,
            weight,
            state,
            qsl,
            seq_idx_from_cu_seqlens(qsl, self.BS * self.K),
            cache_indices,
            torch.ones(self.BS, dtype=torch.bool).cuda(),
            seq_lens,
            *_inert_publish(self.BS, self.DIM),
            None,
            num_extends=0,
            page_size=128,
        )

        expected = pre.clone()
        for i in (0, 1, 3, 4):
            slot = int(cache_indices[i])
            L0 = int(seq_lens[i]) - self.K
            for j in range(self.K):
                expected[slot, (L0 + j) % self.R] = chunk[i * self.K + j]
        self.assertTrue(torch.equal(state, expected))

    def test_channel_slice_ring_write(self):
        """The kernel on a channel-offset slice only touches that slice
        (the fused K+V call updates a sub-range of conv_dim)."""
        from tokenspeed_kernel.ops.conv import (
            inkling_ring_sconv,
            seq_idx_from_cu_seqlens,
        )

        pool = self._make_pool()
        off, dim = 2, 4
        torch.manual_seed(11)
        weight = torch.randn(dim, self.W, device="cuda")
        full = pool.layer_state_wd(2)
        state = full[:, :, off : off + dim]
        pre = pool.conv_state.clone()
        cache_indices = torch.arange(1, self.BS + 1, dtype=torch.int32).cuda()
        seq_lens = torch.full((self.BS,), 24, dtype=torch.int32).cuda()
        qsl = torch.arange(0, self.BS * self.K + 1, self.K, dtype=torch.int32).cuda()
        chunk = torch.randn(self.BS * self.K, dim).cuda()

        inkling_ring_sconv(
            chunk,
            weight,
            state,
            qsl,
            seq_idx_from_cu_seqlens(qsl, self.BS * self.K),
            cache_indices,
            torch.ones(self.BS, dtype=torch.bool).cuda(),
            seq_lens,
            *_inert_publish(self.BS, dim),
            None,
            num_extends=0,
            page_size=128,
        )

        # Outside the channel slice (and other layers): unchanged.
        self.assertTrue(torch.equal(full[:, :, :off], pre[2][:, :, :off]))
        self.assertTrue(torch.equal(full[:, :, off + dim :], pre[2][:, :, off + dim :]))
        self.assertTrue(torch.equal(pool.conv_state[0], pre[0]))
        # Inside: the K chunk rows landed at their positions' ring rows.
        for i in range(self.BS):
            slot = int(cache_indices[i])
            for j in range(self.K):
                self.assertTrue(
                    torch.equal(state[slot, (20 + j) % self.R], chunk[i * self.K + j])
                )


@unittest.skipUnless(torch.cuda.is_available(), "needs a CUDA device")
class TestSconvUnifiedKernel(unittest.TestCase):
    """The single sconv compute kernel: decode = T=1 case, in-kernel ring
    persistence and speculative boundary-checkpoint publish."""

    W = 4
    R = 9
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
        return torch.randn(num_slots, self.R, self.DIM, device="cuda")

    def _ring_prefix(self, state, slot, pre_len):
        """The last W-1 pre-chunk rows read from the ring (zeros before 0)."""
        rows = []
        for p in range(pre_len - (self.W - 1), pre_len):
            if p >= 0:
                rows.append(state[slot, p % self.R])
            else:
                rows.append(torch.zeros(self.DIM, device="cuda"))
        return torch.stack(rows)

    def test_decode_is_t1_case(self):
        """Decode = the unified kernel with T=1 rows: y matches the reference
        conv over [ring history || x_t]; the kernel persists the token's own
        ring row and touches nothing else."""
        from tokenspeed_kernel.ops.conv import inkling_ring_sconv

        weight = self._weight()
        state = self._state()
        pre = state.clone()
        B = 3
        cache_indices = torch.tensor([1, 2, -1], dtype=torch.int32, device="cuda")
        seq_lens = torch.tensor([37, 129, 5], dtype=torch.int32, device="cuda")
        x = torch.randn(B, self.DIM, device="cuda")

        qsl = torch.arange(B + 1, dtype=torch.int32, device="cuda")
        y = inkling_ring_sconv(
            x,
            weight,
            state,
            qsl,
            qsl[:B],
            cache_indices,
            torch.ones(B, dtype=torch.bool, device="cuda"),
            seq_lens,
            *_inert_publish(B, self.DIM),
            None,
            num_extends=0,
            page_size=128,
        )

        expected = pre.clone()
        for b, slot in enumerate([1, 2, None]):
            L = int(seq_lens[b])
            if slot is not None:
                prefix = self._ring_prefix(pre, slot, L - 1)
                expected[slot, (L - 1) % self.R] = x[b]
            else:
                prefix = torch.zeros(self.W - 1, self.DIM, device="cuda")
            ref = self._ref_conv(x[b : b + 1], prefix, weight)
            self.assertTrue(torch.allclose(y[b : b + 1], ref, atol=1e-4), f"req {b}")
        self.assertTrue(torch.equal(state, expected))

    def _publish_setup(self, B, pages=40):
        from tokenspeed_kernel.ops.conv import seq_idx_from_cu_seqlens

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
        window (borrowing ring rows when the boundary falls early in the
        chunk), uncovered/padded requests and untouched pages stay clean —
        all independent of any accept decision."""
        from tokenspeed_kernel.ops.conv import inkling_ring_sconv

        weight = self._weight()
        state = self._state()
        pre = state.clone()
        # S0 = [4, 2, 6, 5] -> boundary L=8 covered for reqs 0 (p*=4) and
        # 2 (p*=2, borrows one ring row); req1 uncovered; req3 padded.
        cache_indices = torch.tensor([1, 2, 3, -1], dtype=torch.int32).cuda()
        seq_lens = torch.tensor([8, 6, 10, 9], dtype=torch.int32).cuda()
        qsl, seq_idx, table, checkpoint = self._publish_setup(4)
        x = torch.randn(4 * self.K, self.DIM, device="cuda")

        inkling_ring_sconv(
            x,
            weight,
            state,
            qsl,
            seq_idx,
            cache_indices,
            torch.ones(4, dtype=torch.bool, device="cuda"),
            seq_lens,
            table,
            checkpoint,
            None,
            num_extends=0,
            page_size=8,
        )

        # req0: p*=4 -> page table[0,0]=11
        self.assertTrue(
            torch.equal(
                checkpoint[11],
                self._ref_window(self._ring_prefix(pre, 1, 4), x[0 : self.K], 4),
            )
        )
        # req2: p*=2 -> page table[2,0]=15, borrows one ring row
        self.assertTrue(
            torch.equal(
                checkpoint[15],
                self._ref_window(
                    self._ring_prefix(pre, 3, 6), x[2 * self.K : 3 * self.K], 2
                ),
            )
        )
        touched = {11, 15}
        for page in range(checkpoint.shape[0]):
            if page not in touched:
                self.assertTrue(bool((checkpoint[page] == -7).all()), f"page {page}")

    def test_publish_prefill_interior_boundaries(self):
        """A prefill chunk spanning several pages publishes EVERY interior
        boundary."""
        from tokenspeed_kernel.ops.conv import (
            inkling_ring_sconv,
            seq_idx_from_cu_seqlens,
        )

        weight = self._weight()
        state = self._state()
        T, page_size = 16, 4
        qsl = torch.tensor([0, T], dtype=torch.int32, device="cuda")
        seq_idx = seq_idx_from_cu_seqlens(qsl, T)
        cache_indices = torch.tensor([1], dtype=torch.int32, device="cuda")
        # Fresh prefill from length 0: boundaries at 4, 8, 12, 16.
        seq_lens = torch.tensor([T], dtype=torch.int32, device="cuda")
        table = torch.arange(21, 21 + 4, dtype=torch.int32, device="cuda").reshape(1, 4)
        checkpoint = torch.full((40, self.W - 1, self.DIM), -7.0, device="cuda")
        x = torch.randn(T, self.DIM, device="cuda")

        inkling_ring_sconv(
            x,
            weight,
            state,
            qsl,
            seq_idx,
            cache_indices,
            torch.zeros(1, dtype=torch.bool, device="cuda"),  # fresh: no taps
            seq_lens,
            table,
            checkpoint,
            None,
            num_extends=1,
            page_size=page_size,
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
        from tokenspeed_kernel.ops.conv import inkling_ring_sconv

        weight = self._weight()
        state = self._state()
        cache_indices = torch.tensor([1], dtype=torch.int32).cuda()
        seq_lens = torch.tensor([8], dtype=torch.int32).cuda()  # p*=4
        qsl, seq_idx, table, _ = self._publish_setup(1)
        field_a = torch.zeros(40, self.W - 1, 2, dtype=torch.bfloat16, device="cuda")
        field_b = torch.zeros(40, self.W - 1, 6, dtype=torch.float8_e5m2, device="cuda")
        x = torch.randn(self.K, self.DIM, device="cuda")

        inkling_ring_sconv(
            x,
            weight,
            state,
            qsl,
            seq_idx,
            cache_indices,
            torch.ones(1, dtype=torch.bool, device="cuda"),
            seq_lens,
            table,
            field_a,
            field_b,
            num_extends=0,
            page_size=8,
        )

        window = self._ref_window(torch.zeros(3, self.DIM, device="cuda"), x, 4)
        self.assertTrue(torch.equal(field_a[11], window[:, :2].to(torch.bfloat16)))
        self.assertTrue(
            torch.equal(
                field_b[11].view(torch.uint8),
                window[:, 2:].to(torch.float8_e5m2).view(torch.uint8),
            )
        )

    def test_publish_overwrites_rejected_round(self):
        """Round 1 publishes candidate rows past its accepted length; round 2
        covering the same boundary overwrites with the committed rows — and
        the ring's own speculative rows feed round 2's borrow correctly."""
        from tokenspeed_kernel.ops.conv import inkling_ring_sconv

        weight = self._weight()
        state = self._state()
        pre = state.clone()
        cache_indices = torch.tensor([1], dtype=torch.int32).cuda()
        seq_lens = torch.tensor([10], dtype=torch.int32).cuda()  # S0=6, p*=2
        qsl, seq_idx, table, checkpoint = self._publish_setup(1)
        ones = torch.ones(1, dtype=torch.bool, device="cuda")
        x1 = torch.randn(self.K, self.DIM, device="cuda")

        inkling_ring_sconv(
            x1,
            weight,
            state,
            qsl,
            seq_idx,
            cache_indices,
            ones,
            seq_lens,
            table,
            checkpoint,
            None,
            num_extends=0,
            page_size=8,
        )
        self.assertTrue(
            torch.equal(
                checkpoint[11],
                self._ref_window(self._ring_prefix(pre, 1, 6), x1, 2),
            )
        )

        # accept=1: the frontier advances by one committed row; S0=7 -> p*=1.
        # No state maintenance — round 1's ring write at position 6 IS the
        # committed row round 2 borrows.
        seq_lens.fill_(11)
        x2 = torch.randn(self.K, self.DIM, device="cuda")
        inkling_ring_sconv(
            x2,
            weight,
            state,
            qsl,
            seq_idx,
            cache_indices,
            ones,
            seq_lens,
            table,
            checkpoint,
            None,
            num_extends=0,
            page_size=8,
        )
        expect = torch.stack([pre[1, 5 % self.R], x1[0], x2[0]])
        self.assertTrue(torch.equal(checkpoint[11], expect))

    def test_cuda_graph_replay(self):
        """All inputs are stable buffers: replays after in-place updates
        reproduce the eager result — ring writes and publish included."""
        from tokenspeed_kernel.ops.conv import inkling_ring_sconv

        weight = self._weight()
        state = self._state()
        cache_indices = torch.tensor([1, 2], dtype=torch.int32).cuda()
        seq_lens = torch.tensor([10, 12], dtype=torch.int32).cuda()
        qsl, seq_idx, table, checkpoint = self._publish_setup(2)
        ones = torch.ones(2, dtype=torch.bool, device="cuda")
        x = torch.randn(2 * self.K, self.DIM, device="cuda")

        def run():
            inkling_ring_sconv(
                x,
                weight,
                state,
                qsl,
                seq_idx,
                cache_indices,
                ones,
                seq_lens,
                table,
                checkpoint,
                None,
                num_extends=0,
                page_size=8,
            )

        run()  # warmup compiles outside capture
        torch.cuda.synchronize()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            run()

        for round_lens in ([10, 12], [11, 16]):
            checkpoint.fill_(-7)
            seq_lens.copy_(torch.tensor(round_lens, dtype=torch.int32, device="cuda"))
            x.copy_(torch.randn_like(x))
            state.copy_(torch.randn_like(state))
            snapshot = state.clone()
            graph.replay()
            torch.cuda.synchronize()
            for req in range(2):
                base = round_lens[req] - self.K
                slot = int(cache_indices[req])
                # Ring rows: all K chunk rows at their positions.
                for j in range(self.K):
                    self.assertTrue(
                        torch.equal(
                            state[slot, (base + j) % self.R],
                            x[req * self.K + j],
                        ),
                        f"ring req {req} row {j} lens {round_lens}",
                    )
                # Publish: first covered boundary, if any.
                p = 8 - base % 8
                if p > self.K:
                    continue
                page = 11 + req * 2 + (base + p) // 8 - 1
                expect = self._ref_window(
                    self._ring_prefix(snapshot, slot, base),
                    x[req * self.K : (req + 1) * self.K],
                    p,
                )
                self.assertTrue(
                    torch.equal(checkpoint[page], expect),
                    f"publish req {req} lens {round_lens}",
                )


@unittest.skipUnless(torch.cuda.is_available(), "needs a CUDA device")
class TestCheckpointMetadata(unittest.TestCase):
    W = 4
    DIM = 8

    def test_draft_lookback_window_keeps_paged_checkpoints(self):
        """enter_draft_lookback_window must carry the paged bridges into the
        widened metadata: the in-kernel publish resolves pages from
        col_page_table by position, so dropping it (the old refusal path)
        would silently disable draft publication on every decode round."""
        from types import SimpleNamespace

        from tokenspeed.runtime.layers.attention.backends.inkling import (
            InklingAttnBackend,
            InklingConvMetadata,
        )

        k, lookback, bs = 4, 2, 1
        backend = InklingAttnBackend.__new__(InklingAttnBackend)
        backend.conv_pool = SimpleNamespace(kernel_size=self.W)
        backend.conv_columns = {
            "block_tokens": 128,
            "group_block_tokens": {"state": 128},
        }
        backend.conv_is_draft = True
        backend.conv_spec_num_tokens = k
        backend._draft_lookback = lookback
        backend.inner = SimpleNamespace()
        table = torch.tensor([[11, 12, 13]], dtype=torch.int32, device="cuda")
        seq_lens = torch.tensor([132], dtype=torch.int32, device="cuda")
        qsl = torch.arange(0, bs * k + 1, k, dtype=torch.int32, device="cuda")
        backend.conv_metadata = InklingConvMetadata(
            query_start_loc=qsl,
            cache_indices=torch.tensor([2], dtype=torch.int32, device="cuda"),
            has_initial_state=torch.ones(bs, dtype=torch.bool, device="cuda"),
            is_decode=False,
            seq_idx=torch.zeros(bs * k, dtype=torch.int32, device="cuda"),
            seq_lens=seq_lens,
            col_page_table={"state": table[:bs]},
        )

        self.assertTrue(backend.enter_draft_lookback_window(bs))

        md = backend.conv_metadata
        self.assertEqual(
            md.query_start_loc.tolist(), [0, k + lookback], "widened chunk"
        )
        self.assertIsNotNone(md.col_page_table)
        self.assertTrue(torch.equal(md.col_page_table["state"], table[:bs]))
        # Same through-chunk end: the boundary the accept landed on (128)
        # is inside the widened chunk (132 - 6, 132], so the kernel's
        # positional publish re-covers it with committed rows.
        self.assertEqual(md.seq_lens.tolist(), [132])

    def test_advance_draft_metadata_keeps_paged_checkpoints(self):
        """The classic per-step rebuild (catch-up -> T=1 decode metadata)
        must also carry the paged bridges, or single-token draft steps
        landing on a boundary silently skip their publish."""
        from types import SimpleNamespace

        from tokenspeed.runtime.layers.attention.backends.inkling import (
            InklingAttnBackend,
            InklingConvMetadata,
        )

        k, bs = 4, 2
        backend = InklingAttnBackend.__new__(InklingAttnBackend)
        backend.inner = SimpleNamespace()
        backend._graph_has_initial_state = None
        backend._decode_qsl = None
        table = torch.tensor([[11, 12], [21, 22]], dtype=torch.int32, device="cuda")
        qsl = torch.arange(0, bs * k + 1, k, dtype=torch.int32, device="cuda")
        backend.conv_metadata = InklingConvMetadata(
            query_start_loc=qsl,
            cache_indices=torch.tensor([2, 3], dtype=torch.int32, device="cuda"),
            has_initial_state=torch.ones(bs, dtype=torch.bool, device="cuda"),
            is_decode=False,
            seq_idx=torch.zeros(bs * k, dtype=torch.int32, device="cuda"),
            seq_lens=torch.tensor([200, 260], dtype=torch.int32, device="cuda"),
            col_page_table={"state": table},
        )

        backend.advance_draft_forward_metadata()

        md = backend.conv_metadata
        self.assertTrue(md.is_decode)
        self.assertTrue(torch.equal(md.col_page_table["state"], table))


if __name__ == "__main__":
    unittest.main()
