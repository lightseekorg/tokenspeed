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

import pytest
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def _inert_publish(bs, dim, rows=3):
    """Publish plumbing that never fires: a hole-only two-slot table."""
    table = torch.zeros(bs, 2, dtype=torch.int32, device="cuda")
    ckpt = torch.zeros(1, rows, dim, device="cuda")
    return table, ckpt


class TestInklingCacheContract:
    def test_wrapper_consumes_history_and_checkpoint_state(self):
        from tokenspeed.runtime.layers.attention.backends.specific.inkling import (
            InklingAttnBackend,
        )

        class HistoryBackend:
            cache_consumer_attn_kinds = frozenset({"full_history", "sliding_window"})

        backend = InklingAttnBackend.__new__(InklingAttnBackend)
        backend.inner = HistoryBackend()

        assert (backend.cache_consumer_attn_kinds) == (
            frozenset({"full_history", "sliding_window", "state"})
        )

    def test_remote_restore_pending_is_consumed_once_and_cleared_on_reuse(self):
        from tokenspeed.runtime.layers.attention.backends.specific.inkling import (
            InklingAttnBackend,
            InklingConvStatePool,
        )

        pool = InklingConvStatePool(
            num_layers=1,
            num_slots=5,
            conv_dim=2,
            ring_size=5,
            dtype=torch.float32,
            device="cpu",
        )
        backend = InklingAttnBackend.__new__(InklingAttnBackend)
        backend.conv_pool = pool

        backend.mark_remote_cache_ready(2)
        mask = backend._consume_remote_restore_mask(
            torch.tensor([2, 3, -1], dtype=torch.int32),
            out=torch.zeros(3, dtype=torch.bool),
        )
        assert (mask.tolist()) == ([True, False, False])
        one = torch.zeros(1, dtype=torch.bool)
        assert not (
            backend._consume_remote_restore_mask(torch.tensor([2]), out=one).item()
        )

        backend.mark_remote_cache_ready(2)
        backend.prepare_remote_cache_slots([2])
        assert not (
            backend._consume_remote_restore_mask(torch.tensor([2]), out=one).item()
        )

    @pytest.mark.skipif(not (torch.cuda.is_available()), reason=("needs CUDA"))
    def test_committed_window_round_trip_through_the_live_slot(self):
        """A verify round's commit publishes the window at the accepted
        frontier from the ring into the live slot; a remote restore reads the
        landed slot back into the ring at the prompt end."""
        from tokenspeed.runtime.layers.attention.backends.specific.inkling import (
            InklingAttnBackend,
            InklingConvMetadata,
            InklingConvStatePool,
        )

        ring_size, k = 7, 4
        pool = InklingConvStatePool(
            num_layers=1,
            num_slots=4,
            conv_dim=2,
            ring_size=ring_size,
            dtype=torch.float32,
            device="cuda",
        )
        backend = InklingAttnBackend.__new__(InklingAttnBackend)
        backend.conv_pool = pool
        backend.conv_spec_num_tokens = k
        backend._checkpoint_streams = {}
        checkpoints = torch.zeros(8, 3, 2, device="cuda")
        backend.register_shortconv_checkpoint_stream(
            layer_id=0,
            channel_offset=0,
            dim=2,
            group_id="state",
            buffers=(checkpoints,),
        )
        state = pool.layer_state_wd(0)
        # The ring holds the verify round's K rows at positions 128..131;
        # the live slot is block 5, the input slot was released.
        for position in range(125, 132):
            state[1, position % ring_size] = torch.tensor(
                [float(position), float(-position)], device="cuda"
            )
        backend.conv_decode_metadata = InklingConvMetadata(
            query_start_loc=torch.tensor([0, k], dtype=torch.int32, device="cuda"),
            cache_indices=torch.tensor([1], dtype=torch.int32, device="cuda"),
            has_initial_state=torch.ones(1, dtype=torch.bool, device="cuda"),
            seq_lens=torch.tensor([132], dtype=torch.int32, device="cuda"),
            col_block_table={
                "state": torch.tensor([[0, 5]], dtype=torch.int32, device="cuda")
            },
            num_extends=0,
        )

        # Two of the four candidates were accepted: the frontier is 130.
        backend.commit_state_after_acceptance(
            torch.tensor([2], dtype=torch.int32, device="cuda"),
            num_extends=0,
            accepted_path=None,
        )
        expected = torch.tensor(
            [[127.0, -127.0], [128.0, -128.0], [129.0, -129.0]], device="cuda"
        )
        assert torch.equal(checkpoints[5], expected)
        assert torch.equal(checkpoints[:5], torch.zeros(5, 3, 2, device="cuda"))

        # An extend round commits nothing: its chunk end is the frontier.
        backend.commit_state_after_acceptance(
            torch.tensor([4], dtype=torch.int32, device="cuda"),
            num_extends=1,
            accepted_path=None,
        )
        assert torch.equal(checkpoints[5], expected)

        state[1].zero_()
        restore_md = InklingConvMetadata(
            query_start_loc=torch.tensor([0, 1], dtype=torch.int32, device="cuda"),
            cache_indices=torch.tensor([1], dtype=torch.int32, device="cuda"),
            has_initial_state=torch.ones(1, dtype=torch.bool, device="cuda"),
            seq_lens=torch.tensor([131], dtype=torch.int32, device="cuda"),
            col_block_table={
                "state": torch.tensor([[0, 5]], dtype=torch.int32, device="cuda")
            },
            remote_restore_mask=torch.tensor([True], device="cuda"),
        )
        backend.restore_shortconv_endpoint(state, (checkpoints,), restore_md, "state")
        actual = torch.stack(
            [state[1, position % ring_size] for position in range(127, 130)]
        )
        assert torch.equal(actual, expected)

        # A Decode that hit the prompt-end checkpoint in its own cache restores
        # from the input slot instead.
        state[1].zero_()
        checkpoints[6] = expected + 1
        restore_md.col_block_table["state"][0, 0] = 6
        backend.restore_shortconv_endpoint(state, (checkpoints,), restore_md, "state")
        actual = torch.stack(
            [state[1, position % ring_size] for position in range(127, 130)]
        )
        assert torch.equal(actual, expected + 1)

    @pytest.mark.skipif(not (torch.cuda.is_available()), reason=("needs CUDA"))
    def test_commit_publishes_every_registered_stream(self):
        """One commit publishes every layer's streams across groups, a stream's
        fields splitting its channels; a stream registered after a commit
        joins the next one."""
        from tokenspeed.runtime.layers.attention.backends.specific.inkling import (
            InklingAttnBackend,
            InklingConvMetadata,
            InklingConvStatePool,
        )

        ring_size, k = 7, 4
        pool = InklingConvStatePool(
            num_layers=2,
            num_slots=4,
            conv_dim=6,
            ring_size=ring_size,
            dtype=torch.float32,
            device="cuda",
        )
        torch.manual_seed(3)
        pool.conv_state.copy_(torch.randn_like(pool.conv_state))
        backend = InklingAttnBackend.__new__(InklingAttnBackend)
        backend.conv_pool = pool
        backend.conv_spec_num_tokens = k
        backend._checkpoint_streams = {}
        k_field = torch.zeros(8, 3, 2, device="cuda")
        v_field = torch.zeros(8, 3, 2, device="cuda")
        hidden = torch.zeros(8, 2, 3, 2, device="cuda")

        def register_hidden(layer):
            backend.register_shortconv_checkpoint_stream(
                layer_id=layer,
                channel_offset=4,
                dim=2,
                group_id="hidden",
                buffers=(hidden[:, layer],),
            )

        backend.register_shortconv_checkpoint_stream(
            layer_id=0,
            channel_offset=0,
            dim=4,
            group_id="kv",
            buffers=(k_field, v_field),
        )
        register_hidden(0)
        backend.conv_decode_metadata = InklingConvMetadata(
            query_start_loc=torch.tensor([0, k], dtype=torch.int32, device="cuda"),
            cache_indices=torch.tensor([2], dtype=torch.int32, device="cuda"),
            has_initial_state=torch.ones(1, dtype=torch.bool, device="cuda"),
            seq_lens=torch.tensor([130], dtype=torch.int32, device="cuda"),
            col_block_table={
                "kv": torch.tensor([[0, 5]], dtype=torch.int32, device="cuda"),
                "hidden": torch.tensor([[0, 6]], dtype=torch.int32, device="cuda"),
            },
            num_extends=0,
        )

        def window(layer, lo, hi, frontier):
            return torch.stack(
                [
                    pool.conv_state[layer, 2, position % ring_size, lo:hi]
                    for position in range(frontier - 3, frontier)
                ]
            )

        def commit(accepted):
            backend.commit_state_after_acceptance(
                torch.tensor([accepted], dtype=torch.int32, device="cuda"),
                num_extends=0,
                accepted_path=None,
            )

        commit(3)  # frontier 130 - 4 + 3
        assert torch.equal(k_field[5], window(0, 0, 2, 129))
        assert torch.equal(v_field[5], window(0, 2, 4, 129))
        assert torch.equal(hidden[6, 0], window(0, 4, 6, 129))
        assert not hidden[6, 1].any()

        register_hidden(1)
        commit(1)
        assert torch.equal(hidden[6, 0], window(0, 4, 6, 127))
        assert torch.equal(hidden[6, 1], window(1, 4, 6, 127))
        assert torch.equal(k_field[5], window(0, 0, 2, 127))
        assert not k_field[:5].any() and not hidden[:6].any()


@pytest.mark.skipif(not (torch.cuda.is_available()), reason=("needs a CUDA device"))
class TestInklingConvRingState:
    W = 4  # sconv kernel size (W-1 = 3 history taps)
    R = 9  # ring rows: >= (W-1) + K
    DIM = 8
    BS = 5
    K = 4  # spec_num_tokens (draft tokens per verify round)
    LAYERS = 3

    def _make_pool(self):
        from tokenspeed.runtime.layers.attention.backends.specific.inkling import (
            InklingConvStatePool,
        )

        pool = InklingConvStatePool(
            num_layers=self.LAYERS,
            num_slots=self.BS + 2,
            conv_dim=self.DIM,
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
        from tokenspeed.runtime.layers.attention.backends.specific.inkling import (
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
        assert (len(backend._checkpoint_streams)) == (1)
        with pytest.raises((RuntimeError)):
            backend.register_shortconv_checkpoint_stream(
                layer_id=0,
                channel_offset=0,
                dim=self.DIM,
                group_id="state",
                buffers=(buf.clone(),),
            )
        with pytest.raises(
            ValueError, match=f"cover {self.DIM + 1} channels, expected {self.DIM}"
        ):
            backend.register_shortconv_checkpoint_stream(
                layer_id=1,
                channel_offset=0,
                dim=self.DIM,
                group_id="state",
                buffers=(buf, buf[..., :1]),
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
            publish_live=False,
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
                _pytest_expr_5 = torch.equal(window, expect)
                _pytest_msg_6 = f"req {i} accept {accept}"
                assert _pytest_expr_5, _pytest_msg_6

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
            publish_live=False,
        )

        expected = pre.clone()
        for i in (0, 1, 3, 4):
            slot = int(cache_indices[i])
            L0 = int(seq_lens[i]) - self.K
            for j in range(self.K):
                expected[slot, (L0 + j) % self.R] = chunk[i * self.K + j]
        assert torch.equal(state, expected)

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
            publish_live=False,
        )

        # Outside the channel slice (and other layers): unchanged.
        assert torch.equal(full[:, :, :off], pre[2][:, :, :off])
        assert torch.equal(full[:, :, off + dim :], pre[2][:, :, off + dim :])
        assert torch.equal(pool.conv_state[0], pre[0])
        # Inside: the K chunk rows landed at their positions' ring rows.
        for i in range(self.BS):
            slot = int(cache_indices[i])
            for j in range(self.K):
                assert torch.equal(
                    state[slot, (20 + j) % self.R], chunk[i * self.K + j]
                )


@pytest.mark.skipif(not (torch.cuda.is_available()), reason=("needs a CUDA device"))
class TestSconvUnifiedKernel:
    """The sconv compute kernels: decode = T=1 case, in-kernel ring
    persistence and chunk-end publication into the live slot."""

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
            publish_live=False,
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
            _pytest_expr_1 = torch.allclose(y[b : b + 1], ref, atol=1e-4)
            _pytest_msg_2 = f"req {b}"
            assert _pytest_expr_1, _pytest_msg_2
        assert torch.equal(state, expected)

    def _publish_setup(self, B, pages=40):
        from tokenspeed_kernel.ops.conv import seq_idx_from_cu_seqlens

        k = self.K
        qsl = torch.arange(0, B * k + 1, k, dtype=torch.int32, device="cuda")
        seq_idx = seq_idx_from_cu_seqlens(qsl, B * k)
        # Two slots per request: a released input slot and a live block.
        table = torch.zeros(B, 2, dtype=torch.int32, device="cuda")
        table[:, 1] = torch.arange(11, 11 + B, dtype=torch.int32, device="cuda")
        checkpoint = torch.full((pages, self.W - 1, self.DIM), -7.0, device="cuda")
        return qsl, seq_idx, table, checkpoint

    def test_publish_draft_chunk_ends(self):
        """Draft-shaped chunks (uniform K, re-anchored at the committed
        frontier) publish each request's chunk-end window to its live slot,
        borrowing ring rows when the chunk is shorter than W-1; padded
        requests and untouched pages stay clean."""
        from tokenspeed_kernel.ops.conv import inkling_ring_sconv

        weight = self._weight()
        state = self._state()
        pre = state.clone()
        cache_indices = torch.tensor([1, 2, 3, -1], dtype=torch.int32).cuda()
        seq_lens = torch.tensor([8, 6, 10, 9], dtype=torch.int32).cuda()
        qsl, seq_idx, table, checkpoint = self._publish_setup(4)
        table[2, 1] = 0  # req2 has no live block
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
            publish_live=True,
        )

        for req, slot in ((0, 1), (1, 2)):
            _pytest_expr_3 = torch.equal(
                checkpoint[11 + req],
                self._ref_window(
                    self._ring_prefix(pre, slot, int(seq_lens[req]) - self.K),
                    x[req * self.K : (req + 1) * self.K],
                    self.K,
                ),
            )
            _pytest_msg_4 = f"req {req}"
            assert _pytest_expr_3, _pytest_msg_4
        touched = {11, 12}
        for page in range(checkpoint.shape[0]):
            if page not in touched:
                _pytest_expr_7 = bool((checkpoint[page] == -7).all())
                _pytest_msg_8 = f"page {page}"
                assert _pytest_expr_7, _pytest_msg_8

    def test_verify_publishes_nothing_in_kernel(self):
        """A target verify's chunk end is speculative: the kernel leaves the
        live slot alone and the commit publishes the accepted window."""
        from tokenspeed_kernel.ops.conv import inkling_ring_sconv

        weight = self._weight()
        state = self._state()
        cache_indices = torch.tensor([1, 2], dtype=torch.int32).cuda()
        seq_lens = torch.tensor([8, 6], dtype=torch.int32).cuda()
        qsl, seq_idx, table, checkpoint = self._publish_setup(2)
        x = torch.randn(2 * self.K, self.DIM, device="cuda")

        inkling_ring_sconv(
            x,
            weight,
            state,
            qsl,
            seq_idx,
            cache_indices,
            torch.ones(2, dtype=torch.bool, device="cuda"),
            seq_lens,
            table,
            checkpoint,
            None,
            num_extends=0,
            publish_live=False,
        )
        assert bool((checkpoint == -7).all())

    def test_publish_prefill_chunk_end_only(self):
        """A prefill chunk spanning many positions publishes exactly its
        chunk-end window; no interior position is a publication point."""
        from tokenspeed_kernel.ops.conv import (
            inkling_ring_sconv,
            seq_idx_from_cu_seqlens,
        )

        weight = self._weight()
        state = self._state()
        T = 15
        qsl = torch.tensor([0, T], dtype=torch.int32, device="cuda")
        seq_idx = seq_idx_from_cu_seqlens(qsl, T)
        cache_indices = torch.tensor([1], dtype=torch.int32, device="cuda")
        seq_lens = torch.tensor([T], dtype=torch.int32, device="cuda")
        table = torch.tensor([[0, 21]], dtype=torch.int32, device="cuda")
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
            publish_live=True,
        )

        zeros = torch.zeros(self.W - 1, self.DIM, device="cuda")
        assert torch.equal(checkpoint[21], self._ref_window(zeros, x, T))
        for page in range(checkpoint.shape[0]):
            if page != 21:
                _pytest_expr_9 = bool((checkpoint[page] == -7).all())
                _pytest_msg_10 = f"page {page}"
                assert _pytest_expr_9, _pytest_msg_10

    def test_publish_two_field_split_and_fp8(self):
        """Fused K+V split across two fields, and an fp8 destination: the
        kernel's store-side casts must match torch's."""
        from tokenspeed_kernel.ops.conv import inkling_ring_sconv

        weight = self._weight()
        state = self._state()
        cache_indices = torch.tensor([1], dtype=torch.int32).cuda()
        seq_lens = torch.tensor([8], dtype=torch.int32).cuda()
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
            publish_live=True,
        )

        window = self._ref_window(torch.zeros(3, self.DIM, device="cuda"), x, self.K)
        assert torch.equal(field_a[11], window[:, :2].to(torch.bfloat16))
        assert torch.equal(
            field_b[11].view(torch.uint8),
            window[:, 2:].to(torch.float8_e5m2).view(torch.uint8),
        )

    def test_cuda_graph_replay(self):
        """All inputs are stable buffers: replays after in-place updates
        reproduce the eager result -- ring writes and publish included."""
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
                publish_live=True,
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
                    _pytest_expr_13 = torch.equal(
                        state[slot, (base + j) % self.R],
                        x[req * self.K + j],
                    )
                    _pytest_msg_14 = f"ring req {req} row {j} lens {round_lens}"
                    assert _pytest_expr_13, _pytest_msg_14
                # Publish: the chunk-end window into the live slot.
                expect = self._ref_window(
                    self._ring_prefix(snapshot, slot, base),
                    x[req * self.K : (req + 1) * self.K],
                    self.K,
                )
                _pytest_expr_11 = torch.equal(checkpoint[11 + req], expect)
                _pytest_msg_12 = f"publish req {req} lens {round_lens}"
                assert _pytest_expr_11, _pytest_msg_12


@pytest.mark.skipif(not (torch.cuda.is_available()), reason=("needs a CUDA device"))
class TestCheckpointMetadata:
    W = 4
    DIM = 8

    def test_draft_frontier_window_reanchors_conv_metadata(self):
        """update_draft_forward_metadata keeps the k-row chunk shape and moves
        only its anchor: seq_lens becomes the committed frontier, the
        checkpoint tables ride through (the draft chunk's end is the frontier,
        so its publish lands the committed window), and the inner backend
        gets the same frontier for its seq_lens/write-loc re-anchor."""
        from types import SimpleNamespace

        from tokenspeed.runtime.layers.attention.backends.specific.inkling import (
            InklingAttnBackend,
            InklingConvMetadata,
        )

        k, bs = 4, 1
        backend = InklingAttnBackend.__new__(InklingAttnBackend)
        inner_calls = []
        backend.inner = SimpleNamespace(
            update_draft_forward_metadata=lambda f: inner_calls.append(f)
        )
        table = torch.tensor([[0, 12]], dtype=torch.int32, device="cuda")
        qsl = torch.arange(0, bs * k + 1, k, dtype=torch.int32, device="cuda")
        backend.conv_decode_metadata = InklingConvMetadata(
            query_start_loc=qsl,
            cache_indices=torch.tensor([2], dtype=torch.int32, device="cuda"),
            has_initial_state=torch.ones(bs, dtype=torch.bool, device="cuda"),
            seq_idx=torch.zeros(bs * k, dtype=torch.int32, device="cuda"),
            seq_lens=torch.tensor([132], dtype=torch.int32, device="cuda"),
            col_block_table={"state": table[:bs]},
        )
        frontier = torch.tensor([130], dtype=torch.int32, device="cuda")

        backend.update_draft_forward_metadata(frontier)

        md = backend.conv_decode_metadata
        assert (md.query_start_loc.tolist()) == ([0, k]), "same k-row chunk"
        assert (md.seq_lens.tolist()) == ([130]), "chunk end at the frontier"
        assert (md.col_block_table) is not None
        assert torch.equal(md.col_block_table["state"], table[:bs])
        assert (len(inner_calls)) == (1)
        assert (inner_calls[0].tolist()) == ([130])

    def test_update_draft_forward_metadata_reanchors_seq_lens(self):
        """The MTP re-anchor must replace the leaf's decode seq_lens with the
        committed frontier (the drafter supplies its own write locations)."""
        from tokenspeed.runtime.layers.attention.backends.paged.mha import (
            MHAAttnBackend,
            MHADecodeMetadata,
        )

        host = MHAAttnBackend.__new__(MHAAttnBackend)
        host.kernel_page_size = 2
        host.spec_num_tokens = 4
        host.seq_lens_buf = torch.tensor([8], dtype=torch.int32, device="cuda")
        host.forward_decode_metadata = MHADecodeMetadata(
            page_table=torch.tensor([[7, 8, 9]], dtype=torch.int32, device="cuda"),
            seq_lens=host.seq_lens_buf[:1],
        )
        frontier = torch.tensor([6], dtype=torch.int32, device="cuda")

        host.advance_draft_forward_metadata(frontier)

        assert (host.forward_decode_metadata.seq_lens.tolist()) == ([6])


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, *sys.argv[1:]]))
