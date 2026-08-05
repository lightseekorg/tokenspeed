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

    def test_checkpoint_publication_masks_padded_rows_before_indexing(self):
        from tokenspeed.runtime.layers.attention.backends.inkling import (
            InklingAttnBackend,
            InklingConvMetadata,
            ShortConvCheckpointMetadata,
        )

        x = torch.arange(6, dtype=torch.float32).view(3, 2)
        state = torch.zeros((2, 2, 2), dtype=torch.float32)
        checkpoint = torch.full((3, 2, 2), -1, dtype=torch.float32)
        metadata = InklingConvMetadata(
            query_start_loc=torch.tensor([0, 3, 3], dtype=torch.int32),
            cache_indices=torch.tensor([1, -1], dtype=torch.int32),
            has_initial_state=torch.tensor([True, False]),
            is_decode=False,
            checkpoints=ShortConvCheckpointMetadata(
                restore_pages={"state": torch.zeros(2, dtype=torch.int32)},
                write_pages={"state": torch.tensor([1, 0], dtype=torch.int32)},
                write_requests=torch.arange(2),
                packed_rows=torch.tensor([[1, 2], [3, 3]], dtype=torch.int64),
                prior_state_rows=torch.tensor([[0, 1], [0, 1]], dtype=torch.int64),
                packed_row_mask=torch.tensor([[True, True], [False, False]]),
            ),
        )

        InklingAttnBackend.publish_shortconv_checkpoints(
            x,
            state,
            (checkpoint,),
            metadata,
            "state",
        )

        self.assertTrue(torch.equal(checkpoint[1], x[1:3]))
        self.assertTrue(torch.equal(checkpoint[0], torch.zeros_like(checkpoint[0])))


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

    def test_verify_publishes_only_accepted_aligned_endpoint(self):
        from tokenspeed.runtime.layers.attention.backends.inkling import (
            InklingAttnBackend,
            InklingConvMetadata,
        )

        pool = self._make_pool()
        backend = InklingAttnBackend.__new__(InklingAttnBackend)
        backend.conv_pool = pool
        backend.conv_spec_num_tokens = self.K
        backend.conv_columns = {
            "mode": "checkpoint",
            "block_tokens": 4,
            "group_block_tokens": {"state": 4},
        }
        backend._verify_stash = torch.randn(
            self.LAYERS,
            3 * self.K,
            self.DIM,
            device="cuda",
        )
        table = torch.tensor(
            [[11, 12, 13], [21, 22, 23], [31, 32, 33]],
            dtype=torch.int32,
            device="cuda",
        )
        cache_indices = torch.tensor([1, 2, 3], dtype=torch.int32, device="cuda")
        backend.conv_metadata = InklingConvMetadata(
            query_start_loc=torch.arange(0, 3 * self.K + 1, self.K, device="cuda"),
            cache_indices=cache_indices,
            has_initial_state=torch.ones(3, dtype=torch.bool, device="cuda"),
            is_decode=False,
            update_mode="stash",
            tokens_per_req=self.K,
            col_page_table={"state": table},
            col_seq_lens=torch.full((3,), 8, dtype=torch.int32, device="cuda"),
        )
        # LCM checkpoint views are backed by tensors allocated while the model
        # executor is in inference mode. The post-verify hook runs afterward,
        # outside that context.
        with torch.inference_mode():
            checkpoint = torch.full(
                (40, self.W - 1, self.DIM),
                -7,
                dtype=pool.conv_state.dtype,
                device="cuda",
            )
            checkpoint[0].zero_()
        backend._checkpoint_streams = {(0, 0, self.DIM, "state"): (checkpoint,)}

        accept = torch.tensor([4, 3, 0], dtype=torch.int32, device="cuda")
        backend.update_mamba_state_after_mtp_verify(accept, None)

        self.assertTrue(torch.equal(checkpoint[12], pool.conv_state[0, 1]))
        self.assertTrue(bool((checkpoint[22] == -7).all()))
        self.assertTrue(bool((checkpoint[31] == -7).all()))
        self.assertTrue(bool((checkpoint[0] == 0).all()))

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


if __name__ == "__main__":
    unittest.main()
