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
