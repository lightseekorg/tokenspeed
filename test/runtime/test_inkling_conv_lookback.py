"""Draft decode-window lookback: lagged conv window recurrence (§14 Step 2).

Positions are encoded as float activation values so window contents can be
checked against the exact position ranges each window must end at:

- main window ends at the committed frontier - 1,
- lag window ends ``D + 1`` positions behind it (what a lookback row follows).
"""

import unittest
from types import SimpleNamespace
from unittest.mock import Mock

import torch

from tokenspeed.runtime.flat_cache_tables import resolve_cache_table_binding
from tokenspeed.runtime.layers.attention.backends.flat_groups import (
    FlatCacheGroupsMixin,
)
from tokenspeed.runtime.layers.attention.backends.inkling import InklingAttnBackend
from tokenspeed.runtime.layers.attention.backends.mha import MHADecodeMetadata


def _acts(start: int, count: int) -> torch.Tensor:
    return torch.arange(start, start + count, dtype=torch.float32).view(count, 1)


class _FlatLookbackBackend(FlatCacheGroupsMixin):
    page_size = 128
    spec_num_tokens = 4
    flat_draft_lookback = 2


class TestInklingConvLookback(unittest.TestCase):
    def _flat_lookback_backend(self, *, with_static_locs: bool):
        bs = 2
        backend = _FlatLookbackBackend()
        page_tables = {"full_attention": torch.zeros((bs, 2), dtype=torch.int32)}
        captured = MHADecodeMetadata(
            page_table=None,
            seq_lens=torch.full((bs,), 4, dtype=torch.int32),
            page_tables=page_tables,
            block_table_base_offsets={
                "full_attention": torch.zeros(bs, dtype=torch.int32)
            },
            out_cache_locs={"full_attention": torch.zeros(bs * 4, dtype=torch.int32)},
        )
        backend.cuda_graph_decode_metadata = {bs: captured}
        backend.cuda_graph_flat_lookback_locs = (
            {"full_attention": torch.arange(bs * (4 + 2), dtype=torch.int32)}
            if with_static_locs
            else {}
        )
        backend.forward_decode_metadata = captured
        backend._compute_flat_decode_out_cache_locs = Mock(
            side_effect=AssertionError("captured metadata entered eager gather")
        )
        return backend, bs, captured

    def test_graph_lookback_uses_captured_metadata_identity(self):
        backend, bs, captured = self._flat_lookback_backend(with_static_locs=True)

        self.assertIs(backend.forward_decode_metadata, captured)
        self.assertTrue(backend.flat_enter_draft_lookback(bs))

        expected = backend.cuda_graph_flat_lookback_locs["full_attention"]
        self.assertIs(backend.forward_decode_metadata, captured)
        actual = backend.forward_decode_metadata.out_cache_locs["full_attention"]
        self.assertTrue(actual.is_set_to(expected))
        backend._compute_flat_decode_out_cache_locs.assert_not_called()

    def test_graph_lookback_missing_static_locs_fails_before_eager_gather(self):
        backend, bs, _ = self._flat_lookback_backend(with_static_locs=False)

        with self.assertRaisesRegex(RuntimeError, "static lookback"):
            backend.flat_enter_draft_lookback(bs)

        backend._compute_flat_decode_out_cache_locs.assert_not_called()

    def test_flat_binding_preserves_group_identity(self):
        backend = InklingAttnBackend.__new__(InklingAttnBackend)
        backend.inner = SimpleNamespace(
            uses_paged_cache_groups=False,
            uses_flat_cache_groups=True,
        )
        group_ids = ("full_attention", "sliding_attention")
        pool = SimpleNamespace(
            flat_memory_plan=None,
            paged_cache_group_specs=tuple(
                SimpleNamespace(group_id=group_id) for group_id in group_ids
            ),
        )

        binding = resolve_cache_table_binding(
            backend=backend,
            pool=pool,
            flat_scheduler_active=True,
        )

        self.assertEqual(binding.kind, "flat")
        self.assertEqual(binding.group_ids, group_ids)
        self.assertTrue(binding.group_keyed_cache_locs)

    def test_decode_window_recurrence_tracks_both_windows(self):
        # w1=3, D=2, k=4: chunk rows cover positions [vc-D, vc+k) per round.
        w1, lookback, k = 3, 2, 4
        tokens_per_req = k + lookback
        idx = torch.tensor([1], dtype=torch.int32)
        main = torch.zeros(2, w1, 1)
        lag = torch.zeros(2, w1, 1)
        # Post-extend seed at frontier vc=10: main ends 9, lag ends 7.
        main[1] = _acts(7, w1)
        lag[1] = _acts(5, w1)

        vc = 10
        for accept in (2, 4, 1):
            chunk = _acts(vc - lookback, tokens_per_req)
            a = torch.tensor([accept])
            # Main first: both writes must read the pre-update lag window.
            InklingAttnBackend._write_window_from(
                main, lag, chunk, idx, tokens_per_req, a + lookback
            )
            InklingAttnBackend._write_window_from(
                lag, lag, chunk, idx, tokens_per_req, a
            )
            vc += accept
            self.assertEqual(
                main[1].view(-1).tolist(), _acts(vc - w1, w1).view(-1).tolist()
            )
            self.assertEqual(
                lag[1].view(-1).tolist(),
                _acts(vc - lookback - w1, w1).view(-1).tolist(),
            )

    def test_write_lag_extend_advances_and_borrows_on_short_chunks(self):
        backend = InklingAttnBackend.__new__(InklingAttnBackend)
        backend._draft_lookback = 2
        backend._draft_lag_conv_state = torch.zeros(1, 2, 3, 1)

        # Request 0: 6-row chunk from position 10; request 1: 2-row chunk
        # from position 20 (shorter than D + W-1, borrows main rows).
        state = torch.zeros(2, 3, 1)
        state[0] = _acts(7, 3)  # main ends 9
        state[1] = _acts(17, 3)  # main ends 19
        x = torch.cat([_acts(10, 6), _acts(20, 2)])
        md = SimpleNamespace(
            query_start_loc=torch.tensor([0, 6, 8], dtype=torch.int32),
            cache_indices=torch.tensor([0, 1], dtype=torch.int32),
            has_initial_state=torch.tensor([True, True]),
        )

        backend._write_lag_extend(state, x, md, 0, 0, 1)

        lag = backend._draft_lag_conv_state[0]
        # Chunk ends: 16 and 22 -> lag windows end at 13 and 19.
        self.assertEqual(lag[0].view(-1).tolist(), [11.0, 12.0, 13.0])
        self.assertEqual(lag[1].view(-1).tolist(), [17.0, 18.0, 19.0])


if __name__ == "__main__":
    unittest.main()
