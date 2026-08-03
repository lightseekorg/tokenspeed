"""Draft decode-window lookback: lagged conv window recurrence (§14 Step 2).

Positions are encoded as float activation values so window contents can be
checked against the exact position ranges each window must end at:

- main window ends at the committed frontier - 1,
- lag window ends ``D + 1`` positions behind it (what a lookback row follows).
"""

import unittest
from types import SimpleNamespace

import torch

from tokenspeed.runtime.layers.attention.backends.inkling import InklingAttnBackend


def _acts(start: int, count: int) -> torch.Tensor:
    return torch.arange(start, start + count, dtype=torch.float32).view(count, 1)


class TestInklingConvLookback(unittest.TestCase):
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
