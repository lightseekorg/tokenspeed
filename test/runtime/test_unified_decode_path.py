# Copyright (c) 2026 LightSeek Foundation
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

"""Unified decode path invariants (docs/design/unified_path.md).

CPU-only checks of the single-decode-path contract:

* eager refresh and padded graph refresh write the SAME persistent buffers
  and produce identical live rows (eager is "refresh + forward", replay is
  "refresh + graph.replay()");
* per-bs metadata views are pointer-stable and lazily built for a bs never
  captured (the above-ladder decode path);
* the persistent buffers are sized by max decode bs, not the capture ladder.
"""

from __future__ import annotations

import os
import sys
import unittest
from types import SimpleNamespace

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from ci_system.ci_register import register_cuda_ci

register_cuda_ci(est_time=10, suite="runtime-1gpu")

MAX_DECODE_BS = 8
LADDER_BS = 4  # capture ladder max, deliberately below MAX_DECODE_BS
MAX_NUM_PAGES = 6


def _decode_mode():
    return SimpleNamespace(
        is_mixed=lambda: False,
        is_extend_or_mixed=lambda: False,
        is_idle=lambda: False,
    )


class _TorchCase(unittest.TestCase):
    def setUp(self):
        try:
            import torch
        except (ImportError, ModuleNotFoundError) as exc:
            self.skipTest(f"needs torch: {exc}")
        self.torch = torch


class _MhaCase(_TorchCase):
    def setUp(self):
        super().setUp()
        try:
            from tokenspeed.runtime.layers.attention.backends.mha import (
                MHAAttnBackend,
            )
        except (ImportError, ModuleNotFoundError) as exc:
            self.skipTest(f"needs tokenspeed_kernel: {exc}")
        torch = self.torch
        backend = MHAAttnBackend.__new__(MHAAttnBackend)
        backend.spec_num_tokens = 1
        backend.is_draft = False
        backend.draft_block_decode = False
        backend.state_group_ids = frozenset()
        backend.engine_owned_group_ids = frozenset()
        backend.group_block_granularities = {"full_attention": 2}
        backend.max_num_pages = MAX_NUM_PAGES
        backend.kernel_page_size = 2
        backend.device = "cpu"
        backend.cuda_graph_decode_metadata = {}
        # Buffers sized by MAX_DECODE_BS (the wrapper passes max_decode_bs to
        # init_cuda_graph_state, never the capture-ladder max).
        backend.cuda_graph_page_table = torch.zeros(
            (MAX_DECODE_BS, MAX_NUM_PAGES), dtype=torch.int32
        )
        backend.cuda_graph_seq_lens = torch.ones(MAX_DECODE_BS, dtype=torch.int32)
        backend._init_group_graph_buffers(MAX_DECODE_BS)
        self.backend = backend

    def _tables(self, bs):
        torch = self.torch
        return {
            "full_attention": (
                torch.arange(1, bs * 2 + 1, dtype=torch.int32).reshape(bs, 2)
            )
        }

    def _refresh(self, bs, actual_bs, seq_lens, replay):
        torch = self.torch
        self.backend.refresh_decode_metadata(
            bs,
            actual_bs,
            torch.arange(bs, dtype=torch.int64),
            seq_lens,
            forward_mode=_decode_mode(),
            page_table=torch.zeros((bs, MAX_NUM_PAGES), dtype=torch.int32),
            for_graph_replay=replay,
            block_tables=self._tables(actual_bs),
        )
        return self.backend.forward_decode_metadata


class EagerMatchesPaddedReplayTest(_MhaCase):
    def test_live_rows_identical_across_paths(self):
        torch = self.torch
        seq = torch.tensor([5, 4], dtype=torch.int32)

        # Padded graph refresh: 2 live rows in a 4-row graph batch.
        padded_seq = torch.cat([seq, torch.ones(2, dtype=torch.int32)])
        self._refresh(LADDER_BS, 2, padded_seq, replay=True)
        replay_tables = {
            gid: buf[:2].clone()
            for gid, buf in self.backend.cuda_graph_page_tables.items()
        }
        replay_locs = {
            gid: buf[:2].clone()
            for gid, buf in self.backend.cuda_graph_out_cache_locs.items()
        }
        # Padded rows landed on the null page.
        for buf in self.backend.cuda_graph_page_tables.values():
            self.assertTrue((buf[2:LADDER_BS] == 0).all())

        # Eager refresh at the true bs (unpadded) over the same buffers.
        md = self._refresh(2, 2, seq, replay=False)
        for gid in replay_tables:
            torch.testing.assert_close(
                self.backend.cuda_graph_page_tables[gid][:2], replay_tables[gid]
            )
            torch.testing.assert_close(
                self.backend.cuda_graph_out_cache_locs[gid][:2], replay_locs[gid]
            )
        # Metadata views the SAME persistent storage on both paths.
        self.assertEqual(
            md.page_tables["full_attention"].data_ptr(),
            self.backend.cuda_graph_page_tables["full_attention"].data_ptr(),
        )


class AboveLadderDecodeTest(_MhaCase):
    def test_refresh_serves_bs_above_capture_ladder(self):
        torch = self.torch
        bs = MAX_DECODE_BS  # above LADDER_BS: no graph exists, same refresh
        seq = torch.arange(3, 3 + bs, dtype=torch.int32)
        md = self._refresh(bs, bs, seq, replay=False)
        self.assertEqual(md.seq_lens.shape[0], bs)
        torch.testing.assert_close(self.backend.cuda_graph_seq_lens[:bs], seq)
        # Lazy per-bs views: built once, pointer-stable on the next refresh.
        md2 = self._refresh(bs, bs, seq + 1, replay=False)
        self.assertIs(md2, md)


if __name__ == "__main__":
    unittest.main()
