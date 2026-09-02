"""Rows past the current batch must never keep a prior batch's page ids.

CUDA-graph replay reads padded_bs rows straight off the draft router's
full-history stack (row i IS batch position i, so the req-pool sink row does
not shield it). A stale id left by an earlier, larger batch routes the
multi-step draft's KV writes into another request's pages; the victim then
mispredicts permanently (#955: M3 EAGLE3 accept 0.665 -> 0.0015). The
router's fill contract carries the scrub now: rows [actual_bs, bs) and every
idle row are the null page 0.
"""

from __future__ import annotations

import os
import sys
import unittest
from types import SimpleNamespace

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from ci_system.ci_register import register_cuda_ci

register_cuda_ci(est_time=5, suite="runtime-1gpu")

from tokenspeed.runtime.execution.forward_batch_info import ForwardMode
from tokenspeed.runtime.execution.model_executor import ModelExecutor
from tokenspeed.runtime.execution.types import DpForwardMetadata
from tokenspeed.runtime.layers.attention.backends.cache_group_geometry import (
    CacheGroupGeometry,
)
from tokenspeed.runtime.layers.attention.backends.router import CacheGroupRouter

FULL = "full_attention"


class _Leaf:
    """Metadata-only leaf stub (the scrub lives in the router's stack)."""

    kernel_page_size = 128
    max_num_pages = 4
    verify_floor = 1

    def init_cuda_graph_state(self, max_bs):
        pass

    def refresh_decode_metadata(self, *args, **kwargs):
        pass

    def set_request_slots(self, req_pool_indices):
        pass

    def init_forward_metadata_capture_cuda_graph(self, *args, **kwargs):
        pass


def _draft_router(rows: int = 8) -> CacheGroupRouter:
    router = CacheGroupRouter(None, is_draft=True, spec_num_tokens=4, device="cpu")
    router.bind(
        CacheGroupGeometry(
            granularities={FULL: 128},
            families={FULL: "history"},
            full_history_group_id=FULL,
            history_block_granularity=128,
        ),
        {FULL: _Leaf()},
    )
    router.init_cuda_graph_state(rows)
    return router


def _refresh(router, bs, actual_bs, table):
    router.refresh_decode_metadata(
        bs,
        actual_bs,
        torch.arange(bs, dtype=torch.int32),
        torch.ones(bs, dtype=torch.int32),
        forward_mode=ForwardMode.DECODE,
        block_tables={FULL: table} if table is not None else {},
    )


class DraftHistoryScrubTest(unittest.TestCase):
    def test_smaller_batch_clears_prior_rows(self):
        router = _draft_router()
        _refresh(router, 8, 6, torch.full((6, 4), 7, dtype=torch.int32))
        _refresh(router, 8, 2, torch.full((2, 4), 3, dtype=torch.int32))
        view = router.draft_history_view()
        self.assertTrue((view.table[2:] == 0).all(), view.table)
        self.assertTrue((view.table[:2] == 3).all())

    def test_padded_replay_rows_resolve_to_the_dummy_slot(self):
        router = _draft_router()
        _refresh(router, 8, 6, torch.full((6, 4), 7, dtype=torch.int32))
        _refresh(router, 4, 1, torch.full((1, 4), 3, dtype=torch.int32))
        out = torch.zeros(8, dtype=torch.int32)
        router.draft_write_locations_uniform(
            out, cache_start=torch.zeros(4, dtype=torch.int32), num_tokens=2
        )
        # Rows 1..3 are padding: their pages are null, so slots are 0.
        self.assertEqual(out[2:].tolist(), [0] * 6)

    def test_idle_refresh_zeroes_every_row(self):
        router = _draft_router()
        _refresh(router, 8, 8, torch.full((8, 4), 9, dtype=torch.int32))
        _refresh(router, 8, 0, None)
        self.assertTrue((router.draft_history_view().table == 0).all())


class IdleReplayScrubTest(unittest.TestCase):
    """A DP rank that served a large batch and then goes idle replays the
    captured drafter graph at padded_bs while another rank decodes; the idle
    refresh must present null rows to the drafter's recorded kernels."""

    def test_idle_replay_sees_zeroed_rows(self):
        padded_bs = 4
        captured = {}
        router = _draft_router()
        _refresh(router, 8, 6, torch.full((6, 4), 7, dtype=torch.int32))

        class _Step:
            def can_run(self, bs, ctx):
                return True

            def padded_bs(self, bs, ctx):
                return padded_bs

            def __call__(self, bs, ctx, sampling_info, **extend_kwargs):
                # The runner's idle metadata prep refreshed the draft router
                # with placeholders before this call; snapshot what the
                # drafter's recorded kernels would read.
                captured["extend_kwargs"] = extend_kwargs
                _refresh(router, padded_bs, 0, None)
                captured["rows"] = router.draft_history_view().table[:padded_bs].clone()

        ex = SimpleNamespace(
            attn_backend=None,
            token_to_kv_pool=None,
            input_buffers=SimpleNamespace(
                req_pool_indices_buf=torch.zeros(8, dtype=torch.int64),
                extend_prefix_lens_buf=torch.zeros(8, dtype=torch.int32),
                extend_prefix_lens_cpu=torch.zeros(8, dtype=torch.int32),
                extend_seq_lens_buf=torch.zeros(8, dtype=torch.int32),
                extend_seq_lens_cpu=torch.zeros(8, dtype=torch.int32),
                fill_dummy_decode_buffers=lambda batch_size, total_tokens: None,
            ),
            runtime_states=SimpleNamespace(
                valid_cache_lengths=torch.zeros(8, dtype=torch.int32),
                vocab_size=32,
            ),
            device="cpu",
            config=SimpleNamespace(output_length=1),
            capturable_grammar=None,
            forward_step=_Step(),
        )
        ModelExecutor.execute_idle_forward(
            ex,
            DpForwardMetadata(
                global_num_tokens=[0],
                global_batch_size=[0],
                global_forward_mode=[int(ForwardMode.IDLE)],
                all_decode_or_idle=True,
                all_extend=False,
                need_idle_forward=True,
            ),
        )
        self.assertTrue((captured["rows"] == 0).all(), captured["rows"])
        # The idle replay hands the runner empty extend slices, never None.
        extend_kwargs = captured["extend_kwargs"]
        self.assertIs(extend_kwargs["extend_with_prefix"], False)
        for name in (
            "extend_prefix_lens",
            "extend_prefix_lens_cpu",
            "extend_seq_lens",
            "extend_seq_lens_cpu",
        ):
            self.assertEqual(extend_kwargs[name].numel(), 0, name)


if __name__ == "__main__":
    unittest.main()
