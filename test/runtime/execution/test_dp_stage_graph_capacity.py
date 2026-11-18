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

"""Run the CED stage planner with real, unequal attention-DP participants."""

import sys
from datetime import timedelta
from pathlib import Path

import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from tokenspeed.runtime.execution.context import ForwardContext
from tokenspeed.runtime.execution.forward_batch_info import (
    CaptureHiddenMode,
    ForwardMode,
)
from tokenspeed.runtime.execution.output_layout import ForwardOutputLayout
from tokenspeed.runtime.execution.prefill_graph import PrefillGraph

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from ci_system.ci_register import register_cuda_ci  # noqa: E402

register_cuda_ci(est_time=20, suite="runtime-1gpu")


class DecoderRows:
    def __init__(self, rows):
        self.view = torch.empty(rows)

    def decoder_rows(self, ctx):
        assert ctx.bs, "an idle peer has no live metadata"
        return self.view.numel()


def _rank(rank, rendezvous):
    dist.init_process_group(
        "gloo",
        init_method=rendezvous,
        rank=rank,
        world_size=4,
        timeout=timedelta(seconds=45),
    )
    try:
        # Only graph selection runs here; the dictionaries stand in for CUDA captures.
        graph = PrefillGraph.__new__(PrefillGraph)
        graph.dp_size = 4
        graph.disable = False
        graph._dp_stage_group = dist.group.WORLD
        graph._captured_hidden_mode = CaptureHiddenMode.NULL
        graph._multimodal_input_embeds = None
        graph._encoders = dict.fromkeys([64, 256])
        graph._decoders = dict.fromkeys([16, 64, 128])
        counts = [200, 6, 0, 64]
        mode = (
            ForwardMode.EXTEND,
            ForwardMode.DECODE,
            ForwardMode.IDLE,
            ForwardMode.EXTEND,
        )[rank]
        ctx = ForwardContext(
            attn_backend=None,
            token_to_kv_pool=None,
            bs=int(bool(counts[rank])),
            num_extends=int(mode.is_extend()),
            input_num_tokens=counts[rank],
            forward_mode=mode,
            output_layout=ForwardOutputLayout(
                int(mode.is_extend()),
                int(mode.is_extend()),
                int(mode.is_decode()),
                6,
            ),
            global_num_tokens=counts,
            global_bs=[1, 1, 0, 1],
            global_num_decodes=[0, 1, 0, 0],
            all_decode_or_idle=False,
        )
        # Identical input counts, different prompt completions. The decoder
        # must use the maximum real tail, including when this rank has zero.
        for tails, buckets in (
            ([128, 6, 0, 64], (256, 128)),
            ([0, 6, 0, 60], (256, 64)),
            ([0, 6, 0, 0], (256, 16)),
        ):
            graph._narrowing = DecoderRows(tails[rank])
            assert graph.select_buckets(ctx) == buckets

        ctx.global_num_tokens = [300, 6, 0, 64]
        ctx.input_num_tokens = ctx.global_num_tokens[rank]
        assert graph.select_buckets(ctx) == (None, 16)
        del graph._decoders[16]
        assert graph.select_buckets(ctx) == (None, 64)

        graph._decoders.clear()
        ctx.global_num_tokens = counts
        ctx.input_num_tokens = counts[rank]
        assert graph.select_buckets(ctx) == (256, None)
        graph._encoders.clear()
        assert graph.select_buckets(ctx) is None
        graph._encoders = {256: None}
        if rank == 1:
            ctx.capture_hidden_mode = CaptureHiddenMode.FULL
        assert graph.select_buckets(ctx) is None

        ctx.all_decode_or_idle = True
        assert graph.select_buckets(ctx) is None
    finally:
        dist.destroy_process_group()


def test_stage_buckets_agree_across_mixed_decode_and_idle_ranks(tmp_path):
    mp.spawn(_rank, args=((tmp_path / "gloo").as_uri(),), nprocs=4)
