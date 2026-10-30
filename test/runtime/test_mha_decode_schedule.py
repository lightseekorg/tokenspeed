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

"""The MHA leaf's per-step decode split schedule on gfx950, through the real
backend, layers and registered kernels."""

import os
import sys

import pytest
import torch
from tokenspeed_kernel.platform import ArchVersion, current_platform

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from ci_system.ci_register import register_cuda_ci  # noqa: E402

register_cuda_ci(est_time=30, suite="runtime-1gpu")

Q_HEADS, KV_HEADS, HEAD_DIM, PAGE_SIZE = 16, 2, 64, 64
MAX_BS, CONTEXT = 4, 1 << 16
WINDOW_LEFT = 511


@pytest.fixture(scope="module", autouse=True)
def gfx950():
    platform = current_platform()
    if not platform.is_amd or platform.arch_version != ArchVersion(9, 5):
        pytest.skip("the split schedule is a gfx950 decode kernel's")


class _Cache:
    """KV storage only; the backend and the kernels stay real."""

    def __init__(self, layers: int, generator: torch.Generator):
        pages = MAX_BS * CONTEXT // PAGE_SIZE
        self.buffers = [
            tuple(
                torch.randn(
                    pages * PAGE_SIZE,
                    KV_HEADS,
                    HEAD_DIM,
                    generator=generator,
                    device="cuda",
                ).bfloat16()
                for _ in range(2)
            )
            for _ in range(layers)
        ]

    def get_key_buffer(self, layer_id):
        return self.buffers[layer_id][0]

    def get_value_buffer(self, layer_id):
        return self.buffers[layer_id][1]


class _Step:
    def __init__(self, *, width: int):
        from tokenspeed.runtime.layers.attention.backends.paged.mha import (
            MHAAttnBackend,
        )
        from tokenspeed.runtime.layers.attention.configs.base import AttnConfig
        from tokenspeed.runtime.layers.attention.configs.mha import MHAConfig
        from tokenspeed.runtime.layers.paged_attention import PagedAttention

        spec = MHAConfig(
            backend_name="mha",
            num_attention_heads=Q_HEADS,
            num_kv_heads=KV_HEADS,
            head_dim=HEAD_DIM,
            attn_tp_size=1,
        )
        config = AttnConfig(
            device="cuda",
            dtype=torch.bfloat16,
            kv_cache_dtype=torch.bfloat16,
            kv_cache_quant_method="none",
            prefix_granularity=PAGE_SIZE,
            context_len=CONTEXT,
            max_bs=MAX_BS,
            speculative_num_draft_tokens=width,
            is_draft=False,
            draft_block_decode=False,
            components=(spec,),
        )
        self.width = width
        self.backend = MHAAttnBackend(config, spec, kernel_page_size=PAGE_SIZE)
        self.backend.init_cuda_graph_state(MAX_BS)
        # Two full-attention layers around a sliding-window one.
        self.layers = [
            PagedAttention(
                Q_HEADS,
                HEAD_DIM,
                HEAD_DIM**-0.5,
                KV_HEADS,
                layer_id,
                sliding_window_size=window,
                rotary_emb=None,
                qk_norm=None,
            )
            for layer_id, window in enumerate((-1, WINDOW_LEFT, -1))
        ]
        generator = torch.Generator(device="cuda").manual_seed(7)
        self.cache = _Cache(len(self.layers), generator)
        pages = CONTEXT // PAGE_SIZE
        self.page_table = (
            torch.randperm(MAX_BS * pages, generator=generator, device="cuda")
            .to(torch.int32)
            .view(MAX_BS, pages)
        )
        self.q = torch.randn(
            MAX_BS * width,
            Q_HEADS,
            HEAD_DIM,
            generator=generator,
            device="cuda",
        ).bfloat16()

    def refresh(self, lengths: list[int], **kwargs) -> None:
        bs = len(lengths)
        self.backend.refresh_decode_metadata(
            bs,
            bs,
            torch.tensor(lengths, dtype=torch.int32, device="cuda"),
            self.page_table[:bs],
            **kwargs,
        )

    def forward(self, layer_id: int, bs: int) -> torch.Tensor:
        return self.backend.forward_decode(
            self.q[: bs * self.width],
            None,
            None,
            self.layers[layer_id],
            None,
            self.cache,
            bs,
        )

    def schedule(self, lengths: list[int]) -> torch.Tensor:
        """The schedule of these lengths, from the registered kernel."""
        from tokenspeed_kernel.ops.attention.mha import mha_decode_schedule

        bs = len(lengths)
        return mha_decode_schedule(
            self.q[: bs * self.width],
            self.cache.get_key_buffer(0).view(-1, PAGE_SIZE, KV_HEADS, HEAD_DIM),
            self.cache.get_value_buffer(0).view(-1, PAGE_SIZE, KV_HEADS, HEAD_DIM),
            torch.tensor(lengths, dtype=torch.int32, device="cuda"),
            self.width,
            window_left=-1,
            logit_cap=0.0,
            sinks=None,
            return_lse=False,
            q_scale=None,
            k_scale=None,
            v_scale=None,
            fp8_kv_scales=None,
            out=None,
            solution=None,
        )

    def held(self, bs: int) -> torch.Tensor:
        return self.backend.decode_schedule_buf[: bs + 1].clone()

    def check(self, output: torch.Tensor, layer_id: int, lengths: list[int]) -> None:
        from tokenspeed_kernel.numerics.torch.attention.mha import (
            torch_dense_attention,
        )

        window = self.layers[layer_id].sliding_window_size
        key, value = (
            buffer.view(-1, PAGE_SIZE, KV_HEADS, HEAD_DIM)
            for buffer in self.cache.buffers[layer_id]
        )
        output = output.view(-1, Q_HEADS, HEAD_DIM)
        for request, length in enumerate(lengths):
            pages = self.page_table[request, : -(-length // PAGE_SIZE)].long()
            rows = slice(request * self.width, (request + 1) * self.width)
            expected, _ = torch_dense_attention(
                self.q[rows],
                key[pages].flatten(0, 1)[:length],
                value[pages].flatten(0, 1)[:length],
                scale=HEAD_DIM**-0.5,
                causal_offset=length - self.width,
                window_left=None if window < 0 else window,
                sinks=None,
            )
            # As the kernel tests: one BF16 rounding of the output, and P
            # rounded to BF16 before P @ V.
            torch.testing.assert_close(
                output[rows].float(), expected, atol=2**-8, rtol=2**-7
            )


@pytest.mark.parametrize("width", [1, 8], ids=["decode", "verify"])
def test_layers_share_one_schedule_until_the_lengths_change(width):
    """The first full-attention layer after the lengths change computes the
    schedule; the sliding layer takes none and leaves it, and the next full
    layer reads it. A drafter's length edit makes the next layer recompute."""
    step = _Step(width=width)
    lengths = [700 * PAGE_SIZE + 5, 3 * PAGE_SIZE, 120 * PAGE_SIZE + 61]
    step.refresh(lengths)
    step.check(step.forward(0, 3), 0, lengths)
    held = step.held(3)
    assert torch.equal(held, step.schedule(lengths))
    step.check(step.forward(1, 3), 1, lengths)
    step.check(step.forward(2, 3), 2, lengths)
    assert torch.equal(step.held(3), held)

    grown = [length + 200 * PAGE_SIZE for length in lengths]
    step.backend.advance_draft_forward_metadata(
        torch.tensor(grown, dtype=torch.int32, device="cuda")
    )
    step.check(step.forward(2, 3), 2, grown)
    assert torch.equal(step.held(3), step.schedule(grown))


def test_replay_recomputes_the_schedule_for_each_step():
    """A captured decode step records the schedule computation, so every
    replay schedules its own lengths, including a different set of long
    requests in the same padded batch."""
    step = _Step(width=8)
    seed = torch.full((MAX_BS,), 8, dtype=torch.int32, device="cuda")
    # Warm the kernels outside the capture, then seed the capture's metadata
    # right before it, as the graph runner does.
    step.backend.init_forward_metadata_capture_cuda_graph(MAX_BS, seed, step.page_table)
    step.forward(0, MAX_BS)
    step.forward(1, MAX_BS)
    torch.cuda.synchronize()
    step.backend.init_forward_metadata_capture_cuda_graph(MAX_BS, seed, step.page_table)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        outputs = [step.forward(layer_id, MAX_BS) for layer_id in (0, 1, 2)]
    for lengths in (
        [1000 * PAGE_SIZE + 33, 8, 8, 300 * PAGE_SIZE],
        [8, 700 * PAGE_SIZE + 1, 64 * PAGE_SIZE, 8],
    ):
        step.refresh(lengths, for_graph_replay=True)
        graph.replay()
        torch.cuda.synchronize()
        assert torch.equal(step.held(MAX_BS), step.schedule(lengths))
        for layer_id, output in enumerate(outputs):
            step.check(output, layer_id, lengths)
