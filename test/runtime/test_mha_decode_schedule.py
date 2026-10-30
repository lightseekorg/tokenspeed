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

"""The MHA leaf's split schedule on gfx950: verify layers share one
balanced schedule per forward, computed by the first of them (inside the
captured graph under replay), through the real backend, layers and
registered kernels."""

import os
import sys

import pytest
import torch
from tokenspeed_kernel.platform import ArchVersion, current_platform

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from ci_system.ci_register import register_cuda_ci  # noqa: E402

register_cuda_ci(est_time=30, suite="runtime-1gpu")

KV_HEADS, HEAD_DIM, PAGE_SIZE = 2, 64, 64
MAX_BS, CONTEXT = 4, 1 << 16
WINDOW_LEFT = 511
# Layer 3's KV heads differ from the leaf's, so it keeps its default kernel
# (the forward's schedule is for the leaf's shape).
WIDE_KV_HEADS = 4


@pytest.fixture(scope="module", autouse=True)
def gfx950():
    platform = current_platform()
    if not platform.is_amd or platform.arch_version != ArchVersion(9, 5):
        pytest.skip("the split schedule is a gfx950 decode kernel's")


class _Cache:
    """KV storage only; the backend and the kernels stay real."""

    def __init__(self, kv_heads: list[int], generator: torch.Generator):
        pages = MAX_BS * CONTEXT // PAGE_SIZE
        self.buffers = [
            tuple(
                torch.randn(
                    pages * PAGE_SIZE,
                    heads,
                    HEAD_DIM,
                    generator=generator,
                    device="cuda",
                ).bfloat16()
                for _ in range(2)
            )
            for heads in kv_heads
        ]

    def get_key_buffer(self, layer_id):
        return self.buffers[layer_id][0]

    def get_value_buffer(self, layer_id):
        return self.buffers[layer_id][1]


class _Step:
    def __init__(self, *, width: int, q_heads: int = 16, is_draft: bool = False):
        from tokenspeed.runtime.layers.attention.backends.paged.mha import (
            MHAAttnBackend,
        )
        from tokenspeed.runtime.layers.attention.configs.base import AttnConfig
        from tokenspeed.runtime.layers.attention.configs.mha import MHAConfig
        from tokenspeed.runtime.layers.paged_attention import PagedAttention

        spec = MHAConfig(
            backend_name="mha",
            num_attention_heads=q_heads,
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
            is_draft=is_draft,
            draft_block_decode=False,
            components=(spec,),
        )
        self.width = width
        self.q_heads = q_heads
        self.backend = MHAAttnBackend(config, spec, kernel_page_size=PAGE_SIZE)
        self.backend.init_cuda_graph_state(MAX_BS)
        # Two full-attention layers around a sliding-window one, and a
        # full-attention layer of another shape.
        layouts = ((-1, KV_HEADS), (WINDOW_LEFT, KV_HEADS), (-1, KV_HEADS))
        layouts += ((-1, WIDE_KV_HEADS),)
        self.layers = [
            PagedAttention(
                q_heads,
                HEAD_DIM,
                HEAD_DIM**-0.5,
                kv_heads,
                layer_id,
                sliding_window_size=window,
                rotary_emb=None,
                qk_norm=None,
            )
            for layer_id, (window, kv_heads) in enumerate(layouts)
        ]
        generator = torch.Generator(device="cuda").manual_seed(7)
        self.cache = _Cache([heads for _, heads in layouts], generator)
        pages = CONTEXT // PAGE_SIZE
        self.page_table = (
            torch.randperm(MAX_BS * pages, generator=generator, device="cuda")
            .to(torch.int32)
            .view(MAX_BS, pages)
        )
        self.q = torch.randn(
            MAX_BS * width,
            q_heads,
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

    def fresh_schedule(self, lengths: list[int]) -> torch.Tensor:
        """The schedule the kernel library computes for these lengths."""
        from tokenspeed_kernel.ops.attention.mha import mha_decode_schedule

        key, value = self.cache.buffers[0]
        return mha_decode_schedule(
            q=self.q[: len(lengths) * self.width],
            k_cache=key.view(-1, PAGE_SIZE, KV_HEADS, HEAD_DIM),
            v_cache=value.view(-1, PAGE_SIZE, KV_HEADS, HEAD_DIM),
            cache_seqlens=torch.tensor(lengths, dtype=torch.int32, device="cuda"),
            max_seqlen_q=self.width,
        )

    def check(self, output: torch.Tensor, layer_id: int, lengths: list[int]) -> None:
        from tokenspeed_kernel.numerics.torch.attention.mha import (
            torch_dense_attention,
        )

        window = self.layers[layer_id].sliding_window_size
        heads = self.layers[layer_id].tp_k_head_num
        key, value = (
            buffer.view(-1, PAGE_SIZE, heads, HEAD_DIM)
            for buffer in self.cache.buffers[layer_id]
        )
        output = output.view(-1, self.q_heads, HEAD_DIM)
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


LONG = 1000 * PAGE_SIZE + 5
UNEVEN = [LONG, 3 * PAGE_SIZE, 8, 120 * PAGE_SIZE + 61]


@pytest.fixture
def schedule_calls(monkeypatch):
    """Record the batch of each schedule the backend computes; the kernels
    stay real."""
    from tokenspeed.runtime.layers.attention.backends.paged import mha

    calls = []

    def counted(*args, **kwargs):
        calls.append(kwargs["cache_seqlens"].shape[0])
        return real(*args, **kwargs)

    real = mha.mha_decode_schedule
    monkeypatch.setattr(mha, "mha_decode_schedule", counted)
    return calls


def test_verify_layers_share_one_schedule_per_forward(schedule_calls):
    """The first verify layer of a forward computes the schedule from the
    lengths the refresh wrote and the next full-attention layer of the
    leaf's shape reuses it; the sliding layer and the layer of another shape
    take their default kernels. The next refresh makes it stale again."""
    step = _Step(width=8)
    step.refresh(UNEVEN)
    assert schedule_calls == []
    for layer_id in range(4):
        step.check(step.forward(layer_id, 4), layer_id, UNEVEN)
    assert schedule_calls == [4]
    assert torch.equal(
        step.backend.decode_schedule_buf[:5], step.fresh_schedule(UNEVEN)
    )

    grown = [length + 16 * PAGE_SIZE for length in UNEVEN]
    step.refresh(grown)
    step.check(step.forward(0, 4), 0, grown)
    step.check(step.forward(2, 4), 2, grown)
    assert schedule_calls == [4, 4]
    assert torch.equal(step.backend.decode_schedule_buf[:5], step.fresh_schedule(grown))


@pytest.mark.parametrize(
    "q_heads, is_draft, reads",
    [
        (16, False, False),  # plain decode: 8 rows per KV head, one wave
        (64, False, True),  # 32 rows per KV head: a multi-wave tile
        (64, True, False),  # a draft keeps the default whatever its shape
    ],
    ids=["decode", "wide-decode", "draft"],
)
def test_plain_decode_reads_a_schedule_only_with_multi_wave_tiles(
    q_heads, is_draft, reads, schedule_calls
):
    """Width-1 decode reads a schedule only where its rows of one KV head
    fill a multi-wave tile, and never in a draft."""
    step = _Step(width=1, q_heads=q_heads, is_draft=is_draft)
    step.refresh(UNEVEN)
    for layer_id in range(3):
        step.check(step.forward(layer_id, 4), layer_id, UNEVEN)
    assert schedule_calls == ([4] if reads else [])


def test_captured_verify_computes_each_steps_schedule_in_the_graph(schedule_calls):
    """The capture records the schedule computation in the first verify
    layer, so every replay balances its own lengths, including padding
    requests at the verify floor, with no eager schedule launch."""
    step = _Step(width=8)
    seed = torch.full((MAX_BS,), 8, dtype=torch.int32, device="cuda")
    step.backend.init_forward_metadata_capture_cuda_graph(MAX_BS, seed, step.page_table)
    step.forward(0, MAX_BS)
    step.forward(1, MAX_BS)
    torch.cuda.synchronize()
    step.backend.init_forward_metadata_capture_cuda_graph(MAX_BS, seed, step.page_table)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        outputs = [step.forward(layer_id, MAX_BS) for layer_id in range(4)]
    captured = len(schedule_calls)
    for lengths in ([LONG, 8, 8, 300 * PAGE_SIZE], [8, 700 * PAGE_SIZE + 1, 8, 8]):
        step.refresh(lengths, for_graph_replay=True)
        graph.replay()
        torch.cuda.synchronize()
        for layer_id, output in enumerate(outputs):
            step.check(output, layer_id, lengths)
        assert torch.equal(
            step.backend.decode_schedule_buf[: MAX_BS + 1],
            step.fresh_schedule(lengths),
        )
    assert len(schedule_calls) == captured
