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

"""The MHA leaf's balanced decode variant on gfx950: its per-step split
schedule, through the real backend, layers and registered kernels."""

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
# Layer 3's KV heads differ from the leaf's, so it keeps its default kernel
# under the balanced variant (the step's schedule is for the leaf's shape).
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
    def __init__(self, *, width: int, context: int = CONTEXT, is_draft: bool = False):
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
            context_len=context,
            max_bs=MAX_BS,
            speculative_num_draft_tokens=width,
            is_draft=is_draft,
            draft_block_decode=False,
            components=(spec,),
        )
        self.width = width
        self.backend = MHAAttnBackend(config, spec, kernel_page_size=PAGE_SIZE)
        self.backend.init_cuda_graph_state(MAX_BS)
        # Two full-attention layers around a sliding-window one, and a
        # full-attention layer of another shape.
        layouts = ((-1, KV_HEADS), (WINDOW_LEFT, KV_HEADS), (-1, KV_HEADS))
        layouts += ((-1, WIDE_KV_HEADS),)
        self.layers = [
            PagedAttention(
                Q_HEADS,
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


LONG = 1000 * PAGE_SIZE + 5
UNEVEN = [LONG, 3 * PAGE_SIZE, 8, 120 * PAGE_SIZE + 61]


@pytest.fixture
def schedule_calls(monkeypatch):
    """Count the backend's schedule computations; the kernels stay real."""
    from tokenspeed.runtime.layers.attention.backends.paged import mha

    calls = []

    def counted(*args, **kwargs):
        calls.append(kwargs["cache_seqlens"].shape[0])
        return real(*args, **kwargs)

    real = mha.mha_decode_schedule
    monkeypatch.setattr(mha, "mha_decode_schedule", counted)
    return calls


def test_balanced_variant_needs_a_long_context_target():
    """The target offers the balanced variant when its context limit reaches
    the balancing floor; it picks it for a few long requests among short
    ones, and keeps the default for short or about equal ones."""
    from tokenspeed.runtime.layers.attention.backends.paged.mha import (
        DECODE_VARIANT_BALANCED,
    )

    backend = _Step(width=8).backend
    assert backend.decode_kernel_variants(4) == (DECODE_VARIANT_BALANCED,)
    # 255 requests of two row blocks leave the schedule's 512 items too
    # little to share.
    assert backend.decode_kernel_variants(254) == (DECODE_VARIANT_BALANCED,)
    assert backend.decode_kernel_variants(255) == ()
    assert backend.prefer_decode_kernel_variant([1000, 3, 1, 1]) == (
        DECODE_VARIANT_BALANCED
    )
    assert backend.prefer_decode_kernel_variant([1000] * 4) is None
    assert backend.prefer_decode_kernel_variant([100, 3, 1, 1]) is None
    assert _Step(width=8, context=8192).backend.decode_kernel_variants(4) == ()
    assert _Step(width=1, is_draft=True).backend.decode_kernel_variants(4) == ()


@pytest.mark.parametrize("width", [1, 8], ids=["decode", "verify"])
def test_balanced_refresh_schedules_once_for_every_layer(width, schedule_calls):
    """Each refresh under the balanced variant computes one schedule from the
    lengths it wrote, before any layer; the full-attention layers of the
    leaf's shape read it, and the sliding layer and the layer of another
    shape take their default kernels. The default variant computes none."""
    from tokenspeed.runtime.layers.attention.backends.paged.mha import (
        DECODE_VARIANT_BALANCED,
    )

    step = _Step(width=width)
    step.backend.use_decode_kernel_variant(DECODE_VARIANT_BALANCED)
    step.refresh(UNEVEN)
    assert schedule_calls == [4]
    for layer_id in range(4):
        step.check(step.forward(layer_id, 4), layer_id, UNEVEN)
    assert schedule_calls == [4]

    grown = [length + 16 * PAGE_SIZE for length in UNEVEN]
    step.refresh(grown)
    assert schedule_calls == [4, 4]
    step.check(step.forward(0, 4), 0, grown)

    step.backend.use_decode_kernel_variant(None)
    step.refresh(UNEVEN)
    assert schedule_calls == [4, 4]
    step.check(step.forward(0, 4), 0, UNEVEN)


def test_captured_balanced_variant_replays_each_steps_lengths(schedule_calls):
    """The capture seeding and every replay refresh compute the schedule
    eagerly into the buffer the captured layers read, so each replay
    balances its own lengths, including padding requests at one page."""
    from tokenspeed.runtime.layers.attention.backends.paged.mha import (
        DECODE_VARIANT_BALANCED,
    )

    step = _Step(width=8)
    step.backend.use_decode_kernel_variant(DECODE_VARIANT_BALANCED)
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
    assert len(schedule_calls) == captured + 2


@pytest.mark.parametrize("window", [128, 2048], ids=["gpt-oss", "muse"])
def test_router_balances_the_full_group_beside_a_sliding_one(window, schedule_calls):
    """A hybrid model's router: the full-history group's leaf offers and
    picks the variant from each request's blocks in its host table, while
    the sliding group's leaf, whose table spans only its window, neither
    votes nor computes a schedule. Padding requests count one page."""
    from tokenspeed.runtime.execution.forward_batch_info import ForwardMode
    from tokenspeed.runtime.layers.attention.backends.paged.cache_group_geometry import (
        CacheGroupGeometry,
    )
    from tokenspeed.runtime.layers.attention.backends.paged.mha import (
        DECODE_VARIANT_BALANCED,
    )
    from tokenspeed.runtime.layers.attention.backends.paged.router import (
        CacheGroupRouter,
    )

    full, swa = "full_attention", "sliding_attention"
    leaves = {full: _Step(width=8).backend, swa: _Step(width=8).backend}
    router = CacheGroupRouter(
        None, is_draft=False, spec_num_tokens=8, device="cuda", consumed_group_ids=None
    )
    router.bind(
        CacheGroupGeometry(
            granularities={full: PAGE_SIZE, swa: PAGE_SIZE},
            attn_kinds={full: "full_history", swa: "sliding_window"},
            full_history_group_id=full,
            row_geometry={full: (PAGE_SIZE, 1), swa: (PAGE_SIZE, 1)},
            windows={full: None, swa: window},
        ),
        leaves,
    )
    router.init_cuda_graph_state(MAX_BS)
    assert router.decode_kernel_variants(MAX_BS) == (DECODE_VARIANT_BALANCED,)
    swa_columns = -(-window // PAGE_SIZE) + 2
    tables = {
        full: torch.full((3, 1000), -1, dtype=torch.int32),
        swa: torch.full((3, swa_columns), 7, dtype=torch.int32),
    }
    tables[full][0] = 7
    tables[full][1:, :3] = 7
    variant = router.select_decode_kernel_variant(MAX_BS, 3, tables)
    assert variant == DECODE_VARIANT_BALANCED
    # A batch whose table is narrower than the policy's floor is not
    # counted at all.
    narrow = {full: tables[full][:, :100], swa: tables[swa]}
    assert router.select_decode_kernel_variant(MAX_BS, 3, narrow) is None
    # Four about equal long requests keep the default.
    long_tables = {
        full: torch.full((4, 1000), 7, dtype=torch.int32),
        swa: torch.full((4, swa_columns), 7, dtype=torch.int32),
    }
    assert router.select_decode_kernel_variant(MAX_BS, 4, long_tables) is None

    router.refresh_decode_metadata(
        MAX_BS,
        3,
        torch.arange(MAX_BS, device="cuda"),
        torch.tensor([LONG, 3 * PAGE_SIZE, 8, 8], dtype=torch.int32, device="cuda"),
        forward_mode=ForwardMode.DECODE,
        block_tables={gid: table.cuda() for gid, table in tables.items()},
        decode_kernel_variant=variant,
    )
    # One schedule, the full leaf's; the sliding leaf runs its default.
    assert schedule_calls == [MAX_BS]


def test_router_counts_each_requests_pages_from_the_host_table():
    """The router hands the leaf each live request's blocks in its group's
    host table (expanded to kernel pages) and one page per padding request,
    and runs the variant only when every leaf offering it picks it."""
    from tokenspeed.runtime.layers.attention.backends.paged.cache_group_geometry import (
        CacheGroupGeometry,
    )
    from tokenspeed.runtime.layers.attention.backends.paged.mha import (
        DECODE_VARIANT_BALANCED,
    )
    from tokenspeed.runtime.layers.attention.backends.paged.router import (
        CacheGroupRouter,
    )

    full = "full_attention"
    step = _Step(width=8)
    router = CacheGroupRouter(
        None, is_draft=False, spec_num_tokens=8, device="cuda", consumed_group_ids=None
    )
    router.bind(
        CacheGroupGeometry(
            granularities={full: PAGE_SIZE},
            attn_kinds={full: "full_history"},
            full_history_group_id=full,
            row_geometry={full: (PAGE_SIZE, 1)},
            windows={full: None},
        ),
        {full: step.backend},
    )
    assert router.decode_kernel_variants(MAX_BS) == (DECODE_VARIANT_BALANCED,)
    table = torch.full((3, 1000), -1, dtype=torch.int32)
    table[0] = 7
    table[1, :3] = 7
    table[2, :1] = 7
    assert router.select_decode_kernel_variant(4, 3, {full: table}) == (
        DECODE_VARIANT_BALANCED
    )
    # Four requests of about equal lengths keep the default.
    table = torch.full((4, 1000), 7, dtype=torch.int32)
    assert router.select_decode_kernel_variant(4, 4, {full: table}) is None
    assert router.select_decode_kernel_variant(4, 0, {full: table[:0]}) is None
