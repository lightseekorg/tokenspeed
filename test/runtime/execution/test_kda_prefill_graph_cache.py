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

"""Regression coverage for the capacity-based prefill graph owner."""

from types import SimpleNamespace

import pytest
import torch

from tokenspeed.runtime.layers.attention.backends.state.kda_prefill_graph import (
    KdaOuterGraphBinding,
    KdaPrefillGraphCache,
    _checkpoint_slot_batch,
    _clone_metadata,
)
from tokenspeed.runtime.layers.attention.backends.state.mamba import (
    MambaForwardMetadata,
)


def metadata(device, page):
    return MambaForwardMetadata(
        query_start_loc=torch.tensor([0, 1], dtype=torch.int32, device=device),
        scan_query_start_loc=torch.tensor([0, 1], device=device),
        query_start_loc_int64=torch.tensor([0, 1], device=device),
        extend_seq_lens_cpu=torch.tensor([1]),
        cu_extend_seq_lens_cpu=torch.tensor([0, 1]),
        state_in_blocks_by_group={"state": torch.tensor([page], device=device)},
        state_out_blocks_by_group={"state": torch.tensor([page], device=device)},
    )


def test_metadata_snapshot_does_not_alias():
    source = metadata("cpu", 1)
    cloned = _clone_metadata(source)
    source.state_in_blocks_by_group["state"].zero_()
    assert cloned.state_in_blocks_by_group["state"].item() == 1
    assert (
        cloned.extend_seq_lens_cpu.data_ptr() != source.extend_seq_lens_cpu.data_ptr()
    )


@pytest.mark.parametrize("count", [0, 1, 2])
def test_checkpoint_slots_keep_shape_and_mask_inactive_requests(count):
    from tokenspeed.runtime.layers.attention.backends.state.mamba import (
        _build_prefill_checkpoint_batch,
    )

    source = metadata("cpu", 1)
    lengths = torch.tensor([868, 869])
    prefixes = torch.tensor([50304, 50304])
    source.extend_seq_lens_cpu = lengths
    source.cu_extend_seq_lens_cpu = torch.cat(
        (torch.zeros(1, dtype=torch.int64), lengths.cumsum(0))
    )
    source.prefill_checkpoint_batch = _build_prefill_checkpoint_batch(
        lengths, prefixes, count, 128, "cpu"
    )
    fixed = _checkpoint_slot_batch(source, 2048, 254)
    assert fixed.rows.tolist() == [0, 1]
    assert fixed.tail_seq_lens_cpu.tolist() == [
        [100, 101][row] if row < count else 1 for row in range(2)
    ]
    assert fixed.state_update_rows.tolist() == [
        row if row < count else -1 for row in range(2)
    ]
    assert fixed.body_token_indices.shape == (2048,)
    assert fixed.tail_token_indices.shape == (254,)
    body = fixed.body_token_indices[fixed.body_token_indices >= 0]
    tail = fixed.tail_token_indices[fixed.tail_token_indices >= 0]
    torch.testing.assert_close(
        torch.cat((body, tail)).sort().values, torch.arange(1737)
    )
    assert not fixed.use_token_views
    sources = fixed.output_sources
    packed_destinations = torch.cat(
        (fixed.body_token_indices, fixed.tail_token_indices)
    )
    torch.testing.assert_close(packed_destinations[sources[:1737]], torch.arange(1737))
    assert torch.all(sources[1737:] == -1)
    assert (
        source.prefill_checkpoint_batch is None
        if count == 0
        else source.prefill_checkpoint_batch.rows.numel() == count
    )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_replay_refreshes_pages_and_writes_exactly_once():
    cache = KdaPrefillGraphCache()
    backend = SimpleNamespace(forward_metadata=metadata("cuda", 1))
    value = torch.ones(1, device="cuda")
    state = torch.zeros(4, device="cuda")

    def forward():
        pages = backend.forward_metadata.state_out_blocks_by_group["state"]
        state[pages] += value
        return state[pages].clone()

    for page in (1, 2, 3, 1, 2):
        live = metadata("cuda", page)
        backend.forward_metadata = live
        before = state.clone()
        result = cache.run(backend, 0, 1, {"value": value}, forward)
        torch.testing.assert_close(result, before[page : page + 1] + 1)
        before[page] += 1
        torch.testing.assert_close(state, before)
        assert backend.forward_metadata is live
    assert cache.captures == 1
    assert cache.replays == 4

    # Another input allocation must not replay the captured old address.
    value = torch.full_like(value, 3)
    before = state.clone()
    cache.run(backend, 0, 1, {"value": value}, forward)
    before[2] += 3
    torch.testing.assert_close(state, before)
    assert cache.replays == 4

    # Every outer token bucket remains eligible, including the ninth and later.
    for bucket in range(2, 11):
        for _ in range(3):
            backend.forward_metadata = metadata("cuda", 2)
            before = state.clone()
            result = cache.run(backend, 0, bucket, {"value": value}, forward)
            before[2] += 3
            torch.testing.assert_close(state, before)
            torch.testing.assert_close(result, before[2:3])
    assert len(cache.schedules) == 10
    assert cache.captures == 10
    assert cache.replays == 22


def test_hybrid_reinitialization_clears_kda_graphs():
    from unittest.mock import Mock

    from tokenspeed.runtime.layers.attention.backends.hybrid.linear import (
        HybridLinearAttnBackend,
    )
    from tokenspeed.runtime.layers.attention.backends.state.kda import KdaAttnBackend

    child = object.__new__(KdaAttnBackend)
    child._prefill_graph_cache = object()
    backend = object.__new__(HybridLinearAttnBackend)
    backend.full_attn_backend = Mock()
    backend.linear_attn_backend = child
    backend.init_prefill_graph_state(1024, 4)
    assert child._prefill_graph_cache is None
    backend.full_attn_backend.init_prefill_graph_state.assert_called_once_with(1024, 4)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_failure_restores_live_metadata():
    cache = KdaPrefillGraphCache()
    live = metadata("cuda", 1)
    backend = SimpleNamespace(forward_metadata=live)

    def forward():
        raise ValueError("test failure")

    with pytest.raises(ValueError, match="test failure"):
        cache.run(backend, 0, 1, {}, forward)
    assert backend.forward_metadata is live


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_pool_sharing_is_scoped_to_replay_stream():
    cache = KdaPrefillGraphCache()
    streams = [torch.cuda.Stream(), torch.cuda.Stream()]
    cases = []

    def bind_forward(value):
        def forward():
            return (value.square() + 3).clone()

        return forward

    for stream in streams:
        with torch.cuda.stream(stream):
            for layer in range(2):
                backend = SimpleNamespace(forward_metadata=metadata("cuda", 1))
                value = torch.full((257,), layer + 1.0, device="cuda")

                forward = bind_forward(value)
                for _ in range(2):
                    cache.run(backend, layer, 257, {"value": value}, forward)
                cases.append((stream, backend, layer, value, forward))
    assert len(cache._capture_resources) == 2
    pools = [resources[0] for resources in cache._capture_resources.values()]
    assert pools[0] != pools[1]
    outputs = []
    # Queue both streams without a host synchronization between replays.
    for stream, backend, layer, value, forward in reversed(cases):
        with torch.cuda.stream(stream):
            value.add_(2)
            result = cache.run(backend, layer, 257, {"value": value}, forward)
            outputs.append((result.clone(), value.square() + 3))
    torch.cuda.synchronize()
    for actual, expected in outputs:
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    assert cache.captures == 4


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_outer_graph_inlines_state_layers_and_retains_full_attention_break():
    from tokenspeed.runtime.execution.breakable_cuda_graph import BreakableCapture
    from tokenspeed.runtime.execution.forward_batch_info import ForwardMode
    from tokenspeed.runtime.layers.attention.backends.base import AttentionBackend
    from tokenspeed.runtime.layers.attention.backends.hybrid.linear import (
        HybridLinearAttnBackend,
    )

    state = torch.zeros(4, device="cuda")
    value = torch.ones(8, 4, device="cuda")

    class Leaf(AttentionBackend):
        def forward_extend(self, q, k, v, layer, pool, bs, **kwargs):
            if layer.layer_id == 1:
                return q + 2
            pages = self.forward_metadata.state_out_blocks_by_group["state"]
            state[pages] += 1
            valid = self.forward_metadata.query_start_loc[-1]
            return (q + state[pages]).masked_fill(
                (torch.arange(q.shape[0], device=q.device) >= valid)[:, None], 0
            )

    leaf = object.__new__(Leaf)
    leaf.cache_pool = object()
    leaf._prefix_granularity = 128
    leaf.forward_metadata = metadata("cuda", 1)
    full = object.__new__(Leaf)
    full.device = torch.device("cuda")
    hybrid = HybridLinearAttnBackend(full, leaf, [1])
    binding = KdaOuterGraphBinding(leaf, 8, leaf.forward_metadata)

    def forward():
        out = value * 2
        for layer_id in (0, 1, 2):
            out = hybrid.forward(
                out,
                None,
                None,
                SimpleNamespace(layer_id=layer_id),
                None,
                ForwardMode.EXTEND,
                1,
                True,
                None,
            )
        return out * 3

    forward()
    torch.cuda.synchronize()
    ordinary = BreakableCapture()
    with ordinary:
        forward()
    assert ordinary.num_segments == 7  # Three eager breaks + four graphs.
    with binding.bind(refresh=False):
        forward()
        torch.cuda.synchronize()
        merged = BreakableCapture()
        with merged:
            output = forward()
    assert merged.num_segments == 3  # Only full attention remains an eager break.

    ctx = SimpleNamespace(forward_mode=ForwardMode.EXTEND, bs=1, num_extends=1)
    for length, page in ((1, 2), (7, 3), (8, 1), (3, 2)):
        live = metadata("cuda", page)
        live.query_start_loc[-1] = length
        live.query_start_loc_int64[-1] = length
        live.scan_query_start_loc[-1] = length
        live.extend_seq_lens_cpu[0] = length
        live.cu_extend_seq_lens_cpu[-1] = length
        leaf.forward_metadata = live
        assert binding.compatible(ctx)
        before = state.clone()
        with binding.bind(refresh=True):
            merged.replay(valid_rows=length)
        expected = torch.zeros_like(output)
        expected[:length] = (value[:length] * 2 + 2 * before[page] + 5) * 3
        torch.testing.assert_close(output, expected, rtol=0, atol=0)
        before[page] += 2
        torch.testing.assert_close(state, before, rtol=0, atol=0)
        assert leaf.forward_metadata is live
        assert not leaf.prefill_graph_inline

    ctx.bs = ctx.num_extends = 2
    assert not binding.compatible(ctx)
    ctx.bs = ctx.num_extends = 1
    ctx.forward_mode = ForwardMode.MIXED
    assert not binding.compatible(ctx)
    ctx.forward_mode = ForwardMode.EXTEND
    leaf.forward_metadata.prefill_checkpoint_batch = SimpleNamespace(
        rows=torch.zeros(1)
    )
    assert binding.compatible(ctx)
    leaf.forward_metadata.prefill_checkpoint_batch = None
    leaf.cache_pool = object()
    assert not binding.compatible(ctx)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_outer_binding_restores_metadata_after_failure():
    from tokenspeed.runtime.layers.attention.backends.base import AttentionBackend

    backend = object.__new__(AttentionBackend)
    backend.cache_pool = object()
    backend._prefix_granularity = 128
    original = metadata("cuda", 1)
    backend.forward_metadata = original
    binding = KdaOuterGraphBinding(backend, 8, original)
    with pytest.raises(ValueError, match="test failure"):
        with binding.bind(refresh=True):
            assert backend.prefill_graph_inline
            raise ValueError("test failure")
    assert backend.forward_metadata is original
    assert not backend.prefill_graph_inline


def test_internal_checkpoint_batch_bypasses_capacity_graph():
    live = metadata("cpu", 1)
    live.prefill_checkpoint_batch = object()
    backend = SimpleNamespace(forward_metadata=live)
    cache = KdaPrefillGraphCache()
    calls = []

    def forward():
        calls.append(backend.forward_metadata)
        return "eager checkpoint result"

    assert cache.run(backend, 0, 8, {}, forward) == "eager checkpoint result"
    assert calls == [live]
    assert cache.schedules == {}


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
@pytest.mark.parametrize("batch_size", [1, 2, 4])
def test_checkpoint_outer_graph_replays_lengths_pages_and_states(batch_size):
    from dataclasses import fields, replace

    from tokenspeed_kernel.ops.attention.gdn.triton import (
        CAUSAL_CONV1D_BLOCK_M,
        build_causal_conv1d_prefill_metadata,
    )

    from tokenspeed.runtime.execution.forward_batch_info import (
        CaptureHiddenMode,
        ForwardMode,
    )
    from tokenspeed.runtime.execution.prefill_graph import PrefillGraph
    from tokenspeed.runtime.layers.attention.backends.state.kda import KdaAttnBackend
    from tokenspeed.runtime.layers.attention.backends.state.mamba import (
        _build_prefill_checkpoint_batch,
    )

    pytest.importorskip("tokenspeed_cutedsl_kda")
    torch.manual_seed(42)
    # Eight BF16 beta heads keep eager tail views 16-byte aligned even when
    # the body has one token, as required by the native scan ABI.
    bucket, heads, dim = 2048, 8, 128
    channels = heads * dim
    conv = torch.randn(12, 3 * channels, 3, device="cuda", dtype=torch.bfloat16)
    states = torch.randn(12, heads, dim, dim, device="cuda", dtype=torch.float32)
    initial_conv, initial_states = conv.clone(), states.clone()
    raw = torch.randn(bucket, 3 * channels, device="cuda", dtype=torch.bfloat16)
    backend = object.__new__(KdaAttnBackend)
    backend.__dict__.update(
        is_draft=False,
        cache_pool=object(),
        _prefix_granularity=128,
        _prefill_graph_enabled=False,
        kda_backend="cutedsl_kda",
        kda_recurrent_layout="v_major",
    )
    backend._layer_state = lambda layer_id: (
        backend.forward_metadata.state_in_blocks_by_group["state"],
        backend.forward_metadata.state_out_blocks_by_group["state"],
        conv,
        states,
    )
    backend._layer_prefill_checkpoint_blocks = lambda layer_id: (
        backend.forward_metadata.state_checkpoint_blocks_by_group["state"]
        if backend.forward_metadata.state_checkpoint_blocks_by_group is not None
        else None
    )
    kwargs = dict(
        conv_weights=torch.randn(3 * channels, 4, device="cuda", dtype=torch.bfloat16)
        * 0.1,
        bias=None,
        activation="silu",
        key_dim=channels,
        value_dim=channels,
        attention_tp_size=1,
        head_k_dim=dim,
        head_v_dim=dim,
        g_raw=torch.randn(bucket, channels, device="cuda", dtype=torch.bfloat16),
        beta_raw=torch.randn(bucket, heads, device="cuda", dtype=torch.bfloat16),
        A_log=torch.zeros(heads, device="cuda"),
        dt_bias=torch.zeros(channels, device="cuda"),
        lower_bound=-5.0,
        layer_id=0,
        seq_len=bucket,
    )

    def forward():
        return backend.forward_extend(
            None,
            None,
            None,
            None,
            None,
            backend.forward_metadata.extend_seq_lens_cpu.numel(),
            ForwardMode.EXTEND,
            save_kv_cache=True,
            mixed_qkv=raw.clone(),
            **kwargs,
        )

    def reset():
        conv.copy_(initial_conv)
        states.copy_(initial_states)

    def live_metadata(lengths, prefixes, page_shift):
        lengths = torch.tensor(lengths)
        host = torch.cat((torch.zeros(1, dtype=torch.int64), lengths.cumsum(0)))
        bounds = host.to(device="cuda", dtype=torch.int32)
        checkpoint = _build_prefill_checkpoint_batch(
            lengths, torch.tensor(prefixes), lengths.numel(), 128, "cuda"
        )
        if checkpoint is not None:
            checkpoint = replace(
                checkpoint,
                body_query_start_loc=checkpoint.body_query_start_loc.long(),
                tail_query_start_loc=checkpoint.tail_query_start_loc.long(),
            )
        pages = (
            torch.arange(lengths.numel(), device="cuda", dtype=torch.int32) + page_shift
        )
        return MambaForwardMetadata(
            query_start_loc=bounds,
            scan_query_start_loc=bounds.long(),
            query_start_loc_int64=bounds.long(),
            extend_seq_lens_cpu=lengths,
            cu_extend_seq_lens_cpu=host,
            state_in_blocks_by_group={
                "state": torch.tensor(
                    [1 if prefix else 0 for prefix in prefixes],
                    device="cuda",
                    dtype=torch.int32,
                )
            },
            state_out_blocks_by_group={"state": pages},
            state_checkpoint_blocks_by_group=(
                {"state": pages + 4} if checkpoint is not None else None
            ),
            prefill_checkpoint_batch=checkpoint,
            conv_prefill_metadata=build_causal_conv1d_prefill_metadata(
                bounds, lengths, CAUSAL_CONV1D_BLOCK_M
            ),
        )

    cases = []
    # Reuse ONE capture across changing checkpoint counts, rows, and lengths.
    for checkpoint_count in range(batch_size + 1):
        for tail_lengths, prefix in [([837, 325, 197, 197], 0), ([70] * 4, 127)]:
            lengths = tail_lengths[:checkpoint_count] + [128] * (
                batch_size - checkpoint_count
            )
            prefixes = [prefix] * checkpoint_count + [0] * (
                batch_size - checkpoint_count
            )
            cases.extend([(lengths, prefixes), (lengths[::-1], prefixes[::-1])])
        cases.append(
            (
                [128] * batch_size,
                [127] * checkpoint_count + [0] * (batch_size - checkpoint_count),
            )
        )
    if batch_size == 1:
        cases.extend([([837], [50304]), ([769], [0]), ([1023], [128])])
    if batch_size == 2:
        cases.extend([([868, 869], [50304, 50304]), ([869, 868], [50304, 50304])])
    cases.append(([1] * batch_size, [0] * batch_size))
    # Exercise the real outer startup loop and shared-pool ownership.
    owner = object.__new__(PrefillGraph)
    owner.config = SimpleNamespace(
        global_rank=1,
        context_len=bucket,
        max_num_seqs=4,
        data_parallel_size=1,
        prefill_graph_capture_batch_sizes=[batch_size],
    )
    owner.dp_size, owner.num_warmup, owner._pool = 1, 1, None
    owner.capture_buckets = [bucket]
    owner._captures, owner._outputs, owner._inline_captures = {}, {}, {}
    owner.input_buffers = SimpleNamespace(input_ids_buf=torch.ones(bucket))
    owner._embed_tokens = lambda ids: ids
    owner._land_input_embeds = lambda *args: None
    owner.attn_backend = backend
    owner._run_inner = lambda bucket: (forward(), None)

    def dummy(bucket, bs):
        backend.forward_metadata = live_metadata([bucket // bs] * bs, [0] * bs, 2)
        return SimpleNamespace(bs=bs, capture_hidden_mode=CaptureHiddenMode.NULL)

    owner.make_dummy_batch = dummy
    backend._prefill_graph_enabled = True
    owner._capture_all_buckets(None)
    backend._prefill_graph_enabled = False
    assert set(owner._inline_captures) == {(bucket, batch_size)}
    capture, captured, (binding,) = owner._inline_captures[bucket, batch_size]
    output = captured.hidden_states
    assert capture.num_segments == 1
    fixed = binding.metadata.prefill_checkpoint_batch
    addresses = {
        field.name: getattr(fixed, field.name).data_ptr()
        for field in fields(fixed)
        if isinstance(getattr(fixed, field.name), torch.Tensor)
    }
    ctx = SimpleNamespace(
        forward_mode=ForwardMode.EXTEND, bs=batch_size, num_extends=batch_size
    )
    for index, (lengths, prefixes) in enumerate(cases * 2):
        backend.forward_metadata = live_metadata(lengths, prefixes, 2 + index % 2)
        assert binding.compatible(ctx)
        reset()
        expected = forward()[: sum(lengths)].clone()
        expected_conv, expected_states = conv.clone(), states.clone()
        reset()
        with binding.bind(refresh=True):
            capture.replay(valid_rows=sum(lengths))
        assert addresses == {
            name: getattr(fixed, name).data_ptr() for name in addresses
        }
        torch.testing.assert_close(output[: sum(lengths)], expected, rtol=0, atol=0)
        torch.testing.assert_close(conv, expected_conv, rtol=0, atol=0)
        torch.testing.assert_close(states, expected_states, rtol=0, atol=0)
        assert torch.count_nonzero(output[sum(lengths) :]) == 0


@pytest.mark.parametrize(
    "compatible,transfer,expected",
    [(True, False, 2), (False, False, 1), (True, True, 1)],
)
@pytest.mark.parametrize("batch_size", [1, 2])
def test_outer_owner_selects_matching_graph_and_refreshes_before_replay(
    compatible, transfer, expected, batch_size
):
    from contextlib import contextmanager, nullcontext
    from unittest.mock import patch

    from tokenspeed.runtime.execution.prefill_graph import CapturedForward, PrefillGraph

    events = []

    class Binding:
        def compatible(self, ctx):
            return compatible

        @contextmanager
        def bind(self, refresh):
            assert refresh
            events.append("refresh")
            try:
                yield
            finally:
                events.append("restore")

    def capture(label):
        return SimpleNamespace(replay=lambda **kwargs: events.append(label))

    owner = object.__new__(PrefillGraph)
    owner._captures = {8: capture("ordinary")}
    owner._outputs = {8: CapturedForward(torch.ones(8, 4), None)}
    owner._inline_captures = {
        (8, batch_size): (
            capture("inline"),
            CapturedForward(torch.full((8, 4), 2.0), None),
            [Binding()],
        )
    }
    owner.attn_backend = SimpleNamespace(step_counter=object() if transfer else None)
    owner._replay_bucket = lambda ctx: 8
    owner._log_engaged_once = lambda *args: None
    owner._embed_tokens = lambda ids: torch.zeros(8, 4)
    owner._land_input_embeds = lambda *args: None
    owner._padded_to = lambda *args: nullcontext()
    owner.text_model = SimpleNamespace(
        lm_head=None, logits_processor=lambda ids, hidden, *args: hidden
    )
    ctx = SimpleNamespace(input_num_tokens=8, bs=batch_size)
    with patch(
        "tokenspeed.runtime.execution.prefill_graph.LogitsMetadata.from_forward_context",
        return_value=None,
    ):
        result = owner.replay(ctx, torch.zeros(8, dtype=torch.int64), None)
    torch.testing.assert_close(result, torch.full((8, 4), float(expected)))
    assert events == (
        ["refresh", "inline", "restore"] if expected == 2 else ["ordinary"]
    )


@pytest.mark.parametrize(
    "sizes,bucket,expected",
    [
        (None, 8, [1]),
        (None, 17, [2]),
        ([1, 2, 4], 8, [1, 2, 4]),
        ([1, 2, 4], 2, [1, 2]),
        ([1, 2, 4], 17, [2, 4]),
        ([4, 2, 2], 8, [2, 4]),
    ],
)
def test_inline_capture_request_counts(sizes, bucket, expected):
    from tokenspeed.runtime.execution.prefill_graph import (
        resolve_prefill_capture_batch_sizes,
    )

    config = SimpleNamespace(
        context_len=16,
        max_num_seqs=8,
        data_parallel_size=2,
        prefill_graph_capture_batch_sizes=sizes,
    )
    assert resolve_prefill_capture_batch_sizes(config, token_bucket=bucket) == expected
    for invalid in [[0], [-1], [5]]:
        config.prefill_graph_capture_batch_sizes = invalid
        with pytest.raises(ValueError, match="capture batch sizes"):
            resolve_prefill_capture_batch_sizes(config, token_bucket=bucket)


def test_outer_capture_records_one_variant_per_configured_request_count():
    from contextlib import contextmanager

    from tokenspeed.runtime.execution.forward_batch_info import CaptureHiddenMode
    from tokenspeed.runtime.execution.prefill_graph import CapturedForward, PrefillGraph

    owner = object.__new__(PrefillGraph)
    owner.config = SimpleNamespace(
        global_rank=1,
        context_len=16,
        max_num_seqs=4,
        data_parallel_size=1,
        prefill_graph_capture_batch_sizes=[1, 2],
    )
    owner.dp_size = 1
    owner.capture_buckets = [8]
    owner._captures, owner._outputs, owner._inline_captures = {}, {}, {}
    owner.input_buffers = SimpleNamespace(input_ids_buf=torch.ones(8))
    owner._embed_tokens = lambda ids: ids
    owner._land_input_embeds = lambda *args: None
    active_count = None

    class Binding:
        @contextmanager
        def bind(self, refresh):
            nonlocal active_count
            assert not refresh
            active_count = self.count
            try:
                yield
            finally:
                active_count = None

    def prepare(bucket):
        assert bucket == 8
        binding = Binding()
        binding.count = owner._ctx.bs
        return [binding]

    owner.attn_backend = SimpleNamespace(prepare_prefill_graph_bindings=prepare)
    owner.make_dummy_batch = lambda bucket, bs: SimpleNamespace(
        bs=bs, capture_hidden_mode=CaptureHiddenMode.NULL
    )

    def capture(bucket, wrapper):
        label = (owner._ctx.bs, active_count)
        owner._captures[bucket] = label
        owner._outputs[bucket] = CapturedForward(torch.ones(bucket, 1), None)

    owner._capture_bucket = capture
    owner._capture_all_buckets(None)
    assert owner._captures[8] == (1, None)
    assert set(owner._inline_captures) == {(8, 1), (8, 2)}
    for (_, bs), (capture, _, _) in owner._inline_captures.items():
        assert capture == (bs, bs)
    assert owner._ctx is None
