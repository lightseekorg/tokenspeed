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

import os
import sys
from types import SimpleNamespace

import pytest
import torch

sys.path.insert(
    0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
)
from ci_system.ci_register import register_cuda_ci  # noqa: E402

register_cuda_ci(est_time=60, suite="runtime-1gpu")

from tokenspeed.runtime.execution.memory_delta import (  # noqa: E402
    NULL_MEMORY_DELTA_OBSERVER,
)
from tokenspeed.runtime.layers.attention.backends.state.kda_prefill_metadata import (  # noqa: E402
    _checkpoint_slot_batch,
    _clone_metadata,
    prepare_kda_prefill_metadata,
)
from tokenspeed.runtime.layers.attention.backends.state.mamba import (  # noqa: E402
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


def configure_prefill(backend):
    backend._prefill_graph_enabled = True
    backend.kda_backend = "cutedsl_kda"
    backend._prefill_metadata = {}
    backend._prefill_metadata_pool = None


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
    fixed = _checkpoint_slot_batch(source, 2048, 254, 2)
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


@pytest.mark.parametrize("count", [0, 1, 3])
def test_checkpoint_slots_pad_requests_without_source_tokens_or_state_updates(count):
    from tokenspeed.runtime.layers.attention.backends.state.mamba import (
        _build_prefill_checkpoint_batch,
    )

    source = metadata("cpu", 1)
    lengths = torch.tensor([129, 130, 131])
    source.extend_seq_lens_cpu = lengths
    source.cu_extend_seq_lens_cpu = torch.cat((lengths.new_zeros(1), lengths.cumsum(0)))
    source.prefill_checkpoint_batch = _build_prefill_checkpoint_batch(
        lengths, lengths.new_zeros(3), count, 128, "cpu"
    )
    fixed = _checkpoint_slot_batch(source, 512, 508, 4)
    assert fixed.body_seq_lens_cpu[-1] == fixed.tail_seq_lens_cpu[-1] == 1
    assert fixed.state_update_rows.tolist() == [
        row if row < count else -1 for row in range(4)
    ]
    for mapping, bounds in [
        (fixed.body_token_indices, fixed.body_cu_seqlens_cpu),
        (fixed.tail_token_indices, fixed.tail_cu_seqlens_cpu),
    ]:
        assert mapping[bounds[-2] : bounds[-1]].tolist() == [-1]
    packed = torch.cat((fixed.body_token_indices, fixed.tail_token_indices))
    assert packed[fixed.output_sources[:390]].tolist() == list(range(390))
    assert torch.all(fixed.output_sources[390:] == -1)
    assert source.extend_seq_lens_cpu.tolist() == [129, 130, 131]


def test_hybrid_initializes_prefill_graph_state_on_both_children():
    from unittest.mock import Mock

    from tokenspeed.runtime.layers.attention.backends.hybrid.linear import (
        HybridLinearAttnBackend,
    )

    backend = object.__new__(HybridLinearAttnBackend)
    backend.full_attn_backend = Mock()
    backend.linear_attn_backend = Mock()
    backend.init_prefill_graph_state(1024, 4)
    backend.full_attn_backend.init_prefill_graph_state.assert_called_once_with(1024, 4)
    backend.linear_attn_backend.init_prefill_graph_state.assert_called_once_with(
        1024, 4
    )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_outer_graph_inlines_state_layers_and_retains_full_attention_break():
    from tokenspeed.runtime.execution.breakable_cuda_graph import BreakableCapture
    from tokenspeed.runtime.execution.forward_batch_info import ForwardMode
    from tokenspeed.runtime.layers.attention.backends.hybrid.linear import (
        HybridLinearAttnBackend,
    )
    from tokenspeed.runtime.layers.attention.backends.state.kda import KdaAttnBackend

    state = torch.zeros(4, device="cuda")
    value = torch.ones(8, 4, device="cuda")

    class Leaf(KdaAttnBackend):
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
    configure_prefill(leaf)
    leaf.cache_pool = object()
    leaf._prefix_granularity = 128
    leaf.forward_metadata = metadata("cuda", 1)
    full = object.__new__(Leaf)
    full.forward_metadata = None
    full.device = torch.device("cuda")
    hybrid = HybridLinearAttnBackend(full, leaf, [1])

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
    leaf.prepare_prefill_metadata(8, 1, ForwardMode.EXTEND, capture=True)
    retained = leaf.forward_metadata
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
        before = state.clone()
        assert leaf.prepare_prefill_metadata(8, 1, ctx.forward_mode, capture=False)
        merged.replay(valid_rows=length)
        expected = torch.zeros_like(output)
        expected[:length] = (value[:length] * 2 + 2 * before[page] + 5) * 3
        torch.testing.assert_close(output, expected, rtol=0, atol=0)
        before[page] += 2
        torch.testing.assert_close(state, before, rtol=0, atol=0)
        assert leaf.forward_metadata is retained
        assert leaf.prefill_metadata_is_capture_ready

    with pytest.raises(ValueError, match="existing captured capacity"):
        leaf.prepare_prefill_metadata(8, 2, ForwardMode.EXTEND, capture=False)
    assert not leaf.prepare_prefill_metadata(8, 1, ForwardMode.MIXED, capture=False)
    leaf.cache_pool = object()
    with pytest.raises(RuntimeError, match="pool changed"):
        leaf.prepare_prefill_metadata(8, 1, ForwardMode.EXTEND, capture=False)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_eager_and_capture_use_the_same_metadata_contract_without_retaining_eager_shapes():
    from tokenspeed.runtime.execution.forward_batch_info import ForwardMode
    from tokenspeed.runtime.layers.attention.backends.state.kda import KdaAttnBackend
    from tokenspeed.runtime.layers.attention.backends.state.kda_prefill_metadata import (
        KdaPrefillMetadata,
    )

    backend = object.__new__(KdaAttnBackend)
    configure_prefill(backend)
    backend.cache_pool = object()
    backend._prefix_granularity = 128
    original = metadata("cuda", 1)
    backend.forward_metadata = original
    backend.prepare_prefill_metadata(8, 1, ForwardMode.EXTEND, capture=False)
    eager = backend.forward_metadata
    assert isinstance(eager, KdaPrefillMetadata)
    assert not backend.prefill_metadata_is_capture_ready
    assert not backend._prefill_metadata
    backend.forward_metadata = original
    backend.prepare_prefill_metadata(8, 1, ForwardMode.EXTEND, capture=True)
    retained = backend.forward_metadata
    assert isinstance(retained, type(eager))
    assert retained.capacity == eager.capacity
    for name in ("body_token_indices", "tail_token_indices", "output_token_sources"):
        torch.testing.assert_close(
            getattr(retained.prefill_checkpoint_batch, name),
            getattr(eager.prefill_checkpoint_batch, name),
        )
    for bucket in (8, 16, 32, 8):
        backend.forward_metadata = metadata("cuda", 2)
        backend.prepare_prefill_metadata(bucket, 1, ForwardMode.EXTEND, capture=False)
        assert (backend.forward_metadata is retained) == (bucket == 8)
        assert set(backend._prefill_metadata) == {(8, 1)}
    backend.init_prefill_graph_state(8, 1)
    assert not backend._prefill_metadata
    assert not backend.prefill_metadata_is_capture_ready


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
@pytest.mark.parametrize("batch_size", [1, 2, 4])
def test_checkpoint_outer_graph_replays_lengths_pages_and_states(batch_size):
    from dataclasses import fields, replace

    from tokenspeed_kernel.ops.attention.gdn.triton import (
        CAUSAL_CONV1D_BLOCK_M,
        build_causal_conv1d_prefill_metadata,
    )
    from tokenspeed_kernel.ops.attention.kda.cute_dsl import cutedsl_kda_supported

    from tokenspeed.runtime.execution.forward_batch_info import (
        CaptureHiddenMode,
        ForwardMode,
    )
    from tokenspeed.runtime.execution.prefill_graph import PrefillGraph
    from tokenspeed.runtime.layers.attention.backends.state.kda import KdaAttnBackend
    from tokenspeed.runtime.layers.attention.backends.state.mamba import (
        _build_prefill_checkpoint_batch,
    )

    if not cutedsl_kda_supported():
        pytest.skip("native CuteDSL KDA requires NVIDIA SM100 or SM103")
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
    configure_prefill(backend)
    backend.__dict__.update(
        is_draft=False,
        cache_pool=object(),
        _prefix_granularity=128,
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
    if batch_size == 4:
        # Reuse BS4 storage across full -> padded -> full transitions. Compare
        # every state block, including the null page and the previous fourth row.
        for actual_bs in (3, 1, 2, 3):
            cases.extend(
                [
                    ([64 + i for i in range(actual_bs)], [50304] * actual_bs),
                    ([128] * actual_bs, [0] * actual_bs),
                    ([837] + [128] * (actual_bs - 1), [50304] * actual_bs),
                ]
            )
        cases.append(([2045, 1, 1], [0, 0, 0]))
    cases.append(([1] * batch_size, [0] * batch_size))
    # Exercise the real outer startup loop and shared-pool ownership.
    owner = object.__new__(PrefillGraph)
    # KDA has no CED narrowing; the split encoder/decoder capture stays off.
    owner._narrowing = None
    owner.config = SimpleNamespace(
        global_rank=1,
        context_len=bucket,
        max_num_seqs=4,
        data_parallel_size=1,
        prefill_graph_capture_batch_sizes=[batch_size],
    )
    owner.disable = False
    owner.dp_size, owner.num_warmup, owner._pool = 1, 1, None
    owner.capture_buckets = [bucket]
    owner._captures = {}
    owner.input_buffers = SimpleNamespace(input_ids_buf=torch.ones(bucket))
    owner._embed_tokens = lambda ids: ids
    owner._land_input_embeds = lambda *args: None
    owner.attn_backend = backend
    owner._run_inner = lambda bucket: (forward(), None)

    def dummy(bucket, bs):
        backend.forward_metadata = live_metadata([bucket // bs] * bs, [0] * bs, 2)
        return SimpleNamespace(
            bs=bs,
            capture_hidden_mode=CaptureHiddenMode.NULL,
            forward_mode=ForwardMode.EXTEND,
        )

    owner.make_dummy_batch = dummy
    owner._capture_all_buckets(None, None, NULL_MEMORY_DELTA_OBSERVER)
    assert set(owner._captures) == {(bucket, None), (bucket, batch_size)}
    capture, captured = owner._captures[bucket, batch_size]
    output = captured.hidden_states
    assert capture.num_segments == 1
    fixed = backend._prefill_metadata[bucket, batch_size].prefill_checkpoint_batch
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
        source = backend.forward_metadata
        reset()
        expected = forward()[: sum(lengths)].clone()
        expected_conv, expected_states = conv.clone(), states.clone()
        # Uncaptured shapes use the same builder with transient storage. Check
        # its eager output/state too, not just eager on a retained graph buffer.
        backend.forward_metadata = prepare_kda_prefill_metadata(
            source, bucket, 128, None
        )
        reset()
        fresh_eager = forward()[: sum(lengths)].clone()
        torch.testing.assert_close(fresh_eager, expected, rtol=0, atol=0)
        torch.testing.assert_close(conv, expected_conv, rtol=0, atol=0)
        torch.testing.assert_close(states, expected_states, rtol=0, atol=0)
        backend.forward_metadata = source
        backend.prepare_prefill_metadata(
            bucket, batch_size, ctx.forward_mode, capture=False
        )
        reset()
        eager = forward()[: sum(lengths)].clone()
        torch.testing.assert_close(eager, expected, rtol=0, atol=0)
        torch.testing.assert_close(conv, expected_conv, rtol=0, atol=0)
        torch.testing.assert_close(states, expected_states, rtol=0, atol=0)
        reset()
        capture.replay(valid_rows=sum(lengths))
        assert addresses == {
            name: getattr(fixed, name).data_ptr() for name in addresses
        }
        torch.testing.assert_close(output[: sum(lengths)], expected, rtol=0, atol=0)
        torch.testing.assert_close(conv, expected_conv, rtol=0, atol=0)
        torch.testing.assert_close(states, expected_states, rtol=0, atol=0)
        assert torch.count_nonzero(output[sum(lengths) :]) == 0


@pytest.mark.parametrize(
    "captured,compatible,transfer,expected",
    [
        (True, True, False, 2),
        (True, False, False, 1),
        (True, True, True, 1),
        (False, True, False, 1),
    ],
)
@pytest.mark.parametrize("batch_size", [1, 2, 3, 4])
def test_outer_owner_selects_matching_graph_and_refreshes_before_replay(
    captured, compatible, transfer, expected, batch_size
):
    from contextlib import nullcontext
    from unittest.mock import patch

    from tokenspeed.runtime.execution.forward_batch_info import ForwardMode
    from tokenspeed.runtime.execution.prefill_graph import CapturedForward, PrefillGraph

    events = []
    capture_bs = 4 if batch_size == 3 else batch_size
    num_tokens = 7 if batch_size == 3 else 8

    def prepare(bucket, bs, mode, *, capture):
        assert (bucket, bs, mode, capture) == (
            8,
            capture_bs if captured else batch_size,
            ForwardMode.EXTEND,
            False,
        )
        events.append("refresh")
        return compatible

    def capture(label):
        return SimpleNamespace(replay=lambda **kwargs: events.append(label))

    owner = object.__new__(PrefillGraph)
    # KDA has no CED narrowing; the split encoder/decoder capture stays off.
    owner._narrowing = None
    owner._captures = {
        (8, None): (capture("ordinary"), CapturedForward(torch.ones(8, 4), None)),
        (8, capture_bs): (
            capture("inline"),
            CapturedForward(torch.full((8, 4), 2.0), None),
        ),
    }
    if not captured:
        del owner._captures[8, capture_bs]
    owner.dp_size = 1
    owner.attn_backend = SimpleNamespace(
        step_counter=object() if transfer else None, prepare_prefill_metadata=prepare
    )
    owner._replay_bucket = lambda ctx: 8
    owner._log_engaged_once = lambda *args: None
    owner._embed_tokens = lambda ids: torch.zeros(8, 4)
    owner._land_input_embeds = lambda *args: None
    owner._padded_to = lambda *args: nullcontext()
    owner.config = SimpleNamespace(model_is_mrope=False)
    owner.input_buffers = SimpleNamespace(
        input_ids_buf=torch.ones(8, dtype=torch.int64),
        positions_buf=torch.ones(8, dtype=torch.int64),
    )
    owner.text_model = SimpleNamespace(
        lm_head=None, logits_processor=lambda ids, hidden, *args: hidden
    )
    ctx = SimpleNamespace(
        input_num_tokens=num_tokens, bs=batch_size, forward_mode=ForwardMode.EXTEND
    )
    with patch(
        "tokenspeed.runtime.execution.prefill_graph.LogitsMetadata.from_forward_context",
        return_value=None,
    ):
        result = owner.replay(ctx, torch.zeros(num_tokens, dtype=torch.int64), None)
    torch.testing.assert_close(result, torch.full((num_tokens, 4), float(expected)))
    assert ctx.bs == batch_size
    assert events == ([] if transfer else ["refresh"]) + [
        "inline" if expected == 2 else "ordinary"
    ]
    assert len(owner._captures) == 1 + int(captured)


@pytest.mark.parametrize(
    "bs,tokens,expected",
    [
        (1, 8, (8, 1)),
        (2, 8, (8, 2)),
        (3, 7, (8, 4)),
        (3, 8, (16, 4)),
        (3, 16, (16, None)),
        (4, 8, (8, 4)),
        (5, 8, (8, None)),
    ],
)
def test_request_bucket_selection_reserves_dummy_scan_tokens(bs, tokens, expected):
    from tokenspeed.runtime.execution.forward_batch_info import ForwardMode
    from tokenspeed.runtime.execution.prefill_graph import PrefillGraph

    owner = object.__new__(PrefillGraph)
    # KDA has no CED narrowing; the split encoder/decoder capture stays off.
    owner._narrowing = None
    owner.disable = False
    owner.dp_size = 1
    owner.attn_backend = SimpleNamespace(step_counter=None)
    owner._captures = {
        (bucket, count): None for bucket in (8, 16) for count in (None, 1, 2, 4)
    }
    owner._captured_hidden_mode = None
    owner._select_bucket = lambda ctx: 8 if ctx.input_num_tokens <= 8 else 16
    ctx = SimpleNamespace(
        bs=bs,
        input_num_tokens=tokens,
        num_extends=bs,
        forward_mode=ForwardMode.EXTEND,
        draft_narrowing=None,
        capture_hidden_mode=None,
    )
    bucket = owner._replay_bucket(ctx)
    assert (bucket, owner._merged_capture_bs(bucket, ctx)) == expected


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
    from tokenspeed.runtime.execution.forward_batch_info import (
        CaptureHiddenMode,
        ForwardMode,
    )
    from tokenspeed.runtime.execution.prefill_graph import CapturedForward, PrefillGraph

    owner = object.__new__(PrefillGraph)
    # KDA has no CED narrowing; the split encoder/decoder capture stays off.
    owner._narrowing = None
    owner.config = SimpleNamespace(
        global_rank=1,
        context_len=16,
        max_num_seqs=4,
        data_parallel_size=1,
        prefill_graph_capture_batch_sizes=[1, 2],
    )
    owner.disable = False
    owner.dp_size = 1
    owner.capture_buckets = [8]
    owner._captures = {}
    owner.input_buffers = SimpleNamespace(input_ids_buf=torch.ones(8))
    owner._embed_tokens = lambda ids: ids
    owner._land_input_embeds = lambda *args: None
    active_count = None

    def prepare(bucket, bs, mode, *, capture):
        nonlocal active_count
        assert (bucket, bs, mode) == (8, owner._ctx.bs, ForwardMode.EXTEND)
        active_count = bs if capture else None
        return True

    owner.attn_backend = SimpleNamespace(
        prepare_prefill_metadata=prepare,
        admits_prefill_graph=lambda *_a, **_k: True,
    )
    owner.make_dummy_batch = lambda bucket, bs: SimpleNamespace(
        bs=bs,
        capture_hidden_mode=CaptureHiddenMode.NULL,
        forward_mode=ForwardMode.EXTEND,
    )

    def capture(bucket, wrapper, observer):
        label = (owner._ctx.bs, active_count)
        return label, CapturedForward(torch.ones(bucket, 1), None)

    owner._capture_bucket = capture
    owner._capture_all_buckets(None, None, NULL_MEMORY_DELTA_OBSERVER)
    assert owner._captures[8, None][0] == (1, None)
    assert set(owner._captures) == {(8, None), (8, 1), (8, 2)}
    for (_, bs), (capture, _) in owner._captures.items():
        if bs is not None:
            assert capture == (bs, bs)
    assert owner._ctx is None


@pytest.mark.parametrize(
    "mode,dp_size,use_graph,prepared",
    [
        ("EXTEND", 1, False, True),
        ("EXTEND", 1, True, False),
        ("EXTEND", 2, False, False),
        ("MIXED", 1, False, False),
        ("DECODE", 1, False, False),
    ],
)
def test_executor_prepares_eager_prefill_metadata_before_any_layer(
    mode, dp_size, use_graph, prepared
):
    from tokenspeed.runtime.execution.forward_batch_info import ForwardMode
    from tokenspeed.runtime.execution.model_executor import ModelExecutor

    events = []
    mode = ForwardMode[mode]
    ctx = SimpleNamespace(forward_mode=mode, bs=2, input_num_tokens=7)
    executor = object.__new__(ModelExecutor)
    executor.config = SimpleNamespace(pp_size=1, data_parallel_size=dp_size)
    executor._active_positions_override = torch.arange(7)
    executor._active_multimodal_context = None
    executor.runtime_states = SimpleNamespace(has_request_token_history=False)
    executor.input_buffers = SimpleNamespace(
        input_ids_buf=torch.arange(7), ngram_model_kwargs=lambda _: {}
    )

    def prepare(capacity, bs, forward_mode, *, capture):
        assert (capacity, bs, forward_mode, capture) == (7, 2, mode, False)
        events.append("prepare")
        return True

    executor.attn_backend = SimpleNamespace(prepare_prefill_metadata=prepare)
    executor.prefill_graph = SimpleNamespace(
        can_run=lambda *_: use_graph,
        replay=lambda *_: events.append("graph"),
    )
    executor.model_runner = SimpleNamespace(
        forward=lambda *_, **__: events.append("eager")
    )
    executor._run_target_forward(ctx)
    assert events == (["prepare"] if prepared else []) + [
        "graph" if use_graph else "eager"
    ]


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q", "-rs"]))
