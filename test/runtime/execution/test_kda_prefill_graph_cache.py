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
from tokenspeed.runtime.layers.attention.backends.state.mamba import (  # noqa: E402
    MambaForwardMetadata,
)
from tokenspeed.runtime.layers.attention.backends.state.prefill_capacity import (  # noqa: E402
    _clone_metadata,
    prepare_capacity_prefill_metadata,
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


def _live_metadata(lengths, device):
    lengths = torch.tensor(lengths)
    host = torch.cat((torch.zeros(1, dtype=torch.int64), lengths.cumsum(0)))
    bounds = host.to(device=device, dtype=torch.int32)
    return MambaForwardMetadata(
        query_start_loc=bounds,
        scan_query_start_loc=bounds.long(),
        query_start_loc_int64=bounds.long(),
        extend_seq_lens_cpu=lengths,
        cu_extend_seq_lens_cpu=host,
        state_in_blocks_by_group={
            "state": torch.zeros(len(lengths), dtype=torch.int32, device=device)
        },
        state_out_blocks_by_group={
            "state": torch.arange(1, len(lengths) + 1, dtype=torch.int32, device=device)
        },
    )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
@pytest.mark.parametrize("live", [1, 2, 3])
def test_capacity_refresh_pads_unused_request_slots_with_one_scan_row(live):
    """A retained shape keeps its request capacity: every unused slot scans one
    row past the live tokens and writes no block, while convolution gives it
    no work at all."""
    lengths = [129, 130, 131][:live]
    retained = prepare_capacity_prefill_metadata(
        _live_metadata([100] * 4, "cuda"), 512, None
    )
    source = _live_metadata(lengths, "cuda")
    assert prepare_capacity_prefill_metadata(source, 512, retained) is retained
    total = sum(lengths)
    padding = 4 - live
    host = source.cu_extend_seq_lens_cpu.tolist()
    expected_scan = host + [total + 1 + i for i in range(padding)]
    assert retained.scan_query_start_loc.tolist() == expected_scan
    assert retained.query_start_loc_int64.tolist() == expected_scan
    assert retained.cu_extend_seq_lens_cpu.tolist() == expected_scan
    assert retained.query_start_loc.tolist() == host + [total] * padding
    assert retained.extend_seq_lens_cpu.tolist() == lengths + [1] * padding
    assert retained.state_out_blocks_by_group["state"].tolist() == (
        list(range(1, live + 1)) + [-1] * padding
    )
    assert retained.prefill_token_extent == 512
    with pytest.raises(ValueError, match="exceed the captured capacity"):
        prepare_capacity_prefill_metadata(_live_metadata([510], "cuda"), 512, retained)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_capacity_scrub_zeroes_rows_past_the_live_total():
    """The padded rows are zeroed by a device-side bound, never a Python int
    the capture would freeze."""
    retained = prepare_capacity_prefill_metadata(
        _live_metadata([2, 2, 2], "cuda"), 8, None
    )
    prepare_capacity_prefill_metadata(_live_metadata([3, 2], "cuda"), 8, retained)
    rows = torch.ones(8, 4, device="cuda")
    extra = torch.ones(8, device="cuda")
    retained.scrub_padding(rows, None, extra)
    assert rows[:5].eq(1).all() and rows[5:].eq(0).all()
    assert extra[:5].eq(1).all() and extra[5:].eq(0).all()


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
    from tokenspeed.runtime.layers.attention.backends.state.prefill_capacity import (
        CapacityPrefillMetadata,
    )

    backend = object.__new__(KdaAttnBackend)
    configure_prefill(backend)
    backend.cache_pool = object()
    original = metadata("cuda", 1)
    backend.forward_metadata = original
    backend.prepare_prefill_metadata(8, 1, ForwardMode.EXTEND, capture=False)
    eager = backend.forward_metadata
    assert isinstance(eager, CapacityPrefillMetadata)
    assert not backend.prefill_metadata_is_capture_ready
    assert not backend._prefill_metadata
    backend.forward_metadata = original
    backend.prepare_prefill_metadata(8, 1, ForwardMode.EXTEND, capture=True)
    retained = backend.forward_metadata
    assert isinstance(retained, type(eager))
    assert retained.capacity == eager.capacity
    for name in (
        "scan_query_start_loc",
        "query_start_loc_int64",
        "cu_extend_seq_lens_cpu",
    ):
        torch.testing.assert_close(getattr(retained, name), getattr(eager, name))
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
def test_outer_graph_replays_lengths_pages_and_states(batch_size):
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
    from tokenspeed.runtime.layers.attention.backends.hybrid.linear import (
        HybridLinearAttnBackend,
    )
    from tokenspeed.runtime.layers.attention.backends.state.kda import KdaAttnBackend

    if not cutedsl_kda_supported():
        pytest.skip("native CuteDSL KDA requires NVIDIA SM100 or SM103")
    torch.manual_seed(42)
    bucket, heads, dim = 2048, 8, 128
    channels = heads * dim
    conv = torch.randn(12, 3 * channels, 3, device="cuda", dtype=torch.bfloat16)
    states = torch.randn(12, heads, dim, dim, device="cuda", dtype=torch.float32)
    initial_conv, initial_states = conv.clone(), states.clone()
    # The live input of the captured projection: merged rows [q|k|v|g|f_a|b],
    # padded to the 128-row grid.
    projection = torch.randn(
        bucket, 4 * channels + 2 * dim, device="cuda", dtype=torch.bfloat16
    )
    backend = object.__new__(KdaAttnBackend)
    configure_prefill(backend)
    backend.__dict__.update(
        is_draft=False,
        cache_pool=object(),
        kda_backend="cutedsl_kda",
        kda_recurrent_layout="v_major",
    )
    backend._layer_state = lambda layer_id: (
        backend.forward_metadata.state_in_blocks_by_group["state"],
        backend.forward_metadata.state_out_blocks_by_group["state"],
        conv,
        states,
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
        f_b_weight=torch.randn(channels, dim, device="cuda", dtype=torch.bfloat16)
        * 0.1,
        A_log=torch.zeros(heads, device="cuda"),
        dt_bias=torch.zeros(channels, device="cuda"),
        lower_bound=-5.0,
        layer_id=0,
        seq_len=bucket,
    )
    # Layer 1 would be full attention; this forward has only the KDA layer 0.
    full = object.__new__(KdaAttnBackend)
    full.forward_metadata = None
    full.device = torch.device("cuda")
    hybrid = HybridLinearAttnBackend(full, backend, [1])

    def forward(compact):
        # As in the model: the captured projection writes fresh rows (the
        # capacity scrub zeroes padded rows in place), the attention gets
        # sibling views of them and the output gate is read after it.
        # ``compact`` is the q|k|v copy the model used to make.
        rows = projection.clone()
        mixed_qkv = rows[:, : 3 * channels]
        core = hybrid.forward(
            None,
            None,
            None,
            None,
            None,
            ForwardMode.EXTEND,
            backend.forward_metadata.extend_seq_lens_cpu.numel(),
            True,
            None,
            mixed_qkv=mixed_qkv.contiguous() if compact else mixed_qkv,
            f_a_out=rows[:, 4 * channels : 4 * channels + dim],
            beta_raw=rows[:, 4 * channels + dim : 4 * channels + dim + heads],
            **kwargs,
        )
        return core.flatten(1) * rows[: core.shape[0], 3 * channels : 4 * channels]

    def reset():
        conv.copy_(initial_conv)
        states.copy_(initial_states)

    def live_metadata(lengths, prefixes, page_shift):
        lengths = torch.tensor(lengths)
        host = torch.cat((torch.zeros(1, dtype=torch.int64), lengths.cumsum(0)))
        bounds = host.to(device="cuda", dtype=torch.int32)
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
            conv_prefill_metadata=build_causal_conv1d_prefill_metadata(
                bounds, lengths, CAUSAL_CONV1D_BLOCK_M
            ),
        )

    cases = []
    # Reuse ONE capture across changing resumed-row counts, rows, and lengths;
    # prefix hits end at arbitrary tokens.
    for resumed in range(batch_size + 1):
        for tail_lengths, prefix in [([837, 325, 197, 197], 0), ([70] * 4, 127)]:
            lengths = tail_lengths[:resumed] + [128] * (batch_size - resumed)
            prefixes = [prefix] * resumed + [0] * (batch_size - resumed)
            cases.extend([(lengths, prefixes), (lengths[::-1], prefixes[::-1])])
        cases.append(
            ([128] * batch_size, [127] * resumed + [0] * (batch_size - resumed))
        )
    if batch_size == 1:
        cases.extend([([837], [50303]), ([769], [0]), ([1023], [129])])
    if batch_size == 2:
        cases.extend([([868, 869], [50303, 50303]), ([869, 868], [50303, 50303])])
    if batch_size == 4:
        # Reuse BS4 storage across full -> padded -> full transitions. Compare
        # every state block, including the null page and the previous fourth row.
        for actual_bs in (3, 1, 2, 3):
            cases.extend(
                [
                    ([64 + i for i in range(actual_bs)], [50303] * actual_bs),
                    ([128] * actual_bs, [0] * actual_bs),
                    ([837] + [128] * (actual_bs - 1), [50303] * actual_bs),
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
    owner.capture_buckets, owner.decoder_buckets = [bucket], []
    owner._handoff_storage, owner._outputs = {}, None
    owner._captures = {}
    owner.input_buffers = SimpleNamespace(input_ids_buf=torch.ones(bucket))
    owner._embed_tokens = lambda ids: ids
    owner._land_input_embeds = lambda *args: None
    owner.attn_backend = hybrid
    owner._run_inner = lambda bucket: (forward(compact=False), None)

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
    # The ordinary capture: projection graph, eager attention break, gate graph.
    ordinary, ordinary_captured = owner._captures[bucket, None]
    ordinary_output = ordinary_captured.hidden_states
    assert ordinary.num_segments == 3
    # Compared bytewise below; same dtypes make same-shape byte views.
    assert output.dtype == ordinary_output.dtype == torch.bfloat16
    fixed = backend._prefill_metadata[bucket, batch_size]
    retained = {
        "query_start_loc": fixed.query_start_loc,
        "scan_query_start_loc": fixed.scan_query_start_loc,
        "query_start_loc_int64": fixed.query_start_loc_int64,
        "state_in": fixed.state_in_blocks_by_group["state"],
        "state_out": fixed.state_out_blocks_by_group["state"],
    }
    addresses = {name: tensor.data_ptr() for name, tensor in retained.items()}
    ctx = SimpleNamespace(
        forward_mode=ForwardMode.EXTEND, bs=batch_size, num_extends=batch_size
    )
    for index, (lengths, prefixes) in enumerate(cases * 2):
        # New live input rows for every forward.
        projection.normal_()
        backend.forward_metadata = live_metadata(lengths, prefixes, 2 + index % 2)
        source = backend.forward_metadata
        reset()
        expected = forward(compact=True)[: sum(lengths)].clone()
        assert expected.dtype == torch.bfloat16
        expected_conv, expected_states = conv.clone(), states.clone()
        # Uncaptured shapes use the same builder with transient storage. Check
        # its eager output/state too, not just eager on a retained graph buffer.
        backend.forward_metadata = prepare_capacity_prefill_metadata(
            source, bucket, None
        )
        reset()
        fresh_eager = forward(compact=False)[: sum(lengths)].clone()
        assert torch.equal(fresh_eager.view(torch.uint8), expected.view(torch.uint8))
        assert torch.equal(conv.view(torch.uint8), expected_conv.view(torch.uint8))
        assert torch.equal(states.view(torch.uint8), expected_states.view(torch.uint8))
        # The ordinary capture hands the eager break views of the projection
        # its graph just wrote, on that transient metadata (an uncaptured
        # request count) or on the plain one (forwards the capacity layout
        # does not admit, such as MIXED batches).
        for metadata in (backend.forward_metadata, source):
            backend.forward_metadata = metadata
            reset()
            ordinary.replay(valid_rows=sum(lengths))
            assert torch.equal(
                ordinary_output[: sum(lengths)].view(torch.uint8),
                expected.view(torch.uint8),
            )
            assert torch.equal(conv.view(torch.uint8), expected_conv.view(torch.uint8))
            assert torch.equal(
                states.view(torch.uint8), expected_states.view(torch.uint8)
            )
            assert torch.count_nonzero(ordinary_output[sum(lengths) :]) == 0
        backend.forward_metadata = source
        backend.prepare_prefill_metadata(
            bucket, batch_size, ctx.forward_mode, capture=False
        )
        reset()
        eager = forward(compact=False)[: sum(lengths)].clone()
        assert torch.equal(eager.view(torch.uint8), expected.view(torch.uint8))
        assert torch.equal(conv.view(torch.uint8), expected_conv.view(torch.uint8))
        assert torch.equal(states.view(torch.uint8), expected_states.view(torch.uint8))
        reset()
        capture.replay(valid_rows=sum(lengths))
        assert addresses == {
            name: tensor.data_ptr() for name, tensor in retained.items()
        }
        assert torch.equal(
            output[: sum(lengths)].view(torch.uint8), expected.view(torch.uint8)
        )
        assert torch.equal(conv.view(torch.uint8), expected_conv.view(torch.uint8))
        assert torch.equal(states.view(torch.uint8), expected_states.view(torch.uint8))
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
    owner._expert_load_rows = None
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
    owner.config = SimpleNamespace(model_is_mrope=False, world_size=1)
    owner.input_buffers = SimpleNamespace(
        input_ids_buf=torch.ones(8, dtype=torch.int64),
        positions_buf=torch.ones(8, dtype=torch.int64),
    )
    owner.text_model = SimpleNamespace(
        lm_head=None, logits_processor=lambda ids, hidden, *args: hidden
    )
    # global_num_tokens: replay reads every rank's live rows (None outside DP).
    ctx = SimpleNamespace(
        input_num_tokens=num_tokens,
        bs=batch_size,
        forward_mode=ForwardMode.EXTEND,
        global_num_tokens=None,
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
    ctx = SimpleNamespace(forward_mode=mode, bs=2, input_num_tokens=7, query_shard=None)
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
