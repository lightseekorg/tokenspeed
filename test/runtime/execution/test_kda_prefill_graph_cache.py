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
from unittest.mock import patch

import numpy as np
import pytest
import torch

# ``test/`` (for ``ci_system``) and the repo root (for ``test.runtime.*``).
_TEST_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, _TEST_DIR)
sys.path.insert(0, os.path.dirname(_TEST_DIR))
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
from tokenspeed.runtime.utils.common import maybe_inference_mode  # noqa: E402


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
    # No Host L2 tier: nothing to wait for before the replay.
    owner.token_to_kv_pool = SimpleNamespace(layerwise_load_tracker=None)
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


_BUCKET = 8


@pytest.fixture
def kimi_host_cache():
    """A Kimi-K3 TP8 pool with a Host L2 tier, the device handle that runs a
    plan's cache ops on it, and a bare prefill graph owner over the pool, which
    captures a test's forward (its ``_run_inner``) in one bucket after the
    Host cache executor exists, as a server does.

    Page 1 of every cache group holds random bytes (finite floats in the first
    KDA layer's recurrent state) and is stored to Host page 1, which a test
    then loads back into another page, as a prefix hit does. The owner's
    backend sends the first KDA layer to a stand-in KDA leaf that adds the
    request's input state to its input. The leaf reads that state through the
    pool, as KDA does, so the read waits for the layer's Host load when it
    runs at an eager break. The ops stand in for ``Cache.WriteBackOp`` and
    ``Cache.LoadBackOp``, which have no Python constructor.
    """
    if not torch.cuda.is_available():
        pytest.skip("requires CUDA")
    ts = pytest.importorskip("tokenspeed_scheduler")
    from test.runtime.conftest import kda_layer_id, make_kimi_pool, mla_layer_id

    from tokenspeed.runtime.cache.l2.executor import HostCacheExecutor
    from tokenspeed.runtime.cache.l2.sizing import RetractionPoolRequest
    from tokenspeed.runtime.execution.context import ForwardContext
    from tokenspeed.runtime.execution.device import DeviceHandle
    from tokenspeed.runtime.execution.forward_batch_info import ForwardMode
    from tokenspeed.runtime.execution.model_executor import ModelExecutor
    from tokenspeed.runtime.execution.output_layout import ForwardOutputLayout
    from tokenspeed.runtime.execution.prefill_graph import PrefillGraph
    from tokenspeed.runtime.layers.attention.backends.hybrid.linear import (
        HybridLinearAttnBackend,
    )
    from tokenspeed.runtime.layers.attention.backends.state.kda import KdaAttnBackend

    class WriteBackOp:
        # A stream-ordered store of page ``src`` of each group to ``dst``.
        def __init__(self, op_id, groups, src, dst):
            self.op_ids = [op_id]
            self.group_ids = [list(groups)]
            self.src_pages = [[src] * len(groups)]
            self.dst_pages = [[dst] * len(groups)]
            self.source_pinned = [False]

    class LoadBackOp:
        def __init__(self, op_id, groups, src, dst):
            self.op_ids = [op_id]
            self.group_ids = [list(groups)]
            self.src_pages = [[src] * len(groups)]
            self.dst_pages = [[dst] * len(groups)]

    class Leaf(KdaAttnBackend):
        def forward_extend(self, q, k, v, layer, pool, bs, **kwargs):
            _, recurrent = self._state_components(layer.layer_id)
            pages = self.forward_metadata.state_in_blocks_by_group["state"].long()
            return q + recurrent[pages].reshape(-1)[: q.numel()].view_as(q)

    pool = make_kimi_pool("cuda", usable_pages=3)
    executor = HostCacheExecutor(
        pool,
        draft_pool=None,
        l2_tier=True,
        host_ratio=1.0,
        host_size_gb=0,
        snapshot_pool=RetractionPoolRequest(
            host_gb=0.0,
            ratio=0.0,
            max_retracted_requests=0,
            tail_lcm_blocks_per_request=0,
        ),
        slot_state_exporters=None,
        io_backend="kernel",
        attn_tp_rank=0,
        kvp_rank=0,
        nprocs_per_node=1,
        world_cpu_group=None,
    )
    # A model executor without a model: the fields the handle reads.
    model_executor = ModelExecutor.__new__(ModelExecutor)
    model_executor.device = "cuda"
    model_executor.device_module = torch.cuda
    model_executor.default_stream = torch.cuda.default_stream()
    model_executor.execution_stream = torch.cuda.Stream()
    model_executor.token_to_kv_pool = pool
    model_executor.draft_token_to_kv_pool = None
    model_executor._cache_runtime_contract = pool.arena.runtime_contract
    model_executor._cache_kvp_rank = 0
    model_executor.drafter = None
    handle = DeviceHandle(model_executor, host_cache_executor=executor)
    handle.bind_forward_thread()

    leaf = object.__new__(Leaf)
    configure_prefill(leaf)
    leaf.cache_pool = pool
    full = object.__new__(Leaf)
    configure_prefill(full)
    full.forward_metadata = None
    full.device = torch.device("cuda")
    hybrid = HybridLinearAttnBackend(full, leaf, [mla_layer_id(pool)])
    ctx = ForwardContext(
        attn_backend=hybrid,
        token_to_kv_pool=pool,
        bs=1,
        num_extends=1,
        output_layout=ForwardOutputLayout(1, 1, 0, 1),
        input_num_tokens=_BUCKET,
        forward_mode=ForwardMode.EXTEND,
    )
    owner = object.__new__(PrefillGraph)
    owner.disable = False
    owner.capture_buckets = [_BUCKET]
    owner.num_warmup = 1
    owner._narrowing = None
    owner._expert_load_rows = None
    owner._engaged_logged = set()
    owner._ctx = None
    owner._pool = None
    owner._handoff_storage = {}
    owner._outputs = None
    owner._captures = {}
    owner._encoders = {}
    owner._decoders = {}
    owner._captured_load_tracker = None
    owner.dp_size = 1
    owner.attn_backend = hybrid
    owner.token_to_kv_pool = pool
    owner.config = SimpleNamespace(
        world_size=1,
        model_is_mrope=False,
        global_rank=0,
        device="cuda",
        gpu_id=torch.cuda.current_device(),
        context_len=_BUCKET,
        max_num_seqs=1,
        data_parallel_size=1,
        prefill_graph_capture_batch_sizes=None,
    )
    owner.input_buffers = SimpleNamespace(
        input_ids_buf=torch.ones(_BUCKET, dtype=torch.int64)
    )
    owner.make_dummy_batch = lambda num_tokens, bs: ctx
    owner._replay_bucket = lambda ctx: _BUCKET
    owner._embed_tokens = torch.nn.Embedding(2, 4)
    owner._land_input_embeds = lambda *args: None
    owner.text_model = SimpleNamespace(
        lm_head=None, logits_processor=lambda ids, hidden, *args: hidden
    )
    groups = [group.group_id for group in executor.layout.groups]
    try:
        with (
            patch.object(ts.Cache, "WriteBackOp", WriteBackOp),
            patch.object(ts.Cache, "LoadBackOp", LoadBackOp),
            patch(
                "tokenspeed.runtime.execution.prefill_graph.LogitsMetadata."
                "from_forward_context",
                return_value=None,
            ),
        ):
            # A first load and layerwise waits (a KDA and an MLA layer: the
            # waiter kernel specializes on the layer index), so a held-back
            # load below sets up no transfer tables or kernels: either would
            # wait for the device, the held-back copy included.
            handle.execute(
                ts.ExecutionPlan(),
                None,
                cache_ops=[LoadBackOp(1, range(len(groups)), 1, 3)],
            )
            pool.get_component(kda_layer_id(pool), "recurrent_state")
            pool.get_key_buffer(mla_layer_id(pool))
            torch.manual_seed(37)
            pool.arena.buffer.random_(0, 256)
            pool.get_component(kda_layer_id(pool), "recurrent_state")[1].normal_()
            # Its plan also points the layerwise waits at no load again, so
            # the tests capture with none, as a server does.
            handle.execute(
                ts.ExecutionPlan(),
                None,
                cache_ops=[WriteBackOp(2, range(len(groups)), 1, 1)],
            )
            torch.cuda.synchronize()
            acks = sorted(int(ack.op_id) for ack in handle.poll_cache_results())
            assert acks == [1, 2]
            yield SimpleNamespace(
                ts=ts,
                pool=pool,
                executor=executor,
                handle=handle,
                groups=groups,
                LoadBackOp=LoadBackOp,
                leaf=leaf,
                hybrid=hybrid,
                owner=owner,
                ctx=ctx,
            )
    finally:
        handle.shutdown_cache()


@pytest.mark.parametrize(
    ("tracker_at_capture", "declared"), [(True, True), (False, True), (True, False)]
)
def test_a_replay_writes_a_loaded_tail_page_after_the_load(
    kimi_host_cache, tracker_at_capture, declared
):
    """A Host tail hit loads the Host block into the hitter's own page, and
    the chunk's MLA prologue writes its new rows into the rest of that page
    inside a captured segment, which cannot hold the layerwise wait (the
    attention break after it waits too late). With the page's H2D held back
    on the load stream, a replay that did not wait for the load would have
    its rows overwritten by the Host block; they survive. The capture notes
    the layer for the replay to wait for before the segment; a capture made
    before the pool held the tracker notes none, and its replay waits for
    the whole load. So does a replay on a pool whose class does not declare
    every access fenced, here a subclass of the Kimi pool's class, which
    does not inherit the declaration; its write goes around the fence and
    notes nothing. Under --disable-kda-prefill-graph, as here, every
    captured segment is ordinary.
    """
    from test.runtime.conftest import MLA_LATENT_DIM, mla_layer_id

    cache = kimi_host_cache
    pool, owner = cache.pool, cache.owner
    cache.leaf._prefill_graph_enabled = False
    layer = mla_layer_id(pool)
    page_size = pool.arena.prefix_granularity
    latent = (
        pool.kv_buffer[layer].view(torch.uint8).view(-1, page_size, 1, MLA_LATENT_DIM)
    )
    # The hit covers the first half of page 2; the chunk writes the rest.
    page, hit = 2, page_size // 2
    host_rows = latent[1].clone()
    slots = torch.arange(page * page_size + hit, (page + 1) * page_size, device="cuda")
    rows = torch.randint(
        0, 256, (page_size - hit, 1, MLA_LATENT_DIM), dtype=torch.uint8, device="cuda"
    )
    hidden = torch.ones(_BUCKET, 4, device="cuda")

    def forward():
        if declared:
            target = pool.kv_write_target(layer, slots, None)
            target.kv_cache.view(torch.uint8)[target.slots] = rows
        else:
            # Around the fence, as an undeclared pool's accessor may write.
            latent[page, hit:] = rows
        return hidden * 2

    owner._run_inner = lambda bucket: (forward(), None)
    # Without the tracker: as if captured before the Host cache executor.
    with patch.object(
        pool,
        "layerwise_load_tracker",
        pool.layerwise_load_tracker if tracker_at_capture else None,
    ):
        owner.capture(None, entries=None, observer=NULL_MEMORY_DELTA_OBSERVER)
    capture, _ = owner._captures[_BUCKET, None]
    assert [bool(waits) for waits in capture.segment_waits] == [
        tracker_at_capture and declared
    ]

    group = cache.groups.index("full_attention")
    pool.arena.zero_blocks({"full_attention": np.asarray([page])})
    with torch.cuda.stream(cache.executor.load_stream):
        torch.cuda._sleep(200_000_000)
    cache.handle.execute(
        cache.ts.ExecutionPlan(),
        None,
        cache_ops=[cache.LoadBackOp(3, [group], 1, page)],
    )
    kimi_pool_class = type(pool)
    if not declared:

        class Undeclared(kimi_pool_class):
            pass

        pool.__class__ = Undeclared
    try:
        # Under inference mode, as the forward step replays.
        with maybe_inference_mode():
            result = owner.replay(
                cache.ctx, torch.ones(_BUCKET, dtype=torch.int64), None
            )
    finally:
        pool.__class__ = kimi_pool_class
    assert not cache.executor.load_stream.query()
    torch.cuda.synchronize()
    assert [int(ack.op_id) for ack in cache.handle.poll_cache_results()] == [3]
    assert torch.equal(result, hidden * 2)
    assert torch.equal(latent[page, :hit], host_rows[:hit])
    assert torch.equal(latent[page, hit:], rows)

    # A plan without cache ops points the layerwise waits at no load.
    assert pool.layerwise_load_tracker.consumer_indices
    cache.handle.execute(cache.ts.ExecutionPlan(), None, cache_ops=[])
    assert pool.layerwise_load_tracker.consumer_indices == ()


def test_an_inline_kda_replay_reads_a_loaded_state_after_the_load(kimi_host_cache):
    """A prefix hit loads the Host State checkpoint into the request's input
    state page, which an inline KDA capture reads inside its segment, with no
    layerwise wait there. With the H2D held back on the load stream, the
    inline replay computes what the ordinary capture computes, whose KDA
    layer reads at an eager break that waits for the load (the path
    --disable-kda-prefill-graph takes): both see the Host state."""
    from test.runtime.conftest import kda_layer_id

    from tokenspeed.runtime.execution.breakable_cuda_graph import BreakableCapture
    from tokenspeed.runtime.execution.forward_batch_info import ForwardMode
    from tokenspeed.runtime.execution.prefill_graph import CapturedForward

    cache = kimi_host_cache
    pool, leaf, owner = cache.pool, cache.leaf, cache.owner
    layer = kda_layer_id(pool)
    group_id = pool.state_group_by_layer[layer]
    page = 2
    value = torch.ones(_BUCKET, 4, device="cuda")

    def forward():
        out = cache.hybrid.forward(
            value * 2,
            None,
            None,
            SimpleNamespace(layer_id=layer),
            None,
            ForwardMode.EXTEND,
            1,
            True,
            None,
        )
        return out * 3

    leaf.forward_metadata = metadata("cuda", page)
    # The owner captures the ordinary graph; the inline one is captured below
    # on the request's own metadata, which the owner's dummy batch does not
    # rebuild.
    leaf._prefill_graph_enabled = False
    owner._run_inner = lambda bucket: (forward(), None)
    owner.capture(None, entries=None, observer=NULL_MEMORY_DELTA_OBSERVER)
    ordinary, _ = owner._captures[_BUCKET, None]
    assert ordinary.num_segments == 3
    leaf._prefill_graph_enabled = True
    leaf.prepare_prefill_metadata(_BUCKET, 1, ForwardMode.EXTEND, capture=True)
    forward()
    torch.cuda.synchronize()
    inline = BreakableCapture()
    with inline:
        inline_output = forward()
    assert inline.num_segments == 1
    owner._captures[_BUCKET, 1] = (inline, CapturedForward(inline_output, None))

    host_state = pool.get_component(layer, "recurrent_state")[1].clone()
    results = {}
    for op_id, inline_kda in ((3, False), (4, True)):
        leaf._prefill_graph_enabled = inline_kda
        leaf.forward_metadata = metadata("cuda", page)
        pool.arena.zero_blocks({group_id: np.asarray([page])})
        with torch.cuda.stream(cache.executor.load_stream):
            torch.cuda._sleep(200_000_000)
        cache.handle.execute(
            cache.ts.ExecutionPlan(),
            None,
            cache_ops=[
                cache.LoadBackOp(op_id, [cache.groups.index(group_id)], 1, page)
            ],
        )
        with maybe_inference_mode():
            results[inline_kda] = owner.replay(
                cache.ctx, torch.ones(_BUCKET, dtype=torch.int64), None
            ).clone()
        assert not cache.executor.load_stream.query()
        torch.cuda.synchronize()
        assert [int(ack.op_id) for ack in cache.handle.poll_cache_results()] == [op_id]
    expected = (value * 2 + host_state.reshape(-1)[: value.numel()].view_as(value)) * 3
    assert torch.equal(results[False], expected)
    assert torch.equal(results[True], results[False])


@pytest.mark.parametrize("released_by", ["flags", "events"])
def test_a_replay_waits_before_each_segment_only_for_the_layers_it_touches(
    kimi_host_cache, released_by
):
    """A replay waits before each captured segment for the highest layer whose
    cache that segment touches, not for the whole Host load. The first segment
    reads a KDA layer's state; the second writes rows of a later MLA layer's
    page, then reads the KDA state again. The owner captures them with the
    executor's tracker. In the load, only the MLA layer's page copy is held
    back, and the layers are released in order, by flag as the Triton copy
    kernel does or by event as the DMA path does. The first segment and the
    break after it run before that copy lands, and the second segment's rows
    land after it and survive. A replay that waited for the whole load, or for
    a lower layer before the second segment, fails here.
    """
    from test.runtime.conftest import MLA_LATENT_DIM, kda_layer_id, mla_layer_id

    from tokenspeed_kernel.ops.kvcache.host_transfer import wait_layer_ready

    from tokenspeed.runtime.execution.breakable_cuda_graph import break_here

    cache = kimi_host_cache
    pool, owner = cache.pool, cache.owner
    cache.leaf._prefill_graph_enabled = False
    tracker = pool.layerwise_load_tracker
    kda, mla = kda_layer_id(pool), mla_layer_id(pool)
    assert kda < mla
    page_size = pool.arena.prefix_granularity
    latent = (
        pool.kv_buffer[mla].view(torch.uint8).view(-1, page_size, 1, MLA_LATENT_DIM)
    )
    page = 2
    slots = torch.arange(page * page_size, (page + 1) * page_size, device="cuda")
    rows, host_rows = torch.randint(
        0, 256, (2, page_size, 1, MLA_LATENT_DIM), dtype=torch.uint8, device="cuda"
    )
    value = torch.ones(_BUCKET, 4, device="cuda")
    handoff = torch.empty_like(value)
    first_segment_done = torch.cuda.Event()

    def mark(hidden):
        first_segment_done.record()
        return hidden

    pool.get_component(kda, "recurrent_state")[page].normal_()

    def state_rows():
        state = pool.get_component(kda, "recurrent_state")[page]
        return state.reshape(-1)[: value.numel()].view_as(value)

    def forward():
        hidden = break_here(mark, handoff, value + state_rows())
        target = pool.kv_write_target(mla, slots, None)
        target.kv_cache.view(torch.uint8)[target.slots] = rows
        return hidden + state_rows()

    owner._run_inner = lambda bucket: (forward(), None)
    owner.capture(None, entries=None, observer=NULL_MEMORY_DELTA_OBSERVER)

    # The copy kernel cannot hold back a single layer, so the test releases
    # the layers itself on the load stream, in layer order, by flag with the
    # waiter the executor binds on the Triton copy path, or by event.
    load = tracker.begin_load()
    events = tracker.event_sets[load]
    flags = None
    if released_by == "flags":
        flags = torch.zeros(pool.layer_num, dtype=torch.int32, device="cuda")
        events.layer_ready_flags, events.wait_layer_ready = flags, wait_layer_ready
    load_stream = cache.executor.load_stream
    load_stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(load_stream):
        for layer in range(pool.layer_num):
            if layer == mla:
                torch.cuda._sleep(200_000_000)
                latent[page].copy_(host_rows)
            if flags is not None:
                flags[layer].fill_(1)
            events.layer_done_events[layer].record()
    tracker.set_consumers(load)
    with maybe_inference_mode():
        result = owner.replay(cache.ctx, torch.ones(_BUCKET, dtype=torch.int64), None)
    first_segment_done.synchronize()
    assert not load_stream.query()
    torch.cuda.synchronize()
    assert torch.equal(latent[page], rows)
    assert torch.equal(result, value + state_rows() + state_rows())


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q", "-rs"]))
