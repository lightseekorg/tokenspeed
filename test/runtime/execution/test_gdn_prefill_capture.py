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

"""GDN layers inside the prefill graph: admission and replay against eager."""

import os
import sys
from dataclasses import replace
from types import SimpleNamespace

import pytest
import torch

sys.path.insert(
    0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
)
from ci_system.ci_register import register_cuda_ci  # noqa: E402

register_cuda_ci(est_time=120, suite="runtime-1gpu")

from tokenspeed_kernel.ops.attention.gdn import (  # noqa: E402
    gdn_chunk_prefill_capturable,
)

from tokenspeed.runtime.execution.forward_batch_info import ForwardMode  # noqa: E402
from tokenspeed.runtime.execution.memory_delta import (  # noqa: E402
    NULL_MEMORY_DELTA_OBSERVER,
)
from tokenspeed.runtime.layers.attention.backends.state.mamba import (  # noqa: E402
    MambaForwardMetadata,
)
from tokenspeed.runtime.layers.attention.configs.linear_attn import (  # noqa: E402
    LinearAttnConfig,
)

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")

K_HEADS, V_HEADS, HEAD_DIM = 2, 4, 128
# The bound pool's prefix granularity, which is also a checkpointing scan's
# interval.
PAGE = 128


def _capturable():
    return gdn_chunk_prefill_capturable(
        torch.bfloat16,
        head_dim=HEAD_DIM,
        value_head_dim=HEAD_DIM,
        num_q_heads=K_HEADS,
        num_v_heads=V_HEADS,
        qk_l2norm=True,
        state_checkpoints=False,
    )


def _config(layer_ids=(0,)):
    """(AttnConfig, softmax spec) of GDN layers; the bound pool sets the page."""
    from tokenspeed.runtime.layers.attention.configs.base import AttnConfig
    from tokenspeed.runtime.layers.attention.configs.mha import MHAConfig

    spec = MHAConfig(
        num_attention_heads=4, num_kv_heads=2, head_dim=128, attn_tp_size=1
    )
    gdn = LinearAttnConfig(
        num_k_heads=K_HEADS,
        num_v_heads=V_HEADS,
        head_k_dim=HEAD_DIM,
        head_v_dim=HEAD_DIM,
        conv_kernel_size=4,
        layer_ids=layer_ids,
        tp_size=1,
    )
    config = AttnConfig(
        device="cuda",
        dtype=torch.bfloat16,
        kv_cache_dtype=torch.bfloat16,
        kv_cache_quant_method="none",
        context_len=4096,
        max_bs=8,
        is_draft=False,
        speculative_num_draft_tokens=1,
        components=(spec, gdn),
    )
    return config, spec


def _backend(*, writes_state_checkpoints_in_forward, layer_ids=(0,)):
    from tokenspeed.runtime.layers.attention.backends.state.gdn import (
        GdnAttnBackend,
    )

    return GdnAttnBackend(
        *_config(layer_ids),
        writes_state_checkpoints_in_forward=writes_state_checkpoints_in_forward,
    )


def _source(lengths, prefixes, page_shift):
    lengths = torch.tensor(lengths)
    host = torch.cat((torch.zeros(1, dtype=torch.int64), lengths.cumsum(0)))
    bounds = host.to(device="cuda", dtype=torch.int32)
    pages = torch.arange(lengths.numel(), device="cuda", dtype=torch.int32)
    pages = pages + page_shift
    from tokenspeed_kernel.ops.attention.gdn.triton import (
        CAUSAL_CONV1D_BLOCK_M,
        build_causal_conv1d_prefill_metadata,
    )

    return MambaForwardMetadata(
        query_start_loc=bounds,
        scan_query_start_loc=bounds.clone(),
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


def _capture(backend, bucket, batch_size, forward, dummy):
    """Capture ``forward`` the way the prefill graph does at startup.

    ``dummy(bucket, bs)`` builds the backend metadata of a capture batch; the
    ordinary and the inline variant are captured for ``batch_size`` requests.
    """
    from tokenspeed.runtime.execution.forward_batch_info import CaptureHiddenMode
    from tokenspeed.runtime.execution.prefill_graph import PrefillGraph

    owner = object.__new__(PrefillGraph)
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
    owner._embedding_model = None
    owner._land_input_embeds = lambda *args: None
    owner.attn_backend = backend
    owner._run_inner = lambda bucket: (forward(), None)

    def make_dummy_batch(bucket, bs):
        dummy(bucket, bs)
        return SimpleNamespace(
            bs=bs,
            capture_hidden_mode=CaptureHiddenMode.NULL,
            forward_mode=ForwardMode.EXTEND,
        )

    owner.make_dummy_batch = make_dummy_batch
    owner._capture_all_buckets(None, None, NULL_MEMORY_DELTA_OBSERVER)
    assert set(owner._captures) == {(bucket, None), (bucket, batch_size)}
    return owner


def test_admission_follows_the_selected_scan():
    """Only a scan that reads its bounds on device is captured; others stay eager."""
    backend = _backend(writes_state_checkpoints_in_forward=False)
    backend.cache_pool = object()
    assert backend.admits_prefill_graph(512, 1, ForwardMode.EXTEND) == _capturable()
    assert not backend.admits_prefill_graph(512, 1, ForwardMode.MIXED)


def test_uncaptured_shapes_keep_the_scheduler_metadata():
    """Eager forwards of shapes without a graph skip the capacity layout."""
    backend = _backend(writes_state_checkpoints_in_forward=False)
    backend.cache_pool = object()
    if not backend.admits_prefill_graph(512, 1, ForwardMode.EXTEND):
        pytest.skip("the selected GDN scan is not graph-capturable here")
    source = _source([300], [0], 2)
    backend.forward_metadata = source
    assert backend.prepare_prefill_metadata(512, 1, ForwardMode.EXTEND, capture=False)
    assert backend.forward_metadata is source
    assert not backend.prefill_metadata_is_capture_ready


@pytest.mark.parametrize("batch_size", [1, 2, 4])
def test_outer_graph_replays_gdn_lengths_pages_and_states(batch_size):
    """Replay equals eager on the retained metadata, which equals plain eager."""
    backend = _backend(writes_state_checkpoints_in_forward=False)
    backend.cache_pool = object()
    if not backend.admits_prefill_graph(2048, batch_size, ForwardMode.EXTEND):
        pytest.skip("the selected GDN scan is not graph-capturable here")
    torch.manual_seed(42)
    bucket = 2048
    key_dim, value_dim = K_HEADS * HEAD_DIM, V_HEADS * HEAD_DIM
    channels = 2 * key_dim + value_dim
    conv = torch.randn(12, channels, 3, device="cuda", dtype=torch.bfloat16)
    states = torch.randn(12, V_HEADS, HEAD_DIM, HEAD_DIM, device="cuda") * 0.1
    initial_conv, initial_states = conv.clone(), states.clone()
    raw = torch.randn(bucket, channels, device="cuda", dtype=torch.bfloat16)
    backend.layer_state = lambda layer_id: (
        backend.forward_metadata.state_in_blocks_by_group["state"],
        backend.forward_metadata.state_out_blocks_by_group["state"],
        conv,
        states,
    )
    kwargs = dict(
        conv_weights=torch.randn(channels, 4, device="cuda", dtype=torch.bfloat16)
        * 0.1,
        bias=None,
        activation="silu",
        key_dim=key_dim,
        value_dim=value_dim,
        attention_tp_size=1,
        head_k_dim=HEAD_DIM,
        head_v_dim=HEAD_DIM,
        a=torch.randn(bucket, V_HEADS, device="cuda", dtype=torch.bfloat16),
        b=torch.randn(bucket, V_HEADS, device="cuda", dtype=torch.bfloat16),
        A_log=torch.log(torch.arange(1, V_HEADS + 1, device="cuda").float()),
        dt_bias=torch.rand(V_HEADS, device="cuda") - 4,
        layer_id=0,
        seq_len=bucket,
    )

    def forward():
        out = backend.forward_extend(
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
        # The scan output is [1, tokens, heads, head_dim]; compare rows.
        return out.reshape(-1, V_HEADS, HEAD_DIM)

    def reset():
        conv.copy_(initial_conv)
        states.copy_(initial_states)

    def live(lengths, prefixes, page_shift):
        return _source(lengths, prefixes, page_shift)

    # Prefix hits end at arbitrary tokens; a resumed row reads block 1.
    cases = []
    for resumed in range(batch_size + 1):
        for tail_lengths, prefix in [([837, 325, 197, 197], 0), ([70] * 4, 127)]:
            lengths = tail_lengths[:resumed] + [128] * (batch_size - resumed)
            prefixes = [prefix] * resumed + [0] * (batch_size - resumed)
            cases.extend([(lengths, prefixes), (lengths[::-1], prefixes[::-1])])
    if batch_size == 1:
        cases.extend([([837], [50303]), ([769], [0]), ([1023], [129]), ([2047], [0])])
    if batch_size == 4:
        for actual_bs in (3, 1, 2, 3):
            cases.append(([64 + i for i in range(actual_bs)], [50303] * actual_bs))
            cases.append(([837] + [128] * (actual_bs - 1), [50303] * actual_bs))
    cases.append(([1] * batch_size, [0] * batch_size))

    def dummy(bucket, bs):
        backend.forward_metadata = live([bucket // bs] * bs, [0] * bs, 2)

    owner = _capture(backend, bucket, batch_size, forward, dummy)
    capture, captured = owner._captures[bucket, batch_size]
    output = captured.hidden_states
    assert capture.num_segments == 1
    fixed = backend._prefill_metadata[bucket, batch_size]
    retained = {
        "query_start_loc": fixed.query_start_loc,
        "scan_query_start_loc": fixed.scan_query_start_loc,
        "query_start_loc_int64": fixed.query_start_loc_int64,
        "state_in": fixed.state_in_blocks_by_group["state"],
        "state_out": fixed.state_out_blocks_by_group["state"],
    }
    addresses = {name: tensor.data_ptr() for name, tensor in retained.items()}
    for index, (lengths, prefixes) in enumerate(cases * 2):
        tokens = sum(lengths)
        source = live(lengths, prefixes, 2 + index % 2)
        backend.forward_metadata = source
        reset()
        plain = forward()[:tokens].clone()
        plain_conv, plain_states = conv.clone(), states.clone()
        backend.forward_metadata = replace(source)
        assert backend.prepare_prefill_metadata(
            bucket, batch_size, ForwardMode.EXTEND, capture=False
        )
        assert backend.prefill_metadata_is_capture_ready
        reset()
        eager = forward()[:tokens].clone()
        eager_conv, eager_states = conv.clone(), states.clone()
        torch.testing.assert_close(eager, plain, rtol=0, atol=0)
        torch.testing.assert_close(eager_conv, plain_conv, rtol=0, atol=0)
        torch.testing.assert_close(eager_states, plain_states, rtol=0, atol=0)
        reset()
        capture.replay(valid_rows=tokens)
        assert addresses == {
            name: tensor.data_ptr() for name, tensor in retained.items()
        }
        torch.testing.assert_close(output[:tokens], eager, rtol=0, atol=0)
        torch.testing.assert_close(conv, eager_conv, rtol=0, atol=0)
        torch.testing.assert_close(states, eager_states, rtol=0, atol=0)
        assert torch.count_nonzero(output[tokens:]) == 0


class _StatePool:
    """GDN layers' conv and recurrent slabs behind a one-group cache contract."""

    def __init__(self, slabs, prefix_granularity=PAGE):
        self.arena = SimpleNamespace(
            runtime_contract=SimpleNamespace(
                prefix_granularity=prefix_granularity,
                group_specs=(SimpleNamespace(group_id="state", attn_kind="state"),),
            )
        )
        self.state_group_by_layer = {layer_id: "state" for layer_id in slabs}
        self._slabs = slabs

    def get_component(self, layer_id, name):
        conv, recurrent = self._slabs[layer_id]
        return conv if name == "conv_state" else recurrent


def test_checkpoint_interval_follows_the_bound_pool():
    """The scheduler places in-forward checkpoints at multiples of the bound
    pool's prefix granularity, so a captured scan stores a state at each of
    those pages; without the option the capacity layout stores none."""
    conv = torch.zeros(
        4,
        2 * K_HEADS * HEAD_DIM + V_HEADS * HEAD_DIM,
        3,
        device="cuda",
        dtype=torch.bfloat16,
    )
    states = torch.zeros(4, V_HEADS, HEAD_DIM, HEAD_DIM, device="cuda")
    backend = _backend(writes_state_checkpoints_in_forward=True)
    backend.set_kv_pool(_StatePool({0: (conv, states)}, prefix_granularity=2 * PAGE))
    if not backend.stores_in_forward_state_checkpoints:
        pytest.skip("no GDN prefill kernel stores state checkpoints here")
    assert backend._capacity_checkpoint_interval() == 2 * PAGE
    backend.set_kv_pool(_StatePool({0: (conv, states)}, prefix_granularity=PAGE))
    assert backend._capacity_checkpoint_interval() == PAGE
    plain = _backend(writes_state_checkpoints_in_forward=False)
    plain.set_kv_pool(_StatePool({0: (conv, states)}, prefix_granularity=PAGE))
    assert plain._capacity_checkpoint_interval() is None


def _scheduler_metadata(backend, rows):
    """Build one forward's metadata the way the scheduler hands it over.

    ``rows`` holds ``(new tokens, cached prefix, checkpoint offset, live
    block, checkpoint block)`` per request; a resumed row reads block 1, and a
    nonzero offset gives the row its third State-table entry. ``backend`` is
    the GDN leaf or a composite above it.
    """
    lengths = torch.tensor([row[0] for row in rows], dtype=torch.int32)
    prefixes = torch.tensor([row[1] for row in rows], dtype=torch.int32)
    offsets = torch.tensor([row[2] for row in rows], dtype=torch.int32)
    table = [[1 if row[1] else 0, row[3]] for row in rows]
    if bool(offsets.any()):
        table = [entry + [row[4] if row[2] else -1] for entry, row in zip(table, rows)]
    backend.init_forward_metadata(
        bs=len(rows),
        num_extends=len(rows),
        req_pool_indices=torch.arange(len(rows), dtype=torch.int32, device="cuda"),
        seq_lens=(prefixes + lengths).cuda(),
        forward_mode=ForwardMode.EXTEND,
        block_tables={"state": torch.tensor(table, dtype=torch.int32, device="cuda")},
        extend_seq_lens=lengths.cuda(),
        extend_seq_lens_cpu=lengths,
        extend_prefix_lens=prefixes.cuda(),
        extend_prefix_lens_cpu=prefixes,
        extend_replay_lens_cpu=torch.zeros_like(prefixes),
        extend_prompt_lens_cpu=prefixes + lengths,
        extend_with_prefix=bool(prefixes.any()),
        query_shard=None,
        extend_state_checkpoint_lens_cpu=offsets,
    )


@pytest.mark.parametrize("state_dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("batch_size", [1, 2, 4])
def test_outer_graph_stores_in_forward_checkpoints_like_eager(batch_size, state_dtype):
    """A capture taken without checkpoints stores the live ones at replay.

    With in-forward State checkpoints the retained shape keeps one fixed slot
    per request slot and its scan stores a state at every page. Replay must
    leave every slab block -- live, checkpoint, and none other -- exactly as
    the eager forward on the scheduler's own metadata leaves it, whichever
    rows carry a checkpoint (none, some, all), with prefix hits and padded
    request slots.
    """
    backend = _backend(writes_state_checkpoints_in_forward=True)
    bucket = 2048
    if not backend.admits_prefill_graph(bucket, batch_size, ForwardMode.EXTEND):
        pytest.skip("the selected GDN scan cannot be captured with checkpoints here")
    torch.manual_seed(7)
    key_dim, value_dim = K_HEADS * HEAD_DIM, V_HEADS * HEAD_DIM
    channels = 2 * key_dim + value_dim
    conv = torch.randn(24, channels, 3, device="cuda", dtype=torch.bfloat16)
    states = (torch.randn(24, V_HEADS, HEAD_DIM, HEAD_DIM, device="cuda") * 0.1).to(
        state_dtype
    )
    initial_conv, initial_states = conv.clone(), states.clone()
    backend.set_kv_pool(_StatePool({0: (conv, states)}))
    backend.init_prefill_graph_state(bucket, batch_size)
    raw = torch.randn(bucket, channels, device="cuda", dtype=torch.bfloat16)
    kwargs = dict(
        conv_weights=torch.randn(channels, 4, device="cuda", dtype=torch.bfloat16)
        * 0.1,
        bias=None,
        activation="silu",
        key_dim=key_dim,
        value_dim=value_dim,
        attention_tp_size=1,
        head_k_dim=HEAD_DIM,
        head_v_dim=HEAD_DIM,
        a=torch.randn(bucket, V_HEADS, device="cuda", dtype=torch.bfloat16),
        b=torch.randn(bucket, V_HEADS, device="cuda", dtype=torch.bfloat16),
        A_log=torch.log(torch.arange(1, V_HEADS + 1, device="cuda").float()),
        dt_bias=torch.rand(V_HEADS, device="cuda") - 4,
        layer_id=0,
        seq_len=bucket,
    )

    def forward():
        out = backend.forward_extend(
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
        return out.reshape(-1, V_HEADS, HEAD_DIM)

    def reset():
        conv.copy_(initial_conv)
        states.copy_(initial_states)

    def dummy(bucket, bs):
        # Like make_dummy_batch: balanced fresh requests on spare blocks.
        lengths = [bucket // bs + (row < bucket % bs) for row in range(bs)]
        _scheduler_metadata(
            backend, [(length, 0, 0, 18 + row, 0) for row, length in enumerate(lengths)]
        )

    owner = _capture(backend, bucket, batch_size, forward, dummy)
    capture, captured = owner._captures[bucket, batch_size]
    output = captured.hidden_states
    assert capture.num_segments == 1
    fixed = backend._prefill_metadata[bucket, batch_size]
    retained = {
        "token_index": fixed.state_checkpoints.token_index,
        "blocks": fixed.state_checkpoints.blocks_by_group["state"],
        "cu_starts": fixed.scan_state_checkpoints.cu_starts,
        "scratch": fixed.scan_state_checkpoints.states,
        "rows": fixed.scan_checkpoint_rows,
    }
    assert fixed.scan_state_checkpoints.every_n_tokens == PAGE
    addresses = {name: tensor.data_ptr() for name, tensor in retained.items()}

    # (new tokens, cached prefix, checkpoint offset): offsets are page
    # multiples inside the chunk, 0 for none.
    shapes = [
        [(837, 4096, 768), (325, 0, 256), (197, 4096, 128), (197, 0, 0)],
        [(640, 0, 0), (325, 4096, 256), (129, 0, 128), (70, 4096, 0)],
        [(129, 0, 128), (129, 4096, 128), (129, 0, 0), (129, 0, 128)],
        [(64, 0, 0), (837, 4096, 0), (65, 0, 0), (1, 0, 0)],
    ]
    cases = []
    for shape in shapes:
        for actual in range(1, batch_size + 1):
            cases.extend([shape[:actual], shape[:actual][::-1]])
    if batch_size == 1:
        cases.extend(
            [
                [(2047, 0, 1920)],
                [(1024, 4096, 896)],
                [(300, 0, 256)],
                [(1, 4096, 0)],
                [(2047, 4096, 128)],
            ]
        )
    for index, case in enumerate(cases * 2):
        shift = 4 * (index % 2)
        rows = [
            (tokens, prefix, offset, 2 + row + shift, 10 + row + shift)
            for row, (tokens, prefix, offset) in enumerate(case)
        ]
        tokens = sum(row[0] for row in rows)
        _scheduler_metadata(backend, rows)
        source = backend.forward_metadata
        assert (source.state_checkpoints is None) == (not any(r[2] for r in rows))
        reset()
        plain = forward()[:tokens].clone()
        plain_conv, plain_states = conv.clone(), states.clone()
        backend.forward_metadata = replace(source)
        assert backend.prepare_prefill_metadata(
            bucket, batch_size, ForwardMode.EXTEND, capture=False
        )
        assert backend.prefill_metadata_is_capture_ready
        reset()
        eager = forward()[:tokens].clone()
        assert torch.equal(eager, plain)
        assert torch.equal(conv, plain_conv)
        assert torch.equal(states, plain_states)
        reset()
        capture.replay(valid_rows=tokens)
        assert addresses == {
            name: tensor.data_ptr() for name, tensor in retained.items()
        }
        assert torch.equal(output[:tokens], plain)
        assert torch.equal(conv, plain_conv)
        assert torch.equal(states, plain_states)
        assert torch.count_nonzero(output[tokens:]) == 0


def test_qwen4_exp_composite_captures_gdn_inline_around_full_attention():
    """The Qwen4-Exp composite hands the prefill graph seams to its GDN child.

    A shape the composite prepares at capture runs both GDN layers inside the
    graph segments while the full-attention layer keeps its break. Replaying
    it with later forwards' scheduler metadata (an in-forward State
    checkpoint, a prefix hit) leaves the outputs and every state block as the
    eager forward on that metadata does. Recapture drops the retained shape,
    and mixed batches and layerwise PD transfer keep the ordinary route.
    """
    from tokenspeed.runtime.execution.breakable_cuda_graph import (
        BreakableCapture,
        active_forward,
    )
    from tokenspeed.runtime.layers.attention.backends.base import AttentionBackend
    from tokenspeed.runtime.layers.attention.backends.hybrid.linear import (
        HybridLinearAttnBackend,
    )
    from tokenspeed.runtime.layers.attention.backends.specific.qwen4_exp import (
        Qwen4ExpBackend,
    )
    from tokenspeed.runtime.layers.attention.backends.state.gdn import (
        GdnAttnBackend,
    )
    from tokenspeed.runtime.pd.utils import StepCounter

    class FullAttention(AttentionBackend):
        """Stands in for the paged full-attention router, which needs a KV arena."""

        def init_forward_metadata(self, *args, **kwargs) -> None:
            pass

        def forward_extend(self, q, k, v, layer, token_to_kv_pool, bs, **kwargs):
            return q * 2 + 1

    config, spec = _config((0, 2))
    gdn = GdnAttnBackend(config, spec, writes_state_checkpoints_in_forward=True)
    root = Qwen4ExpBackend(
        config,
        HybridLinearAttnBackend(FullAttention(config, spec), gdn, [1]),
        None,
        None,
    )
    bucket = 1024
    if not root.admits_prefill_graph(bucket, 1, ForwardMode.EXTEND):
        pytest.skip("the selected GDN scan cannot be captured with checkpoints here")
    assert not root.admits_prefill_graph(bucket, 1, ForwardMode.MIXED)
    torch.manual_seed(11)
    key_dim, value_dim = K_HEADS * HEAD_DIM, V_HEADS * HEAD_DIM
    channels = 2 * key_dim + value_dim
    slabs = {
        layer: (
            torch.randn(16, channels, 3, device="cuda", dtype=torch.bfloat16),
            torch.randn(16, V_HEADS, HEAD_DIM, HEAD_DIM, device="cuda") * 0.1,
        )
        for layer in (0, 2)
    }
    initial = {layer: tuple(t.clone() for t in slab) for layer, slab in slabs.items()}
    root.set_cache_pool(_StatePool(slabs))
    root.init_prefill_graph_state(bucket, 1)
    inputs = {
        layer: dict(
            mixed_qkv=torch.randn(
                bucket, channels, device="cuda", dtype=torch.bfloat16
            ),
            conv_weights=torch.randn(channels, 4, device="cuda", dtype=torch.bfloat16)
            * 0.1,
            bias=None,
            activation="silu",
            key_dim=key_dim,
            value_dim=value_dim,
            attention_tp_size=1,
            head_k_dim=HEAD_DIM,
            head_v_dim=HEAD_DIM,
            a=torch.randn(bucket, V_HEADS, device="cuda", dtype=torch.bfloat16),
            b=torch.randn(bucket, V_HEADS, device="cuda", dtype=torch.bfloat16),
            A_log=torch.log(torch.arange(1, V_HEADS + 1, device="cuda").float()),
            dt_bias=torch.rand(V_HEADS, device="cuda") - 4,
            layer_id=layer,
            seq_len=bucket,
        )
        for layer in (0, 2)
    }
    ctx = SimpleNamespace(forward_mode=ForwardMode.EXTEND, bs=1, num_extends=1)

    def gdn_layer(layer):
        kwargs = dict(inputs[layer], mixed_qkv=inputs[layer]["mixed_qkv"] * 1)
        return root.forward(
            None, None, None, None, None, ForwardMode.EXTEND, 1, **kwargs
        )

    def forward():
        first = gdn_layer(0)
        attended = root.forward(
            first.flatten(1) * 1,
            None,
            None,
            SimpleNamespace(layer_id=1),
            None,
            ForwardMode.EXTEND,
            1,
        )
        return attended + gdn_layer(2).flatten(1)

    def reset():
        for layer, (conv, states) in slabs.items():
            conv.copy_(initial[layer][0])
            states.copy_(initial[layer][1])

    def capture():
        forward()
        torch.cuda.synchronize()
        graph = BreakableCapture()
        with active_forward(ctx), graph:
            output = forward()
        return graph, output

    # A fresh request on spare blocks, like make_dummy_batch.
    _scheduler_metadata(root, [(bucket, 0, 0, 14, 0)])
    assert root.prepare_prefill_metadata(bucket, 1, ForwardMode.EXTEND, capture=False)
    assert not gdn.prefill_metadata_is_capture_ready
    ordinary, _ = capture()
    assert ordinary.num_segments == 7  # Every attention layer is a break.
    _scheduler_metadata(root, [(bucket, 0, 0, 14, 0)])
    assert root.prepare_prefill_metadata(bucket, 1, ForwardMode.EXTEND, capture=True)
    assert gdn.prefill_metadata_is_capture_ready
    merged, output = capture()
    assert merged.num_segments == 3  # Only full attention remains a break.

    for index, case in enumerate(
        [(900, 4096, 768), (1000, 0, 0), (129, 0, 128), (1023, 4096, 896)] * 2
    ):
        tokens = case[0]
        _scheduler_metadata(root, [(*case, 2 + index % 2, 4 + index % 2)])
        source = gdn.forward_metadata
        reset()
        plain = forward()[:tokens].clone()
        plain_slabs = {layer: tuple(t.clone() for t in s) for layer, s in slabs.items()}
        gdn.forward_metadata = replace(source)
        assert root.prepare_prefill_metadata(
            bucket, 1, ForwardMode.EXTEND, capture=False
        )
        reset()
        with active_forward(ctx):
            merged.replay(valid_rows=tokens)
        assert torch.equal(output[:tokens], plain)
        for layer, (conv, states) in slabs.items():
            assert torch.equal(conv, plain_slabs[layer][0])
            assert torch.equal(states, plain_slabs[layer][1])

    root.init_prefill_graph_state(bucket, 1)
    assert not gdn.prefill_metadata_is_capture_ready
    assert not gdn._prefill_metadata
    counter = StepCounter("cuda", torch.cuda.current_device())
    root.register_step_counter(counter)
    assert root.step_counter is counter
    assert not root.admits_prefill_graph(bucket, 1, ForwardMode.EXTEND)
    assert not root.prepare_prefill_metadata(
        bucket, 1, ForwardMode.EXTEND, capture=False
    )
