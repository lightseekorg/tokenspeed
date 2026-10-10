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
from dataclasses import fields, replace
from types import SimpleNamespace

import pytest
import torch

sys.path.insert(
    0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
)
from ci_system.ci_register import register_cuda_ci  # noqa: E402

register_cuda_ci(est_time=60, suite="runtime-1gpu")

from tokenspeed_kernel.ops.attention.gdn import (  # noqa: E402
    gdn_chunk_prefill_capturable,
)

from tokenspeed.runtime.execution.forward_batch_info import ForwardMode  # noqa: E402
from tokenspeed.runtime.execution.memory_delta import (  # noqa: E402
    NULL_MEMORY_DELTA_OBSERVER,
)
from tokenspeed.runtime.layers.attention.backends.state.mamba import (  # noqa: E402
    MambaForwardMetadata,
    _build_prefill_checkpoint_batch,
)
from tokenspeed.runtime.layers.attention.configs.linear_attn import (  # noqa: E402
    LinearAttnConfig,
)

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")

K_HEADS, V_HEADS, HEAD_DIM = 2, 4, 128


def _capturable():
    return gdn_chunk_prefill_capturable(
        torch.bfloat16,
        head_dim=HEAD_DIM,
        value_head_dim=HEAD_DIM,
        num_q_heads=K_HEADS,
        num_v_heads=V_HEADS,
        qk_l2norm=True,
    )


def _backend():
    from tokenspeed.runtime.layers.attention.backends.state.gdn import (
        GdnAttnBackend,
    )
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
        layer_ids=(0,),
        tp_size=1,
    )
    config = AttnConfig(
        device="cuda",
        dtype=torch.bfloat16,
        kv_cache_dtype=torch.bfloat16,
        kv_cache_quant_method="none",
        prefix_granularity=128,
        context_len=4096,
        max_bs=8,
        is_draft=False,
        speculative_num_draft_tokens=1,
        components=(spec, gdn),
    )
    backend = GdnAttnBackend(config, spec)
    backend.cache_pool = object()
    backend._prefix_granularity = 128
    return backend


def _source(lengths, prefixes, page_shift, *, checkpoint_states):
    lengths = torch.tensor(lengths)
    host = torch.cat((torch.zeros(1, dtype=torch.int64), lengths.cumsum(0)))
    bounds = host.to(device="cuda", dtype=torch.int32)
    checkpoint = _build_prefill_checkpoint_batch(
        lengths, torch.tensor(prefixes), lengths.numel(), 128, "cuda"
    )
    pages = torch.arange(lengths.numel(), device="cuda", dtype=torch.int32)
    pages = pages + page_shift
    from tokenspeed_kernel.ops.attention.gdn.triton import (
        CAUSAL_CONV1D_BLOCK_M,
        build_causal_conv1d_prefill_metadata,
    )

    return MambaForwardMetadata(
        query_start_loc=bounds,
        scan_query_start_loc=bounds,
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
            {"state": pages + checkpoint_states} if checkpoint is not None else None
        ),
        prefill_checkpoint_batch=checkpoint,
        conv_prefill_metadata=build_causal_conv1d_prefill_metadata(
            bounds, lengths, CAUSAL_CONV1D_BLOCK_M
        ),
    )


def test_admission_follows_the_selected_scan():
    """Only a scan that reads its bounds on device is captured; others stay eager."""
    backend = _backend()
    assert backend.admits_prefill_graph(512, 1, ForwardMode.EXTEND) == _capturable()
    assert not backend.admits_prefill_graph(512, 1, ForwardMode.MIXED)


def test_uncaptured_shapes_keep_the_scheduler_metadata():
    """Eager forwards of shapes without a graph skip the capacity layout."""
    backend = _backend()
    if not backend.admits_prefill_graph(512, 1, ForwardMode.EXTEND):
        pytest.skip("the selected GDN scan is not graph-capturable here")
    source = _source([300], [0], 2, checkpoint_states=4)
    backend.forward_metadata = source
    assert backend.prepare_prefill_metadata(512, 1, ForwardMode.EXTEND, capture=False)
    assert backend.forward_metadata is source
    assert not backend.prefill_metadata_is_capture_ready


@pytest.mark.parametrize("batch_size", [1, 2, 4])
def test_outer_graph_replays_gdn_lengths_pages_and_states(batch_size):
    """Replay equals eager on the retained metadata, which equals plain eager."""
    from tokenspeed.runtime.execution.forward_batch_info import CaptureHiddenMode
    from tokenspeed.runtime.execution.prefill_graph import PrefillGraph

    backend = _backend()
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
        return _source(lengths, prefixes, page_shift, checkpoint_states=4)

    cases = []
    for checkpoint_count in range(batch_size + 1):
        for tail_lengths, prefix in [([837, 325, 197, 197], 0), ([70] * 4, 127)]:
            lengths = tail_lengths[:checkpoint_count] + [128] * (
                batch_size - checkpoint_count
            )
            prefixes = [prefix] * checkpoint_count + [0] * (
                batch_size - checkpoint_count
            )
            cases.extend([(lengths, prefixes), (lengths[::-1], prefixes[::-1])])
    if batch_size == 1:
        cases.extend([([837], [50304]), ([769], [0]), ([1023], [128]), ([2047], [0])])
    if batch_size == 4:
        for actual_bs in (3, 1, 2, 3):
            cases.append(([64 + i for i in range(actual_bs)], [50304] * actual_bs))
            cases.append(([837] + [128] * (actual_bs - 1), [50304] * actual_bs))
    cases.append(([1] * batch_size, [0] * batch_size))

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
    owner._land_input_embeds = lambda *args: None
    owner.attn_backend = backend
    owner._run_inner = lambda bucket: (forward(), None)

    def dummy(bucket, bs):
        backend.forward_metadata = live([bucket // bs] * bs, [0] * bs, 2)
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
            name: getattr(fixed, name).data_ptr() for name in addresses
        }
        torch.testing.assert_close(output[:tokens], eager, rtol=0, atol=0)
        torch.testing.assert_close(conv, eager_conv, rtol=0, atol=0)
        torch.testing.assert_close(states, eager_states, rtol=0, atol=0)
        assert torch.count_nonzero(output[tokens:]) == 0
