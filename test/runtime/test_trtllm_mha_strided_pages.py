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

"""TRTLLM reads page-major K/V fields without flattening their page gaps."""

import dataclasses
import json
import os
import sys

import pytest
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from ci_system.ci_register import register_cuda_ci

register_cuda_ci(est_time=20, suite="runtime-1gpu")

from tokenspeed.runtime.configs.model_config import ModelConfig
from tokenspeed.runtime.layers.attention.backends.paged.trtllm import (
    TRTLLMMHAAttnBackend,
    TRTLLMMHAMetadata,
    canonicalize_stride,
)
from tokenspeed.runtime.layers.attention.configs.base import AttnConfig
from tokenspeed.runtime.layers.attention.configs.mha import MHAConfig
from tokenspeed.runtime.layers.attention.kv_cache.arena import CacheArena
from tokenspeed.runtime.layers.attention.kv_cache.mha import MHATokenToKVPool
from tokenspeed.runtime.layers.attention.kv_cache.recipes.plan import (
    CacheFieldSpec,
    pack,
)
from tokenspeed.runtime.layers.attention.kv_cache.recipes.spec import CacheGroupSpec
from tokenspeed.runtime.layers.paged_attention import PagedAttention
from tokenspeed.runtime.utils.server_args import ServerArgs


@pytest.mark.parametrize(
    ("algorithm", "is_draft", "backend", "expected"),
    [
        ("DFLASH", True, "trtllm", 64),
        ("DSPARK", True, "trtllm", 64),
        ("DFLASH", False, "trtllm", 64),
        ("MTP", True, "trtllm", 64),
        (None, False, "trtllm", 64),
        ("DFLASH", True, "triton", None),
    ],
)
def test_general_mha_page_defaults(tmp_path, algorithm, is_draft, backend, expected):
    (tmp_path / "config.json").write_text(
        json.dumps(
            {
                "model_type": "qwen3",
                "architectures": ["Qwen3ForCausalLM"],
                "torch_dtype": "bfloat16",
                "vocab_size": 128,
                "hidden_size": 64,
                "intermediate_size": 128,
                "num_hidden_layers": 2,
                "num_attention_heads": 4,
                "num_key_value_heads": 2,
                "head_dim": 16,
                "max_position_embeddings": 256,
            }
        )
    )
    args = ServerArgs(
        model=str(tmp_path),
        device="cpu",
        dtype="bfloat16",
        attention_backend="trtllm",
        drafter_attention_backend=backend,
        speculative_algorithm=algorithm,
        speculative_num_steps=7,
        speculative_num_draft_tokens=8,
        max_num_seqs=2,
    )
    args.mapping.rank = 0
    model = ModelConfig(
        model_path=args.model,
        trust_remote_code=False,
        model_override_args="{}",
        server_args=args,
        dtype=args.dtype,
        context_length=256,
    )
    config = MHAConfig.generate(args, model, is_draft=is_draft)
    if backend == "trtllm":
        assert TRTLLMMHAAttnBackend.resolve_kernel_page_size(config, 128) == expected
        # A supplied config override always wins over both generated and backend defaults.
        for page_size in (64, 128):
            explicit = dataclasses.replace(config, kernel_page_size=page_size)
            assert (
                TRTLLMMHAAttnBackend.resolve_kernel_page_size(explicit, 128)
                == page_size
            )
    else:
        assert config.kernel_page_size is None


def _pool_and_backend(*, heads, dim, device, kernel_page_size, block_draft):
    specs = (
        CacheGroupSpec("target", "full_history", 128, 1, replayable=False),
        CacheGroupSpec("draft", "full_history", 128, 1, replayable=False),
    )
    layout = pack(
        (
            (
                specs[0],
                (
                    CacheFieldSpec(
                        "layer.0.latent_kv",
                        "slot.0",
                        (128, 1, 2 * heads * dim + 16),
                        "bfloat16",
                    ),
                ),
            ),
            (
                specs[1],
                tuple(
                    CacheFieldSpec(
                        f"layer.1.{field}",
                        "slot.0",
                        (128, heads, dim),
                        "bfloat16",
                        exact_page_stride=False,
                    )
                    for field in ("k", "v")
                ),
            ),
        ),
        prefix_granularity=128,
        cache_blocks_per_lcm_block={"target": 5, "draft": 2},
        max_padding_fraction=float("inf"),
    )
    arena = CacheArena(layout.bind(2), device, cache_group_specs=specs)
    pool = MHATokenToKVPool(
        arena, torch.bfloat16, heads, dim, 1, 0, field_layer_offset=1
    )
    component = MHAConfig(
        backend_name="trtllm",
        num_attention_heads=4 * heads,
        num_kv_heads=heads,
        head_dim=dim,
        attn_tp_size=1,
    )
    config = AttnConfig(
        device=device,
        dtype=torch.bfloat16,
        kv_cache_dtype=torch.bfloat16,
        kv_cache_quant_method="none",
        context_len=512,
        max_bs=1,
        components=(component,),
        is_draft=block_draft,
        draft_block_decode=block_draft,
        speculative_num_draft_tokens=8 if block_draft else 1,
    )
    backend = TRTLLMMHAAttnBackend(config, component, kernel_page_size=kernel_page_size)
    layer = PagedAttention(
        num_heads=4 * heads,
        num_kv_heads=heads,
        head_dim=dim,
        scaling=dim**-0.5,
        layer_id=0,
        sliding_window_size=-1,
        rotary_emb=None,
        qk_norm=None,
    )
    return pool, backend, layer


@pytest.mark.parametrize("heads", [1, 2])
def test_paged_views_preserve_field_offsets_and_page_gaps(heads):
    pool, backend, layer = _pool_and_backend(
        heads=heads, dim=16, device="cpu", kernel_page_size=128, block_draft=False
    )
    key, value = pool.get_kv_buffer(0)
    assert key.ndim == value.ndim == 4
    assert key.stride(0) > 2 * 128 * heads * 16
    assert key.untyped_storage().data_ptr() == value.untyped_storage().data_ptr()
    key.copy_(
        torch.arange(key.numel(), dtype=torch.float32).reshape(key.shape).bfloat16()
    )
    value.copy_(-key)
    read_key, read_value = backend._get_kv_cache_permuted(layer, pool)
    for raw, read in ((key, read_key), (value, read_value)):
        assert read.data_ptr() == raw.data_ptr()
        assert read.stride(0) == raw.stride(0)
        torch.testing.assert_close(read, raw.permute(0, 2, 1, 3))


def test_mismatched_padded_pages_raise_a_clear_error():
    pool, backend, layer = _pool_and_backend(
        heads=2, dim=16, device="cpu", kernel_page_size=64, block_draft=False
    )
    with pytest.raises(ValueError, match="page size.*128.*64"):
        backend._get_kv_cache_permuted(layer, pool)


def test_singleton_stride_normalization_preserves_physical_page_stride():
    backing = torch.arange(4096)
    tensor = backing.as_strided((3, 1, 128, 8), (1200, 8, 8, 1), 16)
    result = canonicalize_stride(tensor)
    assert result.stride() == (1200, 1024, 8, 1)
    assert result.data_ptr() == tensor.data_ptr()
    torch.testing.assert_close(result, tensor)


def test_single_page_single_head_stride_normalization():
    backing = torch.arange(1024)
    tensor = backing.as_strided((1, 1, 128, 8), (8, 8, 8, 1))
    result = canonicalize_stride(tensor)
    assert result.stride() == (1024, 1024, 8, 1)
    assert result.data_ptr() == tensor.data_ptr()
    torch.testing.assert_close(result, tensor)


@pytest.mark.parametrize("rank", [3, 4])
def test_contiguous_cache_can_be_viewed_at_a_smaller_page_size(rank):
    pool, backend, layer = _pool_and_backend(
        heads=2, dim=16, device="cpu", kernel_page_size=64, block_draft=False
    )
    # Exercise both supported accessor shapes through the real pool.
    key, value = (cache.contiguous() for cache in pool.get_kv_buffer(0))
    if rank == 3:
        key, value = (cache.view(-1, 2, 16) for cache in (key, value))
    pool.k_buffer[0], pool.v_buffer[0] = key, value
    read_key, read_value = backend._get_kv_cache_permuted(layer, pool)
    for raw, read in ((key, read_key), (value, read_value)):
        assert read.data_ptr() == raw.data_ptr()
        torch.testing.assert_close(read, raw.view(-1, 64, 2, 16).permute(0, 2, 1, 3))


@pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.get_device_capability()[0] != 10,
    reason="requires supported Blackwell SM10x",
)
@pytest.mark.parametrize("heads", [1, 8])
@pytest.mark.parametrize("mode", ["decode", "prefill", "block_draft"])
@pytest.mark.parametrize("graph", [False, True])
def test_trtllm_padded_pages_match_attention_reference(heads, mode, graph):
    prefill = mode == "prefill"
    block_draft = mode == "block_draft"
    pool, backend, layer = _pool_and_backend(
        heads=heads,
        dim=128,
        device="cuda",
        kernel_page_size=128,
        block_draft=block_draft,
    )
    key, value = pool.get_kv_buffer(0)
    generator = torch.Generator(device="cuda").manual_seed(193)
    key.copy_(
        torch.randn(key.shape, dtype=key.dtype, device="cuda", generator=generator)
    )
    value.copy_(
        torch.randn(value.shape, dtype=value.dtype, device="cuda", generator=generator)
    )
    rows = 8 if block_draft else 32 if prefill else 1
    query = torch.randn(
        rows, 4 * heads, 128, dtype=key.dtype, device="cuda", generator=generator
    )
    table = torch.tensor([[1, 3, 0, 0]], dtype=torch.int32, device="cuda")
    lengths = torch.tensor([193], dtype=torch.int32, device="cuda")
    metadata = TRTLLMMHAMetadata(
        cache_seqlens_int32=lengths,
        max_seq_len_q=rows,
        cu_seqlens_q=torch.tensor([0, rows], dtype=torch.int32, device="cuda"),
        cu_seqlens_k=torch.tensor([0, 193], dtype=torch.int32, device="cuda"),
        page_table=table,
    )
    if block_draft:
        backend.init_cuda_graph_state(1)
    if prefill:
        backend.forward_prefill_metadata = metadata
        forward = backend.forward_extend
    else:
        backend.forward_decode_metadata = metadata
        forward = backend.forward_decode

    def run():
        if block_draft:
            backend.refresh_decode_metadata(
                1, 1, lengths, table, num_extends=0, for_graph_replay=False
            )
        return forward(query, None, None, layer, None, pool, 1)

    if graph:
        for _ in range(2):
            run()
        torch.cuda.synchronize()
        capture = torch.cuda.CUDAGraph()
        with torch.cuda.graph(capture):
            output = run()
        # Replay reads changed page tables, lengths, queries and cache contents.
        table.copy_(torch.tensor([[2, 4, 0, 0]], dtype=torch.int32, device="cuda"))
        lengths.fill_(225)
        metadata.cu_seqlens_k.copy_(
            torch.tensor([0, 225], dtype=torch.int32, device="cuda")
        )
        query.add_(0.25)
        key.add_(0.5)
        value.mul_(0.5)
        capture.replay()
    else:
        output = run()
    torch.cuda.synchronize()
    length = int(lengths[0])
    index = table[0].long()
    k = key[index].reshape(-1, heads, 128)[:length].float().repeat_interleave(4, dim=1)
    v = (
        value[index]
        .reshape(-1, heads, 128)[:length]
        .float()
        .repeat_interleave(4, dim=1)
    )
    scores = torch.einsum("qhd,khd->hqk", query.float(), k) * layer.scaling
    positions = torch.arange(length - rows, length, device="cuda")
    if not block_draft:
        scores.masked_fill_(
            torch.arange(length, device="cuda")[None, None, :]
            > positions[None, :, None],
            float("-inf"),
        )
    reference = torch.einsum("hqk,khd->qhd", scores.softmax(-1), v)
    torch.testing.assert_close(
        output.float().view_as(reference), reference, atol=0.01, rtol=0.01
    )
