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

"""Mixed DSA TopK uses decode-sized workspaces and explicit batch row offsets."""

from dataclasses import replace
from test.ci_system.ci_register import register_cuda_ci
from test.runtime.test_dsa_hopper_fp8 import _generate, _pool

import pytest
import torch

from tokenspeed.runtime.execution.context import ForwardContext
from tokenspeed.runtime.execution.forward_batch_info import ForwardMode
from tokenspeed.runtime.execution.output_layout import ForwardOutputLayout
from tokenspeed.runtime.layers.attention.backends.paged.dsa import DSABackend
from tokenspeed.runtime.layers.attention.configs.dsa import DSAConfig
from tokenspeed.runtime.layers.attention.dsa.utils import (
    _prepare_dsa_topk_for_mtp_decode,
)
from tokenspeed.runtime.layers.attention.kv_cache.dsa import DSATokenToKVPool
from tokenspeed.runtime.models.glm5 import (
    GlmDsaDecodeTopK,
    GlmDsaIndexerOutput,
    GlmDsaPrefillTopK,
    GlmMoeDsaAttention,
)
from tokenspeed.runtime.utils.env import global_server_args_dict

register_cuda_ci(est_time=15, suite="runtime-1gpu")


@pytest.mark.parametrize("row_offset", [0, 3])
def test_mtp_gather_translates_mixed_decode_rows(row_offset):
    prefill = GlmDsaPrefillTopK(
        workspace_indices=torch.tensor([[0, 1], [2, -1], [3, 4]], dtype=torch.int32),
        topk_lens=torch.tensor([2, 1, 2], dtype=torch.int32),
        page_table=torch.empty((0, 0), dtype=torch.int32),
        seq_lens=torch.empty(0, dtype=torch.int32),
        kv_seq_lens=torch.empty(0, dtype=torch.int32),
        max_seq_len=0,
        kv_workspace_slots=torch.tensor([10, 11, 12, 13, 14], dtype=torch.int32),
    )
    # Offset zero covers KPool's whole-batch layout; offset three stores only
    # the two decode rows past the three extend tokens.
    leading = [[-1, -1]] * (3 - row_offset)
    decode = GlmDsaDecodeTopK(
        topk_indices=torch.tensor(leading + [[50, 51], [60, 61]], dtype=torch.int32),
        topk_lens=torch.tensor([0] * len(leading) + [2, 1], dtype=torch.int32),
        row_offset=row_offset,
    )
    _, selected = _prepare_dsa_topk_for_mtp_decode(
        (prefill, decode),
        torch.tensor([2, 4, 3]),
        num_prefill_rows=1,
    )
    assert selected.topk_indices.tolist() == [[13, 14], [60, 61], [50, 51]]
    assert selected.topk_lens.tolist() == [2, 1, 2]
    assert selected.row_offset == 0


def test_mtp_gather_can_select_only_decode_rows_from_a_mixed_batch():
    decode = GlmDsaDecodeTopK(
        topk_indices=torch.tensor([[50, 51], [60, 61]], dtype=torch.int32),
        topk_lens=torch.tensor([2, 1], dtype=torch.int32),
        row_offset=3,
    )
    _, selected = _prepare_dsa_topk_for_mtp_decode(
        (None, decode), torch.tensor([4, 3]), num_prefill_rows=0
    )
    assert selected.topk_indices.tolist() == [[60, 61], [50, 51]]
    assert selected.topk_lens.tolist() == [1, 2]
    assert selected.row_offset == 0


@pytest.mark.parametrize("start,end", [(2, 4), (3, 6), (4, 3)])
def test_decode_slice_rejects_rows_outside_its_window(start, end):
    decode = GlmDsaDecodeTopK(
        topk_indices=torch.zeros(2, 4, dtype=torch.int32),
        topk_lens=torch.zeros(2, dtype=torch.int32),
        row_offset=3,
    )
    with pytest.raises(RuntimeError, match="outside the selected top-k rows"):
        GlmMoeDsaAttention._slice_decode_topk(decode, start, end)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA DSA top-k required")
@pytest.mark.parametrize("width", [1, 2, 6])
@torch.inference_mode()
def test_mixed_prefill_reuses_the_decode_topk_workspace(tmp_path, monkeypatch, width):
    args, model, config = _generate(
        tmp_path,
        monkeypatch,
        arch=9,
        rope_dim=64,
        rank=512,
        dtype="bfloat16",
        kv_dtype="bfloat16",
        quant_method="none",
    )
    config = replace(config, speculative_num_draft_tokens=width)
    _, latent_pool = _pool(args, model, config, device="cuda")
    spec = config.component(DSAConfig)
    pool = DSATokenToKVPool(
        latent_pool.arena,
        torch.bfloat16,
        torch.bfloat16,
        "none",
        512,
        64,
        2,
        0,
        index_head_dim=spec.index_head_dim,
    )
    monkeypatch.setitem(global_server_args_dict, "mapping", args.mapping)
    monkeypatch.setitem(global_server_args_dict, "attention_backend", "dsa")
    monkeypatch.setitem(global_server_args_dict, "numerics", "auto")
    monkeypatch.setitem(global_server_args_dict, "dsa_slot_order", "selection")
    attention = GlmMoeDsaAttention(
        config=model.hf_text_config,
        mapping=args.mapping,
        hidden_size=16,
        num_heads=8,
        qk_nope_head_dim=128,
        qk_rope_head_dim=64,
        v_head_dim=128,
        q_lora_rank=64,
        kv_lora_rank=512,
        layer_id=0,
    ).to(device="cuda", dtype=torch.bfloat16)
    backend = DSABackend(config, spec, kernel_page_size=64)
    backend.init_cuda_graph_state(3)
    torch.manual_seed(42)
    pool.set_index_k_buffer(
        0,
        torch.arange(64, 192, device="cuda"),
        torch.randn((128, spec.index_head_dim), device="cuda", dtype=torch.bfloat16),
        write_mask=None,
    )
    query = torch.randn(
        (2 * width, spec.index_n_heads, spec.index_head_dim),
        device="cuda",
        dtype=torch.bfloat16,
    )
    weights = torch.rand((2 * width, spec.index_n_heads), device="cuda")

    def select(prefill_tokens):
        num_extends = int(prefill_tokens > 0)
        bs = num_extends + 2
        lengths = torch.tensor(
            ([prefill_tokens] if num_extends else []) + [64, 64],
            device="cuda",
            dtype=torch.int32,
        )
        pages = torch.zeros((bs, 1), dtype=torch.int32, device="cuda")
        pages[-2:, 0] = torch.tensor([1, 2], device="cuda", dtype=torch.int32)
        backend.refresh_decode_metadata(bs, bs, lengths, pages, num_extends=num_extends)
        queries = torch.cat(
            (torch.zeros_like(query[:1]).expand(prefill_tokens, -1, -1), query)
        )
        head_weights = torch.cat(
            (torch.zeros_like(weights[:1]).expand(prefill_tokens, -1), weights)
        )
        ctx = ForwardContext(
            attn_backend=backend,
            token_to_kv_pool=pool,
            bs=bs,
            num_extends=num_extends,
            input_num_tokens=queries.shape[0],
            forward_mode=ForwardMode.MIXED if num_extends else ForwardMode.DECODE,
            output_layout=ForwardOutputLayout(num_extends, num_extends, 2, width),
        )
        return attention._compute_decode_topk_indices(
            GlmDsaIndexerOutput(
                query=queries, key=torch.empty(0, device="cuda"), weights=head_weights
            ),
            ctx,
        )

    decode = select(0)
    expected = decode.topk_indices.clone(), decode.topk_lens.clone()
    mixed = select(8192)
    assert mixed.topk_indices.shape == (2 * width, spec.index_topk)
    assert mixed.topk_indices.data_ptr() == decode.topk_indices.data_ptr()
    assert mixed.topk_lens.data_ptr() == decode.topk_lens.data_ptr()
    assert attention._retired_decode_workspaces == []
    assert (decode.row_offset, mixed.row_offset) == (0, 8192)
    indices, lengths = attention._slice_decode_topk(mixed, 8192, 8192 + 2 * width)
    # The production native selector does not promise tie/selection order.
    # Both rounds must select the same slots without growing the workspace.
    torch.testing.assert_close(
        indices.sort(dim=-1).values, expected[0].sort(dim=-1).values, rtol=0, atol=0
    )
    torch.testing.assert_close(lengths, expected[1], rtol=0, atol=0)
    assert lengths.tolist() == list(range(65 - width, 65)) * 2
