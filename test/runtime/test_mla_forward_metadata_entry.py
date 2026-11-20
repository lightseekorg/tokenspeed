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

"""The MLA leaves' shared extend/mixed metadata entry."""

import os
import sys

import pytest
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from ci_system.ci_register import register_cuda_ci

register_cuda_ci(
    est_time=60,
    suite="runtime-1gpu",
    disabled_on_runners=["amd-*"],
    disabled_on_runners_reason="The FlashMLA and TRT-LLM MLA leaves are NVIDIA-only.",
)

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")

LEAVES = (
    "mla",
    "flashmla",
    pytest.param(
        "trtllm_mla",
        marks=pytest.mark.skipif(
            torch.cuda.is_available() and torch.cuda.get_device_capability()[0] != 10,
            reason="TRT-LLM MLA kernels need SM100",
        ),
    ),
)


@pytest.fixture
def make_leaf(monkeypatch):
    from test.runtime.conftest import make_kimi_pool

    from tokenspeed.runtime.layers.attention.backends.paged import (
        flashmla,
        mla,
        trtllm_mla,
    )
    from tokenspeed.runtime.layers.attention.configs.base import AttnConfig
    from tokenspeed.runtime.layers.attention.configs.mla import MLAConfig
    from tokenspeed.runtime.utils.env import global_server_args_dict

    monkeypatch.setitem(global_server_args_dict, "mla_disable_ragged", False)
    monkeypatch.setitem(global_server_args_dict, "chunked_prefill_size", 256)
    monkeypatch.setitem(global_server_args_dict, "mla_chunk_multiplier", 1)
    leaves = {
        "mla": mla.MLAAttnBackend,
        "flashmla": flashmla.FlashMLABackend,
        "trtllm_mla": trtllm_mla.TRTLLMMLABackend,
    }

    def make(backend: str, *, is_draft: bool):
        spec = MLAConfig(
            backend_name=backend,
            num_attention_heads=16,
            num_kv_heads=1,
            head_dim=192,
            attn_tp_size=1,
            kv_lora_rank=512,
            qk_nope_head_dim=128,
            qk_rope_head_dim=64,
            v_head_dim=128,
            scaling=192**-0.5,
            kv_cache_dim=576,
        )
        config = AttnConfig(
            device="cuda",
            dtype=torch.bfloat16,
            kv_cache_dtype=torch.bfloat16,
            kernel_page_size=64,
            context_len=512,
            max_bs=4,
            max_resident_requests=4,
            kv_cache_quant_method="",
            is_draft=is_draft,
            components=(spec,),
        )
        leaf = leaves[backend](config, spec, kernel_page_size=64)
        leaf.set_cache_pool(make_kimi_pool("cuda", usable_pages=6))
        leaf.init_cuda_graph_state(max_bs=4)
        return leaf

    return make


def _init(leaf, mode) -> None:
    # Two extend requests with cached prefixes, then one decode request.
    prefix_cpu = torch.tensor([65, 17], dtype=torch.int32)
    extend_cpu = torch.tensor([4, 3], dtype=torch.int32)
    table_cpu = torch.tensor(
        [[2, 5, 1, 0, 0, 0, 0, 0], [4, 3, 1, 0, 0, 0, 0, 0], [1] * 8],
        dtype=torch.int32,
    )
    leaf.init_forward_metadata(
        3,
        2,
        torch.tensor([69, 20, 8], dtype=torch.int32, device="cuda"),
        table_cpu.cuda(),
        mode,
        extend_seq_lens=extend_cpu.cuda(),
        extend_seq_lens_cpu=extend_cpu,
        extend_prefix_lens=prefix_cpu.cuda(),
        extend_prefix_lens_cpu=prefix_cpu,
        extend_with_prefix=True,
        query_shard=None,
        page_table_cpu=table_cpu[:2],
    )


@pytest.mark.parametrize("backend", LEAVES)
def test_mixed_target_refreshes_its_decode_requests(make_leaf, backend):
    from tokenspeed.runtime.execution.forward_batch_info import ForwardMode

    leaf = make_leaf(backend, is_draft=False)
    _init(leaf, ForwardMode.MIXED)
    assert leaf.chunked_prefill_metadata.extend_seq_lens_cpu.tolist() == [4, 3]
    assert leaf.forward_decode_metadata.num_extends == 2


@pytest.mark.parametrize("backend", LEAVES)
def test_mixed_draft_leaves_decode_metadata_to_its_wrapper(make_leaf, backend):
    from tokenspeed.runtime.execution.forward_batch_info import ForwardMode

    leaf = make_leaf(backend, is_draft=True)
    _init(leaf, ForwardMode.MIXED)
    assert leaf.chunked_prefill_metadata.extend_seq_lens_cpu.tolist() == [4, 3]
    assert leaf.forward_decode_metadata is None


@pytest.mark.parametrize("backend", LEAVES)
def test_override_num_extends_restores_after_an_error(make_leaf, backend):
    from tokenspeed.runtime.execution.forward_batch_info import ForwardMode

    leaf = make_leaf(backend, is_draft=False)
    _init(leaf, ForwardMode.MIXED)
    with pytest.raises(KeyError):
        with leaf.override_num_extends(0):
            assert leaf.forward_decode_metadata.num_extends == 0
            raise KeyError("draft step failed")
    assert leaf.forward_decode_metadata.num_extends == 2


def test_flashmla_prefill_plans_from_the_flag_it_is_given(make_leaf):
    leaf = make_leaf("flashmla", is_draft=False)
    extend_cpu = torch.tensor([4, 3], dtype=torch.int32)
    table_cpu = torch.tensor([[2, 5, 1, 0], [4, 3, 1, 0]], dtype=torch.int32)
    for prefix_cpu, with_prefix in (([65, 17], True), ([0, 0], False)):
        prefix_cpu = torch.tensor(prefix_cpu, dtype=torch.int32)
        leaf._init_prefill_metadata(
            seq_lens=(prefix_cpu + extend_cpu).cuda(),
            extend_prefix_lens=prefix_cpu.cuda(),
            extend_prefix_lens_cpu=prefix_cpu,
            extend_seq_lens=extend_cpu.cuda(),
            extend_seq_lens_cpu=extend_cpu,
            extend_with_prefix=with_prefix,
            page_table=table_cpu.cuda(),
            page_table_cpu=table_cpu,
        )
        # Prefixed requests read paged history; prefix-free ones plan ragged.
        assert leaf.forward_prefill_metadata.use_ragged == (not with_prefix)


@pytest.mark.parametrize("backend", LEAVES)
def test_decode_forward_is_refused(make_leaf, backend):
    from tokenspeed.runtime.execution.forward_batch_info import ForwardMode

    leaf = make_leaf(backend, is_draft=False)
    with pytest.raises(RuntimeError, match="init_forward_metadata only serves"):
        _init(leaf, ForwardMode.DECODE)


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
