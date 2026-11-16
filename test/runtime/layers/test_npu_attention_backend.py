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

from types import SimpleNamespace

import numpy as np
import pytest
import torch
from transformers import Qwen3Config

from tokenspeed.runtime.configs.model_config import AttentionArch, ModelConfig
from tokenspeed.runtime.layers import layernorm as layernorm_module
from tokenspeed.runtime.layers.attention.backends.cache_metadata import (
    CacheBatchMetadata,
)
from tokenspeed.runtime.layers.attention.backends.paged.mha import MHAAttnBackend
from tokenspeed.runtime.layers.attention.kv_cache.recipes.cache_runtime import (
    CacheRuntimeContract,
)
from tokenspeed.runtime.layers.attention.kv_cache.recipes.spec import CacheGroupSpec
from tokenspeed.runtime.layers.attention.registry import (
    _get_backend_cls,
    validate_npu_mtp_verify_width,
)
from tokenspeed.runtime.layers.layernorm import RMSNorm
from tokenspeed.runtime.utils.server_args import ServerArgs

_NPU_AVAILABLE = hasattr(torch, "npu") and torch.npu.is_available()


def test_npu_validates_required_options_without_rewriting_them():
    with pytest.raises(ValueError, match="disable-prefill-graph"):
        ServerArgs(
            model="Qwen/Qwen3-0.6B",
            device="npu",
            attention_backend="mha",
            max_num_seqs=1,
        )
    with pytest.raises(ValueError, match="disable-pdl"):
        ServerArgs(
            model="Qwen/Qwen3-0.6B",
            device="npu",
            attention_backend="mha",
            disable_prefill_graph=True,
            max_num_seqs=1,
        )

    args = ServerArgs(
        model="Qwen/Qwen3-0.6B",
        device="npu",
        attention_backend="mha",
        sampling_backend="greedy",
        disable_prefill_graph=True,
        disable_pdl=True,
        max_num_seqs=1,
    )

    assert args.attention_backend == "mha"
    assert args.sampling_backend == "greedy"
    assert args.disable_prefill_graph
    assert args.disable_pdl
    assert not args.enforce_eager


def test_npu_reuses_standard_mha_backend():
    assert _get_backend_cls("mha", AttentionArch.MHA) is MHAAttnBackend


@pytest.mark.skipif(not _NPU_AVAILABLE, reason="Ascend NPU is unavailable")
@pytest.mark.parametrize("width", [5, 6])
def test_npu_mtp_verify_width_uses_registered_mha_limits(tmp_path, width):
    Qwen3Config(architectures=["Qwen3ForCausalLM"]).save_pretrained(tmp_path)
    args = ServerArgs(
        model=str(tmp_path),
        device="npu",
        attention_backend="mha",
        sampling_backend="greedy",
        disable_prefill_graph=True,
        disable_pdl=True,
        speculative_algorithm="MTP",
        speculative_draft_model_path=str(tmp_path),
        speculative_num_steps=width - 1,
        speculative_num_draft_tokens=width,
        max_num_seqs=1,
    )
    args.mapping.rank = 0
    model = ModelConfig(
        args.model,
        server_args=args,
        trust_remote_code=False,
        dtype="bfloat16",
        model_override_args=args.hf_overrides,
    )

    if width == 6:
        with pytest.raises(ValueError, match="Unsupported NPU MHA verify width 6"):
            validate_npu_mtp_verify_width(args, model)
    else:
        validate_npu_mtp_verify_width(args, model)
    assert not torch.distributed.is_initialized()


def test_npu_allreduce_rmsnorm_uses_device_process_group(monkeypatch):
    x = torch.ones(1, 4)
    residual = torch.zeros_like(x)
    process_group = object()

    monkeypatch.setattr(
        layernorm_module,
        "_allreduce_residual_rmsnorm",
        layernorm_module._torch_allreduce_residual_rmsnorm,
    )
    monkeypatch.setattr(
        layernorm_module,
        "_get_process_group",
        lambda group: process_group,
    )

    def all_reduce(tensor, *, group):
        assert group is process_group
        tensor.mul_(2)

    def reference_rmsnorm(x, weight, eps, residual=None, **_):
        if residual is not None:
            x = x + residual
        output = x * torch.rsqrt(x.square().mean(dim=-1, keepdim=True) + eps) * weight
        return (output, x) if residual is not None else output

    monkeypatch.setattr(torch.distributed, "all_reduce", all_reduce)
    monkeypatch.setattr(layernorm_module, "rmsnorm", reference_rmsnorm)
    # RMSNorm binds its kernel when it is constructed.
    norm = RMSNorm(4)

    output, output_residual, aux = norm.forward_with_allreduce_fusion(
        rank=0,
        group=(0, 1),
        x=x,
        residual=residual,
    )

    torch.testing.assert_close(output, torch.ones_like(x))
    torch.testing.assert_close(output_residual, torch.full_like(residual, 2))
    assert aux is None


@pytest.mark.parametrize("world_size", [2, 4])
def test_npu_allows_multi_card_tensor_parallelism(world_size):
    args = ServerArgs(
        model="Qwen/Qwen3-0.6B",
        device="npu",
        attention_backend="mha",
        disable_prefill_graph=True,
        disable_pdl=True,
        world_size=world_size,
        attn_tp_size=world_size,
        dense_tp_size=world_size,
        max_num_seqs=1,
    )

    assert args.mapping.world_size == world_size
    assert args.mapping.attn.tp_size == world_size
    assert args.mapping.dense.tp_size == world_size


@pytest.mark.skipif(not _NPU_AVAILABLE, reason="Ascend NPU is unavailable")
def test_cache_batch_metadata_accepts_npu_tables():
    contract = CacheRuntimeContract(
        prefix_granularity=4,
        num_lcm_blocks=1,
        token_capacity=4,
        group_specs=(
            CacheGroupSpec(
                group_id="full_attention",
                attn_kind="full_history",
                rows_per_page=4,
                entry_stride_tokens=1,
                replayable=False,
            ),
        ),
        group_page_counts={"full_attention": 2},
        group_packing={"full_attention": 1},
    )
    forward_op = SimpleNamespace(
        block_table_windows=lambda: {
            "full_attention": (np.array([[1]], dtype=np.int32), 1)
        },
        block_table_spans=lambda: {
            "full_attention": np.array([[0, 1]], dtype=np.int32)
        },
    )

    metadata = CacheBatchMetadata.from_forward_op(
        forward_op,
        device="npu",
        contract=contract,
        num_requests=1,
        validate_page_ids=True,
        host_tables=False,
    )
    table = metadata.tables(active_forward_op=forward_op)["full_attention"]

    assert table.device.type == "npu"
    assert table.tolist() == [[1]]


@pytest.mark.skipif(not _NPU_AVAILABLE, reason="Ascend NPU is unavailable")
def test_ascend_decode_does_not_gather_stale_page_table_entries():
    from tokenspeed_kernel_npu.ops.mha import mha_decode_with_kvcache

    q = torch.ones((1, 2, 4), dtype=torch.bfloat16, device="npu")
    k_cache = torch.ones((2, 128, 1, 4), dtype=torch.bfloat16, device="npu")
    v_cache = torch.arange(1024, dtype=torch.bfloat16, device="npu").reshape(
        2, 128, 1, 4
    )
    page_table = torch.tensor([[0, 2**30]], dtype=torch.int32, device="npu")
    cache_seqlens = torch.tensor([1], dtype=torch.int32, device="npu")

    output = mha_decode_with_kvcache(
        q,
        k_cache,
        v_cache,
        page_table,
        cache_seqlens,
        max_seqlen_k=256,
    )

    expected = v_cache[0, 0].expand_as(output)
    torch.testing.assert_close(output, expected)
