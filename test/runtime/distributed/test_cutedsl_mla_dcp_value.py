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

"""CuTe MLA KVP decode projecting its values as it merges, on two GPUs.

Each rank runs the CuTe MLA backend as a rank of a KVP 2 group. Given a value
weight, the decode merges the partials and projects them, gated or not, into
the caller's output: the bits of the latent merge followed by
``mla_project_value``, in one launch where the group's fused value tail is
prepared and serves the weight.
"""

import sys
from datetime import timedelta
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

_TEST_DIR = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_TEST_DIR))  # ``ci_system``
from ci_system.ci_register import register_cuda_ci

register_cuda_ci(est_time=120, suite="runtime-2gpu")

_HEADS, _LATENT, _DIM, _VALUE, _PAGE, _GRANULARITY = 12, 512, 576, 128, 64, 128


def _leaf(rank: int, *, kvp: bool, dtype, queries: int, draft: bool, blocks: int):
    """Build a CuTe MLA leaf for one rank of a KVP 2 group, or unsharded."""
    from tokenspeed.runtime.layers.attention.backends.paged import tokenspeed_mla
    from tokenspeed.runtime.layers.attention.configs.base import AttnConfig
    from tokenspeed.runtime.layers.attention.configs.mla import MLAConfig
    from tokenspeed.runtime.utils.env import global_server_args_dict

    mapping = global_server_args_dict["mapping"].attn
    spec = MLAConfig(
        backend_name="tokenspeed_mla",
        num_attention_heads=_HEADS * 2,
        num_kv_heads=1,
        head_dim=_DIM,
        attn_tp_size=2,
        kv_lora_rank=_LATENT,
        qk_nope_head_dim=128,
        qk_rope_head_dim=64,
        v_head_dim=_VALUE,
        scaling=192**-0.5,
        kv_cache_dim=_DIM,
    )
    config = AttnConfig(
        device=torch.device("cuda", rank),
        dtype=torch.bfloat16,
        kv_cache_dtype=dtype,
        kv_cache_quant_method="",
        context_len=blocks * _GRANULARITY,
        max_bs=3,
        max_resident_requests=3,
        speculative_num_draft_tokens=queries,
        is_draft=draft,
        draft_block_decode=draft,
        kvp_size=2 if kvp else 1,
        kvp_rank=mapping.kvp_rank if kvp else 0,
        kvp_group=mapping.kvp_group if kvp else (rank,),
        components=(spec,),
    )
    # Skip unrelated prefill compilation.
    with mock.patch.object(tokenspeed_mla, "warmup_compile_prefill", lambda **kw: None):
        leaf = tokenspeed_mla.CuteDSLMLABackend(config, spec, kernel_page_size=_PAGE)
    leaf.set_cache_pool(object())
    leaf.configure_runtime(
        block_granularity=_GRANULARITY,
        virtual_block_count=blocks + 1,
        shard_count=2 if kvp else 1,
    )
    leaf.init_cuda_graph_state(3)
    return leaf


def _prepare_value_tail(group: tuple[int, ...], rows: int) -> None:
    """Prepare the group's decode exchange and value tail like device setup."""
    from tokenspeed_kernel.ops.communication.fabric import gather_fabric_map

    from tokenspeed.runtime.distributed.process_group_manager import (
        process_group_manager,
    )
    from tokenspeed.runtime.layers.attention.dcp import comm

    gather_fabric_map()
    process_group_manager.register_process_group(
        "gloo", group, dist.new_group(backend="gloo")
    )
    comm.prepare_decode_exchange(
        group, 120, value_heads=_HEADS, decode_rows=rows, float_words=False
    )


def _case(rank: int, dtype, queries: int, draft: bool, value_tail: bool):
    from tokenspeed_kernel.ops.attention.mla import mla_project_value
    from tokenspeed_kernel.ops.attention.mla.cute_dsl import DcpValueTail

    from tokenspeed.runtime.utils.env import global_server_args_dict

    device = torch.device("cuda", rank)
    context, batch, blocks = 700, 3, 8
    leaf = _leaf(
        rank, kvp=True, dtype=dtype, queries=queries, draft=draft, blocks=blocks
    )
    assert leaf.supports_mla_projected_value_decode
    torch.manual_seed(31)
    full_cache = torch.randn(
        blocks + 1, _GRANULARITY, _DIM, device=device, dtype=torch.bfloat16
    ).to(dtype)
    full_cache[0].zero_()
    # Owners (block - 1) % 2: this rank's blocks, behind the zero block.
    kvp_rank = global_server_args_dict["mapping"].attn.kvp_rank
    local_cache = full_cache[[0] + list(range(kvp_rank + 1, blocks + 1, 2))]
    local_cache = local_cache.contiguous()
    pool = SimpleNamespace(get_key_buffer=lambda layer_id: local_cache)
    layer = SimpleNamespace(
        layer_id=0,
        tp_q_head_num=_HEADS,
        head_dim=_DIM,
        v_head_dim=_LATENT,
        scaling=192**-0.5,
        sliding_window_size=-1,
        k_scale_float=1.0,
    )
    order = torch.arange(1, blocks + 1, device=device, dtype=torch.int32)
    pages = (order[:, None] * 2 + torch.arange(2, device=device)).flatten().int()
    table = pages[None].expand(batch, -1).clone()
    # The last request has no history: its rows merge two empty partials.
    ends = torch.tensor([context - 1, 8, 0], device=device, dtype=torch.int32)
    leaf.refresh_decode_metadata(batch, batch - 1, ends, table, num_extends=0)
    rows = batch * queries
    generator = torch.Generator(device=device).manual_seed(57 + rank)
    q = torch.randn(
        rows, _HEADS, _DIM, device=device, dtype=torch.bfloat16, generator=generator
    )

    def forward(**projection):
        return leaf.forward_decode(
            q, None, None, layer, None, pool, batch, **projection
        )

    if value_tail:
        _prepare_value_tail(
            tuple(global_server_args_dict["mapping"].attn.kvp_group), rows
        )
    latent = forward().view(rows, _HEADS, _LATENT)
    # A value-major weight: the tail needs a contiguous latent axis.
    k_major = torch.randn(
        _HEADS,
        _VALUE,
        _LATENT,
        device=device,
        dtype=torch.bfloat16,
        generator=generator,
    ).transpose(1, 2)
    weights = {"k-major": k_major / 16, "value-major": k_major.contiguous() / 16}
    gate = torch.randn(
        rows, _HEADS * _VALUE, device=device, dtype=torch.bfloat16, generator=generator
    )
    for name, weight in weights.items():
        for output_gate in (None, gate):
            expected = mla_project_value(
                latent,
                weight,
                gate=output_gate,
                out=torch.empty(
                    rows, _HEADS * _VALUE, device=device, dtype=torch.bfloat16
                ),
            )
            out = torch.full_like(expected, float("nan"))
            with mock.patch.object(
                DcpValueTail,
                "__call__",
                autospec=True,
                side_effect=DcpValueTail.__call__,
            ) as tail_calls:
                result = forward(
                    value_weight=weight, output_gate=output_gate, projected_output=out
                )
            assert result is out
            assert torch.equal(out.view(torch.int16), expected.view(torch.int16)), name
            fused = value_tail and name == "k-major"
            assert tail_calls.call_count == int(fused), name

    # Without a KVP merge there is nothing to project into.
    plain = _leaf(
        rank, kvp=False, dtype=dtype, queries=queries, draft=draft, blocks=blocks
    )
    assert not plain.supports_mla_projected_value_decode
    with pytest.raises(ValueError, match="only in a KVP merge"):
        plain.forward_decode(
            q,
            None,
            None,
            layer,
            None,
            pool,
            batch,
            value_weight=k_major,
            output_gate=None,
            projected_output=out,
        )
    # A draft tree's attention, through either dispatch, projects none either.
    for dispatch in ("_tree_lane_rows", "_tree_verify_rows"):
        with mock.patch.object(leaf, dispatch, return_value=object()):
            with pytest.raises(ValueError, match="projects no values"):
                forward(value_weight=k_major, output_gate=None, projected_output=out)


def _worker(rank: int, rendezvous: str, *case_args):
    torch.set_num_threads(1)
    torch.cuda.set_device(rank)
    # Runtime imports may initialize CUDA, so select the rank's device first.
    from tokenspeed.runtime.distributed.comm_backend.registry import (
        initialize_comm_backend,
    )
    from tokenspeed.runtime.distributed.mapping import KVP_HISTORY_GROUP_NAME, Mapping
    from tokenspeed.runtime.distributed.process_group_manager import (
        process_group_manager,
    )
    from tokenspeed.runtime.utils.env import global_server_args_dict

    dist.init_process_group(
        "nccl",
        init_method=rendezvous,
        rank=rank,
        world_size=2,
        device_id=torch.device("cuda", rank),
        timeout=timedelta(seconds=120),
    )
    mapping = Mapping(
        rank=rank, world_size=2, attn_tp_size=2, attn_dp_size=1, attn_kvp_size=2
    )
    process_group_manager.register_process_group(
        "nccl", mapping.world_group, dist.group.WORLD
    )
    process_group_manager.init_process_group(
        mapping.attn.kvp_group, backend="nccl", name=KVP_HISTORY_GROUP_NAME
    )
    global_server_args_dict.update(mapping=mapping, chunked_prefill_size=8192)
    initialize_comm_backend(use_pynccl=False)
    _case(rank, *case_args)
    # Clean runs only: a failed run's pending collective would hang teardown.
    dist.destroy_process_group()


@pytest.mark.parametrize("value_tail", [False, True])
@pytest.mark.parametrize("queries, draft", [(1, False), (4, True)])
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float8_e4m3fn])
def test_a_kvp_decode_projects_its_merged_values_bit_for_bit(
    tmp_path, dtype, queries, draft, value_tail
):
    if not torch.cuda.is_available() or torch.cuda.device_count() < 2:
        pytest.skip("requires two GPUs")
    if value_tail and torch.cuda.get_device_capability() not in ((10, 0), (10, 3)):
        pytest.skip("the fused decode value tail runs on SM100 and SM103")
    mp.spawn(
        _worker,
        args=(f"file://{tmp_path / 'rendezvous'}", dtype, queries, draft, value_tail),
        nprocs=2,
        join=True,
    )
