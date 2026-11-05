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

"""Kimi-K3.1 sparse MLA under decode context parallelism, on two GPUs.

Each rank runs one real K3.1 DSA layer as a rank of a KVP 2 group that is
also its attention-TP group: the full-attention blocks, and with them the
index keys, are dealt across the two ranks, so every selection and attention
step exchanges candidates and partials over NCCL.
"""

import sys
from datetime import timedelta
from pathlib import Path

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

_TEST_DIR = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_TEST_DIR))  # ``ci_system``
sys.path.insert(0, str(_TEST_DIR.parent))  # ``test.runtime.*``
from ci_system.ci_register import register_cuda_ci

register_cuda_ci(est_time=180, suite="runtime-2gpu")

_HEADS = 12


def _rank_heads(attention, rank: int) -> None:
    """Give this rank its own query, value and output heads, as attention TP
    does. The query LoRA, the latent row and the indexer stay replicated:
    every rank selects with the same index keys and writes the same latent
    rows to the blocks it owns."""
    from tokenspeed.runtime.models.deepseek_v3 import _prepare_mla_kv_b_proj_weights

    generator = torch.Generator(device="cuda").manual_seed(100 + rank)
    # The fused A projection's rows: query LoRA (128), latent (576), gate.
    gate = attention.fused_qkv_a_proj_with_mqa.weight[128 + 576 :]
    with torch.no_grad():
        for weight in (
            attention.q_b_proj.weight,
            attention.kv_b_proj.weight,
            attention.o_proj.weight,
            gate,
        ):
            weight.normal_(0.0, 0.05, generator=generator)
    attention.w_kc, attention.w_vc = _prepare_mla_kv_b_proj_weights(
        attention.kv_b_proj.weight, attention
    )


def _sharded_layer(rank: int, *, kv_cache_dtype, spec_width: int, index_topk: int):
    from test.runtime.test_kimi_k31 import _k31_attention_stack

    torch.manual_seed(5)
    stack = _k31_attention_stack(
        kv_cache_dtype=kv_cache_dtype,
        spec_width=spec_width,
        heads=_HEADS,
        index_topk=index_topk,
        kvp_size=2,
        kvp_rank=rank,
    )
    _rank_heads(stack[0], rank)
    return stack


def _dense_case(rank: int, kv_cache_dtype, spec_width: int, fraction: float):
    """History within index_topk: sparse MLA must equal dense MLA."""
    from test.runtime.test_kimi_k31 import (
        _assert_close_to_reference,
        _dense_gated_mla,
        _prefill_then_decode_twice,
    )

    attention, comm, router, pool = _sharded_layer(
        rank, kv_cache_dtype=kv_cache_dtype, spec_width=spec_width, index_topk=2048
    )
    prompt = 300
    hidden = torch.randn(
        prompt + 2 * spec_width,
        256,
        device="cuda",
        dtype=torch.bfloat16,
        generator=torch.Generator(device="cuda").manual_seed(7),
    )
    reference = _dense_gated_mla(attention, hidden, _HEADS)
    # Owners (block - 1) % 2: rank 1, rank 0, rank 1.
    prefill, eager, replayed, _ = _prefill_then_decode_twice(
        attention,
        comm,
        router,
        pool,
        hidden,
        prompt=prompt,
        spec_width=spec_width,
        blocks=[4, 7, 2],
    )
    _assert_close_to_reference(prefill, reference[:prompt], fraction)
    _assert_close_to_reference(eager, reference[prompt : prompt + spec_width], fraction)
    _assert_close_to_reference(replayed, reference[prompt + spec_width :], fraction)


def _selection_case(rank: int, kv_cache_dtype, spec_width: int, fraction: float):
    """History past index_topk: the group selects what one rank holding the
    whole history selects, and attends it to the same output."""
    from test.runtime.test_kimi_k31 import (
        _assert_close_to_reference,
        _k31_attention_stack,
        _prefill_then_decode_twice,
    )

    from tokenspeed_kernel.platform import current_platform

    sharded = _sharded_layer(
        rank, kv_cache_dtype=kv_cache_dtype, spec_width=spec_width, index_topk=512
    )
    whole = _k31_attention_stack(
        kv_cache_dtype=kv_cache_dtype,
        spec_width=spec_width,
        heads=_HEADS,
        index_topk=512,
        kvp_size=1,
        kvp_rank=0,
    )
    with torch.no_grad():
        for source, target in zip(
            sharded[0].parameters(), whole[0].parameters(), strict=True
        ):
            target.copy_(source)
    whole[0].w_kc, whole[0].w_vc = sharded[0].w_kc, sharded[0].w_vc
    prompt = 1200
    hidden = torch.randn(
        prompt + 2 * spec_width,
        256,
        device="cuda",
        dtype=torch.bfloat16,
        generator=torch.Generator(device="cuda").manual_seed(7),
    )
    blocks = [9, 2, 7, 4, 11, 6, 1, 8, 3, 10]
    run = dict(prompt=prompt, spec_width=spec_width, blocks=blocks)
    *outputs, (slots, counts) = _prefill_then_decode_twice(*sharded, hidden, **run)
    *expected, (expected_slots, expected_counts) = _prefill_then_decode_twice(
        *whole, hidden, **run
    )
    assert torch.equal(counts, expected_counts)
    assert (counts == 512).all()
    # gfx950 scores KVP candidates with the Triton leaf and one rank with Gluon's.
    if current_platform().is_amd:
        return
    for row in range(spec_width):
        assert sorted(slots[row, :512].tolist()) == sorted(
            expected_slots[row, :512].tolist()
        )
    for got, reference in zip(outputs, expected, strict=True):
        _assert_close_to_reference(got, reference.float(), fraction)


def _prefix_case(rank: int, kv_cache_dtype, spec_width: int, fraction: float):
    """Prefix caching under KVP: the extend selects over and attends the
    rows an earlier forward dealt to both ranks, as a one-shot prefill does."""
    from test.runtime.test_kimi_k31 import (
        _assert_close_to_reference,
        _extend_over_a_cached_prefix,
    )

    stacks = [
        _sharded_layer(
            rank, kv_cache_dtype=kv_cache_dtype, spec_width=spec_width, index_topk=512
        )
        for _ in range(2)
    ]
    hidden = torch.randn(
        1200,
        256,
        device="cuda",
        dtype=torch.bfloat16,
        generator=torch.Generator(device="cuda").manual_seed(7),
    )
    resumed, expected = _extend_over_a_cached_prefix(
        *stacks, hidden, prefix=640, blocks=[9, 2, 7, 4, 11, 6, 1, 8, 3, 10]
    )
    _assert_close_to_reference(resumed, expected.float(), fraction)


def _worker(rank: int, rendezvous: str, case, kv_cache_dtype, spec_width, fraction):
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
    # The dense delegate sizes its prefix chunks from the launch's chunk size.
    global_server_args_dict.update(mapping=mapping, chunked_prefill_size=8192)
    initialize_comm_backend(use_pynccl=False)
    case(rank, kv_cache_dtype, spec_width, fraction)
    # Clean runs only: after a failure a pending collective would make teardown wait.
    dist.destroy_process_group()


def _spawn(tmp_path, case, kv_cache_dtype, spec_width, fraction):
    if not torch.cuda.is_available() or torch.cuda.device_count() < 2:
        pytest.skip("requires two GPUs")
    mp.spawn(
        _worker,
        args=(
            f"file://{tmp_path / 'rendezvous'}",
            case,
            kv_cache_dtype,
            spec_width,
            fraction,
        ),
        nprocs=2,
        join=True,
    )


# One-GPU test tolerances: BF16 activations and probabilities; FP8 rows and queries.
_DTYPES = [
    (torch.bfloat16, 1, 1e-2),
    (torch.bfloat16, 4, 1e-2),
    (torch.float8_e4m3fn, 1, 8e-2),
]


@pytest.mark.parametrize("kv_cache_dtype, spec_width, fraction", _DTYPES)
def test_kvp2_attention_matches_dense_mla_and_replays_in_a_decode_graph(
    tmp_path, kv_cache_dtype, spec_width, fraction
):
    _spawn(tmp_path, _dense_case, kv_cache_dtype, spec_width, fraction)


@pytest.mark.parametrize("kv_cache_dtype, spec_width, fraction", _DTYPES)
def test_kvp2_selects_and_attends_like_one_rank_holding_the_history(
    tmp_path, kv_cache_dtype, spec_width, fraction
):
    _spawn(tmp_path, _selection_case, kv_cache_dtype, spec_width, fraction)


@pytest.mark.parametrize(
    "kv_cache_dtype, fraction",
    # As the one-GPU prefix test: the runs project different row counts.
    [(torch.bfloat16, 1e-2), (torch.float8_e4m3fn, 3e-2)],
)
def test_kvp2_extend_over_a_cached_prefix_matches_one_prefill(
    tmp_path, kv_cache_dtype, fraction
):
    _spawn(tmp_path, _prefix_case, kv_cache_dtype, 1, fraction)
