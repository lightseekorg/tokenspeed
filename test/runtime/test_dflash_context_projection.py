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

"""The block draft's attention-TP-split context projection (``fc``) matches
the replicated one, through weight loading and the projection, and the
target's capture sink writes exactly what the projection of the
concatenated taps gives. The draft's final norm reduces the last layer's
dense-TP MLP partial."""

import functools
import importlib.util
import os
import socket
import sys
import traceback
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from tokenspeed_kernel.platform import current_platform
from transformers import PretrainedConfig

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from ci_system.ci_register import register_cuda_ci  # noqa: E402

register_cuda_ci(est_time=60, suite="runtime-4gpu")

from tokenspeed.runtime.distributed.mapping import Mapping  # noqa: E402
from tokenspeed.runtime.models.dflash import DFlashDraftModel  # noqa: E402

TAPS = 3
ROWS = 24


def _draft(mapping: Mapping, hidden: int) -> DFlashDraftModel:
    """A layerless draft: only its context projection is under test."""
    config = PretrainedConfig(
        hidden_size=hidden,
        num_hidden_layers=0,
        rms_norm_eps=1e-6,
        dflash_config={"target_layer_ids": list(range(TAPS)), "block_size": 8},
    )
    return DFlashDraftModel(config, mapping)


@pytest.mark.parametrize("world_size", [1, 8])
def test_each_rank_loads_and_projects_its_fc_output_columns(world_size):
    hidden = 16
    # Small integers keep every sum exact, so the split must match bit for bit.
    full_weight = torch.randint(-3, 4, (hidden, TAPS * hidden)).float()
    taps = torch.randint(-3, 4, (ROWS, TAPS * hidden)).float()
    columns = []
    for rank in range(world_size):
        mapping = Mapping(rank=rank, world_size=world_size, attn_tp_size=world_size)
        model = _draft(mapping, hidden)
        model.load_weights([("model.fc.weight", full_weight)])
        width = hidden // world_size
        assert torch.equal(
            model.fc.weight, full_weight[rank * width : (rank + 1) * width]
        )
        columns.append(model.fc(taps)[0])
    assert torch.equal(torch.cat(columns, dim=-1), taps @ full_weight.T)


def test_a_draft_replicates_its_mlp_only_under_attention_dp() -> None:
    """Under attention DP a rank may skip the draft, so its MLP is replicated
    (no dense collective); at attention DP 1 it keeps the target's TP shard."""
    dp = Mapping(rank=3, world_size=8, attn_tp_size=1, attn_dp_size=8, dense_tp_size=8)
    assert _draft(dp, 16).mapping.dense.tp_size == 1
    tp = Mapping(rank=3, world_size=8, attn_tp_size=8, dense_tp_size=8)
    assert _draft(tp, 16).mapping is tp


def _embedding(vocab: int, hidden: int, table: torch.Tensor, rank, size):
    """The target's embedding on one rank: vocab-sharded over ``size`` ranks,
    or the whole table when ``size`` is None."""
    from tokenspeed.runtime.layers.vocab_parallel_embedding import (
        VocabParallelEmbedding,
    )

    if size is None:
        embed = VocabParallelEmbedding(vocab, hidden, params_dtype=torch.bfloat16)
    else:
        embed = VocabParallelEmbedding(
            vocab,
            hidden,
            params_dtype=torch.bfloat16,
            tp_rank=rank,
            tp_size=size,
            tp_group=tuple(range(size)),
        )
    embed.weight_loader(embed.weight, table)
    return embed


VOCAB = 256
# Small integers keep every sum exact in BF16.
TABLE = torch.randint(
    -3, 4, (VOCAB, 16), generator=torch.Generator().manual_seed(0)
).bfloat16()


def _check_first_layer_embeds(rank, world_size, port, embed_shards):
    from tokenspeed.runtime.distributed.comm_ops import all_reduce
    from tokenspeed.runtime.distributed.process_group_manager import (
        process_group_manager as pg_manager,
    )
    from tokenspeed.runtime.execution.drafter.dflash import DFlash
    from tokenspeed.runtime.utils.env import global_server_args_dict

    device = torch.device("cuda", rank)
    torch.cuda.set_device(device)
    # Attention DP 1: the draft's dense group is every rank.
    mapping = Mapping(rank=rank, world_size=world_size, attn_tp_size=world_size)
    pg_manager.init_distributed(
        mapping=mapping,
        distributed_init_method=f"tcp://localhost:{port}",
        backend="nccl",
    )
    pg_manager.init_process_group(mapping.dense.tp_group)
    global_server_args_dict["mapping"] = mapping

    drafter = DFlash.__new__(DFlash)
    drafter.model = _draft(mapping, 16)
    drafter.embed_tokens = _embedding(
        VOCAB, 16, TABLE, rank, world_size if embed_shards else None
    ).to(device)
    drafter._first_layer_input = drafter._first_layer_input_mode()
    ids = torch.tensor([0, 5, 37, 100, 255, 128, 64, 1], device=device)

    # The first layer sums its input over the dense group, as here.
    first_layer = all_reduce(
        drafter._first_layer_embeds(ids).clone(), mapping.dense.tp_group
    )
    want = TABLE.to(device)[ids]
    assert torch.equal(first_layer, want), (first_layer / want).nan_to_num().max()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="GPU collective")
@pytest.mark.parametrize("world_size", [2, 4, 8])
@pytest.mark.parametrize("embed_shards", [True, False], ids=["sharded", "whole"])
def test_the_first_layer_sums_to_one_embedding(world_size, embed_shards) -> None:
    """At attention DP 1 the draft's first layer reduces its input over the
    dense group, so that reduce must yield the embedding exactly once (not
    once per rank), whether the target's embedding is vocab-sharded or
    whole."""
    if world_size > torch.cuda.device_count():
        pytest.skip(f"Need {world_size} GPUs, have {torch.cuda.device_count()}")
    error_dict = mp.Manager().dict()
    mp.spawn(
        _worker,
        args=(
            world_size,
            _open_port(),
            error_dict,
            functools.partial(_check_first_layer_embeds, embed_shards=embed_shards),
        ),
        nprocs=world_size,
        join=True,
    )
    if error_dict:
        raise RuntimeError("\n".join(f"Rank {r}: {e}" for r, e in error_dict.items()))


def test_a_replicated_draft_takes_the_whole_embedding() -> None:
    """Under attention DP the draft's dense group is one rank, which reduces
    nothing, so the lookup alone must hand it the whole embedding."""
    from tokenspeed.runtime.execution.drafter.dflash import DFlash

    mapping = Mapping(rank=1, world_size=4, attn_tp_size=1, attn_dp_size=4)
    drafter = DFlash.__new__(DFlash)
    drafter.model = _draft(mapping, 16)
    drafter.embed_tokens = _embedding(VOCAB, 16, TABLE, 0, None)
    drafter._first_layer_input = drafter._first_layer_input_mode()
    ids = torch.tensor([3, 200, 17])
    assert torch.equal(drafter._first_layer_embeds(ids), TABLE[ids])


def test_an_embedding_sharded_over_another_group_is_refused() -> None:
    from tokenspeed.runtime.execution.drafter.dflash import DFlash

    mapping = Mapping(rank=0, world_size=8, attn_tp_size=8)
    drafter = DFlash.__new__(DFlash)
    drafter.model = _draft(mapping, 16)
    drafter.embed_tokens = _embedding(VOCAB, 16, TABLE, 0, 4)
    with pytest.raises(ValueError, match="not the embedding's shard group"):
        drafter._first_layer_input_mode()


def _bf16_steps_apart(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """How many representable BF16 values lie between a and b, elementwise."""

    def ordered(x: torch.Tensor) -> torch.Tensor:
        # Sign-magnitude bits to an integer that increases with the value.
        bits = x.contiguous().view(torch.int16).int()
        return torch.where(bits < 0, -(bits & 0x7FFF), bits)

    return (ordered(a) - ordered(b)).abs()


def _open_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("", 0))
        return sock.getsockname()[1]


def _worker(rank, world_size, port, error_dict, check):
    try:
        check(rank, world_size, port)
    except Exception:
        error_dict[rank] = traceback.format_exc()
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()


class _TapTarget:
    """This target hands taps to the sink instead of storing them itself."""

    def __init__(self, stores_taps: bool):
        self._stores_taps = stores_taps

    def stores_capture_taps(self, ctx) -> bool:
        return self._stores_taps


def _sink_drafter(model, device, max_bs: int, spec_num_tokens: int):
    """A DFlash holding only the state its capture sink runs on."""
    from tokenspeed.runtime.execution.drafter.dflash import DFlash

    drafter = DFlash.__new__(DFlash)
    drafter.target_model = _TapTarget(stores_taps=False)
    drafter.device = device
    drafter.hidden_size = int(model.config.hidden_size)
    drafter.target_layer_ids = list(model.target_layer_ids)
    drafter.spec_num_tokens = spec_num_tokens
    drafter.draft_model_runner = SimpleNamespace(model=model)
    drafter.input_buffers = SimpleNamespace(
        max_bs=max_bs,
        positions_buf=torch.arange(max_bs * spec_num_tokens, device=device),
    )
    drafter.attn_backend = SimpleNamespace(
        decode_window_locations=lambda: (
            torch.arange(max_bs * spec_num_tokens, device=device) + 100
        )
    )
    drafter._fused_kv_enabled = True
    drafter._init_context_taps()
    assert drafter._tap_rows is not None
    return drafter


def _check_projection(rank, world_size, port):
    from tokenspeed.runtime.distributed.process_group_manager import (
        process_group_manager as pg_manager,
    )
    from tokenspeed.runtime.execution.context import ForwardContext
    from tokenspeed.runtime.execution.forward_batch_info import ForwardMode
    from tokenspeed.runtime.execution.output_layout import ForwardOutputLayout
    from tokenspeed.runtime.utils.env import global_server_args_dict

    device = torch.device("cuda", rank)
    torch.cuda.set_device(device)
    mapping = Mapping(rank=rank, world_size=world_size, attn_tp_size=world_size)
    pg_manager.init_distributed(
        mapping=mapping,
        distributed_init_method=f"tcp://localhost:{port}",
        backend="nccl",
    )
    pg_manager.init_process_group(mapping.attn.tp_group)
    global_server_args_dict["mapping"] = mapping
    global_server_args_dict["force_deterministic_rsag"] = False

    hidden = 1024
    torch.manual_seed(11)
    full_weight = (torch.randn(hidden, TAPS * hidden) / 32).bfloat16().to(device)
    norm_weight = torch.randn(hidden).bfloat16().to(device)
    # A target with an FP32 residual hands the sink its FP32 taps.
    taps = [torch.randn(ROWS, hidden, device=device) for _ in range(TAPS)]

    model = _draft(mapping, hidden).to(device=device, dtype=torch.bfloat16)
    # The checkpoint loader hands every rank the full weight.
    model.load_weights(
        [("model.fc.weight", full_weight), ("model.hidden_norm.weight", norm_weight)]
    )
    assert model.fc.weight.shape == (hidden // world_size, TAPS * hidden)

    # The projection: one GEMM over the concatenated taps on this rank's
    # output columns, then the column gather and the norm.
    concatenated = torch.cat(taps, dim=-1).bfloat16()
    projected = model.project_target_hidden(concatenated)
    reference = model.hidden_norm(concatenated @ full_weight.T)
    # The per-rank GEMM over fewer output columns may pick another BLAS
    # kernel, which can round an output 1 BF16 step differently; the norm
    # passes that on as 1 step (measured on MI355X at TP 1/4/8).
    assert _bf16_steps_apart(projected, reference).max() <= 1

    # The capture sink: each tap copied as it lands, one projection and
    # write at the last tap, all on the current stream.
    spec_num_tokens = 4
    drafter = _sink_drafter(model, device, ROWS // spec_num_tokens, spec_num_tokens)
    writes = []
    drafter._write_native_cache_fused = lambda rows, positions, locs: writes.append(
        rows.clone()
    )
    ctx = ForwardContext(
        attn_backend=SimpleNamespace(
            decode_window_locations=lambda: torch.arange(ROWS, device=device)
        ),
        token_to_kv_pool=None,
        bs=ROWS // spec_num_tokens,
        num_extends=0,
        output_layout=ForwardOutputLayout(0, 0, ROWS // spec_num_tokens, 1),
        input_num_tokens=ROWS,
        forward_mode=ForwardMode.DECODE,
    )
    drafter.prepare_target_forward(ctx)
    assert ctx.target_capture_sink is drafter
    for i, tap in enumerate(taps):
        drafter.on_target_capture(i, tap)
    torch.cuda.synchronize()

    assert drafter._context_written
    (written,) = writes
    # The sink rounds each FP32 tap once and projects the same block.
    assert torch.equal(written, projected), (written - projected).abs().max()
    # Every rank writes the same rows.
    gathered = [torch.empty_like(written) for _ in range(world_size)]
    dist.all_gather(gathered, written)
    for other in gathered:
        assert torch.equal(other, written)


def _kernel_test_utils():
    helper = Path(__file__).resolve().parents[2] / "tokenspeed-kernel/test/utils.py"
    spec = importlib.util.spec_from_file_location("kernel_test_utils", helper)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _check_extend_warmup(rank, world_size, port):
    """The drafter's extend warmup compiles the projection's Iris column
    gather for the rows a prefill brings (a long chunk and a short last one)."""
    from tokenspeed_kernel.ops.communication._iris.all_gather import (
        iris_pull_allgather_columns,
    )
    from tokenspeed_kernel.ops.tuning import kernel_warmup

    from tokenspeed.runtime.distributed.comm_backend import set_global_backend
    from tokenspeed.runtime.distributed.comm_backend.iris import IrisBackend
    from tokenspeed.runtime.distributed.comm_backend.nccl import NcclBackend
    from tokenspeed.runtime.distributed.process_group_manager import (
        process_group_manager as pg_manager,
    )
    from tokenspeed.runtime.utils.env import global_server_args_dict

    device = torch.device("cuda", rank)
    torch.cuda.set_device(device)
    mapping = Mapping(rank=rank, world_size=world_size, attn_tp_size=world_size)
    pg_manager.init_distributed(
        mapping=mapping,
        distributed_init_method=f"tcp://localhost:{port}",
        backend="nccl",
    )
    pg_manager.init_process_group(mapping.attn.tp_group)
    global_server_args_dict["mapping"] = mapping
    global_server_args_dict["force_deterministic_rsag"] = False
    # Serving's gather capacity: a long chunk's gather takes two subgroups
    # a program, a short one one.
    set_global_backend(
        IrisBackend(NcclBackend(), producer_direct_max_bytes=8192 * 8192 * 2)
    )

    # Muse Spark's draft: 8192 hidden columns, 1024 a rank at TP8.
    hidden = 8192
    model = _draft(mapping, hidden).to(device=device, dtype=torch.bfloat16)
    drafter = _sink_drafter(model, device, 1, 8)
    with kernel_warmup(enabled=True):
        drafter.warmup_extend(8192)
    taps = torch.zeros((8192, TAPS * hidden), dtype=torch.bfloat16, device=device)
    with _kernel_test_utils().assert_no_triton_compile(iris_pull_allgather_columns):
        for rows in (151, 8192):
            model.project_target_hidden(taps[:rows])
        torch.cuda.synchronize()


def _warmup_worker(rank, world_size, port, error_dict):
    try:
        _check_extend_warmup(rank, world_size, port)
    except Exception:
        error_dict[rank] = traceback.format_exc()
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()


@pytest.mark.skipif(
    not torch.cuda.is_available()
    or torch.cuda.device_count() < 8
    or not current_platform().is_cdna4,
    reason="the Iris column gather needs eight CDNA4 GPUs",
)
def test_extend_warmup_compiles_the_projection_gather():
    error_dict = mp.Manager().dict()
    mp.spawn(_warmup_worker, args=(8, _open_port(), error_dict), nprocs=8, join=True)
    if error_dict:
        raise RuntimeError("\n".join(f"Rank {r}: {e}" for r, e in error_dict.items()))


@pytest.mark.skipif(not torch.cuda.is_available(), reason="GPU projection")
@pytest.mark.parametrize("world_size", [1, 4, 8])
def test_split_context_projection_matches_the_replicated_one(world_size):
    if world_size > torch.cuda.device_count():
        pytest.skip(f"Need {world_size} GPUs, have {torch.cuda.device_count()}")
    error_dict = mp.Manager().dict()
    mp.spawn(
        _worker,
        args=(world_size, _open_port(), error_dict, _check_projection),
        nprocs=world_size,
        join=True,
    )
    if error_dict:
        raise RuntimeError("\n".join(f"Rank {r}: {e}" for r, e in error_dict.items()))


def _check_final_norm(rank, world_size, port, fusion_max_tokens):
    from tokenspeed.runtime.distributed.process_group_manager import (
        process_group_manager as pg_manager,
    )
    from tokenspeed.runtime.execution.context import ForwardContext
    from tokenspeed.runtime.execution.forward_batch_info import ForwardMode
    from tokenspeed.runtime.execution.output_layout import ForwardOutputLayout
    from tokenspeed.runtime.utils.env import global_server_args_dict

    device = torch.device("cuda", rank)
    torch.cuda.set_device(device)
    # Attention DP 1, so the draft keeps the target's dense TP width.
    mapping = Mapping(rank=rank, world_size=world_size, attn_tp_size=world_size)
    pg_manager.init_distributed(
        mapping=mapping,
        distributed_init_method=f"tcp://localhost:{port}",
        backend="nccl",
    )
    pg_manager.init_process_group(mapping.dense.tp_group)
    global_server_args_dict["mapping"] = mapping
    global_server_args_dict["comm_fusion_max_num_tokens"] = fusion_max_tokens

    hidden = 1024
    model = _draft(mapping, hidden).to(device=device, dtype=torch.bfloat16)
    assert model.mapping.dense.tp_size == world_size
    torch.manual_seed(5)
    model.load_weights([("model.norm.weight", torch.rand(hidden).bfloat16() + 0.5)])
    # Small integers keep the cross-rank sum exact in BF16.
    partials = torch.randint(-4, 5, (world_size, ROWS, hidden), device=device)
    partials = partials.bfloat16()
    residual = torch.randint(-4, 5, (ROWS, hidden), device=device).bfloat16()

    ctx = ForwardContext(
        attn_backend=None,
        token_to_kv_pool=None,
        bs=ROWS,
        num_extends=0,
        output_layout=ForwardOutputLayout(0, 0, ROWS, 1),
        input_num_tokens=ROWS,
        forward_mode=ForwardMode.DECODE,
    )
    out = model.final_norm(partials[rank].clone(), residual.clone(), ctx)
    reference = model.norm(partials.sum(dim=0), residual.clone())[0]
    assert torch.equal(out, reference), (out - reference).abs().max()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="GPU collective")
@pytest.mark.parametrize("world_size", [4, 8])
@pytest.mark.parametrize("fusion_max_tokens", [-1, 2048], ids=["unfused", "fused"])
def test_final_norm_reduces_the_last_mlp_partial(world_size, fusion_max_tokens):
    """The layers' MLPs hand back shard-local partials; the last one reaches
    ``final_norm``, which must sum it over the dense group before the norm,
    both as a separate all-reduce and fused with the norm."""
    if world_size > torch.cuda.device_count():
        pytest.skip(f"Need {world_size} GPUs, have {torch.cuda.device_count()}")
    error_dict = mp.Manager().dict()
    mp.spawn(
        _worker,
        args=(
            world_size,
            _open_port(),
            error_dict,
            functools.partial(_check_final_norm, fusion_max_tokens=fusion_max_tokens),
        ),
        nprocs=world_size,
        join=True,
    )
    if error_dict:
        raise RuntimeError("\n".join(f"Rank {r}: {e}" for r, e in error_dict.items()))
