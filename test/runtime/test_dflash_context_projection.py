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
the replicated one, through weight loading, the eager projection, and the
per-tap projection the target's capture sink runs."""

import os
import socket
import sys
import traceback
from types import SimpleNamespace

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
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


def _worker(rank, world_size, port, error_dict):
    try:
        _check_projection(rank, world_size, port)
    except Exception:
        error_dict[rank] = traceback.format_exc()
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()


def _incremental_drafter(model, device, max_bs: int, spec_num_tokens: int):
    """A DFlash holding just the state its capture sink runs on."""
    from tokenspeed.runtime.execution.drafter.dflash import DFlash

    drafter = DFlash.__new__(DFlash)
    drafter.device = device
    drafter.hidden_size = int(model.config.hidden_size)
    drafter.target_layer_ids = list(model.target_layer_ids)
    drafter.spec_num_tokens = spec_num_tokens
    drafter.draft_model_runner = SimpleNamespace(model=model)
    drafter.input_buffers = SimpleNamespace(
        max_bs=max_bs,
        positions_buf=torch.arange(max_bs * spec_num_tokens, device=device),
    )
    drafter._fused_kv_enabled = True
    drafter._kv_aux_stream = torch.cuda.Stream(device=device)
    drafter._kv_join_event = torch.cuda.Event()
    drafter._fused_ctx_rows = torch.empty(
        (max_bs * spec_num_tokens, drafter.hidden_size),
        dtype=model.fc.weight.dtype,
        device=device,
    )
    drafter._init_incremental_proj()
    assert drafter._incremental_proj_enabled
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

    # The replicated projection the split replaces: one fc GEMM per tap,
    # accumulated in BF16, then the norm.
    acc = torch.zeros(ROWS, hidden, dtype=torch.bfloat16, device=device)
    for i, tap in enumerate(taps):
        weight = full_weight[:, i * hidden : (i + 1) * hidden]
        torch.addmm(acc, tap.bfloat16(), weight.t(), out=acc)
    reference = model.hidden_norm(acc)

    # Eager: one GEMM over the concatenated taps, then the column gather.
    concatenated = torch.cat(taps, dim=-1).bfloat16()
    eager = model.project_target_hidden(concatenated)
    eager_reference = model.hidden_norm(concatenated @ full_weight.T)
    # The per-rank GEMM over fewer output columns may pick another BLAS
    # kernel, which can round an output 1 BF16 step differently; the norm
    # passes that on as 1 step (measured on MI355X at TP 1/4/8).
    assert _bf16_steps_apart(eager, eager_reference).max() <= 1

    # The capture sink: per-tap GEMMs on the aux stream, gather on the main.
    spec_num_tokens = 4
    drafter = _incremental_drafter(model, device, ROWS // spec_num_tokens, 4)
    writes = []

    def record_write(rows, positions, locs):
        # Runs on the aux stream after the gathered rows are ready there.
        writes.append(rows.clone())

    drafter._write_native_cache_fused = record_write
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
    torch.cuda.current_stream().wait_event(drafter._kv_join_event)
    torch.cuda.synchronize()

    assert drafter._incremental_kv_write_done
    (written,) = writes
    # Each output column sums the same terms in the same order either way.
    assert torch.equal(written, reference), (written - reference).abs().max()
    # Every rank writes the same rows.
    gathered = [torch.empty_like(written) for _ in range(world_size)]
    dist.all_gather(gathered, written)
    for other in gathered:
        assert torch.equal(other, written)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="GPU projection")
@pytest.mark.parametrize("world_size", [1, 4, 8])
def test_split_context_projection_matches_the_replicated_one(world_size):
    if world_size > torch.cuda.device_count():
        pytest.skip(f"Need {world_size} GPUs, have {torch.cuda.device_count()}")
    error_dict = mp.Manager().dict()
    mp.spawn(
        _worker,
        args=(world_size, _open_port(), error_dict),
        nprocs=world_size,
        join=True,
    )
    if error_dict:
        raise RuntimeError("\n".join(f"Rank {r}: {e}" for r, e in error_dict.items()))
