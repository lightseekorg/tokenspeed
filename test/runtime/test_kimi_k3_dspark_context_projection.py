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

"""The K3 DSpark draft's split context projection matches the replicated one.

It is checked through checkpoint loading, the concatenated and the per-tap
(pipeline) projections, and the context KV every rank writes.
"""

import os
import socket
import sys
import traceback
from types import SimpleNamespace

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from ci_system.ci_register import register_cuda_ci  # noqa: E402

register_cuda_ci(est_time=120, suite="runtime-4gpu")

from tokenspeed.runtime.configs.kimi_k3_dspark_config import (  # noqa: E402
    KimiK3DSparkConfig,
)
from tokenspeed.runtime.distributed.mapping import Mapping  # noqa: E402
from tokenspeed.runtime.models.kimi_k3_dspark import K3DSparkModel  # noqa: E402

HIDDEN = 256
TAPS = 3
ROWS = 24


def _draft(mapping: Mapping, fc_norm: bool) -> K3DSparkModel:
    """A small real draft in bf16 on this rank's GPU, K3's 512 + 64 latent."""
    config = KimiK3DSparkConfig(
        hidden_size=HIDDEN,
        intermediate_size=256,
        num_hidden_layers=2,
        num_attention_heads=8,
        num_key_value_heads=8,
        q_lora_rank=128,
        kv_lora_rank=512,
        qk_nope_head_dim=64,
        qk_rope_head_dim=64,
        v_head_dim=64,
        vocab_size=1024,
        max_position_embeddings=4096,
        rope_theta=10000.0,
        num_target_layers=TAPS,
        target_hidden_size=HIDDEN,
        target_num_hidden_layers=8,
        target_layer_ids=[1, 4, 6],
        fc_norm=fc_norm,
        mask_token_id=1023,
        markov_rank=16,
        enable_confidence_head=False,
    )
    default_dtype = torch.get_default_dtype()
    torch.set_default_dtype(torch.bfloat16)
    try:
        with torch.device("cuda"):
            model = K3DSparkModel(config, mapping)
    finally:
        torch.set_default_dtype(default_dtype)
    # A cached RoPE table may predate the device scope.
    return model.cuda()


def _checkpoint(model: K3DSparkModel) -> list[tuple[str, torch.Tensor]]:
    """A replicated draft's parameters under their checkpoint names."""
    q_lora_rank = int(model.config.q_lora_rank)
    weights = []
    for name, param in model.named_parameters():
        if ".gate_up_proj." in name:
            gate, up = param.detach().chunk(2, dim=0)
            weights.append((name.replace("gate_up_proj", "gate_proj"), gate))
            weights.append((name.replace("gate_up_proj", "up_proj"), up))
        elif ".fused_qkv_a_proj_with_mqa." in name:
            fused = param.detach()
            weights.append(
                (
                    name.replace("fused_qkv_a_proj_with_mqa", "q_a_proj"),
                    fused[:q_lora_rank],
                )
            )
            weights.append(
                (
                    name.replace("fused_qkv_a_proj_with_mqa", "kv_a_proj_with_mqa"),
                    fused[q_lora_rank:],
                )
            )
        else:
            weights.append((name, param.detach()))
    return [(f"model.{name}", tensor) for name, tensor in weights]


def _bf16_steps_apart(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """How many representable BF16 values lie between a and b, elementwise."""

    def ordered(x: torch.Tensor) -> torch.Tensor:
        bits = x.contiguous().view(torch.int16).int()
        return torch.where(bits < 0, -(bits & 0x7FFF), bits)

    return (ordered(a) - ordered(b)).abs()


def _same_on_every_rank(tensor: torch.Tensor) -> bool:
    gathered = [torch.empty_like(tensor) for _ in range(dist.get_world_size())]
    dist.all_gather(gathered, tensor.contiguous())
    return all(
        torch.equal(other.view(torch.uint8), tensor.view(torch.uint8))
        for other in gathered
    )


class _LatentPool:
    """The MLA pool's write target over plain latent planes."""

    def __init__(self, planes: list[torch.Tensor]) -> None:
        self.planes = planes

    def kv_write_target(self, layer_id, slots, write_mask):
        from tokenspeed_kernel.ops.attention.prologue import LatentKVCache

        return LatentKVCache(self.planes[layer_id], True, slots, write_mask)


def _write(model, rows, positions, slots):
    planes = [
        torch.zeros((64, 1, 576), device="cuda", dtype=torch.float8_e4m3fn)
        for _ in model.layers
    ]
    model.write_context_kv(
        rows,
        positions,
        slots,
        _LatentPool(planes),
        attn_backend=SimpleNamespace(cache_placement=lambda layer: None),
    )
    return torch.stack([plane.view(torch.uint8) for plane in planes])


def _check(rank: int, world_size: int, port: int, fc_norm: bool) -> None:
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
    # The all-gather backend sizes its workspace from the serving limits.
    global_server_args_dict["chunked_prefill_size"] = 8192
    global_server_args_dict["max_prefill_tokens"] = 8192
    global_server_args_dict["max_model_len"] = 4096
    global_server_args_dict["max_num_seqs"] = 64

    # Every rank builds the same replicated reference from one seed.
    torch.manual_seed(7)
    reference = _draft(Mapping(rank=0, world_size=1), fc_norm)
    for param in reference.parameters():
        param.data.normal_(0.0, 0.05)
    for layer in reference.layers:
        layer.self_attn.kv_a_layernorm.weight.data.add_(1.0)
    reference.context_norm.weight.data.add_(1.0)
    reference.post_load_weights()

    model = _draft(mapping, fc_norm)
    # The checkpoint loader hands every rank the full tensors.
    model.load_weights(_checkpoint(reference))
    width = HIDDEN // world_size
    assert model.projection_width == width
    assert torch.equal(
        model.context_proj.weight,
        reference.context_proj.weight[rank * width : (rank + 1) * width],
    )

    torch.manual_seed(11)
    taps = [torch.randn(ROWS, HIDDEN, device=device) for _ in range(TAPS)]
    concatenated = torch.cat(taps, dim=-1).bfloat16()

    # One projection of the concatenated taps, gathered over attention TP.
    projected = model.project_target_hidden(concatenated)
    expected = reference.project_target_hidden(concatenated)
    # The per-rank GEMM over fewer output columns may pick another BLAS
    # kernel; a 1-step projection difference can move a normalized value by 2.
    assert _bf16_steps_apart(projected, expected).max() <= 2
    assert _same_on_every_rank(projected)

    # The pipeline form: per-tap FP32 partial sums of this rank's columns.
    partial = torch.zeros((ROWS, width), dtype=torch.float32, device=device)
    full = torch.zeros((ROWS, HIDDEN), dtype=torch.float32, device=device)
    for index, tap in enumerate(taps):
        partial.add_(model.project_target_tap(index, tap.bfloat16()))
        full.add_(reference.project_target_tap(index, tap.bfloat16()))
    staged = model.finalize_target_projection(partial)
    assert (
        _bf16_steps_apart(staged, reference.finalize_target_projection(full)).max() <= 2
    )
    assert _same_on_every_rank(staged)

    # The context KV every rank writes from the gathered rows is the
    # replicated draft's, byte for byte, and the same on every rank.
    positions = torch.arange(ROWS, device=device) * 37
    slots = torch.randperm(63, device=device)[:ROWS] + 1
    dist.broadcast(slots, src=0)
    written = _write(model, projected, positions, slots)
    assert torch.equal(written, _write(reference, projected, positions, slots))
    assert _same_on_every_rank(written)


def _open_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("", 0))
        return sock.getsockname()[1]


def _worker(rank, world_size, port, fc_norm, error_dict):
    try:
        _check(rank, world_size, port, fc_norm)
    except Exception:
        error_dict[rank] = traceback.format_exc()
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="GPU projection")
@pytest.mark.parametrize("fc_norm", [False, True])
@pytest.mark.parametrize("world_size", [1, 2, 4])
def test_split_context_projection_matches_the_replicated_one(world_size, fc_norm):
    if world_size > torch.cuda.device_count():
        pytest.skip(f"Need {world_size} GPUs, have {torch.cuda.device_count()}")
    error_dict = mp.Manager().dict()
    mp.spawn(
        _worker,
        args=(world_size, _open_port(), fc_norm, error_dict),
        nprocs=world_size,
        join=True,
    )
    if error_dict:
        raise RuntimeError("\n".join(f"Rank {r}: {e}" for r, e in error_dict.items()))
