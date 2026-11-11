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

"""Head-TP peer addresses survive a real managed-weight discard (two GPUs)."""

import os
import subprocess
import sys
from datetime import timedelta
from pathlib import Path

import pytest
import torch
import torch.distributed as dist


def _worker():
    from transformers import PretrainedConfig

    from tokenspeed.runtime.distributed.comm_backend.auto import AutoBackend
    from tokenspeed.runtime.distributed.comm_backend.registry import set_global_backend
    from tokenspeed.runtime.distributed.mapping import Mapping
    from tokenspeed.runtime.distributed.process_group_manager import (
        process_group_manager as groups,
    )
    from tokenspeed.runtime.models.deepseek_v3 import DeepseekV3AttentionMLA
    from tokenspeed.runtime.utils.env import global_server_args_dict
    from tokenspeed.runtime.utils.torch_memory_saver_adapter import (
        TorchMemorySaverAdapter,
    )

    rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(rank)
    device = torch.device("cuda", rank)
    dist.init_process_group("nccl", device_id=device, timeout=timedelta(seconds=90))
    group = (0, 1)
    groups.register_process_group("nccl", group, dist.group.WORLD)
    groups.register_process_group("gloo", group, dist.new_group(backend="gloo"))
    backend = AutoBackend()
    set_global_backend(backend)
    mapping = Mapping(
        rank=rank,
        world_size=2,
        attn_tp_size=1,
        attn_dp_size=2,
        attn_head_tp_size=2,
        dense_tp_size=1,
        moe_tp_size=1,
        moe_ep_size=2,
        nprocs_per_node=2,
    )
    global_server_args_dict.update(
        mapping=mapping,
        tp_batch_invariant="attn",
        head_tp_exchange_max_rows=16,
        numerics="auto",
    )
    saver = TorchMemorySaverAdapter.create(True)
    torch.set_default_dtype(torch.bfloat16)
    with torch.device(device), saver.region(tag="weights", enable_cpu_backup=False):
        model = DeepseekV3AttentionMLA(
            config=PretrainedConfig(rms_norm_eps=1e-6),
            mapping=mapping,
            hidden_size=256,
            num_heads=8,
            qk_nope_head_dim=128,
            qk_rope_head_dim=64,
            v_head_dim=128,
            q_lora_rank=128,
            kv_lora_rank=128,
            rope_theta=10000.0,
            rope_scaling=None,
            max_position_embeddings=128,
            quant_config=None,
            layer_id=0,
            prefix="layers.0.self_attn",
        )
    # State is backend-owned, so an inventory of model buffers cannot protect it.
    states = tuple(backend._shard_exchanges.values())
    assert len(states) == 3

    def discard_and_check():
        before = [(s.peers.cpu().clone(), s.control.cpu().clone()) for s in states]
        torch.cuda.synchronize()
        dist.barrier()
        saver.pause(tag="weights")
        saver.resume(tag="weights")
        for state, (peers, control) in zip(states, before):
            assert torch.equal(state.peers.cpu(), peers), "Discard lost peer addresses"
            assert torch.equal(
                state.control.cpu(), control
            ), "Discard lost Lamport epochs"

    discard_and_check()
    channels = model.hidden_size
    rows = 4
    x = torch.empty(rows, channels, dtype=torch.bfloat16, device=device)

    def fill_and_reference(seed):
        gen = torch.Generator().manual_seed(seed + rank)
        bits = torch.randint(-32768, 32768, x.shape, generator=gen, dtype=torch.int16)
        x.copy_(bits.view(torch.bfloat16))
        sent = x.view(rows, 2, channels // 2).transpose(0, 1).contiguous()
        reference = torch.empty(2 * rows, channels // 2, dtype=x.dtype, device=device)
        dist.all_to_all_single(reference.view(-1), sent.view(-1))
        return reference

    reference = fill_and_reference(10)
    for _ in range(3):
        result = backend.try_shard_exchange(x, group, inverse=False)
    assert torch.equal(result.view(torch.int16), reference.view(torch.int16))
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, stream=stream):
        result = backend.try_shard_exchange(x, group, inverse=False)
    torch.cuda.current_stream().wait_stream(stream)
    for cycle in range(2):
        discard_and_check()
        reference = fill_and_reference(20 + cycle)
        graph.replay()
        torch.cuda.synchronize()
        assert torch.equal(result.view(torch.int16), reference.view(torch.int16))
    graph.reset()
    dist.destroy_process_group()


def test_head_tp_graph_survives_weight_discard():
    if torch.cuda.device_count() < 2:
        pytest.skip("requires two CUDA GPUs")
    memory_saver = pytest.importorskip("torch_memory_saver")
    env = dict(
        os.environ,
        NCCL_GRAPH_REGISTER="0",
        PYTORCH_CUDA_ALLOC_CONF="expandable_segments:False",
        PYTORCH_ALLOC_CONF="expandable_segments:False",
    )
    with memory_saver.configure_subprocess():
        # The preload hook is installed only for the new interpreter processes.
        env.update({k: v for k, v in os.environ.items() if k.startswith("LD_")})
        subprocess.run(
            [
                sys.executable,
                "-m",
                "torch.distributed.run",
                "--standalone",
                "--nproc-per-node=2",
                str(Path(__file__).resolve()),
            ],
            env=env,
            check=True,
            timeout=180,
        )


if __name__ == "__main__":
    _worker()
