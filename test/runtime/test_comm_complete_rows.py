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

"""Complete rows and head partials through real CPU collectives."""

import socket

import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from tokenspeed.runtime.distributed.comm_manager import CommManager
from tokenspeed.runtime.distributed.mapping import Mapping
from tokenspeed.runtime.distributed.process_group_manager import process_group_manager
from tokenspeed.runtime.execution.context import ForwardContext
from tokenspeed.runtime.execution.forward_batch_info import ForwardMode
from tokenspeed.runtime.utils.env import global_server_args_dict


class AddResidual(torch.nn.Module):
    def forward(self, hidden, residual):
        return hidden + residual, residual

    def forward_with_allreduce_fusion(self, *args):
        raise AssertionError("Complete rows must never be all-reduced")


def _worker(rank, port):
    process_group_manager.init_distributed(
        Mapping(rank=rank, world_size=2, attn_tp_size=2),
        distributed_init_method=f"tcp://127.0.0.1:{port}",
        backend="gloo",
    )
    group = (0, 1)
    process_group_manager.init_process_group(group, backend="gloo")
    process_group_manager.register_process_group(
        "nccl", group, process_group_manager.get_process_group("gloo", group)
    )
    global_server_args_dict.update(
        force_deterministic_rsag=True,
        comm_fusion_max_num_tokens=1024,
        numerics="auto",
    )
    try:
        for rows in (0, 1, 5):
            full = torch.arange(rows * 4, dtype=torch.float32).reshape(rows, 4)
            residual = full + 7
            ctx = ForwardContext(
                attn_backend=None,
                token_to_kv_pool=None,
                bs=1,
                num_extends=1,
                input_num_tokens=rows,
                forward_mode=ForwardMode.EXTEND,
                output_layout=None,
            )
            for layer_id, is_moe, prev_is_moe, ep in (
                (0, True, False, 1),
                (1, True, False, 1),
                (1, True, False, 2),
                (1, False, True, 2),
            ):
                mapping = Mapping(
                    rank=rank,
                    world_size=2,
                    attn_tp_size=2,
                    dense_tp_size=1,
                    moe_ep_size=ep,
                )
                comm = CommManager(
                    mapping=mapping,
                    layer_id=layer_id,
                    is_moe=is_moe,
                    prev_is_moe=prev_is_moe,
                    dense_batch_invariant=False,
                    post_attn_layernorm=AddResidual(),
                    query_sharded=False,
                )
                counts = comm.attn_tp_group_scattered_num_tokens(ctx)
                begin = sum(counts[:rank])
                mine = slice(begin, begin + counts[rank])
                input_scattered = comm.needs_pre_attn_all_gather()
                hidden = full[mine] if input_scattered else full
                res = residual[mine] if input_scattered else residual
                scattered = not comm.use_all_reduce(is_moe)
                expected_rows = mine if scattered else slice(None)
                for kvp in (1, 2):
                    source_group = (rank,) if kvp == 1 else group
                    source_counts = [rows] if kvp == 1 else counts
                    projected = comm.pre_attn_partition(
                        hidden, ctx, group=source_group, counts=source_counts
                    )
                    expected_projection = full if kvp == 1 else full[mine]
                    torch.testing.assert_close(projected, expected_projection)
                    global_server_args_dict["enable_allreduce_fusion"] = True
                    out, out_res = comm.post_attn_complete_rows_norm(
                        projected * 2,
                        res,
                        ctx,
                        group=source_group,
                        counts=source_counts,
                    )
                    torch.testing.assert_close(
                        out, (full * 2 + residual)[expected_rows]
                    )
                    torch.testing.assert_close(out_res, residual[expected_rows])
                global_server_args_dict["enable_allreduce_fusion"] = False
                partial, partial_res = comm.post_attn_comm(full * (rank + 1), res, ctx)
                torch.testing.assert_close(partial, (full * 3)[expected_rows])
                torch.testing.assert_close(partial_res, residual[expected_rows])
    finally:
        dist.destroy_process_group()


def test_complete_rows_and_partials_preserve_row_ownership():
    with socket.socket() as listener:
        listener.bind(("127.0.0.1", 0))
        port = listener.getsockname()[1]
    mp.spawn(_worker, args=(port,), nprocs=2, join=True)
