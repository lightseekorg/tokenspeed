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

"""Real-shape BF16 TP4 row projection through the 2048-row bucket."""

import gc
import importlib.util
from datetime import timedelta
from pathlib import Path

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp


@torch.no_grad()
def _worker(rank, rendezvous):
    from tokenspeed.runtime.distributed.comm_backend.auto import AutoBackend
    from tokenspeed.runtime.distributed.mapping import DenseLayerMapping
    from tokenspeed.runtime.execution.context import ForwardContext
    from tokenspeed.runtime.execution.forward_batch_info import ForwardMode
    from tokenspeed.runtime.execution.output_layout import ForwardOutputLayout
    from tokenspeed.runtime.layers.linear import (
        DPRowParallelLinear,
        prepare_dp_linear_communication,
        release_dp_linear_communication,
    )
    from tokenspeed.runtime.utils.env import global_server_args_dict

    helper_spec = importlib.util.spec_from_file_location(
        "bf16_projection_kernel_test_utils",
        Path(__file__).resolve().parents[3] / "tokenspeed-kernel/test/utils.py",
    )
    helper = importlib.util.module_from_spec(helper_spec)
    helper_spec.loader.exec_module(helper)
    world_size, input_size, output_size, capacity = 4, 12288, 7168, 2048
    torch.cuda.set_device(rank)
    torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = False
    device = torch.device("cuda", rank)
    dist.init_process_group(
        "nccl",
        init_method=rendezvous,
        rank=rank,
        world_size=world_size,
        timeout=timedelta(seconds=600),
        device_id=device,
    )
    try:
        global_server_args_dict["batch_invariant_collectives"] = False
        global_server_args_dict["force_deterministic_rsag"] = False
        parallel = DenseLayerMapping(
            rank=rank, world_size=world_size, tp_size=world_size, dp_size=1
        )
        with torch.device(device):
            linear = DPRowParallelLinear(
                input_size,
                output_size,
                parallel=parallel,
                params_dtype=torch.bfloat16,
                quant_config=None,
                prefix="o_proj",
            )
        torch.manual_seed(419 + rank)

        def values(shape):
            # Products are multiples of 1/64. Each 3072-wide shard's entire
            # integer sum fits exactly in FP32, independent of GEMM ordering.
            return (
                torch.randint(-3, 4, shape, dtype=torch.int8, device=device)
                .to(torch.bfloat16)
                .div_(8)
            )

        linear.weight.copy_(values(linear.weight.shape))
        linear.quant_method.process_weights_after_loading(linear)
        weight_parts = [torch.empty_like(linear.weight) for _ in range(world_size)]
        dist.all_gather(weight_parts, linear.weight)
        # Preparation used to select an FP8-only cluster at this capacity and
        # fail before serving any BF16 rows. Use the ordinary runtime entry.
        prepare_dp_linear_communication(linear, capacity, torch.bfloat16, AutoBackend())
        state = linear.projection_workspace.fused_row
        assert state is not None
        assert not state.quantized

        def reference(x):
            total = torch.zeros(
                (x.shape[0], output_size), dtype=torch.float32, device=device
            )
            width = input_size // world_size
            for producer, weight in enumerate(weight_parts):
                partial = x[:, producer * width : (producer + 1) * width] @ weight.T
                # Match the public row contract: BF16 GEMM partials, then a
                # fixed rank-ordered FP32 sum and one final BF16 rounding.
                total.add_(partial.float())
            return total.to(torch.bfloat16)

        def caller_output(rows):
            # Only one rank needs an aligned staging buffer. All four must
            # still enter the same fused collective.
            offset = int(rank == 0)
            elements = rows * output_size
            storage = torch.empty(
                elements + offset, dtype=torch.bfloat16, device=device
            )
            return storage[offset:].view(rows, output_size)

        # Cross the cooperative row ACK protocol in both directions while
        # sharing one large-capacity state, including changing empty owners.
        equal_512 = (512,) * world_size
        counts_list = (
            (128, 128, 128, 128),
            (160, 160, 160, 160),
            equal_512,
            (0, 128, 65, 1),
            (0, 512, 257, 1),
            (511, 385, 0, 127),
            (193, 0, 129, 1),
            (128, 0, 1, 32),
            (0, 160, 1, 32),
            (64, 32, 0, 1),
            (0, 128, 1, 32),
            (2048, 0, 1, 32),
        )
        inputs = [values((counts[rank], input_size)) for counts in counts_list]
        contexts = [
            ForwardContext(
                attn_backend=None,
                token_to_kv_pool=None,
                bs=counts[rank],
                num_extends=0,
                input_num_tokens=counts[rank],
                forward_mode=ForwardMode.DECODE,
                output_layout=ForwardOutputLayout(0, 0, counts[rank], 1),
                global_num_tokens=list(counts),
                collective_global_num_tokens=list(counts),
            )
            for counts in counts_list
        ]
        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        retained = []
        with torch.cuda.stream(stream):
            for case_index, (x, ctx) in enumerate(zip(inputs, contexts)):
                expected = reference(x)
                epoch = int(state.input_flags[0])
                out = caller_output(x.shape[0]) if case_index == 0 else None
                actual, bias = linear(x, ctx=ctx, out=out)
                assert bias is None
                if out is not None:
                    assert actual is out
                torch.testing.assert_close(actual, expected, rtol=0, atol=0)
                # Reading after the forward synchronizes this GPU. A native epoch
                # advance proves AutoBackend did not silently take the fallback.
                assert int(state.input_flags[0]) == epoch + 1
                retained.append((actual, expected))
        stream.synchronize()

        # Cancellation and adjacent outputs distinguish rank order and lane order.
        order_input = linear.weight.new_zeros((512, input_size))
        shard_width = input_size // world_size
        order_input[:, ::shard_width] = order_input.new_tensor((2**26, 1, -(2**26), 1))
        order_input[:, 1::shard_width] = order_input.new_tensor(
            (1, 2**26, -(2**26), -4)
        )
        order_weight = torch.zeros_like(linear.weight)
        order_weight[0, 0] = order_weight[1, 1] = 1
        order_expected = order_input.new_zeros((512, output_size))
        order_expected[:, 0] = 1
        order_expected[:, 1] = -4
        order_out = torch.empty_like(order_expected)
        with helper.assert_no_cute_compile():
            state.run(order_input, order_weight, None, equal_512, order_out)
        torch.testing.assert_close(order_out, order_expected, rtol=0, atol=0)

        # Replay bucket changes and changing empty owners without host joins.
        # The full-capacity eager call above also precedes a smaller row stride.
        stream.wait_stream(torch.cuda.current_stream())
        graph = torch.cuda.CUDAGraph()
        alias_case = counts_list.index(equal_512)
        alias_input = inputs[alias_case]
        if rank % 2:
            # This output overlaps the second band of input rows.
            start = 512 * (input_size - output_size)
            alias_output = alias_input.flatten()[start:].view(512, output_size)
        else:
            alias_output = alias_input.new_empty((512, output_size))
        caller_outputs = {
            0: caller_output(inputs[0].shape[0]),
            alias_case: alias_output,
        }
        gc.collect()
        gc_was_enabled = gc.isenabled()
        gc.disable()
        try:
            with (
                helper.assert_no_cute_compile(),
                torch.cuda.graph(graph, stream=stream),
            ):
                outputs = []
                for case_index, (x, ctx) in enumerate(zip(inputs[:-1], contexts[:-1])):
                    torch.cuda._sleep(25000 * ((rank + case_index) % world_size))
                    out = caller_outputs.get(case_index)
                    outputs.append(linear(x, ctx=ctx, out=out)[0])
        finally:
            if gc_was_enabled:
                gc.enable()
        for case_index, out in caller_outputs.items():
            assert outputs[case_index] is out
        for replay in range(3):
            for x in inputs[:-1]:
                if replay == 0:
                    x.zero_()
                else:
                    x.copy_(values(x.shape))
            expected = [reference(x) for x in inputs[:-1]]
            epoch = int(state.input_flags[0])
            torch.cuda._sleep(25000 * ((rank + replay) % world_size))
            graph.replay()
            for actual, ref in zip(outputs, expected):
                torch.testing.assert_close(actual, ref, rtol=0, atol=0)
            assert int(state.input_flags[0]) == epoch + len(outputs)
        graph.reset()
        release_dp_linear_communication(linear)
        for actual, ref in retained + list(zip(outputs, expected)):
            torch.testing.assert_close(actual, ref, rtol=0, atol=0)
    finally:
        dist.destroy_process_group()


def test_bf16_tp4_row_projection_prepares_2048_rows(tmp_path):
    if torch.cuda.device_count() < 4 or torch.cuda.get_device_capability() != (10, 3):
        pytest.skip("requires four node-local GB300 GPUs")
    mp.spawn(
        _worker,
        args=(f"file://{tmp_path / 'rendezvous'}",),
        nprocs=4,
        join=True,
    )
