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

"""Native NPU registry integration: preconverted loading, EP filtering and route combine."""

import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import torch
import torch_npu  # noqa: F401 -- register the NPU backend
from tokenspeed_kernel.ops.moe import moe_apply, moe_plan, moe_process_weights


def write_preconverted_fixture(directory, *, layer_index, ep_rank, ep_size):
    """Serialize synthetic INT4 codes directly; no checkpoint quantizer involved."""
    directory = Path(directory)
    files = {}
    generator = torch.Generator(device="cpu").manual_seed(128)
    for name, n, k, left, right in [
        ("w13", 4608, 5120, 64, 80),
        ("w2", 5120, 2304, 48, 48),
    ]:
        path = directory / f"{name}.pt"
        torch.save(
            dict(
                version=1,
                key=path.stem,
                weight=torch.randint(
                    -(2**31),
                    2**31,
                    (2, n, k // 8),
                    generator=generator,
                    dtype=torch.int32,
                    device="cpu",
                ),
                scales=torch.full((2, n), 0.001, dtype=torch.float32, device="cpu"),
                left=torch.eye(left, dtype=torch.bfloat16, device="cpu"),
                right=torch.eye(right, dtype=torch.bfloat16, device="cpu"),
                clip_ratio=1.0,
                input_size=k,
                output_size=n,
            ),
            path,
        )
        files[name] = path.name
    config = directory / "flatquant.json"
    config.write_text(
        json.dumps(
            {
                "format": "tokenspeed-v41-w4a4",
                "version": 1,
                "ep_size": ep_size,
                "layers": {str(layer_index): {str(ep_rank): files}},
            }
        )
    )
    return config


class Weights(torch.nn.Module):
    def __init__(self, config):
        super().__init__()
        self.tp_size = 1
        self.ep_rank = 1
        self.ep_size = 2
        self.num_local_experts = 2
        self.hidden_size = 5120
        self.intermediate_size = 2304
        self.layer_index = 19
        self.w13_input_layout = "concatenated"
        self.swiglu_beta = None
        self.swiglu_arg = SimpleNamespace(alpha=1.0, limit=10.0)
        self.npu_flatquant_config = config
        self._moe_backend_state = None
        generator = torch.Generator().manual_seed(122)
        for name, n, k in [("w13", 4608, 5120), ("w2", 5120, 2304)]:
            codes = torch.randint(
                0, 256, (2, n, k // 2), generator=generator, dtype=torch.uint8
            )
            scales = torch.full((2, n, k // 32), 120, dtype=torch.uint8)
            self.register_parameter(
                name + "_weight", torch.nn.Parameter(codes.npu(), requires_grad=False)
            )
            self.register_parameter(
                name + "_weight_scale",
                torch.nn.Parameter(scales.npu(), requires_grad=False),
            )


class AscendRegistryTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(8)
        torch.npu.set_device(0)
        torch.npu.config.allow_internal_format = True
        torch.npu.matmul.allow_hf32 = False

    def test_launch_rejects_conflicting_precision_flags(self):
        from tokenspeed.runtime.utils.server_args import ServerArgs

        args = object.__new__(ServerArgs)
        args.moe_backend = "ascend_w4a4"
        args.npu_flatquant_config = "/tmp/explicit-config.json"
        args.enforce_eager = True
        args.resolve_kernel_backends()
        args.moe_mxfp4_fp8_activation = True
        with self.assertRaisesRegex(ValueError, "FP8"):
            args.resolve_kernel_backends()
        args.moe_mxfp4_fp8_activation = False
        args.enforce_eager = False
        args.resolve_kernel_backends()
        args.enforce_eager = True
        args.moe_backend = "ascend_bf16"
        with self.assertRaisesRegex(ValueError, "flatquant-config"):
            args.resolve_kernel_backends()

    def test_runtime_layer_configuration_and_forward(self):
        from unittest.mock import patch

        from tokenspeed.runtime.layers.moe import utils
        from tokenspeed.runtime.layers.moe.expert import MoELayer
        from tokenspeed.runtime.layers.moe.topk import StandardTopKOutput
        from tokenspeed.runtime.layers.quantization.mxfp4 import Mxfp4Config
        from tokenspeed.runtime.utils.env import global_server_args_dict

        with tempfile.TemporaryDirectory() as tmp:
            config = write_preconverted_fixture(
                tmp, layer_index=0, ep_rank=0, ep_size=1
            )
            old_dtype = torch.get_default_dtype()
            torch.set_default_dtype(torch.bfloat16)
            try:
                for mode in ["ascend_w4a4", "ascend_bf16"]:
                    with (
                        self.subTest(mode=mode),
                        patch.object(utils, "MOE_BACKEND", utils.MoeBackend(mode)),
                        patch.object(
                            utils, "ALL2ALL_BACKEND", utils.All2AllBackend.NONE
                        ),
                        patch.dict(
                            global_server_args_dict,
                            {
                                "enforce_eager": True,
                                "npu_flatquant_config": (
                                    str(config) if mode == "ascend_w4a4" else None
                                ),
                                "moe_mxfp4_fp8_activation": False,
                            },
                        ),
                        torch.device("npu"),
                    ):
                        layer = MoELayer(
                            top_k=2,
                            num_experts=2,
                            hidden_size=5120,
                            intermediate_size=2304,
                            quant_config=Mxfp4Config(
                                is_checkpoint_mxfp4_serialized=True
                            ),
                            layer_index=0,
                            activation="swiglu",
                            swiglu_limit=10.0,
                            routing_mode="precomputed_topk",
                        )
                        layer.w13_weight.data.fill_(0x22)
                        layer.w2_weight.data.fill_(0x22)
                        layer.w13_weight_scale.data.fill_(120)
                        layer.w2_weight_scale.data.fill_(120)
                        layer.process_weights_after_loading(layer)
                        x = torch.ones(2, 5120, dtype=torch.bfloat16, device="npu")
                        routes = StandardTopKOutput(
                            torch.tensor([[0.2, 0.3], [0.3, 0.2]], device="npu"),
                            torch.tensor([[0, 1], [1, 0]], device="npu"),
                            None,
                        )
                        actual = layer(
                            x, routes, num_global_tokens=2, max_num_tokens_per_gpu=2
                        )
                        self.assertTrue(bool(torch.isfinite(actual).all().cpu()))
                        self.assertEqual(tuple(actual.shape), (2, 5120))
            finally:
                torch.set_default_dtype(old_dtype)

    def test_preconverted_dispatch_and_combine(self):
        with tempfile.TemporaryDirectory() as tmp:
            config = write_preconverted_fixture(
                tmp, layer_index=19, ep_rank=1, ep_size=2
            )
            for solution in ["ascend_bf16", "ascend_w4a4"]:
                with self.subTest(solution=solution):
                    w = Weights(str(config) if solution == "ascend_w4a4" else None)
                    plan = moe_plan(
                        "mxfp4",
                        input_dtype=torch.bfloat16,
                        activation="swiglu",
                        routing_mode="precomputed_topk",
                        a2a_backend="none",
                        ep_size=2,
                        ispp=2304,
                        hidden=5120,
                        swiglu_form="standard",
                        activation_clamped=True,
                        expert_id_repeats=True,
                        internal_activation_dtype=(
                            "int4" if solution == "ascend_w4a4" else "input"
                        ),
                        fast_math=False,
                        combine_order="rank",
                        solution=solution,
                    )
                    moe_process_weights(plan, w)
                    torch.manual_seed(12)
                    x = torch.randn(5, 5120).bfloat16().npu()
                    ids = torch.tensor(
                        [[3, 0, 2], [2, 2, 1], [0, 1, 0], [3, 3, 2], [2, 0, 3]],
                        device="npu",
                    )
                    routes = torch.tensor(
                        [
                            [0.1, 0.2, 0.3],
                            [0.2, 0.3, 0.4],
                            [0.2, 0.4, 0.3],
                            [0.0, 0.0, 0.0],
                            [0.3, 0.1, 0.4],
                        ],
                        device="npu",
                    )
                    actual = moe_apply(
                        plan, x, w, None, topk_ids=ids, topk_weights=routes
                    )
                    # Independently evaluate each local route, without sorting,
                    # expert-count construction, or vectorized index_add.
                    expected = torch.zeros_like(x, dtype=torch.float32)
                    for row in range(5):
                        for slot in range(3):
                            expert = int(ids[row, slot].cpu()) - 2
                            if expert not in [0, 1]:
                                continue
                            counts = torch.tensor(
                                [int(expert == 0), int(expert == 1)],
                                device="npu",
                                dtype=torch.int64,
                            )
                            expected[row] += w._moe_backend_state(
                                x[row : row + 1], counts, routes[row, slot : slot + 1]
                            )[0].float()
                    expected = expected.bfloat16().float().cpu()
                    actual = actual.float().cpu()
                    self.assertLess(
                        float((actual - expected).norm() / expected.norm()), 0.025
                    )
                    self.assertEqual(int(actual[2:4].count_nonzero()), 0)
                    empty = moe_apply(
                        plan, x[:0], w, None, topk_ids=ids[:0], topk_weights=routes[:0]
                    )
                    self.assertEqual(tuple(empty.shape), (0, 5120))
                    with self.assertRaises(ValueError):
                        moe_apply(
                            plan,
                            x,
                            w,
                            None,
                            topk_ids=ids,
                            topk_weights=routes,
                            do_finalize=False,
                        )


if __name__ == "__main__":
    unittest.main(verbosity=2)
