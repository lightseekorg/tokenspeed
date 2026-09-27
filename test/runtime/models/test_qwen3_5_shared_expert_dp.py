"""The DeepEP shared expert must be data parallel, not tensor parallel.

``_forward_deepep`` runs on this rank's token shard (``post_attn_comm``
reduce-scatters whenever ``attn_tp_size != moe.tp_ep_size``). A tensor-parallel
shared expert there would reduce partial products belonging to *different*
tokens, so its weights have to be replicated instead.
"""

from __future__ import annotations

import unittest
from unittest import mock

import pytest
import torch
import torch.nn.functional as F
from torch import nn

from tokenspeed.runtime.distributed.mapping import Mapping
from tokenspeed.runtime.layers.linear import (
    MergedColumnParallelLinear,
    ReplicatedLinear,
    RowParallelLinear,
)
from tokenspeed.runtime.models import qwen3_5_moe
from tokenspeed.runtime.models.qwen3_5_moe import Qwen3_5MoeMLP

HIDDEN = 256
INTERMEDIATE = 512


def _mlp(world_size: int, replicate: bool) -> Qwen3_5MoeMLP:
    return Qwen3_5MoeMLP(
        hidden_size=HIDDEN,
        intermediate_size=INTERMEDIATE,
        hidden_act="silu",
        mapping=Mapping(rank=0, world_size=world_size),
        quant_config=None,
        reduce_results=False,
        replicate_weights=replicate,
        parallelism="moe_shared",
    )


class TestSharedExpertReplication(unittest.TestCase):
    def test_tensor_parallel_by_default(self):
        mlp = _mlp(world_size=4, replicate=False)
        self.assertIsInstance(mlp.gate_up_proj, MergedColumnParallelLinear)
        self.assertIsInstance(mlp.down_proj, RowParallelLinear)
        # Sharded: each rank holds 1/4 of the intermediate dimension.
        self.assertEqual(mlp.gate_up_proj.weight.shape, (2 * INTERMEDIATE // 4, HIDDEN))
        self.assertEqual(mlp.down_proj.weight.shape, (HIDDEN, INTERMEDIATE // 4))

    def test_replicated_keeps_full_weights(self):
        mlp = _mlp(world_size=4, replicate=True)
        self.assertIsInstance(mlp.gate_up_proj, ReplicatedLinear)
        self.assertIsInstance(mlp.down_proj, ReplicatedLinear)
        self.assertEqual(mlp.gate_up_proj.weight.shape, (2 * INTERMEDIATE, HIDDEN))
        self.assertEqual(mlp.down_proj.weight.shape, (HIDDEN, INTERMEDIATE))

    def test_replication_lifts_the_block_quant_shard_floor(self):
        """Replicated weights make the dense TP degree irrelevant.

        Block-quantized FP8 requires every shard to stay >= the 128-wide
        quantization block, which caps dense TP at ``INTERMEDIATE / 128`` when
        the shared expert is sharded. Replication removes that cap, so degrees
        past it (TP8 for a 512 intermediate) stay loadable.
        """
        for world_size in (4, 8, 16):
            mlp = _mlp(world_size=world_size, replicate=True)
            self.assertEqual(mlp.gate_up_proj.weight.shape, (2 * INTERMEDIATE, HIDDEN))

    def test_replicated_deep_gemm_fuses_swiglu_quant(self):
        class FakeGateUp(nn.Module):
            _use_deep_gemm_fp8 = True

            def __init__(self):
                super().__init__()
                self.output = None

            def forward(self, x):
                self.output = torch.randn(
                    (x.shape[0], 2 * INTERMEDIATE), dtype=torch.bfloat16
                )
                return self.output, None

        class FakeDown(nn.Module):
            _use_deep_gemm_fp8 = True

            def __init__(self):
                super().__init__()
                self.call = None

            def forward(self, x, block_scale=None, output_dtype=None):
                self.call = (x, block_scale, output_dtype)
                return torch.ones((x.shape[0], HIDDEN), dtype=output_dtype), None

        class UnexpectedActivation(nn.Module):
            def forward(self, x):
                raise AssertionError("standalone SiLU must be bypassed")

        mlp = _mlp(world_size=4, replicate=True)
        gate_up = FakeGateUp()
        down = FakeDown()
        mlp.gate_up_proj = gate_up
        mlp.down_proj = down
        mlp.act_fn = UnexpectedActivation()

        quantized = torch.empty((3, INTERMEDIATE), dtype=torch.float32)
        scales = torch.empty((3, INTERMEDIATE // 512), dtype=torch.int32)
        with (
            mock.patch.object(qwen3_5_moe, "_is_blackwell", True),
            mock.patch.object(
                qwen3_5_moe,
                "fused_swiglu_fp8_ue8m0",
                return_value=(quantized, scales),
            ) as fused,
        ):
            output = mlp(torch.randn((3, HIDDEN), dtype=torch.bfloat16))

        fused.assert_called_once_with(gate_up.output)
        self.assertIs(down.call[0], quantized)
        self.assertIs(down.call[1], scales)
        self.assertIs(down.call[2], torch.bfloat16)
        self.assertEqual(output.shape, (3, HIDDEN))


@pytest.mark.parametrize("moe_tp,moe_ep", [(4, 1), (1, 4), (2, 2)])
@pytest.mark.parametrize("stage", [0, 1])
def test_shared_expert_shards_match_moe_reduction(moe_tp, moe_ep, stage):
    torch.manual_seed(42)
    gate = torch.randn(INTERMEDIATE, HIDDEN) * 0.02
    up = torch.randn(INTERMEDIATE, HIDDEN) * 0.02
    down = torch.randn(HIDDEN, INTERMEDIATE) * 0.02
    x = torch.randn(7, HIDDEN)
    expected = F.linear(F.silu(F.linear(x, gate)) * F.linear(x, up), down)
    partials = []
    for local_rank in range(4):
        mapping = Mapping(
            rank=stage * 4 + local_rank,
            world_size=8,
            pp_size=2,
            attn_tp_size=1,
            attn_dp_size=4,
            dense_tp_size=1,
            moe_tp_size=moe_tp,
            moe_ep_size=moe_ep,
        )
        mlp = Qwen3_5MoeMLP(
            HIDDEN,
            INTERMEDIATE,
            "silu",
            mapping,
            reduce_results=False,
            replicate_weights=False,
            parallelism="moe_shared",
        )
        assert mlp.gate_up_proj.tp_group == mapping.moe.tp_ep_group
        assert mlp.gate_up_proj.tp_size == 4
        assert mlp.gate_up_proj.tp_rank == local_rank
        assert mlp.down_proj.tp_group == mapping.moe.tp_ep_group
        assert not mlp.down_proj.reduce_results
        with torch.no_grad():
            mlp.gate_up_proj.weight_loader(mlp.gate_up_proj.weight, gate, 0)
            mlp.gate_up_proj.weight_loader(mlp.gate_up_proj.weight, up, 1)
            mlp.down_proj.weight_loader(mlp.down_proj.weight, down)
        local_gate, local_up = F.linear(x, mlp.gate_up_proj.weight).chunk(2, dim=-1)
        partials.append(F.linear(F.silu(local_gate) * local_up, mlp.down_proj.weight))
    torch.testing.assert_close(sum(partials), expected, atol=1e-6, rtol=1e-5)


@pytest.mark.parametrize("scope,replicate", [("dense", False), ("moe_shared", True)])
def test_dense_dp_and_deepep_shared_still_replicate(scope, replicate):
    mapping = Mapping(
        rank=2,
        world_size=4,
        attn_tp_size=1,
        dense_tp_size=1,
        moe_tp_size=1,
        moe_ep_size=4,
    )
    mlp = Qwen3_5MoeMLP(
        HIDDEN,
        INTERMEDIATE,
        "silu",
        mapping,
        reduce_results=False,
        replicate_weights=replicate,
        parallelism=scope,
    )
    assert isinstance(mlp.gate_up_proj, ReplicatedLinear)
    assert isinstance(mlp.down_proj, ReplicatedLinear)
    assert mlp.down_proj.weight.shape == (HIDDEN, INTERMEDIATE)


if __name__ == "__main__":
    unittest.main()
