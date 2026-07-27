from __future__ import annotations

import pytest

from tokenspeed.runtime.layers.moe.loader import _build_default_expert_plan
from tokenspeed.runtime.layers.moe.schema import ExpertCheckpointSchema

_KIMI3_SCHEMA = ExpertCheckpointSchema(
    gate_proj_name="w1",
    up_proj_name="w3",
    down_proj_name="w2",
)


@pytest.mark.parametrize(
    "ep_rank,first_global,last_global",
    [
        pytest.param(0, 0, 111, id="first-rank"),
        pytest.param(3, 336, 447, id="middle-rank"),
        pytest.param(7, 784, 895, id="last-rank"),
    ],
)
def test_kimi_k3_ep8_checkpoint_plan_owns_contiguous_112_experts(
    ep_rank: int,
    first_global: int,
    last_global: int,
) -> None:
    plan = _build_default_expert_plan(
        _KIMI3_SCHEMA,
        num_experts=896,
        ep_rank=ep_rank,
        ep_size=8,
    )

    assert len(plan) == 112 * 3
    assert plan[0].local_expert_id == 0
    assert plan[0].checkpoint_weight_name == f"experts.{first_global}.w1."
    assert plan[-1].local_expert_id == 111
    assert plan[-1].checkpoint_weight_name == f"experts.{last_global}.w2."


def test_checkpoint_plan_rejects_uneven_or_out_of_range_ep() -> None:
    with pytest.raises(ValueError, match="divide evenly"):
        _build_default_expert_plan(
            _KIMI3_SCHEMA,
            num_experts=895,
            ep_rank=0,
            ep_size=8,
        )
    with pytest.raises(ValueError, match="valid EP ranks"):
        _build_default_expert_plan(
            _KIMI3_SCHEMA,
            num_experts=896,
            ep_rank=8,
            ep_size=8,
        )
