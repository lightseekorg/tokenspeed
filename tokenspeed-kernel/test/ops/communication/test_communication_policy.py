# Copyright (c) 2026 LightSeek Foundation

from __future__ import annotations

import torch
from tokenspeed_kernel.ops import communication


def test_allreduce_fusion_lane_owns_single_row_eligibility(monkeypatch) -> None:
    monkeypatch.setattr(communication, "_ALLREDUCE_FUSION_LANE", None)
    one_row = torch.ones(1, 4)

    lane = communication.allreduce_fusion_lane(one_row, 6)

    assert lane is not None
    assert lane.shape == (1, 6)
    assert communication.allreduce_fusion_lane(torch.ones(2, 4), 6) is None
    assert communication.allreduce_fusion_lane(one_row, 6, enabled=False) is None
    assert communication.allreduce_lane_latent_norm_supported(one_row)
    assert not communication.allreduce_lane_latent_norm_supported(torch.ones(2, 4))
