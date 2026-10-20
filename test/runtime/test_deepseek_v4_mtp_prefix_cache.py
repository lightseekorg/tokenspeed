# Copyright (c) 2026 LightSeek Foundation
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.

from __future__ import annotations

import os
import sys

import pytest

# CI Registration (parsed via AST, runtime no-op)
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from ci_system.ci_register import register_cuda_ci

register_cuda_ci(est_time=30, suite="runtime-1gpu")

from tokenspeed.runtime.execution.drafter.base import BaseDrafter


def test_draft_idle_global_num_tokens_match_multi_step_decode_shape():
    # The Eagle-chain default: step 0 runs the target's rows, the multi-step
    # chain one row per request; one entry per draft forward.
    drafter = BaseDrafter(spec_num_tokens=4, spec_num_steps=3)
    global_num_tokens = [6, 0, 3]
    global_bs = [2, 0, 1]

    steps = drafter.idle_forward_global_num_tokens(
        global_num_tokens, global_bs, global_num_decodes=global_bs
    )

    assert len(steps) == 3
    assert steps[0] is global_num_tokens
    assert steps[1] is global_bs
    assert steps[2] is global_bs
    assert (
        BaseDrafter(spec_num_tokens=1).idle_forward_global_num_tokens(
            global_num_tokens, global_bs, global_num_decodes=global_bs
        )
        == []
    )


def test_block_drafters_idle_with_one_forward_of_the_targets_rows():
    # Block drafters propose the whole block in one draft forward, so an idle
    # rank mirrors them with a single IDLE forward sized by the target's rows.
    from tokenspeed.runtime.execution.drafter.dflash import DFlash
    from tokenspeed.runtime.execution.drafter.dspark import DSpark

    global_num_tokens = [6, 0, 3]
    global_bs = [2, 0, 1]
    for cls in (DFlash, DSpark):
        drafter = cls.__new__(cls)
        drafter.spec_num_steps = 3
        assert drafter.idle_forward_global_num_tokens(
            global_num_tokens, global_bs, global_num_decodes=global_bs
        ) == [global_num_tokens], cls.__name__


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
