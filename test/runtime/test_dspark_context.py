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

"""CPU coverage for local and pipeline DSpark context production."""

import os
import sys
from types import SimpleNamespace

import pytest
import torch
from torch.nn import functional as F

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from ci_system.ci_register import register_cuda_ci

register_cuda_ci(est_time=5, suite="runtime-1gpu")

from tokenspeed.runtime.execution.dspark_context import DSparkContextProducer


@pytest.mark.parametrize(
    "stage_taps", [((0, 1, 2),), ((0,), (1,), (2,)), ((), (0, 1), (2,))]
)
@pytest.mark.parametrize("with_tap_norm", [False, True])
def test_pipeline_projection_matches_concatenated_reference(stage_taps, with_tap_norm):
    generator = torch.Generator().manual_seed(4)
    taps = (2, 4, 7)
    hidden = [torch.randn(5, 6, generator=generator) for _ in taps]
    weights = [torch.randn(4, 6, generator=generator) for _ in taps]
    norms = [
        torch.nn.RMSNorm(6, eps=1e-5) if with_tap_norm else torch.nn.Identity()
        for _ in taps
    ]
    output_norm = torch.nn.RMSNorm(4, eps=1e-5)
    positions = torch.arange(11, 16)
    locations = torch.tensor([17, 18, 29, 30, 31])
    writes = []
    seen = []
    accumulator = None
    for stage, owned in enumerate(stage_taps):

        def project(index, rows):
            assert index in owned
            seen.append(index)
            return F.linear(norms[index](rows), weights[index])

        def write(rows, pos, loc, pool):
            assert pool == "last-stage-cache"
            writes.append((rows, pos, loc))

        model = SimpleNamespace(
            hidden_size=4,
            mapping=SimpleNamespace(
                is_first_pp_rank=stage == 0,
                is_last_pp_rank=stage == len(stage_taps) - 1,
            ),
            project_target_tap=project,
            finalize_target_projection=output_norm,
            write_context_kv=write,
        )
        producer = DSparkContextProducer(
            model, "last-stage-cache" if model.mapping.is_last_pp_rank else None
        )
        accumulator = producer.begin_stage(hidden[0], accumulator)
        for index in owned:
            producer.add_capture(accumulator, index, hidden[index])
        if model.mapping.is_last_pp_rank:
            producer.write_context(accumulator, positions, locations)
        else:
            with pytest.raises(RuntimeError, match="final pipeline stage"):
                producer.write_context(accumulator, positions, locations)
    expected = output_norm(
        F.linear(
            torch.cat([norm(row) for norm, row in zip(norms, hidden)], dim=-1),
            torch.cat(weights, dim=1),
        )
    )
    assert seen == [0, 1, 2]
    assert len(writes) == 1
    torch.testing.assert_close(writes[0][0], expected)
    torch.testing.assert_close(writes[0][1], positions)
    torch.testing.assert_close(writes[0][2], locations)


def test_context_accumulators_do_not_alias_between_inflight_chunks():
    model = SimpleNamespace(
        hidden_size=4,
        mapping=SimpleNamespace(is_first_pp_rank=True, is_last_pp_rank=True),
    )
    producer = DSparkContextProducer(model, object())
    first = producer.begin_stage(torch.ones(3, 6), None)
    second = producer.begin_stage(torch.ones(3, 6), None)
    first.fill_(7)
    assert torch.count_nonzero(second) == 0


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
