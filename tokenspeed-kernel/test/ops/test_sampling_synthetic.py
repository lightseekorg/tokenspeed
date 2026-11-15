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

"""Synthetic-acceptance verify outputs match the reference in one launch."""

from test.utils import assert_no_triton_compile

import pytest
import torch
from tokenspeed_kernel.ops.sampling import write_synthetic_verify_outputs
from tokenspeed_kernel.ops.sampling.triton.synthetic import _write_synthetic_outputs

if not torch.cuda.is_available():
    pytest.skip("CUDA required", allow_module_level=True)


def _outputs(rows, width, device):
    sentinel = 7777
    return (
        torch.full((rows * width,), sentinel, dtype=torch.int32, device=device),
        torch.full((rows, width), sentinel, dtype=torch.int32, device=device),
        torch.full((rows,), sentinel, dtype=torch.int32, device=device),
    )


def _inputs(rows, width, target_rows, seed):
    generator = torch.Generator().manual_seed(seed)
    # A wider buffer gives the candidates a row stride past their width.
    candidates = torch.randint(
        0, 1 << 20, (rows, width + 3), dtype=torch.int32, generator=generator
    )[:, :width]
    shape = (rows, width) if target_rows else (rows,)
    targets = torch.randint(0, 1 << 20, shape, dtype=torch.int64, generator=generator)
    # Lengths past the width are clamped to it.
    lengths = torch.randint(
        1, width + 3, (rows,), dtype=torch.int32, generator=generator
    )
    return candidates, targets, lengths


def _expected(candidates, targets, lengths):
    rows, width = candidates.shape
    predict, index, accepted = (
        torch.zeros(rows, width, dtype=torch.int32),
        torch.full((rows, width), -1, dtype=torch.int32),
        torch.empty(rows, dtype=torch.int32),
    )
    for row in range(rows):
        length = min(int(lengths[row]), width)
        target = targets[row, length - 1] if targets.ndim == 2 else targets[row]
        predict[row, : length - 1] = candidates[row, 1:length]
        predict[row, length - 1] = target
        index[row, :length] = torch.arange(row * width, row * width + length)
        accepted[row] = length - 1
    return predict.view(-1), index, accepted


@pytest.mark.parametrize("target_rows", [False, True])
@pytest.mark.parametrize("width", [1, 2, 7, 8, 16])
def test_outputs_match_the_reference_on_both_devices(width, target_rows):
    rows = 37
    candidates, targets, lengths = _inputs(rows, width, target_rows, seed=width)
    expected = _expected(candidates, targets, lengths)
    for device in ("cpu", "cuda"):
        outputs = _outputs(rows, width, device)
        write_synthetic_verify_outputs(
            candidates.to(device),
            targets.to(device),
            lengths.to(device),
            *outputs,
        )
        for actual, reference in zip(outputs, expected):
            assert torch.equal(actual.cpu(), reference), device


def test_batch_sizes_and_lengths_reuse_one_kernel():
    width = 8
    candidates, targets, lengths = _inputs(64, width, True, seed=3)
    candidates, targets, lengths = (t.cuda() for t in (candidates, targets, lengths))
    outputs = _outputs(64, width, "cuda")
    write_synthetic_verify_outputs(candidates, targets, lengths, *outputs)
    with assert_no_triton_compile(_write_synthetic_outputs):
        for rows in (1, 3, 16, 64):
            lengths[:rows].random_(1, width + 1)
            sliced = [o[: rows * width] for o in outputs[:1]] + [
                outputs[1][:rows],
                outputs[2][:rows],
            ]
            write_synthetic_verify_outputs(
                candidates[:rows], targets[:rows], lengths[:rows], *sliced
            )
            reference = _expected(
                candidates[:rows].cpu(), targets[:rows].cpu(), lengths[:rows].cpu()
            )
            for actual, ref in zip(sliced, reference):
                assert torch.equal(actual.cpu(), ref)


def test_misshaped_or_misplaced_inputs_are_refused():
    candidates, targets, lengths = _inputs(4, 8, True, seed=1)
    with pytest.raises(ValueError, match=r"targets must be \[4\] or \[4, 8\]"):
        write_synthetic_verify_outputs(
            candidates, targets[:, :3], lengths, *_outputs(4, 8, "cpu")
        )
    with pytest.raises(ValueError, match="share the candidates' device"):
        write_synthetic_verify_outputs(
            candidates, targets, lengths, *_outputs(4, 8, "cuda")
        )
