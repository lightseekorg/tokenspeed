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

"""Compare TP captures at identical input histories, including every vocabulary logit."""

import argparse
import json
from pathlib import Path

import torch


def compare_logits(actual, expected):
    if (
        actual.shape != expected.shape
        or not torch.isfinite(actual).all()
        or not torch.isfinite(expected).all()
    ):
        raise ValueError(
            "Finite logits with identical full-vocabulary shapes are required"
        )
    a, b = actual.float().reshape(-1), expected.float().reshape(-1)
    diff = (a - b).abs()
    log_actual = torch.log_softmax(actual.float(), dim=-1)
    log_expected = torch.log_softmax(expected.float(), dim=-1)
    result = dict(
        max_abs=diff.max().item(),
        rms=diff.square().mean().sqrt().item(),
        cosine=torch.nn.functional.cosine_similarity(a, b, dim=0).item(),
        greedy_equal=actual.argmax(-1).tolist() == expected.argmax(-1).tolist(),
        mean_signed=(actual.float() - expected.float()).mean().item(),
        kl_reference_to_parallel=(log_expected.exp() * (log_expected - log_actual))
        .sum(-1)
        .max()
        .item(),
        reference_rms=b.square().mean().sqrt().item(),
        normalized_rms=(
            diff.square().mean().sqrt() / b.square().mean().sqrt().clamp_min(1e-12)
        ).item(),
        reference_top1_margin=(
            expected.float().topk(2, dim=-1).values.diff(dim=-1).abs().min().item()
        ),
        top1_regret=(
            expected.float().max(dim=-1).values
            - expected.float().gather(-1, actual.argmax(-1, keepdim=True)).squeeze(-1)
        )
        .max()
        .item(),
    )
    result["passed"] = (
        result["max_abs"] <= 0.75
        and result["rms"] <= 0.10
        and result["cosine"] >= 0.999
    )
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--baseline", type=Path, required=True)
    p.add_argument("--parallel", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--steps", type=int, required=True)
    args = p.parse_args()
    records = []
    for directory in sorted(args.baseline.glob("length-*")):
        paths = sorted(directory.glob("[0-9][0-9][0-9][0-9].pt"))
        if not paths:
            continue
        other = args.parallel / "rank-0" / directory.name
        assert args.steps > 0 and len(paths) >= args.steps, directory
        assert len(list(other.glob("[0-9][0-9][0-9][0-9].pt"))) >= args.steps, directory
        # The overlap pipeline can issue an extra forward after the final committed token.
        for path in paths[: args.steps]:
            baseline = torch.load(path, weights_only=True)
            parallel = torch.load(other / path.name, weights_only=True)
            rank1 = torch.load(
                args.parallel / "rank-1" / directory.name / path.name, weights_only=True
            )
            torch.testing.assert_close(
                baseline["input_ids"], parallel["input_ids"], atol=0, rtol=0
            )
            torch.testing.assert_close(
                baseline["positions"], parallel["positions"], atol=0, rtol=0
            )
            torch.testing.assert_close(
                parallel["logits"], rank1["logits"], atol=0, rtol=0
            )
            record = compare_logits(parallel["logits"], baseline["logits"])
            record.update(request=directory.name, step=int(path.stem))
            records.append(record)
    assert records, "No capture files found"
    result = dict(
        thresholds=dict(max_abs=0.75, rms=0.10, cosine=0.999),
        compared_steps=len(records),
        ranks_agree_exactly=True,
        all_passed=all(r["passed"] for r in records),
        records=records,
    )
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({k: v for k, v in result.items() if k != "records"}))
    assert result["all_passed"], "TP logits differ beyond the predeclared gates"


if __name__ == "__main__":
    main()
