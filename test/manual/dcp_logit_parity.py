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

"""Compare full pre-sampling TP and DCP logits captured by ModelExecutor.

Start eager TP and DCP servers with ``TOKENSPEED_TEST_LOGIT_DUMP_DIR`` pointing
at separate empty directories, issue the same request, and compare the dumps.
The hook records full FP32 logits before sampling, plus input IDs and positions,
so a generated decode step is compared only when its complete input context is
identical. This avoids the sampled-logprob harness's ambiguity after token
divergence.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch


def _load(directory: Path) -> list[dict]:
    paths = sorted(directory.glob("step-*.pt"))
    if not paths:
        raise AssertionError(f"no logit dumps found in {directory}")
    return [torch.load(path, map_location="cpu", weights_only=False) for path in paths]


def _largest_extend_window(steps: list[dict]) -> list[dict]:
    """Select the captured request with the largest extend input.

    Engine and control-server health probes can race an otherwise empty dump
    directory. They use one-token extends, so select the largest extend and its
    following decode steps up to the next extend.
    """
    extend_indices = [
        index
        for index, step in enumerate(steps)
        if step["forward_mode"] in ("EXTEND", "MIXED")
    ]
    if not extend_indices:
        raise AssertionError("logit dumps contain no extend step")
    start = max(
        extend_indices,
        key=lambda index: int(steps[index]["input_ids"].numel()),
    )
    end = len(steps)
    for index in range(start + 1, len(steps)):
        if steps[index]["forward_mode"] in ("EXTEND", "MIXED"):
            end = index
            break
    return steps[start:end]


def _same_context(actual: dict, expected: dict) -> bool:
    if actual["forward_mode"] != expected["forward_mode"]:
        return False
    if not torch.equal(actual["input_ids"], expected["input_ids"]):
        return False
    actual_positions = actual["positions"]
    expected_positions = expected["positions"]
    if actual_positions is None or expected_positions is None:
        return actual_positions is expected_positions
    return torch.equal(actual_positions, expected_positions)


def _compare_steps(
    actual_steps: list[dict],
    expected_steps: list[dict],
    *,
    atol: float,
    rtol: float = 0.0,
    require_argmax: bool = False,
    topk_report: int = 64,
    compare_logprobs: bool = False,
) -> None:
    """Compare fixed-context full logits and raise on any parity failure."""
    if len(actual_steps) != len(expected_steps):
        raise AssertionError(
            f"forward-step count differs: {len(actual_steps)} != {len(expected_steps)}"
        )

    failures = []
    for index, (actual, expected) in enumerate(zip(actual_steps, expected_steps)):
        if not _same_context(actual, expected):
            failures.append(f"step {index}: input context or forward mode differs")
            continue
        actual_logits = actual["logits"].float()
        expected_logits = expected["logits"].float()
        if actual_logits.shape != expected_logits.shape:
            failures.append(
                f"step {index}: logit shape differs: "
                f"{tuple(actual_logits.shape)} != {tuple(expected_logits.shape)}"
            )
            continue
        difference = (actual_logits - expected_logits).abs()
        max_abs = float(difference.max()) if difference.numel() else 0.0
        mean_abs = float(difference.mean()) if difference.numel() else 0.0
        actual_logprobs = torch.log_softmax(actual_logits, dim=-1)
        expected_logprobs = torch.log_softmax(expected_logits, dim=-1)
        logprob_difference = (actual_logprobs - expected_logprobs).abs()
        max_logprob_abs = (
            float(logprob_difference.max()) if logprob_difference.numel() else 0.0
        )
        mean_logprob_abs = (
            float(logprob_difference.mean()) if logprob_difference.numel() else 0.0
        )
        argmax_equal = torch.equal(
            actual_logits.argmax(dim=-1),
            expected_logits.argmax(dim=-1),
        )
        report_k = min(topk_report, expected_logits.shape[-1])
        expected_topk = torch.topk(
            expected_logprobs, k=report_k, dim=-1, sorted=False
        ).indices
        actual_topk = torch.topk(
            actual_logprobs, k=report_k, dim=-1, sorted=False
        ).indices
        expected_topk_errors = torch.gather(logprob_difference, -1, expected_topk)
        max_expected_topk_logprob_abs = float(expected_topk_errors.max())
        topk_overlap = sum(
            len(set(actual_row.tolist()) & set(expected_row.tolist()))
            for actual_row, expected_row in zip(actual_topk, expected_topk)
        )
        compared_actual = actual_logprobs if compare_logprobs else actual_logits
        compared_expected = expected_logprobs if compare_logprobs else expected_logits
        close = torch.isclose(
            compared_actual,
            compared_expected,
            atol=atol,
            rtol=rtol,
        )
        mismatches = int((~close).sum())
        print(
            f"step {index} mode={actual['forward_mode']} "
            f"max_abs={max_abs:.6g} mean_abs={mean_abs:.6g} "
            f"max_logprob_abs={max_logprob_abs:.6g} "
            f"mean_logprob_abs={mean_logprob_abs:.6g} "
            f"top{report_k}_overlap={topk_overlap}/{actual_topk.numel()} "
            f"max_expected_top{report_k}_logprob_abs="
            f"{max_expected_topk_logprob_abs:.6g} "
            f"mismatches={mismatches} argmax_equal={argmax_equal}"
        )
        if mismatches:
            failures.append(
                f"step {index}: {mismatches} logits exceed atol={atol}, "
                f"rtol={rtol}; max_abs={max_abs}; "
                f"max_logprob_abs={max_logprob_abs}"
            )
        if require_argmax and not argmax_equal:
            failures.append(f"step {index}: next-token argmax differs")
    if failures:
        raise AssertionError("\n".join(failures))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--reference-dir", type=Path, required=True)
    parser.add_argument("--actual-dir", type=Path, required=True)
    parser.add_argument("--atol", type=float, required=True)
    parser.add_argument("--rtol", type=float, default=0.0)
    parser.add_argument("--require-argmax", action="store_true")
    parser.add_argument("--topk-report", type=int, default=64)
    parser.add_argument(
        "--compare-logprobs",
        action="store_true",
        help="Apply tolerances after full-vocabulary log-softmax normalization.",
    )
    args = parser.parse_args()

    expected_steps = _largest_extend_window(_load(args.reference_dir))
    actual_steps = _largest_extend_window(_load(args.actual_dir))
    _compare_steps(
        actual_steps,
        expected_steps,
        atol=args.atol,
        rtol=args.rtol,
        require_argmax=args.require_argmax,
        topk_report=args.topk_report,
        compare_logprobs=args.compare_logprobs,
    )
    print("full pre-sampling TP/DCP logit parity passed")


if __name__ == "__main__":
    main()
