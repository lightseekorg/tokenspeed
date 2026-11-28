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

"""Compare exact full-vocabulary scoring on an idle GPU using CUDA graphs.

Synthetic BF16-rounded FP32 logits; excludes model forward, verification, D2H
and publishing. The accepted lengths model verify output already on the GPU.
The caller registers any optional normalizer provider before running this file.
"""

import argparse
import gc
import json
import statistics
from pathlib import Path

import torch
from tokenspeed_kernel.ops.sampling.full_vocab import (
    FullVocabScores,
    FullVocabTopK,
    _helper,
)
from tokenspeed_kernel.ops.sampling.torch import (
    torch_full_vocab_head,
    torch_full_vocab_tail,
)
from tokenspeed_kernel.thirdparty.cuda import full_vocab_scoring


def timed(fn):
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        for _ in range(4):
            retained = fn()
    torch.cuda.current_stream().wait_stream(stream)
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        retained = fn()
    samples = []
    for _ in range(7):
        for _ in range(5):
            graph.replay()
        start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(
            enable_timing=True
        )
        start.record()
        for _ in range(100):
            graph.replay()
        end.record()
        end.synchronize()
        samples.append(start.elapsed_time(end) * 10)
    return {"median_us": statistics.median(samples), "samples_us": samples}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rows", type=int, nargs="+", required=True)
    parser.add_argument("--vocab", type=int, required=True)
    parser.add_argument("--k", type=int, required=True)
    parser.add_argument("--width", type=int, required=True)
    parser.add_argument("--accepted", type=int, required=True)
    parser.add_argument("--normalizer-solution", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    assert full_vocab_scoring.available(), "build the optional exact CUDA library first"
    assert 0 <= args.accepted <= args.width
    assert all(rows > 0 and rows % args.width == 0 for rows in args.rows)
    scorer = FullVocabTopK(
        k=args.k,
        vocab_block=32768,
        form="negative-cross-entropy",
        solution=args.normalizer_solution,
    )
    report = {
        "device": torch.cuda.get_device_name(),
        "torch": torch.__version__,
        "vocab": args.vocab,
        "k": args.k,
        "width": args.width,
        "accepted": args.accepted,
        "normalizer": scorer.plan.kernel.name,
        "input": "synthetic BF16-rounded FP32 logits",
        "method": "CUDA Graph replay; CUDA events; median of 7 samples x 100 replays",
        "results": [],
    }
    for rows in args.rows:
        torch.manual_seed(37)
        logits = torch.randn((rows, args.vocab), device="cuda").bfloat16().float()
        tokens = torch.arange(rows, device="cuda")
        lengths = torch.full(
            (rows // args.width,), args.accepted, dtype=torch.int32, device="cuda"
        )
        pools = torch.arange(1, lengths.numel() + 1, device="cuda")
        valid = torch.arange(rows, device="cuda") % args.width < args.accepted
        all_rows = torch.ones(rows, dtype=torch.bool, device="cuda")

        def denominator(shifted):
            partials = scorer.plan.kernel(shifted, block_size=scorer.vocab_block)
            return _helper("full_vocab_sum", partials)(partials).log()

        def reference(*, fused_tail):
            ids = torch.topk(logits, args.k, dim=-1).indices
            maximum = logits.max(dim=-1).values
            shifted = logits - maximum[:, None]
            log_z = denominator(shifted)
            head = -(log_z[:, None] - shifted.gather(1, ids))
            tail_fn = (
                _helper("full_vocab_tail", shifted)
                if fused_tail
                else torch_full_vocab_tail
            )
            tail = tail_fn(logits, ids, log_z, row_maximum=maximum, active_rows=None)
            return FullVocabScores(
                shifted,
                log_z,
                ids,
                head,
                tail,
                scorer.form,
                scorer.vocab_block,
                scorer.plan.kernel.name,
            )

        def selected():
            scores = scorer.begin(logits, defer_head=True).complete_accepted(
                lengths,
                pools,
                num_prefill_outputs=0,
                num_extends=0,
                decode_width=args.width,
                padding_index=lengths.numel() + 1,
                prediction_rows=None,
            )
            return scores.pack(tokens, prediction_rows=None)

        baseline = reference(fused_tail=False)
        expected = baseline.pack(tokens, prediction_rows=None)
        for actual, ref in zip(selected(), expected, strict=True):
            assert torch.equal(
                actual[valid].contiguous().view(torch.int32),
                ref[valid].contiguous().view(torch.int32),
            )
        for actual, ref in zip(
            scorer.prepare(logits).pack(tokens, prediction_rows=None),
            expected,
            strict=True,
        ):
            assert torch.equal(
                actual.contiguous().view(torch.int32),
                ref.contiguous().view(torch.int32),
            )
        values = torch.topk(logits, args.k, dim=-1).values
        maximum = logits.max(dim=-1).values
        stages = {
            "baseline_full": lambda: reference(fused_tail=False).pack(
                tokens, prediction_rows=None
            ),
            "tail_fusion_only_full": lambda: reference(fused_tail=True).pack(
                tokens, prediction_rows=None
            ),
            "all_rows_maximum_reuse_full": lambda: scorer.prepare(logits).pack(
                tokens, prediction_rows=None
            ),
            "accepted_rows_full": selected,
            "baseline_topk": lambda: torch.topk(logits, args.k, dim=-1),
            "selected_topk_all": lambda: _helper("full_vocab_topk", logits)(
                logits, k=args.k, active_rows=all_rows
            ),
            "selected_topk_accepted": lambda: _helper("full_vocab_topk", logits)(
                logits, k=args.k, active_rows=valid
            ),
            "baseline_maximum": lambda: logits.max(dim=-1).values,
            "reuse_maximum": lambda: _helper("full_vocab_maximum", logits)(
                logits, values
            ),
            "baseline_head": lambda: torch_full_vocab_head(
                baseline.shifted, baseline.ids, baseline.log_z, form=scorer.form
            ),
            "fused_head": lambda: _helper("full_vocab_head", logits)(
                baseline.shifted, baseline.ids, baseline.log_z, form=scorer.form
            ),
            "baseline_tail": lambda: torch_full_vocab_tail(
                logits,
                baseline.ids,
                baseline.log_z,
                row_maximum=maximum,
                active_rows=None,
            ),
            "fused_tail_all": lambda: _helper("full_vocab_tail", logits)(
                logits,
                baseline.ids,
                baseline.log_z,
                row_maximum=maximum,
                active_rows=all_rows,
            ),
            "fused_tail_accepted": lambda: _helper("full_vocab_tail", logits)(
                logits,
                baseline.ids,
                baseline.log_z,
                row_maximum=maximum,
                active_rows=valid,
            ),
        }
        result = {
            "rows": rows,
            "bitwise": True,
            "timings": {name: timed(fn) for name, fn in stages.items()},
        }
        report["results"].append(result)
        print(json.dumps(result), flush=True)
        args.output.write_text(json.dumps(report, indent=2))
        del (
            stages,
            baseline,
            expected,
            logits,
            tokens,
            lengths,
            pools,
            valid,
            all_rows,
            values,
        )
        gc.collect()
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
