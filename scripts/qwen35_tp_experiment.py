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

"""Run real Ascend TP inference, rank-aware captures and concurrent benchmarks.

All execution choices are explicit. Large captures and traces stay outside Git.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import statistics
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from qwen35_probe import engine_kwargs, install

# Spawned workers execute this module before loading the model.
if os.environ.get("QWEN35_PROBE_DIR"):
    install()


def summarize_wave(requests, elapsed):
    """Aggregate simultaneous requests using the whole wave's wall time."""
    if elapsed <= 0 or not requests:
        raise ValueError("A completed wave needs requests and positive elapsed time")
    return dict(
        wall_seconds=elapsed,
        output_tokens=sum(len(r["output_ids"]) for r in requests),
        output_tokens_per_second=sum(len(r["output_ids"]) for r in requests) / elapsed,
        ttft_ms=statistics.median(r["ttft_ms"] for r in requests),
        tpot_ms=statistics.median(r["tpot_ms"] for r in requests),
        requests=requests,
    )


def run_wave(engine, ids, output_tokens, concurrency, *, ignore_eos):
    barrier = threading.Barrier(concurrency)

    def one(index):
        barrier.wait()
        started = time.perf_counter()
        times, tokens = [], []
        final = None
        for frame in engine.generate(
            input_ids=ids,
            sampling_params=dict(
                temperature=0, max_new_tokens=output_tokens, ignore_eos=ignore_eos
            ),
            stream=True,
        ):
            times.append(time.perf_counter())
            tokens.extend(frame["output_ids"])
            final = frame
        assert final is not None and tokens, final
        assert len(tokens) == final["meta_info"]["completion_tokens"]
        if ignore_eos:
            assert len(tokens) == output_tokens, (len(tokens), output_tokens)
        return dict(
            index=index,
            input_ids=ids,
            output_ids=tokens,
            text=final["text"],
            meta_info=final["meta_info"],
            ttft_ms=(times[0] - started) * 1000,
            tpot_ms=(times[-1] - times[0]) * 1000 / max(1, len(tokens) - 1),
            wall_seconds=times[-1] - started,
        )

    with ThreadPoolExecutor(max_workers=concurrency) as pool:
        started = time.perf_counter()
        requests = list(pool.map(one, range(concurrency)))
        elapsed = time.perf_counter() - started
    return summarize_wave(requests, elapsed)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--tp", type=int, choices=[1, 2], required=True)
    parser.add_argument(
        "--mode", choices=["smoke", "capture", "benchmark", "profile"], required=True
    )
    parser.add_argument("--lengths", type=int, nargs="+", required=True)
    parser.add_argument("--concurrencies", type=int, nargs="+", required=True)
    parser.add_argument("--output-tokens", type=int, required=True)
    parser.add_argument("--warmups", type=int, required=True)
    parser.add_argument("--repeats", type=int, required=True)
    parser.add_argument("--overlap", choices=["on", "off"], required=True)
    parser.add_argument("--work-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.mode in ("capture", "smoke") and args.concurrencies != [1]:
        parser.error("Capture and smoke use one request for interpretable evidence")
    if (
        min(args.lengths + args.concurrencies + [args.output_tokens, args.repeats]) < 1
        or args.warmups < 0
    ):
        parser.error("Lengths, concurrency, output tokens and repeats must be positive")
    if args.work_dir.exists():
        parser.error("Use a fresh work directory")
    devices = os.environ["ASCEND_RT_VISIBLE_DEVICES"].split(",")
    if len(devices) != args.tp or len(set(devices)) != args.tp:
        parser.error("Expose exactly one distinct physical chip per rank")
    if args.tp == 2 and int(devices[0]) // 2 != int(devices[1]) // 2:
        parser.error("The homework requires the two chips of the same board")
    args.work_dir.mkdir(parents=True)
    os.environ["QWEN35_PROBE_DIR"] = str(args.work_dir.resolve())
    install()
    from transformers import AutoTokenizer

    from tokenspeed.runtime.entrypoints.engine import Engine

    tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=True)
    pattern = tokenizer.encode(
        "The research group studies efficient language model inference. Explain the result clearly. ",
        add_special_tokens=False,
    )
    config = engine_kwargs(args.model)
    config.update(
        world_size=args.tp,
        attn_tp_size=args.tp,
        dense_tp_size=args.tp,
        max_num_seqs=max(args.concurrencies),
        max_model_len=max(4096, max(args.lengths) + args.output_tokens),
        max_total_tokens=max(
            16384,
            max(args.concurrencies) * (max(args.lengths) + args.output_tokens + 512),
        ),
        disable_overlap_schedule=args.overlap == "off",
        enable_allreduce_fusion=False,
    )
    trials = []
    result = dict(config=config, arguments=vars(args).copy(), trials=trials)
    repository = Path(__file__).resolve().parents[1]
    result["source_sha256"] = {
        name: hashlib.sha256((repository / name).read_bytes()).hexdigest()
        for name in (
            "python/tokenspeed/runtime/distributed/comm_backend/hccl.py",
            "tokenspeed-kernel/python/tokenspeed_kernel/ops/conv/_triton/causal_conv1d.py",
            "tokenspeed-kernel-npu/python/tokenspeed_kernel_npu/ops/gdn.py",
        )
    }
    result["visible_devices"] = os.environ["ASCEND_RT_VISIBLE_DEVICES"]
    result["arguments"] = {
        k: str(v) if isinstance(v, Path) else v for k, v in result["arguments"].items()
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    control = args.work_dir / "active.json"
    with Engine(**config) as engine:
        for length in args.lengths:
            ids = (pattern * (length // len(pattern) + 1))[:length]
            for concurrency in args.concurrencies:
                for repeat in range(args.warmups + args.repeats):
                    name = f"length-{length}-c{concurrency}-r{repeat}"
                    mode = {
                        "capture": "capture",
                        "profile": "profile",
                        "smoke": "timing",
                        "benchmark": "benchmark",
                    }[args.mode]
                    if repeat < args.warmups:
                        mode = "benchmark"
                    # Atomic replacement: no worker can read a partial JSON document.
                    temporary = args.work_dir / "active.tmp"
                    temporary.write_text(json.dumps(dict(id=name, mode=mode)))
                    temporary.replace(control)
                    trial = run_wave(
                        engine, ids, args.output_tokens, concurrency, ignore_eos=True
                    )
                    trial.update(
                        id=name,
                        input_tokens=length,
                        concurrency=concurrency,
                        warmup=repeat < args.warmups,
                    )
                    waves = args.work_dir / "waves"
                    waves.mkdir(exist_ok=True)
                    (waves / f"{name}.json").write_text(json.dumps(trial) + "\n")
                    # Keep full requests outside Git; summaries retain measured values.
                    trial["requests"] = [
                        {
                            k: v
                            for k, v in request.items()
                            if k != "input_ids"
                            and (
                                args.mode in ("capture", "smoke")
                                or k not in ("output_ids", "text", "meta_info")
                            )
                        }
                        | {
                            "input_tokens": len(request["input_ids"]),
                            "output_tokens": len(request["output_ids"]),
                            "finish_reason": request["meta_info"]["finish_reason"],
                        }
                        for request in trial["requests"]
                    ]
                    trials.append(trial)
                    print(
                        json.dumps(
                            {
                                k: trial[k]
                                for k in (
                                    "id",
                                    "ttft_ms",
                                    "tpot_ms",
                                    "output_tokens_per_second",
                                )
                            }
                        ),
                        flush=True,
                    )
                    args.output.write_text(json.dumps(result, indent=2) + "\n")
        if args.mode in ("smoke", "capture"):
            control.unlink(missing_ok=True)
            ids = tokenizer.apply_chat_template(
                [dict(role="user", content="What is 17 + 25? Answer briefly.")],
                tokenize=True,
                add_generation_prompt=True,
                enable_thinking=False,
                return_dict=False,
            )
            result["eos_check"] = run_wave(engine, ids, 64, 1, ignore_eos=False)
            print("EOS check:", result["eos_check"]["requests"][0]["text"], flush=True)
            result["eos_check_passed"] = (
                result["eos_check"]["requests"][0]["meta_info"]["finish_reason"]["type"]
                == "stop"
            )
            assert result["eos_check_passed"], result["eos_check"]
        control.unlink(missing_ok=True)
    result["completed"] = True
    args.output.write_text(json.dumps(result, indent=2) + "\n")


if __name__ == "__main__":
    main()
