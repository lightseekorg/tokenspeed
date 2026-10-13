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

"""Run one engine over a fixed prompt on a truncated HY V4 and dump its layers.

Each engine gets its own subprocess (``--engine tokenspeed|reference``) because
both install a variant of the same runtime in site-packages and cannot coexist
in one interpreter. The parent ``hy_v4_parity.py`` orchestrates the two runs and
compares the dumps.

The reference engine is vLLM 0.29.0, the first release that ships
``vllm/models/hy_v4``; point ``--reference-python`` at the interpreter that can
import it, and ``--reference-repo`` at its checkout.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

# The prompt is fixed so both engines see identical tokens; the values are
# arbitrary but cover multiple byte-lengths and a repeated stretch, which is
# what a mis-sliced rope or a broken sink would react to differently.
PROMPT_IDS = [
    15496,
    995,
    11,
    314,
    716,
    257,
    1712,
    13,
    1,
    1,
    1,
    257,
    1712,
    13,
    1111,
    42,
    351,
    217,
    1049,
    314,
    716,
    257,
    1712,
    13,
    2447,
    25,
    1566,
    481,
    314,
    716,
]
DECODE_STEPS = int(os.environ.get("HY_V4_DECODE_STEPS", "4"))


def run_tokenspeed(args, dump_path: Path) -> dict:
    """Boot TokenSpeed's engine and dump the truncated model.

    Instrumentation wraps the model class inside whichever process constructs it
    (engine workers are separate processes), so the patch is installed here and
    fires there.
    """
    from test.hy_v4_parity.hy_v4_dump import instrument_hy_v4_classes

    from tokenspeed.runtime.entrypoints.engine import Engine

    instrumented = instrument_hy_v4_classes(
        path=dump_path,
        num_layers=args.num_layers,
        tags=["engine=tokenspeed", f"backend={args.attention_backend}"],
    )
    if not instrumented:
        raise SystemExit(
            "TokenSpeed's HY V4 model class was not importable in this "
            "interpreter; run with the TokenSpeed environment."
        )
    print(f"[tokenspeed] instrumented {instrumented}")

    engine = Engine(
        model=args.model_path,
        load_format=args.load_format,
        attention_backend=args.attention_backend,
        quantization=args.quantization,
        dtype="bfloat16",
        # Eager only: the comparison wants bit-comparable kernels, not graph replay.
        enforce_eager=True,
        skip_tokenizer_init=True,
        enable_prefix_caching=False,
        disable_kvstore=True,
        max_model_len=args.max_model_len,
        max_num_seqs=4,
        gpu_memory_utilization=args.gpu_memory_utilization,
        enable_output_logprobs=True,
        log_level="warning",
        **json.loads(os.environ.get("HY_V4_TS_ENGINE_KWARGS", "{}")),
    )
    try:
        output = engine.generate(
            input_ids=PROMPT_IDS,
            sampling_params={"temperature": 0.0, "max_new_tokens": DECODE_STEPS},
            return_logprob=True,
        )
    finally:
        engine.shutdown()
    result = output if isinstance(output, dict) else output[0]
    return {
        "output_ids": [int(t) for t in result["output_ids"]],
        "logprobs": [
            float(entry[0]) for entry in result["meta_info"]["output_token_logprobs"]
        ],
    }


def run_reference(args, dump_path: Path) -> dict:
    """Boot the reference vLLM engine and dump the same taps.

    vLLM is imported from ``--reference-repo``. Its HY V4 model
    names the sub-blocks the same way TokenSpeed does, so the same instrumentation
    attaches without a second hook implementation.
    """
    reference_repo = Path(args.reference_repo)
    if not reference_repo.is_dir():
        raise SystemExit(
            f"--reference-repo {reference_repo} is not a directory; point it at "
            "the checkout that ships vllm/models/hy_v4"
        )
    sys.path.insert(0, str(reference_repo))

    from test.hy_v4_parity.hy_v4_dump import instrument_hy_v4_classes

    from vllm import LLM, SamplingParams

    instrumented = instrument_hy_v4_classes(
        path=dump_path,
        num_layers=args.num_layers,
        tags=["engine=reference", f"backend={args.reference_attention_backend}"],
    )
    if not instrumented:
        raise SystemExit(
            f"no HY V4 model class was importable under {reference_repo}; "
            "confirm the checkout actually contains vllm/models/hy_v4"
        )
    print(f"[reference] instrumented {instrumented}")

    llm = LLM(
        model=args.model_path,
        tokenizer=args.model_path,
        enforce_eager=True,
        # Sink-capable sparse MLA is the backend the reference binds for HY V4.
        attention_backend=args.reference_attention_backend,
        max_model_len=args.max_model_len,
        gpu_memory_utilization=args.gpu_memory_utilization,
        dtype="bfloat16",
        disable_log_stats=True,
        # The truncated checkpoint carries no tokenizer of its own.
        skip_tokenizer_init=False,
    )
    outputs = llm.generate(
        [{"prompt_token_ids": PROMPT_IDS}],
        SamplingParams(temperature=0.0, max_tokens=DECODE_STEPS, logprobs=1),
    )

    completion = outputs[0].outputs[0]
    logprobs = [
        float(next(iter(step_logprobs.values())).logprob)
        for step_logprobs in (completion.logprobs or [])
    ]
    return {
        "output_ids": [int(t) for t in completion.token_ids],
        "logprobs": logprobs,
    }


def write_summary(path: Path, summary: dict) -> None:
    import json

    path.write_text(json.dumps(summary, indent=2))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--engine",
        choices=("tokenspeed", "reference"),
        required=True,
        help="Which engine this process runs.",
    )
    parser.add_argument("--model-path", required=True, help="Truncated checkpoint.")
    parser.add_argument("--dump-path", required=True, help="Layer dump output.")
    parser.add_argument("--summary-path", required=True, help="Run summary output.")
    parser.add_argument("--num-layers", type=int, default=3)
    parser.add_argument("--max-model-len", type=int, default=2048)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.4)
    parser.add_argument(
        "--load-format",
        default="auto",
        help="TokenSpeed load format; use 'dummy' to skip weight loading.",
    )
    parser.add_argument(
        "--quantization",
        default=None,
        help="Optional TokenSpeed quantization (e.g. fp8) to exercise the "
        "online-quantized path.",
    )
    parser.add_argument(
        "--attention-backend",
        default="dsa",
        help="TokenSpeed attention backend. 'dsa' is the production path; "
        "'trtllm_mla'/'tokenspeed_mla' exercise the dense fallback.",
    )
    parser.add_argument(
        "--reference-repo",
        default=os.environ.get("HY_V4_REFERENCE_REPO", ""),
        help="vLLM checkout (reference engine only).",
    )
    parser.add_argument(
        "--reference-attention-backend",
        default=os.environ.get("HY_V4_REFERENCE_ATTENTION_BACKEND", "FLASHMLA_SPARSE"),
        help="Reference engine's attention backend for the sink-capable MLA path.",
    )
    args = parser.parse_args()

    dump_path = Path(args.dump_path)
    summary_path = Path(args.summary_path)
    if args.engine == "tokenspeed":
        summary = run_tokenspeed(args, dump_path)
    else:
        summary = run_reference(args, dump_path)
    write_summary(summary_path, summary)
    print(f"[{args.engine}] output_ids={summary['output_ids']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
