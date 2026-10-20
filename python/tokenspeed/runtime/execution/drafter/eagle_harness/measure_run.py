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

"""EAGLE-Harness cost measurement driver.

    python -m tokenspeed.runtime.execution.drafter.eagle_harness.measure_run \\
        --model <target> --draft-model <EAGLE-4 draft> --run-dir <dir> \\
        --tp-size 4 --compute-intensity chain_intensity --topk 1 \\
        --num-steps 15 --num-draft-tokens 16 --max-bs 8 --mamba-ssm-dtype float32

For each sampled batch size the driver writes ``config/measure_config.json``,
serves that many identical greedy requests on an exec-mode-2 engine (whose
rounds record the draft and verify stage samples, see :mod:`measurement`),
then times one attention layer of the target's verify and of the draft's
lanes over the context range with the kernels serving runs
(:func:`attention_samples`), and processes everything into
``cost/bs_<n>/model_{slm,llm}`` (:mod:`cost`). A batch whose averages do not
converge is measured again (three retries).

Serving reads the run directory as its ``--speculative-data-path``, with the
same draft shape, TP size, attention backends and state dtype.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys

import torch
from tokenspeed_kernel.ops.attention.mha.flashinfer import (
    trtllm_batch_decode_with_kv_cache,
)
from tokenspeed_kernel.ops.attention.tree import tree_window_attention

from tokenspeed.runtime.execution.drafter.eagle_harness.cost import (
    CHAIN_RULE_MAX_BATCH_SIZE,
    COST_BATCH_SIZES,
    MeasureRecord,
    max_batch_size,
    process_cache,
    process_model,
    process_query,
    save_records,
)
from tokenspeed.runtime.execution.drafter.eagle_harness.measurement import (
    draft_widths,
    verify_widths,
)
from tokenspeed.runtime.layers.attention.backends.paged.trtllm import (
    canonicalize_stride,
)
from tokenspeed.runtime.layers.attention.kernel_page_sizes import TRTLLM_MHA_PAGE_SIZE

# Tokens a measured request may emit; the engine ends the batch with an EOS well before.
_MAX_NEW_TOKENS = 1024
_WORKSPACE_BYTES = 256 << 20
# The fixed prompt every measured request starts from.
_PROMPT = (
    "Please write a detailed, step-by-step explanation of how a modern operating "
    "system schedules threads across CPU cores, covering run queues, priorities, "
    "time slices, preemption, load balancing between cores, and how I/O-bound "
    "and CPU-bound workloads are treated differently. "
) * 8


def attention_samples(
    *,
    batch_size: int,
    rows: int,
    window: int,
    num_q_heads: int,
    num_kv_heads: int,
    head_dim: int,
    sliding_window: int,
    context_step: int,
    context_stop: int,
    fast: bool,
    repeat: int,
    widths: list[int],
    category: str,
) -> list[MeasureRecord]:
    """One attention layer of the decode tree cascade over the context range:
    ``rows`` query rows per request against its ``kv_len``-key prefix (the
    trtllm-gen decode; a sliding window decodes each row alone over the
    window ending there) plus its ``window``-key tree window.

    Serving forwards all ``rows`` whatever the round's width, so one timing
    per context stands for every swept width.

    Returns:
        A sample per (context, width): microseconds over ``repeat`` layers.
    """
    device = torch.device("cuda")
    dtype = torch.bfloat16
    page = TRTLLM_MHA_PAGE_SIZE
    interval = context_step // (16 if fast else 32)
    lengths = [
        kv for kv in range(context_step // 2, context_stop + 1, interval)
        if kv >= context_step
    ]  # fmt: skip
    pages_per_req = math.ceil((context_stop + window) / page)
    num_pages = batch_size * pages_per_req
    k_rows = torch.randn(
        (num_pages * page, num_kv_heads, head_dim), device=device, dtype=dtype
    )
    v_rows = torch.randn_like(k_rows)

    def paged(rows_: torch.Tensor) -> torch.Tensor:
        cache = rows_.view(-1, page, num_kv_heads, head_dim).permute(0, 2, 1, 3)
        return canonicalize_stride(cache) if num_kv_heads == 1 else cache

    k_cache, v_cache = paged(k_rows), paged(v_rows)
    page_table = torch.arange(num_pages, device=device, dtype=torch.int32).view(
        batch_size, pages_per_req
    )
    q = torch.randn(
        (batch_size * rows, num_q_heads, head_dim), device=device, dtype=dtype
    )
    workspace = torch.zeros(_WORKSPACE_BYTES, dtype=torch.uint8, device=device)
    scale = head_dim**-0.5
    causal = sliding_window <= 0
    # Row r sees its path's window keys: a chain for verify rows, one ancestor per depth for lanes.
    if causal:
        bits = [(1 << (r + 1)) - 1 for r in range(rows)]
    else:
        bits = [
            sum(1 << (d * rows + r) for d in range(window // rows)) for r in range(rows)
        ]
    # Window key j is bit j % 64 of the row's word j // 64; bit 63 is the sign.
    words = [
        (b >> (64 * w)) & ((1 << 64) - 1)
        for b in bits
        for w in range((window + 63) // 64)
    ]
    mask = torch.tensor(
        [x - (1 << 64) if x >= 1 << 63 else x for x in words] * batch_size,
        dtype=torch.int64,
        device=device,
    )
    samples = []
    for kv_len in lengths:
        prefix = torch.full((batch_size,), kv_len, dtype=torch.int32, device=device)
        seq_lens = prefix + window
        if causal:
            tables, prefix_lens, q_len = page_table, prefix, rows
        else:
            tables = page_table.repeat_interleave(rows, dim=0)
            prefix_lens, q_len = prefix.repeat_interleave(rows), 1

        def layer() -> torch.Tensor:
            out, lse = trtllm_batch_decode_with_kv_cache(
                query=q,
                kv_cache=(k_cache, v_cache),
                workspace_buffer=workspace,
                block_tables=tables,
                seq_lens=prefix_lens,
                max_seq_len=context_stop + window,
                bmm1_scale=scale,
                bmm2_scale=1.0,
                out_dtype=dtype,
                q_len_per_req=q_len,
                window_left=-1 if causal else sliding_window - 1,
                return_lse=True,
            )
            return tree_window_attention(
                q, k_rows, v_rows, page_table, seq_lens, mask, out, lse,
                rows_per_req=rows, window=window, causal_prefix=causal,
                page_size=page, sm_scale=scale,
            )  # fmt: skip

        layer()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            for _ in range(repeat):
                layer()
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        graph.replay()
        end.record()
        end.synchronize()
        elapsed = max(int(start.elapsed_time(end) * 1000), 1)
        samples += [
            MeasureRecord(elapsed, repeat, width, kv_len, category, batch_size)
            for width in widths
        ]
    return samples


def _attention_config(config) -> tuple[int, int, int, int]:
    """(query heads, key/value heads, head dim, sliding window) of a model's
    full-attention layers."""
    text = getattr(config, "text_config", None) or config
    heads = text.num_attention_heads
    kv_heads = text.num_key_value_heads
    head_dim = getattr(text, "head_dim", None) or text.hidden_size // heads
    window = getattr(text, "sliding_window", None) or 0
    return heads, kv_heads, head_dim, window


def _full_attention_layers(config) -> int:
    """A hybrid model's softmax-attention layers, or every layer."""
    text = getattr(config, "text_config", None) or config
    # Nemotron-H's layer_types name block kinds, so a hybrid config's full_attention_layer_ids wins.
    layer_ids = getattr(text, "full_attention_layer_ids", None)
    if layer_ids is not None:
        return len(layer_ids)
    types = getattr(text, "layer_types", None)
    if types is None:
        return text.num_hidden_layers
    return sum(t == "full_attention" for t in types)


def measure_batch(args, engine, input_ids, batch_size, target_cfg, draft_cfg) -> bool:
    """Measure one batch size; whether its averages converged."""
    ci = args.compute_intensity
    # The verify window: the widest swept verify width.
    verify_num = max(verify_widths(batch_size, ci))
    config_dir = os.path.join(args.run_dir, "config")
    os.makedirs(config_dir, exist_ok=True)
    with open(os.path.join(config_dir, "measure_config.json"), "w") as f:
        json.dump({"batch_size": batch_size}, f)
    engine.generate(
        input_ids=[input_ids] * batch_size,
        sampling_params={"temperature": 0.0, "max_new_tokens": _MAX_NEW_TOKENS},
    )
    test = os.path.join(args.run_dir, "test", f"bs_{batch_size}")
    cost = os.path.join(args.run_dir, "cost", f"bs_{batch_size}")
    drafts = draft_widths(batch_size, ci, args.topk)
    verifies = sorted(
        w for w in verify_widths(batch_size, ci) if w <= args.num_draft_tokens
    )
    heads, kv_heads, head_dim, _ = _attention_config(target_cfg)
    d_heads, d_kv_heads, d_head_dim, d_window = _attention_config(draft_cfg)
    tp = args.tp_size
    repeat = 8 if args.fast else 16
    if args.context_stop >= 16384:
        repeat = max(repeat // 2, 1)
    for kind, widths, rows, window, shape, sliding in (
        ("llm", verifies, args.num_draft_tokens, args.num_draft_tokens,
         (heads, kv_heads, head_dim), 0),
        ("slm", drafts, args.topk, (args.num_steps - 1) * args.topk,
         (d_heads, d_kv_heads, d_head_dim), d_window),
    ):  # fmt: skip
        records = attention_samples(
            batch_size=batch_size, rows=rows, window=window,
            num_q_heads=shape[0] // tp, num_kv_heads=max(1, shape[1] // tp),
            head_dim=shape[2], sliding_window=sliding, context_step=args.context_step,
            context_stop=args.context_stop, fast=args.fast, repeat=repeat,
            widths=widths, category=f"cache_{kind}",
        )  # fmt: skip
        save_records(test, records)
    rates = []
    for kind, widths, max_q, layers, window_layers, sliding in (
        ("slm", drafts, args.topk, 1, 1 if d_window else 0, d_window),
        ("llm", verifies, verify_num, _full_attention_layers(target_cfg), 0, 0),
    ):
        rates.append(process_cache(
            os.path.join(test, f"cache_{kind}"), os.path.join(cost, f"cache_{kind}"),
            widths=widths, max_query_length=max_q, cache_length_step=args.context_step,
            cache_length_stop=args.context_stop, batch_size=batch_size, fast=args.fast,
        ) < (0.20 if args.fast else 0.25))  # fmt: skip
        rates.append(process_query(
            os.path.join(test, f"model_{kind}"), os.path.join(cost, f"cache_{kind}"),
            os.path.join(cost, f"query_{kind}"), widths=widths,
            cache_length_step=args.context_step, attention_layers=layers,
            batch_size=batch_size, loose=kind == "slm" and (not args.fast or tp > 1),
            window_layers=window_layers, window=sliding,
        ) < 0.30)  # fmt: skip
        process_model(
            os.path.join(cost, f"query_{kind}"), os.path.join(cost, f"cache_{kind}"),
            os.path.join(cost, f"model_{kind}"), widths=widths,
            max_query_length=max_q, cache_length_step=args.context_step,
            attention_layers=layers, batch_size=batch_size,
            window_layers=window_layers, window=sliding,
        )  # fmt: skip
    return all(rates)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--model", required=True)
    parser.add_argument("--draft-model", required=True)
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--tp-size", type=int, required=True)
    parser.add_argument(
        "--compute-intensity",
        required=True,
        choices=["high_intensity", "low_intensity", "chain_intensity"],
    )
    parser.add_argument(
        "--topk", type=int, required=True, help="draft children per node"
    )
    parser.add_argument("--num-steps", type=int, required=True, help="draft depth")
    parser.add_argument(
        "--num-draft-tokens",
        type=int,
        required=True,
        help="verify window (the widest swept verify width)",
    )
    parser.add_argument("--start-bs", type=int, default=1)
    parser.add_argument("--max-bs", type=int, required=True)
    parser.add_argument(
        "--mamba-ssm-dtype",
        required=True,
        choices=["float32", "bfloat16"],
        help="linear-attention state dtype, as serving runs it (it picks the "
        "state's verify kernels)",
    )
    parser.add_argument(
        "--gpu-memory-utilization",
        type=float,
        default=0.85,
        help="the engine's share of each GPU; the rest leaves room for the "
        "attention microbenchmark on rank 0's GPU",
    )
    parser.add_argument("--context-step", type=int, default=64)
    parser.add_argument("--context-stop", type=int, default=8192)
    parser.add_argument(
        "--fast",
        action="store_true",
        help="fewer attention repetitions and context points, checked against "
        "a stricter convergence bar",
    )
    args = parser.parse_args(argv)  # fmt: skip
    if args.max_bs > min(
        max_batch_size(args.compute_intensity), CHAIN_RULE_MAX_BATCH_SIZE
    ):
        raise SystemExit(f"--max-bs above the {args.compute_intensity} batch cap")

    from transformers import AutoConfig, AutoTokenizer

    from tokenspeed.runtime.configs.model_config import get_hf_text_config
    from tokenspeed.runtime.entrypoints.engine import Engine
    from tokenspeed.runtime.utils.hf_transformers_utils import get_config

    # Read as serving does: a multimodal checkpoint's config class may not load without remote code.
    target_cfg = get_hf_text_config(get_config(args.model, trust_remote_code=True))
    draft_cfg = AutoConfig.from_pretrained(args.draft_model, trust_remote_code=True)
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    input_ids = list(tokenizer(_PROMPT, add_special_tokens=False)["input_ids"])
    engine = Engine(
        model=args.model, attn_tp_size=args.tp_size, dtype="bfloat16",
        max_model_len=max(4096, len(input_ids) + _MAX_NEW_TOKENS + 2 * args.num_draft_tokens),
        max_num_seqs=args.max_bs, attention_backend="trtllm", sampling_backend="greedy",
        disable_kvstore=True, speculative_algorithm="EAGLE4",
        speculative_draft_model_path=args.draft_model, speculative_eagle_topk=args.topk,
        speculative_num_steps=args.num_steps, speculative_num_draft_tokens=args.num_draft_tokens,
        speculative_exec_mode=2, speculative_data_path=args.run_dir,
        speculative_compute_intensity=args.compute_intensity, log_level="warning",
        mamba_ssm_dtype=args.mamba_ssm_dtype,
        gpu_memory_utilization=args.gpu_memory_utilization,
    )  # fmt: skip
    try:
        for batch_size in COST_BATCH_SIZES:
            if not args.start_bs <= batch_size <= args.max_bs:
                continue
            for attempt in range(4):
                if measure_batch(
                    args, engine, input_ids, batch_size, target_cfg, draft_cfg
                ):
                    print(f"[measure] batch {batch_size}: done", flush=True)
                    break
                if attempt < 3:
                    print(
                        f"[measure] batch {batch_size}: unstable, retry {attempt + 1}/3",
                        flush=True,
                    )
            else:
                raise SystemExit(f"batch {batch_size} stayed unstable")
    finally:
        engine.shutdown()


if __name__ == "__main__":
    sys.exit(main())
