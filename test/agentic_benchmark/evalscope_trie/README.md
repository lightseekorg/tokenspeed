# EvalScope Trie Agentic Benchmark Protocol

This directory contains backend-neutral tooling for running EvalScope trie
agentic workloads against an already-running OpenAI-compatible serving endpoint.
It is intended for TokenSpeed/TRT-LLM/VLLM cross-stack comparisons where server
launch, model placement, and backend-specific flags are controlled separately.

## Scope

The protocol fixes the client-side benchmark shape and metric definitions:

- workload: EvalScope `trie_agentic_coding`, `trie_code_qa`, or
  `trie_office_work`
- endpoint: OpenAI-compatible `/v1/chat/completions` or `/v1/completions`
- generation: streaming, multi-turn, `max_tokens=500`, greedy temperature
- warmup: one small sanity warmup and one high-concurrency warmup
- formal sweep: fixed request counts per concurrency
- output: CSV rows plus a throughput-vs-per-user-speed SVG

The server process is not launched by this protocol. Start TokenSpeed, TRT-LLM,
or another backend first, wait until it is healthy, then run the protocol against
that endpoint.

Backend-specific launch flags, model sharding, KV cache sizing, and profiler
wrapping remain outside this directory. Record those settings next to each run so
TokenSpeed and reference-backend results can be compared with the same client
workload but their own serving commands.

## Why This Exists

Earlier agentic scripts in this repository were tied to a specific backend and
the SWE-Smith dataset. For V4-Pro work, this protocol makes the client side
identical when comparing TokenSpeed and a reference backend:

- same trie workload file
- same request parameters and request count per concurrency
- same prompt serialization
- same warmup sequence
- same temperature, streaming, and `ignore_eos` settings
- same metric formulas

This keeps perf discussions focused on serving/runtime behavior instead of
benchmark harness drift.

## Run

Start a server separately, then run:

```bash
cd test/agentic_benchmark/evalscope_trie

MODEL=DeepSeek-V4-Pro \
API_URL=http://127.0.0.1:8000/v1/chat/completions \
TOKENIZER_PATH=/path/to/tokenizer \
DATASET=trie_agentic_coding \
DATASET_PATH=/path/to/agentic_coding_8k.jsonl \
NUM_GPUS=8 \
./run_evalscope_trie_sweep.sh
```

The default sweep is:

```text
warmup 1: parallel=2, number=2
warmup 2: parallel=8, number=16
formal:  parallel=1/2/4/8, number=8/16/32/64
```

Override with environment variables:

```bash
FORMAL_PARALLELS="1 2 4 8 16" \
FORMAL_NUMBERS="8 16 32 64 128" \
./run_evalscope_trie_sweep.sh
```

## Prompt Serialization

DeepSeek-V4-Pro model snapshots may not include a usable chat template. When a
backend cannot accept `/v1/chat/completions` with templated messages, use a plain
prompt path consistently across all compared stacks:

```bash
API_URL=http://127.0.0.1:8001/v1/completions \
TOKENIZE_PROMPT=1 \
./run_evalscope_trie_sweep.sh
```

Only compare TokenSpeed and reference results if both sides use the same prompt
serialization, tokenizer path, and temperature/sampling settings.

## Collect Existing Results

```bash
python3 collect_outputs.py outputs/<timestamp> \
  --num-gpus 8 \
  --csv sweep.csv \
  --svg sweep.svg
```

The collector reads each `parallel_<P>_number_<N>` directory emitted by
EvalScope and computes:

- `steady_completion_tok_s`: `workload_throughput.json`, row
  `Completion tok/s`, field `steady_state`
- `completion_tps_per_user`: `steady_completion_tok_s / parallel`
- `output_token_min_per_gpu`: `steady_completion_tok_s * 60 / num_gpus`
- `decoded_tok_iter` and `spec_accept_rate`: `benchmark_summary.json`
- `success`: `Success Requests / Total Requests`

The SVG uses the same axes used in the V4-Pro comparison discussions:

- X axis: `completion_tps_per_user`
- Y axis: `output_token_min_per_gpu`

Treat the CSV as the source of truth. The SVG is a convenience view for quickly
checking the throughput/per-user-speed tradeoff.
