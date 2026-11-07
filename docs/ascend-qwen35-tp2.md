# Qwen3.5 on two Ascend chips

The homework branch extends the verified FP16 single-chip configuration to
attention TP=2 and dense TP=2, with DP=1 and PP=1. Both ranks execute every
layer of one model. Eager execution, greedy sampling, FP16 KV/conv state and
FP32 GDN matrix state are held fixed. Prefix reuse, offload, speculation and
all-reduce/norm fusion remain disabled in this baseline.

## Environment and launch

Create a personal `.venv` with `--system-site-packages` from the course `aics`
environment. Follow the fixed installation versions in `setup.md`; build the
scheduler from this checkout. Source `scripts/ascend_env.sh` from the repository
root after explicitly choosing one or two distinct physical device IDs. For
TP=2 the experiment driver requires a same-board pair (0/1, 2/3, 4/5 or 6/7 on
the observed host). Record PCIe addresses with `npu-smi info`; do not infer the
board relationship from logical `npu:0` and `npu:1` alone.

```bash
export ASCEND_RT_VISIBLE_DEVICES=0,1 OMP_NUM_THREADS=4
source scripts/ascend_env.sh
python -m torch.distributed.run --standalone --nproc-per-node=2 \
  scripts/qwen35_hccl_check.py
python scripts/qwen35_tp_experiment.py \
  --model /path/to/Qwen3.5-0.8B --tp 2 --mode capture \
  --lengths 1 128 2048 --concurrencies 1 --output-tokens 32 \
  --warmups 0 --repeats 1 --overlap on \
  --work-dir /path/to/fresh-capture-directory --output results/tp2-capture.json
```

The HCCL backend uses `all_gather_into_tensor` and `reduce_scatter_tensor`,
which exist in the course's PyTorch 2.9. Names `all_gather_single` and
`reduce_scatter_single` are retained at TokenSpeed's backend interface only;
the newer torch aliases are not present in that environment. The real HCCL
check exercises the backend with equal and uneven token counts, including a
rank owning zero rows.

## Tensor partition and state

Use PyTorch's stored linear weight convention `[out_features, in_features]`.
For 0.8B, MLP gate/up weights `[3584,1024]` each become `[1792,1024]` on each
rank, and down `[1024,3584]` becomes `[1024,1792]`. Each rank reads the same
`[T,1024]` hidden state, computes its local SwiGLU features and produces a
`[T,1024]` partial sum. `CommManager` all-reduces it before the next dependent
residual/norm boundary. The row-parallel projections intentionally set
`reduce_results=False`; adding another reduction inside them would double
count the sum.

Full attention splits 8 Q heads into 4 per rank and 2 KV heads into 1 per
rank (head dimension 256). The Q output gate follows the same local Q heads.
Local attention output `[T,1024]` is projected through a local
`[1024,1024]` output weight into the hidden-state partial sum. Head-wise Q/K
normalization and RoPE stay local. KV history is sharded by KV head, while
positions, request identities and scheduling/page metadata are replicated.

GDN splits each packed Q/K/V/Z and B/A segment independently, not the entire
concatenation in half. Depthwise convolution channels, `A_log`, `dt_bias`,
head gating and output projection must follow the identical head ownership.
The 0.8B recurrent state `[page,16,128,128]` becomes
`[page,8,128,128]`; convolution history `[page,6144,3]` becomes
`[page,3072,3]`. GDN state remains FP32 and convolution history FP16. GDN does
not exchange its recurrent matrix after every token; its local output
projection produces a hidden-state partial sum for the layer boundary.
The LCM arena continues to allocate, reuse and free every rank's local state.

Norm weights on replicated hidden states remain replicated. Vocabulary-sharded
embedding combines masked local lookups with an all-reduce. A vocabulary-sharded
LM head computes local logits; with output logprobs enabled the measured path
gathers full logits before greedy sampling. Include those collectives in the
communication census rather than assuming exactly two collectives per layer.

## Evidence and measurement

`qwen35_probe.py` writes separate `rank-0/` and `rank-1/` directories under TP=2;
TP=1 preserves the earlier layout. Weight audits contain rank, PID, visible and
logical devices, loaded parameter counts and every stored parameter shape.
Cache audits record actual conv/recurrent state shapes. Every forward records
mode, actual batch size, token count and allocator memory. Diagnostic modes
also record collective input shapes, dtypes and bytes. These bytes describe
logical inputs, not measured physical link traffic.

Capture mode stores full logits after completed forwards, checks EOS separately,
and supports comparisons at identical generated histories using
`qwen35_tp_compare.py`. Both TP ranks must return exactly equal complete logits;
TP=1/2 differences use the predeclared whole-model gates (RMS <=0.10,
max absolute <=0.75, cosine >=0.999), recording greedy agreement separately.
`qwen35_reference.py` supplies the unmodified Transformers 5.12 CPU FP32 oracle,
including untied LM heads in larger models.

Benchmark mode submits simultaneous requests using a thread-safe Engine facade.
It computes output throughput from the sum of output tokens divided by the
whole wave's elapsed time. It performs no per-forward device synchronization,
NPU event timing or CPU logits copies. It does retain per-forward host metadata
and memory logs; that fixed observer overhead is shared by TP=1 and TP=2.
Profiler runs are separate and deliberately synchronize diagnostic boundaries.

Use input lengths 128/2048 and concurrency 1/4 for TP=1/2, fixed output length
128, three warmups and ten formal waves per configuration. Actual device batch
sizes come from forward records; client concurrency is not assumed to equal
batch size. The communication experiment compares the two concurrency settings:
measure aggregate throughput, request latency, collective payload/call counts
and exposed communication time. A larger batch can amortize launch latency
without reducing the logical bytes per token, and may increase request latency.

The timeline analysis must use interval unions for compute, communication and
their intersection. Do not sum rank times or overlapping kernel durations into
end-to-end latency. Preserve profiler traces outside Git. Report shared-host
occupancy and avoid attributing unrelated driver memory to model parameters.

## Precision regression found during validation

The causal convolution's accumulator was FP32, but each product of two half
inputs was computed in the input dtype before promotion into that accumulator.
The fix explicitly promotes both operands before an FP32 FMA, for prefill,
chain/decode and tree kernels. A cancellation case fails before the fix and
passes after it, and all 40 Ascend convolution tests pass. Stored history and
output dtypes do not change. See the convolution operator README for the exact
regression formula.

This operator fix does not by itself make every larger model satisfy the
0.8B model's strict whole-logit gates. Keep token agreement, full-logit error,
cosine and probability KL as separate measurements. A failed strict gate must
remain failed even if greedy tokens agree; the supplied comparison tool exits
nonzero for that case. Do not relax it to turn diagnostic failures into passes.
The final report lists the observed limitations explicitly.

Final experiments use the `v2` run label. The complete raw requests, logits and
traces are retained under the experiment work directory; small JSON summaries
stay in `results/`. Each run records the critical HCCL/conv/GDN source hashes
and physical chip visibility. Benchmark and profile results must be generated
from the same source hashes as the correctness captures. Profiler Level1
captures the HCCL operation breakdown; its wait and SDMA copy counters are
reported separately from the interval-union compute/communication timeline.

## Reproduce the final matrix

On the course host, the verified model directories are `/tmp/hw1-qwen35-model`
(0.8B) and `~/models/Qwen3.5-{4B,9B,27B}`. Each capture uses a fresh work directory.
For the three larger models use input length 128; the 0.8B capture also uses
lengths 1 and 2048. Capture 32 output tokens per length, then compare both runs:

```bash
python scripts/qwen35_tp_compare.py \
  --baseline /path/to/tp1-capture --parallel /path/to/tp2-capture \
  --steps 32 --output results/tp-correctness.json
```

Run the following twice for the 0.8B benchmark, selecting `TP=1` with physical
chip `0` and `TP=2` with physical chips `0,1`, and changing both output paths:

```bash
python scripts/qwen35_tp_experiment.py \
  --model /tmp/hw1-qwen35-model --tp 2 --mode benchmark \
  --lengths 128 2048 --concurrencies 1 4 --output-tokens 128 \
  --warmups 3 --repeats 10 --overlap on \
  --work-dir /path/to/fresh-benchmark --output results/tp2-benchmark.json
```

For 27B TP=2, change the model path, use 32 output tokens and 3 formal repeats.
For separate profiling use `--mode profile --output-tokens 8 --warmups 3
--repeats 1`; retain the two input lengths and two concurrency settings. Summarize
completed results with `qwen35_tp_summary.py --inputs ... --output ...` and
`qwen35_tp_profile_summary.py --work-dirs ... --output ...`.

For an OpenAI-compatible service, the explicit launcher accepts any of the four
verified model paths, TP size, unused localhost port and served model name:

```bash
ASCEND_RT_VISIBLE_DEVICES=0,1 bash scripts/serve_qwen35_tp.sh \
  /tmp/hw1-qwen35-model 2 32239 qwen35-08b
```

The 0.8B TP=2 service was checked with `qwen35_http_smoke.py` for arithmetic EOS
and Chinese continuous generation. Wait for a successful chat request before
starting this test: a reachable `/v1/models` does not by itself prove that the
gateway's asynchronous tokenizer registration has completed. The service is
stopped after verification; rerun the launcher when needed.
