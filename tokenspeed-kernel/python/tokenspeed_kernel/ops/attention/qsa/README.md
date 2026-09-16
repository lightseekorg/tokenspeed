# QSA block scoring

The materialized `qwen4_exp_qsa_block_topk` solution scores compressed K blocks
with `sum_heads(relu(Q @ K))`, then selects blocks independently for each query.
Its Triton scoring kernel uses 128-block tiles and four warps per CTA.

`queries_per_request` describes row layout, not a separate execution mode:

- A positive integer means consecutive runs of that many query rows belong to
  one request. It must divide the total row count. Request IDs may appear in any
  order, but each run must have one owner.
- `None` means ragged or mixed queries, so every query is scored independently.
- The kernel groups up to four queries using a power-of-two divisor of the
  uniform width. The grouped, padded head count is capped at 16. For four heads,
  widths 1/2/3/4/6/8 therefore use groups of 1/2/1/4/2/4.

A group loads one K tile and places its queries along the matrix multiply's M
dimension. Each query retains its own head reduction, complete-block frontier,
logits row and top-k selection. K loads use the group's maximum frontier, while
output masks use each query's frontier. Invalid scores, including every score
of a padded request, are still written as `-inf` on every invocation.

The same kernel handles group size one and larger groups. Strided Q views and
page tables remain supported. Uniform widths come from the existing runtime
query-layout helper; no device-to-host length reads or new cache state are
needed. Eager and captured forwards use the same arguments. PDL still waits
before reading query/cache/metadata and releases successor setup after the wait.

For four 128-dimensional BF16 heads, the old 512-block tile used 132,096 bytes
of shared memory, allowing only one CTA per B300 SM. A 128-block tile with four
queries uses 36,864 bytes, while sharing K avoids four independent reads and
uses 16 rows of the matrix multiply. This reduces resource pressure without
changing scoring or selection semantics. The streaming solution keeps its
existing score-and-select implementation.

## Isolated B300 measurements

The following comparison uses four BF16 heads of dimension 128, a strided Q
view, random physical page mappings, 20,544 output blocks, and PDL enabled.
Effective contexts range from about 51K to 61K tokens; BS1 uses 55,523 tokens.
The baseline scores one query per 512-block tile with eight warps. Timings are
microseconds per scoring call, with identical inputs for both implementations.

| Batch | Queries/request | Baseline, repeated input | Grouped/tile128, repeated input | Baseline, cache flush | Grouped/tile128, cache flush |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 1 | 4.10 | 2.41 | 12.29 | 8.19 |
| 1 | 4 | 6.61 | 2.69 | 12.29 | 8.19 |
| 32 | 1 | 46.34 | 30.65 | 59.39 | 43.01 |
| 32 | 4 | 157.90 | 31.75 | 167.94 | 40.96 |

Repeated-input timings use 100 calls per CUDA Graph and the median of 15
replays over three alternating-order rounds. Cache-flush timings write a
256 MiB buffer before each call and take the median of 108 samples, excluding
the buffer write. The latter include CUDA event boundaries and show timing
quantization; no hardware counters establish the actual cache miss rate.
Outputs match the baseline bit for bit. These measurements cover the scoring
kernel only, not complete inference latency or throughput.

## Tests

Regression coverage lives in `test/ops/test_qwen4_exp_qsa.py`: uniform and ragged
layouts, odd widths, padded requests, non-power-of-two head shapes, frontier
boundaries, strided inputs and metadata changes during CUDA Graph replay.
`test/nvidia/ops/test_qsa_pdl.py` additionally checks producer/consumer ordering
with PDL both enabled and disabled.
