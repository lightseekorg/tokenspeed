# PD-disaggregation simulation bench

Measures what a PD-disaggregated Kimi-K3 deployment cares about, on the
attention-DP 16 x MoE EP 16 layout (`configs/attn_dp16_moe_ep16.sh`, 4 nodes
x 4 GPUs): prefill-node throughput (P-sim, split into fresh and cached) and
decode-node throughput (D-sim), measured separately. The agentic bench in
`../tokenspeed` cannot answer this — its numbers mix prefill interference into
every decode figure.

## Data

Both sweeps build the 71-conversation SWE-smith dataset with the Kimi-K3
tokenizer (same recipe as `../tokenspeed/agentic_bench.slurm`: 50K-token first
turn, 800-token later turns, 10-15 turns) and then run
`duplicate_agentic_dataset.py` on it with its default of 32 replicas, each
with a unique `[rid:<hash>]` mark prepended to its first user message, so
replicas are distinct prefixes rather than cache hits. The result,
`agentic_dataset_x32.json` (2272 conversations, all distinct first turns), is
the only file the client reads; the 128-wide decode rung needs 128 of them.
Replicas are interleaved, so the first 71 entries are the originals.

Synthetic random-token prompts are deliberately NOT used: speculative
acceptance and MoE expert routing are content-sensitive.

Real-workload anchors (measured): fresh prefill = ~50K tokens, 1 per
conversation (~8% of requests, ~75% of prefill compute); cached prefill =
~0.8-1.3K new tokens on a 50K->67K cached prefix (~92% of requests).

## Routing under attention-DP

Every pd_client request carries `rid=pdsim-conv-<index>`. The gateway runs
`--dp-aware --sticky-sessions --policy round_robin`: each DP rank is a
routable worker, a conversation's first request is placed round-robin, and
every later request with the same rid is pinned to that rank. The engine's own
DP dispatch is load-based and each rank's cache is private, so without this a
resent prompt reaches the rank that primed it only by chance.

## P-sim: prefill-node simulation

Batch contains prefill only (`max_tokens 1` kills the decode phase). One boot
per config, fresh phase then cached phase; a warmup of one cold prefill per DP
rank on spare conversations 64-79 (excluded from every metric) absorbs
first-touch costs on every rank before the timed rungs.

**P-fresh (compute-bound):** send unique first turns cold. Ladder: parallel
1/2/4/8/16, number = 2 x parallel, offsets advance so no prompt is reused
(62 conversations). Ranking: **prefill tok/s / GPU**; secondary TTFT p50/p99.
Validity guard: cache hit <= 5%.

**P-cached (bandwidth-bound, distinct prefixes):** per conversation, prime
turn 1 with `max_tokens 500` (excluded from measurement), then measure turn 2
with `max_tokens 1`: a cached prefill of the ~50K prefix plus the ~800-token
turn increment. The prime's reasoning_content and content are passed back in
the replayed assistant turn; whatever the re-rendered turn fails to match is
recomputed and counted. Ranking: **computed tok/s / GPU = (prompt_tokens -
cached_tokens) / time**; secondary requests/s and TTFT p50/p99. Validity
guard: cache hit >= 95%.

## D-sim: decode-node simulation

A D node's defining state: the KV already exists (computed by P, delivered by
transfer). The prefix cache plays the transfer's role: a prime request puts
the KV on the rank, the measured request resends the identical first turn and
its prefill collapses to a near-full cache hit.

Each rung is its own prime-measure loop:

1. **Prime** the rung's conversations (0 .. parallel - 1) with
   `max_tokens 1` at concurrency 16, one cold prefill per DP rank.
2. **Settle 10s**: the host writebacks of the primed contexts are issued
   within a second of each request finishing, and launching the measure
   client as a job step adds about 20 s on top.
3. **Measure**: resend the same first turns, `max_tokens 2000` + `ignore_eos`,
   as one lockstep wave of exactly parallel requests. Sticky placement fixes
   each conversation's rank, so rolling admission could not hold per-rank
   concurrency constant anyway; one wave keeps every rank at parallel/16
   requests for the whole rung.
4. Rungs: parallel 16/32/64/128 (one to eight requests per rank).
   Ranking: **Output Throughput (tok/s) / GPU**; secondary TPOT p50/p99.

Why prime per rung: a finished request keeps prefix-cache blocks not only for
its prompt but for every 128 tokens it generated (one recurrent-state
checkpoint block per state group each), so a measured rung roughly triples the
footprint of the contexts it replays. Under a single up-front prime, that
growth evicts the contexts a later rung needs (observed: rung 128 missing
exactly the half rung 64 had not touched). Priming right before each rung
refreshes exactly what the rung will read.

Validity guard, recorded per rung in the collect output: **cache hit >= 95%**
on the measure wave — below it the rung is VOID (primed KV evicted, or a
request reached a rank other than the one holding its KV).

## Knobs

- Input anchor 50K (first turns); D output 2000 (decode >= 95% of measured
  wall clock at observed TPOT).
- Sampling: greedy (`temperature 0`), matching evalscope's default in the
  agentic bench, so acceptance lengths are comparable across the two. A
  rung's output throughput is output divided by the wall time set by its
  slowest request, so D-sim differences below the measured noise band need
  repeated runs; TPOT p50 is the steadier column.
- Config: `--max-num-seqs 128` (8 slots per rank), chunked prefill 8192 with
  `--gpu-memory-utilization 0.8` — the trtllm MoE workspace scales with the
  tokens gathered from all 16 ranks per step, and 16 x 8192 needs the headroom
  0.8 leaves (0.9 runs out of memory on that gather).
- P:D sizing helper in collect: give it the per-conversation token mix
  (`--mix-fresh-tokens/--mix-cached-tokens/--mix-decode-tokens`, defaults =
  the agentic anchors) and it converts the measured rates into GPU-seconds per
  conversation on each side, printing `P gpu-s/conv / D gpu-s/conv = P-GPUs
  per D-GPU`.

## Implementation notes

- evalscope's swe_smith plugin has no prime/measure phase control, so ALL
  phases run through pd_client.py (thin stdlib client, fixed concurrency). Any
  failed request aborts the sweep with a non-zero exit: a rung measured after
  a failure carries a cold prefill inside its wall time, and every later rung
  inherits the gap.
- Every summary and collect report carries the boundaries statement: no
  KV-transfer cost modeled; prime-as-transfer is the core approximation;
  single deployment; TTFT is approximated by full-request latency at
  max_tokens 1; TPOT (d-measure only) amortizes the cache-hit KV load into
  per-token time.
- The KV host tier (`kvstore`) holds finished contexts once they leave the
  device; measured hits are then host-to-device load-backs. Its pool is pinned
  host memory sized per rank, and the config caps it at `--kvstore-size 80`
  (GB, about one device arena): the default 2x ratio would pin ~162 GB per
  rank, and four ranks share one node's Slurm memory limit (880 GiB here), so
  the boot dies with a Slurm `Out Of Memory` before the server is ready. 80 GB
  holds the widest rung's primed set (8 contexts x ~26 blocks) several times
  over. The boot log's `Allocated ... compact Host L2` line reports the pool
  actually allocated.
- Server logs land in `/tmp/tokenspeed_server_disagg_<config>_<sweep-ts>.log`
  on the node running the wrapper, one file per sweep.

## Running on Slurm

```
# hold a 4-node allocation, then from the repo root (under /data):
SLURM_JOB_ID=<jobid> CONTAINER_IMAGE=<sqsh-or-ref> \
    nohup bash test/agentic_benchmark/kimi_k3/tokenspeed_disagg/d_bench.slurm > /tmp/d_bench.log 2>&1 &
python3 test/agentic_benchmark/kimi_k3/tokenspeed_disagg/collect_outputs.py \
    test/agentic_benchmark/kimi_k3/tokenspeed_disagg/outputs/p_<ts> \
    test/agentic_benchmark/kimi_k3/tokenspeed_disagg/outputs/d_<ts>
```

The wrapper is a foreground loop; run it under nohup (or sbatch it, see the
usage header) so a dropped session does not stop the sweep.

## Layout

```
configs/attn_dp16_moe_ep16.sh   # the serve config (gateway flags included)
duplicate_agentic_dataset.py    # cache-bust replicas of the 71-conversation build
p_bench.slurm                   # P sweep: boot -> warmup -> P-fresh ladder -> P-cached ladder -> kill
d_bench.slurm                   # D sweep: boot -> per rung (prime -> settle -> measure) -> kill
pd_client.py                    # phased client shared by both sweeps
collect_outputs.py              # tables + guards + P:D sizing; accepts multiple sweep dirs
outputs/p_<ts>/, outputs/d_<ts>/   # per-sweep artifacts (gitignored)
```
