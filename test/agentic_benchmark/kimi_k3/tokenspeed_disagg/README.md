# PD-disaggregation simulation bench

Measure prefill request capacity and decode token throughput separately for
Kimi-K3. `p_bench.slurm` runs one prefill conversation workload;
`d_bench.slurm` sweeps cached-prompt decode concurrency. The shared client is
`pd_client.py`. Its `TARGET_RESPONSE_TOKENS` constant (500) sets both P's
synthetic assistant length and the target output length used to normalize D
request throughput.

## Data

Both scripts use the SWE-smith dataset with 4,000-token first turns,
800-token user increments and 30–50 turns. The current build has 69 source
conversations; 32 replicas produce 2,208 conversations in
`agentic_dataset_x32.json`. Each replica has a distinct `[rid:<hash>]` prefix.
Existing dataset files are reused, so regenerate both files when changing
the dataset settings.

The dataset stores user-message increments, without assistant responses.
P inserts synthetic assistant text to model the work from previous decode
outputs. D concatenates the original user messages into one representative
long prompt, approximately 27K–43K tokens. These are workload approximations:
synthetic text can change MoE routing, and D does not replay real assistant
history.

## Routing and placement

Every request carries `rid=pdsim-conv-<index>`. The gateway uses
`--dp-aware --sticky-sessions --policy round_robin` to preserve rank-local
prefix reuse. Server steps select the first four allocated nodes for DP16,
or the first eight for DP32, with four GPU workers per node. Dataset
preparation and clients run on the selected head node.

## P: conversation replay

`p_bench.slurm` uses DP16/EP16. It warms up with 16 concurrent conversations
from indices 1024–1039, then measures 1,024 conversations from indices
0–1023 at concurrency 512.

Each worker sends one turn at a time with `max_tokens=1`, appends a prepared
500-token synthetic assistant response locally, and advances to the next
user turn. A finished conversation is immediately replaced by another from
the pool. First turns are cold; subsequent turns reuse the accumulated
prefix and normally add roughly 500 assistant tokens plus 800 user tokens,
subject to chat formatting and cache block boundaries.

Synthetic responses are seeded by conversation index, exclude special token
IDs, and are validated to re-encode to 500 tokens. All responses are prepared
before timing. P never generates or separately primes those 500 tokens.

The primary capacity is **mu_P = completed turn requests / wall time** for
the entire engine. The summary also reports computed tokens/s, cache-hit
rate and mean/p50/p99 request latency. Latency includes queueing at this
load and must not be substituted for queue-free prefill processing time.
Check batch occupancy and client/server load before treating requests/s as
saturated capacity.

## D: cached-prompt decode

Each concurrency point has a separate prime/measure sequence:

1. Concatenate all user turns for each selected conversation and prime that
   prompt with `max_tokens=1` at concurrency 16.
2. Settle for 10 seconds to allow host cache writebacks to complete.
3. Replay the identical prompts and conversation IDs with `max_tokens=2000`
   and `ignore_eos`, issuing one wave of requests at the target concurrency.

The ladder is 16/32/64/128/256/384/512. Report output tokens/s/GPU, equivalent
requests/s at the target OSL, cache-hit rate and TPOT p50/p99. Concurrency is the client setting; actual
GPU batches can vary during admission and completion. Each point is primed
again because generated-token checkpoints can evict prefixes needed later.

This simulation substitutes prefix-cache reuse for real KV transfer. Its
reported TPOT divides full request latency by output length, amortizing the
cache load and first-token work over 2,000 output tokens.

## P:D sizing

`collect_outputs.py` prints `Requests/s @ target OSL` and `Target Response
Tokens` in both tables. P uses its measured request rate; D uses output
tokens/s divided by the target response length. For example, 8,760 output
tokens/s gives 17.52 equivalent requests/s at OSL=500, while the measured
2,000-token request rate is 4.38 requests/s. Summary JSON retains the raw
measured `Requests/s` and records `Target Response Tokens` for both phases.

The sizing helper pairs P and D measurements with the same engine
configuration and target OSL. It consumes the displayed rates directly and
reports `ideal n_P = mu_D / mu_P` with one D instance. The ratio is fractional
and excludes real KV-transfer costs. Integer allocation and end-to-end
performance still need validation.

## Server settings

The current config uses `--max-num-seqs 512`, 4,096-token prefill chunks,
`--gpu-memory-utilization 0.9`, and `--kvstore-size 80` GB per rank. Sampling
is greedy. The host KV tier stores finished contexts, so a reported cache
hit may involve a host-to-device load rather than an already resident GPU
prefix. Server logs are written to
`/tmp/tokenspeed_server_disagg_<config>_<sweep-ts>.log`.

## Running on Slurm

Use an existing allocation with initialized containers. P currently needs
four nodes; the D config list includes DP32 and therefore needs eight.
Run each wrapper once, with the scripts' container settings:

```bash
SLURM_JOB_ID=<jobid> CONTAINER_IMAGE=<sqsh-or-ref> \
    bash test/agentic_benchmark/kimi_k3/tokenspeed_disagg/p_bench.slurm
SLURM_JOB_ID=<jobid> CONTAINER_IMAGE=<sqsh-or-ref> \
    bash test/agentic_benchmark/kimi_k3/tokenspeed_disagg/d_bench.slurm
python3 test/agentic_benchmark/kimi_k3/tokenspeed_disagg/collect_outputs.py \
    test/agentic_benchmark/kimi_k3/tokenspeed_disagg/outputs/p_<ts> \
    test/agentic_benchmark/kimi_k3/tokenspeed_disagg/outputs/d_<ts>
```

Warmup is excluded from collected results. Failed requests stop the run;
cache-hit rates are reported for the user to assess without filtering.

## Layout

```
configs/attn_dp16_moe_ep16.sh   # the serve config (gateway flags included)
duplicate_agentic_dataset.py    # cache-bust replicas of the source conversations
p_bench.slurm                   # P run: boot -> warmup -> conversation replay -> stop
d_bench.slurm                   # D sweep: boot -> per rung (prime -> settle -> measure) -> kill
pd_client.py                    # phased client shared by both sweeps
collect_outputs.py              # tables + P:D sizing; accepts multiple sweep dirs
outputs/p_<ts>/, outputs/d_<ts>/   # per-sweep artifacts (gitignored)
```
