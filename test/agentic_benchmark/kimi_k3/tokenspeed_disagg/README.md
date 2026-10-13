# Kimi-K3 P/D benchmarks

Measure prefill and decode capacity separately, or run a complete 1P1D
benchmark with KV transfer.

| Script | Workload | Nodes (4 GPUs each) |
| --- | --- | --- |
| `agentic_bench.slurm` | 1P1D, both ADP16/EP16 | 8 |
| `p_bench.slurm` | Prefill simulation, ADP16/EP16 | 4 |
| `d_bench.slurm` | Decode simulation, TP8/TP8 then ADP16/EP16 and ADP32/EP32 | 8 |

## Running

From this directory, use an existing allocation with initialized `ts`
containers. Run one launcher at a time:

```bash
SLURM_JOB_ID="$JOBID" \
CONTAINER_IMAGE=/path/to/tokenspeed-dev.sqsh \
./agentic_bench.slurm
```

Use the same invocation with `p_bench.slurm` or `d_bench.slurm` for the
separate simulations. Server settings live in `configs/`. `CONTAINER_NAME`
and `CONTAINER_MOUNTS` override the container name and mounts; the defaults
are `ts` and `/data:/data`. For 1P1D, `DISAGGREGATION_IB_DEVICE` defaults to
`mlx5_0,mlx5_1,mlx5_2,mlx5_3`. Override it for a different NIC layout.

## Dataset

The scripts install a pinned EvalScope revision and use its SWE-smith
builder to create `agentic_dataset_x128.json`: 4,000-token first turns,
800-token user increments, 30–50 turns, and 128 replicas. Replicas have
unique prompt prefixes and conversation IDs. The file is reused on later
runs; regenerate it when changing dataset settings.

## End-to-end 1P1D

`agentic_bench.slurm` launches prefill on the first four nodes and decode
on the next four. Both use `configs/attn_dp16_moe_ep16.sh --grpc`, with
Slurm step topology detected automatically. Mooncake transfers KV caches
between them. One SMG gateway uses DP-aware sticky sessions with
round-robin assignment (`delegate`) and EvalScope conversation IDs.

After a separate 16-conversation warmup, the client sweeps concurrency
16/32/64/128/256/384/512 with two conversations per slot. Each turn generates
500 tokens with greedy sampling.

Normalize throughput by **32 GPUs**. Total token throughput includes cached
input tokens; distinguish it from output-only throughput.

## Separate P/D simulations

`pd_client.py` uses `TARGET_RESPONSE_TOKENS=500` for synthetic assistant
responses and decode request-rate normalization. Both simulations use
sticky conversation IDs for prefix reuse.

- **Prefill:** warm up with 16 conversations, then replay 1,024 conversations
  at concurrency 512. Each turn generates one token; the client appends
  500 synthetic assistant tokens before the next user turn.
- **Decode:** sweep concurrency 1/2/4/8/16/32 for TP8, 16/32/64/128/256/384/512
  for ADP16, and 32/64/128/256/384/512/768/1024 for ADP32. TP8 runs
  4/8/8/16/32/64 conversations respectively; ADP runs one conversation per
  concurrent slot. TP8 uses the first two nodes, ADP16 the first four, and
  ADP32 all eight. At each point,
  concatenate all stored user turns, prime at concurrency 16 with the last
  token removed after chat templating, then wait 10 seconds and generate 2,000
  tokens from each full prompt. The shorter prime works around
  aligned KDA checkpoint misses. Assistant responses are not included.

These simulations exclude real KV-transfer costs. Prefill latency includes
queueing. Decode requests stream output: TTFT measures the wait for the first
content or reasoning chunk; TPOT is the time from the first to the last output
chunk divided by output tokens minus one. Throughput uses the entire measured
run's wall time, including startup and completion tails. Cache hits can involve the
host KV tier.

Results are under `outputs/p_<timestamp>/` and `outputs/d_<timestamp>/`;
server logs are `/tmp/tokenspeed_server_disagg_<config>_<timestamp>.log`.
Collect both sweeps together:

```bash
python3 collect_outputs.py outputs/p_<timestamp> outputs/d_<timestamp>
```

The collector excludes warmup and reports cache-hit rates without filtering.
`Requests/s @ target OSL` is the measured P request rate or D output tokens/s
divided by 500. For matching configs, it estimates `n_P = mu_D / mu_P` with
one D instance. This fractional ratio excludes transfer costs. The collector
handles the simulations only; use EvalScope reports for the 1P1D benchmark.
