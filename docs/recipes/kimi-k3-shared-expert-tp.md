# Kimi-K3 DEP16 with sharded shared experts

This recipe shards only the BF16 shared-expert MLP. Attention and caches remain
TP1/DP16, and routed experts remain TP1/EP16. Use real checkpoint weights and
the full model by default; reduced-depth performance tests are not full-model
capacity or accuracy results.

Set `TOKENSPEED_KIMI_K3_SHARED_EXPERT_TP_SIZE` to a positive divisor strictly
smaller than world size. `1` disables sharding; DEP16 supports TP2, TP4 and TP8.
The shared MLP intermediate width must also be divisible by the selected TP
size. The command below uses TP4 as an example.

## Allocation and launch

On Slurm, reserve a persistent allocation first, using your account, partition,
and fabric-placement constraints. For a four-GPU-per-node deployment:

```bash
salloc --account ACCOUNT --partition PARTITION --nodes 4 --exclusive \
  --ntasks-per-node 1 --gpus-per-node 4 --time 02:00:00
```

Run the serving command through your site's `submit` wrapper or `srun` inside
that allocation, with the same container, checkpoint mounts, and environment
on all nodes. Use one serving launcher per node and four workers per launcher.
Replace `HEAD_NODE` with the allocated head node, not localhost.

```bash
export TOKENSPEED_KIMI_K3_SHARED_EXPERT_TP_SIZE=4
python -m tokenspeed.cli serve \
  --model MODEL_DIR --quantization nvfp4 --dtype bfloat16 \
  --world-size 16 --nprocs-per-node 4 \
  --attn-tp-size 1 --data-parallel-size 16 --dense-tp-size 1 \
  --moe-tp-size 1 --expert-parallel-size 16 \
  --all2all-backend flashinfer --moe-backend flashinfer_trtllm \
  --attention-backend tokenspeed_mla --kda-backend cutedsl_kda \
  --dist-init-addr HEAD_NODE:29500 \
  --gpu-memory-utilization 0.83 --max-num-seqs 512 \
  --chunked-prefill-size 8192 --max-prefill-tokens 8192 \
  --prefix-granularity 128 --enable-prefix-caching \
  --max-cudagraph-capture-size 32 --cudagraph-capture-sizes 32
```

For the DEP16 baseline, set the environment variable to `1`. Keep all other
settings fixed. Here `max-num-seqs=512` means 32 requests per rank. Increase
both the global sequence budget and graph batch size together for higher
concurrency, and verify that cache/state admission succeeds. Weight loading
alone does not establish serving capacity. Do not silently reduce model depth
to bypass capacity failures.

## Communication and validation

The forward chain is AllGather, gate/up, activation, down, ReduceScatter.
For TP2/4/8/16 with positive, 128-aligned hidden widths, TRT-LLM one-shot
collectives cover up to 128 padded rows/rank on CUDA-IPC-accessible groups.
Group size and buffer shapes come from runtime parameters; AllGather subdivides
rows into aligned views within the native hidden-width limit. Other TP sizes,
unaligned widths and larger batches use NCCL. One-shot address-space limits still
apply. TP4/H7168 has prior GPU validation; newly enabled geometries require GPU
validation on the target topology before deployment.
`SharedExpertCommunication` owns the persistent scratch and runs these
collectives; it is initialized before graph capture. Empty
owners still participate when their subgroup is active. AllGather completes
before routed dispatch; shared GEMMs finish before routed BMM. Shared
ReduceScatter runs after dispatch and completes before combine, so shared
collectives never overlap routed dispatch/combine.

The plain collective wrappers live in `tokenspeed_kernel.ops.communication.trtllm`
as `trtllm_allgather` and `trtllm_reduce_scatter`, with explicit
`TrtllmAllGatherState` and `TrtllmReduceScatterState` instances. These instances
own independent IPC workspaces, separate from the existing fusion workspace.
AllGather returns a borrowed buffer; ReduceScatter returns owned output.
Their `stateful_allgather` and `stateful_reduce_scatter` registry modes do not
participate in stateless auto dispatch. Runtime padding and backend selection
remain in `SharedExpertCommunication`.

Run the model orchestration and stream-ordering tests:

These tests are registered in the per-commit `runtime-1gpu` CI suite. The
distributed real-weight validator below is separate and is not automatically
run by that suite. A green unit-test job does not establish real collective
or real-weight validation. Multi-node Slurm CI skips fork PRs; running that
validation requires a reviewed dispatch with suitable GPUs and checkpoint access.

```bash
python -m pytest -q test/runtime/test_kimi_k3_moe_attn_dp.py test/runtime/test_cuda_stream.py \
  test/runtime/test_kimi_k3_config.py
```

Use `test/runtime/distributed/validate_kimi_k3_shared_expert_tp.py` under
`torchrun` with 16 workers and explicit `--model MODEL_DIR --layer 1 --tp-size 4`
for real-weight validation; repeat with `--tp-size 2` and `--tp-size 8`.
It covers cross-rank setting agreement using real collectives, exact weight
shards, uneven/empty owners, retained outputs, and changing inputs during graph
replay. Eager warmup also checks actual collective participation and backend
selection at the 128/129-row boundary without substituting collective results.
Synthetic communication cases also cover narrow, subdivided, and unaligned
hidden widths with one token per rank in eager execution; the real-weight cases
above cover empty owners, row-count boundaries, and CUDA-graph replay.
These shared-expert checks live in the GPU validator, not a separate CPU suite.

For E2E comparison, use fixed per-rank request affinity, identical prompts and
generation lengths, warmup, and repeated unprofiled rounds. Report prefill-plus-
decode workflow latency separately from full-batch decode intervals. Record
weight memory and allocated cache as well as peak device use: a fixed memory
budget can convert weight savings into cache rather than lower device usage.
