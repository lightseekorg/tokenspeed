# Persistent hyperconnection mix

The low-row gated-residual mix uses one resident Triton grid for the down/inject
projection, SiLU, up projection and gated branch reduction. The production
specialization accepts contiguous BF16/FP16 inputs with 1–16 rows, HC=4,
hidden size 2560 and rank 320. These HC weights are replicated across TP ranks;
TP4 does not divide the kernel's 10240-wide input or projection weights.

## Scratch lifetime and synchronization

Scratch belongs to a `(device, CUDA stream, projection rows)` workspace. It is
temporary kernel storage, not per-request model or attention state. Each
workspace contains two FP32 projection buffers and two signed 64-bit counters:
cumulative CTA arrivals and the next launch's generation. With inject enabled,
the buffers occupy 40.5 KiB, plus 16 bytes of counters.

Both buffers and counters start at zero. For generation `g`, each CTA:

1. Waits for its PDL producer before accessing any workspace state. Ordinary
   stream ordering provides this dependency with PDL disabled.
2. Reads the device generation and accumulates into buffer `g & 1`, which the
   previous generation cleared. In parallel, CTAs clear disjoint slices of
   buffer `(g + 1) & 1` for the next launch.
3. Publishes its writes with a GPU `acq_rel` arrival atomic and waits with GPU
   acquire loads until cumulative arrivals reach `(g + 1) * num_ctas`.
4. Consumes the completed projection. CTA zero publishes generation `g + 1`
   after the barrier. The arrival counter stays monotonic so late pollers can
   still observe completion.

This removes the initial clearing barrier and the final reader counter. The
buffer being cleared is never read in the same launch. Publishing the next
generation does not permit a following launch to access scratch early: its
stream/PDL dependency still waits for the preceding launch.

The grid size is fixed to the device SM count for each workspace. Changing this
policy requires a fresh workspace or a revised cumulative-arrival contract.
The grid must remain resident for its global barrier to make progress. The
64-bit counters avoid the frequent wrap that cumulative 32-bit arrivals would
encounter in long-running decode.

Generation is device state: eager calls and graph replays execute exactly the
same protocol, including graphs containing an odd number of mix calls. Host
capture-time parity must never select the buffer. Projection weights and inputs
are read on every invocation; graph replay does not cache their values.

GPU acquire semantics follow the [PTX memory consistency model and load
instructions](https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#data-movement-and-conversion-instructions-ld).
Projection accumulation still uses FP32 atomics, so the specialization remains
excluded from deterministic dispatch. Input precision and numerical tolerances
are unchanged.

## Validation and measurements

`tokenspeed-kernel/test/ops/test_hyperconnection.py` checks FP64 reference parity,
BF16/FP16, partial row tiles, optional inject outputs, projection scaling,
PDL on/off, changing shapes in one workspace, concurrent eager/graph streams,
odd/even graph replay counts with changing inputs/weights, and generations
crossing `2**32`.

On NVIDIA B300 SXM6 AC (148 SMs), CUDA 13.0, PyTorch 2.13.0+cu130 and
tokenspeed-triton 3.8.10, baseline and optimized validation measured:

| BF16 rows | Original (µs) | Optimized (µs) | Reduction |
| --- | ---: | ---: | ---: |
| 1 | 10.73 | 6.95 | 35.2% |
| 4 | 10.60 | 7.10 | 33.0% |
| 8 | 10.75 | 7.21 | 32.9% |
| 16 | 10.95 | 7.24 | 33.9% |

These are CUDA-event timings of repeated graphs containing 64 complete mix
calls with resident weights, PDL enabled and inject outputs included. They
exclude host launch overhead and transfers. Validation also passed FP16,
non-unit projection scale, no-inject, partial-row and PDL-disabled workloads;
their medians were 6.84–7.34 µs.

The repository benchmark also compared every row count from 1 through 16 in
both BF16 and FP16 against the original implementation on the same B300:
all 32 cases improved by 29.0–35.0%, with optimized latency at most 7.50 µs.
Four concurrent per-rank microbenchmark processes measured T=4 at
7.20, 7.16, 7.11 and 7.18 µs on four B300 GPUs.

Cache conditions matter. A separate paired measurement on another B300 rotated
32 independent weight pairs (about 403 MiB, exceeding the 126 MiB L2): T=4
improved from 11.92 to 9.47 µs. The sub-8 µs result applies to resident weights;
it is not a claim of sub-8 µs latency for this rotating-weight case or of
end-to-end model throughput.

From the repository root, activate the Python environment and run:

```bash
PYTHONPATH=tokenspeed-kernel/python python \
  tokenspeed-kernel/test/ops/bench_hyperconnection.py \
  --backend persistent --operation mix --rows 1,4,8,16 --mode graph

PYTHONPATH=tokenspeed-kernel/python python \
  tokenspeed-kernel/test/ops/bench_hyperconnection.py \
  --backend persistent --operation mix --rows 4 --mode graph --weight-banks 32
```

The benchmark warms the capture stream before recording, batches calls within
each graph to avoid timing Python replay gaps, and reports latency per call.
Use `--operation chain` for normalization + mix + combine, and `--dtype fp16`
for the other supported input dtype. Compare baseline and candidate with the
same benchmark, GPU and cache conditions.

## Gated residual normalization

Qwen4-Exp carries residual streams shaped `[tokens, branches * hidden_size]`.
Each branch is independently Gemma RMS-normalized over `hidden_size`, with
effective per-feature scale `1 + weight`. Mixing uses the normalized streams;
sublayer injection updates the unnormalized residual streams.

### Adjacent combine and norm

When one sublayer's combine directly precedes the next mixer, the next mixer
accepts the previous sublayer output and injection logits together with the
original residual. `tokenspeed-kernel.gated_residual_combine_norm` computes:

```text
updated[t, g, :] = residual[t, g, :]
                  + output[t, :] * 2 * sigmoid(inject_logits[t, g])
normalized[t, g, :] = GemmaRMSNorm(updated[t, g, :], next_norm_weight[g, :])
```

Both tensors are outputs. The next sublayer's combine needs `updated`, so
normalization must not overwrite or replace that residual. The normalizer's
weight belongs to the consuming mixer, while the injection logits belong to
the preceding mixer. Shared branch weights remain supported.

The grouped RMSNorm Triton kernel owns both the ordinary norm and this optional
combine prologue. Its grid is `(tokens, branches)`; each CTA handles a full
branch with `BLOCK = next_power_of_2(hidden_size)`. The wrapper ensures contiguous
storage, so the kernel derives row offsets directly from the tensor widths.
Combine, normalization and both stores occur within that CTA, without a second
launch or a reload of the
updated residual. Masked lanes do not participate in the mean square.

The updated residual is rounded to its storage dtype **before** computing the
RMS statistics. This preserves the BF16/FP16 rounding boundary of separate
combine and norm kernels; retaining the FP32 sum through normalization would
change model numerics. FP32 arithmetic and the `1 + weight` scale otherwise
match standalone grouped Gemma RMSNorm.

### Execution boundaries

The shared decoder code fuses attention combine into MLP normalization after
attention output communication and residual-row alignment. GDN, full attention
and the MTP draft use the same path, including idle and CUDA graph execution.
No normalized tensors are cached across forwards.

`preload_residual=True` explicitly promises that both the residual and norm
weight are already visible before the current PDL producer begins. The fused
kernel loads them before `gdc_wait`; block output and injection logits remain
after the wait. Attention-to-MLP can make this promise because attention has
already consumed the residual, communication preserves it or changes its row
view, and a newly gathered residual is consumed by `norm_for` before injection.
The MTP row selection likewise materializes its residual before its norm and
gate selections. Normal grouped RMSNorm waits before reading its activation.
If the kernel wrapper must copy a noncontiguous residual or weight, it disables
preloading for that invocation so the copy's output is not read prematurely.

The MLP's final combine stays materialized at the layer boundary. The next
attention preparation can communicate rows, apply PLE, or follow a multimodal
deepstack addition. A norm must consume the result of those operations; it
cannot be moved ahead of them. Likewise, the final output mixer runs after the
model's final communication and retains the unnormalized HC output for MTP.

### Validation

`tokenspeed-kernel/test/ops/test_hyperconnection.py` compares fused and separate
kernels for BF16, FP16 and FP32, shared/per-branch weights, strided inputs,
non-power-of-two widths, empty inputs, and CUDA graph replay with PDL on/off.
An intentionally delayed producer and poisoned output buffers check that
activation loads remain behind the PDL wait.
`test/runtime/test_hyperconnection_kernel_boundary.py` checks consumer norm
weights, subsequent residual injection, and attention-to-MLP row alignment.

Compare separate combine and norm with the fused kernel, with and without
preloading:

```bash
PYTHONPATH=tokenspeed-kernel/python python \
  tokenspeed-kernel/test/ops/bench_hyperconnection.py \
  --operation combine_norm --dtype bf16 --rows 1,4,16,128,2048 --mode graph
```

Graph timings capture repeated operations in one graph to amortize host launch
overhead and report the median of five device-event measurements. These are
combine-plus-norm timings, not end-to-end model throughput.
