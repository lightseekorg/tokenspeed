# Gated residual normalization

Qwen4-Exp carries residual streams shaped `[tokens, branches * hidden_size]`.
Each branch is independently Gemma RMS-normalized over `hidden_size`, with
effective per-feature scale `1 + weight`. Mixing uses the normalized streams;
sublayer injection updates the unnormalized residual streams.

## Adjacent combine and norm

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

## Execution boundaries

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

## Validation

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
