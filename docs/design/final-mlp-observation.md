# Final-MLP observation and replay

`tokenspeed.runtime.inspection.final_mlp.FinalMLPObserver` is an opt-in developer
tool for collecting actual inference intermediates and testing a local
activation intervention. It does not classify neurons as malicious, authorize
agent actions, or establish model integrity.

## Scope and ownership

The initial adapter accepts the ordinary dense `Qwen3ForCausalLM`, unquantized
BF16 or FP32 tensors, one device (TP=PP=DP=1), and eager execution. Fused
GEMM/SwiGLU paths, quantization, speculative execution, monokernels, and graph
capture/replay are unsupported. There are no changes to the model's forward,
scheduler, cache, sampling, or output protocol. Without creating an observer,
there are no additional hooks, buffers, or computations. Construction allocates
the private buffers; entering the context attaches hooks. Attached but idle
hooks still execute their guard checks. Closing removes hooks but retains the
observer's owned buffers until the object is released.

An observer owns fixed device buffers for one physical token-row count and one
selected-logit-row count. It runs on the forward's thread and stream. Do not
share it across concurrent forwards, move the model, reload weights, change
kernel or numerics configuration, or update weights during its lifetime. This
is a local analysis contract, not an integrity boundary against arbitrary
memory writes.

`record()` brackets exactly one forward. Every required hook must execute
once; missing or repeated hooks fail the record, and exceptions invalidate it.
This also prevents an existing captured graph from silently producing a stale
record: graph replay does not execute Python hooks. Snapshot tensors own copies
and survive subsequent forwards or observer removal. The private recording
buffers are never exposed. Exporting snapshots to CPU is an explicit caller
operation; the API does not add a background exporter or request-slot state.

## What is captured

| Tensor | Exact source |
| --- | --- |
| `mlp_input` | Output of the final post-attention RMSNorm. |
| `residual_before_mlp` | Its separate residual output, copied before a later norm can overwrite it. |
| `gate_up` | Input to the final MLP's activation module, arranged `[gate \| up]`. |
| `activation` | Actual SiLU-times-up output consumed by the down projection. |
| `mlp_output` | Actual down-projection output, before final residual addition/norm. |
| `normalized_hidden` | Actual final RMSNorm output, before head row selection. |
| `selected_logits` | Full vocabulary logits returned by the logits processor, before sampling. |
| `logit_rows` | Effective physical hidden-row indices selected by that processor. |

The normal `GatedMLP` calls `down_proj.forward_with_activation`, so a hook on
the down projection's module `forward` would miss the activation seam. The
observer hooks the actual activation module instead. Supported unquantized
methods call it exactly once; fused methods are refused rather than replaced.
Hooks copy tensors and return `None`, preserving the original computation.

All physical rows are retained. Replaying only selected rows can change GEMM
dispatch and accumulation order. For T rows, intermediate width I, hidden
width H, S selected logits and vocabulary V, BF16 recording buffers occupy
`2 * (3*T*I + 4*T*H + S*V) + 8*S` bytes, excluding snapshots and model weights.
No latency or throughput guarantee follows from that storage formula.

## Usage

The caller already owns the initialized TokenSpeed model and its normal
forward context. The example wraps that existing forward; it does not build a
second inference path.

```python
from tokenspeed.runtime.inspection.final_mlp import FinalMLPObserver

with FinalMLPObserver(
    model,
    num_rows=input_ids.shape[0],
    num_logits_rows=1,
    execution="eager",
    speculative=False,
    monokernel=False,
) as observer:
    with observer.record():
        output = model(ctx, input_ids, positions)
    trace = observer.snapshot()
    replay = observer.replay(trace, ablate_channels=(17,))
```

The trace records physical row indices, not request identifiers, token ids, or
absolute positions. An analysis driver must retain that mapping, the model
revision, source commit, dtype, norm epsilon, kernel/numerics configuration and
weight provenance alongside any exported tensors. CPU or another framework's
calculations are separate numerical references, not automatically identical
engine replays.

## Three distinct suffix results

Let A be the captured activation, D the down projection, M its captured output,
and R the separate pre-MLP residual. Replay returns:

1. `suffix_logits`: run the original final norm/head suffix from saved M and R.
2. `recomputed_logits`: recompute D(A) at the original physical shape, then run
   the same suffix. Its difference from the first result measures replay error
   independently of any intervention.
3. `intervened_logits`: copy A, zero the requested channels, recompute the full
   down projection, then run the same suffix. No model weight or KV entry is
   modified.

Replay uses the model's projection method, norm implementation, and logits
processor. Direct suffix calls bypass only the recording hooks so the replay
cannot overwrite the record under inspection. Scratch clones prevent in-place
norms from modifying captured tensors.

The final residual-add mode matters. `fused` feeds the FP32 sum of M and R into
the norm. `unfused` first materializes their sum in the tensor's dtype. Capturing
only the rounded post-add residual would lose information needed by the fused
form. The observer preserves both addends and replays the configured form.

Zeroing an activation and recomputing GEMM is not equivalent in finite precision
to subtracting `A[:, i] * D[:, i]` from the saved output. The API implements the
former. It intervenes only on this forward's final MLP: it does not replay an
altered prefix, earlier layers, future tokens, or a changed model's KV cache.

Before interpreting channel effects, compare the complete original logits with
both replay baselines. Exact equality is a validation result for the tested
device, shape and configuration, not a guarantee across different kernels,
row counts, dtypes or machines. Large activation or causal effect alone is not
evidence of a backdoor.

## Validation

`test/runtime/test_final_mlp_observer.py` exercises real Qwen3 suffix classes,
snapshot independence, malformed records, row selection, channel interventions,
configuration rejection, and same-shape replay. CPU cases use numerical kernel
references because those entry points require GPU execution; GPU cases run the
actual selected BF16 kernels.

`examples/inspect_final_mlp.py` loads a real checkpoint through `ModelRunner`
and runs every layer using the normal paged-attention backend. It owns a single
request's pages, resets KV storage, and uses a zero cached prefix for every
comparison. It does not start a scheduler or server. In an installed TokenSpeed
GPU environment, create a JSON array of raw prompt strings and run:

```bash
python examples/inspect_final_mlp.py \
  --model Qwen/Qwen3-0.6B \
  --revision c1899de289a04d12100db370d81485cdf75e47ca \
  --prompts prompts.json \
  --channels 0,17 \
  --norm-mode fused \
  --attention-backend mha \
  --output-dir observation-fused \
  --export-suffix-weights
```

Repeat with `--norm-mode unfused` and a new output directory to validate that
normalization path separately. The example uses BF16 and disables graph capture
and autotuning. It rejects prompts above `--max-tokens` instead of truncating
them. A local checkpoint directory is also accepted; the revision argument
then records the caller's declared source, not a verification of local bytes.

The report compares complete selected-logit arrays with observation off/on,
both suffix baselines, and a separate full-model run whose native activation
hook zeros the same channels. It also checks snapshot independence after a
different same-shape forward, restored baseline logits after hook removal,
and explicit unsupported-configuration refusal. Any nonfinite value or exact
equality failure makes the example fail; error magnitudes are retained for
diagnosis. No tolerance is silently substituted for identity.

Per-prompt SafeTensors files retain the trace, token IDs, positions and compared
logits. Optional suffix weights retain the actual runtime gate/up projection,
down projection, final norm and padded head for a separate offline reference.
An offline reconstruction must account for the recorded row selection, dtype,
norm mode, logical vocabulary, head scaling and softcap. Agreement between two
reconstructions alone does not establish agreement with the recorded forward.

This example validates cold eager prefill only. Decode, prefix-cache reuse,
graph replay, serving lifecycle, sampling and multi-device execution need
their own integration coverage. Recording and host-export overhead are not
measured here.

See the [research results](../research/final-mlp-observation-results.md) for
separate cached-decode checks, bounded timing measurements, and mitigation limits.
