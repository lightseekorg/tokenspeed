# MLP channel profiling and selective observation

`ChannelActivityProfiler` and `ChannelWatch` provide an opt-in path from an
offline workload sample to sparse activation records during eager inference.
They do not decide whether a checkpoint or agent action is safe.

## Measurement and ownership

The profiler observes the actual output of each Qwen3 MLP's `SiluAndMul`: one
activation per physical token row and intermediate channel. It collects all
channels in all transformer blocks. A channel here is one coordinate of the
MLP intermediate vector, not an attention head or a uniquely interpretable
concept. The offline profile records the row count, mean absolute activation,
RMS, observed maximum, fraction above a caller-set activation threshold, and
an absolute-activation histogram for each channel. It also records
`abs(activation) * ||down_projection_column||_2` summaries. This last quantity
estimates the size of the channel's direct MLP output contribution before
later layers; it is neither a logit effect nor a causal effect.

The caller supplies the workload and a nonempty model identity. The identity
binds the two artifacts by equality, but the caller must independently verify
checkpoint provenance. The model must remain fixed while a profile or watcher
is attached. A profile describes only the sampled input distribution. Rare
activity on that sample does not establish that a channel was dormant during
training, benign, or susceptible to a particular modification.

Profiling runs only inside an explicit `record()` around a complete model
forward. The caller may select physical rows with `record(row_indices=...)`;
the default records every row. A selected-row profile can, for example,
compare the same output-decision position across requests. The report records
both physical rows seen and rows selected. All supported layers must call
their activation module once. A missing or repeated call fails the record
instead of being interpreted as zero activity. Statistics are accumulated on
the device; `profile()` transfers aggregates for JSON export after recording.

## Freezing a watchlist

`freeze_watchlist()` selects channels by either of two explicit rules after a
minimum row count. The quiet rule applies caller-set ceilings to observed
active fraction, mean contribution estimate, and maximum contribution
estimate. The weight-outlier rule requires a down-projection column norm above
a caller-set multiple of its layer's median, plus a caller-set ceiling on
active fraction. A layer with zero median cannot satisfy that rule. The
profile exposes the norm, median, and ratio so the reason for a nomination can
be inspected. Large output weight on a rarely active channel may be useful to
review, but is not evidence of a malicious edit. The method emits a sorted
list of `(layer, channel, contribution_limit)` entries under the same model
identity. The runtime limit is `max(observed_max * multiplier, floor)`; both
parameters are explicit. A frozen limit is a comparison with the calibration sample, not a
confidence interval or a guarantee about future ordinary traffic.

Freeze thresholds and candidate selection before examining any held-out event
that will be used to evaluate the watcher. Keep the calibration workload,
thresholds, selected-channel count, held-out alarms, and misses with the
result. A watchlist may be empty; in that case there is no runtime watcher to
attach. Recalibrate after changing weights, model configuration, workload, or
the meaning of physical rows.

## Sparse inference records

`ChannelWatch` attaches hooks only to MLP layers represented in its frozen
watchlist and copies only their selected activation columns. It computes the
same direct contribution estimate on the device and compares it with each
frozen limit. The hook does not copy full MLP activations to host, change model
output, or synchronously export events. The caller wraps exactly one forward
with `record(rows=...)`, supplying a `RowIdentity` for each physical row.
The caller may also pass `selected_rows` to watch only designated physical
rows; exported events retain their original physical-row index and supplied
request/token identity. `snapshot()` owns copies of sparse device tensors.
`events(snapshot)` transfers only over-limit samples to the host after the
forward; with no hits it transfers no activation values. The caller owns the
event store, retention policy, request correlation, and later tool-action
correlation; those are not inferred by the hook.

An event means that a watched channel's direct contribution estimate exceeded
its frozen threshold on one recorded row. It is a lead for inspection. It is
not a model decision, an action authorization failure, or a proven backdoor.
Some harmful behavior may involve channels absent from the watchlist, ordinary
channels, attention, or many weak contributions. A large activation may also
be legitimate. Tool permission checks belong at the action boundary.

## Supported execution

The initial path supports one dense, unquantized BF16 or FP32
`Qwen3ForCausalLM` on one device, with eager, non-speculative,
non-monokernel execution and unfused `SiluAndMul`. Fused MLPs, quantization,
graph capture or replay, parallel shards, concurrent forwards, and weight
changes during observation are unsupported. A record fails when its expected
hooks do not fire. Without either object attached, the model's execution path
is unchanged. Attached hooks and device reductions have overhead; this design
does not claim a production latency budget. Benchmark the selected watchlist
and row lengths before deployment.

The companion example in `examples/profile_mlp_channels.py` runs ordinary
TokenSpeed cold eager forwards, writes a full profile, a ranked candidate
preview, and a watchlist, then can apply the watchlist to separate inputs.
`--rows all|last` makes the measured scope explicit in both phases. Input
records contain exact token IDs; an optional rendered text field is checked
against them without adding special tokens. It is a bounded integration
example rather than a server integration or a tool policy.

The example parses and hashes the same input bytes. It builds the ranking,
watchlist, and preview from one exported profile, using the same channel
selection rule for the watchlist and its displayed reasons.

```bash
python examples/profile_mlp_channels.py \
  --mode profile --model /path/to/checkpoint --revision pinned-revision \
  --model-identity declared-identity --inputs calibration.json \
  --output-dir profile-output --norm-mode fused --attention-backend mha \
  --rows last --activation-threshold 0.01 \
  --histogram-edges 0.01,0.1,1,10 \
  --max-active-fraction 0.05 --max-mean-contribution 0.5 \
  --max-observed-contribution 2 --min-observed-rows 24 \
  --min-down-norm-ratio 5 --max-outlier-active-fraction 0.25 \
  --runtime-multiplier 8 --contribution-floor 1

python examples/profile_mlp_channels.py \
  --mode watch --model /path/to/checkpoint --revision pinned-revision \
  --model-identity declared-identity --inputs heldout.json \
  --output-dir watch-output --norm-mode fused --attention-backend mha \
  --rows last --watchlist profile-output/watchlist.json
```

The numeric thresholds above illustrate the interface; derive them from a
recorded calibration protocol and evaluate false alarms and misses on held-out
traffic before using an event for any operational decision. The example
compares complete model logits with and without watching on every input, and
fails its report if they differ. It does not run an agent or enforce tool
permissions.
