# MLP channel profiling and selective observation

`ChannelActivityProfiler` measures MLP activations over a calibration workload
in eager TokenSpeed inference. Use `ChannelWatch` on a second workload to
identify which token positions exceed a recorded channel limit. The watcher
retains the activation, layer, channel, and physical row for each event.

## Measurement and ownership

The profiler observes the actual output of each Qwen3 MLP's `SiluAndMul`: one
activation per physical token row and intermediate channel. It collects all
channels in all transformer blocks. A channel is one coordinate of the MLP
intermediate vector. The offline profile records the row count, mean absolute
activation, RMS, observed maximum, fraction of rows whose absolute activation
exceeds a caller-set threshold, and an absolute-activation histogram for each
channel. It also records
`abs(activation) * ||down_projection_column||_2` summaries. This last quantity
estimates the magnitude of the channel's contribution at the down projection.
The final logits also depend on the residual stream and subsequent operations.

The caller supplies the workload and a nonempty model identity. The watcher
requires the same identity string as the profile. Keep the model fixed while
recording and retain the checkpoint revision with the profile. Each statistic
describes only the sampled inputs and selected rows.

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
estimate. The weight-outlier rule requires a down-projection column norm at
least a caller-set multiple of its layer's median, plus a caller-set ceiling on
active fraction. A layer with zero median cannot satisfy that rule. The
profile records the norm, median, and ratio used in the comparison.
The resulting watchlist contains sorted `(layer, channel, contribution_limit)`
entries. For each channel, the limit is `max(observed_max * multiplier, floor)`,
where `observed_max` is its largest estimated contribution on the calibration
rows. The caller supplies the multiplier and floor.

Freeze the watchlist before running the comparison workload so both the
selection and the limits come from calibration alone. Record the workload,
thresholds, selected-channel count, and number of sampled rows alongside the
events. If calibration selects zero channels, skip attaching a watcher.
Build a new reference profile when changing weights, model configuration, the
calibration workload, or the rule for selecting rows.

## Sparse inference records

`ChannelWatch` attaches hooks only to MLP layers represented in its frozen
watchlist and copies only their selected activation columns. It computes the
same direct contribution estimate on the device and compares it with each
frozen limit. The hook does not copy full MLP activations to host, change model
output, or synchronously export events. The caller wraps exactly one forward
with `record(rows=...)`, supplying a `RowIdentity` for each physical row.
The caller may also pass `selected_rows` to watch only designated physical
rows; exported events retain their original physical-row index and supplied
request/token identity. A snapshot owns device tensors covering every selected
row and channel, including samples below the limit.
`events(snapshot)` transfers only over-limit samples to the host after the
forward. With no hits it transfers no activation values. The caller stores
events and supplies the mapping from physical rows to requests and positions.

An event gives the contribution estimate and the limit it exceeded on one
recorded row. Use the request and token position to locate that sample in
the workload. For channels in the final MLP, a separate replay can measure
how zeroing the channel changes the logits on the same input. See
[final-MLP replay](final-mlp-observation.md).

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
against them without adding special tokens.

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

The numeric thresholds above illustrate the interface. Choose thresholds
using a representative calibration workload, then inspect the event counts
on held-out inputs. For each input the example compares the complete logits
with and without the watcher and fails the report if they differ or contain
nonfinite values.
