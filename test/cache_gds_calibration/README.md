# GDS cost calibration

These tools collect prefill and Disk read measurements, fit the explicit cost models used by GDS, and write configuration only after held-out validation passes. They are independent of model family, accelerator vendor and deployment system. Supply the model, engine arguments, representative corpus, cache geometry and local SSD for the deployment being measured. There are no pretrained coefficients or hardware presets.

Run Python commands from the repository root with its Python environment activated. Offline fitting and tests need NumPy; prompt collection also needs Transformers and the serving environment. Native collection uses TokenSpeed's `WireDriver`. The Disk benchmark is a Rust example in `tokenspeed-engine/crates/ts-disk`, where it can use the serving executor directly.

| File | Responsibility |
| --- | --- |
| `calibrate.py` | CLI for HTTP collection, timing import, Disk sampling and fitting |
| `wire_collect.py` | Native collection without an HTTP frontend |
| `data.py` | Shared workload generation, geometry checks and JSONL I/O |
| `prefill.py` | HTTP requests and server timing validation |
| `disk.py` | Invocation and validation of the native Disk benchmark |
| `fitting.py` | Nonnegative fitting, validation reports and configuration export |
| `cases.json` | Full native prefix/new-token matrix, selected explicitly with `--cases` |
| `cases_quick.json` | Default native matrix: eight representative geometries |
| `test_calibrate.py` | CPU behavioral tests using synthetic data and a local HTTP server |

## Cost models and units

Following the prefill model in [FlashStates, Appendix C](https://www.alphaxiv.org/pdf/2611.flashstate), fit prompt-processing time in **seconds**:

```text
C(a,b) = d * (b-a) + e * (b*b-a*a) / 200000
R(route) = A * unique_missing_host_blocks + B * aligned_bytes

d > 0, e >= 0; A >= 0, B > 0
```

`a` is the actual cached prefix and `b` is the full prompt length. `d` and `e` are in seconds/token; the marginal prefill cost at depth `x` is `d + e*x/100000`. `A` is seconds/block and `B` is seconds/byte. Disk pricing is TokenSpeed's extension: a route sums the missing Host blocks' aligned payload sizes, without counting shared keys twice. The scoring contract is in [RFC 0007](../../docs/rfcs/0007-dependency-aware-hybrid-cache-management.md).

Multiplying all four coefficients by the same positive constant preserves relative scores, aside from rounding. Changing only Disk costs changes the preference for Disk recovery versus recomputation. Overlapped I/O still consumes bandwidth, buffers and credits, so a positive Disk penalty remains useful. This model does not predict queue delay, exposed request latency or Host-to-Device cost.

## 1. Isolate and record the deployment

Record model revision, dtype/quantization, runtime and kernel revisions, parallelism, chunk size, graph/overlap settings, accelerator topology and resolved engine arguments. Keep them fixed during a run. Choose one `--deployment` label for its prefill and Disk samples; the label does not verify the configuration automatically.

Use a dedicated idle replica, one request at a time, and a fixed attention-DP rank. Disable Host L2, Disk, peer prefix sharing and P/D transfer for compute sampling. Keep Device prefix caching enabled for warm continuation, and select `--prefix-cache-evict-policy lru` to measure costs independently of GDS. Typical isolation arguments are:

```text
--disaggregation-mode null
--disable-kvstore
--enable-prefix-caching
--enable-log-request-stats
--prefix-cache-evict-policy lru
```

Add these to the deployment's complete engine arguments; they are not a model launch command on their own. The largest prompt and its prime must fit on Device without retraction. Match the production kernels and chunk size. Cover the intended context range and short continuations with separate validation geometries. Do not infer accuracy at larger contexts or different parallelism from a smaller run.

## 2. Collect prefill requests

Each entry in a case file names the two input lengths explicitly:

```json
{"prefix_tokens": 131072, "new_tokens": 32768, "split": "train"}
```

| Field | Meaning |
| --- | --- |
| `prefix_tokens` | Tokens to compute and cache before the measured request; zero means cold prefill |
| `new_tokens` | Additional input tokens after that prefix; excludes generated output tokens |
| `split` | `train` supplies timings for coefficient fitting; `validation` supplies independent timings for error checks |

The example first primes a 128K-token prefix, then sends a 160K-token prompt to measure its additional 32K input tokens. `prompt_tokens = prefix_tokens + new_tokens` is derived, so cases do not need a separately entered total length. In the mathematical formula above, `a` is the actual cached depth and `b` is the total prompt length; the requested geometry is `a=prefix_tokens`, `b=prefix_tokens+new_tokens`. Fitting uses the measured cache hit if it differs within the explicitly allowed tolerance.

Here **training means fitting the small cost formula's coefficients**: `d,e` for prefill or `A,B` for Disk. Both splits run ordinary inference or I/O measurements; model weights are never updated. The fitter estimates coefficients using only `train` timings, then compares predictions with `validation` timings from different geometries. Validation data does not change the coefficients. It checks whether the formula predicts unseen measurements well enough to export configuration; excessive error prevents export. For example, fitting may include a 64K prefix plus 8K new tokens, while validation checks a 96K prefix plus 8K new tokens.

Both transports use the same workload generator: warm every geometry, shuffle formal samples, prime exactly `prefix_tokens` token IDs, then send the full `prompt_tokens` IDs with one output token. Random leading tokens isolate each pair, while the remainder uses contiguous spans from the supplied corpus. Prime requests and warmups never enter the fit. Use representative text/code for the model's workload, especially for routed experts.

The supplied `cases.json` varies both `prefix_tokens` and `new_tokens`. Long-prefix continuation measures how prefill cost changes with existing context. Writing `p=prefix_tokens` and `n=new_tokens`, the expected cost is `d*n + e*(2*p*n+n*n)/200000`. Holding `new_tokens` fixed while varying `prefix_tokens` checks context dependence, and varying `new_tokens` checks how the same prefix behaves across different amounts of work and chunk counts.

| Split | prefix_tokens | new_tokens | Cases |
| --- | --- | --- | ---: |
| Coefficient fitting (`train`), cold | 0 | 4K, 8K, 16K, 32K, 64K, 128K, 256K | 7 |
| Coefficient fitting (`train`), continuation | 16K, 64K, 128K, 256K | 4K, 8K, 16K, 32K, 64K, crossed with every prefix | 20 |
| Validation, cold | 0 | 96K, 192K | 2 |
| Validation, continuation | 32K, 96K, 192K | 4K, 8K, 16K, 32K, 64K, crossed with every prefix | 15 |

Here K means 1024 tokens. Validation holds out complete `(prefix_tokens,new_tokens)` geometries and continuation depths, while preserving coverage of each suffix size. The full 44-case matrix produces 132 formal measurements and 44 warmups with `--repeats 3 --warmup-repeats 1`, plus separate prime requests and the native smoke request. The largest prompt is 320K tokens; the engine needs context capacity for that prompt plus its output token and enough Device cache for the run. Adjust the matrix before starting if the deployment supports a smaller range, and report the resulting coverage. Do not treat these new example cases as already measured results.

Native collection defaults to `cases_quick.json` when `--cases` is omitted. The file is resolved relative to the collector module, independently of the working directory. With `--repeats 3 --warmup-repeats 1`, this keeps five fitting geometries and three validation geometries: cold 32K; cached 16K+4K, 64K+8K, 128K+16K and 256K+32K for fitting; 32K+16K, 96K+8K and 32K+64K for validation. It produces 24 formal measurements plus eight warmups, and its largest prompt is 288K. Each continuation also needs a separate prime request, so long prefixes dominate collection time. The quick matrix covers all five suffix sizes but does not cross every size with every prefix depth; select `--cases test/cache_gds_calibration/cases.json` explicitly when broader validation is needed. An explicitly supplied case file is used as given. Repeat counts remain required arguments.

For native collection, create `engine-args.json` containing a JSON array of the complete CLI argument strings for the isolated engine. `--model-path` selects its matching tokenizer. Set `--num-engines` to the deployment's native engine count and choose a valid attention-DP rank; these are explicit inputs, not inferred from the model. Run, adjusting the example values to the deployment:

```bash
mkdir -p calibration-output
python -m test.cache_gds_calibration.wire_collect \
  --engine-args /path/to/engine-args.json \
  --model-path /path/to/model --text-file /path/to/corpus.txt \
  --trust-remote-code no --deployment calibration-run \
  --num-engines 1 --data-parallel-rank 0 --routing round_robin \
  --warmup-repeats 1 --repeats 3 --seed 41 \
  --max-prefix-shortfall 1 --startup-timeout 2400 \
  --output-dir calibration-output/prefill \
  > calibration-output/prefill.log 2>&1
```

The output directory must not exist. It contains `manifest.json`, `engines.json`, `smoke.json`, `requests.jsonl` and, after collection and driver cleanup return, `complete.json`. Preserve the combined log, including shutdown errors. The native client explicitly sets request creation time. A failed completion or unexpected prefix match stops collection after recording the sample. Prefix shortfall is explicit: choose zero for exact reuse, or a small justified allowance; fitting always uses the actual recovered depth.

To shorten an interrupted run, choose a smaller case file and pass `--resume-requests /path/to/previous/requests.jsonl` with a new output directory. Keep the deployment label, execution configuration, corpus, split assignment and formal repeat count unchanged. The native collector warms the selected cases again and skips their already completed formal repeat indices; it rejects duplicate repeats, failed completions and mismatched deployments or splits. A different seed can be used for newly collected inputs. Import both the reused records and new records with their original server logs. Explicitly select the retained geometries from the old request file before importing if it also contains cases that were removed; retain the complete original file for audit. Warmups remain excluded. The completion marker separately reports newly collected and reused request counts.

If an isolated native `/generate` HTTP endpoint is already running, use this alternative. It must preserve token IDs, request ID and the explicit DP rank:

```bash
python -m test.cache_gds_calibration.calibrate prefill \
  --url http://127.0.0.1:8000 --model-path /path/to/model \
  --text-file /path/to/corpus.txt --trust-remote-code no \
  --deployment calibration-run --data-parallel-rank 0 \
  --prefix-lengths 0 16384 65536 131072 262144 \
  --suffix-lengths 4096 8192 16384 32768 65536 \
  --warmup-repeats 1 --repeats 5 --seed 41 --split train \
  --output calibration-output/train-requests.jsonl

python -m test.cache_gds_calibration.calibrate prefill \
  --url http://127.0.0.1:8000 --model-path /path/to/model \
  --text-file /path/to/corpus.txt --trust-remote-code no \
  --deployment calibration-run --data-parallel-rank 0 \
  --prefix-lengths 32768 98304 196608 \
  --suffix-lengths 4096 8192 16384 32768 65536 \
  --warmup-repeats 1 --repeats 5 --seed 42 --split validation \
  --output calibration-output/validation-requests.jsonl
```

These HTTP commands collect the 40 grid cases. To match all 44 native cases, collect two additional cold runs with `--prefix-lengths 0`: `--suffix-lengths 131072 262144 --split train` and `--suffix-lengths 98304 196608 --split validation`, each into a separate request JSONL file with the remaining collection options unchanged. Keep the server log separately. HTTP wall time and native `wire_elapsed_s` are diagnostic only; neither is a fitting target.

## 3. Import validated server time

For native collection:

```bash
python -m test.cache_gds_calibration.calibrate import-prefill \
  --requests calibration-output/prefill/requests.jsonl \
  --server-logs calibration-output/prefill.log \
  --timing-source plain-request-stats \
  --queue-source server-accepted \
  --max-queue-ms 20 --max-prefix-shortfall 1 \
  --output calibration-output/prefill.jsonl \
  --rejected calibration-output/prefill-rejected.jsonl
```

For HTTP, pass all request JSONL files, including the additional cold runs, to `--requests` and the isolated server log to `--server-logs`. `plain-request-stats` uses `RequestStats.prefill_ms`: first forward dispatch to prompt completion, including normal chunk scheduling and submission gaps. It is not pure device kernel time. If P/D must remain enabled, collect P-side request statistics and 100% TTFT traces, select `--timing-source pd-prefill-trace --queue-source pd-admission`, and provide only P-side logs. The importer uses the trace's prefill phase, excluding transfer and decode waits.

Import joins request IDs, checks actual prompt/prefix lengths, one output token, completion, finite positive time, reported preemption and queue isolation. It rejects missing or conflicting records. Choose the queue source explicitly: `server-accepted` measures the server's `ts_request_accepted` event to first dispatch and is recommended for isolated native collection; `request-stats` starts at frontend request creation and also includes client encoding and transport; `pd-admission` uses the P-side trace's admission phase. Long token-ID prompts can spend appreciable time before server acceptance, which must not be confused with admission waiting. The selected source is recorded in imported data. Missing events or creation timestamps are rejected, without falling back to another timing source. Neither queue choice changes the fitting target, `prefill_ms`.

The `server-accepted` source accepts both UTC (`Z`) and numeric-offset (`+HH:MM` or `-HH:MM`) timestamps, including the engine's local-time logging format. Timestamps without a timezone are not interpreted using the collector's local timezone. If no matching acceptance event can be parsed, the rejection explains the required event and timestamp formats.

For compatibility with archived measurements, request JSONL retains `expected_a` (requested cached prefix) and `expected_b` (total prompt); imported timing JSONL retains `a` (actual cached tokens) and `b` (actual total prompt tokens). These persisted fields keep their original meanings. The case configuration uses `prefix_tokens` and `new_tokens`, and collector logs name the prefix, additional input and total prompt lengths explicitly.

Rejected samples are preserved separately and cause a nonzero exit. Investigate and recollect; do not silently discard failed requests. `preempt_count` is not a complete prefill replay counter. Audit scheduler batches to confirm computed prefill tokens equal `prompt_tokens - cache_tokens`, and exclude retraction or background load. A CPU forward range may time only asynchronous launch; use device events or an associated device trace when checking kernel execution time.

## 4. Measure Disk service cost

Record the serving SSD, filesystem, backend, queue limits and maximum I/O size. Use `specs.host_cache_block_bytes` from resolved initialization for each group: compact Host payload aligned to 4096 bytes. Do not substitute Device parent sizes, logical token counts or unaligned tensor bytes. Do not multiply a logical block's bytes by world size. Record owner geometry when a group is sharded.

Build the benchmark:

```bash
cargo build --release --manifest-path tokenspeed-engine/Cargo.toml \
  -p tokenspeed-disk --example gds_calibrate
```

Create a scratch parent on the actual local SSD, outside live cache files. The native benchmark exclusively creates and initializes its own directory and file, performs direct I/O through `ts_disk::Executor`, and removes its files on normal return or unwinding. There is no buffered-I/O fallback. Killing the process abruptly can leave its uniquely named scratch directory behind.

The sizes below illustrate an independent train/validation split. Replace or extend them to cover every real group's aligned size and any splitting at `max_io_bytes`. A model-specific calibration is incomplete without that coverage.

```bash
python -m test.cache_gds_calibration.calibrate disk \
  --bench tokenspeed-engine/target/release/examples/gds_calibrate \
  --directory /local/ssd/calibration-scratch --deployment calibration-run \
  --backend uring --inflight 1 --queue-depth 1 --max-io-bytes 4194304 \
  --file-bytes 134217728 --block-bytes 16384 65536 1048576 4194304 16777216 \
  --warmup-seconds 1 --seconds 5 --repeats 3 --seed 41 \
  --time-basis job-latency --split train \
  --output calibration-output/disk-train.jsonl

python -m test.cache_gds_calibration.calibrate disk \
  --bench tokenspeed-engine/target/release/examples/gds_calibrate \
  --directory /local/ssd/calibration-scratch --deployment calibration-run \
  --backend uring --inflight 1 --queue-depth 1 --max-io-bytes 4194304 \
  --file-bytes 134217728 --block-bytes 32768 262144 2097152 8388608 \
  --warmup-seconds 1 --seconds 5 --repeats 3 --seed 42 \
  --time-basis job-latency --split validation \
  --output calibration-output/disk-validation.jsonl
```

Benchmark `uring` corresponds to serving `--disk-tier-io io_uring`; `psync` corresponds to `--disk-tier-io psync`. The file budget rounds down to whole blocks and must hold every in-flight buffer. Each run writes the entire file before measuring reads: the example matrix initializes up to 3.375 GiB across 27 runs, plus warmup and measured read traffic.

Use several block sizes: at one size, the block-count and byte columns are proportional, so `A` and `B` cannot be identified separately. QD=1 `job-latency` is a service-cost baseline. Repeat separately with production concurrency to inspect p95 and bandwidth; `amortized` means `elapsed_s/completed_blocks`, a throughput cost rather than per-job latency. Never mix concurrency, backends or timing bases in one fit. The benchmark does not simulate Host landing credits, cross-owner ACKs, concurrent cleaning or all serving ranks sharing a drive.

## 5. Fit, validate and export

```bash
python -m test.cache_gds_calibration.calibrate fit \
  --prefill calibration-output/prefill.jsonl \
  --disk calibration-output/disk-train.jsonl calibration-output/disk-validation.jsonl \
  --weighting absolute --disk-scale 1 \
  --max-mape 0.15 --max-error 0.30 --max-relative-mad 0.10 \
  --output-dir calibration-output/fit
```

The fitter takes medians per geometry and solves normalized, two-column nonnegative least squares. It requires at least three training geometries, two held-out geometries and three repeats per geometry. Entire geometries, not repeated samples of the same geometry, must be held out. `absolute` fits time error; `relative` weights relative error. Select the objective and validation limits before collecting data. The example limits are engineering choices, not thresholds specified by the paper.

`report.json` records coefficients, conditioning, per-geometry errors, relative median absolute deviations, observed ranges and input SHA256 hashes. Failed validation keeps the report but writes no configuration. Passing validation additionally writes `gds_args.json` and `gds_args.txt`. Inspect systematic size/context errors even when the aggregate limits pass; more repeats cannot fix a mismatched cost model.

Merge the generated fragment into a deployment that supports GDS. Restore Host L2; Disk requires write-back L2. Retain the Disk capacity, backend, credits and write-budget configuration. All mirrored schedulers must use the same coefficients. `--disk-scale` multiplies only exported A/B; the report retains the measured values. A scale above one expresses a more conservative Disk preference, not a new bandwidth measurement. Revalidate after changing kernels, quantization, parallelism, chunk size or the supported context range.

## 6. Check cache-policy benefit separately

Calibration establishes cost estimates, not a GDS hit-rate improvement. Compare LRU and GDS with identical capacities, checkpoint settings, token IDs, request concurrency and tool gaps. Restart and warm the service for each parameter set. Use actual generated continuations so recurrent State identities remain reusable. Run both low-load and sustained I/O pressure workloads.

Measure missed computed prefill tokens/request, TTFT percentiles, throughput, Disk bytes/request, read bandwidth, prefetch waits and cleaning traffic. Without retraction, `prompt_tokens - cache_tokens` is a useful missed-token proxy; repeated replay requires summing actual computed batches. Keep cost-fit validation separate from policy tuning and report both recomputation saved and additional Disk work.

## Development checks

```bash
python -m unittest test.cache_gds_calibration.test_calibrate -v
cargo fmt --manifest-path tokenspeed-engine/Cargo.toml --all --check
cargo test --release --manifest-path tokenspeed-engine/Cargo.toml \
  -p tokenspeed-disk --example gds_calibrate
cargo clippy --release --manifest-path tokenspeed-engine/Cargo.toml \
  -p tokenspeed-disk --example gds_calibrate -- -D warnings
```

Run Rust formatting through Cargo so it uses the workspace's edition and formatting settings; the repository's pre-commit hooks do not run this check.

Python tests need no accelerator, model weights or serving runtime import. They exercise the real fitter, log importer, shared sampler, local HTTP exchange and module CLI. Rust tests exercise actual aligned I/O, split blocks, invalid geometry and scratch-file ownership; use a Linux filesystem supporting direct I/O. These checks do not replace measurements on the deployment being calibrated.
