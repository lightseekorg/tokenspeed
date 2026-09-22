# Kernel Benchmarks

This directory contains suites for measuring TokenSpeed operations and their
selected kernel registrations. The benchmark harness separates operation-specific
input and correctness logic from shared device timing and result reporting.

Suites are organized as `<vendor>/<arch>.json`, one per target platform.
Benchmark cases are grouped by model and operation family in files referenced by
the hardware suite. They use representative model inputs and exercise normal
operation dispatch, covering both kernel selection and execution.

Each operation family and mode owns one generator under
`tokenspeed_kernel/benchmark/generators/`. Built-in generators are loaded by the
harness, and additional ones are registered with `set_benchmark_generator`.
Every generator reuses the same harness, timer, and validators.

## Benchmark Requests

Each request identifies an operation family and mode, supplies parameters for
that operation's generator, and may select a solution or exact registration.
The generator interprets parameters such as shapes and data types and returns
the callable, arguments, and correctness work needed by the harness. For
example, a dense BF16 batched GEMM request against one exact registration:

```python
from tokenspeed_kernel.benchmark import (
    BenchmarkRequest,
    GraphBenchmarkConfig,
    GraphTimer,
    KernelBenchmarkHarness,
)
from tokenspeed_kernel.platform import current_platform

harness = KernelBenchmarkHarness(
    GraphTimer(
        GraphBenchmarkConfig(
            eager_warmup_iterations=5,
            replay_warmup_iterations=3,
        )
    ),
    platform_provider=current_platform,
)
result = harness.run(
    BenchmarkRequest(
        family="gemm",
        mode="bmm",
        parameters={
            "batch": 12,
            "M": 1,
            "N": 512,
            "K": 128,
            "dtype": "bfloat16",
            "validation": {
                "runs": 5,
                "atol": 0.015,
                "rtol": 0.015,
            },
        },
        solution=None,
        registration="gluon_bmm_a16w16_gfx950",
        cold_cache=True,
        seed=42,
    ),
    measurement_blocks=30,
)
```

Exact-registration benchmarks invoke the named registration through its normal
public behavior. Any internal fallback remains owned by the operation and is
not changed by the benchmark harness.

## Timing

The shared timer measures warmed graph replay of the selected registration.
Input creation, selection, compilation, eager warmup, graph capture, replay
warmup, correctness checks, and result serialization are outside the reported
device time.

By default, each captured invocation clears the device caches immediately
before the operation runs. Device events surround only the operation, so cache
clearing is excluded from its reported time. A case can set `cold_cache` to
`false` for hot-cache experiments.

Each measurement block reports the device time for one captured operation
invocation. Cases inherit the suite's `measurement_blocks` value unless they set
an explicit override. Eager and graph-replay warmup counts remain suite-wide.
Warmup settings and cache mode must match across revisions before measurements
can be compared; measurement-block counts may differ.

The result contains the raw device-time samples, resolved registration, timing
mode, and structured failure information. Suite-level comparison uses the
sample median and relative median absolute deviation.

## Correctness

Correctness is opt-in and owned by the operation generator. The generator
selects a registered reference solution, constructs candidate and reference
calls over the same fresh inputs, and declares a validator for each output that
needs checking. Outputs that do not require validation use no validator.

The generator declares how many fresh input sets to run, and each validator
receives every run's candidate/reference pair for its output along with
validator-specific options. The built-in `close` validator accepts absolute
and relative tolerances; additional validators are registered with
`set_output_validator`. Generators typically compare against the operation's
registered `reference` solution.

Correctness runs before timing. A failure prevents the case from producing a
successful measurement. Correctness remains within the revision-local process,
and tensor values are never serialized for cross-revision comparison.

## Suite Contract

A suite declares:

- the required hardware vendor and architecture;
- shared eager and graph-replay warmup counts, plus a default measurement-block
  count;
- stable case IDs, comparison epochs, and definitions; and
- per-case relative regression, absolute regression, and noise limits.

A case may set `measurement_blocks` when its noise profile requires more samples
than the suite default. The effective value is recorded with that case's result;
the suite's single benchmark harness receives that value for each case.

The optional `case_files` list composes model-and-operation case files into the
hardware suite. Case IDs remain unique across the composed suite.

A top-level parameter may be a non-empty list. The loader expands all such lists
as a Cartesian product and appends `_0`, `_1`, and so on to the case ID. Parameter
names are sorted alphabetically to determine dimension order; list order determines
each dimension's value order. A one-element product retains the original case ID.
Use separate case entries when shape classes need
different policies or do not form a Cartesian product. This expansion applies only
to direct parameter values; lists inside nested parameter objects remain literal.
Wrap a literal top-level list in an outer expansion list, such as `[[1, 2, 3]]`.
Changing list contents or order can remap numbered IDs, so treat the grid as part of
the case identity.

`comparison_epoch` is an opaque equality token used only to decide whether
baseline and candidate measurements are comparable. It has no ordering and
creates no backward-compatibility requirement. Advance it when an operation's
performance-relevant semantics change while retaining the same operation and
case identity. For example, adding a consumer fusion without renaming the
operation starts a new comparison epoch. Do not advance it for ordinary kernel
implementation changes or benchmark harness and CI changes. Those kernel
implementation changes are what the benchmark is intended to compare, while
timing-infrastructure compatibility is represented and checked separately.

Compatible cases must have the same ID, comparison epoch, definition, warmup
settings, cache mode, and recorded hardware. Their measurement-block counts may
differ. The resolved registration is reported but may change because dispatch is
part of the behavior under test. Added and changed cases are reported but not
compared. A baseline case missing from the candidate is also reported. If any
otherwise-compatible run is too noisy, its result is inconclusive rather than a
regression.

Regression policy comes from the merge-base suite, so a candidate cannot weaken
its own gate by changing a threshold. A benchmark is a regression only when its
median slowdown exceeds both the configured relative and absolute limits.

## Running A Suite

From the repository root and a prepared TokenSpeed kernel environment, run the
revision-local worker:

```bash
python3 -m tokenspeed_kernel.benchmark.ci \
  --suite tokenspeed-kernel/benchmarks/amd/gfx950.json \
  --revision "$(git rev-parse HEAD)" \
  --output /tmp/tokenspeed-kernel-result.json
```

Run a complete base/candidate comparison from the repository root:

```bash
python3 test/ci_system/kernel_benchmark_ci.py \
  --base-ref <target-commit> \
  --candidate-ref <candidate-commit> \
  --output-dir /tmp/tokenspeed-kernel-benchmark
```

The default comparison creates separate worktrees and virtual environments for
the two revisions and installs each revision's ROCm kernel requirements. Use
`--environment-mode current` to reuse an already prepared environment during
local development.

The output directory contains revision-local JSON results and logs, setup logs,
the structured comparison, and a Markdown summary. When the merge base does not
contain the suite, the comparison degrades to a candidate-only bootstrap
because no compatible baseline exists.

See the [CI documentation](../../test/ci/README.md#registration-level-kernel-benchmarks)
for GitHub Actions triggers, artifacts, pull request comments, and runner setup.
