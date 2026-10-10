# PD scheduler: remote admission and sliding recovery regressions

These are scheduler-only regressions for prefill/decode disaggregation (PD).
They need no GPU, model weights, model-specific code, or TokenSpeed runtime.
Reproduction conditions and recorded results are separated below.
No new configuration, API, or version change is required.

## Remote admission must cover the whole suffix

**Trigger:** on D, a mixed full-history/sliding-window (SWA) prefix cache has
a common hit `P` shorter than the full-history hit `Q`, because SWA lacks the
lookback needed at `Q`. Local prefill must stop at that promotion boundary;
remote admission must not, since the peer transfers the whole unmatched suffix.
In `csrc/scheduler/operations/forward.cpp`, `schedulePrefillFirstChunk` must
use `unscheduled` for a remote source and retain `PrefillChunkTokens` for a
local source.

[`python/tests/test_remote_admission.py`](../python/tests/test_remote_admission.py)
exercises the following standalone cases:

- D role, prefix cache enabled, L2 disabled, prefix grain 64, Full grain 64,
  SWA grain 32/window 513, chunk budget 1024; decode widths 1/2/4 and overlap
  depths 0/1.
- Seed with `list(range(seed_length))`, complete remote prefill and one decode,
  then finish. Probe with `list(range(577)) + [9001] * 62` (639 tokens).
  Both seeds publish Full through 576, but the 634-token seed lacks SWA slot 2
  needed to resume at 576; the 600-token control retains it.

| Seed length | Expected common hit | Expected remote input length |
| --- | --- | --- |
| 634 | 0 | 639 |
| 600 | 576 | 63 |

The baseline reports input length 576 instead of 639 for the first case.
Both cases must keep `prefill_lengths == [639]`,
release the PD pin on remote completion, and retain valid Full/SWA slots through
four subsequent decode forwards, including the SWA page-boundary crossing.
Finishing must leave no active LCM blocks.

**Local control:** `test_decode_role_local_recovery_still_chunks_at_promotion`
uses width 4/overlap 1 and a 634-token prompt with `max_new_tokens = 8192`.
After remote completion and one decode, raising the reserve to 8192 forces
capacity retraction. The rebased 636-token prompt must still recover locally
as `(prefix, input_length) = (0, 576), (576, 60)`, with no remote operation.
The remote exception must not disable local promotion/chunking.

## Sliding capacity must cover local recovery

**Trigger:** a D-role request is retracted for capacity and recomputes locally,
even when L2 is disabled. A local recovery chunk plus retained lookback can
exceed the remote landing window. Sizing only for landing can admit a request
whose later recovery chunk cannot fit, even after other requests finish.

The D sliding branch in `csrc/scheduler/capacity_model.cpp` must cover the larger
of remote landing and local-prefill peak, capped by dense storage. Concurrent
sizing charges landing per live request and the recovery excess only once:
only one request can recover locally at a time. See
[Scheduler §1.4](../../docs/design/scheduler.md#14-what-bounds-a-single-request)
for the formulas. For grain 32, window 513, chunk 1024, width 1, overlap 0 and
a sufficiently long context, landing needs 33 pages (16 lookback + 17 window),
while recovery needs 49 (16 lookback + 33 for the chunk and decode reserve).

- `CapacityModelTest.DecodeSlidingCapacityCoversLocalRecovery` in
  [`cpp/test_capacity_model.cpp`](cpp/test_capacity_model.cpp) checks prefix
  caching on/off, overlap 0/1, dense caps for short contexts, and concurrent
  sizing including zero live requests.
- `PdSlidingRecoveryTestSuite.CapacityRetractionCompletesEveryRecoveryChunkAndResumesDecode`
  in [`cpp/test_outside_event_handler.cpp`](cpp/test_outside_event_handler.cpp)
  checks deterministic cold recovery of **one request**, not just first-chunk
  admission:
  1. Use D, prefix grain 64, Full grain 64 plus three SWA grain-32/window-513
     groups packed 1/1/5, chunk 1024, batch limit 2, width 1, overlap 0, no L2
     and prefix caching disabled. Size usable LCM blocks with
     `LcmBlocksNeededFor(SingleRequestGroupPages(6208))`, plus the null block.
  2. Submit only `running`, with 2048 prompt tokens and `max_new_tokens = 4100`.
     Complete remote prefill with bootstrap token 42, then deliver 4096 decode
     results to reach 6145 tokens.
  3. Send `forward::UpdateReserveNumTokens` with
     `reserve_num_tokens_in_next_schedule_event = 8192`. The next plan must
     retract: zero active LCM blocks and one waiting request. Require
     `ASSERT_TRUE(ClearL1Cache())` before readmission to evict the released
     request's own prefix; otherwise a hit at 6080 can mask the missing capacity.
  4. Expect all seven local recovery chunks from 0 through 6145, with no remote
     admission or cache transfer. The final prefill result brings the token
     count to 6146; two decodes reach 6148. Finish `running` and check empty
     waiting/decoding queues and zero active LCM blocks.

The forced reserve increase and L1 flush are **test controls**, not deployment
prerequisites. Real triggers are memory pressure followed by eviction of the
released prefix pages.

With the baseline model's 170 usable parents, the first 1024-token chunk fits:
live group pages `[97, 32, 32, 32]` consume 168 parents. The second chunk needs
`[97, 48, 48, 48]`, or 203 parents with the declared packing. Each SWA group now
holds 16 lookback + 32 new pages. This intermediate chunk has no final decode
reserve, so its actual demand is 48, not the worst-case bound of 49 (205 parents
across all groups). The baseline produced 11 empty plans after the first chunk,
with no PD pin, cache transfer, or in-flight output left to unblock it.
Controls with 203 or 205 parents completed; 202 still stalled.

The existing `PdSlidingSparseDecodeAdmissionTestSuite` tiny-pool fixture needs
10 total device pages instead of 8 (including the null page) under the stronger
recovery bound; its sparse-placement assertions remain unchanged.

## Recorded results

Baseline results use a fresh build of unfixed main (`c44c7f29`) with these
regression tests, without model-specific code or prefix-hash lookahead. The
C++ recovery test fails at prefix 1024 on that build; an independent Python
exercise also checks repeated empty plans and the 202/203-parent controls.

| Check | Baseline | Fixed |
| --- | --- | --- |
| Remote-admission Python tests | 6 failed / 7 passed; all seed-634 cases report 576 instead of 639 | 13 passed |
| Single-request bound at 6208 | 33 pages per SWA group; 170 usable parents | 49 pages per SWA group; 205 usable parents |
| Cold recovery | Stops at 1024; 11 subsequent empty plans | All 7 chunks complete; decode reaches 6148 |
| Targeted C++ capacity/recovery tests | Both fail: underestimated bound and empty second-chunk plan | Both pass |
| Whole scheduler suites | Not run on baseline | 582 C++ tests and 284 Python tests passed |

## Standalone build and run

Run from the repository root with a local Python virtual environment activated.
Install the scheduler build prerequisites from `tokenspeed-scheduler/pyproject.toml`
into that environment; CMake also needs a C++20 toolchain and OpenSSL development
files and may fetch GoogleTest. Use checkouts containing the standalone regression
tests, with identical tests for baseline and fixed implementations, and rebuild
each implementation. The unfixed base does not contain the new tests; an empty
test selection is not a regression result.

```sh
cmake -S tokenspeed-scheduler -B tokenspeed-scheduler/build-pd-regression \
  -G Ninja -DCMAKE_BUILD_TYPE=Release \
  -DTOKENSPEED_SCHEDULER_BUILD_TESTS=ON \
  -DTOKENSPEED_SCHEDULER_BUILD_PYTHON=ON
cmake --build tokenspeed-scheduler/build-pd-regression \
  --target tokenspeed_scheduler_tests tokenspeed_scheduler_ext -j4
tokenspeed-scheduler/build-pd-regression/tokenspeed_scheduler_tests \
  --gtest_filter='CapacityModelTest.*:PdSlidingRecoveryTestSuite.*:PdSlidingSparseDecodeAdmissionTestSuite.*'
```

Building the extension alone does not install the Python package. With the
scheduler prerequisites and pytest already installed in the environment:

```sh
python -m pip install --no-build-isolation --no-deps -e ./tokenspeed-scheduler
python -m pytest --noconftest -q tokenspeed-scheduler/python/tests/test_remote_admission.py
# Full suites:
tokenspeed-scheduler/build-pd-regression/tokenspeed_scheduler_tests
python -m pytest --noconftest -q tokenspeed-scheduler/python/tests
```

`--noconftest` avoids the root runtime conftest's GPU requirements. Verify that
`tokenspeed_scheduler` and its compiled extension come from the checkout under
test, not another installed build.
