# MiniMax-M3 vision encoder CUDA Graph A/B

This benchmark isolates the real MiniMax-M3 vision encoder CUDA Graph from
language-model graphing and from request concurrency. The benchmark client is
read-only with respect to server lifecycle: an external shell or the CI
pipeline must start a fresh server for each arm and stop it after collection.

## Fixed contract

Run four fresh TP4 server starts in counterbalanced `E1-G1-G2-E2` order from
the same TokenSpeed SHA, model revision, package set, physical GPUs, and server
arguments. `E1/G1` and `E2/G2` are the two independent launch pairs. The graph
arm adds exactly
`--enable-mm-encoder-cuda-graph`; the eager arm omits that one flag and the
collector verifies that the server reports the default `false`. Both arms use:

- `--enforce-eager --disable-prefill-graph`, so the language model cannot
  contribute CUDA Graph speedup;
- `--enable-log-mm-timing`, so every rank reports synchronized encoder time;
- `--no-enable-prefix-caching --multimodal-pixel-cache-mb 0`;
- greedy generation, one output token, and request concurrency one;
- the immutable `dog.jpg` fixture with SHA256
  `e1cd91db28149f21f3c410ffa074d0fb8bc8950740ba140c7eaae130f0493464`;
- the prompt and shape contract in
  `test/ci/reference/minimax_m3_dog_visual_reference.json`:
  input IDs `[1, 254]`, pixels `[308, 1176]`, grid `[[1, 14, 22]]`, and 77
  merged image tokens.

The collector always runs 10 warmup requests followed by 50 measured requests.
Those counts are intentionally not CLI options. It snapshots the server log
before every serialized request, requires one `encoder_ms` row from each TP
rank, and defines request encoder latency as `max(rank0, rank1, rank2, rank3)`.
This maximum is the TP critical path; averaging rank times would understate the
latency observed by the request.

Every arm artifact retains the raw HTTP response, raw per-request server-log
window and digest, per-rank H2D/encode time, rank spread, critical-path encode
time, client end-to-end time, and distribution statistics. Warmup compile
events are recorded but permitted. Statistics include median/P90/P95/P99,
mean, range, and population standard deviation. Any measured-window JIT,
Triton compile, or autotune event fails the arm.

## Collection

The manual paired CI contract is
`test/ci/perf/minimax-m3-mxfp8-encoder-graph-ab.yaml`. One B200 TP4 job creates
a fresh isolated Python environment, installs and audits the published SMG
dependency set, and then invokes the external lifecycle driver
`test/ci_system/minimax_m3_encoder_graph_ab_ci.py`.
The runtime-environment audit is a distinct `preflight` stage, so split GitHub
install/execute steps run it in the exact environment inherited by the four
server launches rather than attesting only the earlier install process.

That driver runs `E1-G1-G2-E2` on the same four GPU UUIDs and the same
installation. The reversed second pair counterbalances monotonic cache,
temperature, and runner drift, while two independent starts per arm let the
launch-cluster bootstrap estimate between-launch variation. Between every arm
it sends `SIGTERM` only to the real server root and requires exit code zero,
closed HTTP/control ports, vanished descendants/process group, and clean
selected GPUs. It also scans each final server log, records stable GPU
identity/driver/application and maximum clocks/power limit, and then runs
`compare`; therefore the numerical A/B gates directly determine the CI exit
status. Lifecycle remains outside the reusable collector.

The hosted task deliberately omits placement arguments, so the driver defaults
to `--base-gpu-id 0 --gpu-id-step 1` and selects physical GPUs `0,1,2,3`. For
the local acceptance machine whose reserved cards are `4,5,6,7`, pass
`--base-gpu-id 4 --gpu-id-step 1` to the CI driver. The driver derives exactly
four physical indices from those arguments and uses the same indices for the
server argv, hardware snapshot, idle check, managed-server ownership, and
shutdown validation. Do not combine this with `CUDA_VISIBLE_DEVICES`; the
runtime-environment preflight rejects visible-device masks.

For a locally managed server, invoke the collector directly. An eager example
is:

```bash
python3 test/quality_benchmark/tokenspeed/minimax_m3_encoder_graph_ab.py collect \
  --arm eager \
  --launch-id launch-1 \
  --base-url http://127.0.0.1:8000 \
  --server-info-base-url http://127.0.0.1:8001 \
  --model minimax-m3 \
  --dog .ci-artifacts/minimax-m3-encoder-graph-ab/fixtures/dog.jpg \
  --reference test/ci/reference/minimax_m3_dog_visual_reference.json \
  --server-log .ci-artifacts/server.log \
  --server-sha "$(git rev-parse HEAD)" \
  --provenance-file hardware=.ci-artifacts/minimax-m3-encoder-graph-ab/launch-1/eager/hardware.json \
  --provenance-file runtime_environment=.ci-artifacts/minimax-m3-encoder-graph-ab/runtime-environment.json \
  --provenance-file smg_packages=.ci-artifacts/minimax-m3-encoder-graph-ab/smg-packages.json \
  --provenance-file pip_install_report=.ci-artifacts/minimax-m3-encoder-graph-ab/smg-pip-install.json \
  --output .ci-artifacts/minimax-m3-encoder-graph-ab/launch-1/eager/arm.json
```

The paired CI driver creates one hardware JSON file for each of the four arms,
and the install/preflight stages create the other three shared provenance
files. A manual run must produce the same four provenance inputs for every
arm; an arm can be collected without them for debugging, but the release
comparator fails closed when they are absent.

Stop each server, prove its process group and GPUs are clean, and collect the
remaining `G1`, `G2`, and `E2` arms from the same SHA in that order. Keep the
same launch ID within each pair and use a distinct output path for every arm.
Never collect arms from concurrently running servers.

Compare the artifacts with:

```bash
python3 test/quality_benchmark/tokenspeed/minimax_m3_encoder_graph_ab.py compare \
  --eager .ci-artifacts/minimax-m3-encoder-graph-ab/launch-1/eager/arm.json \
  --eager .ci-artifacts/minimax-m3-encoder-graph-ab/launch-2/eager/arm.json \
  --graph .ci-artifacts/minimax-m3-encoder-graph-ab/launch-1/graph/arm.json \
  --graph .ci-artifacts/minimax-m3-encoder-graph-ab/launch-2/graph/arm.json \
  --output .ci-artifacts/minimax-m3-encoder-graph-ab/comparison.json
```

The comparator accepts repeated ordered `--eager` and `--graph` arguments,
rejects duplicate paths/launch IDs, and uses a paired launch-cluster bootstrap.
A one-pair comparison remains useful as a debug artifact but fails the release
gate; at least two independent pairs are required. Give each pair a distinct
shared `--launch-id`, then list corresponding eager and graph artifacts in pair
order.

## Gates

Collection fails unless every request returns exactly `Dog`, reports 254 prompt
tokens, and stays within 0.02 absolute sampled-logprob delta of the independent
reference. The comparator also requires every corresponding graph/eager
measured logprob pair to stay within 0.02.

Every comparison also requires identical hashes for the stable hardware,
runtime-environment audit, published SMG package manifest, and pip install
report. This prevents a runner or dependency change from being attributed to
CUDA Graph.

Each graph arm must report exactly one wrapper initialization, one nine-budget
completion, and one installation from each TP rank 0-3. When detailed capture
markers are enabled, every rank must also report each of the nine budgets
exactly once. Thus the 36-graph claim is a rank-keyed `4 x 9` matrix rather than
an uncorrelated total. Its structured graph-capture summary must remain
identical before and after requests, so request-time recapture fails closed.
The eager arm must report no encoder graph capture markers.

The performance gates use critical-path encoder time:

- at least two independent launch pairs must be present;
- pooled graph/eager median ratio must be at most 0.90 (at least 10% faster);
- the deterministic 95% bootstrap CI upper bound for that median ratio must be
  strictly below 1.0; repeated runs use a paired launch-cluster bootstrap with
  request resampling inside each selected arm;
- no individual launch-pair median ratio may exceed 1.05.

Client end-to-end latency and its bootstrap CI are retained as informational
evidence because HTTP, image preprocessing, and scheduling noise are outside
the isolated encoder path. W8A8 tuning warnings during ordinary model startup
affect both arms and are not encoder-only failures; compile or autotune activity
inside a measured request window is still a hard failure.
