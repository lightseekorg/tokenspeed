# Kimi K3 on Hopper with PD, DeepEP and DSpark

This layout uses 64 GPUs: four eight-GPU H200 nodes for each engine.
The runtime supports the configuration below; the GPU validation steps at
the end are required before treating a deployment as validated.

| Engine | Attention | Routed experts | Shared experts | Execution |
| --- | --- | --- | --- | --- |
| P | PP4, TP8, DP1 | EP8 within each stage, MoE TP1 | Stage TP8 | Eager, DeepEP normal |
| D | PP1, TP8, DP4 | EP32 across four nodes, MoE TP1 | Each replica's TP8 | Decode graph, DeepEP auto (low latency for decode) |

EP uses the same GPUs as attention. P and D both use attention TP8, so
CachePD does not change the target cache's TP shard width at handoff.

## Dependencies and model contract

Use the MXFP4 K3 target and a compatible `k3_dspark` checkpoint. Marlin
executes packed W4A16 routed experts with BF16 activations. This Hopper
example uses FlashMLA with BF16 target and draft caches. Keep target cache
precision identical on P and D. The current `tokenspeed_mla` implementation
uses Blackwell instructions and must not be selected on H100/H200.
Build the kernel package with its Marlin CUDA extension on the GPU hosts.
`--kda-backend flashkda` requires the matching FlashKDA package; it selects
the prefill scan, while KDA decode/verify uses the platform-selected kernels.

For Linux H100/H200 hosts with CUDA Toolkit 12.9 and an active Python 3.11
virtual environment, install this checkout through the dedicated recipe:

```bash
CUDA_VARIANT=cu129 bash test/ci_system/install_deps.sh
```

The [Hopper/cu129 installation guide](hopper-cu129.md) describes the host
prerequisites and dependency sources. The command rebuilds the checkout's
kernel and scheduler packages; it does not start the P, D, or SMG services.

DeepEP must provide the legacy `Buffer` API with BF16 low-latency support
for the checkpoint's latent width and top-k. The standard K3 geometry is
3584 latent dimensions, top-k 16, and 896 experts: 112 per P rank and 28 per
D rank. The normal interface returns local expert IDs, which feed Marlin's
local routing entry directly. The public global-ID entry translates IDs once
before using the same local computation. Low latency retains its compact,
device-counted schedule and shares the SiTU elementwise implementation with
the dense layout. K3 does not require an identity-expert `x_ori` argument on
low-latency combine.

EP32 low latency requires the installed DeepEP build's inter-node transport
prerequisites. Use consistent dependency builds on all ranks. For initial
validation, `--load-format auto` keeps checkpoint loading independent of
InstantTensor; optional InstantTensor context loading uses each P stage's
TP group because stages consume different weight subsets.

## Engine arguments

Add these argument sets to the existing engine launcher. Supply target and
draft paths, node rank, rendezvous address, gateway transport, context
length and memory utilization separately. Run one launcher per node with
eight GPU workers and use separate P/D rendezvous addresses. These argument
sets do not configure or start an SMG gateway.

Common arguments:

```text
--model <K3_MODEL_PATH>
--speculative-draft-model-path <K3_DSPARK_MODEL_PATH>
--speculative-algorithm DSPARK
--speculative-num-draft-tokens 8
--dtype bfloat16
--kv-cache-dtype bfloat16
--attention-backend flashmla
--drafter-attention-backend flashmla
--kda-backend flashkda
--moe-backend marlin
--all2all-backend deepep
--world-size 32
--nnodes 4
--nprocs-per-node 8
--attn-tp-size 8
--dense-tp-size 8
--moe-tp-size 1
--disable-kvstore
```

Verify width includes the anchor: eight target rows use seven DSpark
proposal queries. Select a width supported by the draft checkpoint.

P arguments:

```text
--disaggregation-mode prefill
--pipeline-parallel-size 4
--data-parallel-size 1
--expert-parallel-size 8
--deepep-mode normal
--mm-encoder-tp-mode data
--max-num-seqs 32
--chunked-prefill-size 8192
```

P is eager by role. Its layer windows partition the target decoder layers,
excluding draft layers. The automatic split of 93 layers is `24,23,23,23`.
`--pp-layer-partition 24,24,24,21` is another candidate: its first three
boundaries align with the standard AttnRes blocks and the last stage owns
fewer target layers alongside context writing and draft execution. Measure
stage time before choosing a partition for throughput. KDA geometry can
reduce the effective prefill chunk size to align state checkpoints;
startup logs report it.

D arguments:

```text
--disaggregation-mode decode
--pipeline-parallel-size 1
--data-parallel-size 4
--expert-parallel-size 32
--deepep-mode auto
--max-num-seqs 128
--max-cudagraph-capture-size 32
```

`auto` keeps ordinary decode on low latency and CUDA graph. If cache
pressure retracts a request and the scheduler performs a recovery prefill,
that extend-shaped work uses normal dispatch and eager execution. Disabling
KVStore does not eliminate recovery prefills.

`max-num-seqs` is global across DP: this example allows 32 requests per D
replica and captures through that local batch size. Keep graph padding
enabled and omit `--enforce-eager`. An explicit capture-size list must
cover the intended workload too. These settings do not establish a tested
maximum context capacity or memory utilization.

## Token layout and capacity

Each TP rank takes a disjoint slice of its attention batch before routing
and latent projection. DeepEP carries routed latent work and returns the
combined result to the source slice. A TP8 token gather restores that
replica's latent batch before normalization and the shared/up-projection
TP reduction. No full hidden-state or residual gather replicates the four
attention-DP batches across EP32.

Automatic low-latency capacity for token-sliced K3 in `auto` mode is:

```text
ceil((max_num_seqs // attention_DP) * verify_width / attention_TP)
```

The D example allocates 32 source rows per EP rank. A smaller graph ladder
does not reduce this capacity: eager batches above the ladder remain valid.
An explicit `--low-latency-max-num-tokens-per-gpu` below the required bound
fails at startup; a larger explicit value is respected. Other model paths
retain their previous automatic value of 256. If `low_latency` is pinned,
the capacity also covers the configured prefill/recovery chunk beside the
decode batch before TP slicing, because no normal buffers exist. For the
D example and an 8192-token chunk this requires 1056 rows per source rank
and substantially more memory;
an explicit decode-only capacity such as 32 is rejected. Prefer `auto` for
the D engine.

DeepEP's own receive buffers still reserve expert capacity. The Marlin
bridge uses device counts to construct aligned work and bound intermediate
storage by source routes, avoiding SiTU work over the entire
`experts * capacity` padding extent. Communication buffers are prepared
with weights before KV memory profiling. Measure persistent memory and
capture/runtime peaks separately.

Empty slices and idle DP ranks participate in both EP legs. The target's
persistent cache write slots identify graph padding on the GPU; those
routes use ID -1 and weight zero. Python counts recorded during capture
are not used as the live-token mask.

## Context production and handoff

A *tap* is one selected intermediate target hidden stream, with one vector
per token. The draft checkpoint specifies which layers and stream to read.
For example, `target_layer_ids=[2,23,47,71,89]` selects five taps; these
zero-based IDs refer to completed target layers. The number of taps is
independent of the draft network's depth and the four PP stages.

For `aux_hidden_stream=prefix`, each tap reads the prefix stream after its
named layer. For `attn_res`, it reads the result of the next consumer's
attention mixing; the final layer uses the output mixing. A tap's owner is
the stage with the weights needed to produce that stream.

K3 capture selection is configured once after model loading through
`K3DSpark.configure_target`, on every P stage. Later K3 runtime wiring
binds model references and shared weights. Per-forward hooks use that
configuration without selecting the taps again. This setup is specific
to the K3 drafter.

P stages project DSpark taps as they appear and pass one FP32
`[tokens, draft_hidden]` accumulator alongside ordinary PP state. Per-tap
`fc_norm` precedes projection; `context_norm` follows the complete sum.
An AttnRes tap requiring the next stage's first layer is evaluated at that
stage's entrance using the incoming PP state.

Each stage loads the tap projection parameters it uses. Only the last P
stage also owns the complete DSpark proposal network and the draft cache.
It writes every draft layer's prompt context KV, samples the target's first
output token, and runs the ordinary drafter to generate real candidates
for that token. Proposal execution uses the target's output head.

The draft checkpoint must include `embed_tokens.weight` as a frozen copy
of the target embedding. The final P stage loads its TP shard because the
target embedding lives on the first stage; non-PP execution on D continues
to borrow the target embedding. Include the final stage's proposal weights,
embedding shard, activations and proposal execution time when sizing memory
and choosing the PP partition.

All P stages share one logical cache geometry, with target physical fields
assigned by stage and draft fields owned only by the last stage. The same
ownership drives producer readiness, sender filtering and receiver routes.
A final producer barrier follows the completed drafter call, covering all
draft cache writes. Bootstrap metadata explicitly distinguishes the target
cache-layer count from the draft continuation layers, so adding the draft
cannot move PP boundaries.

P sends its sampled bootstrap token and the complete candidate window:
the anchor followed by DSpark proposals. D installs this window before its
first ordinary verify round, which can accept the supplied proposals and
produce the next candidate window through its local drafter. MLA, KDA
recurrent/conv state and draft KV must all arrive before that round. The
handoff uses the existing candidate transfer and verify path.

## Validation

Run the existing cache-transfer, model-configuration and Marlin tests in the
configured runtime environment. Validate the full serving topology on the
GPU hosts:

1. Compare Marlin with the dequantized reference, then test DeepEP normal
   EP8 and low latency EP32 at actual K3 geometry. Include nonzero EP ranks,
   empty/unequal source batches and skewed expert routing.
2. Compare eager and repeated graph MoE results while changing routes,
   active DP ranks and live counts inside one capture bucket. Include
   batches below TP8 and padded verify rows.
3. Compare concatenated context projection with PP accumulation, including
   `fc_norm`, both configured tap streams and PP boundaries. Floating-point
   addition order differs; compare errors and finite values rather than
   requiring BF16 bitwise equality.
4. Validate P-to-D target/cache behavior without speculation, then enable
   DSpark and compare initial logits, generation and acceptance. Verify
   that P transfers its sampled anchor and actual proposals, and compare
   the first D verify round in eager mode and with graph replay.
5. Exercise chunked prompts, prefix hits, page boundaries, concurrency,
   cancellation and repeated slot reuse. Keep graph enabled with unequal
   DP batches. `TOKENSPEED_GRAPH_DEBUG=1` checks metadata pointer stability
   during correctness runs.
6. Measure realistic prompts across concurrency levels. Record P computed
   input tokens/s, D output tokens/s, TTFT/TPOT, acceptance, per-rank memory
   peaks and errors, with exact arguments and dependency versions.

Disabling graph is a diagnostic comparison, not completion of the decode
graph requirement. No GPU performance or stability result is implied by
the presence of this configuration.

### Focused test commands

From the repository root, run the existing cache-transfer and model tests
in the configured runtime environment:

```bash
PYTHONPATH=python:tokenspeed-kernel/python:tokenspeed-scheduler/python:test \
python -m pytest -q \
  test/runtime/distributed/test_pd_transfer_plan.py \
  test/runtime/distributed/test_cache_pd_manifest.py \
  test/runtime/test_cli_config_compat.py \
  test/runtime/test_kimi_k3_cache_spec.py \
  test/runtime/test_model_executor_cache_state.py
```

Also run the existing K3 DSpark/capture tests. Keep runtime and kernel test
roots in separate pytest
invocations to avoid their conflicting `conftest` module names.

For the existing one-node EP8 normal-dispatch smoke test:

```bash
TEST_DEEPEP_MODE=normal \
PYTHONPATH=python:tokenspeed-kernel/python:tokenspeed-kernel/test \
torchrun --standalone --nproc-per-node=8 -m pytest -q \
  tokenspeed-kernel/test/ops/moe/test_marlin_deepep_distributed.py
```

For the corresponding EP8 low-latency smoke test:

```bash
TEST_DEEPEP_MODE=low_latency \
PYTHONPATH=python:tokenspeed-kernel/python:tokenspeed-kernel/test \
torchrun --standalone --nproc-per-node=8 -m pytest -q \
  tokenspeed-kernel/test/ops/moe/test_marlin_deepep_distributed.py
```

These smoke tests use small expert matrices. Full K3 geometry, EP32 transport
and graph replay need the GPU validation described above. Run normal and
low-latency tests in separate process invocations; the DeepEP buffer is
process-scoped.
