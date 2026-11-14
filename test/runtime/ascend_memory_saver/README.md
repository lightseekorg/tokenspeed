# Ascend memory saver acceptance tests

## Regression after shared persistent-buffer integration

The targeted integration is tested as source tree
`9cdaec776e2b349e6c4388a4e5f9477b6b932fa9` (3,815 paths), based on main
`0cecb81ee588775224c7177c1db8595cc6937a0f` and common protocol source
`97400c6eb136254021a2b1880a3e35104e4e9e5a`. The current integration pins common
head `a034c3cc585f5a5aa9f6f1d811f8624e66549466`, whose later changes are
CUDA test/README changes, separate from the executed NPU test tree. Its CPU/native
gates are accepted. All ten targeted NPU hardware cases are accepted: four
standalone combinations,
TP2 eager/graph recovery, two-cycle synthetic MTP, live cancellation/retry and
eager/graph callback ordering. One actual integrated CUDA TP2/EP1 eager v2
CPU-backup full-update/fresh-reference case is accepted separately. The executed
C tree is exactly commit `ddf4af615e7890e3ad5f8b2c53778f61ebf6b3b3`. The later
delivery delta is limited to eight CUDA test/README paths and four documentation
paths; production, native and NPU-test code remain unchanged. Earlier hardware
passes are not inherited. The current result summary
and source boundaries are in the
[validation report](../../../docs/serving/ascend-memory-saver-test-report.md#regression-after-shared-persistent-buffer-integration).

The four named owners use a distinct weights-backed `CPU_BACKUP` pool even
when parameters use `DISCARD`. These tests distinguish persistent contents from
parameter, KV and graph policies. They require the real qualified PTA/native
stack, with no additional PTA patches, and matching newly built TokenSpeed
native artifacts. Run from the repository root, after configuring the documented
CANN/PTA environment and saver/host variables. Select only the devices assigned
to the run. The commands below are opt-in instructions, not recorded passes.

### Standalone policy and execution-mode matrix

Expose one physical NPU for this matrix. Bind the saver digest to the qualified
artifact before running; do not substitute a different library to satisfy a
failure.

```bash
suite=test/runtime/ascend_memory_saver
RUN_OUTPUT=$(mktemp -d)
PTA_GIT=ee1742b06d2c0e59cb0b357288ec29ee190cfd2d
: "${QUALIFIED_SAVER_SHA256:?Set the saver SHA256 from the matching qualified build receipt}"

for policy in cpu_backup discard; do
  for mode in eager graph; do
    python "$suite/test_persistent_buffer_roundtrip.py" \
      --source "$PWD" \
      --weight-policy "$policy" --execution-mode "$mode" \
      --expected-pta-git "$PTA_GIT" \
      --expected-library-sha256 "$QUALIFIED_SAVER_SHA256" \
      --output "$RUN_OUTPUT/persistent-$policy-$mode.json"
  done
done
```

The probe checks actual named owner/workspace allocation, the distinct real PTA
pool and native backing, full persistent values and the same virtual addresses
after weight release/resume. Graph mode additionally checks its declared replay
scope. These standalone tensors do not qualify TP2 model constants or a full
model-serving workload.

### TP2 full disk update and ownership witnesses

Expose two physical NPUs. Set `INITIAL_CHECKPOINT` and `UPDATED_CHECKPOINT` to
distinct compatible local Qwen3 checkpoints and set `TOKENSPEED_HOST_BINARY` to
the matching native host. Retain the package source on `PYTHONPATH` so spawned
workers resolve the same tree. Each output subdirectory must be new.

```bash
suite=test/runtime/ascend_memory_saver
RUN_OUTPUT=$(mktemp -d)
export PYTHONPATH="$PWD/python${PYTHONPATH:+:$PYTHONPATH}"

for mode in eager graph; do
  python "$suite/test_tp2_weight_recovery.py" \
    --initial "$INITIAL_CHECKPOINT" --updated "$UPDATED_CHECKPOINT" \
    --trace-dir "$PWD/$suite/weight_trace" \
    --execution-mode "$mode" \
    --output-dir "$RUN_OUTPUT/tp2-$mode"
done
```

The recovery test requires full updated-B output equality with an independently
loaded fresh B, real parameter discard/reload, mandatory named-buffer
PTA/native ownership witnesses, and original graph identity/replay with zero
recapture in graph mode. Stable addresses alone are not ownership or content
proof. Accepted TP2 recovery evidence does not qualify full model-constant
contents. The ownership checker keeps positive half-open ranges, full native/PTA
joins, policy/tag classification and source-origin guards; do not disable these
checks to turn a failure into a pass.

The accepted targeted scope also includes live cancellation/refusal with healthy
retry, eager/graph callback-return versus frontend-response observations, and
two cycles with an existing Qwen3 synthetic-MTP fixture. A synthetic MTP smoke
does not qualify actual EAGLE3, DFLASH or DSPARK checkpoints, complete draft
refresh, or a broad speculative-decoding matrix. The callback's artificial
300 ms hold does not prove native asymmetric KV drain, native ACK timestamps,
global exactly-once wire behavior or production performance.

This round does not rerun the historical full GSM8K or TP4/TP8 matrix. Existing
result statements below keep their original source qualification. The command
reference remains available without implying a new execution of every probe.

## Existing command reference

These opt-in tests exercise the real Ascend allocator and TokenSpeed serving
path. Use one of the [documented PTA stacks](../../../docs/serving/ascend-memory-saver.md#before-you-start)
and load its CANN environment. The original runs used Atlas A2, CANN 9.0.0,
and backported PTA `ee1742b`. The supplemental A2 Docker run used stock
TorchNPU 2.13.0 with CANN 9.0.0. Build the saver library with
[the backend instructions](../../../tokenspeed-kernel-npu/memory-saver/README.md).
When running the Python package from source, use the `ts-host` binary built
from the same TokenSpeed revision.

Run the following commands from the repository root. Set the physical NPU index
and supply an absolute path to the saver library.

```bash
export ASCEND_RT_VISIBLE_DEVICES=0
export TASK_QUEUE_ENABLE=1
export PYTORCH_NPU_ALLOC_CONF=expandable_segments:False
export TOKENSPEED_NPU_MEMORY_SAVER_LIBRARY=/path/to/libascend_memory_saver.so
# Set TOKENSPEED_HOST_BINARY if the matching ts-host is not in the environment.

suite=test/runtime/ascend_memory_saver
work=$(mktemp -d)
python "$suite/test_npugraph_vmm_roundtrip.py" \
  --library "$TOKENSPEED_NPU_MEMORY_SAVER_LIBRARY" \
  --output "$work/vmm.json"
python "$suite/test_ascend_pool_policy_probe.py" --output "$work/pool-policy.json"
```

The VMM probe checks weight and graph-pool backup, KV discard, physical memory
release, preserved addresses, replay of two graphs sharing a pool, and final
cleanup. Use `--teardown-paused` to check cleanup with the regions paused.
The pool-policy probe uses synthetic target and draft tensors to check weight
replacement, RoPE backup, resident storage, KV rebuild, and replay with shared
and independent pools after their allocations are remapped.

## Serving decode graph discard

```bash
python "$suite/test_acl_graph_discard.py" \
  --model /path/to/Qwen3-0.6B --sampler greedy --cycles 3 \
  --output-dir "$work/graph-discard"
```

The rank trace checks both startup batch buckets and verifies that the
private pool remaps at the original addresses without a Host snapshot. It
fills the remapped pool with nonzero patterns before the first replay of
each bucket after wake, then checks the tokens against a fresh engine.
Pass `--drafter DUMMY` for model-free verification or `--drafter MTP` with
`--draft-checkpoint /path/to/draft` for the three-step MTP chain. Use
`--drafter EAGLE3`, `--drafter DFLASH` or `--drafter DSPARK` with a Llama
target and the corresponding native draft checkpoint.
A draft model must explicitly declare `supports_acl_graph_discard` after
its capture path has been audited. Use `--expect-backup` to check the
compatibility path for a draft that has no declaration.

## Complete checkpoint replacement

Use distinct local checkpoints for the same dense BF16 Qwen3 model. The
reference loads the updated checkpoint in a fresh engine.

```bash
python "$suite/test_weight_discard_checkpoint.py" \
  --initial /path/to/checkpoint-a --updated /path/to/checkpoint-b \
  --output-dir "$work/weight-discard"
```

The check verifies discard and remapping of the weight blocks, in-place disk
reload, reuse of the original graph, and matching output tokens. Pass
`--draft /path/to/mtp-draft` to exercise a compatible MTP draft with the
`retain` policy. The test then checks for actual verify rounds and independent
draft backup blocks.

Add `--mooncake-fixture` to exercise the same cycle through the external Model
Updater SDK interface. Add `--snapshot` to include publication metadata, or
`--incomplete /path/to/incomplete-checkpoint` to verify a partial update
blocks KV wake. `--draft` exercises retain and `--updated-draft` exercises
refresh with a compatible draft that opts in to the complete update contract.
Add `--verify-namespace` to the incomplete SDK test to check that L3 still uses
its original namespace after the load fails.
Use a local checkpoint with one `model.safetensors` file per model for this
fixture.

With the official non-CUDA Mooncake Python package installed, pass
`--store-master 127.0.0.1:PORT` for a local Mooncake master. A separate
publisher and worker use the Store's TCP transport with CPU staging. The
checkpoint object layout and snapshot directory in this mode belong to the
test fixture. Production publication and transport compatibility depend on
the selected SDK provider.

## Qwen3 GSM8K benchmark

Fetch the pinned official 1,319-question test split. The benchmark validates its
SHA-256 independently.

```bash
dataset_revision=3101c7d5072418e28b9008a6636bde82a006892c
dataset_url="https://raw.githubusercontent.com/openai/grade-school-math/$dataset_revision/grade_school_math/data/test.jsonl"
curl --fail --location "$dataset_url" -o "$work/gsm8k-test.jsonl"
printf '%s  %s\n' 3730d312f6e3440559ace48831e51066acaca737f6eabec99bccb9e4b3c39d14 \
  "$work/gsm8k-test.jsonl" | sha256sum -c -

run="$work/tp1"
mkdir "$run"
set -o pipefail
python "$suite/test_gsm8k_pause_resume.py" \
  --model /path/to/Qwen3-0.6B --dataset "$work/gsm8k-test.jsonl" \
  --output-dir "$run/benchmark" --tp-size 1 \
  --concurrency 32 --max-tokens 512 --warmup 32 2>&1 | tee "$run/runner.log"
python "$suite/test_summarize_gsm8k_runs.py" --output-dir "$work/summary" "$run"
```

The benchmark runs both passes in one headless service and defaults to pausing
only the graph pool. Use `--tags kv_cache weights acl_graph` for all managed
regions. Use `--limit 4 --concurrency 2 --max-tokens 16` for a smoke run.
The probes accept `--pta-commit` to verify the expected PTA identity.
Use `--expected-graph-policy backup` for a model that uses Host backup.

Both passes use BF16, zero-shot prompts, greedy decoding and Qwen's non-thinking
chat template. Accuracy compares the last boxed numeric answer with the reference.
Wire TPOT measures first-to-last token time per remaining token. The server
estimate uses the logged `decode_tps`. Wall time covers first submission through
last completion, excluding startup, warmup and the separately reported pause
and resume calls. The baseline warms up before measurement. The resumed pass
starts with its first measured request. The per-rank trace checks address
preservation, physical release and remapping, and reuse of captured graphs
without another capture.

Run targeted controls using the result directory of a completed full pass.

```bash
python "$suite/test_gsm8k_targeted_control.py" \
  --model /path/to/Qwen3-0.6B --dataset "$work/gsm8k-test.jsonl" \
  --output-dir "$work/serial-control" --tp-size 1 \
  --original "$run/benchmark" --selection correctness --correctness-cap 12 \
  --concurrency 1 --mode aa-then-pause

python "$suite/test_gsm8k_targeted_control.py" \
  --model /path/to/Qwen3-0.6B --dataset "$work/gsm8k-test.jsonl" \
  --output-dir "$work/concurrent-control" --tp-size 1 \
  --original "$run/benchmark" --selection changed-plus-control --target-count 96 \
  --concurrency 32 --mode aa
```

## Multi-rank HCCL and Qwen3 validation

Run on a single node with 2, 4, or 8 free NPUs exposed. Reserve all selected
devices for the duration of each run. Use the same PTA build and environment
as above. The probe uses the ordinary HCCL subgroup over the visible ranks.

```bash
tp=2
export ASCEND_RT_VISIBLE_DEVICES=0,1
probe="$suite/test_npugraph_hccl_roundtrip.py"
for mode in eager graph; do
  python -m torch.distributed.run --standalone --nproc-per-node="$tp" "$probe" "$mode"
done
python -m torch.distributed.run --standalone --nproc-per-node="$tp" "$probe" vmm \
  --library "$TOKENSPEED_NPU_MEMORY_SAVER_LIBRARY"
```

Eager checks communication setup and input-dependent AllReduce and AllGather.
Graph captures those collectives with ordinary allocation. VMM uses a managed
graph pool and checks both collectives on every rank before and after each
release and remap. It checks stable addresses and real physical-byte release.
The HCCL group remains live throughout. Host barriers coordinate only the
transitions. Explicit HCCL memory registration and native communication windows
have separate lifetimes. This probe covers ordinary collective capture only.

Run the Qwen3 benchmark for each TP width with `--tp-size "$tp"`. Start with
serial greedy decode and a short smoke run, first with `--no-pause-control`
and then with `--tags kv_cache weights acl_graph`. Repeat with the full dataset
and intended concurrency. Compare token IDs, exact-answer accuracy, capture
count and memory transitions against the same-TP no-pause control. Historical TP4 and TP8 runs had token differences even with serial
requests and no pause. At a TP width supported by the model's discard
contract, add `--weight-memory-policy discard` to exercise the rejected
release and wake orders, successful retry, and the checkpoint commit before
KV wake. Test greedy MTP separately with an eligible
draft checkpoint because its verify shapes exercise different kernels.
