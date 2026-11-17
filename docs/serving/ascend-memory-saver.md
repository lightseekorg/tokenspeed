# Ascend ACL Graph Memory Saver user guide

Memory Saver releases selected physical NPU memory between rollout windows.
Workers stay alive and wake with their original decode graphs and virtual
addresses. Only managed backing is released; budget separately for resident
executables, HCCL groups/workspaces and unmanaged allocations.

## Before you start

The primary baseline is single-host Atlas A2 (Ascend 910B3), Linux aarch64,
Python 3.12, PyTorch `2.9.0+cpu`, PTA `2.9.0-730` plus exactly six official
backports, and CANN/HCCL `9.0.0`. No additional PTA patches or system-library
replacements are required. The July 30 PTA base is
[`9d40ac3a2b3714df7e5d164bda28a42eff3282cd`](https://gitcode.com/Ascend/pytorch/commit/9d40ac3a2b3714df7e5d164bda28a42eff3282cd).
Build its wheel for the deployment's Python, PyTorch C++ ABI and CANN from the
[public branch](https://gitcode.com/Huyanlong/pytorch/tree/backport/pta-730-acl-graph-offload)
or [pinned snapshot `74e6508b896f630e5d5cacd39a3877047e91c19a`](https://gitcode.com/Huyanlong/pytorch/commit/74e6508b896f630e5d5cacd39a3877047e91c19a).
That snapshot and tested build `ee1742b06d2c0e59cb0b357288ec29ee190cfd2d`
have identical source tree `546ac928ce6199c1e2f221d7f5279b4eef6713a8`;
the reported commit can be either checkout identity.

| Required backport | Behavior |
| --- | --- |
| [!39221](https://gitcode.com/Ascend/pytorch/merge_requests/39221) | MemPool allocation/lifetime handling |
| [!45270](https://gitcode.com/Ascend/pytorch/merge_requests/45270) | Preserve allocator ownership during pool capture |
| [!30625](https://gitcode.com/Ascend/pytorch/merge_requests/30625) | Normal ProcessGroupHCCL shutdown |
| [!32154](https://gitcode.com/Ascend/pytorch/merge_requests/32154) | Stream synchronization during HCCL shutdown |
| [!32564](https://gitcode.com/Ascend/pytorch/merge_requests/32564) | Multi-stream capture task-queue ordering |
| [!38431](https://gitcode.com/Ascend/pytorch/merge_requests/38431) | Remove HCCL pending-event wait during capture |

Stock, unpatched TorchNPU `2.13.0` already contains these changes. It passed an
**earlier** single-host A2 compatibility run with CANN `9.0.0`; its official
pairing is CANN `9.2.0`. This route has not been rerun for the current integration.
Use a separate environment and the
[official matrix](https://github.com/Ascend/pytorch/blob/master/COMPATIBILITY.en.md)
to reproduce that historical route.

### Install

Use a dedicated virtual environment. Install the matching PyTorch/PTA and
TorchVision packages first; the primary environment used `torch==2.9.0+cpu`,
`torch-npu==2.9.0+gitee1742b`, `torchvision==0.24.0`. Preserve them during
subsequent installation with metadata-derived constraints (no device import):

```bash
export PIP_CONSTRAINT="$(mktemp)"
python - <<'PYVERSIONS' > "$PIP_CONSTRAINT"
from importlib.metadata import version
for name in ("torch", "torch-npu", "torchvision"):
    print(f"{name}=={version(name)}")
PYVERSIONS
```

Build runtime, kernels, saver and native host from the same checkout. The
matching SMG gateway supplies HTTP control. The kernel setup installs
Triton-Ascend 3.2.1 and Transformers 5.12.0.

```bash
export TOKENSPEED_CANN_ROOT=/usr/local/Ascend/cann-9.0.0
source "$TOKENSPEED_CANN_ROOT/set_env.sh"
TOKENSPEED_TVM_FFI_VERSION=0.1.13.post3 bash test/ci_system/install_triton_ascend.sh
TOKENSPEED_KERNEL_BACKEND=npu python -m pip install \
  -e ./tokenspeed-kernel/python ./tokenspeed-engine -e ./python

# Build the saver and export its library path as described in the native README.
export TORCH_DEVICE_BACKEND_AUTOLOAD=1
export PYTORCH_NPU_ALLOC_CONF=expandable_segments:False
export TORCH_HCCL_ZERO_COPY=0
export TASK_QUEUE_ENABLE=1 ASCEND_LAUNCH_BLOCKING=0
```

Follow the [native build and host ABI instructions](https://github.com/lightseekorg/tokenspeed/blob/main/tokenspeed-kernel-npu/memory-saver/README.md#build-and-abi)
before launch. Ascend requires protocol 3 and matching host/runtime modules;
an older host is rejected before model initialization. NPU exit protection is
automatic, including with Memory Saver disabled.

Keep backend autoload enabled. Leave `PYTORCH_NO_NPU_MEMORY_CACHING` unset or
`0`; bypassing the caching allocator bypasses managed pools and is rejected.
PTA must provide `NPUPluggableAllocator`, `MemPool`, `use_mem_pool` and user-pool
ACL capture with preserved ownership. Startup checks these APIs and verifies
parameter, KV and graph backing against live native VMM segments. Views may
share segments; eager serving needs no graph pool and captured pools may be
allocation-free. Qualify other PTA/CANN builds separately.

## Launch and policies

The example uses a complete local dense BF16 Qwen3 checkpoint as `CHECKPOINT_A`.
Memory
saving is opt-in and requires an explicit weight policy: `cpu_backup` restores
original parameters; `discard` avoids their host copy and requires complete
replacement before generation. This example chooses discard:

```bash
tokenspeed serve "$CHECKPOINT_A" \
  --host 127.0.0.1 --port 8000 --enable-rl \
  --device npu --tensor-parallel-size 1 --dtype bfloat16 \
  --attention-backend mha --sampling-backend greedy \
  --disable-prefill-graph --disable-pdl \
  --max-model-len 4096 --max-num-seqs 4 --max-total-tokens 16384 \
  --cudagraph-capture-sizes 1 4 \
  --enable-memory-saver --weight-memory-policy discard
```

| Tag | Policy |
| --- | --- |
| `weights` | Parameters follow the explicit weight policy. Persistent executor inputs, runtime state, attention metadata and shared workspaces always use a separate CPU-backed pool under this same tag, including with parameter discard. |
| `kv_cache` | Discard shared target/draft device KV; clear and synchronize on wake. |
| `acl_graph` | Discard graph-private temporaries while retaining the original executable and virtual addresses. Replay rewrites consumed values. Releasing this tag also discards resident device KV. Graph-only wake keeps KV asleep. |

Persistent backup consumes pinned host memory and adds transfer time. HCCL and
its workspaces stay resident with the default `TORCH_HCCL_ZERO_COPY=0`;
explicitly registered communication windows also have resident storage. Measure
reclaimed memory with driver free-memory readings: virtual reservation counters
can stay high. Free the trainer's device allocations before restoring serving.

## One checkpoint cycle

Send control calls through SMG's `/v1/rl` gateway or the headless message
transport. Stop rollout submission first. Wait for each call's successful
**result**, on every serving replica, before advancing; HTTP success alone is
insufficient. Each replica frontend relays to its TP ranks.
[Coordination and transition IDs](#memory-transition-protocol)

```bash
base=http://127.0.0.1:8000/v1/rl
rl() {
  curl --fail-with-body -sS -H 'Content-Type: application/json' \
    -d "$2" "$base/$1"
}

# 1. Drain and release, then train and publish checkpoint B.
rl release_memory_occupation '{"tags":["kv_cache","weights","acl_graph"]}'
# 2. Restore mappings, keeping KV asleep.
rl resume_memory_occupation '{"tags":["weights","acl_graph"]}'
# 3. Load the complete new target and commit on every rank.
rl update_weights_from_disk \
  '{"model_path":"/path/to/checkpoint-b","load_format":"auto","weight_version":"step-42","flush_cache":true}'
# 4. Only after successful update everywhere, clear/wake KV and resume rollouts.
rl resume_memory_occupation '{"tags":["kv_cache"]}'
```

Ascend updates require `flush_cache=true`, clearing L1/L2 and deleting the old
L3 namespace before parameter writes. Publish immutable checkpoints of the same
architecture at paths visible to every worker. Discard requires a complete
checkpoint and a model loading contract that can verify complete replacement.
File selection and loading follow the existing checkpoint loader; Memory Saver
does not introduce another format allowlist. The existing InstantTensor reader
requires CUDA/NCCL and is unavailable on NPU. Do not modify files during reload.

With `cpu_backup`, omit the update when keeping the original weights, or supply
a partial disk checkpoint or SDK update to replace selected weights. All target weight
allocations are backed up and restored; weights omitted from the update keep
their restored values. Partial reloads use the same general loader as CUDA and
retain the validation limits below.
Device KV still rebuilds; same-weight host prefixes remain reusable until an
update flushes them. With discard, always wake explicit tags so KV cannot
precede the new committed weights. A refusal before device/update work may be retried in
correct order; a device transition failure, failed allocation rollback or load
failure after a weight transaction begins requires worker restart.

### Memory transition protocol

The native host and Python runtime negotiate these control guarantees:

| Protocol | Meaning | Startup address payload |
| --- | --- | --- |
| `WEIGHT_UPDATE_PROTOCOL` (v2) | Existing weight-load preflight, result agreement and commit; memory operations retain mainline drain and synchronization. | Address string |
| `WEIGHT_UPDATE_WITH_MEMORY_TRANSITIONS_PROTOCOL` (v3) | v2 plus memory-domain agreement on release/resume requests, direction and tags, readiness/drain checks, and device-completion acknowledgements from every participating rank. | `(version, address)` tuple |

V3 extends v2's guarantees, but their startup formats cannot be mixed within
one model instance. A partial device failure keeps the replica unavailable;
v3 does not promise automatic rollback of every rank.

This change includes v3 in the runtime and native host. Checked startup reads
the Ascend adapter's protocol-integration capability
without creating an allocator or initializing a device. CUDA, ROCm and disabled
savers retain protocol 2. The Gloo address rendezvous rejects mixed protocols
before the Rust mesh or READY; model and graph startup options must also agree.

Send each participating ingress the same direction, canonical tags and optional
single-string `transition_id` (`rid` on the direct wire). Use a fresh ID for each
operation, or omit it for ordered calls. Success waits for every TP rank's device
completion. A pre-device refusal changes no mappings. HCCL stays resident;
communication release tags are rejected before any memory is released.

`continue_generation` cancels the whole observed pending batch before device
work and leaves already released memory asleep. If participants are missing,
the 30-second queue-head timeout refuses that whole observed batch while no
earlier release is draining. Late deliveries can reply later: collect continue
and every outstanding memory reply before retrying. Continue success alone
does not fence earlier fanout. A device-apply or post-readiness L1-clear failure
requires restarting every affected worker before training or serving resumes.

### Partial disk update limits

Safetensors preflight compares overlapping names with held checkpoint headers;
binary sources lack equivalent preflight. The shared loader does not validate
every global source shape before TP slicing or agree supplied inventories across
ranks. Strengthening these checks remains outside this change under
[#3456](https://github.com/lightseekorg/tokenspeed/issues/3456).

### Model Updater SDK

Install an Ascend-capable provider in every worker. It must supply ready CPU
or local-NPU tensors and retain their storage until copies finish. Set all four
flags explicitly:

| Flag | Value |
| --- | --- |
| `--model-update-config` | Provider configuration JSON object |
| `--model-update-sdk-module` | Installed provider import path |
| `--model-update-engine-type` | Provider's engine type |
| `--model-update-draft-weights` | `retain` for frozen independent drafts; `refresh` for eligible complete target/draft replacement in one version |

Replace step 3 with `rl update_weights_from_mooncake '{"version":42,"flush_cache":true}'`.
Optional `snapshot` carries publication metadata. With flush, the version string
becomes the cache namespace unless `weight_version` overrides it. Disk reload
replaces only the target. Under `cpu_backup`, independent draft weights keep
their restored values even when the SDK policy is `refresh`. Under `discard`,
a target-only disk reload is refused if the independent draft also needs
replacement. SDK refresh must publish both independent models together; the
target updates shared tensors.
Send full-shaped HF tensors for each rank's required names in the provider's
target-first model order; loaders perform rank-local slicing/fusion and validate
each model's inventory before all-rank commit.
[Provider and request fields](../configuration/server.md#mooncake-weight-updates)

## Validation coverage and limitations

- Memory Saver adds no model, attention, sampling or speculative-algorithm
  allowlist. Actual execution support remains with the existing runtime and
  kernels. Weight discard requires the existing complete-replacement and
  preserved-state contract; `cpu_backup` does not require that contract.
- Hardware qualification has covered single-host A2 TP1/2/4/8. Dense BF16
  Qwen3 and selected speculative configurations are test coverage, not startup
  eligibility. Other configurations, especially cross-host layouts, require
  their own qualification; removing a Saver gate does not implement a kernel.
- Captured narrowed draft attention uses Triton lengths computed on device and
  keeps overlap enabled. Do not force
  `TOKENSPEED_KERNEL_OVERRIDE_ATTENTION_MHA_DECODE_WITH_KVCACHE=torch_npu_mha_decode_with_kvcache`
  for this path: a narrowed query needs its causal KV frontier or absolute mask,
  not the full verification KV length.
- Complete target/draft SDK refresh used a synthetic Qwen3 MTP draft and test
  provider. Real TCP with CPU staging covered local TP2 and remote publisher/TP1;
  production providers, RDMA and native draft refresh require qualification.

For token-level restoration checks, match the actual per-rank execution batches,
not just the client request list. The wire driver submits requests individually,
so admission can split the same list differently before and after resume.
Use the existing scheduler pause/resume controls to queue each complete batch,
then verify its execution shapes and compare the first restored output.

On the primary PTA/CANN baseline, BF16 Qwen3 TP4/TP8 passed eight controlled
restoration cycles with this method. Different admission groups reproduced the
previously differing tokens with Saver disabled. These diagnostic runs used
`HCCL_DETERMINISTIC=true`; they do not establish batch-independent logits or
production throughput.

[Design invariants](../design/ascend-acl-graph-memory-saver.md)
