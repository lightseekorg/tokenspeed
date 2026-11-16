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

Follow the [native build and host ABI instructions](../../tokenspeed-kernel-npu/memory-saver/README.md#build-and-abi)
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

Set `CHECKPOINT_A` to a complete local dense BF16 Qwen3 checkpoint. Memory
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
| `acl_graph` | Runtime-selected discard for audited BF16 MHA/greedy captures; CPU backup for other supported captures. Releasing this tag also releases resident KV, under either policy. Graph-only wake keeps KV asleep. |

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
[Shared coordination and transition IDs](weight-memory.md#one-update-cycle)

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
checkpoint. Direct readers `auto`, `safetensors`, `pt` and `mistral` are supported
within the model contract;
NPCACHE, InstantTensor, dummy and custom loaders are refused under either
weight policy. `auto` refuses mixed binary/Safetensors families; an explicit
format can select one family. Declared exact checkpoint files must exist.
Format preflight precedes cache invalidation and the transaction, so a corrected
request can retry the same version. Do not modify files during reload.

With `cpu_backup`, omit the update when keeping the original weights, or supply
a partial disk checkpoint to replace selected weights. All target weight
allocations are backed up and restored; weights omitted from the update keep
their restored values. Partial reloads use the same general loader as CUDA and
retain the validation limits below.
Device KV still rebuilds; same-weight host prefixes remain reusable until an
update flushes them. With discard, always wake explicit tags so KV cannot
precede the new committed weights. A refusal before device/update work may be retried in
correct order; a device transition failure, failed allocation rollback or load
failure after a weight transaction begins requires worker restart.

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
with a draft requires `retain` and replaces only the target. SDK refresh must
publish both independent models together; the target updates shared tensors.
Send full-shaped HF tensors for each rank's required names in the provider's
target-first model order; loaders perform rank-local slicing/fusion and validate
each model's inventory before all-rank commit.
[Provider and request fields](../configuration/server.md#mooncake-weight-updates)

## Supported scope and limitations

- Single-node aggregate TP1/2/4/8 only; attention, dense, MoE and vision TP
  groups must span the whole worker group. Cross-node serving needs separate
  enablement, hardware validation and deployment supervision.
- Dense BF16 Qwen3 has the complete target disk/SDK discard contract.
  Quantization and online expert rebalancing are outside it. Graph discard and
  weight discard have independent eligibility checks.
- Greedy chain speculation uses eligible model/kernel combinations. Fixed-width
  Qwen3 MTP with default paged MHA checks supported verify widths before process
  groups start; A2 registrations support widths through five. Audited TP1 Llama
  EAGLE3, DFLASH and vanilla DSPARK graphs use backed-up weights; block drafters
  require full attention and verify widths 2–4. Native draft complete refresh
  is unqualified. DUMMY has no draft parameters.
- Captured narrowed draft attention uses Triton lengths computed on device and
  keeps overlap enabled. Do not force
  `TOKENSPEED_KERNEL_OVERRIDE_ATTENTION_MHA_DECODE_WITH_KVCACHE=torch_npu_mha_decode_with_kvcache`
  for this path: a narrowed query needs its causal KV frontier or absolute mask,
  not the full verification KV length.
- Complete target/draft SDK refresh used a synthetic Qwen3 MTP draft and test
  provider. Real TCP with CPU staging covered local TP2 and remote publisher/TP1;
  production providers, RDMA and native draft refresh require qualification.

[Design invariants](../design/ascend-acl-graph-memory-saver.md) ·
[Direct graph/weight smoke tests](../../test/runtime/ascend_memory_saver/README.md) ·
[Source-qualified results and numerical limitations](ascend-memory-saver-test-report.md)
