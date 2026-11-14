# Ascend ACL Graph Memory Saver user guide

When a trainer and a TokenSpeed worker share Ascend NPUs, Memory Saver lets
the worker release selected device memory between rollout windows. The
worker stays alive. When it wakes, it reuses its captured decode graph and the
same device addresses.

## Before you start

The single-host Atlas A2 (Ascend 910B3) tests used Linux aarch64 and Python 3.12.
Choose the PyTorch and TorchNPU packages together in a dedicated virtual environment.

The primary deployment baseline is PyTorch `2.9.0+cpu`, PTA `2.9.0-730`
plus the six required backports, and CANN/HCCL `9.0.0`. The PTA base is
`9d40ac3a2b3714df7e5d164bda28a42eff3282cd` (July 30). Build the PTA wheel
for the deployment's Python version and PyTorch C++ ABI from the
[pinned backport source](https://gitcode.com/Huyanlong/pytorch/tree/74e6508b896f630e5d5cacd39a3877047e91c19a).
Its source tree matches tested build `ee1742b`. See the
[allocator requirements](../../tokenspeed-kernel-npu/memory-saver/README.md#pta-options)
for links to all six official changes. No additional PTA patches or
system-library replacements are required. Install that matching PyTorch/PTA
pair before building TokenSpeed below. Install the matching TorchVision package first as well.
The primary environment used `torch==2.9.0+cpu`,
`torch-npu==2.9.0+gitee1742b`, and `torchvision==0.24.0`.

Keep those installed versions constrained while installing TokenSpeed and
its dependencies. Generate the constraints from package metadata so the
custom PTA version is preserved; this command does not import PyTorch or
initialize a device.

```bash
export PIP_CONSTRAINT="$(mktemp)"
python - <<'PYVERSIONS' > "$PIP_CONSTRAINT"
from importlib.metadata import version
for name in ("torch", "torch-npu", "torchvision"):
    print(f"{name}=={version(name)}")
PYVERSIONS
```

Stock TorchNPU `2.13.0` is a separately recorded historical compatibility
route. The earlier A2 test used CANN `9.0.0`; Ascend's official release pairs
that wheel with CANN `9.2.0`. It has not been rerun for the current integration.
[Historical compatibility coverage](ascend-memory-saver-test-report.md#component-and-historical-boundaries)
· [Historical stock-wheel installation](#historical-stock-2130-installation)
· [Ascend version matrix](https://github.com/Ascend/pytorch/blob/master/COMPATIBILITY.en.md)

### Build and install TokenSpeed

The NPU feature depends on the common memory-protocol changes in
[PR #3397](https://github.com/lightseekorg/tokenspeed/pull/3397), pinned here
at `2e0975c31d88661b2fe110e0fac860594ed62479`. It is open and Ready for
Review, not merged. This dependency is separate from the Ascend allocator and model support.

Build the TokenSpeed Python runtime, kernel packages, native saver library,
and `ts-host` from the same checkout. The Ascend saver uses the shared
memory-control protocol 3 for release and resume. For HTTP control, install
the matching SMG gateway from the TokenSpeed runtime dependencies. The Ascend
kernel setup below installs Triton-Ascend 3.2.1, Transformers 5.12.0, and
`apache-tvm-ffi` 0.1.13.post3.
[Ascend kernel setup](../../tokenspeed-kernel-npu/README.md)
· [Native host package](../../tokenspeed-engine/README.md)

From the TokenSpeed checkout, point `TOKENSPEED_CANN_ROOT` at your installed
CANN toolkit. The example uses the tested 9.0.0 installation. Build the
native library with CMake 3.18 or later and a C++17 compiler.

```bash
export TOKENSPEED_CANN_ROOT=/usr/local/Ascend/cann-9.0.0
source "$TOKENSPEED_CANN_ROOT/set_env.sh"
TOKENSPEED_TVM_FFI_VERSION=0.1.13.post3 bash test/ci_system/install_triton_ascend.sh
TOKENSPEED_KERNEL_BACKEND=npu python -m pip install \
  -e ./tokenspeed-kernel/python ./tokenspeed-engine -e ./python

saver_build="$PWD/build/ascend-memory-saver"
cmake -S tokenspeed-kernel-npu/memory-saver -B "$saver_build" \
  -DASCEND_HOME_PATH="$ASCEND_HOME_PATH" -DCMAKE_BUILD_TYPE=Release
cmake --build "$saver_build"

export TOKENSPEED_NPU_MEMORY_SAVER_LIBRARY="$saver_build/libascend_memory_saver.so"
export TORCH_DEVICE_BACKEND_AUTOLOAD=1
export PYTORCH_NPU_ALLOC_CONF=expandable_segments:False
export TORCH_HCCL_ZERO_COPY=0
export TASK_QUEUE_ENABLE=1 ASCEND_LAUNCH_BLOCKING=0
```

Set `TOKENSPEED_HOST_BINARY` to the matching `ts-host` executable if you are
running from source and the installed executable is from another checkout.
The saver library must provide ABI 2. Checked Ascend Saver serving requires
protocol 3 and refuses an older native host before model initialization.
CUDA and ROCm retain protocol 2, `torch_memory_saver==0.0.9.post1`, and
their existing allocator and dependency paths.
[Allocator requirements](../../tokenspeed-kernel-npu/memory-saver/README.md)

Keep PyTorch backend autoload enabled in the serving environment so a fresh
worker registers the installed TorchNPU backend before importing kernels.

Leave `PYTORCH_NO_NPU_MEMORY_CACHING` unset or set it to `0`. Disabling the
PTA caching allocator bypasses the managed MemPools and is rejected at startup.
Startup also checks that target and draft parameters, the shared KV backing,
and captured graph allocations belong to their live managed pools. An
allocation failure with a failed rollback faults the worker. Restart it rather
than retrying a partially cleaned-up allocation.

## Start a worker

The example runs a dense BF16 Qwen3 model on one NPU. Set `CHECKPOINT_A`
to a complete local checkpoint. Use `cpu_backup` to restore the same weights
after every pause. Use `discard` when each training window produces a complete
replacement checkpoint and you want to avoid copying old target weights to
host memory. The following launch selects `discard`.

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

The graph tag is `acl_graph`. An audited BF16 MHA decode capture with greedy
sampling can discard temporary graph pages. Other supported captures back up
the graph pool on the host. The runtime selects this graph policy from the
model and capture configuration. The weight policy flag controls weights.
Releasing `acl_graph` also releases `kv_cache`, including target and draft KV
in the shared device arena. Resume the graph pool first and wake KV explicitly
when weights are ready. This pairing also applies to graph-pool CPU backup.
[Server options](../configuration/server.md)

| Tag | Released memory | How it becomes ready |
| --- | --- | --- |
| `kv_cache` | Device KV cache arena | TokenSpeed maps and clears the arena on wake. |
| `weights` | Model weight regions | CPU backup restores the old bytes. Discard requires a complete weight update after remapping. |
| `acl_graph` | ACL decode graph private pool | The original graph replays with restored or freshly written temporary bytes. |

The ACL Graph executable stays resident; the saver releases its managed
private allocation pages. HCCL groups and their workspaces stay resident.
Device memory available to training also depends on persistent graph inputs
and other runtime allocations.

Measure reclaimed device memory with driver free-memory readings. Allocator
reservation counters can still include the virtual ranges kept for replay.
Release the trainer's device allocations before waking the serving worker;
restoring physical backing requires enough free device memory.

## Release memory and load the next checkpoint

Use the SMG gateway's `/v1/rl` routes, or send the same control calls over the
engine's message transport when running headless. For a worker group, let each
call finish successfully on every serving replica before moving on. The
replica's frontend relays the call to its TP ranks.
[Control call coordination](weight-memory.md#one-update-cycle)

The following examples use the gateway on localhost. Run each step after
checking the previous response. An HTTP success code alone does not establish
that the operation succeeded. Check the returned success result.

```bash
base=http://127.0.0.1:8000/v1/rl
rl() {
  curl --fail-with-body -sS -H 'Content-Type: application/json' \
    -d "$2" "$base/$1"
}
```

1. Stop submitting rollouts. Release all three managed regions. The worker
   drains before releasing device memory.

   ```bash
   rl release_memory_occupation '{"tags":["kv_cache","weights","acl_graph"]}'
   ```

2. Train and make checkpoint B available on the workers. Resume the weights
   and graph pool while KV is still released.

   ```bash
   rl resume_memory_occupation '{"tags":["weights","acl_graph"]}'
   ```

3. Load the complete replacement target checkpoint. Use a complete checkpoint
   of the same dense BF16 Qwen3 architecture and a path visible to every
   worker. This request lets the direct checkpoint loader detect the file
   format and sets a cache namespace version.
   Ascend Memory Saver requires `flush_cache=true` for weight updates. Before
   loading any rank, TokenSpeed clears Host L2 and deletes old L3 namespace
   objects. A refusal leaves the update unapplied and can be retried.

   Publish each checkpoint in an immutable version directory. For `auto`, keep
   one selectable inference file family there: a folder containing both
   Safetensors and binary weights is refused. An explicit format can select
   the intended family. When a supported model declares exact checkpoint
   files, only those files are selected; a missing declared file is an error.
   Direct readers `auto`, `safetensors`, `pt`, and
   `mistral` are supported within the existing model contract. NPCACHE,
   InstantTensor, dummy and custom loaders are rejected for Ascend Saver
   updates under both weight policies. These format checks run before cache
   invalidation and the weight transaction, so correcting the request permits
   a retry with the same version. Do not modify files during a reload.

   ```bash
   rl update_weights_from_disk \
     '{"model_path":"/path/to/checkpoint-b","load_format":"auto","weight_version":"step-42","flush_cache":true}'
   ```

4. After the update succeeds everywhere, resume KV and submit the next
   rollout.

   ```bash
   rl resume_memory_occupation '{"tags":["kv_cache"]}'
   ```

With `cpu_backup`, omit step 3 when you want to serve the original weights.
Device KV is still discarded during graph suspend and rebuilt on wake, even
when the weight version stays the same. Host prefixes can be reused for the
same weights until a weight update flushes them.
With `discard`, always send explicit tags during wake so KV stays released
until the new weights are committed. A sequencing refusal before work starts
can be retried in the correct order. If the weight load starts and then fails,
or a device memory transition fails, restart the worker before submitting
new requests.
[Weight update contract](weight-memory.md)

### Causal attention in a captured draft

FIA causal masking works for multi-query verification. For example, five
queries over 105 KV positions use suffix-causal bounds 100 through 104.
Query row 1 cannot attend positions 102 through 104. After narrowing that
row into a single-query draft call, the native wrapper supplies no mask or
absolute query position. It must receive the correct KV frontier, 102 in
this example, or an explicit position mask. A KV length of 105 by itself
does not preserve the original row's causal boundary.

Automatic native draft MHA capture uses Triton to consume the lengths
computed on device inside the graph. Draft overlap remains enabled. Avoid
forcing `TOKENSPEED_KERNEL_OVERRIDE_ATTENTION_MHA_DECODE_WITH_KVCACHE` to
`torch_npu_mha_decode_with_kvcache` for this captured narrowed-draft path.
See the [causal-length control results](ascend-memory-saver-test-report.md#fia-causal-length-controls).

## Load a published Mooncake version

Install your Ascend-capable Model Updater SDK provider in every worker
environment. The provider must supply ready CPU tensors or tensors on the
worker's own NPU. Configure all four launch flags:

| Flag | Value |
| --- | --- |
| `--model-update-config` | Provider configuration as a JSON object |
| `--model-update-sdk-module` | Import path of the installed provider |
| `--model-update-engine-type` | Engine type accepted by that provider |
| `--model-update-draft-weights` | `retain` to preserve independent draft weights, or `refresh` to replace them in the same version |

Keep KV asleep during the weight update. Replace the disk update in step 3
with the published integer version, then wake KV after the update succeeds.
The resumed worker builds fresh KV for new requests.

```bash
rl update_weights_from_mooncake '{"version":42,"flush_cache":true}'
```

If the provider publishes snapshot metadata, include it in the optional
`snapshot` object. With a flush, the version string becomes the cache
namespace unless the request supplies `weight_version`. For disk replacement
after discard with a draft model loaded, set `--model-update-draft-weights retain` and
replace only the target. For SDK refresh, publish complete target and
independent draft weights together. Shared tensors are updated by the target.
[Mooncake fields and provider contract](../configuration/server.md#mooncake-weight-updates)

## Supported configurations

- Use a single Atlas A2 node in aggregate serving with TP1, TP2, TP4, or TP8.
  The runtime requires all serving ranks on that node in a single TP replica.
  Use ordinary ranks with prefill and decode served together.
- Speculative decoding uses greedy chain drafting. Graph support depends on
  the target model, drafter, and attention kernels.
- Dense BF16 Qwen3 supports complete target replacement from disk or the SDK.
  Quantized weights and online expert rebalancing are outside the discard
  contract.
- Graph discard and weight discard have separate eligibility checks. The
  audited TP1 Llama EAGLE3, DFLASH, and vanilla DSPARK graph configurations
  use backed-up weights. The validated complete target-and-draft refresh
  used a synthetic Qwen3-compatible MTP draft and a test SDK provider.
  Mooncake TCP tests also cover a remote publisher with one serving rank
  and a local publisher with two serving ranks. Production SDK transport
  and native draft refresh have separate qualification requirements.
  [Mooncake validation](ascend-memory-saver-test-report.md#latest-integration-cpu-and-hardware-acceptance)
- Real-device acceptance tests exercise address preservation, graph replay,
  and per-rank memory accounting. Run them against the stack you will
  deploy. [Hardware acceptance tests](../../test/runtime/ascend_memory_saver/README.md)

See the [design](../design/ascend-acl-graph-memory-saver.md) for the CUDA
comparison and how release and recovery work. The
[validation report](ascend-memory-saver-test-report.md#latest-integration-cpu-and-hardware-acceptance)
records root-accepted latest functional qualification separately from earlier
N18 timing components and historical coverage. Physical NVIDIA/AMD feature testing was unavailable; inspection
and CPU tests are the available GPU boundary checks.

## Historical stock 2.13.0 installation

To reproduce the earlier compatibility environment, use a separate
Python 3.12 virtual environment. With it active, download the three Linux
aarch64 wheels below into the same directory. Most serving tests used Ascend's
published PyTorch CPU build with its ARM allocator fix. Use the matching
TorchVision build for Torch 2.13.0. Compressed Tensors 0.15.0.1 removes the
Torch upper bound in the older 0.15.0 release.
[PyTorch 2.13.0 CPU wheel](https://gitcode.com/Ascend/pytorch/releases/download/v26.2.0-pytorch2.13.0/torch-2.13.0%2bcpu-cp312-cp312-manylinux_2_28_aarch64.whl)
· [TorchNPU 2.13.0 wheels](https://pypi.org/project/torch-npu/2.13.0/#files)
· [TorchVision CPU wheels](https://download.pytorch.org/whl/cpu/torchvision/)
· [Compressed Tensors release](https://github.com/vllm-project/compressed-tensors/releases/tag/0.15.0.1)

Set `wheel_dir` to the download directory. Keep the constraints active while
installing TokenSpeed so dependency resolution keeps the tested versions.

```bash
wheel_dir=/path/to/downloaded/wheels
export PIP_CONSTRAINT="$(mktemp)"
cat > "$PIP_CONSTRAINT" <<'EOF'
torch==2.13.0+cpu
torch-npu==2.13.0
torchvision==0.28.0+cpu
compressed-tensors==0.15.0.1
EOF
python -m pip install \
  "$wheel_dir/torch-2.13.0+cpu-cp312-cp312-manylinux_2_28_aarch64.whl" \
  "$wheel_dir/torch_npu-2.13.0-cp312-cp312-manylinux_2_28_aarch64.whl" \
  "$wheel_dir/torchvision-0.28.0+cpu-cp312-cp312-manylinux_2_28_aarch64.whl" \
  compressed-tensors
```
