# CUDA memory protocol on two logical nodes

This manual test exercises the current public source over the production
serving wire. It runs four CUDA protocol-v2 cases: EP4 and DP2×TP2+EP4, each
with `weights` + `kv_cache`, then with the additional `nccl` tag. It uses
Eager execution, with graph suspend/resume disabled. It does not enable CUDA
protocol v3 or require a separate candidate source tree.

The two logical nodes have separate Linux network namespaces connected by a
veth pair. Each node uses two GPUs (default device sets `0,1` and `4,5`).
NCCL uses Socket transport across the veth link; the FlashInfer CUTLASS MoE
kernel still uses the production gather/reduce path. The runner requires an
otherwise idle eight-GPU Linux container, user/network namespace support,
`unshare`, `nsenter`, `ip`, `ping`, `flock`, `timeout`, `sha256sum`, and
`nvidia-smi`.

## Prepare the current source

Use a fresh absolute run directory and explicitly select the public checkout
being validated. Activate its supported CUDA environment first. Install the
source's declared dependencies, including controller dependencies `msgpack`
and `pyzmq`, the fixture dependencies PyTorch, `safetensors`, `tokenizers`,
and `transformers`, and the supported FlashInfer/CUDA toolchain.

Build the engine native extension and `ts-host` from this same checkout,
using the active Python interpreter. Build/install its scheduler and kernel
packages using the project's installation instructions. The source tree
must contain `tokenspeed-engine/python/tokenspeed_engine/_native*.so` and
`tokenspeed-engine/python/tokenspeed_engine/bin/ts-host`; its kernel Python
package must have compatible native objects available. For example, after
installing the native build prerequisites in the active environment:

```bash
SOURCE=/absolute/path/to/current-public-checkout
RUN=/absolute/path/to/fresh-run
HARNESS="$SOURCE/test/runtime/distributed/cuda_memory_protocol"
export TS_TEST_PYTHON="$(command -v python3)"
export PYO3_PYTHON="$TS_TEST_PYTHON"
cd "$SOURCE"
"$TS_TEST_PYTHON" -m pip install --no-build-isolation --no-deps -e "$SOURCE/tokenspeed-scheduler"
"$TS_TEST_PYTHON" -m pip install --no-build-isolation --no-deps -e "$SOURCE/tokenspeed-engine"
mkdir -p "$RUN/models" "$RUN/logs"
"$TS_TEST_PYTHON" "$HARNESS/test_make_cuda_two_node_fixture.py" "$RUN/models/moe4"
```

A standalone replay bundle uses its own `test` directory as `HARNESS`, with
the public source supplied separately as `SOURCE`. Both logical nodes run
that source and its native host. No historical checkout or private patch is
needed. The fixture writes the canonical `num_experts` field explicitly so
Transformers 5's serialization alias cannot change its eight-expert layout.

## Select installation paths and warm FlashInfer

The runner normally uses the active environment's library paths. These
optional variables select installation-specific directories; set only those
needed by the environment:

- `TS_TEST_NCCL_LIB_DIR`: NCCL library directory. The `nccl` cases require
  working native ProcessGroupNCCL suspend/resume support.
- `TS_TEST_PYTHON_LIB_DIR`: Python environment's library directory, such as
  `$CONDA_PREFIX/lib` when using Conda.
- `TS_TEST_CUDA_RUNTIME_LIB_DIR`: CUDA runtime library directory.
- `TS_TEST_EXTRA_PYTHONPATH`: dependency overlays or separately built Python
  native packages. Use this for dependencies, not another runtime checkout.
- `TS_TEST_CARDS_0` / `TS_TEST_CARDS_1`: distinct two-GPU lists per logical
  node, defaulting to `0,1` / `4,5`.
- `TS_TEST_LOCK_FILE`: optional lock location shared by validation runs.

For example, an environment with a custom NCCL build can set
`TS_TEST_NCCL_LIB_DIR=/absolute/path/to/nccl/lib`. The recorded H200 setup used
PyTorch 2.9, FlashInfer 0.7.4, NVCC 12.8, and NCCL 2.30.7 with native
suspend/resume support; these paths are not built into the runner.

Warm the FlashInfer Hopper kernel in the same environment/cache used for
serving. The cold build can take tens of minutes. Adjust `MAX_JOBS` and
scratch space to the container's CPU/memory allocation:

```bash
export FLASHINFER_CUDA_ARCH_LIST=9.0a FLASHINFER_JIT_DEBUG=0
for directory in "${TS_TEST_CUDA_RUNTIME_LIB_DIR:-}" "${TS_TEST_PYTHON_LIB_DIR:-}" "${TS_TEST_NCCL_LIB_DIR:-}"; do
  if [[ -n "$directory" ]]; then
    export LD_LIBRARY_PATH="$directory${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
  fi
done
ep_build_tmp=$(mktemp -d /dev/shm/ts-flashinfer.XXXXXXXX)
CUDA_VISIBLE_DEVICES= MAX_JOBS="${MAX_JOBS:-8}" TMPDIR="$ep_build_tmp" \
  timeout -s TERM -k 30 2700 "$TS_TEST_PYTHON" "$HARNESS/test_warm_flashinfer_ep.py"
rm -rf -- "$ep_build_tmp"
bash "$HARNESS/test_run_ep4_warm.sh" "$RUN" "$SOURCE"
```

The runner clears inherited `LD_PRELOAD` and uses the normal launcher's
automatic memory-saver hook setup. It records source/native hashes in
`logs/source-inputs.sha256`, actual imported module paths and package versions
in `logs/runtime-imports.json`, and the four checked results in
`logs/cuda-v2-serving-summary.json`.

## Check the evidence

Each case must select `flashinfer_cutlass_unquant_moe_apply` on both nodes.
The evidence checker requires exactly one memory-saver hook mapping per
rank, every rank's completed release/resume apply phase, all four EP4
ingresses or both DP2×TP2 ingresses, and exactly eight integer token IDs
before release and after resume. Those token lists must match exactly.
Missing, extra, empty, truncated, or non-integer ACK records are rejected.

For each `nccl` case, the probe passively records the exact backend list
returned by production `_nccl_backends` and the original suspend/resume
return counts. It records only scalar metadata: backend IDs, read-only
`backend.options.group_name` and the existing process-group registry names.
It calls no additional communicator lookup, synchronization or collective,
and returns the original objects and results unchanged. Each rank must have
one positive count per direction, matching its unique backend IDs and unique
registered group names within that apply phase. Python wrapper IDs can change
or be reused between calls, so suspend/resume must cover the same group-name
set, with release before resume. Missing, duplicate, undercounted,
out-of-phase or unpaired records fail validation. Counts follow the actual
inventory, including cached split communicators, rather than topology
constants. Historical `nccl` logs without this metadata need a new run;
the checker does not waive this requirement. Non-`nccl` cases do not require
inventory records.

Protocol v2 retains its original one-apply-per-direction and local ACK
semantics, including when `nccl` is requested. The checker must not impose
v3's all-rank ACK barrier on v2. Its regression tests separately cover v3's
split NCCL phases and reject ACKs sent before every rank finishes.

```bash
"$TS_TEST_PYTHON" "$HARNESS/test_summarize_cuda_two_node_serving.py" \
  "$RUN"/logs/serve-flashinfer-*/
cd "$SOURCE"
"$TS_TEST_PYTHON" -m pytest -q \
  test/runtime/distributed/test_memory_protocol_startup.py \
  "$HARNESS/test_cuda_two_node_evidence.py" \
  "$HARNESS/test_cuda_two_node_cleanup.py" \
  test/runtime/test_model_executor_cache_state.py
```

`network.log` and `socket-proof.log` preserve distinct network namespaces and
NCCL `NET/Socket` data-channel evidence. The two `-nccl` cases also require
four ranks' native suspend/resume message counts to match their validated
inventory counts. These appear in `nccl-suspend-resume.log`; per-rank IDs
and operation counts remain in the rank JSONL records. Raw controller, launcher, rank, and NCCL logs stay
in each case directory so the summary can be independently checked.

The four Python regression modules are registered in `runtime-1gpu` CI.
They are CPU behavioral tests using that job's complete runtime dependency
environment. The startup tests exercise protocol-v2/v3 mismatch detection
with real multiprocess Gloo, without enabling CUDA v3 serving.

This matrix uses one physical host and two isolated network stacks. It does
not validate production PP/PD with an in-flight remote KV transfer, physical
cross-host failure modes, or the v3 all-rank success barrier on hardware.
