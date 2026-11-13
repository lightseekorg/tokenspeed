# CUDA memory protocol on two logical nodes

This manual H200 test drives the production serving wire for EP4 and
DP2×TP2+EP4. It assigns GPUs 0,1 and 4,5 to separate Linux network namespaces
and forces NCCL Socket over their veth link. Both layouts use the default
FlashInfer CUTLASS MoE kernel with production gather/reduce. The test uses
Eager execution and omits the CUDA graph memory tag.

The runner requires an otherwise idle eight-GPU Linux container with
`unshare`, `nsenter`, `ip`, `ping`, `flock`, and `nvidia-smi`. The recorded
environment used PyTorch 2.9.0 with native ProcessGroupNCCL suspend/resume,
NCCL 2.30.7 mt2.1, FlashInfer 0.7.3, and NVCC 12.8. The runner expects
NCCL under `/opt/nccl-mt/lib`, and Python, its CUDA runtime, and its
`torch_memory_saver` CUDA 12 preload under `/usr/local/conda`.
Install `msgpack` and `pyzmq` for the controller, plus PyTorch,
`safetensors`, `tokenizers`, and `transformers` for the local fixture.

Use a fresh absolute `RUN` directory. Build each revision's Rust host and
Python native extension in its matching source tree. Put the current CUDA v2
Draft source at `$RUN/offline/scoped` (base `622af556` plus the review
fixes) and the independent CUDA v3 source at `$RUN/offline/candidate` (base
`071d7055` plus the same runtime and regression-test fixes). Preserve the
candidate's CUDA v3/NCCL tests when applying these fixes before building;
the Draft keeps CUDA v2 and only the independent source enables CUDA v3.
Each tree must contain
its native extension and `tokenspeed-engine/python/tokenspeed_engine/bin/ts-host`.
Put the compatible kernel package under
`$RUN/offline/kernel/tokenspeed-kernel/python`. The recorded run used the
shared kernel Python source from those commits and compatible installed CUDA
native objects. The separate replay archive includes offline build instructions.

From the repository root, set `HARNESS` to the directory below. For a
standalone bundle, set it to the directory containing the launch scripts.
Precompile FlashInfer in the same environment and cache used for serving.

```bash
HARNESS="${HARNESS:-$(pwd)/test/runtime/distributed/cuda_memory_protocol}"
export LD_LIBRARY_PATH="/opt/nccl-mt/lib:/usr/local/conda/lib:/usr/local/conda/lib/python3.11/site-packages/nvidia/cuda_runtime/lib:${LD_LIBRARY_PATH:-}"
mkdir -p "$RUN/models" "$RUN/logs"
python3 "$HARNESS/test_make_cuda_two_node_fixture.py" "$RUN/models/moe4"

ep_build_tmp=$(mktemp -d /dev/shm/ts-flashinfer.XXXXXXXX)
trap 'rm -rf -- "$ep_build_tmp"' EXIT
CUDA_VISIBLE_DEVICES= FLASHINFER_CUDA_ARCH_LIST=9.0a FLASHINFER_JIT_DEBUG=0 \
  MAX_JOBS=56 TMPDIR="$ep_build_tmp" \
  timeout -s TERM -k 30 2700 python3 "$HARNESS/test_warm_flashinfer_ep.py" &&
bash "$HARNESS/test_run_ep4_warm.sh" "$RUN"
```

Adjust `MAX_JOBS` and temporary space for the container's CPU and memory
allocation. The recorded cold build took about 16 minutes. Keep its cache
available for the serving run.

For each layout, the runner compares eight tokens per ingress before
release and after resume. It runs all v2, all v3, all v3 with an explicit
`nccl` tag, and both orientations of mixed v2/v3. Homogeneous v3 must
acknowledge after the last rank's apply returns. The probe delays rank 3
before release and records apply completion. Rank 3 is an ingress in EP4
and a TP follower in DP2×TP2+EP4. In mixed runs, both launchers must exit
with a protocol mismatch and the wire must show zero HELLO, READY, or
input registrations.

The runner clears inherited `LD_PRELOAD` and uses the normal launcher's
automatic memory-saver setup. Each rank must map exactly one preload hook.
Both source trees must include the launcher environment-order fix; the v3
tree must also include the coordinated cancellation fix. The replay archive
contains these patches and matching source trees.

```bash
python3 "$HARNESS/test_summarize_cuda_two_node_serving.py" "$RUN"/logs/serve-flashinfer-*/
```

Inspect `network.log` and `nccl.*.log` in each case directory for data
channels using `NET/Socket`. Check the two `*-v3-nccl` cases' rank logs for
two suspended and resumed NCCL groups per rank in EP4 and three per rank
in DP2×TP2+EP4. The recorded evidence can also be checked by passing its
`serve-flashinfer-*/` directories to the same Python script.
The result records `tokens_per_ingress` before release and
`tokens_after_resume_per_ingress` after resume. The checker requires all four
EP4 ingresses or both DP2×TP2 ingresses and compares eight token IDs exactly.
For both release and resume it also requires four ACK timestamps in EP4 or
two in DP2×TP2+EP4 before checking the earliest reply against all-rank apply.
Missing, extra, empty, truncated, or non-integer ACK records are rejected.
Older archives lacking the resumed output or hook evidence must be rerun with
the current harness; they cannot independently prove these checks.

Run the evidence regressions with
`python3 -m pytest -q "$HARNESS/test_cuda_two_node_evidence.py"`.
The standalone CPU probe runs the actual launch function with real children,
without importing the GPU runtime. Exercise both context behaviors:

```bash
for context in overwrite prepend; do
  python3 "$HARNESS/test_launcher_context_cpu.py" \
    "$RUN/offline/candidate/python/tokenspeed/runtime/engine/rust_host.py" \
    --context "$context"
done
```

Run the serving regressions in each source tree, including the independent
candidate, so its same-round cancellation expectations cannot drift from its
implementation:

```bash
export CARGO_HOME="$RUN/offline/cargo" CARGO_NET_OFFLINE=true
export CARGO_TARGET_DIR="$RUN/offline/target" CARGO_BUILD_JOBS=8
export PYO3_PYTHON="$(command -v python3)"
export FETCHCONTENT_SOURCE_DIR_XGRAMMAR="$RUN/offline/build-deps/xgrammar-0.2.8"
export PYTHONPATH="$RUN/offline/build-python:$RUN/offline/runtime-extra:$RUN/offline/kernel/tokenspeed-kernel/python:$RUN/offline/candidate/python:$RUN/offline/candidate/tokenspeed-engine/python${PYTHONPATH:+:$PYTHONPATH}"
cargo test --manifest-path "$RUN/offline/candidate/tokenspeed-engine/Cargo.toml" \
  --locked -p tokenspeed-control --test serving
```

This matrix exercises one physical host with two isolated network stacks.
The artificial apply delay does not exercise production PP/PD with an
in-flight remote KV transfer.
