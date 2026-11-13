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
Python native extension in its matching source tree. Put the CUDA v2
Draft source at `$RUN/offline/scoped` (`7ac36393`) and the independent CUDA
v3 source at `$RUN/offline/candidate` (`071d7055`). Each tree must contain
its native extension and `tokenspeed-engine/python/tokenspeed_engine/bin/ts-host`.
Put the compatible kernel package under
`$RUN/offline/kernel/tokenspeed-kernel/python`. The recorded run used the
shared kernel Python source from those commits and compatible installed CUDA
native objects. The separate replay archive includes offline build instructions.

From the repository root, set `HARNESS` to the directory below. For a
standalone bundle, set it to the directory containing the launch scripts.
Precompile FlashInfer in the same environment and cache used for serving.

```bash
HARNESS="$(pwd)/test/runtime/distributed/cuda_memory_protocol"
mkdir -p "$RUN/models" "$RUN/logs"
python3 "$HARNESS/test_make_cuda_two_node_fixture.py" "$RUN/models/moe4"

export LD_LIBRARY_PATH="/opt/nccl-mt/lib:/usr/local/conda/lib:/usr/local/conda/lib/python3.11/site-packages/nvidia/cuda_runtime/lib:${LD_LIBRARY_PATH:-}"
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

```bash
python3 "$HARNESS/test_summarize_cuda_two_node_serving.py" "$RUN"/logs/serve-flashinfer-*/
```

Inspect `network.log` and `nccl.*.log` in each case directory for data
channels using `NET/Socket`. Check the two `*-v3-nccl` cases' rank logs for
two suspended and resumed NCCL groups per rank in EP4 and three per rank
in DP2×TP2+EP4. The recorded evidence can also be checked by passing its
`serve-flashinfer-*/` directories to the same Python script.

This matrix exercises one physical host with two isolated network stacks.
The artificial apply delay does not exercise production PP/PD with an
in-flight remote KV transfer.
