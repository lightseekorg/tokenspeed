# Kimi K3 AgentX: DWDP16 prefill + DEP16 decode

Reproduces the disaggregated Kimi K3 AgentX results on 32 GB300 GPUs: eight
four-GPU nodes in one NVLink domain. The results were measured at the parent
commit of this recipe.

| Role | GPUs | Configuration |
| --- | --- | --- |
| Prefill | 0 and 2 of every node | DWDP16: 16 attention-DP ranks; each rank stores 1/16 of the routed experts and fetches the rest from peers; FlashInfer TRT-LLM MoE; 16K-token chunks; shared L2 host pool |
| Decode | 1 and 3 of every node | DEP16: 16 attention-DP ranks, MegaMoE EP16, TP4 QKV/O projections and shared experts, CUDA graphs for batch 1–64 |
| Gateway | head node | SMG; prefill rank by least load, decode rank by power-of-two least requests |

Both roles use EAGLE3 with three speculative steps. Decode fixes the
acceptance length at 3.0 (`--synthetic-acceptance-length`), so generated text is
synthetic and only valid for performance measurement. Settings live in
`configs/`.

## Pinned revisions

| Component | Revision |
| --- | --- |
| TokenSpeed | parent commit of this recipe (`a4bd0d5d`) |
| Base image | `lightseekorg/tokenspeed-runner:cu130-torch-2.14.0-flashinfer-0.6.18` |
| FlashInfer | `flashinfer-python==0.7.2`, repacked by `build_flashinfer_wheel.py` |
| SMG gateway, gRPC servicer and proto | [syuoni/smg](https://github.com/syuoni/smg) `fe2c6cefc243e02abe4f30c6f3923478f7892691` + `smg-gateway-dp-rank-loads.patch` |
| InferenceX | [SemiAnalysisAI/InferenceX](https://github.com/SemiAnalysisAI/InferenceX) `1f60c15b5aefe948b7fdc60898b439831413d299` |
| AIPerf | [SemiAnalysisAI/aiperf](https://github.com/SemiAnalysisAI/aiperf) `89b21867872a5bbc4b0676bf5005c404da5e9f94` + `aiperf-smg-routing-key-source.patch` |
| Model | `nvidia/Kimi-K3-NVFP4` `b2428a0b83a8b712ff2e1a8448a103d4175341f1` |
| Draft model | `lightseekorg/kimi-k3-eagle3-mla` `4c48d2bb72134094340067e3012ebe0e822fac37` |
| Trace | `semianalysisai/cc-traces-weka-062126` `23f152f6f0f9399a85901b89a6458def0ef16729` |

## Prepare

Prepare and Run use one held allocation of at least eight four-GPU nodes.
Request the GPUs explicitly, for example `salloc --nodes=8 --gpus-per-node=4
--exclusive`: job steps get GPUs only from the allocation's GPU request. Use
shared, absolute paths without spaces, mounted at the same paths on every
node.

Run every Prepare step in one shell, in a container started from the
TokenSpeed runner image on one node. Exiting the shell saves the container as
the image every node runs:

```bash
SHARED=/path/to/shared
srun --jobid=YOUR_HELD_JOB_ID --overlap --nodes=1 --ntasks=1 --pty \
  --container-image=lightseekorg/tokenspeed-runner:cu130-torch-2.14.0-flashinfer-0.6.18 \
  --container-mounts="$SHARED:$SHARED" --container-remap-root --no-container-mount-home \
  --container-save="$SHARED/tokenspeed-agentx.sqsh" bash
```

### Server

Install TokenSpeed from this checkout for GB300 as `docker/Dockerfile.nvidia`
does, then the packages this recipe adds:

```bash
SOURCE_ROOT=/path/to/this/checkout
BUILD_ROOT=/path/to/shared/build
export PIP_BREAK_SYSTEM_PACKAGES=1
apt-get update && apt-get install -y --no-install-recommends \
  libssl-dev libopenmpi-dev libnuma1 pkg-config
python3 -m pip install cmake ninja
MAX_JOBS=16 FLASHINFER_CUDA_ARCH_LIST=10.3a TOKENSPEED_KERNEL_BACKEND=cuda \
  python3 -m pip install "$SOURCE_ROOT/tokenspeed-kernel/python/" --no-build-isolation
python3 -m pip install "$SOURCE_ROOT/tokenspeed-scheduler/"
python3 -m pip install "$SOURCE_ROOT/python" --no-build-isolation
python3 -m pip install --no-deps --force-reinstall "$SOURCE_ROOT/tokenspeed-mla"

# FlashInfer 0.7.2 with the Kimi K3 MoE backport.
git clone https://github.com/flashinfer-ai/flashinfer.git "$BUILD_ROOT/flashinfer"
python3 -m pip download --no-deps flashinfer-python==0.7.2 --dest "$BUILD_ROOT"
python3 "$SOURCE_ROOT/test/agentic_benchmark/kimi_k3/tokenspeed_agentx/build_flashinfer_wheel.py" \
  --source-wheel "$BUILD_ROOT/flashinfer_python-0.7.2-py3-none-any.whl" \
  --repo "$BUILD_ROOT/flashinfer" --work "$BUILD_ROOT/flashinfer-work" --output "$BUILD_ROOT/wheels"
python3 -m pip install --no-deps --force-reinstall "$BUILD_ROOT/wheels/flashinfer_python-0.7.2-1k3gb300-py3-none-any.whl"

# SMG: one source tree for the gateway, the gRPC servicer and its proto.
git clone https://github.com/syuoni/smg.git "$BUILD_ROOT/smg"
git -C "$BUILD_ROOT/smg" checkout fe2c6cefc243e02abe4f30c6f3923478f7892691
git -C "$BUILD_ROOT/smg" apply \
  "$SOURCE_ROOT/test/agentic_benchmark/kimi_k3/tokenspeed_agentx/smg-gateway-dp-rank-loads.patch"
python3 -m pip uninstall -y smg-grpc-servicer smg-grpc-proto tokenspeed-smg-grpc-servicer tokenspeed-smg-grpc-proto
python3 -m pip install --no-deps "$BUILD_ROOT/smg/crates/grpc_client/python" "$BUILD_ROOT/smg/grpc_servicer"

# The gateway build needs rustup (cargo picks the toolchain pinned in the SMG
# checkout's rust-toolchain.toml, so run it there) and protoc 31.1 with its
# well-known type includes, which distribution protoc packages lack. rustup
# defaults to $HOME, which is not shared; keep the toolchain under BUILD_ROOT.
export RUSTUP_HOME="$BUILD_ROOT/rustup" CARGO_HOME="$BUILD_ROOT/cargo"
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y --profile minimal --default-toolchain none
source "$CARGO_HOME/env"
curl -fsSL -o "$BUILD_ROOT/protoc-31.1.zip" \
  https://github.com/protocolbuffers/protobuf/releases/download/v31.1/protoc-31.1-linux-aarch_64.zip
python3 -m zipfile -e "$BUILD_ROOT/protoc-31.1.zip" "$BUILD_ROOT/protoc-31.1"
chmod +x "$BUILD_ROOT/protoc-31.1/bin/protoc"
(cd "$BUILD_ROOT/smg" && PROTOC="$BUILD_ROOT/protoc-31.1/bin/protoc" cargo build -p smg --locked --release)
SMG_BINARY="$BUILD_ROOT/smg/target/release/smg"
```

The gateway patch scopes each worker's polled load to its own attention-DP
rank.

### Client

The client gets its own Python 3.11 environment, because its pinned packages
conflict with the server's. The image has no uv; install it, and keep uv's
Python under `CLIENT_ROOT` so the environment also works in later containers:

```bash
CLIENT_ROOT=/path/to/shared/agentx-client
mkdir -p "$CLIENT_ROOT"
git clone https://github.com/SemiAnalysisAI/InferenceX.git "$CLIENT_ROOT/inferencex"
git -C "$CLIENT_ROOT/inferencex" checkout 1f60c15b5aefe948b7fdc60898b439831413d299
git clone https://github.com/SemiAnalysisAI/aiperf.git "$CLIENT_ROOT/aiperf"
git -C "$CLIENT_ROOT/aiperf" checkout 89b21867872a5bbc4b0676bf5005c404da5e9f94
git -C "$CLIENT_ROOT/aiperf" apply \
  "$SOURCE_ROOT/test/agentic_benchmark/kimi_k3/tokenspeed_agentx/aiperf-smg-routing-key-source.patch"
python3 -m pip install uv
export UV_PYTHON_INSTALL_DIR="$CLIENT_ROOT/python" UV_CACHE_DIR="$CLIENT_ROOT/uv-cache"
uv venv --python 3.11 --seed "$CLIENT_ROOT/venv"
source "$CLIENT_ROOT/venv/bin/activate"
python -m pip install -e "$CLIENT_ROOT/aiperf" \
  'transformers==4.57.1' 'numpy>=1.24' 'pandas>=2.0' 'aiohttp>=3.10' \
  'xlsxwriter>=3.2.1' 'tqdm>=4.66' 'datasets>=4.7' tiktoken matplotlib \
  'huggingface_hub[cli]>=0.25' urllib3 requests
python -m pip check
```

The AIPerf patch adds `AIPERF_HTTP_X_SMG_ROUTING_KEY_SOURCE`, which selects
the identity sent in the `X-SMG-Routing-Key` header; `agentx_client.sh` sends
each request's branch correlation ID.

### Models and trace

With the client environment active, which provides `hf`:

```bash
MODEL_CACHE=/path/to/shared/hf-cache/hub
hf download nvidia/Kimi-K3-NVFP4 --revision b2428a0b83a8b712ff2e1a8448a103d4175341f1 --cache-dir "$MODEL_CACHE"
hf download lightseekorg/kimi-k3-eagle3-mla --revision 4c48d2bb72134094340067e3012ebe0e822fac37 --cache-dir "$MODEL_CACHE"
MODEL_PATH="$MODEL_CACHE/models--nvidia--Kimi-K3-NVFP4/snapshots/b2428a0b83a8b712ff2e1a8448a103d4175341f1"
DRAFT_MODEL_PATH="$MODEL_CACHE/models--lightseekorg--kimi-k3-eagle3-mla/snapshots/4c48d2bb72134094340067e3012ebe0e822fac37"

# The tokenizer must load from real files: its code imports a sibling module,
# which transformers cannot resolve through the cache's blob symlinks.
TOKENIZER_PATH=/path/to/shared/kimi-k3-tokenizer
mkdir -p "$TOKENIZER_PATH"
cp -L "$MODEL_PATH"/*.py "$MODEL_PATH"/*.json "$MODEL_PATH"/tiktoken.model "$TOKENIZER_PATH"/

CLIENT_CACHE=/path/to/shared/client-cache
hf download --repo-type dataset semianalysisai/cc-traces-weka-062126 \
  --revision 23f152f6f0f9399a85901b89a6458def0ef16729 --cache-dir "$CLIENT_CACHE/hub"
# The client reads the trace offline; build its datasets cache once, pinned.
HF_HUB_CACHE="$CLIENT_CACHE/hub" HF_DATASETS_CACHE="$CLIENT_CACHE/datasets" python -c \
  "import datasets; datasets.load_dataset('semianalysisai/cc-traces-weka-062126', revision='23f152f6f0f9399a85901b89a6458def0ef16729')"
```

Then exit the shell; pyxis saves the image to
`$SHARED/tokenspeed-agentx.sqsh`.

## Run

The first eight nodes of the allocation are used, and the first of them is the
head node. Start the server from this directory and keep it in the foreground:

```bash
SLURM_JOB_ID=YOUR_HELD_JOB_ID \
CONTAINER_IMAGE=/path/to/shared/tokenspeed-agentx.sqsh CONTAINER_NAME=ts \
CONTAINER_MOUNTS="/path/to/shared:/path/to/shared" \
MODEL_PATH="$MODEL_PATH" TOKENIZER_PATH="$TOKENIZER_PATH" DRAFT_MODEL_PATH="$DRAFT_MODEL_PATH" \
SMG_BINARY="$SMG_BINARY" \
bash serve_dwdp16_dep16.sh
```

It starts the shared L2 owner and its 16 providers, then decode, prefill and
the gateway, and prints the run directory once the gateway is ready. The first
start compiles and autotunes kernels and takes much longer than later starts. `DISAGGREGATION_IB_DEVICE`
overrides the KV-transfer NICs (default `mlx5_0,mlx5_1,mlx5_2,mlx5_3`). The
shared L2 providers assume 144 cores and NIC pairs `mlx5_0,mlx5_1` and
`mlx5_2,mlx5_3` per node; edit `configs/shared_l2_services.sh` for other nodes.
Interrupting the script stops all of its steps. `CONTAINER_NAME` names a
container kept on each node; pyxis reuses an existing container of that name
as is and ignores `CONTAINER_IMAGE`, so use a new name after rebuilding the
image.

From another shell, run the benchmark against it:

```bash
SLURM_JOB_ID=YOUR_HELD_JOB_ID \
CONTAINER_IMAGE=/path/to/shared/tokenspeed-agentx.sqsh CONTAINER_NAME=ts \
CONTAINER_MOUNTS="/path/to/shared:/path/to/shared" \
SERVE_DIR=outputs/serve_YYYYMMDD_HHMMSS CLIENT_ROOT="$CLIENT_ROOT" CLIENT_CACHE="$CLIENT_CACHE" \
bash bench_agentx.sh
```

Edit `CONCURRENCIES` near the top of `bench_agentx.sh`; the default sweeps
128/256/384/512/576. Before each point the engines must be idle; their GPU
caches are flushed and the shared L2 pool is emptied. Each point replays the
full trace: ten warmup requests per lane (1800-second warmup grace period),
a 3600-second scored window, then up to 900 seconds of drain.

## Outputs

- `outputs/serve_<timestamp>/`: per-node engine and shared L2 logs,
  `gateway.log` and `endpoints.env`.
- `outputs/agentx_<timestamp>/c<N>/inferencex_result.json`: InferenceX report
  for concurrency `N`; `aiperf_artifacts/` holds the native AIPerf exports and
  `client-command.txt` the exact replay command.

Normalize throughput by 32 GPUs. Reported output throughput includes requests
still draining after the scored window.
