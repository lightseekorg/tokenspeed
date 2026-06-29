#!/bin/bash
# Dump host, driver, toolkit and pinned-package versions.
#
# Called once from install_deps.sh after every install step has finished,
# and again from the CI workflow just before `Execute CI task` so that
# post-mortems on runner-only failures (e.g. illegal memory access seen
# on the b200-8gpu agentic perf job) can diff state without rerunning.
# Best-effort: never fail the caller if a probe tool is missing.

set +e

LABEL="${1:-env}"

echo ""
echo "=========================================="
echo "=== ${LABEL}: Environment info dump   ==="
echo "=========================================="
echo "--- env vars ---"
echo "CUDA_VERSION=${CUDA_VERSION:-unset}"
echo "SM=${SM:-unset}"
echo "CUINDEX=${CUINDEX:-unset}"
echo "FI_ARCH=${FI_ARCH:-unset}"
echo "INSTALL_TOKENSPEED_MLA_FROM_SOURCE=${INSTALL_TOKENSPEED_MLA_FROM_SOURCE:-unset}"
echo "PIP_EXTRA_INDEX_URL=${PIP_EXTRA_INDEX_URL:-unset}"
echo "MAX_JOBS=${MAX_JOBS:-unset}"
echo "CI_RUNNER_LABEL=${CI_RUNNER_LABEL:-unset}"
echo "WORKSPACE=${WORKSPACE:-unset}"

echo "--- OS / kernel / glibc ---"
{ cat /etc/os-release || true; } 2>&1 | sed 's/^/[os-release] /'
uname -a 2>&1 | sed 's/^/[uname] /' || true
ldd --version 2>&1 | head -n 1 | sed 's/^/[glibc] /' || true

echo "--- NVIDIA driver / GPUs ---"
nvidia-smi --query-gpu=index,name,driver_version,vbios_version,pstate,memory.total,memory.free,compute_cap \
    --format=csv 2>&1 | sed 's/^/[nvidia-smi] /' || true
nvidia-smi -q -d CLOCK,POWER,TEMPERATURE,ECC 2>&1 | head -n 80 | sed 's/^/[nvidia-smi] /' || true

echo "--- CUDA toolkit ---"
{ command -v nvcc && nvcc --version; } 2>&1 | sed 's/^/[nvcc] /' || true
{ ls -l /usr/local/cuda 2>/dev/null; ls /usr/local/cuda/version.json 2>/dev/null \
    && cat /usr/local/cuda/version.json; } 2>&1 | sed 's/^/[cuda] /' || true

echo "--- Python ---"
python3 --version 2>&1 | sed 's/^/[python] /' || true
python3 -c "import sys; print(sys.executable); print(sys.prefix)" 2>&1 \
    | sed 's/^/[python] /' || true
python3 -m pip --version 2>&1 | sed 's/^/[pip] /' || true

echo "--- Key Python packages ---"
python3 - <<'PY' 2>&1 | sed 's/^/[pkg] /' || true
from importlib.metadata import PackageNotFoundError, version
pkgs = [
    "torch",
    "triton",
    "flashinfer-python",
    "flashinfer-cubin",
    "flashinfer-jit-cache",
    "nvidia-cutlass-dsl",
    "nvidia-cutlass-dsl-libs-cu12",
    "nvidia-cutlass-dsl-libs-cu13",
    "nvidia-nccl-cu12",
    "nvidia-nccl-cu13",
    "nvidia-nvshmem-cu12",
    "nvidia-nvshmem-cu13",
    "nvidia-cudnn-cu12",
    "nvidia-cudnn-cu13",
    "cuda-python",
    "cuda-bindings",
    "tokenspeed",
    "tokenspeed-kernel",
    "tokenspeed-mla",
    "tokenspeed-scheduler",
    "tokenspeed-smg",
    "tokenspeed-grpc-servicer",
    "tokenspeed-grpc-proto",
    "tokenspeed-triton",
    "tokenspeed-triton-kernels",
    "tokenspeed-trtllm-kernel",
    "transformers",
    "huggingface-hub",
    "evalscope",
]
for p in pkgs:
    try:
        print(f"{p}=={version(p)}")
    except PackageNotFoundError:
        print(f"{p}==<not installed>")
PY

echo "--- torch CUDA build info ---"
python3 - <<'PY' 2>&1 | sed 's/^/[torch] /' || true
try:
    import torch
    print("version:", torch.__version__)
    print("cuda:", torch.version.cuda)
    print("cudnn:", torch.backends.cudnn.version() if torch.cuda.is_available() else None)
    print("nccl:", torch.cuda.nccl.version() if torch.cuda.is_available() else None)
    print("device_count:", torch.cuda.device_count())
    for i in range(torch.cuda.device_count()):
        print(f"device{i}:", torch.cuda.get_device_name(i), torch.cuda.get_device_capability(i))
except Exception as exc:
    print("torch probe failed:", exc)
PY

echo "--- JIT cache state ---"
for d in "${HOME}/.triton" "${HOME}/.cache/triton" "${HOME}/.cache/tokenspeed_triton" \
         "${HOME}/.cache/flashinfer" "${HOME}/.cache/flashinfer_jit_cache" \
         "${HOME}/.nv/ComputeCache" "${HOME}/.cache/torch_inductor" \
         "${HOME}/.cache/torchinductor"; do
    if [ -e "${d}" ]; then
        sz="$(du -sh "${d}" 2>/dev/null | awk '{print $1}')"
        n="$(find "${d}" -type f 2>/dev/null | wc -l)"
        echo "[cache] ${d}: ${sz:-?} (${n} files)"
    fi
done

echo "--- pip freeze ---"
python3 -m pip freeze 2>&1 | sed 's/^/[freeze] /' || true

echo "=========================================="
echo "=== End of ${LABEL} env dump          ==="
echo "=========================================="
