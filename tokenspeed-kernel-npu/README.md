# TokenSpeed-Kernel-NPU

TokenSpeed-Kernel-NPU contains the Ascend-specific operators used by
TokenSpeed. Keeping these implementations in a standalone package mirrors the
AMD package layout and keeps the TokenSpeed runtime vendor-neutral.

The initial Ascend path provides paged MHA, RMSNorm, Q/K RMSNorm, rotary
embedding, and the Triton-Ascend import adapter required by CANN 9.0.0.

For development from this repository:

```bash
test/ci_system/install_triton_ascend.sh
```

The validated stack is CANN 9.0.0, PyTorch 2.9.0, `torch_npu` 2.9.0.post2,
Transformers 5.12.0, Triton 3.2.0, and Triton-Ascend 3.2.1. The setup script
records every Python package mutation needed by this Ascend path, including the
`apache-tvm-ffi==0.1.13` build dependency and editable install of this package.

TokenSpeed applications should continue to import operators from
`tokenspeed-kernel`; it owns registration and dispatch to this package.

Run the operator correctness suite on a visible NPU with:

```bash
source /usr/local/Ascend/cann-9.0.0/set_env.sh
PYTHONPATH="${PWD}/python:${PWD}/tokenspeed-kernel/python:${PWD}/tokenspeed-kernel-npu/python:${PYTHONPATH:-}" \
    pytest -q tokenspeed-kernel-npu/test
```

For the complete Qwen3-0.6B launch command, ACL Graph capture sizes, serving
limits, and a request example, see the
[Ascend model recipe](../docs/recipes/models.md#qwen3-06b-on-ascend-npu).

## Serving Dependencies in a Source Checkout

When running TokenSpeed directly from a checkout through `PYTHONPATH`, install
the SMG serving packages explicitly. Run these commands from the repository
root with the same Python interpreter used to launch TokenSpeed:

```bash
source /usr/local/Ascend/cann-9.0.0/set_env.sh

python -m pip install \
    "tokenspeed-smg==1.9.0.post20260823" \
    "tokenspeed-smg-grpc-proto==0.4.14.post20260823" \
    "tokenspeed-smg-grpc-servicer==0.8.0.post20260823" \
    "grpcio==1.81.1" \
    "grpcio-health-checking==1.81.1" \
    "grpcio-reflection==1.81.1" \
    "protobuf>=5.26.0,<7" \
    "viztracer"
```

If the host requires an outbound HTTP proxy, configure it for the installation
without committing credentials:

```bash
export PROXY_URL="http://<username>:<password>@<proxy-host>:<port>"
export HTTP_PROXY="${PROXY_URL}"
export HTTPS_PROXY="${PROXY_URL}"
export http_proxy="${PROXY_URL}"
export https_proxy="${PROXY_URL}"
```

Verify the gRPC servicer and the source paths together:

```bash
PYTHONPATH="${PWD}/python:${PWD}/tokenspeed-kernel/python:${PWD}/tokenspeed-kernel-npu/python:${PYTHONPATH:-}" \
    python -m smg_grpc_servicer.tokenspeed --help
```
