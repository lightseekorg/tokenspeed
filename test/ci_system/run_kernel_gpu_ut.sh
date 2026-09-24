#!/usr/bin/env bash
set -euo pipefail

export PYTHONPATH="test/ci_system:python${PYTHONPATH:+:$PYTHONPATH}"

multi_gpu_tests=(
    tokenspeed-kernel/test/nvidia/ops/communication/test_multimem_distributed.py
    tokenspeed-kernel/test/nvidia/ops/moe/test_latent_tail_distributed.py
    tokenspeed-kernel/test/nvidia/ops/moe/test_marlin_deepep_distributed.py
    tokenspeed-kernel/test/nvidia/thirdparty/test_trtllm_mnnvl_comm.py
    tokenspeed-kernel/test/nvidia/thirdparty/test_trtllm_mnnvl_twoshot.py
    tokenspeed-kernel/test/ops/test_attention_dsv41_index_scan.py
    tokenspeed-kernel/test/ops/test_communcation.py
)

python3 -m pytest "${multi_gpu_tests[@]}" -v --junitxml=/tmp/kernel-multi-gpu.xml

ignores=(
    --ignore=tokenspeed-kernel/test/amd
    --ignore=tokenspeed-kernel/test/test_numerics.py
    --ignore=tokenspeed-kernel/test/nvidia/thirdparty/test_trtllm_comm.py
    --ignore=tokenspeed-kernel/test/nvidia/thirdparty/test_cuda.py
)
for test_file in "${multi_gpu_tests[@]}"; do
    ignores+=("--ignore=${test_file}")
done

python3 -m pytest -p kernel_gpu_worker -n 4 --dist loadfile \
    --max-worker-restart=0 tokenspeed-kernel/test/ -v "${ignores[@]}" \
    --junitxml=/tmp/kernel-rest.xml
