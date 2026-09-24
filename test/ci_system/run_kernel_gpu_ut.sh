#!/usr/bin/env bash
set -euo pipefail

export PYTHONPATH="test/ci_system:python:tokenspeed-kernel/python${PYTHONPATH:+:$PYTHONPATH}"

check_reports() {
    python3 - "$@" <<'PY'
import sys
import xml.etree.ElementTree as ET

for path in sys.argv[1:]:
    root = ET.parse(path).getroot()
    suite = root if root.tag == "testsuite" else root.find("testsuite")
    assert suite is not None, path
    assert int(suite.attrib["tests"]) > 0, path
    assert int(suite.attrib["skipped"]) == 0, path
PY
}

run_distributed() {
    local name="$1"
    shift
    # The child shell expands these rank variables.
    # shellcheck disable=SC2016
    python3 -m torch.distributed.run --standalone --nproc-per-node=4 --no-python \
        bash -c 'exec python3 -m pytest "$@" -v --junitxml="/tmp/kernel-${0}-${RANK}.xml"' \
        "$name" "$@"
    local reports=()
    for rank in 0 1 2 3; do
        reports+=("/tmp/kernel-${name}-${rank}.xml")
    done
    check_reports "${reports[@]}"
}

python3 -m pytest tokenspeed-kernel/test/ops/test_communcation.py -v \
    --junitxml=/tmp/kernel-multi-gpu.xml
check_reports /tmp/kernel-multi-gpu.xml
python3 -m pytest tokenspeed-kernel/test/ops/test_attention_dsv41_index_scan.py \
    -k 'not tp4' -v --junitxml=/tmp/kernel-index-scan.xml
check_reports /tmp/kernel-index-scan.xml

TOKENSPEED_TEST_TP4=1 run_distributed tp4 \
    tokenspeed-kernel/test/ops/test_attention_dsv41_index_scan.py -k tp4
run_distributed marlin \
    tokenspeed-kernel/test/nvidia/ops/moe/test_marlin_deepep_distributed.py
run_distributed mnnvl \
    tokenspeed-kernel/test/nvidia/thirdparty/test_trtllm_mnnvl_comm.py
run_distributed mnnvl-twoshot \
    tokenspeed-kernel/test/nvidia/thirdparty/test_trtllm_mnnvl_twoshot.py

ignores=(
    --ignore=tokenspeed-kernel/test/amd
    --ignore=tokenspeed-kernel/test/test_numerics.py
    --ignore=tokenspeed-kernel/test/nvidia/thirdparty/test_trtllm_comm.py
    --ignore=tokenspeed-kernel/test/nvidia/thirdparty/test_cuda.py
    --ignore=tokenspeed-kernel/test/ops/test_communcation.py
    --ignore=tokenspeed-kernel/test/ops/test_attention_dsv41_index_scan.py
    --ignore=tokenspeed-kernel/test/nvidia/ops/moe/test_marlin_deepep_distributed.py
    --ignore=tokenspeed-kernel/test/nvidia/thirdparty/test_trtllm_mnnvl_comm.py
    --ignore=tokenspeed-kernel/test/nvidia/thirdparty/test_trtllm_mnnvl_twoshot.py
    # These suites require eight or sixteen ranks; this allocation has four GPUs.
    --ignore=tokenspeed-kernel/test/nvidia/ops/communication/test_multimem_distributed.py
    --ignore=tokenspeed-kernel/test/nvidia/ops/moe/test_latent_tail_distributed.py
)

python3 -m pytest -p kernel_gpu_worker -n 4 --dist loadfile \
    --max-worker-restart=0 tokenspeed-kernel/test/ -v "${ignores[@]}" \
    --junitxml=/tmp/kernel-rest.xml
