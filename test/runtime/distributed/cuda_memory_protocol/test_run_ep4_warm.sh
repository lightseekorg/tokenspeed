#!/usr/bin/env bash
# Copyright (c) 2026 LightSeek Foundation
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

set -eu
if [[ $# != 1 ]]; then
    echo "Usage: $0 RUN_DIR" >&2
    exit 2
fi
harness=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
cd "$1"
mkdir -p logs

sp=/usr/local/conda/lib/python3.11/site-packages
export LD_LIBRARY_PATH=/opt/nccl-mt/lib:/usr/local/conda/lib:$sp/nvidia/cuda_runtime/lib:${LD_LIBRARY_PATH:-}
export FLASHINFER_CUDA_ARCH_LIST=9.0a FLASHINFER_JIT_DEBUG=0
export TOKENSPEED_KERNEL_VERBOSE=1
unset TOKENSPEED_KERNEL_OVERRIDE_MOE_APPLY
# Exercise the normal launcher's automatic hook setup rather than inheriting it.
unset LD_PRELOAD

scoped="$PWD/offline/scoped"
shared="$PWD/offline/runtime-extra:$PWD/offline/kernel/tokenspeed-kernel/python"
export PYTHONPATH="$shared:$scoped/python:$scoped/tokenspeed-engine/python${PYTHONPATH:+:$PYTHONPATH}"

run() {
    local label=$1 layout=$2 versions=$3 a=$4 b=$5
    shift 5
    local aa="$PWD/offline/$a" bb="$PWD/offline/$b"
    local rc=0
    bash "$harness/test_run_logical_two_node.sh" "$PWD" "$label" -- \
        --layout "$layout" --protocols "$versions" --model "$PWD/models/moe4" \
        --cards-0 0,1 --cards-1 4,5 \
        --ts-host-0 "$aa/tokenspeed-engine/python/tokenspeed_engine/bin/ts-host" \
        --ts-host-1 "$bb/tokenspeed-engine/python/tokenspeed_engine/bin/ts-host" \
        --pythonpath-0 "$shared:$aa/python:$aa/tokenspeed-engine/python" \
        --pythonpath-1 "$shared:$bb/python:$bb/tokenspeed-engine/python" \
        --startup-timeout 90 --deadline 150 "$@" \
        > "logs/$label-supervisor.log" 2>&1 || rc=$?
    local node
    for node in 0 1; do
        awk -v node="$node" '/moe\.apply.*-> flashinfer_cutlass_unquant_moe_apply/ {
            print "node-" node ": " $0
            found = 1
            exit
        } END { if (!found) exit 1 }' "logs/$label/serving/node-$node.log" || {
            if (( rc == 0 )); then rc=1; fi
        }
    done > "logs/$label/kernel-selection.log"
    printf '%s %s\n' "$label" "$rc"
    return "$rc"
}

failed=0
for layout in ep4 dp2-tp2-ep4; do
    label="serve-flashinfer-$layout"
    run "$label-v2" "$layout" 2,2 scoped scoped || failed=1
    run "$label-v3" "$layout" 3,3 candidate candidate --delay-rank 3 --fanout-delay 0.3 || failed=1
    run "$label-v3-nccl" "$layout" 3,3 candidate candidate --nccl --delay-rank 3 --fanout-delay 0.3 || failed=1
    run "$label-mixed-23" "$layout" 2,3 scoped candidate || failed=1
    run "$label-mixed-32" "$layout" 3,2 candidate scoped || failed=1
done
exit "$failed"
