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

set -euo pipefail

if [[ $# -lt 2 ]]; then
    echo "Usage: $0 RUN_DIR RUN_LABEL [-- CONTROLLER_ARGS ...]" >&2
    exit 2
fi
root=$(cd "$1" && pwd)
harness=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
label=$2
shift 2
if [[ ${1:-} == -- ]]; then shift; fi
cd "$root"
mkdir -p "logs/$label"
out="$root/logs/$label"

if [[ ${TS_TEST_LOGICAL_NS:-0} != 1 ]]; then
    exec 9>"${TS_TEST_LOCK_FILE:-/var/tmp/tokenspeed-logical-two-node-validation.lock}"
    flock -n 9
    nvidia-smi --query-gpu=index,memory.used,utilization.gpu \
        --format=csv,noheader,nounits > "$out/devices-before.csv"
    awk -F, '($2+0)>0 || ($3+0)>0 {bad=1} END {exit bad || NR!=8}' "$out/devices-before.csv"
    rc=0
    TS_TEST_LOGICAL_NS=1 unshare -Urn bash "$harness/test_run_logical_two_node.sh" \
        "$root" "$label" -- "$@" || rc=$?
    nvidia-smi --query-gpu=index,memory.used,utilization.gpu \
        --format=csv,noheader,nounits > "$out/devices-after.csv"
    exit "$rc"
fi

ip link set lo up
unshare -n bash -c 'printf "%s\n" "$$" > "$1"; exec sleep 1500' _ "$out/node1.pid" &
ns_process=$!
cleanup() {
    kill "$ns_process" 2>/dev/null || true
    wait "$ns_process" 2>/dev/null || true
}
trap cleanup EXIT
trap 'exit 130' INT
trap 'exit 143' TERM
for _ in {1..100}; do [[ -s $out/node1.pid ]] && break; sleep .05; done
peer=$(cat "$out/node1.pid")
ip link add tsnet type veth peer name tspeer
ip link set tspeer netns "$peer"
ip address add 10.199.0.1/30 dev tsnet
ip link set tsnet up
nsenter -t "$peer" -n ip link set lo up
nsenter -t "$peer" -n ip link set tspeer name tsnet
nsenter -t "$peer" -n ip address add 10.199.0.2/30 dev tsnet
nsenter -t "$peer" -n ip link set tsnet up
{
    printf 'node0 %s\n' "$(readlink /proc/self/ns/net)"
    ip -br address
    nsenter -t "$peer" -n bash -c 'printf "node1 %s\n" "$(readlink /proc/self/ns/net)"; ip -br address; ping -n -c1 -W2 10.199.0.1'
} > "$out/network.log"

export CUDA_DEVICE_ORDER=PCI_BUS_ID OMP_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1
export NCCL_P2P_DISABLE=1 NCCL_SHM_DISABLE=1 NCCL_IB_DISABLE=1 NCCL_NET=Socket
export NCCL_CUMEM_HOST_ENABLE=0
export NCCL_SOCKET_IFNAME='=tsnet' GLOO_SOCKET_IFNAME=tsnet
export NCCL_DEBUG=INFO NCCL_DEBUG_SUBSYS=INIT,NET,GRAPH,P2P NCCL_GRAPH_REGISTER=0
export NCCL_DEBUG_FILE="$out/nccl.%p.log"
rc=0
# Allow a 250s case, 40s shared shutdown grace, two 5s reaps, and startup margin.
# The final kill must also leave enough time to finish bounded cleanup on TERM.
timeout -s TERM -k 60 330 "${TS_TEST_PYTHON:-python3}" "$harness/test_cuda_two_node.py" \
    --netns-1 "/proc/$peer/ns/net" --output "$out/serving" "$@" || rc=$?
printf '%s\n' "$rc" > "$out/serving.exit"
exit "$rc"
