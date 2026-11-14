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

# Test only: current merged source, CUDA v2 on both logical nodes.
# Generate models/moe4 and prewarm FlashInfer in the serving environment first.
set -euo pipefail
if [[ $# != 2 ]]; then
    echo "Usage: $0 RUN_DIR SOURCE_DIR" >&2
    exit 2
fi
root=$(cd "$1" && pwd)
harness=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
source_root=$(cd "$2" && pwd)
python=${TS_TEST_PYTHON:-$(command -v python3)}
export TS_TEST_PYTHON="$python"
cd "$root"
mkdir -p logs
test -f "$root/models/moe4/config.json"
test -x "$source_root/tokenspeed-engine/python/tokenspeed_engine/bin/ts-host"
test -f "$harness/test_cuda_two_node.py"
test -f "$harness/test_run_logical_two_node.sh"

# Keep installation-specific libraries explicit; otherwise use the active environment.
for library_dir in "${TS_TEST_CUDA_RUNTIME_LIB_DIR:-}" "${TS_TEST_PYTHON_LIB_DIR:-}" "${TS_TEST_NCCL_LIB_DIR:-}"; do
    if [[ -n "$library_dir" ]]; then
        test -d "$library_dir"
        export LD_LIBRARY_PATH="$library_dir${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
    fi
done
export FLASHINFER_CUDA_ARCH_LIST=${FLASHINFER_CUDA_ARCH_LIST:-9.0a} FLASHINFER_JIT_DEBUG=0
export TOKENSPEED_KERNEL_VERBOSE=1
unset TOKENSPEED_KERNEL_OVERRIDE_MOE_APPLY
# Normal startup must install exactly one saver hook without caller injection.
unset LD_PRELOAD
runtime_path="$source_root/python:$source_root/tokenspeed-engine/python:$source_root/tokenspeed-kernel/python:$source_root"
if [[ -n "${TS_TEST_EXTRA_PYTHONPATH:-}" ]]; then
    runtime_path="$TS_TEST_EXTRA_PYTHONPATH:$runtime_path"
fi
export PYTHONPATH="$runtime_path${PYTHONPATH:+:$PYTHONPATH}"

sha256sum \
    "$source_root/python/tokenspeed/runtime/engine/rust_host.py" \
    "$source_root/python/tokenspeed/runtime/engine/memory_protocol.py" \
    "$source_root/python/tokenspeed/runtime/execution/weight_memory.py" \
    "$source_root/tokenspeed-engine/crates/ts-control/src/event_loop.rs" \
    "$source_root/tokenspeed-engine/crates/ts-control/src/control.rs" \
    "$source_root/tokenspeed-engine/crates/ts-host/src/main.rs" \
    "$source_root/tokenspeed-engine/python/tokenspeed_engine/bin/ts-host" \
    "$source_root"/tokenspeed-engine/python/tokenspeed_engine/_native*.so \
    "$harness/test_cuda_two_node.py" \
    "$harness/cuda_two_node_probe/sitecustomize.py" \
    "$harness/test_summarize_cuda_two_node_serving.py" \
    > logs/source-inputs.sha256

"$python" - <<'PY_IMPORTS' > logs/runtime-imports.json
import importlib
import importlib.metadata
import json

modules = ["transformers", "torch", "numpy", "tokenspeed_scheduler",
           "tokenspeed_scheduler.tokenspeed_scheduler_ext", "tokenspeed_engine._native",
           "tokenspeed_kernel"]
paths = {name: importlib.import_module(name).__file__ for name in modules}
versions = {name: importlib.metadata.version(name) for name in
            ["transformers", "torch", "numpy", "compressed-tensors"]}
print(json.dumps({"versions": versions, "module_paths": paths}, indent=2))
PY_IMPORTS

case_dirs=()
run() {
    local label=$1 layout=$2
    shift 2
    local rc=0
    case_dirs+=("$root/logs/$label")
    bash "$harness/test_run_logical_two_node.sh" "$root" "$label" -- \
        --layout "$layout" --protocols 2,2 --model "$root/models/moe4" \
        --cards-0 "${TS_TEST_CARDS_0:-0,1}" --cards-1 "${TS_TEST_CARDS_1:-4,5}" \
        --ts-host-0 "$source_root/tokenspeed-engine/python/tokenspeed_engine/bin/ts-host" \
        --ts-host-1 "$source_root/tokenspeed-engine/python/tokenspeed_engine/bin/ts-host" \
        --python-0 "$python" --python-1 "$python" \
        --pythonpath-0 "$runtime_path" \
        --pythonpath-1 "$runtime_path" \
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
    label="serve-flashinfer-$layout-v2"
    run "$label" "$layout" || failed=1
    run "$label-nccl" "$layout" --nccl || failed=1
done
if (( failed != 0 )); then exit "$failed"; fi
"$python" "$harness/test_summarize_cuda_two_node_serving.py" "${case_dirs[@]}" \
    > logs/cuda-v2-serving-summary.json

# Preserve network and native suspend/resume proof in compact public excerpts.
"$python" - "${case_dirs[@]}" <<'PYTHON'
import json
import re
import sys
from pathlib import Path

for name in sys.argv[1:]:
    directory = Path(name)
    result = json.loads((directory / "serving/result.json").read_text())
    assert result["protocols"] == [2, 2] and result["status"] == "passed"
    socket_lines = []
    for log in sorted(directory.glob("nccl.*.log")):
        rows = [line for line in log.read_text(errors="replace").splitlines()
                if "NET/Socket" in line]
        assert rows, (directory.name, log.name)
        socket_lines.extend(line[line.index("NCCL INFO"):] for line in rows)
    assert socket_lines, directory.name
    (directory / "socket-proof.log").write_text("\n".join(socket_lines) + "\n")
    if "nccl" in result["tags"]:
        text = "\n".join(log.read_text(errors="replace")
                         for log in sorted((directory / "serving").glob("node-*.log")))
        groups = 2 if result["layout"] == "ep4" else 3
        assert re.findall(r"Suspended (\d+) NCCL", text) == [str(groups)] * 4
        assert re.findall(r"Resumed (\d+) NCCL", text) == [str(groups)] * 4
        excerpts = [line for line in text.splitlines()
                    if "Suspended " in line or "Resumed " in line]
        clean = re.sub(r"\x1b\[[0-9;]*m", "", "\n".join(excerpts))
        (directory / "nccl-suspend-resume.log").write_text(clean + "\n")
    print(directory.name, "passed")
PYTHON
