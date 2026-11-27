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

# Post-checkout entry for the NPU install smoke and NPU runtime unit tests.
# Phases:
#   install-smoke: run test/ci_system/install_triton_ascend.sh end to end, then
#     assert `tokenspeed.cli env` sees NPU devices and the ulimit lines, and
#     that `tokenspeed.cli --help` imports the runtime.
#   runtime: run the in-tree NPU runtime unit tests (pure single-NPU tests; no
#     weights, no network). Scheduler/engine native bindings are not tracked
#     in the checkout, so those packages come from the deployed CI venv.
set -Eeuo pipefail

NPU_VENV_PYTHON=/srv/tokenspeed-npu/cann9-torch29-py312-root/venv/bin/python
NPU_ENV_SCRIPT=/srv/tokenspeed-npu/cann9-torch29-py312-root/env.sh

check_platform() {
  [[ "$1" = Linux ]] || { echo "Linux is required; actual system: $1" >&2; return 1; }
  case "$2" in
    aarch64|arm64) ;;
    *) echo "ARM64 is required; actual architecture: $2" >&2; return 1 ;;
  esac
}

[[ "$#" = 2 ]]
phase=$1
work=$2
case "$phase" in install-smoke|runtime) ;; *) echo "Unknown phase: $phase" >&2; exit 1 ;; esac
echo "Environment checks started: phase=$phase, work=$work"
check_platform "$(uname -s)" "$(uname -m)"
cd "$work/checkout"
actual_sha=$(git rev-parse HEAD)
[[ "$actual_sha" = "$GITHUB_SHA" ]] || { echo "Checkout SHA mismatch: $actual_sha" >&2; exit 1; }
status=$(git status --porcelain --untracked-files=all)
[[ -z "$status" ]] || { echo 'Checkout contains modified or untracked files' >&2; exit 1; }
echo 'Acquiring host lock: /srv/npu-ci/locks/host-all.lock'
test -f /srv/npu-ci/locks/host-all.lock || { echo 'Host lock file does not exist' >&2; exit 1; }
exec 9<> /srv/npu-ci/locks/host-all.lock
flock -n 9 || { echo 'Host lock is held; tests were not started' >&2; exit 1; }
# Vendor setup may reference unset optional variables.
echo 'Loading NPU environment: /srv/tokenspeed-npu/cann9-torch29-py312-root/env.sh'
set +u
source "$NPU_ENV_SCRIPT"
set -Eeuo pipefail
# Keep the deployed CI venv at the front of PATH for `python`/`pip` calls in
# test/ci_system/install_triton_ascend.sh and the pytest invocations below.
export PATH="/srv/tokenspeed-npu/cann9-torch29-py312-root/venv/bin:${PATH:-}"
echo 'Preparing cache directories for this run'
export TMPDIR="$work/tmp"
export XDG_CACHE_HOME="$work/cache"
export TRITON_CACHE_DIR="$work/cache/triton"
export PYTHONPYCACHEPREFIX="$work/cache/pycache"
export PYTHONUNBUFFERED=1
mkdir -p "$TMPDIR" "$XDG_CACHE_HOME" "$TRITON_CACHE_DIR" "$PYTHONPYCACHEPREFIX"
# Expose the three in-tree source packages (python, kernel, kernel-npu) exactly
# as docs/recipes/models.md describes. The scheduler/engine source directories
# are NOT added: their native bindings are gitignored in the checkout, so the
# installed scheduler/engine in the deployed CI venv must be used instead.
export PYTHONPATH="${PWD}/python:${PWD}/tokenspeed-kernel/python:${PWD}/tokenspeed-kernel-npu/python:${PYTHONPATH:-}"

if [[ "$phase" = install-smoke ]]; then
  echo "Running install smoke: test/ci_system/install_triton_ascend.sh"
  bash test/ci_system/install_triton_ascend.sh
  echo 'Running runtime self-check: python -m tokenspeed.cli env'
  env_output=$("$NPU_VENV_PYTHON" -m tokenspeed.cli env 2>&1)
  printf '%s\n' "$env_output"
  grep -q '^NPU available: True' <<<"$env_output"
  grep -qE '^NPU [0-9]' <<<"$env_output"
  grep -q '^ulimit -l (soft):' <<<"$env_output"
  grep -q '^triton-ascend:' <<<"$env_output"
  echo 'Running import smoke: python -m tokenspeed.cli --help'
  "$NPU_VENV_PYTHON" -m tokenspeed.cli --help >/dev/null
  echo "Install smoke passed."
else
  echo "Running NPU runtime unit tests"
  "$NPU_VENV_PYTHON" -m pytest -p no:cacheprovider -q \
    test/runtime/layers/test_npu_attention_backend.py \
    test/runtime/execution/test_breakable_cuda_graph_npu.py \
    test/runtime/test_longcat_ngram_history_embedding.py \
    test/runtime/test_ascend_dsa_cache_npu.py \
    test/runtime/test_npu_version_evidence.py
fi
