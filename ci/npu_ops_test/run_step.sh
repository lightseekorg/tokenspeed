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

# Common post-checkout entry for both authored workflow steps.
set -Eeuo pipefail

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
case "$phase" in accuracy|performance) ;; *) echo "Unknown phase: $phase" >&2; exit 1 ;; esac
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
source /srv/tokenspeed-npu/cann9-torch29-py312-root/env.sh
set -Eeuo pipefail
echo 'Preparing cache directories for this run'
export TMPDIR="$work/tmp"
export XDG_CACHE_HOME="$work/cache"
export TRITON_CACHE_DIR="$work/cache/triton"
export PYTHONPYCACHEPREFIX="$work/cache/pycache"
export PYTHONUNBUFFERED=1
mkdir -p "$TMPDIR" "$XDG_CACHE_HOME" "$TRITON_CACHE_DIR" "$PYTHONPYCACHEPREFIX"
echo "Starting phase executor: $phase"
# Keep the lock in the executor process; it cleans up the child process group.
exec /srv/tokenspeed-npu/cann9-torch29-py312-root/venv/bin/python -B -u \
  ci/npu_ops_test/run_phase.py --repo-root "$PWD" --phase "$phase" \
  --output-dir "$work/$phase" --timeout-seconds 600
