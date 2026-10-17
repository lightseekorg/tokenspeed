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

if (( $# != 0 )); then
    echo "Usage: bash $0" >&2
    exit 2
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../../.." && pwd)"
if [[ -z "${VIRTUAL_ENV:-}" ]]; then
    for VENV in "${REPO_ROOT}/.venv" "${REPO_ROOT}/venv"; do
        if [[ -f "${VENV}/bin/activate" ]]; then
            source "${VENV}/bin/activate"
            break
        fi
    done
fi
for COMMAND in ts curl setsid; do
    if ! command -v "$COMMAND" > /dev/null; then
        echo "Missing required command: $COMMAND" >&2
        exit 1
    fi
done

# Check before installing tools, and never benchmark an unrelated server.
python3 - <<'PY'
import socket

with socket.socket() as sock:
    try:
        sock.bind(("127.0.0.1", 8000))
    except OSError as exc:
        raise SystemExit(f"Port 8000 is unavailable: {exc}") from exc
PY

bash "${SCRIPT_DIR}/prepare.sh"
SWEEP_DIR="$(mktemp -d "${SCRIPT_DIR}/outputs/$(date -u +%Y%m%d_%H%M%S)_XXXXXX")"
SERVER_LOG="${SWEEP_DIR}/server.log"
SERVER_PID=

stop_server() {
    if [[ -n "$SERVER_PID" ]]; then
        # Workers can outlive the launcher; always signal the entire group.
        kill -TERM -- "-${SERVER_PID}" 2>/dev/null || true
        for _ in {1..20}; do
            kill -0 -- "-${SERVER_PID}" 2>/dev/null || break
            sleep 0.5
        done
        kill -KILL -- "-${SERVER_PID}" 2>/dev/null || true
        wait "$SERVER_PID" 2>/dev/null || true
        SERVER_PID=
    fi
}

trap stop_server EXIT
trap 'exit 130' INT
trap 'exit 143' TERM

echo "Sweep outputs: ${SWEEP_DIR}"
setsid bash "${SCRIPT_DIR}/configs/attn_tp4_moe_ep4.sh" > "$SERVER_LOG" 2>&1 &
SERVER_PID=$!
START=$SECONDS
until curl --fail --silent --max-time 5 -o /dev/null http://127.0.0.1:8000/readiness; do
    if ! kill -0 "$SERVER_PID" 2>/dev/null; then
        echo "Server exited before becoming ready:" >&2
        tail -100 "$SERVER_LOG" >&2
        exit 1
    fi
    if (( SECONDS - START >= 7200 )); then
        echo "Timed out waiting for the server after 7200s:" >&2
        tail -100 "$SERVER_LOG" >&2
        exit 1
    fi
    sleep 5
done
echo "Server ready after $((SECONDS - START))s"

bash "${SCRIPT_DIR}/run_benchmark.sh" \
    "${SCRIPT_DIR}/.venv-evalscope/bin/evalscope" \
    "${SCRIPT_DIR}/data/agentic_dataset.json" \
    "$SWEEP_DIR"
