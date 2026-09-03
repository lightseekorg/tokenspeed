#!/bin/bash
# Stand in for the Python interpreter that pytest-xdist starts its workers with.
#
# xdist launches each worker as "<python> -u -c <bootstrap>" and then speaks
# execnet's protocol over that process's stdin and stdout. Naming this script as
# the interpreter puts every worker under its own rocJITsu daemon, so the workers
# do not queue their kernels behind one shared emulator.
set -euo pipefail

here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Nothing in this path may write to stdout; execnet would read it as protocol
# data and the worker would fail to hand shake.
exec bash "${here}/run_mi450_rocjitsu.sh" \
    "${MI450_SIM_WORKER_PYTHON:-python3}" "$@"
