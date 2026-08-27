#!/bin/bash
# Run the test suite as several concurrent rocJITsu workers on one runner.
#
# A single emulator saturates about two cores, so running the suite serially
# leaves most of a 32-core runner idle. Six pytest-xdist workers under one
# shared daemon are no faster than serial, because the daemon serialises their
# kernels; on one directory that measured 207s against 80s. Pointing xdist's
# --tx gateway at a wrapper that starts its own daemon gives each worker a
# private emulator, which recovers the speed (83s on the same directory) while
# leaving the scheduling to xdist, so no per-file cost table has to be kept in
# step with the tests as they are added and removed.
set -uo pipefail

if [ "$#" -eq 0 ]; then
    echo "usage: $0 <pytest args...>" >&2
    exit 2
fi

here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
workers="${MI450_SIM_WORKERS:-6}"
test_root="${MI450_SIM_TEST_ROOT:-tokenspeed-kernel/test}"
log_dir="${RUNNER_TEMP:-/tmp}/mi450-sim"
# An emulator that aborts mid-kernel leaves its worker blocked in a HIP call
# that pytest-timeout's signal cannot interrupt, and the controller waits on
# that worker before printing anything. Bound the run so the digest below still
# gets to report what the other workers finished.
run_timeout="${MI450_SIM_RUN_TIMEOUT:-2700}"

rm -rf "${log_dir}"
mkdir -p "${log_dir}"

report_log="${log_dir}/reports.jsonl"
# One cache for all workers: a kernel compiled while one file runs is then
# reused when a later file lands on a different worker.
export TRITON_CACHE_DIR="${log_dir}/triton-cache"
# The wrapper has to launch the controller's interpreter, not whichever
# python3 the emulator's environment puts first on PATH.
MI450_SIM_WORKER_PYTHON="$(python3 -c 'import sys; print(sys.executable)')"
export MI450_SIM_WORKER_PYTHON

# rocJITsu warns once per unsupported hardware register access, and kernels that
# poll one produce millions of identical lines: a single run reached 18 GB of log
# in ten minutes. Keep a few examples of each warning and tally the rest. The
# daemons share this stream, so a warning can land mid-line on top of a test
# result; dropping such a line is safe because the digest, not the console, is
# what reports results.
collapse_emulator_warnings() {
    awk -v limit=20 '
        match($0, /\[rj warn\][[:space:]]+[A-Za-z_]+/) {
            key = substr($0, RSTART, RLENGTH)
            if (++seen[key] > limit) next
        }
        { print; fflush() }
        END {
            for (key in seen)
                if (seen[key] > limit)
                    printf "%s suppressed %d further identical warnings\n", \
                        key, seen[key] - limit
        }
    '
}

timeout -k 30 "${run_timeout}" \
    bash "${here}/run_mi450_rocjitsu.sh" python3 -m pytest \
    -p no:cacheprovider \
    --dist loadfile \
    --tx "${workers}*popen//python=${here}/mi450_rocjitsu_worker_python.sh" \
    --report-log="${report_log}" \
    "${test_root}" "$@" 2>&1 | collapse_emulator_warnings
status="${PIPESTATUS[0]}"

case "${status}" in
    124 | 137)
        echo "the run was killed after ${run_timeout}s without finishing" >&2
        ;;
    5)
        echo "pytest collected no tests" >&2
        ;;
esac

echo
python3 "${here}/mi450_sim_report_digest.py" "${report_log}"
digest_status="$?"

if [ "${status}" -ne 0 ] || [ "${digest_status}" -ne 0 ]; then
    exit 1
fi
