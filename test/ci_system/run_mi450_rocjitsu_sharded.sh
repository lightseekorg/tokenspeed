#!/bin/bash
# Run pytest as several concurrent rocJITsu shards, each with its own emulator
# daemon and Triton cache. A single emulator saturates about two cores, so a
# serial run leaves most of a many-core runner idle. Shards hold disjoint test
# files, which keeps them from recompiling the same kernels and lets one crashed
# emulator fail its own shard instead of stalling the others.
set -uo pipefail

if [ "$#" -eq 0 ]; then
    echo "usage: $0 <pytest args...>" >&2
    exit 2
fi

here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
shards="${MI450_SIM_SHARDS:-6}"
test_root="${MI450_SIM_TEST_ROOT:-tokenspeed-kernel/test}"
log_dir="${RUNNER_TEMP:-/tmp}/mi450-sim-shards"
# An emulator that aborts mid-kernel leaves its client blocked in a HIP call
# that pytest-timeout's signal cannot interrupt, so the shard would run until
# the job timeout and take the other shards' results with it.
shard_timeout="${MI450_SIM_SHARD_TIMEOUT:-2400}"

rm -rf "${log_dir}"
mkdir -p "${log_dir}"

mapfile -t plan < <(python3 "${here}/mi450_sim_shard_plan.py" \
    "${shards}" "${here}/mi450_sim_test_weights.txt" "${test_root}")
if [ "${#plan[@]}" -eq 0 ] || [ "${#plan[@]}" -gt "${shards}" ]; then
    echo "shard planner emitted ${#plan[@]} shards, expected 1 to ${shards}" >&2
    exit 1
fi
# The planner drops shards it cannot fill, so follow its count, not the request.
shards="${#plan[@]}"

# rocJITsu warns once per unsupported hardware register access, and kernels that
# poll one produce millions of identical lines: a single shard reached 18 GB of
# log in ten minutes. Keep a few examples of each warning and tally the rest.
collapse_emulator_warnings() {
    awk -v limit=20 '
        /^\[rj warn\]/ {
            key = $3
            if (++seen[key] > limit) next
        }
        { print; fflush() }
        END {
            for (key in seen)
                if (seen[key] > limit)
                    printf "[rj warn] %s suppressed %d further identical warnings\n", \
                        key, seen[key] - limit
        }
    '
}

pids=()
for shard in $(seq 0 $((shards - 1))); do
    (
        export TRITON_CACHE_DIR="${log_dir}/triton-cache-${shard}"
        started=${SECONDS}
        # Unquoted on purpose: the planner emits one argument list per shard.
        # shellcheck disable=SC2086
        timeout -k 30 "${shard_timeout}" \
            bash "${here}/run_mi450_rocjitsu.sh" python3 -m pytest \
            -p no:cacheprovider ${plan[shard]} "$@" 2>&1 |
            collapse_emulator_warnings >"${log_dir}/shard-${shard}.log"
        echo "${PIPESTATUS[0]} $((SECONDS - started))" >"${log_dir}/shard-${shard}.status"
    ) &
    pids+=("$!")
done

for pid in "${pids[@]}"; do
    wait "${pid}"
done

failed=0
empty=0
statuses=()
durations=()
for shard in $(seq 0 $((shards - 1))); do
    status=""
    seconds="?"
    if [ -f "${log_dir}/shard-${shard}.status" ]; then
        read -r status seconds <"${log_dir}/shard-${shard}.status"
    fi

    case "${status}" in
        # Exit 5 means the shard collected nothing, which is normal once every
        # test in its files skips on this architecture.
        5) empty=$((empty + 1)) ;;
        0) ;;
        124)
            echo "shard ${shard} was killed after ${shard_timeout}s" >&2
            failed=1
            ;;
        "")
            status="none"
            echo "shard ${shard} exited without reporting a status" >&2
            failed=1
            ;;
        *) failed=1 ;;
    esac

    statuses+=("${status}")
    durations+=("${seconds}")

    echo "::group::rocJITsu shard ${shard} (exit ${status}, ${seconds}s)"
    cat "${log_dir}/shard-${shard}.log" 2>/dev/null
    echo "::endgroup::"
done

if [ "${empty}" -eq "${shards}" ]; then
    echo "every shard collected zero tests" >&2
    failed=1
fi

echo
echo "shard  exit  seconds"
for shard in $(seq 0 $((shards - 1))); do
    printf '%5s  %4s  %7s\n' "${shard}" "${statuses[shard]}" "${durations[shard]}"
done

exit "${failed}"
