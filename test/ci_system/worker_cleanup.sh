# Shared bounded shutdown for multi-worker CI launchers.

_worker_descendants() {
  local parent=$1
  local child
  while read -r child; do
    [[ -n "$child" ]] || continue
    _worker_descendants "$child"
    echo "$child"
  done < <(pgrep -P "$parent" 2>/dev/null || true)
}

stop_worker_pids() {
  local label=$1
  local timeout=$2
  shift 2
  local -a worker_pids=("$@")
  local -a alive=()
  local -a descendants=()
  local pid
  local deadline

  ((${#worker_pids[@]})) || return 0
  echo "[$label] stopping ${#worker_pids[@]} processes"
  kill "${worker_pids[@]}" 2>/dev/null || true

  deadline=$((SECONDS + timeout))
  while ((SECONDS < deadline)); do
    alive=()
    for pid in "${worker_pids[@]}"; do
      kill -0 "$pid" 2>/dev/null && alive+=("$pid")
    done
    ((${#alive[@]})) || break
    sleep 1
  done

  alive=()
  for pid in "${worker_pids[@]}"; do
    kill -0 "$pid" 2>/dev/null && alive+=("$pid")
  done
  if ((${#alive[@]})); then
    echo "[$label] forcing ${#alive[@]} processes to stop after ${timeout}s"
    for pid in "${alive[@]}"; do
      mapfile -t descendants < <(_worker_descendants "$pid")
      kill -KILL "$pid" "${descendants[@]}" 2>/dev/null || true
    done
  fi
  wait "${worker_pids[@]}" 2>/dev/null || true
}

# After `wait -n` returns: name every worker that has exited and show the tail
# of its log, so a crash mid-run is diagnosable from the job log alone. The
# two arrays are parallel: worker_logs[i] is the log of worker_pids[i].
report_exited_workers() {
  local label=$1
  local -n _logs=$2
  local -n _pids=$3
  local i
  for i in "${!_pids[@]}"; do
    kill -0 "${_pids[$i]}" 2>/dev/null && continue
    echo "[$label] worker exited early: pid=${_pids[$i]} log=${_logs[$i]}" >&2
    tail -n 200 "${_logs[$i]}" >&2 || true
  done
}
