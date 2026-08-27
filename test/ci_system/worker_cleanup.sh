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

# True only for a running process. `kill -0` succeeds on zombies, which made
# wait_serving hang for the full startup timeout after a worker crashed on
# bind (the parent had not wait(2)ed yet).
pid_is_live() {
  local pid=$1
  local state
  [[ -n "$pid" && -d "/proc/$pid" ]] || return 1
  state=$(awk '/^State:/{print $2; exit}' "/proc/$pid/status" 2>/dev/null) || return 1
  [[ "$state" != "Z" ]]
}

_listen_pids() {
  local port=$1
  if command -v lsof >/dev/null 2>&1; then
    lsof -tiTCP:"$port" -sTCP:LISTEN 2>/dev/null || true
  elif command -v fuser >/dev/null 2>&1; then
    fuser -n tcp "$port" 2>/dev/null || true
  fi
}

# Drop leftover LISTEN sockets from a previous EPD/PD job on this runner.
free_listen_ports() {
  local label=$1
  shift
  local port pids
  for port in "$@"; do
    [[ -n "$port" ]] || continue
    pids=$(_listen_pids "$port")
    [[ -n "$pids" ]] || continue
    echo "[$label] killing stale listener(s) on port $port: $pids"
    # shellcheck disable=SC2086
    kill -TERM $pids 2>/dev/null || true
    sleep 1
    pids=$(_listen_pids "$port")
    if [[ -n "$pids" ]]; then
      # shellcheck disable=SC2086
      kill -KILL $pids 2>/dev/null || true
    fi
  done
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
      pid_is_live "$pid" && alive+=("$pid")
    done
    ((${#alive[@]})) || break
    sleep 1
  done

  alive=()
  for pid in "${worker_pids[@]}"; do
    pid_is_live "$pid" && alive+=("$pid")
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
