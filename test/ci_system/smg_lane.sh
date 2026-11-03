# Shared pieces of the CI launchers that serve headless TokenSpeed engines
# behind SMG's gateway on its direct ZMQ lane: the gateway binds each worker's
# sockets, and the engine's ranks dial its handshake.

# A private directory for the workers' ipc:// sockets: the gateway refuses
# to bind in a directory another user owns.
lane_socket_dir() {
  mktemp -d "${TMPDIR:-/tmp}/smg-zmq-ci.XXXXXX"
}

# The TCP handshake port the gateway derives from an ipc:// worker URL.
lane_handshake_port() {
  python3 -c 'import sys; from tokenspeed.cli._proc import zmq_handshake_port; print(zmq_handshake_port(sys.argv[1]))' "$1"
}

# The engine flags that dial a worker's handshake at host:port; the engine
# identities on a worker count from 0.
lane_dial_args() {
  local host=$1
  local port=$2
  echo --data-parallel-address "$host" --data-parallel-rpc-port "$port" --zmq-engine-index 0
}

# Wait until url answers 2xx, or fail once deadline (a SECONDS value) passes
# or any of the given processes exits.
lane_wait_http() {
  local label=$1
  local name=$2
  local url=$3
  local deadline=$4
  shift 4
  local pid
  until curl --connect-timeout 2 --max-time 5 -fsS "$url" >/dev/null 2>&1; do
    for pid in "$@"; do
      if ! kill -0 "$pid" 2>/dev/null; then
        echo "[$label] process $pid exited before $name was ready" >&2
        return 1
      fi
    done
    if ((SECONDS >= deadline)); then
      echo "[$label] timed out waiting for $name at $url" >&2
      return 1
    fi
    sleep 5
  done
  echo "[$label] $name ready at $url"
}
