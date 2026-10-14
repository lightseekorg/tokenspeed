#!/usr/bin/env bash
# Shared transport wiring for local PD/EPD smoke launchers.
# Source after SCRIPT_DIR is set; do not execute this file directly.

SMG_CONNECTION_MODE=${SMG_CONNECTION_MODE:-grpc}
SMG_ZMQ_HANDSHAKE_BASE=${SMG_ZMQ_HANDSHAKE_BASE:-30700}
SMG_ZMQ_DIR=""
SMG_WORKER_SPECS=()
SMG_ZMQ_WORKER_COUNT=0
SMG_ENGINE_TRANSPORT_ARGS=()
SMG_GATEWAY_WORKER_ARGS=()
case "$SMG_CONNECTION_MODE" in
  grpc) ;;
  zmq) SMG_ZMQ_DIR=$(mktemp -d /tmp/ts-zmq.XXXXXX) ;;
  *) echo "SMG_CONNECTION_MODE must be grpc or zmq" >&2; exit 2 ;;
esac

configure_smg_worker() {
  local label=$1 role=$2 port=$3 bootstrap=$4
  SMG_ENGINE_TRANSPORT_ARGS=()
  if [[ "$SMG_CONNECTION_MODE" == grpc ]]; then
    SMG_GATEWAY_WORKER_ARGS+=("--$role" "grpc://127.0.0.1:$port")
    if [[ -n "$bootstrap" ]]; then
      SMG_GATEWAY_WORKER_ARGS+=("$bootstrap")
    fi
    return
  fi
  local handshake=$((SMG_ZMQ_HANDSHAKE_BASE + SMG_ZMQ_WORKER_COUNT))
  if [[ "$role" != encode ]]; then
    SMG_ENGINE_TRANSPORT_ARGS=(--zmq-msgpack --data-parallel-address 127.0.0.1 --data-parallel-rpc-port "$handshake")
  fi
  local spec
  spec=$(python3 - "$SMG_ZMQ_DIR/$label" "$role" "$port" "$handshake" "$bootstrap" <<'PYSPEC'
import json
import sys
path, role, port, handshake, bootstrap = sys.argv[1:]
control = f"grpc://127.0.0.1:{port}"
spec = {"url": control if role == "encode" else f"ipc://{path}",
        "runtime_type": "tokenspeed", "worker_type": role,
        "connection_mode": "grpc" if role == "encode" else "zmq",
        "max_connection_attempts": 7200}
if role != "encode":
    spec["labels"] = {"zmq_control_url": control, "zmq_connect_timeout_secs": "7200", "zmq_pd": "true"}
    spec["zmq_handshake_address"] = f"tcp://127.0.0.1:{handshake}"
if bootstrap:
    spec["bootstrap_port"] = int(bootstrap)
print(json.dumps(spec))
PYSPEC
  )
  SMG_WORKER_SPECS+=("$spec")
  SMG_ZMQ_WORKER_COUNT=$((SMG_ZMQ_WORKER_COUNT + 1))
}

configure_smg_gateway() {
  if [[ "$SMG_CONNECTION_MODE" == zmq ]]; then
    local specs
    specs=$(python3 - "${SMG_WORKER_SPECS[@]}" <<'PYSPECS'
import json
import sys
print(json.dumps([json.loads(value) for value in sys.argv[1:]]))
PYSPECS
    )
    SMG_GATEWAY_WORKER_ARGS=(--worker-specs "$specs" --worker-startup-timeout-secs "${SMG_WORKER_STARTUP_TIMEOUT:-7200}")
  fi
}

cleanup_smg_transport() {
  if [[ -n "$SMG_ZMQ_DIR" ]]; then
    rm -rf -- "$SMG_ZMQ_DIR"
  fi
}
