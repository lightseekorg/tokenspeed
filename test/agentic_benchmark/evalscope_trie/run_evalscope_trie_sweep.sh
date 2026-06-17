#!/usr/bin/env bash
set -Eeuo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

MODEL=${MODEL:-DeepSeek-V4-Pro}
API_URL=${API_URL:-http://127.0.0.1:8000/v1/chat/completions}
API=${API:-openai}
API_KEY=${API_KEY:-EMPTY}
TOKENIZER_PATH=${TOKENIZER_PATH:-}

DATASET=${DATASET:-trie_agentic_coding}
DATASET_PATH=${DATASET_PATH:-}
MAX_TOKENS=${MAX_TOKENS:-500}
EXTRA_ARGS=${EXTRA_ARGS:-'{"ignore_eos": true, "temperature": 0.0}'}

NUM_GPUS=${NUM_GPUS:-8}
OUTPUT_DIR=${OUTPUT_DIR:-"$SCRIPT_DIR/outputs/$(date +%Y%m%d_%H%M%S)"}
EVALSCOPE_VENV=${EVALSCOPE_VENV:-"$OUTPUT_DIR/.evalscope-venv"}
EVALSCOPE_BIN=${EVALSCOPE_BIN:-"$EVALSCOPE_VENV/bin/evalscope"}

WARMUP1_PARALLEL=${WARMUP1_PARALLEL:-2}
WARMUP1_NUMBER=${WARMUP1_NUMBER:-2}
WARMUP2_PARALLEL=${WARMUP2_PARALLEL:-8}
WARMUP2_NUMBER=${WARMUP2_NUMBER:-16}
FORMAL_PARALLELS=${FORMAL_PARALLELS:-"1 2 4 8"}
FORMAL_NUMBERS=${FORMAL_NUMBERS:-"8 16 32 64"}

STREAM=${STREAM:-1}
TOKENIZE_PROMPT=${TOKENIZE_PROMPT:-0}
NO_TEST_CONNECTION=${NO_TEST_CONNECTION:-1}
HEALTH_URL=${HEALTH_URL:-}

log() {
  printf '[%s] %s\n' "$(date '+%F %T %Z')" "$*" >&2
}

usage() {
  cat <<'EOF'
Run EvalScope trie agentic perf against an already-running server.

Required:
  DATASET_PATH=/path/to/trie/workloads/agentic_coding_8k.jsonl

Common overrides:
  MODEL=DeepSeek-V4-Pro
  API_URL=http://127.0.0.1:8000/v1/chat/completions
  TOKENIZER_PATH=/path/to/tokenizer
  OUTPUT_DIR=/path/to/output
  FORMAL_PARALLELS="1 2 4 8"
  FORMAL_NUMBERS="8 16 32 64"
  TOKENIZE_PROMPT=1  # useful with /v1/completions plain-prompt replay
EOF
}

require_dataset() {
  if [[ -z "$DATASET_PATH" ]]; then
    usage >&2
    exit 2
  fi
  if [[ ! -f "$DATASET_PATH" ]]; then
    echo "DATASET_PATH does not exist: $DATASET_PATH" >&2
    exit 2
  fi
}

ensure_evalscope() {
  if [[ -x "$EVALSCOPE_BIN" ]]; then
    return
  fi
  log "installing evalscope[perf] into $EVALSCOPE_VENV"
  if command -v uv >/dev/null 2>&1; then
    uv venv --seed --clear "$EVALSCOPE_VENV"
    uv pip install --python "$EVALSCOPE_VENV/bin/python" "evalscope[perf]"
  elif python3 -m uv --version >/dev/null 2>&1; then
    python3 -m uv venv --seed --clear "$EVALSCOPE_VENV"
    python3 -m uv pip install --python "$EVALSCOPE_VENV/bin/python" "evalscope[perf]"
  else
    python3 -m venv --clear "$EVALSCOPE_VENV"
    "$EVALSCOPE_VENV/bin/python" -m pip install --upgrade pip
    "$EVALSCOPE_VENV/bin/python" -m pip install "evalscope[perf]"
  fi
}

wait_for_health() {
  if [[ -z "$HEALTH_URL" ]]; then
    return
  fi
  log "waiting for health endpoint: $HEALTH_URL"
  for _ in {1..120}; do
    if curl --max-time 10 -fsS "$HEALTH_URL" >/dev/null; then
      log "server is healthy"
      return
    fi
    sleep 5
  done
  echo "Timed out waiting for $HEALTH_URL" >&2
  exit 1
}

check_pair_counts() {
  local numbers=$1
  local parallels=$2
  local n_count p_count
  read -r -a n_count <<< "$numbers"
  read -r -a p_count <<< "$parallels"
  if [[ ${#n_count[@]} -ne ${#p_count[@]} ]]; then
    echo "number/parallel list lengths differ: '$numbers' vs '$parallels'" >&2
    exit 2
  fi
}

run_evalscope() {
  local label=$1
  local numbers=$2
  local parallels=$3
  local out_dir="$OUTPUT_DIR/$label"
  local log_file="$OUTPUT_DIR/${label}.log"
  local number_args parallel_args cmd

  check_pair_counts "$numbers" "$parallels"
  read -r -a number_args <<< "$numbers"
  read -r -a parallel_args <<< "$parallels"

  rm -rf "$out_dir"
  mkdir -p "$out_dir"

  cmd=(
    "$EVALSCOPE_BIN" perf
    --model "$MODEL"
    --url "$API_URL"
    --api "$API"
    --api-key "$API_KEY"
    --dataset "$DATASET"
    --dataset-path "$DATASET_PATH"
    --max-tokens "$MAX_TOKENS"
    --multi-turn
    --number "${number_args[@]}"
    --parallel "${parallel_args[@]}"
    --extra-args "$EXTRA_ARGS"
    --no-timestamp
    --outputs-dir "$out_dir"
  )

  if [[ -n "$TOKENIZER_PATH" ]]; then
    cmd+=(--tokenizer-path "$TOKENIZER_PATH")
  fi
  if [[ "$STREAM" == "1" ]]; then
    cmd+=(--stream)
  fi
  if [[ "$TOKENIZE_PROMPT" == "1" ]]; then
    cmd+=(--tokenize-prompt)
  fi
  if [[ "$NO_TEST_CONNECTION" == "1" ]]; then
    cmd+=(--no-test-connection)
  fi

  log "running $label: numbers=[$numbers], parallels=[$parallels], out=$out_dir"
  "${cmd[@]}" 2>&1 | tee "$log_file"
}

collect_results() {
  local csv="$OUTPUT_DIR/sweep.csv"
  local svg="$OUTPUT_DIR/sweep.svg"
  log "collecting results into $csv and $svg"
  python3 "$SCRIPT_DIR/collect_outputs.py" "$OUTPUT_DIR" \
    --num-gpus "$NUM_GPUS" \
    --csv "$csv" \
    --svg "$svg"
}

main() {
  require_dataset
  mkdir -p "$OUTPUT_DIR"
  log "output_dir=$OUTPUT_DIR"
  log "model=$MODEL api_url=$API_URL dataset=$DATASET dataset_path=$DATASET_PATH"
  ensure_evalscope
  wait_for_health
  run_evalscope "warmup_p${WARMUP1_PARALLEL}_n${WARMUP1_NUMBER}" "$WARMUP1_NUMBER" "$WARMUP1_PARALLEL"
  run_evalscope "warmup_p${WARMUP2_PARALLEL}_n${WARMUP2_NUMBER}" "$WARMUP2_NUMBER" "$WARMUP2_PARALLEL"
  run_evalscope "sweep" "$FORMAL_NUMBERS" "$FORMAL_PARALLELS"
  collect_results
  log "done"
}

main "$@"
