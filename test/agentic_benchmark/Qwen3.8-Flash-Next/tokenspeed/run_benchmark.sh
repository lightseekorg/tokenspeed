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

if (( $# != 3 )); then
    echo "Usage: bash $0 EVALSCOPE_BIN DATASET_PATH SWEEP_DIR" >&2
    exit 2
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/model_config.sh"
EVALSCOPE_BIN=$1
DATASET_PATH=$2
SWEEP_DIR=$3
mkdir -p "$SWEEP_DIR"

python3 - "$DATASET_PATH" "$QWEN_MODEL_ID" "$QWEN_MODEL_REVISION" <<'PY'
import json
import sys
from pathlib import Path

data = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
expected = {
    "model_path": sys.argv[2],
    "tokenizer_revision": sys.argv[3],
    "first_turn_length": 50000,
    "subsequent_turn_length": 800,
    "chat_template_included": True,
}
if any(data.get("metadata", {}).get(key) != value for key, value in expected.items()):
    raise SystemExit("Dataset tokenizer or token budgets do not match; run prepare.sh")
if len(data["conversations"]) < 70:
    raise SystemExit("The sweep requires 68 measured and 2 warmup conversations")
PY
if [[ ! -s "${QWEN_TOKENIZER_PATH}/tokenizer_config.json" ]]; then
    echo "Qwen tokenizer is missing; run ${SCRIPT_DIR}/prepare.sh" >&2
    exit 1
fi

# Warmup conversations 68-69 do not overlap the measured conversations 0-67.
"$EVALSCOPE_BIN" perf \
    --model "$QWEN_MODEL_ID" \
    --tokenizer-path "$QWEN_TOKENIZER_PATH" \
    --url http://127.0.0.1:8000/v1/chat/completions \
    --api openai \
    --dataset swe_smith \
    --dataset-path "$DATASET_PATH" \
    --max-tokens 500 \
    --multi-turn \
    --number 2 \
    --parallel 2 \
    --dataset-offset 68 \
    --extra-args '{"ignore_eos": true}' \
    --outputs-dir "${SWEEP_DIR}/.warmup" \
    2>&1 | tee "${SWEEP_DIR}/warmup.log"

"$EVALSCOPE_BIN" perf \
    --model "$QWEN_MODEL_ID" \
    --tokenizer-path "$QWEN_TOKENIZER_PATH" \
    --url http://127.0.0.1:8000/v1/chat/completions \
    --api openai \
    --dataset swe_smith \
    --dataset-path "$DATASET_PATH" \
    --max-tokens 500 \
    --multi-turn \
    --number 4 4 4 8 16 32 \
    --parallel 1 2 4 8 16 32 \
    --extra-args '{"ignore_eos": true}' \
    --name attn_tp4_moe_ep4 \
    --outputs-dir "$SWEEP_DIR" \
    --no-timestamp \
    2>&1 | tee "${SWEEP_DIR}/benchmark.log"

python3 "${SCRIPT_DIR}/collect_outputs.py" "$SWEEP_DIR" \
    --expected-concurrencies 1 2 4 8 16 32 \
    --output "${SWEEP_DIR}/summary.csv"
cat "${SWEEP_DIR}/summary.csv"
echo
