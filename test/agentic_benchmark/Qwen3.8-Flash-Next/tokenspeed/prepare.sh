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
source "${SCRIPT_DIR}/model_config.sh"
EVALSCOPE_COMMIT=acd09b44384d53174768bb1063f675420f76fae9
PERF_VENV="${SCRIPT_DIR}/.venv-evalscope"

# Keep EvalScope's dependencies separate from the serving environment.
# The serving environment's pip can also seed systems without ensurepip.
python3 -m venv --without-pip "$PERF_VENV"
python3 -m pip --python "${PERF_VENV}/bin/python" install \
    "evalscope[perf] @ git+https://github.com/modelscope/evalscope.git@${EVALSCOPE_COMMIT}"
mkdir -p "${SCRIPT_DIR}/data/source" "${SCRIPT_DIR}/outputs"
"${PERF_VENV}/bin/python" - "${SCRIPT_DIR}/data/source" \
    "$QWEN_MODEL_ID" "$QWEN_MODEL_REVISION" "$QWEN_TOKENIZER_PATH" <<'PY'
import sys

from huggingface_hub import hf_hub_download, snapshot_download

hf_hub_download(
    repo_id="lightseekorg/agentic-dataset",
    filename="agentic_dataset.json",
    repo_type="dataset",
    revision="2779acd48a05c211f2f82eee71bdbdf688d984ca",
    local_dir=sys.argv[1],
)
# EvalScope loads local assets so its ModelScope wrapper cannot resolve a
# different repository or revision for the tokenizer.
snapshot_download(
    repo_id=sys.argv[2],
    revision=sys.argv[3],
    local_dir=sys.argv[4],
    allow_patterns=[
        "tokenizer*", "chat_template*", "special_tokens_map*", "config.json",
        "vocab*", "merges*", "*.model", "*.tiktoken",
    ],
)
PY
"${PERF_VENV}/bin/python" "${SCRIPT_DIR}/build_dataset.py" \
    --source-path "${SCRIPT_DIR}/data/source/agentic_dataset.json" \
    --output-path "${SCRIPT_DIR}/data/agentic_dataset.json" \
    --model-path "$QWEN_MODEL_ID" \
    --tokenizer-path "$QWEN_TOKENIZER_PATH" \
    --tokenizer-revision "$QWEN_MODEL_REVISION" \
    --source-revision 2779acd48a05c211f2f82eee71bdbdf688d984ca \
    --first-turn-length 50000 \
    --subsequent-turn-length 800
test -s "${SCRIPT_DIR}/data/agentic_dataset.json"
