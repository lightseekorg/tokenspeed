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
model_path="${1:?Provide the local model directory}"
tp="${2:?Provide TP size 1 or 2}"
service_port="${3:?Provide an unused localhost port}"
model_name="${4:?Provide the served model name}"
[[ "$tp" == 1 || "$tp" == 2 ]] || { echo 'TP must be 1 or 2.' >&2; exit 1; }
: "${ASCEND_RT_VISIBLE_DEVICES:?Choose physical chip IDs}"
IFS=, read -r -a chips <<< "$ASCEND_RT_VISIBLE_DEVICES"
[[ "${#chips[@]}" == "$tp" ]] || { echo 'Expose exactly one chip per rank.' >&2; exit 1; }
source scripts/ascend_env.sh
if [[ "$tp" == 2 && $((10#${chips[0]} / 2)) != $((10#${chips[1]} / 2)) ]]; then
    echo 'Choose the two chips of the same board.' >&2
    exit 1
fi
export OMP_NUM_THREADS=4
exec python -m tokenspeed.cli serve "$model_path" \
  --served-model-name "$model_name" --device npu --dtype float16 \
  --kv-cache-dtype float16 --attention-backend mha --sampling-backend greedy \
  --disable-prefill-graph --disable-pdl --enforce-eager \
  --disable-prefix-caching --disable-kvstore --language-model-only \
  --enable-output-logprobs --seed 2026 --mamba-ssm-dtype float32 \
  --world-size "$tp" --attn-tp-size "$tp" --dense-tp-size "$tp" \
  --max-model-len 4096 --max-num-seqs 4 --max-total-tokens 16384 \
  --chunked-prefill-size 4096 --prefix-granularity 128 \
  --retraction-snapshot-ratio 0 --disable-autotune \
  --host 127.0.0.1 --port "$service_port"
