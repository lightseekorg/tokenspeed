#!/usr/bin/env bash
# Stress scenario: Kimi-K2.5 NVFP4 TP8 + reality_mix.
#
# Boots our standard TP8 server (custom all-reduce ON — the unmitigated path)
# and drives the reality_mix workload. A server-wide decode wedge trips the
# harness's fatal global-stall detector, which exits non-zero (2): this script
# propagates that, so it doubles as the AR-fusion hang regression gate.
#
# Self-contained and config-owning: the workflow just calls it. Tunables come
# from the environment (sensible defaults so it also runs by hand):
#   OUT_DIR          output dir for events.jsonl / summary.txt (default ./stress-out)
#   DURATION_S       reality_mix load duration in seconds       (default 900 = 15 min)
#   MAX_CONCURRENCY  peak in-flight requests (sawtooth)         (default 40)
#
# NB: no `set -e` — we must still summarize and propagate the harness exit code
# after a fatal abort.
set -uo pipefail

# Run from the repo root so `python -m test.stress` resolves regardless of cwd.
cd "$(dirname "$0")/../../.."

OUT_DIR="${OUT_DIR:-$PWD/stress-out}"
DURATION_S="${DURATION_S:-900}"
MAX_CONCURRENCY="${MAX_CONCURRENCY:-40}"
mkdir -p "$OUT_DIR"

SERVE="tokenspeed serve \
  --model nvidia/Kimi-K2.5-NVFP4 \
  --served-model-name nvidia/Kimi-K2.5-NVFP4 \
  --trust-remote-code \
  --host 127.0.0.1 --port 8000 \
  --attn-tp-size 8 --moe-tp-size 8 \
  --max-model-len 262144 --max-num-seqs 40 \
  --gpu-memory-utilization 0.95 --disable-cuda-graph-padding \
  --attention-backend tokenspeed_mla --moe-backend flashinfer_trtllm \
  --quantization nvfp4 --kv-cache-dtype fp8 \
  --reasoning-parser kimi_k25 --tool-call-parser kimik2 \
  --grammar-backend xgrammar \
  --speculative-algorithm EAGLE3 \
  --speculative-draft-model-path lightseekorg/kimi-k2.5-eagle3-mla \
  --speculative-num-steps 3 --speculative-eagle-topk 1 \
  --speculative-num-draft-tokens 4 \
  --drafter-attention-backend tokenspeed_mla \
  --speculative-draft-model-quantization unquant \
  --weight-loader-prefetch-checkpoints \
  --enable-cache-report --enable-metrics"

python3 -m test.stress run \
  --launch-cmd "$SERVE" \
  --launch-timeout 2400 \
  --base-url http://127.0.0.1:8000 \
  --model nvidia/Kimi-K2.5-NVFP4 \
  --workload reality_mix \
  --workload-arg grammar_fraction=0.6 \
  --workload-arg cancel_fraction=0.15 \
  --workload-arg cached_fraction=0.5 \
  --workload-arg max_tokens_cap=65536 \
  --workload-arg very_long_weight=20 \
  --arrival sawtooth --min-concurrency 1 \
  --max-concurrency "$MAX_CONCURRENCY" --triangle-period 180 \
  --duration "$DURATION_S" --request-timeout 2400 \
  --stall-timeout 20 --global-stall-timeout 20 \
  --metrics-interval 10 --accept-len-min 1.1 \
  --out "$OUT_DIR"
rc=$?

# Regenerate a clean summary for the artifact bundle and the workflow run page.
if [ -f "$OUT_DIR/events.jsonl" ]; then
  python3 -m test.stress summarize --events "$OUT_DIR/events.jsonl" \
    | tee "$OUT_DIR/summary.txt" || true
fi

exit "$rc"
