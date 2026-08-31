#!/usr/bin/env bash
set -euo pipefail

DEEPSWE_REPOSITORY=https://github.com/datacurve-ai/deep-swe.git
DEEPSWE_REVISION=0b9fabbb63b9104d678fe965e1632f2dd9eaa2ea
KIMI_CODE_VERSION=0.23.6

task_count=${DEEPSWE_TASK_COUNT:-10}
sample_seed=${DEEPSWE_SAMPLE_SEED:-0}
concurrency=${DEEPSWE_CONCURRENCY:-4}
minimum_score=${DEEPSWE_MINIMUM_SCORE:-0.0}

[[ "$task_count" =~ ^[0-9]+$ ]] && (( task_count >= 1 && task_count <= 113 )) || {
  echo "DEEPSWE_TASK_COUNT must be between 1 and 113" >&2
  exit 2
}
[[ "$sample_seed" =~ ^-?[0-9]+$ ]] || {
  echo "DEEPSWE_SAMPLE_SEED must be an integer" >&2
  exit 2
}
[[ "$concurrency" =~ ^[0-9]+$ ]] && (( concurrency >= 1 && concurrency <= 16 )) || {
  echo "DEEPSWE_CONCURRENCY must be between 1 and 16" >&2
  exit 2
}
[[ "${POD_IP:-}" =~ ^[0-9]+\.[0-9]+\.[0-9]+\.[0-9]+$ ]] || {
  echo "POD_IP must contain the runner Pod IPv4 address" >&2
  exit 2
}

docker info >/dev/null
docker compose version
curl -fsS --max-time 10 "http://${POD_IP}/readiness" >/dev/null

artifact_dir=${PWD}/.ci-artifacts/deepswe
jobs_dir=${artifact_dir}/jobs
dataset_root=${RUNNER_TEMP:-/tmp}/deep-swe-${GITHUB_RUN_ID:-local}-${GITHUB_RUN_ATTEMPT:-1}
job_name=b300-kimi-k3-${GITHUB_RUN_ID:-local}-${GITHUB_RUN_ATTEMPT:-1}
mkdir -p "$artifact_dir" "$jobs_dir"

git clone --filter=blob:none --no-checkout "$DEEPSWE_REPOSITORY" "$dataset_root"
git -C "$dataset_root" checkout --detach "$DEEPSWE_REVISION"

export PYTHONPATH="${PWD}/test/ci/deepswe${PYTHONPATH:+:${PYTHONPATH}}"
pier run \
  --path "$dataset_root/tasks" \
  --agent-import-path kimi_code_agent:KimiCodeAgent \
  --model kimi/kimi-k3 \
  --agent-kwarg "version=${KIMI_CODE_VERSION}" \
  --agent-env "KIMI_MODEL_BASE_URL=http://${POD_IP}/v1" \
  --agent-env KIMI_MODEL_API_KEY=EMPTY_TOKEN \
  --agent-env KIMI_MODEL_PROVIDER_TYPE=kimi \
  --agent-env KIMI_MODEL_MAX_CONTEXT_SIZE=80000 \
  --agent-env KIMI_MODEL_CAPABILITIES=thinking,always_thinking,tool_use \
  --agent-env KIMI_MODEL_TEMPERATURE=1.0 \
  --agent-env KIMI_MODEL_TOP_P=0.95 \
  --agent-env KIMI_MODEL_THINKING_EFFORT=max \
  --agent-env KIMI_MODEL_THINKING_KEEP=all \
  --env docker \
  --delete \
  --n-concurrent "$concurrency" \
  --n-tasks "$task_count" \
  --sample-seed "$sample_seed" \
  --jobs-dir "$jobs_dir" \
  --job-name "$job_name" \
  --yes

result=${jobs_dir}/${job_name}/result.json
summary=${artifact_dir}/summary.md
cp "$result" "${artifact_dir}/result.json"
python3 test/ci/deepswe/summarize.py \
  "$result" \
  --minimum-score "$minimum_score" \
  --output "$summary"
