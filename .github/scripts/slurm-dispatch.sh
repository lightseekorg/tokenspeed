#!/usr/bin/env bash
set -euo pipefail
COMMIT=${COMMIT:-}
EXACT_RUNNER=${EXACT_RUNNER:-}
if [ -n "$COMMIT" ] && [ -n "$PR" ]; then
  echo "Set only one of pr or commit" >&2
  exit 2
fi
source_repo=$PWD
if [ -n "$COMMIT" ]; then
  [[ "$COMMIT" =~ ^[0-9a-f]{40}$ ]] || {
    echo "commit must be a full lowercase hexadecimal SHA" >&2
    exit 2
  }
  git fetch --no-tags origin "$COMMIT"
  source_repo="$RUNNER_TEMP/slurm-source"
  git worktree add --detach "$source_repo" "$COMMIT"
  [[ "$(git -C "$source_repo" rev-parse HEAD)" == "$COMMIT" ]] || exit 2
  trap 'git worktree remove --force "$source_repo"' EXIT
fi
if [ -n "$EXACT_RUNNER" ] && [ "$YAML_SELECTION" = off ]; then
  echo "An exact runner requires one CI YAML" >&2
  exit 2
fi
if [ -n "$CONTAINER_IMAGE" ]; then
  [[ "$CONTAINER_IMAGE" =~ ^ghcr\.io/lightseekorg/tokenspeed-runner:[A-Za-z0-9_][A-Za-z0-9._-]{0,127}@sha256:[0-9a-f]{64}$ ]] || {
    echo "container_image must be an immutable ghcr.io/lightseekorg/tokenspeed-runner image" >&2
    exit 2
  }
  export TS_CI_CONTAINER_IMAGE="$CONTAINER_IMAGE"
fi
args=(--wait --report-dir "$RUNNER_TEMP/slurm-report")
[ -z "$COMMIT" ] || args+=(--repo-root "$source_repo")
[ -z "$PR" ] || args+=(--pr "$PR")

case "$CLUSTER" in
  gb200)
    runner_prefixes=(b200- gb200- slurm-gb200-)
    ;;
  gb300)
    runner_prefixes=(gb300- slurm-gb300-)
    coordinator_user=${USER:-$(id -un)}
    export TS_CI_ARTIFACT_ROOT="/data/home/$coordinator_user/tokenspeed-slurm"
    export TS_CI_CACHE_DIR="/data/home/$coordinator_user/tokenspeed-cache"
    ;;
  *)
    echo "Unsupported Slurm cluster: $CLUSTER" >&2
    exit 2
    ;;
esac

runner_is_supported() {
  local runner=$1 prefix
  for prefix in "${runner_prefixes[@]}"; do
    [[ "$runner" == "$prefix"* ]] && return 0
  done
  return 1
}

gb300_runner_for() {
  local runner=$1
  case "$runner" in
    b200-*) echo "gb300-${runner#b200-}" ;;
    gb200-*) echo "gb300-${runner#gb200-}" ;;
    slurm-b200-*) echo "slurm-gb300-${runner#slurm-b200-}" ;;
    slurm-gb200-*) echo "slurm-gb300-${runner#slurm-gb200-}" ;;
    gb300-*|slurm-gb300-*) echo "$runner" ;;
    *) return 1 ;;
  esac
}

if [ "$YAML_SELECTION" != off ]; then
  args+=(--config "$YAML_SELECTION")
  yaml_metadata_output=$(
    python3 - "$YAML_SELECTION" "$CLUSTER" "$PR" "$source_repo" <<'PY'
import sys
from pathlib import Path

sys.path.insert(0, "test/ci_system")
from pipeline import normalize_task
from slurm_submit import pr_worktree

repo = Path(sys.argv[4])
selection = Path(sys.argv[1])
pr = sys.argv[3]
if pr:
    with pr_worktree(repo, pr) as checkout:
        task = normalize_task(checkout / selection, checkout)
else:
    task = normalize_task(repo / selection, repo)
prefixes = {
    "gb200": ("b200-", "gb200-", "slurm-gb200-"),
    "gb300": (
        "b200-",
        "gb200-",
        "slurm-b200-",
        "slurm-gb200-",
        "gb300-",
        "slurm-gb300-",
    ),
}[sys.argv[2]]
print(f"type:{task['type']}")
for label in task["runner"]["labels"]:
    if label.startswith(prefixes):
        print(label)
PY
  )
  mapfile -t yaml_metadata <<< "$yaml_metadata_output"
  [[ "${yaml_metadata[0]-}" == type:* ]] || {
    echo "Failed to read task metadata from $YAML_SELECTION" >&2
    exit 2
  }
  yaml_task_type="${yaml_metadata[0]#type:}"
  yaml_runners=("${yaml_metadata[@]:1}")
  ((${#yaml_runners[@]})) || {
    echo "No supported Slurm runners declared by $YAML_SELECTION" >&2
    exit 2
  }
  args+=(--type "$yaml_task_type")
  if [ -n "$EXACT_RUNNER" ]; then
    exact_found=false
    for yaml_runner in "${yaml_runners[@]}"; do
      [ "$yaml_runner" != "$EXACT_RUNNER" ] || exact_found=true
    done
    [ "$exact_found" = true ] || {
      echo "Exact runner is not declared by the selected YAML" >&2
      exit 2
    }
    yaml_runners=("$EXACT_RUNNER")
    RUNNERS=$EXACT_RUNNER
  fi
  if [ "$CLUSTER" = gb300 ]; then
    selected_runners=()
    if [[ "${RUNNERS//[[:space:]]/}" == "b200-4gpu,gb200-4gpu" || \
          "${RUNNERS//[[:space:]]/}" == "b200-4gpu,gb200-4gpu,slurm-gb200-4gpu" ]]; then
      selected_runners=("${yaml_runners[@]}")
    else
      IFS=',' read -ra values <<< "$RUNNERS"
      for value in "${values[@]}"; do
        value="${value//[[:space:]]/}"
        [ -z "$value" ] && continue
        matches=0
        for yaml_runner in "${yaml_runners[@]}"; do
          effective_runner=$(gb300_runner_for "$yaml_runner") || continue
          if [ "$value" = "$yaml_runner" ] || [ "$value" = "$effective_runner" ]; then
            for selected_runner in "${selected_runners[@]}"; do
              [ "$selected_runner" != "$yaml_runner" ] || {
                echo "Runner $value selects $yaml_runner more than once" >&2
                exit 2
              }
            done
            selected_runners+=("$yaml_runner")
            ((matches += 1))
          fi
        done
        ((matches == 1)) || {
          echo "Runner $value does not identify exactly one runner declared by $YAML_SELECTION" >&2
          exit 2
        }
      done
    fi
    for yaml_runner in "${selected_runners[@]}"; do
      effective_runner=$(gb300_runner_for "$yaml_runner") || {
        echo "Runner $yaml_runner cannot run on the GB300 cluster" >&2
        exit 2
      }
      args+=(--runner-alias "$yaml_runner=$effective_runner")
    done
  else
    for value in "${yaml_runners[@]}"; do
      args+=(--runner "$value")
    done
  fi
else
  args+=(--all)
  IFS=',' read -ra values <<< "$RUNNERS"
  for value in "${values[@]}"; do
    value="${value//[[:space:]]/}"
    [ -z "$value" ] && continue
    if [ "$CLUSTER" = gb300 ]; then
      effective_runner=$(gb300_runner_for "$value") || {
        echo "Runner $value cannot run on the GB300 cluster" >&2
        exit 2
      }
      args+=(--runner-alias "$value=$effective_runner")
    else
      runner_is_supported "$value" || {
        echo "Runner $value is not supported by the $CLUSTER cluster" >&2
        exit 2
      }
      args+=(--runner "$value")
    fi
  done

  IFS=',' read -ra values <<< "$TASK_TYPES"
  for value in "${values[@]}"; do
    value="${value//[[:space:]]/}"
    [ -z "$value" ] || args+=(--type "$value")
  done

  IFS=',' read -ra values <<< "$MATCH"
  for value in "${values[@]}"; do
    value="${value#"${value%%[![:space:]]*}"}"
    value="${value%"${value##*[![:space:]]}"}"
    [ -z "$value" ] || args+=(--match "$value")
  done

  [ "$INCLUDE_MMLU" = true ] || args+=(--exclude-match mmlu)
  [ "$TRIGGER" = all ] || args+=(--trigger "$TRIGGER")
fi
if [ -n "$COMMIT" ] && [ -n "$EXACT_RUNNER" ]; then
  mkdir -p "$RUNNER_TEMP/slurm-report"
  python3 - "$source_repo" "$YAML_SELECTION" "$EXACT_RUNNER" "$CLUSTER" <<'PY'
import json
import os
import subprocess
import sys
from pathlib import Path

commit = subprocess.run(["git", "rev-parse", "HEAD"], cwd=sys.argv[1], check=True, capture_output=True, text=True).stdout.strip()
Path(os.environ["RUNNER_TEMP"], "slurm-report/source.json").write_text(json.dumps({"commit": commit, "tasks": [{"config": sys.argv[2], "runner": sys.argv[3]}], "cluster": sys.argv[4]}))
PY
fi
bash test/ci/run_slurm.sh "${args[@]}"
