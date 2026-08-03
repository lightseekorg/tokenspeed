#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
artifact_root="${TS_CI_ARTIFACT_ROOT:-/mnt/nfs01/${USER}/tokenspeed-slurm}"
cache_dir="${TS_CI_CACHE_DIR:-/mnt/lustre01/${USER}/tokenspeed-cache}"
container_image="${TS_CI_CONTAINER_IMAGE:-ghcr.io/lightseekorg/tokenspeed-runner:cu130-torch-2.11.0-flashinfer-0.6.16@sha256:9ea554656cb57d3b9f6dd548b9621debcd7a3e0e5229309a24184589c97cb1d4}"

if [ "$#" -gt 0 ] && [[ "$1" != -* ]]; then
    config="$1"
    shift
    set -- --config "${config}" "$@"
fi

exec python3 "${repo_root}/test/ci_system/slurm_submit.py" \
    --repo-root "${repo_root}" \
    --artifact-root "${artifact_root}" \
    --cache-dir "${cache_dir}" \
    --container-image "${container_image}" \
    "$@"
