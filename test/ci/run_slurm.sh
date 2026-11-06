#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
# Cron may start runners without USER in their environment.
ci_user="${USER:-$(id -un)}"
artifact_root="${TS_CI_ARTIFACT_ROOT:-/mnt/nfs01/${ci_user}/tokenspeed-slurm}"
cache_dir="${TS_CI_CACHE_DIR:-/mnt/lustre01/${ci_user}/tokenspeed-cache}"
container_image="${TS_CI_CONTAINER_IMAGE:-ghcr.io/lightseekorg/tokenspeed-runner:cu130-torch-2.14.0-flashinfer-0.7.3@sha256:e359ef0b3d761357f52bd3675fcc831896288448ee06b11807d09f237ec8b3bd}"

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
