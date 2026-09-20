#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
artifact_root="${TS_CI_ARTIFACT_ROOT:-/mnt/nfs01/${USER}/tokenspeed-slurm}"
cache_dir="${TS_CI_CACHE_DIR:-/mnt/lustre01/${USER}/tokenspeed-cache}"
container_image="${TS_CI_CONTAINER_IMAGE:-ghcr.io/lightseekorg/tokenspeed-runner:cu130-torch-2.14.0-flashinfer-0.6.18@sha256:f8169f28800619c6e798440128ba00a944512d4256bc14f7174dcbae0aca2c44}"

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
