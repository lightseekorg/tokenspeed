#!/bin/bash

# b200v2 runners mount persistent, node-local storage under /raid/cache.
# Keep this opt-in so clusters with different storage layouts are unchanged.
configure_b200v2_package_cache() {
    if [[ "${CI_RUNNER_LABEL:-}" != b200v2-* ]]; then
        return 0
    fi

    local cache_root="${CI_CACHE_ROOT:-}"
    if [ -z "${cache_root}" ] && [ -n "${FLASHINFER_CACHE_DIR:-}" ]; then
        cache_root="$(dirname "${FLASHINFER_CACHE_DIR}")"
    fi
    cache_root="${cache_root:-/raid/cache}"

    export PIP_CACHE_DIR="${PIP_CACHE_DIR:-${cache_root}/pip}"
    export CI_WHEEL_CACHE_DIR="${CI_WHEEL_CACHE_DIR:-${cache_root}/wheelhouse}"
    mkdir -p "${PIP_CACHE_DIR}" "${CI_WHEEL_CACHE_DIR}"
    echo "b200v2 package cache: pip=${PIP_CACHE_DIR}, wheels=${CI_WHEEL_CACHE_DIR}"
}

cache_remote_wheel() {
    local wheel_url="$1"
    if [ -z "${CI_WHEEL_CACHE_DIR:-}" ]; then
        printf '%s\n' "${wheel_url}"
        return 0
    fi

    local filename="${wheel_url%%\?*}"
    filename="${filename##*/}"
    local cache_path="${CI_WHEEL_CACHE_DIR}/${filename}"

    (
        flock 9
        if [ ! -s "${cache_path}" ]; then
            local tmp_path="${cache_path}.tmp.$$"
            trap 'rm -f "${tmp_path}"' EXIT
            echo "Downloading ${wheel_url} to persistent b200v2 cache" >&2
            curl --fail --location --retry 5 --retry-all-errors \
                --connect-timeout 30 --output "${tmp_path}" "${wheel_url}"
            mv "${tmp_path}" "${cache_path}"
            trap - EXIT
        else
            echo "Using cached wheel ${cache_path}" >&2
        fi
    ) 9>"${cache_path}.lock"

    printf '%s\n' "${cache_path}"
}
