#!/bin/bash

configure_package_cache() {
    local cache_root="${CI_CACHE_ROOT:-}"
    case "${CI_RUNNER_LABEL:-}" in
        b200v2-*)
            if [ -z "${cache_root}" ] && [ -n "${FLASHINFER_CACHE_DIR:-}" ]; then
                cache_root="$(dirname "${FLASHINFER_CACHE_DIR}")"
            fi
            cache_root="${cache_root:-/raid/cache}"
            ;;
        slurm-*) cache_root="${cache_root:-${XDG_CACHE_HOME:-/home/runner/.cache}}" ;;
        *) return 0 ;;
    esac

    export PIP_CACHE_DIR="${PIP_CACHE_DIR:-${cache_root}/pip}"
    export CI_WHEEL_CACHE_DIR="${CI_WHEEL_CACHE_DIR:-${cache_root}/wheelhouse}"
    mkdir -p "${PIP_CACHE_DIR}" "${CI_WHEEL_CACHE_DIR}"
    echo "Package cache: pip=${PIP_CACHE_DIR}, wheels=${CI_WHEEL_CACHE_DIR}"
}

cache_remote_wheel() {
    local wheel_url="$1"
    local expected_sha256="${2:-}"
    if [ -z "${CI_WHEEL_CACHE_DIR:-}" ]; then
        printf '%s\n' "${wheel_url}"
        return 0
    fi

    local filename="${wheel_url%%\?*}"
    filename="${filename##*/}"
    local cache_path="${CI_WHEEL_CACHE_DIR}/${filename}"

    (
        flock 9
        local cached_sha256=""
        if [ -s "${cache_path}" ] && [ -n "${expected_sha256}" ]; then
            cached_sha256="$(sha256sum "${cache_path}")"
            cached_sha256="${cached_sha256%% *}"
        fi
        if [ ! -s "${cache_path}" ] || { [ -n "${expected_sha256}" ] && [ "${cached_sha256}" != "${expected_sha256}" ]; }; then
            local tmp_path="${cache_path}.tmp.$$"
            trap 'rm -f "${tmp_path}"' EXIT
            rm -f "${cache_path}"
            echo "Downloading ${wheel_url} to persistent cache" >&2
            curl --fail --location --retry 5 --retry-all-errors \
                --connect-timeout 30 --output "${tmp_path}" "${wheel_url}"
            if [ -n "${expected_sha256}" ]; then
                local downloaded_sha256
                downloaded_sha256="$(sha256sum "${tmp_path}")"
                downloaded_sha256="${downloaded_sha256%% *}"
                if [ "${downloaded_sha256}" != "${expected_sha256}" ]; then
                    echo "SHA256 mismatch for ${wheel_url}: expected ${expected_sha256}, got ${downloaded_sha256}" >&2
                    return 1
                fi
            fi
            mv "${tmp_path}" "${cache_path}"
            trap - EXIT
        else
            echo "Using cached wheel ${cache_path}" >&2
        fi
    ) 9>"${cache_path}.lock"

    printf '%s\n' "${cache_path}"
}
