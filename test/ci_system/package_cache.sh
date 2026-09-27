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
    export CI_CCACHE_DIR="${CI_CCACHE_DIR:-${cache_root}/ccache}"
    mkdir -p "${PIP_CACHE_DIR}" "${CI_WHEEL_CACHE_DIR}"
    echo "Package cache: pip=${PIP_CACHE_DIR}, wheels=${CI_WHEEL_CACHE_DIR}, ccache=${CI_CCACHE_DIR}"
}

configure_nvcc_cache() {
    if [ -z "${CI_CCACHE_DIR:-}" ] || ! command -v ccache >/dev/null 2>&1; then
        return 0
    fi

    export TOKENSPEED_KERNEL_NVCC_LAUNCHER="${TOKENSPEED_KERNEL_NVCC_LAUNCHER:-ccache}"
    export CCACHE_DIR="${CCACHE_DIR:-${CI_CCACHE_DIR}}"
    export CCACHE_BASEDIR="${CCACHE_BASEDIR:-${WORKSPACE:?WORKSPACE is required}}"
    # Checkout paths are unique per matrix job. No debug flags are used for
    # these objects, so the working directory must not split identical keys.
    export CCACHE_NOHASHDIR="${CCACHE_NOHASHDIR:-1}"
    export CCACHE_COMPILERTYPE="${CCACHE_COMPILERTYPE:-nvcc}"
    export CCACHE_COMPILERCHECK="${CCACHE_COMPILERCHECK:-%compiler% --version; g++ --version}"
    # Fresh checkouts give headers new timestamps even though their content is
    # immutable during the build. Keep content hashing while allowing caching.
    export CCACHE_SLOPPINESS="${CCACHE_SLOPPINESS:-include_file_ctime,include_file_mtime}"
    export CCACHE_MAXSIZE="${CCACHE_MAXSIZE:-100G}"
    export CCACHE_UMASK="${CCACHE_UMASK:-002}"
    export CCACHE_TEMPDIR="${CCACHE_TEMPDIR:-${WORKSPACE}/.ccache-tmp}"
    export CCACHE_STATSLOG="${CCACHE_STATSLOG:-${WORKSPACE}/.ci-artifacts/ccache-stats.log}"
    if [ "${TOKENSPEED_CI_FORK_PR:-false}" = "true" ]; then
        export CCACHE_READONLY=1
    fi

    local stats_dir
    stats_dir=$(dirname "${CCACHE_STATSLOG}")
    if ! mkdir -p "${CCACHE_DIR}" "${CCACHE_TEMPDIR}" "${stats_dir}" \
        || [ ! -w "${CCACHE_DIR}" ] \
        || [ ! -w "${CCACHE_TEMPDIR}" ] \
        || [ ! -w "${stats_dir}" ]; then
        echo "NVCC cache directory is unavailable; continuing without ccache" >&2
        unset CI_CCACHE_DIR TOKENSPEED_KERNEL_NVCC_LAUNCHER
        return 0
    fi
    echo "NVCC cache: dir=${CCACHE_DIR}, base=${CCACHE_BASEDIR}, read_only=${CCACHE_READONLY:-0}"
}

show_nvcc_cache_stats() {
    local phase="$1"
    if [ -z "${CI_CCACHE_DIR:-}" ] || ! command -v ccache >/dev/null 2>&1; then
        return 0
    fi

    echo "=== NVCC cache stats (${phase}) ==="
    ccache --show-stats || true
    if [ "${phase}" = "after" ] && [ -s "${CCACHE_STATSLOG:-}" ]; then
        echo "=== NVCC cache stats for this build ==="
        ccache --show-log-stats || true
    fi
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
