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

# tokenspeed-kernel/python's setup.py selects its dependency set from this
# variable; on an Ascend host no CUDA/ROCm is detectable, so default to npu.
export TOKENSPEED_KERNEL_BACKEND="${TOKENSPEED_KERNEL_BACKEND:-npu}"

TOKENSPEED_CANN_ROOT="${TOKENSPEED_CANN_ROOT:-/usr/local/Ascend/cann-9.0.0}"
TOKENSPEED_TVM_FFI_VERSION="${TOKENSPEED_TVM_FFI_VERSION:-0.1.13.post3}"
TOKENSPEED_TRITON_ASCEND_VERSION="${TOKENSPEED_TRITON_ASCEND_VERSION:-3.2.1}"
TOKENSPEED_TRANSFORMERS_VERSION="${TOKENSPEED_TRANSFORMERS_VERSION:-5.12.0}"
TOKENSPEED_ASCEND_PYPI="${TOKENSPEED_ASCEND_PYPI:-https://mirrors.huaweicloud.com/ascend/repos/pypi}"
TOKENSPEED_PIP_TIMEOUT="${TOKENSPEED_PIP_TIMEOUT:-600}"
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
TOKENSPEED_REPO_ROOT="$(cd -- "${SCRIPT_DIR}/../.." && pwd)"

if [[ ! -r "${TOKENSPEED_CANN_ROOT}/set_env.sh" ]]; then
    echo "CANN environment script not found: ${TOKENSPEED_CANN_ROOT}/set_env.sh" >&2
    exit 1
fi

# shellcheck disable=SC1091
set +u
source "${TOKENSPEED_CANN_ROOT}/set_env.sh"
set -u

python -m pip install \
    --timeout "${TOKENSPEED_PIP_TIMEOUT}" \
    "apache-tvm-ffi==${TOKENSPEED_TVM_FFI_VERSION}" \
    "transformers==${TOKENSPEED_TRANSFORMERS_VERSION}" \
    "triton-ascend==${TOKENSPEED_TRITON_ASCEND_VERSION}" \
    --extra-index-url "${TOKENSPEED_ASCEND_PYPI}"

python -m pip install --no-deps -e "${TOKENSPEED_REPO_ROOT}/tokenspeed-kernel-npu"

# Install the remaining in-repo runtime packages editable so runtime tests run
# against this checkout. --no-deps keeps the pinned PyPI versions above intact.
for _pkg in tokenspeed-scheduler tokenspeed-engine tokenspeed-kernel/python python; do
    _pkg_dir="${TOKENSPEED_REPO_ROOT}/${_pkg}"
    if [[ ! -d "${_pkg_dir}" ]]; then
        echo "Missing in-repo package directory: ${_pkg_dir}" >&2
        exit 1
    fi
    # tokenspeed-kernel's setup.py shells out to `python -m pip` to install its
    # npu backend requirements. pip's isolated build env strips the venv
    # site-packages from that subprocess, so the nested pip cannot import
    # itself. Its build requirements (setuptools/wheel/apache-tvm-ffi) are
    # already installed above, so build it in place instead.
    _build_iso_flag=()
    if [[ "${_pkg}" == "tokenspeed-kernel/python" ]]; then
        _build_iso_flag=(--no-build-isolation)
    fi
    python -m pip install --no-deps "${_build_iso_flag[@]}" -e "${_pkg_dir}"
done

# tokenspeed-smg / tokenspeed-mooncake / tokenspeed-smg-grpc-proto /
# tokenspeed-smg-grpc-servicer / tokenspeed-spdlog are pinned in
# python/pyproject.toml (and the scheduler/engine build requirements) and
# resolve from the public PyPI. The editable ``--no-deps`` install above skips
# dependencies, so install them explicitly; the smg/mooncake pins come from
# python/pyproject.toml, tokenspeed-spdlog from the scheduler/engine build.
python -m pip install \
    --timeout "${TOKENSPEED_PIP_TIMEOUT}" \
    "tokenspeed-smg" \
    "tokenspeed-mooncake" \
    "tokenspeed-smg-grpc-proto" \
    "tokenspeed-smg-grpc-servicer" \
    "tokenspeed-spdlog" \
    --extra-index-url "${TOKENSPEED_ASCEND_PYPI}"

PYTHONPATH="${TOKENSPEED_REPO_ROOT}/python:${TOKENSPEED_REPO_ROOT}/tokenspeed-kernel/python:${TOKENSPEED_REPO_ROOT}/tokenspeed-kernel-npu/python:${PYTHONPATH:-}" \
    python "${SCRIPT_DIR}/verify_triton_ascend.py"
