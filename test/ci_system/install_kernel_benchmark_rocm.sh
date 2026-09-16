#!/bin/bash
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

TORCH_VERSION=${TORCH_VERSION:-2.13.0}
TORCH_INDEX_URL=${TORCH_INDEX_URL:-https://download.pytorch.org/whl/rocm7.2}

pip_install_with_retry() {
    local attempt
    local delay=10
    for attempt in 1 2 3 4 5; do
        if "$@"; then
            return 0
        fi
        if [ "${attempt}" -eq 5 ]; then
            echo "pip install failed after ${attempt} attempts: $*" >&2
            return 1
        fi
        echo "pip install attempt ${attempt} failed; retrying in ${delay}s" >&2
        sleep "${delay}"
        delay=$((delay * 2))
    done
}

if ! python3 -c 'import torch; assert torch.cuda.is_available(); assert torch.version.hip'; then
    pip_install_with_retry \
        python3 -m pip install --force-reinstall "torch==${TORCH_VERSION}" \
        --index-url "${TORCH_INDEX_URL}"
fi

python3 -c 'import torch; assert torch.cuda.is_available(); assert torch.version.hip; print(torch.__version__, torch.cuda.get_device_name(0))'
python3 -m venv --help >/dev/null
