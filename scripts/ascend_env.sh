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

# Source from the repository root after choosing the physical chips explicitly.
if [[ ! "${ASCEND_RT_VISIBLE_DEVICES:-}" =~ ^[0-9]+(,[0-9]+)?$ ]]; then
    echo 'Set ASCEND_RT_VISIBLE_DEVICES to one or two physical device IDs.' >&2
    return 1
fi
if [[ "${ASCEND_RT_VISIBLE_DEVICES}" == *,* ]]; then
    IFS=, read -r _ascend_first _ascend_second <<< "${ASCEND_RT_VISIBLE_DEVICES}"
    if [[ "${_ascend_first}" == "${_ascend_second}" ]]; then
        echo 'Physical device IDs must be distinct.' >&2
        return 1
    fi
    unset _ascend_first _ascend_second
fi
source .venv/bin/activate
# CANN's script expands optional variables, including under a caller's `set -u`.
_ascend_env_nounset=false
[[ "$-" == *u* ]] && _ascend_env_nounset=true
set +u
source /usr/local/Ascend/cann-9.0.0/set_env.sh
if $_ascend_env_nounset; then set -u; fi
unset _ascend_env_nounset
export PYTHONPATH="${PWD}/python:${PWD}/tokenspeed-kernel/python:${PWD}/tokenspeed-kernel-npu/python:${PYTHONPATH:-}"
export TOKENIZERS_PARALLELISM=false
