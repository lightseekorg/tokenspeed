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

set -eo pipefail
: "${ASCEND_HOME_PATH:?Source the CANN environment first}"
script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
package_root=$(cd -- "$script_dir/../.." && pwd)
output_dir="$package_root/python/tokenspeed_kernel_npu/lib"
compiler="$ASCEND_HOME_PATH/$(uname -m)-linux/ccec_compiler/bin/bisheng"
if [[ ! -x "$compiler" ]]; then
    compiler="$ASCEND_HOME_PATH/bin/bisheng"
fi
mkdir -p "$output_dir"
build_dir=$(mktemp -d)
trap 'rm -rf "$build_dir"' EXIT
"${CXX:-g++}" -O2 -std=c++17 -fPIC -c "$script_dir/tiling.cpp" \
    -o "$build_dir/tiling.o" -I"$ASCEND_HOME_PATH/include" \
    -I"$ASCEND_HOME_PATH/include/ascendc/highlevel_api"
"$compiler" -xasc --npu-arch=dav-2201 -O2 -std=c++17 -fPIC -shared \
    "$script_dir/grouped_w4a4.asc" -x none "$build_dir/tiling.o" \
    -o "$output_dir/libdsv41_w4a4.so" -I"$ASCEND_HOME_PATH/include" \
    -I"$ASCEND_HOME_PATH/include/ascendc/highlevel_api" \
    -L"$ASCEND_HOME_PATH/lib64" -ltiling_api -lplatform -lregister -lgraph \
    -lruntime -lascendcl -ldl
