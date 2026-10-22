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
# Source CANN before invoking this script. Build only this optional kernel.
: "${ASCEND_HOME_PATH:?Source the CANN environment first}"
script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
package_root=$(cd -- "$script_dir/../.." && pwd)
output_dir="$package_root/python/tokenspeed_kernel_npu/lib"
compiler="$ASCEND_HOME_PATH/$(uname -m)-linux/ccec_compiler/bin/bisheng"
if [[ ! -x "$compiler" ]]; then
    compiler="$ASCEND_HOME_PATH/bin/bisheng"
fi
mkdir -p "$output_dir"
"$compiler" -xasc --npu-arch=dav-2201 -O2 -std=c++17 -fPIC -shared \
    "$script_dir/token_shards.asc" -o "$output_dir/libtoken_shards.so" \
    -I"$ASCEND_HOME_PATH/include" -L"$ASCEND_HOME_PATH/lib64" -lruntime -lascendcl
