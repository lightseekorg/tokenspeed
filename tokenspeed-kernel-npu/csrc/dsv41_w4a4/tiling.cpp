// Copyright (c) 2026 LightSeek Foundation
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in
// all copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
// SOFTWARE.

#include "tiling/matmul/matmul_tiling.h"
#include <cstring>
using namespace matmul_tiling;
extern "C" int make_tiling(void* out, int k, int n) {
    constexpr int bm = 64, bn = 512, bk = 256;
    PlatformInfo p;
    p.socVersion = platform_ascendc::SocVersion::ASCEND910B;
    p.l1Size = 512 * 1024;
    p.l0CSize = 128 * 1024;
    p.ubSize = 192 * 1024;
    p.l0ASize = 64 * 1024;
    p.l0BSize = 64 * 1024;
    MatmulApiTiling mm(p);
    mm.SetAType(TPosition::GM, CubeFormat::ND, DataType::DT_INT4, false);
    mm.SetBType(TPosition::GM, CubeFormat::NZ, DataType::DT_INT4, true);
    mm.SetCType(TPosition::GM, CubeFormat::ND, DataType::DT_FLOAT16);
    mm.SetDequantType(DequantType::TENSOR);
    mm.SetBias(false);
    mm.SetOrgShape(bm, n, k);
    mm.SetShape(bm, bn, k);
    mm.SetFixSplit(bm, bn, bk);
    AscendC::tiling::TCubeTiling t;
    if (mm.GetTiling(t) != 0) return -1;
    memcpy(out, &t, sizeof(t));
    return sizeof(t);
}
