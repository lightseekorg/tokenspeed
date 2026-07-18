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

"""Gluon attention kernels."""

from __future__ import annotations

from tokenspeed_kernel_amd.ops.attention.gluon.dsa_gfx950 import (  # noqa: F401
    gluon_dsa_decode_gfx950,
    gluon_dsa_prefill_gfx950,
)
from tokenspeed_kernel_amd.ops.attention.gluon.dsa_topk_gfx950 import (  # noqa: F401
    gluon_dsa_decode_topk_fp8_gfx950,
    gluon_dsa_prefill_topk_fp8_gfx950,
)
from tokenspeed_kernel_amd.ops.attention.gluon.mha_decode_gfx950 import (  # noqa: F401
    gluon_mha_decode_gfx950,
)
from tokenspeed_kernel_amd.ops.attention.gluon.mha_extend_gfx950 import (  # noqa: F401
    gluon_mha_extend_gfx950,
)
from tokenspeed_kernel_amd.ops.attention.gluon.mha_prefill_gfx950 import (  # noqa: F401
    gluon_mha_prefill_gfx950,
)
from tokenspeed_kernel_amd.ops.attention.gluon.mha_prefill_gfx1250 import (  # noqa: F401
    gluon_mha_prefill_gfx1250,
)
from tokenspeed_kernel_amd.ops.attention.gluon.mla_decode_bf16_gfx950 import (  # noqa: F401
    gluon_mla_decode_bf16_gfx950,
)
from tokenspeed_kernel_amd.ops.attention.gluon.mla_prefill_bf16_gfx950 import (  # noqa: F401
    gluon_mla_prefill_bf16_gfx950,
)
from tokenspeed_kernel_amd.ops.attention.gluon.rel_mha_decode_gfx950 import (  # noqa: F401
    gluon_rel_mha_decode_gfx950,
)
from tokenspeed_kernel_amd.ops.attention.gluon.rel_mha_extend_gfx950 import (  # noqa: F401
    gluon_rel_mha_extend_gfx950,
)
from tokenspeed_kernel_amd.ops.attention.gluon.rel_mha_prefill_gfx950 import (  # noqa: F401
    gluon_rel_mha_prefill_gfx950,
)
