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

from __future__ import annotations

from tokenspeed_kernel.ops.attention.dsa import *  # noqa: F403
from tokenspeed_kernel.ops.attention.dsa import __all__ as _dsa_all
from tokenspeed_kernel.ops.attention.dsv4 import *  # noqa: F403
from tokenspeed_kernel.ops.attention.dsv4 import __all__ as _dsv4_all
from tokenspeed_kernel.ops.attention.gdn import *  # noqa: F403
from tokenspeed_kernel.ops.attention.gdn import __all__ as _gdn_all
from tokenspeed_kernel.ops.attention.kda import *  # noqa: F403
from tokenspeed_kernel.ops.attention.kda import __all__ as _kda_all
from tokenspeed_kernel.ops.attention.kpool import *  # noqa: F403
from tokenspeed_kernel.ops.attention.kpool import __all__ as _kpool_all
from tokenspeed_kernel.ops.attention.merge_state import attn_merge_state

# Preserve the long-standing module aliases exposed by this package.
from tokenspeed_kernel.ops.attention.mha import *  # noqa: F403
from tokenspeed_kernel.ops.attention.mha import __all__ as _mha_all
from tokenspeed_kernel.ops.attention.mha import flash_attn
from tokenspeed_kernel.ops.attention.mla import *  # noqa: F403
from tokenspeed_kernel.ops.attention.mla import __all__ as _mla_all
from tokenspeed_kernel.ops.attention.mla import tokenspeed_mla
from tokenspeed_kernel.ops.attention.msa import *  # noqa: F403
from tokenspeed_kernel.ops.attention.msa import __all__ as _msa_all
from tokenspeed_kernel.ops.attention.msa import score as msa_score
from tokenspeed_kernel.ops.attention.qsa import *  # noqa: F403
from tokenspeed_kernel.ops.attention.qsa import __all__ as _qsa_all
from tokenspeed_kernel.ops.attention.rmha import *  # noqa: F403
from tokenspeed_kernel.ops.attention.rmha import __all__ as _rmha_all

__all__ = [
    *_mha_all,
    *_rmha_all,
    *_mla_all,
    *_kpool_all,
    *_dsa_all,
    *_msa_all,
    *_dsv4_all,
    *_gdn_all,
    *_kda_all,
    *_qsa_all,
    "attn_merge_state",
]
