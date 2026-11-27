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

"""Topology lookup can run before the communication implementation loads."""

import subprocess
import sys
from test.ci_system.ci_register import register_cuda_ci
from textwrap import dedent

register_cuda_ci(est_time=3, suite="runtime-1gpu")


def test_mapping_is_usable_without_communication_imports() -> None:
    probe = dedent("""
        import sys
        from tokenspeed.runtime.distributed.mapping import Mapping
        from tokenspeed.runtime.distributed import Mapping as PublicMapping

        assert Mapping is PublicMapping
        mapping = Mapping(rank=0, world_size=4, attn_tp_size=1, attn_dp_size=4)
        assert (mapping.attn.tp_size, mapping.attn.dp_size) == (1, 4)
        assert 'tokenspeed.runtime.distributed.comm_manager' not in sys.modules
        assert 'tokenspeed.runtime.distributed.comm_ops' not in sys.modules
        """)
    result = subprocess.run(
        [sys.executable, "-c", probe], capture_output=True, text=True, timeout=30
    )
    assert result.returncode == 0, result.stderr
