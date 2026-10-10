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

"""``--retraction-snapshot-host-gb`` / ``--retraction-snapshot-max-requests`` resolution."""

import pytest

from tokenspeed.runtime.utils.server_args import ServerArgs


def test_default_is_no_pool():
    args = ServerArgs(model="x")
    assert args.retraction_snapshot_host_gb == 0.0
    assert args.retraction_snapshot_max_requests == 0


def test_a_pool_needs_its_slot_rows_and_vice_versa():
    args = ServerArgs(
        model="x", retraction_snapshot_host_gb=4.0, retraction_snapshot_max_requests=16
    )
    assert args.retraction_snapshot_host_gb == 4.0
    with pytest.raises(ValueError, match="go together"):
        ServerArgs(model="x", retraction_snapshot_host_gb=4.0)
    with pytest.raises(ValueError, match="go together"):
        ServerArgs(model="x", retraction_snapshot_max_requests=16)
    with pytest.raises(ValueError, match="non-negative"):
        ServerArgs(model="x", retraction_snapshot_host_gb=-1.0)


@pytest.mark.parametrize("role", ["prefill", "encode"])
def test_non_retracting_roles_refuse_a_pool(role):
    with pytest.raises(ValueError, match="never retracts"):
        ServerArgs(
            model="x",
            disaggregation_mode=role,
            retraction_snapshot_host_gb=1.0,
            retraction_snapshot_max_requests=2,
        )


def test_the_pool_is_independent_of_the_kvstore():
    args = ServerArgs(
        model="x",
        disable_kvstore=True,
        retraction_snapshot_host_gb=1.0,
        retraction_snapshot_max_requests=2,
    )
    assert args.enable_kvstore is False
    assert args.retraction_snapshot_host_gb == 1.0
