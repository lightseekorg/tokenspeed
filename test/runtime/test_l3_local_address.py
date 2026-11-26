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

"""Mooncake reuses the launcher's address while preserving explicit overrides."""

import pytest

from tokenspeed.runtime.cache.l3.config import MooncakeStoreSettings


@pytest.mark.parametrize(
    ("extra", "env", "expected"),
    [
        ({}, {"TOKENSPEED_HOST_IP": "192.0.2.11"}, "192.0.2.11"),
        ({}, {"TOKENSPEED_HOST_IP": "192.0.2.12"}, "192.0.2.12"),
        (
            {"local_hostname": "explicit"},
            {
                "TOKENSPEED_HOST_IP": "192.0.2.11",
                "MOONCAKE_LOCAL_HOSTNAME": "mc",
                "LOCAL_HOSTNAME": "legacy",
            },
            "explicit",
        ),
        (
            {},
            {
                "TOKENSPEED_HOST_IP": "192.0.2.11",
                "MOONCAKE_LOCAL_HOSTNAME": "mc",
                "LOCAL_HOSTNAME": "legacy",
            },
            "mc",
        ),
        (
            {},
            {"TOKENSPEED_HOST_IP": "192.0.2.11", "LOCAL_HOSTNAME": "legacy"},
            "legacy",
        ),
        (
            {"local_hostname": ""},
            {
                "TOKENSPEED_HOST_IP": "192.0.2.11",
                "MOONCAKE_LOCAL_HOSTNAME": "",
                "LOCAL_HOSTNAME": "",
            },
            "192.0.2.11",
        ),
        ({}, {}, "localhost"),
    ],
)
def test_mooncake_client_address(monkeypatch, extra, env, expected):
    for name in ("MOONCAKE_LOCAL_HOSTNAME", "LOCAL_HOSTNAME", "TOKENSPEED_HOST_IP"):
        monkeypatch.delenv(name, raising=False)
    for name, value in env.items():
        monkeypatch.setenv(name, value)
    settings = MooncakeStoreSettings.resolve(
        {
            "master_server_address": "store.example:50051",
            "global_segment_size": 0,
            **extra,
        },
        library="store.so",
        tp_size=1,
        pp_size=1,
    )
    assert settings.local_hostname == expected
