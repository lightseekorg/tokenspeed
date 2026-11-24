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

"""The L3 tier's settings, resolved once at startup: the scheduler's storage
limits, the executor's workers, the store client and what the engine host
starts its executor from (``L3Settings``)."""

"""Certificate-directory configuration for the native Store and direct reader."""

import os

import pytest

from tokenspeed.runtime.cache.l3.tls import configure_etcd_tls


@pytest.fixture
def certificates(tmp_path, monkeypatch):
    monkeypatch.setenv("MOONCAKE_ETCD_CERT_DIR", str(tmp_path))
    names = {
        "MOONCAKE_ETCD_CA_CERT": "ca_bundle.crt",
        "MOONCAKE_ETCD_CLIENT_CERT": "client.crt",
        "MOONCAKE_ETCD_CLIENT_KEY": "client.key",
    }
    for variable, name in names.items():
        monkeypatch.delenv(variable, raising=False)
        (tmp_path / name).write_text("fixture")
    return tmp_path, names


def test_certificate_directory_resolves_complete_sdk_configuration(certificates):
    directory, names = certificates
    configure_etcd_tls()
    for variable, name in names.items():
        assert os.environ[variable] == str(directory / name)


def test_explicit_certificate_overrides_directory(certificates, tmp_path, monkeypatch):
    explicit = tmp_path / "explicit.pem"
    explicit.write_text("fixture")
    monkeypatch.setenv("MOONCAKE_ETCD_CA_CERT", str(explicit))
    configure_etcd_tls()
    assert os.environ["MOONCAKE_ETCD_CA_CERT"] == str(explicit)


def test_incomplete_directory_does_not_partly_publish_configuration(certificates):
    directory, names = certificates
    (directory / "client.key").unlink()
    with pytest.raises(ValueError, match="CLIENT_KEY"):
        configure_etcd_tls()
    assert all(variable not in os.environ for variable in names)


def test_no_directory_preserves_existing_sdk_settings(monkeypatch):
    monkeypatch.delenv("MOONCAKE_ETCD_CERT_DIR", raising=False)
    monkeypatch.setenv("MOONCAKE_ETCD_CA_CERT", "/explicit/ca.crt")
    configure_etcd_tls()
    assert os.environ["MOONCAKE_ETCD_CA_CERT"] == "/explicit/ca.crt"
