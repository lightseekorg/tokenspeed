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

"""Resolve a certificate directory into the Mooncake SDK's TLS environment."""

import os
from pathlib import Path


def configure_etcd_tls() -> None:
    """Use MOONCAKE_ETCD_CERT_DIR; explicit SDK file variables take precedence.

    A directory contains a CA bundle (ca*.crt), one client certificate and one
    private key (*.key). Conventional ca.crt/client.crt/client.key also work.
    File contents and authentication are validated by the Mooncake SDK.
    """
    directory = os.environ.get("MOONCAKE_ETCD_CERT_DIR")
    if not directory:
        return
    root = Path(directory).expanduser()
    if not root.is_dir():
        raise ValueError("MOONCAKE_ETCD_CERT_DIR is not a directory")
    candidates = {
        "MOONCAKE_ETCD_CA_CERT": sorted(root.glob("ca*.crt")),
        "MOONCAKE_ETCD_CLIENT_CERT": sorted(
            p for p in root.glob("*.crt") if not p.name.startswith("ca")
        ),
        "MOONCAKE_ETCD_CLIENT_KEY": sorted(root.glob("*.key")),
    }
    resolved = {}
    for name, paths in candidates.items():
        explicit = os.environ.get(name)
        if explicit:
            path = Path(explicit).expanduser()
        elif len(paths) == 1:
            path = paths[0]
        else:
            raise ValueError(f"Certificate directory requires one file for {name}")
        if not path.is_file():
            raise ValueError(f"{name} is not a file")
        resolved[name] = str(path.resolve())
    # Resolve the whole set before publishing any paths to the native SDK.
    os.environ.update(resolved)
