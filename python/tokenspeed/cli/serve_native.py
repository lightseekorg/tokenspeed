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

"""In-tree HTTP server path.

This is the body that used to live inline in ``cli.py::_serve``. It is the
implementation behind ``ts serve`` today; commit 1 of the ``ts-serve``
migration only extracts it into its own module without changing behavior.
"""

from __future__ import annotations

import argparse
import os
import traceback


def run_native(args: argparse.Namespace) -> None:
    from tokenspeed.runtime.entrypoints.http_server import api_server
    from tokenspeed.runtime.utils.process import kill_process_tree
    from tokenspeed.runtime.utils.server_args import ServerArgs

    server_args = ServerArgs.from_cli_args(args)

    try:
        api_server(server_args)
    except Exception:
        traceback.print_exc()
    finally:
        kill_process_tree(os.getpid(), include_parent=False)
