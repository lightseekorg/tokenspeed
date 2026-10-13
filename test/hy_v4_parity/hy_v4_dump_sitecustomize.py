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

"""Sitecustomize that makes any engine dump HY V4 activations on load.

Copy this file onto ``PYTHONPATH`` (keeping the name ``sitecustomize.py``) and
set ``HY_V4_DUMP_PATH``. Every interpreter that starts afterwards -- including
engine worker subprocesses, which a patch in the parent cannot reach -- attaches
a layer dumper as soon as a HY V4 model finishes loading, and writes the dump on
interpreter exit.

    HY_V4_DUMP_PATH=/tmp/ts.json PYTHONPATH=/path/to/inject ./run_tokenspeed.sh

The instrumentation is inert unless both conditions hold: the module must be
imported as ``sitecustomize`` (Python only auto-imports the first one on
``sys.path``, hence the explicit chaining below) and ``HY_V4_DUMP_PATH`` must be
set. This mirrors the injection pattern used by
``test/runtime/test_slab_capacity_prefix_hits.py``.
"""

import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))


def _chain_next_sitecustomize() -> None:
    """Run any other ``sitecustomize`` later on ``sys.path``.

    Python imports only the first one it finds, so skipping the rest would
    silently disable an unrelated injection (e.g. another test's).
    """
    for entry in sys.path:
        if not entry or os.path.abspath(entry) == _HERE:
            continue
        candidate = os.path.join(entry, "sitecustomize.py")
        if os.path.isfile(candidate):
            with open(candidate) as handle:
                source = handle.read()
            namespace = {"__name__": "sitecustomize", "__file__": candidate}
            try:
                exec(compile(source, candidate, "exec"), namespace)
            except Exception as error:  # pragma: no cover - diagnostics only
                print(
                    f"[hy_v4_dump] chained sitecustomize failed: {error}",
                    file=sys.stderr,
                )
            return


_chain_next_sitecustomize()


def _install() -> None:
    dump_target = os.environ.get("HY_V4_DUMP_PATH")
    if not dump_target:
        return

    # Import torch before patching: the hook stores tensors, and failing here
    # (loudly) is better than failing deep inside a forward pass.
    import torch  # noqa: F401

    module_name = os.environ.get("HY_V4_DUMP_MODULE", "test.hy_v4_parity.hy_v4_dump")
    module = __import__(module_name, fromlist=["instrument_hy_v4_classes"])
    instrumented = module.instrument_hy_v4_classes(
        path=dump_target,
        num_layers=int(os.environ.get("HY_V4_DUMP_LAYERS", "0")) or None,
        tags=[f"tag={os.environ.get('HY_V4_DUMP_TAG', '')}"],
    )
    print(
        f"[hy_v4_dump] armed target={dump_target} instrumented={instrumented}",
        file=sys.stderr,
    )


try:
    _install()
except Exception as error:  # pragma: no cover - never break the interpreter
    # A dump that cannot be armed must be loud: a silently empty dump would make
    # the parity comparison look like a pass.
    print(f"[hy_v4_dump] install failed: {error}", file=sys.stderr)
