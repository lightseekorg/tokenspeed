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

"""The shared L2 pool's owner for the Rust client's tests: the real owner
service over real ZMQ, as ``test/runtime/shared_l2_test_utils.py`` serves it,
over a pool that keeps its books in Python.

Run with one JSON argument: ``client_timeout_s``, the provider's ``base`` and
``blocks``, and the ``clients`` that join. Prints a line with the owner's
``address`` once they joined, then answers each JSON line on its input with
one JSON line: ``books`` (the pool's books), ``stall`` and ``resume`` (stop
and start answering), ``fail`` and ``heal`` (make a pool method raise). Ends
with its input.
"""

import json
import os
import sys

# The repository's root, for the runtime's test utilities.
sys.path.insert(
    0, os.path.join(os.path.dirname(os.path.abspath(__file__)), *[".."] * 5)
)

from test.runtime.shared_l2_test_utils import (  # noqa: E402
    GEOMETRY,
    LCM_BYTES,
    ServedOwner,
    deployment_id,
)


def main() -> None:
    settings = json.loads(sys.argv[1])
    owner = ServedOwner(
        joined_providers=0, client_timeout_s=settings["client_timeout_s"]
    )
    owner.admin.request(
        "add_provider",
        session="node0",
        base_address=settings["base"],
        bytes=settings["blocks"] * LCM_BYTES,
    )
    for client in settings["clients"]:
        joined = owner.admin.request(
            "hello", client=client, geometry=GEOMETRY, deployment_id=deployment_id()
        )
        assert joined["ready"]
    print(json.dumps({"address": owner.address}), flush=True)
    for line in sys.stdin:
        command = json.loads(line)
        answer = {}
        if command["do"] == "books":
            answer = {
                "stats": owner.stats(),
                "touched": owner.pool.touched,
                "disconnected": owner.pool.disconnected,
            }
        elif command["do"] == "stall":
            owner.stall()
        elif command["do"] == "resume":
            owner.resume()
        elif command["do"] == "fail":
            owner.pool.failing.add(command["method"])
        elif command["do"] == "heal":
            owner.pool.failing.discard(command["method"])
        else:
            raise ValueError(f"unknown command {command}")
        print(json.dumps(answer), flush=True)
    owner.close()


if __name__ == "__main__":
    main()
