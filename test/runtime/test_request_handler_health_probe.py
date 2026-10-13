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

"""How the scheduler answers the gRPC servicer's deep health probe.

The probe is a one-token generate that passes on any scheduler output within
its timeout. Queued behind a long-prefill backlog it would see none for
minutes, so a scheduler that holds requests answers it at once instead.
"""

from __future__ import annotations

import os
import sys
import unittest
from unittest import mock

# CPU-only tests scheduled in runtime-1gpu because they import the full runtime.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from ci_system.ci_register import register_cuda_ci  # noqa: E402

register_cuda_ci(est_time=10, suite="runtime-1gpu")

from tokenspeed.runtime.engine.io_struct import (  # noqa: E402
    HEALTH_CHECK_RID_PREFIX,
    HealthCheckOutput,
    TokenizedGenerateReqInput,
)
from tokenspeed.runtime.engine.request_handler import RequestHandler  # noqa: E402

PROBE_RID = f"{HEALTH_CHECK_RID_PREFIX}1760000000.5"


def _generate(rid: str) -> TokenizedGenerateReqInput:
    req = mock.Mock(spec=TokenizedGenerateReqInput)
    req.rid = rid
    return req


class TestHealthProbe(unittest.TestCase):
    def _handler(self, num_running: int) -> RequestHandler:
        handler = RequestHandler.__new__(RequestHandler)
        handler.send_func = mock.Mock()
        handler.num_running_fn = mock.Mock(return_value=num_running)
        handler.handle_generate_request = mock.Mock(
            side_effect=lambda req: (f"spec-{req.rid}", f"state-{req.rid}", None)
        )
        handler.attn_dp_size = 1
        handler.attn_dp_cpu_group = None
        handler._rendezvous_replica_flush = mock.Mock(return_value=(True, None, None))
        return handler

    def test_busy_scheduler_answers_the_probe_without_queueing_it(self):
        handler = self._handler(num_running=3)

        specs, states, _, _ = handler.process_requests(
            [_generate(PROBE_RID), _generate("user")]
        )

        self.assertEqual(specs, ["spec-user"])
        self.assertEqual(states, ["state-user"])
        sent = handler.send_func.send_pyobj.call_args_list
        self.assertEqual(len(sent), 1)
        self.assertIsInstance(sent[0].args[0], HealthCheckOutput)

    def test_idle_scheduler_runs_the_probe_as_a_generate(self):
        handler = self._handler(num_running=0)

        specs, _, _, _ = handler.process_requests([_generate(PROBE_RID)])

        self.assertEqual(specs, [f"spec-{PROBE_RID}"])
        handler.send_func.send_pyobj.assert_not_called()

    def test_only_the_probe_prefix_short_circuits(self):
        handler = self._handler(num_running=3)

        specs, _, _, _ = handler.process_requests([_generate("HEALTH-user")])

        self.assertEqual(specs, ["spec-HEALTH-user"])
        handler.send_func.send_pyobj.assert_not_called()


if __name__ == "__main__":
    unittest.main()
