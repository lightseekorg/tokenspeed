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

"""Regression coverage for generation weight-version metadata."""

from __future__ import annotations

import asyncio
import os
import sys
import unittest
from types import SimpleNamespace
from unittest import mock

from fastapi.testclient import TestClient

# CI registration (AST-parsed, runtime no-op).
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from ci_system.ci_register import register_cuda_ci  # noqa: E402

register_cuda_ci(est_time=10, suite="runtime-1gpu")

from runtime.rl_fakes import FakeLLM  # noqa: E402

from tokenspeed.runtime.engine.collector import RequestOutputCollector  # noqa: E402
from tokenspeed.runtime.engine.io_struct import BatchEmbeddingOut  # noqa: E402
from tokenspeed.runtime.engine.output_processor import (  # noqa: E402
    OutputProcessor,
    ReqState,
)
from tokenspeed.runtime.entrypoints import (  # noqa: E402
    control_server,
    sglang_compat_http,
)
from tokenspeed.runtime.entrypoints.sglang_compat_http import (  # noqa: E402
    build_sglang_compat_app,
)
from tokenspeed.runtime.utils.server_args import (  # noqa: E402
    ServerArgs,
    prepare_server_args,
)


class TestWeightVersionHTTP(unittest.TestCase):
    def test_sglang_version_endpoints(self):
        llm = FakeLLM()
        client = TestClient(build_sglang_compat_app(llm))

        self.assertEqual(
            client.get("/get_weight_version").json(),
            {"weight_version": "default"},
        )
        self.assertEqual(
            client.get("/model_info").json(),
            {"model_path": "model-x", "weight_version": "default"},
        )
        response = client.post("/update_weight_version", json={"new_version": 7})
        self.assertEqual(response.status_code, 200)
        self.assertEqual(llm.server_args.weight_version, "7")
        self.assertEqual(
            client.post("/update_weight_version", json={}).status_code,
            400,
        )

    def test_sglang_direct_version_update_rejects_l3_without_mutation(self):
        llm = FakeLLM()
        llm.server_args.kvstore_storage_backend = "mooncake"
        client = TestClient(build_sglang_compat_app(llm))

        for new_version in ("new-checkpoint", 7, "default"):
            with self.subTest(new_version=new_version):
                response = client.post(
                    "/update_weight_version", json={"new_version": new_version}
                )
                self.assertEqual(response.status_code, 400)
                self.assertFalse(response.json()["success"])
                self.assertIn(
                    "/update_weights_from_distributed", response.json()["message"]
                )
                self.assertEqual(llm.server_args.weight_version, "default")
                self.assertEqual(
                    client.get("/get_weight_version").json(),
                    {"weight_version": "default"},
                )
                self.assertEqual(
                    client.get("/model_info").json()["weight_version"], "default"
                )
        self.assertEqual(llm.updates, [])
        self.assertEqual(llm.scheduler_calls, [])

    def test_sglang_updates_stamp_only_after_success(self):
        llm = FakeLLM()
        client = TestClient(build_sglang_compat_app(llm))

        response = client.post(
            "/update_weights_from_distributed",
            json={
                "names": ["weight"],
                "dtypes": ["float32"],
                "shapes": [[1]],
                "weight_version": "v8",
            },
        )
        self.assertEqual(response.status_code, 200)
        self.assertEqual(llm.updates[-1].weight_version, "v8")
        self.assertEqual(llm.server_args.weight_version, "v8")

        llm.succeed = False
        # This build refuses the disk route with 501 before the engine sees it
        # (the scheduler has no branch for it), so widen the supported set to
        # keep exercising "stamp only after success" on a non-distributed route.
        with mock.patch.object(
            sglang_compat_http,
            "SUPPORTED_WEIGHT_UPDATE_SOURCES",
            frozenset({"disk", "tensor", "distributed"}),
        ):
            response = client.post(
                "/update_weights_from_disk",
                json={"model_path": "/tmp/model", "weight_version": "failed"},
            )
        self.assertEqual(response.status_code, 400)
        self.assertEqual(llm.server_args.weight_version, "v8")

    def test_sglang_scheduler_memory_and_load_endpoints(self):
        llm = FakeLLM()
        client = TestClient(build_sglang_compat_app(llm))

        self.assertEqual(client.post("/pause_generation").status_code, 200)
        self.assertEqual(client.post("/continue_generation").status_code, 200)
        self.assertEqual(
            client.post("/abort_request", json={"abort_all": True}).status_code,
            200,
        )
        self.assertEqual(
            llm.scheduler_calls,
            [
                ("pause", "wait"),
                ("resume", None),
                ("pause", "abort"),
                ("resume", None),
            ],
        )
        self.assertEqual(llm.admission_calls, ["block", "allow", "block", "allow"])

        self.assertEqual(
            client.post(
                "/release_memory_occupation", json={"tags": ["kv_cache"]}
            ).status_code,
            200,
        )
        self.assertEqual(
            client.post(
                "/resume_memory_occupation", json={"tags": ["weights"]}
            ).status_code,
            200,
        )
        self.assertEqual(
            llm.memory_calls,
            [("release", ["kv_cache"]), ("resume", ["weights"])],
        )
        self.assertEqual(
            client.get("/v1/loads").json(),
            {
                "loads": [
                    {
                        "dp_rank": 0,
                        "num_reqs": 2,
                        "num_waiting_reqs": 1,
                        "num_pages": 3,
                    }
                ]
            },
        )

    def test_control_server_exposes_only_sglang_rl_routes(self):
        routes = {
            (route.path, frozenset(route.methods or []))
            for route in control_server.app.routes
        }
        self.assertIn(("/get_weight_version", frozenset({"GET"})), routes)
        self.assertIn(("/model_info", frozenset({"GET"})), routes)
        self.assertIn(("/update_weight_version", frozenset({"POST"})), routes)
        self.assertIn(("/v1/loads", frozenset({"GET"})), routes)
        removed_vllm_routes = {
            ("/init_weight_transfer_engine", frozenset({"POST"})),
            ("/start_weight_update", frozenset({"POST"})),
            ("/update_weights", frozenset({"POST"})),
            ("/finish_weight_update", frozenset({"POST"})),
            ("/pause", frozenset({"POST"})),
            ("/resume", frozenset({"POST"})),
            ("/get_world_size", frozenset({"GET"})),
            ("/is_paused", frozenset({"GET"})),
        }
        self.assertTrue(removed_vllm_routes.isdisjoint(routes))


class _Request:
    stream = False
    sampling_params = {}
    return_logprob = False
    log_metrics = False


class TestGenerationVersionStamp(unittest.TestCase):
    def test_output_meta_info_carries_current_version(self):
        engine = SimpleNamespace(
            server_args=SimpleNamespace(
                weight_version="v-output",
                speculative_algorithm=None,
            ),
            rid_to_state={},
            enable_metrics=False,
            dump_requests_folder=False,
        )
        state = ReqState(
            RequestOutputCollector(),
            False,
            asyncio.Event(),
            _Request(),
            created_time=0.0,
        )
        engine.rid_to_state["rid"] = state

        OutputProcessor(engine).handle_batch_output(
            BatchEmbeddingOut(
                rids=["rid"],
                finished_reasons=[None],
                embeddings=[[1.0]],
                prompt_tokens=[3],
            )
        )

        self.assertEqual(
            state.collector.take()["meta_info"]["weight_version"],
            "v-output",
        )

    def test_server_args_default_version(self):
        self.assertEqual(ServerArgs(model="model-x").weight_version, "default")

    def test_server_args_accepts_initial_version_from_cli(self):
        server_args = prepare_server_args(["model-x", "--weight-version", "policy-v1"])
        self.assertEqual(server_args.weight_version, "policy-v1")


if __name__ == "__main__":
    unittest.main(verbosity=2)
