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
from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient

# CI registration (AST-parsed, runtime no-op).
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(
    0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
)
from ci_system.ci_register import register_cuda_ci  # noqa: E402

register_cuda_ci(est_time=10, suite="runtime-1gpu")

from test.runtime.rl_fakes import FakeLLM  # noqa: E402

from tokenspeed.runtime.engine.collector import RequestOutputCollector  # noqa: E402
from tokenspeed.runtime.engine.io_struct import BatchEmbeddingOut  # noqa: E402
from tokenspeed.runtime.engine.output_processor import (  # noqa: E402
    OutputProcessor,
    ReqState,
)
from tokenspeed.runtime.entrypoints import control_server  # noqa: E402
from tokenspeed.runtime.entrypoints.sglang_compat_http import (  # noqa: E402
    build_sglang_compat_app,
)
from tokenspeed.runtime.utils.server_args import (  # noqa: E402
    ServerArgs,
    prepare_server_args,
)


class TestWeightVersionHTTP:
    def test_sglang_version_endpoints(self):
        llm = FakeLLM()
        client = TestClient(build_sglang_compat_app(llm))

        assert (client.get("/get_weight_version").json()) == (
            {"weight_version": "default"}
        )
        assert (client.get("/model_info").json()) == (
            {"model_path": "model-x", "weight_version": "default"}
        )
        response = client.post("/update_weight_version", json={"new_version": 7})
        assert (response.status_code) == (200)
        assert (llm.server_args.weight_version) == ("7")
        assert (client.post("/update_weight_version", json={}).status_code) == (400)

    def test_sglang_direct_version_update_rejects_l3_without_mutation(self, subtests):
        llm = FakeLLM()
        llm.server_args.kvstore_storage_backend = "mooncake"
        client = TestClient(build_sglang_compat_app(llm))

        for new_version in ("new-checkpoint", 7, "default"):
            with subtests.test(new_version=new_version):
                response = client.post(
                    "/update_weight_version", json={"new_version": new_version}
                )
                assert (response.status_code) == (400)
                assert not (response.json()["success"])
                assert ("/update_weights_from_distributed") in (
                    response.json()["message"]
                )
                assert (llm.server_args.weight_version) == ("default")
                assert (client.get("/get_weight_version").json()) == (
                    {"weight_version": "default"}
                )
                assert (client.get("/model_info").json()["weight_version"]) == (
                    "default"
                )
        assert (llm.updates) == ([])
        assert (llm.scheduler_calls) == ([])

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
        assert (response.status_code) == (200)
        assert (llm.updates[-1].weight_version) == ("v8")
        assert (llm.server_args.weight_version) == ("v8")

        llm.succeed = False
        response = client.post(
            "/update_weights_from_disk",
            json={"model_path": "/tmp/model", "weight_version": "failed"},
        )
        assert (response.status_code) == (400)
        assert (llm.server_args.weight_version) == ("v8")

    def test_sglang_disk_update_passes_flush_cache_and_weight_version(self):
        llm = FakeLLM()
        client = TestClient(build_sglang_compat_app(llm))

        response = client.post(
            "/update_weights_from_disk",
            json={"model_path": "/tmp/model", "flush_cache": False},
        )
        assert (response.status_code) == (200)
        assert (llm.updates[-1].flush_cache) is (False)
        assert (llm.updates[-1].weight_version) is (None)

        client.post(
            "/update_weights_from_disk",
            json={"model_path": "/tmp/model", "weight_version": "v2"},
        )
        assert (llm.updates[-1].flush_cache) is (True)
        assert (llm.updates[-1].weight_version) == ("v2")

    def test_sglang_scheduler_memory_and_load_endpoints(self):
        llm = FakeLLM()
        client = TestClient(build_sglang_compat_app(llm))

        assert (client.post("/pause_generation").status_code) == (200)
        assert (client.post("/continue_generation").status_code) == (200)
        assert (
            client.post("/abort_request", json={"abort_all": True}).status_code
        ) == (200)
        assert (llm.scheduler_calls) == (
            [
                ("pause", "wait"),
                ("resume", None),
                ("pause", "abort"),
                ("resume", None),
            ]
        )
        assert (llm.admission_calls) == (["block", "allow", "block", "allow"])

        assert (
            client.post(
                "/release_memory_occupation", json={"tags": ["kv_cache"]}
            ).status_code
        ) == (200)
        assert (
            client.post(
                "/resume_memory_occupation", json={"tags": ["weights"]}
            ).status_code
        ) == (200)
        assert (llm.memory_calls) == (
            [("release", ["kv_cache"]), ("resume", ["weights"])]
        )
        assert (client.get("/v1/loads").json()) == (
            {
                "loads": [
                    {
                        "dp_rank": 0,
                        "num_reqs": 2,
                        "num_waiting_reqs": 1,
                        "num_pages": 3,
                    }
                ]
            }
        )

    def test_control_server_exposes_only_sglang_rl_routes(self):
        routes = {
            (route.path, frozenset(route.methods or []))
            for route in control_server.app.routes
        }
        assert (("/get_weight_version", frozenset({"GET"}))) in (routes)
        assert (("/model_info", frozenset({"GET"}))) in (routes)
        assert (("/update_weight_version", frozenset({"POST"}))) in (routes)
        assert (("/update_weights_from_mooncake", frozenset({"POST"}))) in (routes)
        assert (("/v1/loads", frozenset({"GET"}))) in (routes)
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
        assert removed_vllm_routes.isdisjoint(routes)


class _Request:
    stream = False
    sampling_params = {}
    return_logprob = False
    log_metrics = False


class TestGenerationVersionStamp:
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

        assert (state.collector.take()["meta_info"]["weight_version"]) == ("v-output")

    def test_server_args_default_version(self):
        assert (ServerArgs(model="model-x").weight_version) == ("default")

    def test_server_args_accepts_initial_version_from_cli(self):
        server_args = prepare_server_args(["model-x", "--weight-version", "policy-v1"])
        assert (server_args.weight_version) == ("policy-v1")


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, *sys.argv[1:]]))
