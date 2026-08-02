"""Tests for in-engine native generation."""

from fastapi.testclient import TestClient

from tokenspeed.runtime.entrypoints.control_server import _needs_engine_generate
from tokenspeed.runtime.entrypoints.native_generate_http import (
    build_native_generate_app,
)


class _FakeAsyncLLM:
    def __init__(self):
        self.request = None

    async def generate_request(self, request):
        self.request = request
        yield {
            "text": "answer",
            "output_ids": [1, 2, 42],
            "meta_info": {
                "output_token_logprobs": [(-0.1, 42, None)],
                "output_top_logprobs": [[(-0.1, 42, None)]],
                "completion_tokens": 1,
            },
        }


def test_detailed_logprob_fields_reach_engine():
    llm = _FakeAsyncLLM()
    client = TestClient(build_native_generate_app(llm))

    response = client.post(
        "/_engine/generate",
        json={
            "input_ids": [[1, 2], [3, 4]],
            "sampling_params": [
                {"max_new_tokens": 1},
                {"max_new_tokens": 1},
            ],
            "return_logprob": [True, True],
            "logprob_start_len": [-1, 0],
            "top_logprobs_num": [4, 128],
            "token_ids_logprob": [[7, 9], [11]],
        },
    )

    assert response.status_code == 200
    assert response.json()["output_ids"] == [42]
    assert llm.request.return_logprob == [True, True]
    assert llm.request.logprob_start_len == [-1, 0]
    assert llm.request.top_logprobs_num == [4, 128]
    assert llm.request.token_ids_logprob == [[7, 9], [11]]


def test_non_object_body_is_rejected():
    client = TestClient(build_native_generate_app(_FakeAsyncLLM()))

    response = client.post("/_engine/generate", json=[1, 2, 3])

    assert response.status_code == 400


def test_only_detailed_requests_use_in_engine_route():
    assert not _needs_engine_generate(
        b'{"return_logprob": true, "sampling_params": {"max_new_tokens": 1}}'
    )
    assert _needs_engine_generate(b'{"return_logprob": true, "top_logprobs_num": 4}')
    assert _needs_engine_generate(
        b'{"return_logprob": [true, true], "top_logprobs_num": [16, 128]}'
    )
    assert _needs_engine_generate(b'{"return_logprob": true, "logprob_start_len": 0}')
    assert _needs_engine_generate(
        b'{"return_logprob": true, "token_ids_logprob": [7, 9]}'
    )
