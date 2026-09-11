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

"""V4.1 gateway selection and real SMG HTTP-to-token parity, without GPUs.

Set DEEPSEEK_V41_REFERENCE_DIR to a trusted snapshot containing encoding/encoding.py
and tokenizer.json. The HTTP tests launch ts serve with this module as its CPU
fake engine; they never import the model or load weights. Also executable as that
fake engine via TS_SERVE_ENGINE_MODULE.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import signal
import socket
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

import pytest


def _serve_fake_engine():
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    from concurrent.futures import ThreadPoolExecutor
    from test.cli._fixtures.fake_engine import _FakeHealth

    import grpc
    from grpc_health.v1 import health_pb2_grpc
    from smg_grpc_proto import tokenspeed_scheduler_pb2 as pb
    from smg_grpc_proto import tokenspeed_scheduler_pb2_grpc as pbg
    from tokenizers import Tokenizer

    parser = argparse.ArgumentParser()
    parser.add_argument("--host", required=True)
    parser.add_argument("--port", type=int, required=True)
    parser.add_argument("--model", required=True)
    args, _ = parser.parse_known_args()
    tokenizer = Tokenizer.from_file(str(Path(args.model) / "tokenizer.json"))
    capture = Path(os.environ["TOKENSPEED_V41_TEST_CAPTURE"])

    class Engine(pbg.TokenSpeedSchedulerServicer):
        def GetModelInfo(self, request, context):
            return pb.GetModelInfoResponse(
                model_path=args.model,
                tokenizer_path=args.model,
                served_model_name="v41-text-test",
                model_type="deepseek_v41",
                architectures=["DeepseekV41ForCausalLM"],
                max_context_length=16384,
                max_req_input_len=16384,
                vocab_size=tokenizer.get_vocab_size(),
                eos_token_ids=[tokenizer.token_to_id("<｜end▁of▁sentence｜>")],
                bos_token_id=tokenizer.token_to_id("<｜begin▁of▁sentence｜>"),
                weight_version="test",
                supports_vision=False,
                supports_multimodal=False,
            )

        def HealthCheck(self, request, context):
            return pb.HealthCheckResponse(healthy=True)

        def GetServerInfo(self, request, context):
            return pb.GetServerInfoResponse()

        def GetLoads(self, request, context):
            return pb.GetLoadsResponse()

        def Generate(self, request, context):
            thinking = request.tokenized.input_ids[-1] == tokenizer.token_to_id(
                "<think>"
            )
            output = tokenizer.encode(
                "stub reasoning.</think>OK" if thinking else "OK",
                add_special_tokens=False,
            ).ids
            with capture.open("a", encoding="utf-8") as f:
                f.write(json.dumps(list(request.tokenized.input_ids)) + "\n")
            if request.stream:
                yield pb.GenerateResponse(
                    request_id=request.request_id,
                    chunk=pb.GenerateStreamChunk(
                        token_ids=output,
                        prompt_tokens=len(request.tokenized.input_ids),
                        completion_tokens=len(output),
                    ),
                )
            yield pb.GenerateResponse(
                request_id=request.request_id,
                complete=pb.GenerateComplete(
                    output_ids=output,
                    finish_reason="stop",
                    prompt_tokens=len(request.tokenized.input_ids),
                    completion_tokens=len(output),
                ),
            )

    server = grpc.server(ThreadPoolExecutor(max_workers=4))
    health_pb2_grpc.add_HealthServicer_to_server(_FakeHealth(), server)
    pbg.add_TokenSpeedSchedulerServicer_to_server(Engine(), server)
    server.add_insecure_port(f"{args.host}:{args.port}")
    signal.signal(signal.SIGTERM, lambda _sig, _frame: server.stop(0))
    server.start()
    try:
        server.wait_for_termination(timeout=90)
    finally:
        server.stop(0).wait(timeout=5)


@pytest.mark.parametrize(
    "model",
    ["deepseek-ai/DeepSeek-V4.1-Flash", "DeepSeek_V4_1_Flash", "DeepSeekV41-Flash"],
)
def test_v41_never_selects_v4(model):
    from tokenspeed.cli import serve_smg as serve

    assert serve._is_deepseek_v41_model(model)
    assert not serve._is_deepseek_v4_model(model)
    engine, gateway = serve._args_with_default_model_parsers(
        ["--model", model], ["--model", model]
    )
    assert serve._get_from_args(engine, "--reasoning-parser", None) == "deepseek_v31"
    assert serve._get_from_args(gateway, "--tool-call-parser", None) == "passthrough"
    assert "deepseek_v4" not in gateway


@pytest.mark.parametrize(
    "config",
    [{"model_type": "deepseek_v41"}, {"architectures": ["DeepseekV41ForCausalLM"]}],
)
def test_v41_detects_renamed_local_snapshot(tmp_path, config):
    from tokenspeed.cli import serve_smg as serve

    (tmp_path / "config.json").write_text(json.dumps(config))
    assert serve._is_deepseek_v41_model(str(tmp_path))
    assert not serve._is_deepseek_v4_model(str(tmp_path))


def test_v41_rejects_incompatible_tool_parser():
    from tokenspeed.cli import serve_smg as serve

    with pytest.raises(ValueError, match="V4 DSML parser is incompatible"):
        serve._args_with_default_model_parsers(
            ["--model", "DeepSeek-V4.1-Flash"], ["--tool-call-parser", "deepseek_v4"]
        )


@pytest.mark.parametrize("custom_template", [False, True])
def test_serve_registers_template_for_gateway_lifetime(
    monkeypatch, tmp_path, custom_template
):
    from tokenspeed.cli import serve_smg as serve

    model = str(tmp_path)
    (tmp_path / "config.json").write_text(json.dumps({"model_type": "deepseek_v41"}))
    captured = []

    async def run(**kwargs):
        path = serve._get_from_args(kwargs["gateway_args"], "--chat-template", None)
        captured.append(path)
        if custom_template:
            assert path == "operator-template.jinja"
        else:
            assert Path(path).read_text() == serve._DEEPSEEK_V41_CHAT_TEMPLATE
        return 0

    monkeypatch.setattr(serve, "run_smg", run)
    monkeypatch.setattr(serve, "_check_serve_extra_installed", lambda: None)
    monkeypatch.setattr(serve, "print_logo", lambda: None)
    argv = ["--model", model, "--node-rank", "0"]
    if custom_template:
        argv.extend(["--chat-template", "operator-template.jinja"])
    with pytest.raises(SystemExit) as exc:
        serve.run_smg_from_args(argparse.Namespace(), argv)
    assert exc.value.code == 0
    assert len(captured) == 1
    if not custom_template:
        assert not Path(captured[0]).exists()


@pytest.fixture(scope="module")
def reference():
    from tokenizers import Tokenizer

    directory = os.environ.get("DEEPSEEK_V41_REFERENCE_DIR")
    if directory is None:
        pytest.skip("set DEEPSEEK_V41_REFERENCE_DIR to a trusted V4.1 snapshot")
    snapshot = Path(directory)
    spec = importlib.util.spec_from_file_location(
        "_v41_gateway_reference", snapshot / "encoding" / "encoding.py"
    )
    assert spec is not None and spec.loader is not None
    encoder = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(encoder)
    return snapshot, encoder, Tokenizer.from_file(str(snapshot / "tokenizer.json"))


def _free_port():
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


@pytest.fixture(scope="module")
def gateway(reference, tmp_path_factory):
    pytest.importorskip("smg")
    pytest.importorskip("smg_grpc_proto")
    snapshot, _, _ = reference
    directory = tmp_path_factory.mktemp("v41-gateway")
    capture = directory / "requests.jsonl"
    capture.touch()
    port = _free_port()
    url = f"http://127.0.0.1:{port}"
    root = Path(__file__).resolve().parents[2]
    env = os.environ.copy()
    env.update(
        TS_SERVE_ENGINE_MODULE="test.cli.test_serve_smg_deepseek_v41",
        TOKENSPEED_V41_TEST_CAPTURE=str(capture),
        HF_HUB_OFFLINE="1",
        TRANSFORMERS_OFFLINE="1",
        PYTHONPATH=str(root) + os.pathsep + env.get("PYTHONPATH", ""),
    )
    with (directory / "gateway.log").open("w+") as log:
        process = subprocess.Popen(
            [
                sys.executable,
                "-m",
                "tokenspeed.cli",
                "serve",
                "--model",
                str(snapshot),
                "--trust-remote-code",
                "--node-rank",
                "0",
                "--host",
                "127.0.0.1",
                "--port",
                str(port),
                "--control-port",
                str(_free_port()),
                "--prometheus-host",
                "127.0.0.1",
                "--prometheus-port",
                str(_free_port()),
                "--worker-startup-check-interval",
                "1",
                "--engine-startup-timeout",
                "20",
                "--gateway-startup-timeout",
                "20",
                "--drain-timeout",
                "3",
            ],
            cwd=root,
            env=env,
            stdout=log,
            stderr=subprocess.STDOUT,
        )
        try:
            deadline = time.monotonic() + 40
            while time.monotonic() < deadline and process.poll() is None:
                try:
                    with urllib.request.urlopen(
                        url + "/readiness", timeout=1
                    ) as response:
                        if response.status == 200:
                            break
                except (urllib.error.URLError, TimeoutError):
                    time.sleep(0.1)
            else:
                log.seek(0)
                pytest.fail("CPU fake-engine gateway failed to start:\n" + log.read())
            yield url, capture
        finally:
            if process.poll() is None:
                process.terminate()
                try:
                    process.wait(timeout=15)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait(timeout=5)


_MESSAGES = [
    [{"role": "user", "content": "What is 2+2?"}],
    [
        {"role": "system", "content": " Be helpful. "},
        {"role": "user", "content": "你好"},
    ],
    [
        {"role": "user", "content": "First question"},
        {
            "role": "assistant",
            "reasoning_content": "old reasoning",
            "content": "First answer",
        },
        {"role": "user", "content": "Next question"},
    ],
    [
        {"role": "user", "content": "First question"},
        {
            "role": "assistant",
            "reasoning_content": "old reasoning",
            "content": "First answer",
        },
        {"role": "system", "content": "Now answer in French."},
    ],
    [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "one"},
                {"type": "text", "text": "two"},
            ],
        },
        {"role": "user", "content": "three"},
    ],
]
_OPTIONS = [
    {},
    {"enable_thinking": False},
    {"enable_thinking": True},
    {"enable_thinking": True, "reasoning_effort": "low"},
    {"enable_thinking": True, "reasoning_effort": 37},
    {"enable_thinking": True, "reasoning_effort": 1},
    {"enable_thinking": True, "reasoning_effort": 100},
    {"enable_thinking": True, "reasoning_effort": "high"},
    {"enable_thinking": True, "reasoning_effort": "max", "drop_thinking": False},
]


@pytest.mark.parametrize("messages", _MESSAGES)
@pytest.mark.parametrize(
    ("options", "top_level_effort"),
    [(options, None) for options in _OPTIONS]
    + [
        (options, effort)
        for options in ({}, {"enable_thinking": True})
        for effort in ("low", "high", "max")
    ],
)
def test_http_chat_matches_official_token_ids(
    gateway, reference, messages, options, top_level_effort
):
    _, encoder, tokenizer = reference
    url, capture = gateway
    body = {
        "model": "v41-text-test",
        "messages": messages,
        "max_tokens": 8,
        "chat_template_kwargs": options,
    }
    if top_level_effort is not None:
        body["reasoning_effort"] = top_level_effort
    thinking = options.get("enable_thinking", False)
    request = urllib.request.Request(
        url + "/v1/chat/completions",
        data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(request, timeout=5) as response:
        result = json.load(response)
    message = result["choices"][0]["message"]
    assert message["content"] == "OK"
    if thinking:
        assert message["reasoning_content"] == "stub reasoning."
    expected = encoder.encode_messages(
        messages,
        thinking_mode="thinking" if thinking else "chat",
        context=None,
        drop_thinking=options.get("drop_thinking", True),
        add_default_bos_token=True,
        reasoning_effort=options.get("reasoning_effort", top_level_effort),
        return_multi_modal_data=False,
    )
    actual_ids = json.loads(capture.read_text().splitlines()[-1])
    assert actual_ids == tokenizer.encode(expected, add_special_tokens=False).ids
    assert actual_ids.count(tokenizer.token_to_id("<｜begin▁of▁sentence｜>")) == 1


@pytest.mark.parametrize(
    "extra",
    [
        {
            "tools": [
                {
                    "type": "function",
                    "function": {"name": "lookup", "parameters": {"type": "object"}},
                }
            ]
        },
        {
            "messages": [
                {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [
                        {
                            "id": "call1",
                            "type": "function",
                            "function": {"name": "lookup", "arguments": "{}"},
                        }
                    ],
                },
                {"role": "tool", "tool_call_id": "call1", "content": "42"},
            ]
        },
        {
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image_url",
                            "image_url": {"url": "data:image/png;base64,invalid"},
                        }
                    ],
                }
            ]
        },
        {
            "chat_template_kwargs": {
                "enable_thinking": True,
                "reasoning_effort": "medium",
            }
        },
        {"chat_template_kwargs": {"enable_thinking": True, "reasoning_effort": 0}},
        {"chat_template_kwargs": {"enable_thinking": True, "reasoning_effort": 101}},
        {"chat_template_kwargs": {"enable_thinking": True, "reasoning_effort": True}},
        {"chat_template_kwargs": {"thinking": True}},
        {"chat_template_kwargs": {"thinking_mode": "thinking"}},
    ],
)
def test_http_unsupported_requests_never_reach_engine(gateway, extra):
    url, capture = gateway
    before = capture.read_text()
    body = {
        "model": "v41-text-test",
        "messages": [{"role": "user", "content": "hello"}],
        "max_tokens": 8,
        **extra,
    }
    request = urllib.request.Request(
        url + "/v1/chat/completions",
        data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json"},
    )
    with pytest.raises(urllib.error.HTTPError) as exc:
        urllib.request.urlopen(request, timeout=5)
    detail = exc.value.read().decode()
    assert exc.value.code in (400, 422), detail
    assert (
        "not supported" in detail
        or "reasoning_effort" in detail
        or "multimodal" in detail
    ), detail
    assert capture.read_text() == before


def test_http_streaming_chat(gateway):
    url, _ = gateway
    request = urllib.request.Request(
        url + "/v1/chat/completions",
        data=json.dumps(
            {
                "model": "v41-text-test",
                "messages": [{"role": "user", "content": "hello"}],
                "max_tokens": 8,
                "stream": True,
                "chat_template_kwargs": {
                    "enable_thinking": True,
                    "reasoning_effort": "low",
                },
            }
        ).encode(),
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(request, timeout=5) as response:
        events = response.read().decode().splitlines()
    deltas = [json.loads(line[6:]) for line in events if line.startswith("data: {")]
    content = "".join(
        choice["delta"].get("content") or ""
        for event in deltas
        for choice in event["choices"]
    )
    assert content == "OK"
    assert "data: [DONE]" in events


if __name__ == "__main__":
    _serve_fake_engine()
