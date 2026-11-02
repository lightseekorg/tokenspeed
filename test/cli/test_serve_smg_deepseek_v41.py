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
import socket
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

import pytest


def _serve_fake_engine():
    """The CPU fake engine (``test.cli._fixtures.fake_engine``) answering as a
    V4.1 checkpoint: "OK", after stub reasoning when the prompt opens a think
    block. It records each request's prompt ids, and its multimodal items as
    JSON, beside ``TOKENSPEED_V41_TEST_CAPTURE``."""
    from test.cli._fixtures.fake_engine import serve

    from tokenizers import Tokenizer

    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    args, _ = parser.parse_known_args()
    tokenizer = Tokenizer.from_file(str(Path(args.model) / "tokenizer.json"))
    capture = Path(os.environ["TOKENSPEED_V41_TEST_CAPTURE"])

    def respond(request):
        prompt = request.input_ids_unpadded or request.input_ids
        if request.multimodal_inputs is not None:
            items = [
                {
                    "feature_shape": list(item.feature.shape),
                    "model_specific_data": {
                        name: tensor.flatten().tolist()
                        for name, tensor in item.model_specific_data.items()
                    },
                    "offsets": item.offsets,
                    "pad_value": item.pad_value,
                }
                for item in request.multimodal_inputs.mm_items
            ]
            capture.with_suffix(".mm.json").write_text(
                json.dumps(
                    {
                        "items": items,
                        "input_ids": request.input_ids,
                        "input_ids_unpadded": prompt,
                    }
                )
            )
        thinking = prompt[-1] == tokenizer.token_to_id("<think>")
        output = tokenizer.encode(
            "stub reasoning.</think>OK" if thinking else "OK",
            add_special_tokens=False,
        ).ids
        with capture.open("a", encoding="utf-8") as f:
            f.write(json.dumps(list(prompt)) + "\n")
        return output

    serve(respond=respond, max_model_len=16384)


@pytest.mark.parametrize(
    "model",
    ["deepseek-ai/DeepSeek-V4.1-Flash", "DeepSeek_V4_1_Flash", "DeepSeekV41-Flash"],
)
def test_v41_selects_v41_parsers(model):
    from tokenspeed.cli import serve_smg as serve

    assert serve._is_deepseek_v41_model(model)
    engine, gateway = serve._args_with_default_model_parsers(
        ["--model", model], ["--model", model]
    )
    assert serve._get_from_args(engine, "--reasoning-parser", None) == "deepseek_v41"
    assert serve._get_from_args(gateway, "--tool-call-parser", None) == "deepseek_v41"


@pytest.mark.parametrize(
    "config",
    [{"model_type": "deepseek_v41"}, {"architectures": ["DeepseekV41ForCausalLM"]}],
)
def test_v41_detects_renamed_local_snapshot(tmp_path, config):
    from tokenspeed.cli import serve_smg as serve

    (tmp_path / "config.json").write_text(json.dumps(config))
    assert serve._is_deepseek_v41_model(str(tmp_path))


def test_v41_preserves_explicit_tool_parser():
    from tokenspeed.cli import serve_smg as serve

    _, gateway = serve._args_with_default_model_parsers(
        ["--model", "DeepSeek-V4.1-Flash"], ["--tool-call-parser", "passthrough"]
    )
    assert serve._get_from_args(gateway, "--tool-call-parser", None) == "passthrough"


@pytest.mark.parametrize("custom_template", [False, True])
def test_serve_uses_native_renderer_or_explicit_template(
    monkeypatch, tmp_path, custom_template
):
    from tokenspeed.cli import serve_smg as serve

    model = str(tmp_path)
    (tmp_path / "config.json").write_text(json.dumps({"model_type": "deepseek_v41"}))
    captured = []

    async def run(**kwargs):
        path = serve._get_from_args(kwargs["gateway_args"], "--chat-template", None)
        captured.append(path)
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
    assert captured == ["operator-template.jinja" if custom_template else None]


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
    pytest.importorskip("zmq")
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
                "--served-model-name",
                "v41-text-test",
                "--trust-remote-code",
                "--node-rank",
                "0",
                "--host",
                "127.0.0.1",
                "--port",
                str(port),
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
    {"enable_thinking": True, "reasoning_effort": "low"},
    {"enable_thinking": True, "reasoning_effort": 37},
    {"enable_thinking": True, "reasoning_effort": 1},
    {"enable_thinking": True, "reasoning_effort": 100},
    {"enable_thinking": True, "reasoning_effort": "high"},
    {"enable_thinking": True, "reasoning_effort": "max", "drop_thinking": False},
]


@pytest.mark.parametrize(
    ("messages", "options", "top_level_effort"),
    [
        (messages, {"enable_thinking": thinking}, None)
        for messages in _MESSAGES
        for thinking in (False, True)
    ]
    + [(_MESSAGES[2], options, None) for options in _OPTIONS]
    + [(_MESSAGES[0], {}, effort) for effort in ("low", "high", "max")],
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
    thinking = options.get("enable_thinking", True)
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
    # SMG and the checkpoint use different names for the same numeric budgets.
    effort = options.get("reasoning_effort", top_level_effort)
    effort = {None: 50, "low": 25, "high": 50, "xhigh": 75, "max": 100}.get(
        effort, effort
    )
    expected = encoder.encode_messages(
        messages,
        thinking_mode="thinking" if thinking else "chat",
        context=None,
        drop_thinking=options.get("drop_thinking", True),
        add_default_bos_token=True,
        reasoning_effort=effort,
        return_multi_modal_data=False,
    )
    actual_ids = json.loads(capture.read_text().splitlines()[-1])
    assert actual_ids == tokenizer.encode(expected, add_special_tokens=False).ids
    assert actual_ids.count(tokenizer.token_to_id("<｜begin▁of▁sentence｜>")) == 1


@pytest.mark.parametrize(
    "extra",
    [
        {"chat_template_kwargs": {"enable_thinking": True, "reasoning_effort": effort}}
        for effort in (0, 101, True)
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
    assert "reasoning_effort" in detail, detail
    assert capture.read_text() == before


def test_http_image_reaches_engine_in_v41_format(gateway):
    import base64
    import io

    from PIL import Image

    url, capture = gateway
    pixels = io.BytesIO()
    Image.new("RGB", (48, 32), color=(220, 30, 80)).save(pixels, format="PNG")
    image_url = "data:image/png;base64," + base64.b64encode(pixels.getvalue()).decode()
    body = {
        "model": "v41-text-test",
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "Describe this image."},
                    {"type": "image_url", "image_url": {"url": image_url}},
                ],
            }
        ],
        "max_tokens": 8,
        "chat_template_kwargs": {"enable_thinking": False},
    }
    request = urllib.request.Request(
        url + "/v1/chat/completions",
        data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(request, timeout=30) as response:
        assert json.load(response)["choices"][0]["message"]["content"] == "OK"
    generated = json.loads(capture.with_suffix(".mm.json").read_text())
    assert len(generated["items"]) == 1
    item = generated["items"][0]
    assert item["feature_shape"][0] > 0
    assert item["feature_shape"][1:] == [3, 14, 14]
    metadata = item["model_specific_data"]
    assert {"vit_grid", "llm_grid", "types"} <= metadata.keys()
    height, width = metadata["llm_grid"]
    types = metadata["types"]
    assert types == [0] + ([1] * width + [2]) * height + [3]
    # One inclusive span of image tokens, which the wire carries as the
    # item's pad value.
    [(start, end)] = item["offsets"]
    assert end - start + 1 == len(types)
    assert generated["input_ids_unpadded"][start : end + 1] == [129264] * len(types)
    assert generated["input_ids"][start : end + 1] == [item["pad_value"]] * len(types)


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
