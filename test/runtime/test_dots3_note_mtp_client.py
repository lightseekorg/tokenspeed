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

"""Opt-in client for an already-running dots3-note MTP control sidecar.

Set DOTS3_MTP_URL to the sidecar root (e.g. http://127.0.0.1:18001),
DOTS3_MTP_MODEL to the served model ID, and DOTS3_MTP_TOKENIZER to matching
local tokenizer assets or an already-cached HF ID. From the repo root run:
  python -m pytest --noconftest -q -s test/runtime/test_dots3_note_mtp_client.py

Requires pytest and transformers/tokenizers. Uses only the sidecar's /generate
with rendered token IDs; the sidecar unwraps SMG's singleton response list.
No downloads, remote tokenizer code, runtime/kernel imports, server lifecycle
changes, or cache flush. --noconftest avoids the parent CUDA availability probe.
Each HTTP socket operation has a 180-second timeout. Any DOTS3_MTP_* setting
opts in; incomplete/unsupported settings and missing telemetry fail, not skip.

Requires MTP >=3 proposal steps, width=steps+1, prefix caching at grain64, and
context room for >max(2048, chunked_prefill_size) input plus 128 output tokens.
Run without concurrent cache pressure. Seven generation requests check strict
outputs and warm/cold cache safety, not acceptance rate, PD, or kernel dispatch.
"""

from __future__ import annotations

import json
import os
import urllib.error
import urllib.parse
import urllib.request
from uuid import uuid4

import pytest

_MARKER = "__DOTS3_CLIENT_PADDING__"
_MAX_TOKENS = 128


def _settings(env):
    options = {
        key.removeprefix("DOTS3_MTP_"): value
        for key, value in env.items()
        if key.startswith("DOTS3_MTP_")
    }
    if not options:
        pytest.skip("Set DOTS3_MTP_URL/MODEL/TOKENIZER to opt in")
    assert set(options) == {"URL", "MODEL", "TOKENIZER"} and all(
        value.strip() for value in options.values()
    ), "Set exactly DOTS3_MTP_URL, DOTS3_MTP_MODEL, DOTS3_MTP_TOKENIZER"
    url = options["URL"].rstrip("/")
    parsed = urllib.parse.urlsplit(url)
    assert parsed.scheme in ("http", "https") and parsed.netloc
    assert not (
        parsed.path or parsed.query or parsed.fragment
    ), "Use the sidecar root URL"
    return url, options["MODEL"], options["TOKENIZER"]


def _http(url, *, body):
    request = urllib.request.Request(
        url,
        data=None if body is None else json.dumps(body).encode(),
        headers={"Content-Type": "application/json"},
    )
    try:
        with urllib.request.urlopen(request, timeout=180) as response:
            data = json.load(response)
    except urllib.error.HTTPError as exc:
        pytest.fail(f"HTTP {exc.code} at {url}: {exc.read().decode(errors='replace')}")
    assert isinstance(data, dict) and not data.get("error"), data
    return data


def _complete(prompt, *, url, model, label):
    data = _http(
        f"{url}/generate",
        body={
            "model": model,
            "input_ids": prompt,
            "stream": False,
            "sampling_params": {
                "max_new_tokens": _MAX_TOKENS,
                "temperature": 0,
                "top_p": 1,
                "seed": 42,
            },
        },
    )
    text, usage = data["text"], data["meta_info"]
    finish = usage["finish_reason"]
    if isinstance(finish, dict):
        finish = finish["type"]
    cached = usage["cached_tokens"]
    assert isinstance(text, str) and text.strip(), data
    assert finish == "stop", data
    assert usage["prompt_tokens"] == len(prompt), "Token IDs changed or were truncated"
    assert 0 < usage["completion_tokens"] <= _MAX_TOKENS, data
    assert isinstance(cached, int) and 0 <= cached <= len(prompt), data
    print(
        f"{label}: prompt={len(prompt)} cached={cached} "
        f"output={usage['completion_tokens']} text={text!r}"
    )
    return text, usage["completion_tokens"], cached


def _render(tokenizer, content):
    return tokenizer.apply_chat_template(
        [{"role": "user", "content": content}],
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False,
    )


def _padding_parts(tokenizer, instruction):
    # A fresh ID inside the first grain isolates runs without flushing others' KV.
    rendered = _render(
        tokenizer, f"ID {uuid4().hex}\nPadding: {_MARKER}\n{instruction}"
    )
    assert rendered.count(_MARKER) == 1, "Tokenizer template changed the user content"
    before, after = rendered.split(_MARKER)
    prefix, suffix, filler = [
        tokenizer.encode(text, add_special_tokens=False)
        for text in (before, after, " x")
    ]
    assert 0 < len(prefix) < 64, "Tokenizer template leaves no room in the first grain"
    assert len(filler) == 1 and filler[0] not in tokenizer.all_special_ids
    return prefix, suffix, filler


def _divergent_prompts(tokenizer):
    prefix, suffix, filler = _padding_parts(
        tokenizer,
        "Repeat the last word on the Padding line sixteen times, separated by "
        "single spaces. Output only those words.",
    )
    prefix += filler * (64 - len(prefix))
    branches = [
        tokenizer.encode(word, add_special_tokens=False) for word in (" red", " blue")
    ]
    assert all(
        len(ids) == 1 and ids[0] not in tokenizer.all_special_ids for ids in branches
    )
    prompts = [prefix + ids + suffix for ids in branches]
    assert prompts[0][:64] == prompts[1][:64] and prompts[0][64] != prompts[1][64]
    assert prompts[0][65:] == prompts[1][65:]
    return prompts


def test_running_dots3_note_mtp():
    url, model, tokenizer_path = _settings(os.environ)
    info = _http(f"{url}/get_server_info", body=None)
    args = info["server_args"]
    assert args["speculative_algorithm"] == "MTP", "Requires native MTP"
    steps = int(args["speculative_num_steps"])
    width = int(args["speculative_num_draft_tokens"])
    assert steps >= 3 and width == steps + 1, "Requires >=3 proposals plus anchor"
    assert args["enable_prefix_caching"] is True and args["prefix_granularity"] == 64
    chunk = int(args["chunked_prefill_size"])
    assert chunk > 0, "Requires chunked prefill"
    model_info = _http(f"{url}/get_model_info", body=None)
    assert model_info["model_type"] == "dots3_note"
    assert model_info["served_model_name"] == model
    long_length = max(2049, chunk + 64)
    assert long_length + _MAX_TOKENS <= int(
        model_info["max_context_length"]
    ), "Reduce the server's chunked-prefill-size to leave room for a multi-chunk prompt"
    assert long_length <= int(model_info["max_req_input_len"])
    print(f"MTP proposals={steps} width={width} grain=64 chunk={chunk}")

    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(
        tokenizer_path,
        local_files_only=True,
        trust_remote_code=False,
    )
    assert (
        tokenizer.chat_template
    ), "Supply the checkpoint's actual chat-template assets"

    def complete(prompt, label):
        return _complete(prompt, url=url, model=model, label=label)

    short = tokenizer.encode(
        _render(
            tokenizer, "What is 17 + 25? Reply with only the number, no other text."
        ),
        add_special_tokens=False,
    )
    assert complete(short, "short-greedy")[0].strip() == "42"

    prefix, suffix, filler = _padding_parts(
        tokenizer,
        "Ignore the ID and padding. List the integers from 1 through 16 "
        "in order, separated by single spaces. Output only the numbers.",
    )
    long_prompt = prefix + filler * (long_length - len(prefix) - len(suffix)) + suffix
    assert len(long_prompt) == long_length
    cold = complete(long_prompt, "long-cold-chunked")
    assert cold[0].split() == [str(i) for i in range(1, 17)]
    assert (
        cold[1] > width and cold[2] == 0
    ), "Need cold multi-chunk prefill and multiple verify rounds"
    warm = complete(long_prompt, "long-repeat")
    assert (
        warm[:2] == cold[:2] and warm[2] >= 64
    ), "Repeated prompt must actually hit cache"

    prompts = _divergent_prompts(tokenizer)
    cold_branches = []
    for prompt, word in zip(prompts, ("red", "blue")):
        result = complete(prompt, f"divergent-{word}-cold")
        assert result[0].split() == [word] * 16 and result[1] > width
        # MTP row 63 consumes x[64]. A different x[64] must NOT reuse that grain,
        # even if target KV alone would safely share the first 64 tokens.
        assert result[2] == 0, "Unsafe grain-64 reuse across different next tokens"
        cold_branches.append(result)
    for prompt, word, cold in zip(prompts, ("red", "blue"), cold_branches):
        warm = complete(prompt, f"divergent-{word}-repeat")
        assert (
            warm[:2] == cold[:2] and warm[2] >= 64
        ), "Warm/cold divergence or no cache hit"


def test_client_contract_offline(monkeypatch):
    """Check opt-in and the sidecar wire contract without sockets or assets."""
    with pytest.raises(pytest.skip.Exception):
        _settings({})
    with pytest.raises(AssertionError, match="Set exactly"):
        _settings({"DOTS3_MTP_URL": "http://example.invalid"})
    url, model, _ = _settings(
        {
            "DOTS3_MTP_URL": "http://example.invalid",
            "DOTS3_MTP_MODEL": "dots3-note",
            "DOTS3_MTP_TOKENIZER": "unused",
        }
    )
    usage = {
        "prompt_tokens": 2,
        "completion_tokens": 1,
        "cached_tokens": 0,
        "finish_reason": {"type": "stop"},
    }

    def fake_http(endpoint, *, body):
        assert endpoint == f"{url}/generate"
        assert body["model"] == model and body["input_ids"] == [10, 11]
        assert body["stream"] is False and body["sampling_params"]["temperature"] == 0
        return {"text": "42", "meta_info": usage}

    monkeypatch.setitem(globals(), "_http", fake_http)
    assert _complete([10, 11], url=url, model=model, label="offline") == ("42", 1, 0)
    del usage["cached_tokens"]
    with pytest.raises(KeyError, match="cached_tokens"):
        _complete([10, 11], url=url, model=model, label="missing-telemetry")
