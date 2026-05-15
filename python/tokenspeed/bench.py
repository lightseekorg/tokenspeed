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

r"""Benchmark online serving throughput.

On the server side, launch a TokenSpeed OpenAI-compatible API server:
    tokenspeed serve --model <your_model> <engine arguments>

On the client side, run:
    tokenspeed bench serve \
        --backend <backend or endpoint type. Default 'openai'> \
        --label <benchmark result label. Default using backend> \
        --model <your_model. Optional, defaults to first model from server> \
        --dataset-name <dataset_name. Default 'random'> \
        --input-len <general input length. Optional, maps to dataset-specific args> \
        --output-len <general output length. Optional, maps to dataset-specific args> \
        --request-rate <request_rate. Default inf> \
        --num-prompts <num_prompts. Default 1000>
"""

from __future__ import annotations

import argparse
import asyncio
import codecs
import contextlib
import json
import logging
import math
import os
import random
import resource
import ssl
import sys
import time
import traceback
import warnings
from collections.abc import AsyncGenerator, Coroutine
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Literal
from urllib.parse import urlparse

import aiohttp
import numpy as np
import requests
from tqdm.asyncio import tqdm
from transformers import AutoTokenizer, PreTrainedTokenizerBase

from tokenspeed.runtime.utils.env import envs

# Streaming HTTP timeouts. ``total=6h`` keeps the session umbrella generous so
# whole-run benches don't get cut off; the per-socket sub-timeouts catch a
# legitimately stuck stream without false-failing slow legitimate prefills.
#
# ``sock_read`` defaults to 30 minutes — well above the largest TTFT one would
# expect on real hardware (a 64k-context prefill on a single-GPU consumer card
# is still well under 10 minutes) yet far below ``total``, so an indefinitely
# silent socket still surfaces as a ``aiohttp.ServerTimeoutError`` rather than
# blocking the outer ``asyncio.gather`` at high concurrency. Long-haul or
# pathologically large prefill workloads can bump it via env. ``sock_connect``
# is the dial-tone timeout for the TCP handshake itself.
AIOHTTP_TOTAL_TIMEOUT_SEC = float(
    os.environ.get("TOKENSPEED_BENCH_TOTAL_TIMEOUT_SEC", str(6 * 60 * 60))
)
AIOHTTP_SOCK_CONNECT_TIMEOUT_SEC = float(
    os.environ.get("TOKENSPEED_BENCH_SOCK_CONNECT_TIMEOUT_SEC", "30")
)
AIOHTTP_SOCK_READ_TIMEOUT_SEC = float(
    os.environ.get("TOKENSPEED_BENCH_SOCK_READ_TIMEOUT_SEC", str(30 * 60))
)
AIOHTTP_TIMEOUT = aiohttp.ClientTimeout(
    total=AIOHTTP_TOTAL_TIMEOUT_SEC,
    sock_connect=AIOHTTP_SOCK_CONNECT_TIMEOUT_SEC,
    sock_read=AIOHTTP_SOCK_READ_TIMEOUT_SEC,
)
# Per-request hard ceiling so a single misbehaving stream cannot block the
# whole gather. 1h is generous enough for the longest practical decode and
# still bounded for CI / smoke benches. Override via env when running
# unusually long sequences.
PER_REQUEST_TIMEOUT_SEC = float(
    os.environ.get("TOKENSPEED_BENCH_PER_REQUEST_TIMEOUT_SEC", str(60 * 60))
)
DEFAULT_NUM_PROMPTS = 1000
MILLISECONDS_TO_SECONDS_CONVERSION = 1000
SHAREGPT_URL = "https://huggingface.co/datasets/anon8231489123/ShareGPT_Vicuna_unfiltered/resolve/main/ShareGPT_V3_unfiltered_cleaned_split.json"
OPENAI_COMPATIBLE_BACKENDS = frozenset({"openai", "tokenspeed"})
logger = logging.getLogger(__name__)

# Type alias: a single float applies to both ISL and OSL; a dict allows
# specifying them independently via ``{"input": ..., "output": ...}``.
RangeRatio = float | dict[str, float]


def _print_section_header(title: str, fill: str) -> None:
    print(f"{title:{fill}^50}")


def _print_metric_row(label: str, value: Any, precision: int | None = None) -> None:
    formatted_value = (
        f"{value:<10}" if precision is None else f"{value:<10.{precision}f}"
    )
    print(f"{label:<40} {formatted_value}")


class StreamedResponseHandler:
    """Accumulate SSE bytes until complete `data:` messages are available."""

    def __init__(self) -> None:
        self.buffer = ""
        self._decoder = codecs.getincrementaldecoder("utf-8")()

    def add_chunk(self, chunk_bytes: bytes) -> list[str]:
        self.buffer += self._decoder.decode(chunk_bytes)
        messages: list[str] = []

        while "\n\n" in self.buffer:
            message, self.buffer = self.buffer.split("\n\n", 1)
            message = message.strip()
            if message:
                messages.append(message)

        if self.buffer.startswith("data: "):
            message_content = self.buffer.removeprefix("data: ").strip()
            if message_content == "[DONE]":
                messages.append(self.buffer.strip())
                self.buffer = ""
            elif message_content:
                try:
                    json.loads(message_content)
                except json.JSONDecodeError:
                    pass
                else:
                    messages.append(self.buffer.strip())
                    self.buffer = ""

        return messages


@dataclass
class SampleRequest:
    prompt: str
    prompt_len: int
    expected_output_len: int
    multi_modal_data: dict | list[dict] | None = None
    lora_request: Any | None = None
    request_id: str | None = None


@dataclass
class RequestFuncInput:
    """The input for the request function."""

    prompt: str | list[str]
    api_url: str
    prompt_len: int
    output_len: int
    model: str
    model_name: str | None = None
    logprobs: int | None = None
    extra_headers: dict | None = None
    extra_body: dict | None = None
    multi_modal_content: dict | list[dict] | None = None
    ignore_eos: bool = False
    language: str | None = None
    request_id: str | None = None


@dataclass
class RequestFuncOutput:
    """The output of the request function including metrics."""

    generated_text: str = ""
    success: bool = False
    latency: float = 0.0
    output_tokens: int = 0
    ttft: float = 0.0  # Time to first token
    itl: list[float] = field(default_factory=list)  # list of inter-token latencies
    tpot: float = 0.0  # avg next-token latencies
    prompt_len: int = 0
    error: str = ""
    start_time: float = 0.0
    input_audio_duration: float = 0.0  # in seconds


async def await_with_per_request_timeout(
    coro: Coroutine[Any, Any, RequestFuncOutput],
    *,
    prompt_len: int,
    pbar: tqdm | None = None,
) -> RequestFuncOutput:
    """Run a request coroutine under :data:`PER_REQUEST_TIMEOUT_SEC`.

    Wraps the per-request ``asyncio.wait_for`` so a single stuck stream
    cannot deadlock the outer ``asyncio.gather`` in :func:`benchmark`.  On
    :class:`asyncio.TimeoutError`, returns a standard
    :class:`RequestFuncOutput` with ``success=False`` so the gather can
    complete and the metrics output reports the failure normally.
    """
    try:
        return await asyncio.wait_for(coro, timeout=PER_REQUEST_TIMEOUT_SEC)
    except asyncio.TimeoutError:
        output = RequestFuncOutput()
        output.prompt_len = prompt_len
        output.success = False
        output.error = (
            f"per-request timeout {PER_REQUEST_TIMEOUT_SEC:.1f}s "
            "(TOKENSPEED_BENCH_PER_REQUEST_TIMEOUT_SEC)"
        )
        if pbar is not None:
            pbar.update(1)
        return output


class TaskType(Enum):
    GENERATION = "generation"


@dataclass
class BenchmarkMetrics:
    completed: int
    failed: int
    total_input: int
    total_output: int
    request_throughput: float
    request_goodput: float
    output_throughput: float
    total_token_throughput: float
    mean_ttft_ms: float
    median_ttft_ms: float
    std_ttft_ms: float
    percentiles_ttft_ms: list[tuple[float, float]]
    mean_tpot_ms: float
    median_tpot_ms: float
    std_tpot_ms: float
    percentiles_tpot_ms: list[tuple[float, float]]
    mean_itl_ms: float
    median_itl_ms: float
    std_itl_ms: float
    percentiles_itl_ms: list[tuple[float, float]]
    mean_e2el_ms: float
    median_e2el_ms: float
    std_e2el_ms: float
    percentiles_e2el_ms: list[tuple[float, float]]
    max_output_tokens_per_s: float
    max_concurrent_requests: int


def set_ulimit(target_soft_limit: int = 65535) -> None:
    resource_type = resource.RLIMIT_NOFILE
    current_soft, current_hard = resource.getrlimit(resource_type)
    if current_soft < target_soft_limit:
        try:
            resource.setrlimit(resource_type, (target_soft_limit, current_hard))
        except ValueError as e:
            print(f"Fail to set RLIMIT_NOFILE: {e}")


def join_host_port(host: str, port: int) -> str:
    return (
        f"[{host}]:{port}"
        if ":" in host and not host.startswith("[")
        else f"{host}:{port}"
    )


def _validate_api_url(
    api_url: str,
    api_name: str,
    expected_suffixes: str | set[str],
) -> None:
    if isinstance(expected_suffixes, str):
        expected_suffixes = {expected_suffixes}

    expected_suffixes = {*expected_suffixes, "profile"}

    if not api_url.endswith(tuple(expected_suffixes)):
        raise ValueError(f"{api_name} URL must end with one of: {expected_suffixes}.")


def _update_payload_common(
    payload: dict[str, Any],
    request_func_input: RequestFuncInput,
) -> None:
    if request_func_input.ignore_eos:
        payload["ignore_eos"] = request_func_input.ignore_eos
    if request_func_input.extra_body:
        payload.update(request_func_input.extra_body)


def _update_headers_common(
    headers: dict[str, Any],
    request_func_input: RequestFuncInput,
) -> None:
    if request_func_input.extra_headers:
        headers |= request_func_input.extra_headers
    if request_func_input.request_id:
        headers["x-request-id"] = request_func_input.request_id


def _get_headers(content_type: str | None = None) -> dict[str, str]:
    headers = {}
    if content_type:
        headers["Content-Type"] = content_type
    api_key = os.environ.get("OPENAI_API_KEY")
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    return headers


async def async_request_openai_completions(
    request_func_input: RequestFuncInput,
    session: aiohttp.ClientSession,
    pbar: tqdm | None = None,
) -> RequestFuncOutput:
    """The async request function for the OpenAI Completions API.

    Args:
        request_func_input: The input for the request function.
        pbar: The progress bar to display the progress.

    Returns:
        The output of the request function.
    """
    api_url = request_func_input.api_url
    _validate_api_url(api_url, "OpenAI Completions API", "completions")

    payload = {
        "model": (
            request_func_input.model_name
            if request_func_input.model_name
            else request_func_input.model
        ),
        "prompt": request_func_input.prompt,
        "repetition_penalty": 1.0,
        "max_tokens": request_func_input.output_len,
        "logprobs": request_func_input.logprobs,
        "stream": True,
        "stream_options": {
            "include_usage": True,
        },
    }
    _update_payload_common(payload, request_func_input)

    headers = _get_headers()
    _update_headers_common(headers, request_func_input)

    output = RequestFuncOutput()
    output.prompt_len = request_func_input.prompt_len

    generated_text = ""
    st = time.perf_counter()
    output.start_time = st
    most_recent_timestamp = st
    try:
        async with session.post(url=api_url, json=payload, headers=headers) as response:
            if response.status == 200:
                first_chunk_received = False
                handler = StreamedResponseHandler()

                async for chunk_bytes in response.content.iter_any():
                    chunk_bytes = chunk_bytes.strip()
                    if not chunk_bytes:
                        continue

                    messages = handler.add_chunk(chunk_bytes)
                    for message in messages:
                        if message.startswith(":"):
                            continue

                        chunk = message.removeprefix("data: ")

                        if chunk != "[DONE]":
                            data = json.loads(chunk)

                            if choices := data.get("choices"):
                                text = choices[0].get("text")
                                timestamp = time.perf_counter()
                                if not first_chunk_received:
                                    first_chunk_received = True
                                    ttft = time.perf_counter() - st
                                    output.ttft = ttft
                                else:
                                    output.itl.append(timestamp - most_recent_timestamp)

                                most_recent_timestamp = timestamp
                                generated_text += text or ""
                            elif usage := data.get("usage"):
                                output.output_tokens = usage.get("completion_tokens")
                                if (pt := usage.get("prompt_tokens")) is not None:
                                    output.prompt_len = pt
                if first_chunk_received:
                    output.success = True
                else:
                    output.success = False
                    output.error = (
                        "Never received a valid chunk to calculate TTFT."
                        "This response will be marked as failed!"
                    )
                output.generated_text = generated_text
                output.latency = most_recent_timestamp - st
            else:
                output.error = response.reason or ""
                output.success = False
    except Exception:
        output.success = False
        exc_info = sys.exc_info()
        output.error = "".join(traceback.format_exception(*exc_info))

    if pbar:
        pbar.update(1)
    return output


def _get_chat_content(
    request_func_input: RequestFuncInput,
    mm_position: Literal["first", "last"] = "last",
) -> list[dict[str, Any]]:
    text_contents = [{"type": "text", "text": request_func_input.prompt}]

    mm_contents = []
    if request_func_input.multi_modal_content:
        mm_content = request_func_input.multi_modal_content
        if isinstance(mm_content, list):
            mm_contents.extend(request_func_input.multi_modal_content)
        elif isinstance(mm_content, dict):
            mm_contents.append(request_func_input.multi_modal_content)
        else:
            raise TypeError(
                "multi_modal_content must be a dict or list[dict] for openai-chat"
            )

    if mm_position == "first":
        return mm_contents + text_contents

    return text_contents + mm_contents


async def async_request_openai_chat_completions(
    request_func_input: RequestFuncInput,
    session: aiohttp.ClientSession,
    pbar: tqdm | None = None,
    mm_position: Literal["first", "last"] = "last",
) -> RequestFuncOutput:
    api_url = request_func_input.api_url
    _validate_api_url(api_url, "OpenAI Chat Completions API", "chat/completions")

    content = _get_chat_content(request_func_input, mm_position=mm_position)

    payload = {
        "model": (
            request_func_input.model_name
            if request_func_input.model_name
            else request_func_input.model
        ),
        "messages": [
            {"role": "user", "content": content},
        ],
        "max_completion_tokens": request_func_input.output_len,
        "stream": True,
        "stream_options": {
            "include_usage": True,
        },
    }
    _update_payload_common(payload, request_func_input)

    headers = _get_headers("application/json")
    _update_headers_common(headers, request_func_input)

    output = RequestFuncOutput()
    output.prompt_len = request_func_input.prompt_len

    generated_text = ""
    ttft = 0.0
    st = time.perf_counter()
    output.start_time = st
    most_recent_timestamp = st
    try:
        async with session.post(url=api_url, json=payload, headers=headers) as response:
            if response.status == 200:
                handler = StreamedResponseHandler()
                async for chunk_bytes in response.content.iter_any():
                    chunk_bytes = chunk_bytes.strip()
                    if not chunk_bytes:
                        continue

                    messages = handler.add_chunk(chunk_bytes)
                    for message in messages:
                        if message.startswith(":"):
                            continue

                        chunk = message.removeprefix("data: ")

                        if chunk != "[DONE]":
                            timestamp = time.perf_counter()
                            data = json.loads(chunk)

                            if choices := data.get("choices"):
                                content = choices[0]["delta"].get("content")
                                if ttft == 0.0:
                                    ttft = timestamp - st
                                    output.ttft = ttft
                                else:
                                    output.itl.append(timestamp - most_recent_timestamp)

                                generated_text += content or ""
                            elif usage := data.get("usage"):
                                output.output_tokens = usage.get("completion_tokens")
                                if (pt := usage.get("prompt_tokens")) is not None:
                                    output.prompt_len = pt

                            most_recent_timestamp = timestamp

                output.generated_text = generated_text
                output.success = True
                output.latency = most_recent_timestamp - st
            else:
                output.error = response.reason or ""
                output.success = False
    except Exception:
        output.success = False
        exc_info = sys.exc_info()
        output.error = "".join(traceback.format_exception(*exc_info))

    if pbar:
        pbar.update(1)
    return output


ASYNC_REQUEST_FUNCS = {
    "openai": async_request_openai_completions,
    "tokenspeed": async_request_openai_completions,
    "openai-chat": async_request_openai_chat_completions,
}


def get_model(pretrained_model_name_or_path: str) -> str:
    if envs.TOKENSPEED_USE_MODELSCOPE.get():
        import huggingface_hub.constants
        from modelscope import snapshot_download

        return snapshot_download(
            model_id=pretrained_model_name_or_path,
            local_files_only=huggingface_hub.constants.HF_HUB_OFFLINE,
            ignore_file_pattern=[".*.pt", ".*.safetensors", ".*.bin"],
        )
    return pretrained_model_name_or_path


def get_tokenizer(
    pretrained_model_name_or_path: str,
) -> PreTrainedTokenizerBase:
    if pretrained_model_name_or_path is not None and not os.path.exists(
        pretrained_model_name_or_path
    ):
        pretrained_model_name_or_path = get_model(pretrained_model_name_or_path)
    return AutoTokenizer.from_pretrained(
        pretrained_model_name_or_path, trust_remote_code=True
    )


def download_and_cache_file(url: str, filename: str | None = None) -> str:
    if filename is None:
        filename = os.path.join("/tmp", os.path.basename(urlparse(url).path))
    if os.path.exists(filename):
        return filename

    print(f"Downloading from {url} to {filename}")
    response = requests.get(url, stream=True)
    response.raise_for_status()
    total_size = int(response.headers.get("content-length", 0))
    with open(filename, "wb") as f, tqdm(
        desc=filename,
        total=total_size,
        unit="B",
        unit_scale=True,
        unit_divisor=1024,
    ) as bar:
        for chunk in response.iter_content(chunk_size=1024):
            f.write(chunk)
            bar.update(len(chunk))
    return filename


def is_valid_sequence(
    prompt_len: int,
    output_len: int,
    max_model_len: int | None,
    skip_min_tokens_check: bool,
) -> bool:
    if not skip_min_tokens_check and (prompt_len < 4 or output_len < 4):
        return False
    if max_model_len is not None and prompt_len + output_len > max_model_len:
        return False
    return True


def _resolve_range_ratios(
    range_ratio: RangeRatio,
) -> tuple[float, float]:
    """Return ``(input_range_ratio, output_range_ratio)`` from *range_ratio*.

    *range_ratio* is either a single float (used for both input and output)
    or a dict with ``"input"`` and ``"output"`` keys.
    """
    if isinstance(range_ratio, dict):
        try:
            return float(range_ratio["input"]), float(range_ratio["output"])
        except KeyError as exc:
            raise ValueError(
                "When range_ratio is a dict it must contain 'input' and "
                f"'output' keys, got: {sorted(range_ratio)}"
            ) from exc
    ratio = float(range_ratio)
    return ratio, ratio


def get_sampling_params(
    rng: np.random.Generator,
    num_requests: int,
    range_ratio: RangeRatio,
    input_len: int,
    output_len: int,
    tokenizer: PreTrainedTokenizerBase,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Sample per-request input/output token lengths and vocab offsets.

    Lengths are drawn uniformly from integer ranges around the configured
    means, controlled by *range_ratio*.  It may be a single ``float``
    (applied to both input and output) or a ``dict`` with ``"input"`` and
    ``"output"`` keys for independent control.

    Tokenizer special tokens are subtracted from ``input_len`` before
    computing the sampling interval.

    Returns:
        (input_lens, output_lens, offsets) - three 1-D ``np.ndarray`` of
        shape ``(num_requests,)``.
    """
    input_range_ratio, output_range_ratio = _resolve_range_ratios(range_ratio)

    if not (0.0 <= input_range_ratio < 1.0):
        raise ValueError("input_range_ratio must be in [0, 1).")
    if not (0.0 <= output_range_ratio < 1.0):
        raise ValueError("output_range_ratio must be in [0, 1).")
    num_special_tokens = int(tokenizer.num_special_tokens_to_add())
    real_input_len = max(0, int(input_len) - num_special_tokens)
    input_low = math.floor(real_input_len * (1 - input_range_ratio))
    input_high = math.ceil(real_input_len * (1 + input_range_ratio))
    output_low = math.floor(output_len * (1 - output_range_ratio))
    output_high = math.ceil(output_len * (1 + output_range_ratio))
    # Ensure the lower bound for output length is at least 1 to
    # prevent sampling 0 tokens.
    output_low = max(output_low, 1)
    output_high = max(output_high, 1)

    if input_low > input_high:
        raise ValueError(
            f"Invalid input sampling interval: low={input_low} > high={input_high}"
        )
    if output_low > output_high:
        raise ValueError(
            f"Invalid output sampling interval: low={output_low} > high={output_high}"
        )

    logger.info(
        "Sampling input_len from [%s, %s] and output_len from [%s, %s]",
        input_low,
        input_high,
        output_low,
        output_high,
    )

    input_lens = rng.integers(input_low, input_high + 1, size=num_requests)
    output_lens = rng.integers(output_low, output_high + 1, size=num_requests)
    offsets = rng.integers(0, tokenizer.vocab_size, size=num_requests)
    return input_lens, output_lens, offsets


def gen_prompt_decode_to_target_len(
    tokenizer: PreTrainedTokenizerBase,
    token_sequence: list[int],
    target_token_len: int,
    max_retry: int = 10,
    add_special_tokens: bool = False,
    rng: np.random.Generator | None = None,
) -> tuple[str, list[int], int]:
    """
    Ensure decoded-then-encoded prompt length matches the target token length.

    This function decodes an initial token sequence to text and re-encodes it
    , iteratively adjusting the token sequence length to match a target.
    This is necessary because some tokenizers do not guarantee a 1:1 mapping
    between consecutive tokens and the decoded-then-encoded sequence length.
    For example, for GPT2Tokenizer:
    [6880, 6881] -> ['Ġcalls', 'here'] ->
    [1650, 939, 486] -> ['Ġcall', 'sh', 'ere']

    Returns a tuple of the final prompt string, the adjusted token sequence,
    and the token mismatch (final_len - target_token_len) if the retry budget
    is exhausted.
    """
    remain_num_try = max_retry
    token_mismatch = 0
    while True:
        prompt = tokenizer.decode(token_sequence)
        token_sequence = tokenizer.encode(prompt, add_special_tokens=add_special_tokens)
        if remain_num_try <= 0:
            if len(token_sequence) != target_token_len:
                token_mismatch = len(token_sequence) - target_token_len
            break

        if len(token_sequence) == target_token_len:
            break
        elif len(token_sequence) < target_token_len:
            if rng is not None:
                extra_tokens = rng.integers(
                    0,
                    tokenizer.vocab_size,
                    size=target_token_len - len(token_sequence),
                ).tolist()
            else:
                extra_tokens = np.random.randint(
                    0,
                    tokenizer.vocab_size,
                    size=target_token_len - len(token_sequence),
                ).tolist()
            token_sequence.extend(extra_tokens)
        elif len(token_sequence) > target_token_len:
            token_sequence = token_sequence[:target_token_len]

        remain_num_try -= 1

    return prompt, token_sequence, token_mismatch


class BenchmarkDataset:
    DEFAULT_SEED = 0

    def __init__(
        self,
        dataset_path: str | None = None,
        random_seed: int = DEFAULT_SEED,
        disable_shuffle: bool = False,
        **kwargs,
    ) -> None:
        """
        Initialize the BenchmarkDataset with an optional dataset path and random
        seed.
        """
        self.dataset_path = dataset_path
        self.random_seed = random_seed if random_seed is not None else self.DEFAULT_SEED
        self.disable_shuffle = disable_shuffle
        self.data: Any | None = None

    def get_lora_request(
        self,
        index: int,
        max_loras: int | None = None,
        lora_path: str | None = None,
        lora_assignment: str = "random",
    ) -> None:
        return None


# fmt: off
class RandomDataset(BenchmarkDataset):
    """
    Synthetic text-only dataset for serving/throughput benchmarks.

    Strategy:
    - Sample input/output token lengths per request from integer-uniform ranges
      around configured means (controlled by range_ratio).
    - Prepend a fixed random prefix of length prefix_len.
    - Generate the remaining tokens as a reproducible sequence:
      (offset + index + arange(input_len)) % vocab_size.
    - Decode then re-encode/truncate to ensure prompt token counts match.
    - Uses numpy.default_rng seeded with random_seed for reproducible sampling.
    """

    DEFAULT_PREFIX_LEN = 0
    DEFAULT_RANGE_RATIO = 0.0
    DEFAULT_INPUT_LEN = 1024
    DEFAULT_OUTPUT_LEN = 128

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        # Use numpy's default_rng for deterministic sampling
        # Do not use random.seed() or np.random.seed() elsewhere in this class.
        # This ensures that the RNG is isolated from global RNG state.
        self._rng = np.random.default_rng(self.random_seed)

    def sample(
        self,
        tokenizer: PreTrainedTokenizerBase,
        num_requests: int,
        request_id_prefix: str = "",
        no_oversample: bool = False,
        prefix_len: int = DEFAULT_PREFIX_LEN,
        range_ratio: RangeRatio = DEFAULT_RANGE_RATIO,
        input_len: int = DEFAULT_INPUT_LEN,
        output_len: int = DEFAULT_OUTPUT_LEN,
        batchsize: int = 1,
        max_loras: int | None = None,
        lora_path: str | None = None,
        lora_assignment: str = "random",
        **kwargs,
    ) -> list[SampleRequest]:
        resolved_input_rr, _ = _resolve_range_ratios(range_ratio)

        num_special = int(tokenizer.num_special_tokens_to_add())
        real_input_len = max(0, int(input_len) - num_special)
        min_sampled_input = math.floor(
            real_input_len * (1.0 - float(resolved_input_rr))
        )
        min_total_input = int(prefix_len) + min_sampled_input
        if min_total_input < 1:
            raise ValueError(
                "--random-input-len is too small: with tokenizer special "
                f"tokens {num_special} and "
                f"input range ratio {resolved_input_rr}, "
                "the minimum possible total input tokens (prefix + sampled) is "
                f"{min_total_input}. Increase --random-input-len and/or "
                "--random-prefix-len, or decrease the input range ratio "
                "so that prefix_len + floor(max(0, random_input_len - "
                "num_special)) * (1 - input_range_ratio) >= 1."
            )

        input_lens, output_lens, offsets = get_sampling_params(
            self._rng,
            num_requests,
            range_ratio,
            input_len,
            output_len,
            tokenizer,
        )

        vocab_size = tokenizer.vocab_size
        prohibited_tokens = tokenizer.all_special_ids
        all_tokens = np.arange(vocab_size)
        allowed_tokens = np.array(list(set(all_tokens) - set(prohibited_tokens)))

        # Generate prefix once
        prefix_token_ids = self.get_prefix(tokenizer, allowed_tokens, prefix_len)

        requests = []
        token_mismatch_total = 0
        for i in range(num_requests):
            prompt, total_input_len, token_mismatch = self.generate_token_sequence(  # noqa: E501
                tokenizer=tokenizer,
                prefix_token_ids=prefix_token_ids,
                prefix_len=prefix_len,
                vocab_size=vocab_size,
                input_len=int(input_lens[i]),
                offset=int(offsets[i]),
                index=i,
                allowed_tokens=allowed_tokens,
            )
            token_mismatch_total += token_mismatch
            lora_req = self.get_lora_request(
                index=i,
                max_loras=max_loras,
                lora_path=lora_path,
                lora_assignment=lora_assignment,
            )
            requests.append(
                SampleRequest(
                    prompt=prompt,
                    prompt_len=total_input_len,
                    expected_output_len=int(output_lens[i]),
                    lora_request=lora_req,
                    request_id=request_id_prefix + str(i),
                )
            )
        # only used for embeddings benchmark.
        if batchsize > 1:
            batch_requests = []
            # Create batched requests
            for i in range(0, num_requests, batchsize):
                batch = requests[i : i + batchsize]
                batch_requests.append(
                    SampleRequest(
                        prompt=[req.prompt for req in batch],
                        prompt_len=sum(req.prompt_len for req in batch),
                        expected_output_len=0,
                        request_id=request_id_prefix + str(i // batchsize),
                    )
                )
            requests = batch_requests

        if token_mismatch_total != 0:
            sign = "more" if token_mismatch_total > 0 else "fewer"
            logger.warning(
                "Across all generated prompts, there were %d %s tokens "
                "than expected after decoding and re-encoding. This is "
                "expected due to the imperfect nature of the sampling "
                "procedure.",
                abs(token_mismatch_total),
                sign,
            )

        return requests

    def get_prefix(
        self,
        tokenizer: PreTrainedTokenizerBase,
        allowed_tokens: np.ndarray,
        prefix_len: int,
    ) -> list[int]:
        """
        Get the prefix for the dataset.
        """
        if prefix_len <= 0:
            return []

        prefix_tokens = allowed_tokens[
            self._rng.integers(0, len(allowed_tokens), size=prefix_len)
        ].tolist()
        _, adjusted_tokens, token_mismatch = gen_prompt_decode_to_target_len(
            tokenizer=tokenizer,
            token_sequence=prefix_tokens,
            target_token_len=prefix_len,
            add_special_tokens=False,
            rng=self._rng,
        )
        if token_mismatch != 0:
            sign = "more" if token_mismatch > 0 else "fewer"
            logger.warning(
                "Prefix tokenization produced %d %s tokens than expected "
                "after decoding and re-encoding. This is expected due to "
                "the imperfect nature of the sampling procedure",
                abs(token_mismatch),
                sign,
            )
        return adjusted_tokens

    def generate_token_sequence(
        self,
        *,
        tokenizer: PreTrainedTokenizerBase,
        prefix_token_ids: list[int],
        prefix_len: int,
        vocab_size: int,
        input_len: int,
        offset: int,
        index: int,
        allowed_tokens: np.ndarray,
    ) -> tuple[str, int, int]:
        """
        Returns (prompt, total_input_len).

        NOTE: After decoding the prompt we have to encode and decode it again.
        This is done because in some cases N consecutive tokens
        give a string tokenized into != N number of tokens.
        For example for GPT2Tokenizer:
        [6880, 6881] -> ['Ġcalls', 'here'] ->
        [1650, 939, 486] -> ['Ġcall', 'sh', 'ere']
        To avoid uncontrolled change of the prompt length,
        the encoded sequence is truncated before being decoded again.
        """
        # Build the inner sequence by sampling
        # sequentially from the allowed tokens
        inner_seq = allowed_tokens[
            (offset + index + np.arange(input_len)) % len(allowed_tokens)
        ].tolist()
        token_sequence = prefix_token_ids + inner_seq

        # Decode, then re-encode and truncate to preserve token count invariants
        total_input_len = prefix_len + int(input_len)
        prompt, adjusted_token_sequence, token_mismatch = (
            gen_prompt_decode_to_target_len(
                tokenizer=tokenizer,
                token_sequence=token_sequence,
                target_token_len=total_input_len,
                add_special_tokens=False,
                rng=self._rng,
            )
        )
        total_input_len = len(adjusted_token_sequence)
        return prompt, total_input_len, token_mismatch
# fmt: on


def sample_sharegpt_requests(
    dataset_path: str | None,
    num_requests: int,
    tokenizer: PreTrainedTokenizerBase,
    fixed_output_len: int | None = None,
    max_model_len: int | None = None,
    apply_chat_template: bool = False,
    skip_min_tokens_check: bool = False,
) -> list[SampleRequest]:
    if fixed_output_len is not None and fixed_output_len < 4:
        raise ValueError("output_len too small")
    if not dataset_path:
        dataset_path = download_and_cache_file(SHAREGPT_URL)

    with open(dataset_path, encoding="utf-8") as f:
        dataset = json.load(f)

    conversations = []
    for data in dataset:
        turns = data.get("conversations", data.get("conversation", []))
        if len(turns) >= 2:
            conversations.append((turns[0]["value"], turns[1]["value"]))
    random.shuffle(conversations)

    samples: list[SampleRequest] = []
    for prompt, completion in conversations:
        if len(samples) == num_requests:
            break
        if apply_chat_template:
            prompt = tokenizer.apply_chat_template(
                [{"role": "user", "content": prompt}],
                add_generation_prompt=True,
                tokenize=False,
            )
            if tokenizer.bos_token:
                prompt = prompt.replace(tokenizer.bos_token, "")
        prompt_len = len(tokenizer.encode(prompt))
        output_len = fixed_output_len or len(tokenizer.encode(completion))
        if not is_valid_sequence(
            prompt_len, output_len, max_model_len, skip_min_tokens_check
        ):
            continue
        samples.append(SampleRequest(prompt, prompt_len, output_len))

    print(f"#Input tokens: {sum(x.prompt_len for x in samples)}")
    print(f"#Output tokens: {sum(x.expected_output_len for x in samples)}")
    return samples


def sample_random_requests(
    input_len: int,
    output_len: int,
    num_prompts: int,
    range_ratio: float,
    tokenizer: PreTrainedTokenizerBase,
    dataset_path: str | None,
    prefix_len: int = 0,
    random_seed: int = 0,
    request_id_prefix: str = "",
) -> list[SampleRequest]:
    if dataset_path is not None:
        raise ValueError("Cannot use 'random' dataset with --dataset-path.")

    samples = RandomDataset(random_seed=random_seed).sample(
        tokenizer=tokenizer,
        num_requests=num_prompts,
        request_id_prefix=request_id_prefix,
        prefix_len=prefix_len,
        range_ratio=range_ratio,
        input_len=input_len,
        output_len=output_len,
    )

    print(f"#Input tokens: {sum(x.prompt_len for x in samples)}")
    print(f"#Output tokens: {sum(x.expected_output_len for x in samples)}")
    return samples


def get_samples(
    args: argparse.Namespace, tokenizer: PreTrainedTokenizerBase
) -> list[SampleRequest]:
    if args.dataset_name == "sharegpt":
        return sample_sharegpt_requests(
            dataset_path=args.dataset_path,
            num_requests=args.num_prompts,
            tokenizer=tokenizer,
            fixed_output_len=args.sharegpt_output_len,
            max_model_len=args.max_model_len,
            apply_chat_template=args.apply_chat_template,
            skip_min_tokens_check=args.skip_min_tokens_check,
        )
    if args.dataset_name == "random":
        return sample_random_requests(
            input_len=args.random_input_len,
            output_len=args.random_output_len,
            num_prompts=args.num_prompts,
            range_ratio=args.random_range_ratio,
            tokenizer=tokenizer,
            dataset_path=args.dataset_path,
            prefix_len=args.random_prefix_len,
            random_seed=args.seed,
            request_id_prefix=args.request_id_prefix,
        )
    raise ValueError(f"Unknown dataset: {args.dataset_name}")


def get_current_request_rate(
    ramp_up_strategy: Literal["linear", "exponential"] | None,
    ramp_up_start_rps: int | None,
    ramp_up_end_rps: int | None,
    request_index: int,
    total_requests: int,
    request_rate: float,
) -> float:
    if (
        ramp_up_strategy
        and ramp_up_start_rps is not None
        and ramp_up_end_rps is not None
    ):
        progress = request_index / max(total_requests - 1, 1)
        if ramp_up_strategy == "linear":
            return ramp_up_start_rps + (ramp_up_end_rps - ramp_up_start_rps) * progress
        if ramp_up_strategy == "exponential":
            ratio = ramp_up_end_rps / ramp_up_start_rps
            return ramp_up_start_rps * (ratio**progress)
        raise ValueError(f"Unknown ramp-up strategy: {ramp_up_strategy}")
    return request_rate


async def get_request(
    input_requests: list[SampleRequest],
    request_rate: float,
    burstiness: float = 1.0,
    ramp_up_strategy: Literal["linear", "exponential"] | None = None,
    ramp_up_start_rps: int | None = None,
    ramp_up_end_rps: int | None = None,
) -> AsyncGenerator[tuple[SampleRequest, float], None]:
    assert (
        burstiness > 0
    ), f"A positive burstiness factor is expected, got {burstiness}."
    total_requests = len(input_requests)
    assert total_requests > 0, "No requests provided."

    delay_ts = []
    request_rates = []
    for request_index, _request in enumerate(input_requests):
        current_request_rate = get_current_request_rate(
            ramp_up_strategy,
            ramp_up_start_rps,
            ramp_up_end_rps,
            request_index,
            total_requests,
            request_rate,
        )
        assert (
            current_request_rate > 0.0
        ), f"Non-positive request rate {current_request_rate}."
        request_rates.append(current_request_rate)
        if current_request_rate == float("inf"):
            delay_ts.append(0.0)
        elif burstiness == float("inf"):
            delay_ts.append(1.0 / current_request_rate)
        else:
            theta = 1.0 / (current_request_rate * burstiness)
            delay_ts.append(float(np.random.gamma(shape=burstiness, scale=theta)))

    for i in range(1, len(delay_ts)):
        delay_ts[i] += delay_ts[i - 1]
    if ramp_up_strategy is None and delay_ts[-1] != 0:
        target_total_delay_s = total_requests / request_rate
        normalize_factor = target_total_delay_s / delay_ts[-1]
        delay_ts = [delay * normalize_factor for delay in delay_ts]

    start_ts = time.time()
    for request_index, request in enumerate(input_requests):
        if delay_ts[request_index] > 0:
            sleep_interval_s = start_ts + delay_ts[request_index] - time.time()
            if sleep_interval_s > 0:
                await asyncio.sleep(sleep_interval_s)
        yield request, request_rates[request_index]


async def get_first_model_from_server(
    base_url: str,
    headers: dict[str, str] | None = None,
    ssl_context: ssl.SSLContext | bool | None = None,
) -> tuple[str, str]:
    connector = aiohttp.TCPConnector(ssl=ssl_context)
    async with aiohttp.ClientSession(connector=connector) as session:
        models_url = f"{base_url}/v1/models"
        try:
            async with session.get(models_url, headers=headers) as response:
                response.raise_for_status()
                data = await response.json()
                if data.get("data"):
                    model = data["data"][0]
                    return model["id"], model.get("root", model["id"])
                raise ValueError(f"No models found on the server at {base_url}.")
        except (aiohttp.ClientError, json.JSONDecodeError) as e:
            raise RuntimeError(f"Failed to fetch models from {models_url}: {e}") from e


async def wait_for_endpoint(
    request_func,
    test_input: RequestFuncInput,
    session: aiohttp.ClientSession,
    timeout_seconds: int = 600,
    retry_interval: int = 5,
) -> RequestFuncOutput:
    deadline = time.perf_counter() + timeout_seconds
    output = RequestFuncOutput(success=False)
    print(f"Waiting for endpoint to become up in {timeout_seconds} seconds")
    with tqdm(
        total=timeout_seconds,
        bar_format="{desc} |{bar}| {elapsed} elapsed, {remaining} remaining",
        unit="s",
    ) as pbar:
        while True:
            remaining = deadline - time.perf_counter()
            elapsed = timeout_seconds - remaining
            pbar.update(min(elapsed - pbar.n, timeout_seconds - pbar.n))
            pbar.refresh()
            if remaining <= 0:
                break
            try:
                output = await request_func(test_input, session=session)
                if output.success:
                    return output
                err_last_line = str(output.error).rstrip().rsplit("\n", 1)[-1]
                print(f"Endpoint is not ready. Error='{err_last_line}'")
            except aiohttp.ClientConnectorError:
                pass
            await asyncio.sleep(min(retry_interval, max(remaining, 0)))
    return output


def calculate_metrics(
    input_requests: list[SampleRequest],
    outputs: list[RequestFuncOutput],
    dur_s: float,
    tokenizer: PreTrainedTokenizerBase | None,
    selected_percentiles: list[float],
    goodput_config_dict: dict[str, float],
) -> tuple[BenchmarkMetrics, list[int]]:
    actual_output_lens: list[int] = []
    total_input = 0
    completed = 0
    good_completed = 0
    itls: list[float] = []
    tpots: list[float] = []
    all_tpots: list[float] = []
    ttfts: list[float] = []
    e2els: list[float] = []

    for output in outputs:
        if output.success:
            output_len = output.output_tokens
            if not output_len:
                output_len = (
                    len(
                        tokenizer.encode(
                            output.generated_text, add_special_tokens=False
                        )
                    )
                    if tokenizer
                    else 1
                )
            actual_output_lens.append(output_len)
            total_input += output.prompt_len
            tpot = 0.0
            if output_len > 1:
                tpot = (output.latency - output.ttft) / (output_len - 1)
                tpots.append(tpot)
            all_tpots.append(tpot)
            itls.extend(output.itl)
            ttfts.append(output.ttft)
            e2els.append(output.latency)
            completed += 1
        else:
            actual_output_lens.append(0)

    if goodput_config_dict:
        valid_metrics = []
        slo_values = []
        if "ttft" in goodput_config_dict:
            valid_metrics.append(ttfts)
            slo_values.append(
                goodput_config_dict["ttft"] / MILLISECONDS_TO_SECONDS_CONVERSION
            )
        if "tpot" in goodput_config_dict:
            valid_metrics.append(all_tpots)
            slo_values.append(
                goodput_config_dict["tpot"] / MILLISECONDS_TO_SECONDS_CONVERSION
            )
        if "e2el" in goodput_config_dict:
            valid_metrics.append(e2els)
            slo_values.append(
                goodput_config_dict["e2el"] / MILLISECONDS_TO_SECONDS_CONVERSION
            )
        for req_metric in zip(*valid_metrics):
            if all(slo >= metric for slo, metric in zip(slo_values, req_metric)):
                good_completed += 1

    if completed == 0:
        warnings.warn(
            "All requests failed. This is likely due to a misconfiguration on the benchmark arguments.",
            stacklevel=2,
        )

    successful_outputs = [output for output in outputs if output.success]
    failed_outputs = [output for output in outputs if not output.success]
    if failed_outputs:
        print("Failed requests during benchmark run detected (capping to 10):")
        for i, err in enumerate(failed_outputs[:10]):
            print(f"Error {i}: {err.error}")

    max_output_tokens_per_s = 0.0
    max_concurrent_requests = 0
    if successful_outputs:
        min_start_time = min(output.start_time for output in successful_outputs)
        max_end_time = max(
            output.start_time + output.latency for output in successful_outputs
        )
        duration_seconds = int(np.ceil(max_end_time - min_start_time)) + 1
        tokens_per_second = np.zeros(duration_seconds)
        concurrent_requests_per_second = np.zeros(duration_seconds)
        for output in successful_outputs:
            token_times = [output.start_time + output.ttft]
            current_time = token_times[0]
            for itl_value in output.itl:
                current_time += itl_value
                token_times.append(current_time)
            for token_time in token_times:
                second_bucket = int(token_time - min_start_time)
                if 0 <= second_bucket < duration_seconds:
                    tokens_per_second[second_bucket] += 1
            request_start_second = int(output.start_time - min_start_time)
            request_end_second = int(
                (output.start_time + output.latency) - min_start_time
            )
            for second in range(request_start_second, request_end_second + 1):
                concurrent_requests_per_second[second] += 1
        max_output_tokens_per_s = (
            float(np.max(tokens_per_second)) if len(tokens_per_second) else 0.0
        )
        max_concurrent_requests = (
            int(np.max(concurrent_requests_per_second))
            if len(concurrent_requests_per_second)
            else 0
        )

    metrics = BenchmarkMetrics(
        completed=completed,
        failed=len(failed_outputs),
        total_input=total_input,
        total_output=sum(actual_output_lens),
        request_throughput=completed / dur_s,
        request_goodput=good_completed / dur_s,
        output_throughput=sum(actual_output_lens) / dur_s,
        total_token_throughput=(total_input + sum(actual_output_lens)) / dur_s,
        mean_ttft_ms=np.mean(ttfts or 0) * 1000,
        std_ttft_ms=np.std(ttfts or 0) * 1000,
        median_ttft_ms=np.median(ttfts or 0) * 1000,
        percentiles_ttft_ms=[
            (p, np.percentile(ttfts or 0, p) * 1000) for p in selected_percentiles
        ],
        mean_tpot_ms=np.mean(tpots or 0) * 1000,
        std_tpot_ms=np.std(tpots or 0) * 1000,
        median_tpot_ms=np.median(tpots or 0) * 1000,
        percentiles_tpot_ms=[
            (p, np.percentile(tpots or 0, p) * 1000) for p in selected_percentiles
        ],
        mean_itl_ms=np.mean(itls or 0) * 1000,
        std_itl_ms=np.std(itls or 0) * 1000,
        median_itl_ms=np.median(itls or 0) * 1000,
        percentiles_itl_ms=[
            (p, np.percentile(itls or 0, p) * 1000) for p in selected_percentiles
        ],
        mean_e2el_ms=np.mean(e2els or 0) * 1000,
        std_e2el_ms=np.std(e2els or 0) * 1000,
        median_e2el_ms=np.median(e2els or 0) * 1000,
        percentiles_e2el_ms=[
            (p, np.percentile(e2els or 0, p) * 1000) for p in selected_percentiles
        ],
        max_output_tokens_per_s=max_output_tokens_per_s,
        max_concurrent_requests=max_concurrent_requests,
    )
    return metrics, actual_output_lens


async def benchmark(
    task_type: TaskType,
    backend: str,
    api_url: str,
    base_url: str,
    model_id: str,
    model_name: str | None,
    tokenizer: PreTrainedTokenizerBase | None,
    input_requests: list[SampleRequest],
    logprobs: int | None,
    request_rate: float,
    burstiness: float,
    disable_tqdm: bool,
    num_warmups: int,
    profile: bool,
    profile_num_steps: int | None,
    selected_percentile_metrics: list[str],
    selected_percentiles: list[float],
    ignore_eos: bool,
    goodput_config_dict: dict[str, float],
    max_concurrency: int | None,
    extra_headers: dict[str, str] | None,
    extra_body: dict[str, Any] | None,
    ramp_up_strategy: Literal["linear", "exponential"] | None = None,
    ramp_up_start_rps: int | None = None,
    ramp_up_end_rps: int | None = None,
    ready_check_timeout_sec: int = 600,
    ssl_context: ssl.SSLContext | bool | None = None,
) -> dict[str, Any]:
    try:
        request_func = ASYNC_REQUEST_FUNCS[backend]
    except KeyError:
        raise ValueError(f"Unknown backend: {backend}") from None

    connector = aiohttp.TCPConnector(ssl=ssl_context)
    session = aiohttp.ClientSession(
        connector=connector, trust_env=True, timeout=AIOHTTP_TIMEOUT
    )

    test_request = input_requests[0]
    test_input = RequestFuncInput(
        model=model_id,
        model_name=model_name,
        prompt=test_request.prompt,
        api_url=api_url,
        prompt_len=test_request.prompt_len,
        output_len=test_request.expected_output_len,
        logprobs=logprobs,
        ignore_eos=ignore_eos,
        extra_headers=extra_headers,
        extra_body=extra_body,
    )

    if ready_check_timeout_sec > 0:
        print("Starting initial single prompt test run...")
        test_output = await wait_for_endpoint(
            request_func, test_input, session, timeout_seconds=ready_check_timeout_sec
        )
        if not test_output.success:
            raise ValueError(
                "Initial test run failed - Please make sure benchmark arguments are correctly specified. "
                f"Error: {test_output.error}"
            )
        print("Initial test run completed.")
    else:
        print("Skipping endpoint ready check.")

    if num_warmups > 0:
        print(f"Warming up with {num_warmups} requests...")
        warmup_pbar = None if disable_tqdm else tqdm(total=num_warmups)
        warmup_semaphore = (
            asyncio.Semaphore(max_concurrency)
            if max_concurrency
            else contextlib.nullcontext()
        )

        async def warmup_limited_request_func():
            async with warmup_semaphore:
                return await request_func(test_input, session=session, pbar=warmup_pbar)

        await asyncio.gather(
            *(
                asyncio.create_task(warmup_limited_request_func())
                for _ in range(num_warmups)
            )
        )
        if warmup_pbar:
            warmup_pbar.close()
        print("Warmup run completed.")

    if profile:
        if profile_num_steps is None:
            print("Starting profiler...")
        else:
            print(f"Starting profiler for {profile_num_steps} steps...")
        extra_body = dict(extra_body or {})
        if profile_num_steps is not None:
            extra_body["num_steps"] = profile_num_steps
        profile_input = RequestFuncInput(
            model=model_id,
            model_name=model_name,
            prompt=test_request.prompt,
            api_url=base_url + "/start_profile",
            prompt_len=test_request.prompt_len,
            output_len=test_request.expected_output_len,
            logprobs=logprobs,
            ignore_eos=ignore_eos,
            extra_headers=extra_headers,
            extra_body=extra_body,
        )
        profile_output = await request_func(profile_input, session=session)
        if profile_output.success:
            print("Profiler started")

    distribution = "Poisson process" if burstiness == 1.0 else "Gamma distribution"
    if ramp_up_strategy:
        print(f"Traffic ramp-up strategy: {ramp_up_strategy}.")
        print(f"Will increase RPS from {ramp_up_start_rps} to {ramp_up_end_rps} RPS.")
    else:
        print(f"Traffic request rate: {request_rate}")
    print(f"Burstiness factor: {burstiness} ({distribution})")
    print(f"Maximum request concurrency: {max_concurrency}")

    pbar = None if disable_tqdm else tqdm(total=len(input_requests))
    semaphore = (
        asyncio.Semaphore(max_concurrency)
        if max_concurrency
        else contextlib.nullcontext()
    )

    async def limited_request_func(request_func_input, session, pbar):
        async with semaphore:
            coro = request_func(request_func_input, session=session, pbar=pbar)
            return await await_with_per_request_timeout(
                coro,
                prompt_len=request_func_input.prompt_len,
                pbar=pbar,
            )

    print("Starting main benchmark run...")
    benchmark_start_time = time.perf_counter()
    tasks: list[asyncio.Task] = []
    rps_change_events = []
    last_int_rps = -1
    if ramp_up_strategy is not None and ramp_up_start_rps is not None:
        last_int_rps = ramp_up_start_rps
        rps_change_events.append(
            {"rps": last_int_rps, "timestamp": datetime.now().isoformat()}
        )

    async for request, current_request_rate in get_request(
        input_requests,
        request_rate,
        burstiness,
        ramp_up_strategy,
        ramp_up_start_rps,
        ramp_up_end_rps,
    ):
        if ramp_up_strategy is not None:
            current_int_rps = int(current_request_rate)
            if current_int_rps > last_int_rps:
                timestamp = datetime.now().isoformat()
                for rps_val in range(last_int_rps + 1, current_int_rps + 1):
                    rps_change_events.append({"rps": rps_val, "timestamp": timestamp})
                last_int_rps = current_int_rps
        request_func_input = RequestFuncInput(
            model=model_id,
            model_name=model_name,
            prompt=request.prompt,
            api_url=api_url,
            prompt_len=request.prompt_len,
            output_len=request.expected_output_len,
            logprobs=logprobs,
            ignore_eos=ignore_eos,
            extra_headers=extra_headers,
            extra_body=extra_body,
            request_id=request.request_id,
        )
        tasks.append(
            asyncio.create_task(limited_request_func(request_func_input, session, pbar))
        )

    outputs = await asyncio.gather(*tasks)
    if pbar:
        pbar.close()
    benchmark_duration = time.perf_counter() - benchmark_start_time

    metrics, actual_output_lens = calculate_metrics(
        input_requests,
        outputs,
        benchmark_duration,
        tokenizer,
        selected_percentiles,
        goodput_config_dict,
    )

    _print_section_header(" Serving Benchmark Result ", "=")
    _print_metric_row("Successful requests:", metrics.completed)
    _print_metric_row("Failed requests:", metrics.failed)
    if max_concurrency is not None:
        _print_metric_row("Maximum request concurrency:", max_concurrency)
    if request_rate != float("inf"):
        _print_metric_row("Request rate configured (RPS):", request_rate, precision=2)
    _print_metric_row("Benchmark duration (s):", benchmark_duration, precision=2)
    _print_metric_row("Total input tokens:", metrics.total_input)
    _print_metric_row("Total generated tokens:", metrics.total_output)
    _print_metric_row(
        "Request throughput (req/s):", metrics.request_throughput, precision=2
    )
    if goodput_config_dict:
        _print_metric_row(
            "Request goodput (req/s):", metrics.request_goodput, precision=2
        )
    _print_metric_row(
        "Output token throughput (tok/s):", metrics.output_throughput, precision=2
    )
    _print_metric_row(
        "Peak output token throughput (tok/s):",
        metrics.max_output_tokens_per_s,
        precision=2,
    )
    _print_metric_row(
        "Peak concurrent requests:", metrics.max_concurrent_requests, precision=2
    )
    _print_metric_row(
        "Total token throughput (tok/s):",
        metrics.total_token_throughput,
        precision=2,
    )

    result: dict[str, Any] = {
        "duration": benchmark_duration,
        "completed": metrics.completed,
        "failed": metrics.failed,
        "total_input_tokens": metrics.total_input,
        "total_output_tokens": metrics.total_output,
        "request_throughput": metrics.request_throughput,
        "request_goodput": metrics.request_goodput if goodput_config_dict else None,
        "output_throughput": metrics.output_throughput,
        "total_token_throughput": metrics.total_token_throughput,
        "input_lens": [output.prompt_len for output in outputs],
        "output_lens": actual_output_lens,
        "ttfts": [output.ttft for output in outputs],
        "itls": [output.itl for output in outputs],
        "start_times": [output.start_time for output in outputs],
        "generated_texts": [output.generated_text for output in outputs],
        "errors": [output.error for output in outputs],
        "max_output_tokens_per_s": metrics.max_output_tokens_per_s,
        "max_concurrent_requests": metrics.max_concurrent_requests,
    }
    if rps_change_events:
        result["rps_change_events"] = rps_change_events

    def process_one_metric(
        metric_attribute_name: str, metric_name: str, metric_header: str
    ) -> None:
        if metric_attribute_name not in selected_percentile_metrics:
            return
        _print_section_header(metric_header, "-")
        _print_metric_row(
            f"Mean {metric_name} (ms):",
            getattr(metrics, f"mean_{metric_attribute_name}_ms"),
            precision=2,
        )
        _print_metric_row(
            f"Median {metric_name} (ms):",
            getattr(metrics, f"median_{metric_attribute_name}_ms"),
            precision=2,
        )
        result[f"mean_{metric_attribute_name}_ms"] = getattr(
            metrics, f"mean_{metric_attribute_name}_ms"
        )
        result[f"median_{metric_attribute_name}_ms"] = getattr(
            metrics, f"median_{metric_attribute_name}_ms"
        )
        result[f"std_{metric_attribute_name}_ms"] = getattr(
            metrics, f"std_{metric_attribute_name}_ms"
        )
        for p, value in getattr(metrics, f"percentiles_{metric_attribute_name}_ms"):
            p_word = str(int(p)) if int(p) == p else str(p)
            _print_metric_row(f"P{p_word} {metric_name} (ms):", value, precision=2)
            result[f"p{p_word}_{metric_attribute_name}_ms"] = value

    process_one_metric("ttft", "TTFT", "Time to First Token")
    process_one_metric("tpot", "TPOT", "Time per Output Token (excl. 1st token)")
    process_one_metric("itl", "ITL", "Inter-token Latency")
    process_one_metric("e2el", "E2EL", "End-to-end Latency")

    print("=" * 50)

    if profile and profile_num_steps is None:
        print("Stopping profiler...")
        profile_input = RequestFuncInput(
            model=model_id,
            model_name=model_name,
            prompt=test_request.prompt,
            api_url=base_url + "/stop_profile",
            prompt_len=test_request.prompt_len,
            output_len=test_request.expected_output_len,
            logprobs=logprobs,
            ignore_eos=ignore_eos,
        )
        profile_output = await request_func(profile_input, session=session)
        if profile_output.success:
            print("Profiler stopped")

    await session.close()
    return result


def parse_goodput(slo_pairs: list[str] | None) -> dict[str, float]:
    goodput_config_dict: dict[str, float] = {}
    if not slo_pairs:
        return goodput_config_dict
    try:
        for slo_pair in slo_pairs:
            slo_name, slo_val = slo_pair.split(":")
            goodput_config_dict[slo_name] = float(slo_val)
    except ValueError as err:
        raise argparse.ArgumentTypeError(
            'Specify service level objectives for goodput as "KEY:VALUE" pairs.'
        ) from err
    for slo_name, slo_val in goodput_config_dict.items():
        if slo_name not in {"ttft", "tpot", "e2el"}:
            raise ValueError(f"Invalid goodput metric {slo_name!r}.")
        if slo_val < 0:
            raise ValueError(f"Goodput SLO {slo_name!r} must be non-negative.")
    return goodput_config_dict


def compute_result_filename(
    args: argparse.Namespace, model_id: str, label: str | None, current_dt: str
) -> str | None:
    if not (args.save_result or args.append_result or args.output_file):
        return None
    if args.output_file:
        return args.output_file
    base_model_id = model_id.split("/")[-1]
    max_concurrency_str = (
        f"-concurrency{args.max_concurrency}"
        if args.max_concurrency is not None
        else ""
    )
    result_label = label or args.backend
    if args.ramp_up_strategy is not None:
        file_name = f"{result_label}-ramp-up-{args.ramp_up_strategy}-{args.ramp_up_start_rps}qps-{args.ramp_up_end_rps}qps{max_concurrency_str}-{base_model_id}-{current_dt}.json"
    else:
        file_name = f"{result_label}-{args.request_rate}qps{max_concurrency_str}-{base_model_id}-{current_dt}.json"
    if args.result_dir:
        os.makedirs(args.result_dir, exist_ok=True)
        file_name = os.path.join(args.result_dir, file_name)
    return file_name


def add_dataset_parser(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--dataset-name",
        type=str,
        default="random",
        choices=["sharegpt", "random"],
        help="Name of the dataset to benchmark on.",
    )
    parser.add_argument(
        "--dataset-path", type=str, default=None, help="Path to the dataset."
    )
    parser.add_argument("--num-prompts", type=int, default=DEFAULT_NUM_PROMPTS)
    parser.add_argument("--input-len", type=int, default=None)
    parser.add_argument("--output-len", type=int, default=None)
    parser.add_argument("--max-model-len", type=int, default=None)
    parser.add_argument("--skip-min-tokens-check", action="store_true")
    parser.add_argument("--sharegpt-output-len", type=int, default=None)
    parser.add_argument("--random-input-len", type=int, default=1024)
    parser.add_argument("--random-output-len", type=int, default=128)
    parser.add_argument("--random-range-ratio", type=float, default=0.0)
    parser.add_argument("--random-prefix-len", type=int, default=0)
    parser.add_argument("--request-id-prefix", type=str, default="bench-")


def add_serving_cli_args(parser: argparse.ArgumentParser) -> None:
    add_dataset_parser(parser)
    parser.add_argument("--label", type=str, default=None)
    parser.add_argument(
        "--backend",
        type=str,
        default="openai",
        choices=list(ASYNC_REQUEST_FUNCS.keys()),
        help="The backend type to use for the benchmark.",
    )
    parser.add_argument("--base-url", type=str, default=None)
    parser.add_argument("--host", type=str, default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--endpoint", type=str, default="/v1/completions")
    parser.add_argument("--header", metavar="KEY=VALUE", nargs="*")
    parser.add_argument("--model", type=str, default=None)
    parser.add_argument("--served-model-name", type=str, default=None)
    parser.add_argument("--tokenizer", type=str, default=None)
    parser.add_argument("--skip-tokenizer-init", action="store_true")
    parser.add_argument("--trust-remote-code", action="store_true", default=True)
    parser.add_argument("--request-rate", type=float, default=float("inf"))
    parser.add_argument("--burstiness", type=float, default=1.0)
    parser.add_argument("--max-concurrency", type=int, default=None)
    parser.add_argument("--num-warmups", type=int, default=0)
    parser.add_argument("--ready-check-timeout-sec", type=int, default=600)
    parser.add_argument("--disable-tqdm", action="store_true")
    parser.add_argument("--profile", action="store_true")
    parser.add_argument("--profile-num-steps", type=int, default=None)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--ignore-eos", action="store_true")
    parser.add_argument("--disable-ignore-eos", action="store_true")
    parser.add_argument("--apply-chat-template", action="store_true")
    parser.add_argument("--logprobs", type=int, default=None)
    parser.add_argument("--extra-body", type=json.loads, default={})
    parser.add_argument("--extra-request-body", type=json.loads, default=None)
    parser.add_argument("--goodput", nargs="*", default=None)
    parser.add_argument("--percentile-metrics", type=str, default=None)
    parser.add_argument("--metric-percentiles", type=str, default="99")
    parser.add_argument(
        "--ramp-up-strategy", choices=["linear", "exponential"], default=None
    )
    parser.add_argument("--ramp-up-start-rps", type=int, default=None)
    parser.add_argument("--ramp-up-end-rps", type=int, default=None)
    parser.add_argument("--insecure", action="store_true")
    parser.add_argument("--save-result", action="store_true")
    parser.add_argument("--append-result", action="store_true")
    parser.add_argument("--save-detailed", action="store_true")
    parser.add_argument("--result-dir", type=str, default=None)
    parser.add_argument("--output-file", type=str, default=None)
    parser.set_defaults(dispatch_function=BenchmarkServingSubcommand.cmd)


async def main_async(args: argparse.Namespace) -> dict[str, Any]:
    print(args)
    set_ulimit()
    random.seed(args.seed)
    np.random.seed(args.seed)

    if args.disable_ignore_eos:
        args.ignore_eos = False
    if args.extra_request_body is not None:
        args.extra_body = args.extra_request_body
    if args.profile_num_steps is not None:
        if args.profile_num_steps <= 0:
            raise ValueError("--profile-num-steps must be positive.")
        if not args.profile:
            raise ValueError("--profile-num-steps requires --profile.")
    if args.input_len is not None:
        args.random_input_len = args.input_len
    if args.output_len is not None:
        args.random_output_len = args.output_len
        args.sharegpt_output_len = args.output_len

    if args.ramp_up_strategy is not None:
        if args.request_rate != float("inf"):
            raise ValueError("When using ramp-up, do not specify --request-rate.")
        if args.ramp_up_start_rps is None or args.ramp_up_end_rps is None:
            raise ValueError(
                "Ramp-up requires --ramp-up-start-rps and --ramp-up-end-rps."
            )
        if args.ramp_up_start_rps > args.ramp_up_end_rps:
            raise ValueError("Ramp-up start RPS must be less than end RPS.")
        if args.ramp_up_strategy == "exponential" and args.ramp_up_start_rps == 0:
            raise ValueError("For exponential ramp-up, start RPS cannot be 0.")

    if args.base_url is not None:
        api_url = f"{args.base_url}{args.endpoint}"
        base_url = args.base_url
    else:
        host_port = join_host_port(args.host, args.port)
        api_url = f"http://{host_port}{args.endpoint}"
        base_url = f"http://{host_port}"

    headers = None
    if args.header:
        headers = {}
        for item in args.header:
            if "=" not in item:
                raise ValueError("Invalid header format. Please use KEY=VALUE format.")
            key, value = item.split("=", 1)
            headers[key.strip()] = value.strip()

    ssl_context: ssl.SSLContext | bool | None = (
        False if args.insecure else True if base_url.startswith("https://") else None
    )

    if args.model is None:
        print("Model not specified, fetching first model from server...")
        model_name, model_id = await get_first_model_from_server(
            base_url, headers, ssl_context
        )
        print(f"First model name: {model_name}, first model id: {model_id}")
    else:
        model_name = args.served_model_name
        model_id = args.model

    tokenizer = None
    tokenizer_id = None
    if not args.skip_tokenizer_init:
        tokenizer_id = args.tokenizer or model_id
        tokenizer = get_tokenizer(tokenizer_id)

    if args.dataset_name == "random" and args.backend in OPENAI_COMPATIBLE_BACKENDS:
        args.ignore_eos = True

    input_requests = get_samples(args, tokenizer)
    goodput_config_dict = parse_goodput(args.goodput)
    extra_body = args.extra_body or {}
    percentile_metrics = args.percentile_metrics or "ttft,tpot,itl"

    if "temperature" not in extra_body:
        print(
            "WARNING: tokenspeed bench serve no longer sets temperature==0 in requests by default. "
            "The server decides its own default. Include --extra-body '{\"temperature\": 0}' for greedy decoding."
        )

    benchmark_result = await benchmark(
        task_type=TaskType.GENERATION,
        backend=args.backend,
        api_url=api_url,
        base_url=base_url,
        model_id=model_id,
        model_name=model_name,
        tokenizer=tokenizer,
        input_requests=input_requests,
        logprobs=args.logprobs,
        request_rate=args.request_rate,
        burstiness=args.burstiness,
        disable_tqdm=args.disable_tqdm,
        num_warmups=args.num_warmups,
        profile=args.profile,
        profile_num_steps=args.profile_num_steps,
        selected_percentile_metrics=percentile_metrics.split(","),
        selected_percentiles=[float(p) for p in args.metric_percentiles.split(",")],
        ignore_eos=args.ignore_eos,
        goodput_config_dict=goodput_config_dict,
        max_concurrency=args.max_concurrency,
        extra_headers=headers,
        extra_body=extra_body,
        ramp_up_strategy=args.ramp_up_strategy,
        ramp_up_start_rps=args.ramp_up_start_rps,
        ramp_up_end_rps=args.ramp_up_end_rps,
        ready_check_timeout_sec=args.ready_check_timeout_sec,
        ssl_context=ssl_context,
    )

    current_dt = datetime.now().strftime("%Y%m%d-%H%M%S")
    result_json = {
        "date": current_dt,
        "backend": args.backend,
        "label": args.label,
        "model_id": model_id,
        "tokenizer_id": tokenizer_id,
        "num_prompts": args.num_prompts,
        "request_rate": (
            args.request_rate if args.request_rate < float("inf") else "inf"
        ),
        "burstiness": args.burstiness,
        "max_concurrency": args.max_concurrency,
        **benchmark_result,
    }

    if not args.save_detailed:
        for field_name in [
            "input_lens",
            "output_lens",
            "start_times",
            "ttfts",
            "itls",
            "generated_texts",
            "errors",
        ]:
            result_json.pop(field_name, None)

    file_name = compute_result_filename(args, model_id, args.label, current_dt)
    if file_name:
        with open(
            file_name, mode="a+" if args.append_result else "w", encoding="utf-8"
        ) as outfile:
            if args.append_result and outfile.tell() != 0:
                outfile.write("\n")
            json.dump(result_json, outfile)

    return result_json


def run_benchmark(args: argparse.Namespace) -> dict[str, Any]:
    return asyncio.run(main_async(args))


class BenchmarkSubcommandBase:
    help: str
    name: str

    @classmethod
    def add_cli_args(cls, parser: argparse.ArgumentParser) -> None:
        raise NotImplementedError

    @staticmethod
    def cmd(args: argparse.Namespace) -> None:
        raise NotImplementedError


class BenchmarkServingSubcommand(BenchmarkSubcommandBase):
    name = "serve"
    help = "Benchmark online serving throughput."

    @classmethod
    def add_cli_args(cls, parser: argparse.ArgumentParser) -> None:
        add_serving_cli_args(parser)

    @staticmethod
    def cmd(args: argparse.Namespace) -> None:
        run_benchmark(args)


class BenchmarkSubcommand:
    name = "bench"
    help = "TokenSpeed bench subcommand."

    @staticmethod
    def cmd(args: argparse.Namespace) -> None:
        args.dispatch_function(args)

    def subparser_init(
        self, subparsers: argparse._SubParsersAction
    ) -> argparse.ArgumentParser:
        bench_parser = subparsers.add_parser(
            self.name,
            help=self.help,
            description=self.help,
            usage=f"tokenspeed {self.name} <bench_type> [options]",
        )
        bench_subparsers = bench_parser.add_subparsers(required=True, dest="bench_type")
        for cmd_cls in BenchmarkSubcommandBase.__subclasses__():
            cmd_subparser = bench_subparsers.add_parser(
                cmd_cls.name,
                help=cmd_cls.help,
                description=cmd_cls.help,
                usage=f"tokenspeed {self.name} {cmd_cls.name} [options]",
            )
            cmd_subparser.set_defaults(dispatch_function=cmd_cls.cmd)
            cmd_cls.add_cli_args(cmd_subparser)
        return bench_parser


def is_legacy_serving_args(argv: list[str]) -> bool:
    return bool(argv) and argv[0].startswith("-") and argv[0] not in ("-h", "--help")


def main(argv: list[str] | None = None) -> None:
    argv = list(sys.argv[1:] if argv is None else argv)
    if is_legacy_serving_args(argv):
        parser = argparse.ArgumentParser(description=BenchmarkServingSubcommand.help)
        BenchmarkServingSubcommand.add_cli_args(parser)
        args = parser.parse_args(argv)
        BenchmarkServingSubcommand.cmd(args)
        return

    parser = argparse.ArgumentParser(
        prog="tokenspeed", description="TokenSpeed benchmark commands."
    )
    subparsers = parser.add_subparsers(required=True, dest="command")
    BenchmarkSubcommand().subparser_init(subparsers)
    args = parser.parse_args(["bench", *argv])
    BenchmarkSubcommand.cmd(args)


if __name__ == "__main__":
    main()
