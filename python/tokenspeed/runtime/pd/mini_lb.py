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

"""Minimal HTTP load balancer for prefill and decode servers used in tests."""

import asyncio
import contextlib
import dataclasses
import logging
import random
import urllib
from itertools import chain
from typing import List, Optional

import aiohttp
import orjson
import uvicorn
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import ORJSONResponse, Response, StreamingResponse

from tokenspeed.runtime.pd.utils import PDRegistryRequest


def setup_logger():
    logger = logging.getLogger("pdlb")
    logger.setLevel(logging.INFO)

    formatter = logging.Formatter(
        "[PDLB (Python)] %(asctime)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    handler = logging.StreamHandler()
    handler.setFormatter(formatter)
    logger.addHandler(handler)

    return logger


logger = setup_logger()


def normalize_base_url(url: str) -> str:
    return url.rstrip("/")


@dataclasses.dataclass
class PrefillConfig:
    url: str
    bootstrap_port: Optional[int] = None

    def __post_init__(self):
        self.url = normalize_base_url(self.url)


class MiniLoadBalancer:
    def __init__(
        self,
        prefill_configs: List[PrefillConfig],
        decode_servers: List[str],
        enable_cache_report: bool = False,
    ):
        self.prefill_configs = prefill_configs
        self.prefill_servers = [p.url for p in prefill_configs]
        self.decode_servers = [normalize_base_url(url) for url in decode_servers]
        self.bootstrap_room_cnt = 0
        self.bootstrap_room_lock = asyncio.Lock()
        self.enable_cache_report = enable_cache_report
        self.prefill_index = 0  # Round-robin cursor for prefill_configs.
        self.decode_index = 0  # Round-robin cursor for decode_servers.

    async def get_bootstrap_room(self):
        async with self.bootstrap_room_lock:
            self.bootstrap_room_cnt += 1
            return self.bootstrap_room_cnt

    def add_prefill_server(self, new_prefill_config: PrefillConfig):
        self.prefill_configs.append(new_prefill_config)
        self.prefill_servers.append(new_prefill_config.url)

    def add_decode_server(self, new_decode_server: str):
        self.decode_servers.append(normalize_base_url(new_decode_server))

    def select_pair_round_robin(self):
        # 校验服务列表非空
        assert len(self.prefill_configs) > 0, "No prefill servers available"
        assert len(self.decode_servers) > 0, "No decode servers available"

        # 轮询选择prefill_config
        prefill_config = self.prefill_configs[self.prefill_index]
        # 更新prefill索引（循环递增，超过长度后重置为0）
        self.prefill_index = (self.prefill_index + 1) % len(self.prefill_configs)

        # 轮询选择decode_server
        decode_server = self.decode_servers[self.decode_index]
        # 更新decode索引
        self.decode_index = (self.decode_index + 1) % len(self.decode_servers)

        return prefill_config.url, prefill_config.bootstrap_port, decode_server

    def select_pair(self):
        assert len(self.prefill_configs) > 0, "No prefill servers available"
        assert len(self.decode_servers) > 0, "No decode servers available"

        prefill_config = random.choice(self.prefill_configs)
        decode_server = random.choice(self.decode_servers)
        return prefill_config.url, prefill_config.bootstrap_port, decode_server

    async def generate(
        self, modified_request, prefill_server, decode_server, endpoint, raw_request=None
    ) -> ORJSONResponse:
        assert endpoint[0] != "/", f"Endpoint should not start with '/': {endpoint}"

        async with aiohttp.ClientSession(
            timeout=aiohttp.ClientTimeout(
                total=3600
            )  # Add timeout for request reliability
        ) as session:
            tasks = [
                asyncio.create_task(
                    session.post(f"{prefill_server}/{endpoint}", json=modified_request)
                ),
                asyncio.create_task(
                    session.post(f"{decode_server}/{endpoint}", json=modified_request)
                ),
            ]

            try:
                # Wait for both responses to complete. Prefill should end first.
                prefill_response, decode_response = await self._gather_with_disconnect(
                    tasks, raw_request
                )
            except asyncio.CancelledError:
                raise HTTPException(status_code=499, detail="Client disconnected")

            if modified_request is not None:
                if "return_logprob" in modified_request or self.enable_cache_report:
                    prefill_json = await prefill_response.json()
                    ret_json = await decode_response.json()
                    # merge `meta_info.input_token_logprobs` from prefill to decode
                    if "meta_info" in ret_json:
                        if "input_token_logprobs" in ret_json["meta_info"]:
                            ret_json["meta_info"]["input_token_logprobs"] = (
                                prefill_json["meta_info"]["input_token_logprobs"]
                                + ret_json["meta_info"]["input_token_logprobs"]
                            )

                    # Inject cached_tokens from prefill if needed
                    if self.enable_cache_report:
                        # Extract cached_tokens from prefill response (which may have usage field)
                        cached_tokens = 0

                        # Try to get cached_tokens from prefill's usage field (new format)
                        if (
                            "usage" in prefill_json
                            and "prompt_tokens_details" in prefill_json["usage"]
                        ):
                            if (
                                prefill_json["usage"]["prompt_tokens_details"]
                                is not None
                            ):
                                cached_tokens = prefill_json["usage"][
                                    "prompt_tokens_details"
                                ].get("cached_tokens", 0)

                        # If not found in usage, try to get from meta_info (fallback for compatibility)
                        if cached_tokens == 0 and "meta_info" in prefill_json:
                            cached_tokens = prefill_json["meta_info"].get(
                                "cached_tokens", 0
                            )

                        # Inject cached_tokens into decode response
                        if cached_tokens > 0 or "usage" in ret_json:
                            if "usage" not in ret_json:
                                ret_json["usage"] = {}
                            if "prompt_tokens_details" not in ret_json["usage"]:
                                ret_json["usage"]["prompt_tokens_details"] = {}
                            ret_json["usage"]["prompt_tokens_details"][
                                "cached_tokens"
                            ] = cached_tokens
                else:
                    # For non-logprob requests, also get prefill response to merge cached_tokens
                    prefill_json = await prefill_response.json()
                    ret_json = await decode_response.json()
                    # Merge cached_tokens from both prefill and decode responses
                    if "meta_info" in ret_json and "meta_info" in prefill_json:
                        prefill_cached = prefill_json["meta_info"].get(
                            "cached_tokens", 0
                        )
                        ret_json["meta_info"]["cached_tokens"] = prefill_cached

            else:
                ret_json = "profile"

            return ORJSONResponse(
                content=ret_json,
                status_code=decode_response.status,
            )

    @staticmethod
    async def _gather_with_disconnect(tasks, raw_request):
        gather_task = asyncio.gather(*tasks)
        try:
            while True:
                done, _ = await asyncio.wait({gather_task}, timeout=1.0)
                if gather_task in done:
                    return gather_task.result()
                if raw_request is not None and await raw_request.is_disconnected():
                    for task in tasks:
                        task.cancel()
                    gather_task.cancel()
                    with contextlib.suppress(asyncio.CancelledError, Exception):
                        await gather_task
                    raise asyncio.CancelledError
        finally:
            if not gather_task.done():
                gather_task.cancel()
            for task in tasks:
                if not task.done():
                    task.cancel()

    async def generate_stream(
        self, modified_request, prefill_server, decode_server, endpoint="generate"
    ):
        assert endpoint[0] != "/", f"Endpoint should not start with '/': {endpoint}"

        async def stream_results():
            async with aiohttp.ClientSession(
                timeout=aiohttp.ClientTimeout(
                    total=3600
                )  # Add timeout for request reliability
            ) as session:
                # Create the tasks for both prefill and decode requests
                tasks = [
                    session.post(f"{prefill_server}/{endpoint}", json=modified_request),
                    session.post(f"{decode_server}/{endpoint}", json=modified_request),
                ]
                # Wait for both responses to complete. Since this is streaming, they return immediately.
                prefill_response, decode_response = await asyncio.gather(*tasks)

                if (
                    modified_request.get("return_logprob", False)
                    or self.enable_cache_report
                ):
                    prefill_chunks = []
                    cached_from_prefill = {}
                    prefill_usage = None
                    async for chunk in prefill_response.content:
                        prefill_chunks.append(chunk)

                    # Process all prefill chunks, collect cached_tokens and first chunk's logprobs
                    first_prefill_chunk_json = None
                    for chunk in prefill_chunks:
                        decoded_chunk = chunk.decode("utf-8")
                        if (
                            decoded_chunk
                            and decoded_chunk.startswith("data:")
                            and "[DONE]" not in decoded_chunk
                        ):
                            chunk_json = orjson.loads(decoded_chunk[5:].strip("\n"))

                            # Collect first chunk's logprobs (for existing logic)
                            if first_prefill_chunk_json is None:
                                first_prefill_chunk_json = chunk_json

                            # Collect cached_tokens from usage (from new usage field)
                            if "usage" in chunk_json and self.enable_cache_report:
                                prefill_usage = chunk_json.get("usage")

                            # Collect each choice's cached_tokens (from meta_info, for compatibility)
                            if "choices" in chunk_json:
                                for choice in chunk_json["choices"]:
                                    if (
                                        "index" in choice
                                        and "meta_info" in choice
                                        and "cached_tokens" in choice["meta_info"]
                                    ):
                                        idx = choice["index"]
                                        cached_from_prefill[idx] = choice["meta_info"][
                                            "cached_tokens"
                                        ]

                    async for chunk in decode_response.content:
                        # Note: This is inefficient
                        # merge prefill input_token_logprobs, output_token_logprobs to decode
                        decoded_chunk = chunk.decode("utf-8")
                        if (
                            decoded_chunk
                            and decoded_chunk.startswith("data:")
                            and "[DONE]" not in decoded_chunk
                        ):
                            ret_json = orjson.loads(decoded_chunk[5:].strip("\n"))

                            # Merge logprobs (existing logic)
                            if (
                                first_prefill_chunk_json
                                and "meta_info" in ret_json
                                and "input_token_logprobs" in ret_json["meta_info"]
                            ):
                                ret_json["meta_info"]["input_token_logprobs"] = (
                                    first_prefill_chunk_json["meta_info"][
                                        "input_token_logprobs"
                                    ]
                                    + ret_json["meta_info"]["input_token_logprobs"]
                                )

                            # Check if it's the last chunk (has usage) and inject cached_tokens
                            if (
                                "usage" in ret_json
                                and ret_json["usage"] is not None
                                and self.enable_cache_report
                            ):
                                # Extract cached_tokens from prefill's usage
                                cached_tokens = 0
                                if (
                                    prefill_usage
                                    and "prompt_tokens_details" in prefill_usage
                                ):
                                    if (
                                        prefill_usage["prompt_tokens_details"]
                                        is not None
                                    ):
                                        cached_tokens = prefill_usage[
                                            "prompt_tokens_details"
                                        ].get("cached_tokens", 0)

                                # If not obtained from usage, try from cached_from_prefill (compatibility)
                                if cached_tokens == 0 and cached_from_prefill:
                                    cached_tokens = cached_from_prefill.get(0, 0)

                                # Inject cached_tokens into decode response's usage
                                if cached_tokens > 0 or self.enable_cache_report:
                                    if "prompt_tokens_details" not in ret_json["usage"]:
                                        ret_json["usage"]["prompt_tokens_details"] = {}
                                    ret_json["usage"]["prompt_tokens_details"][
                                        "cached_tokens"
                                    ] = cached_tokens

                            yield b"data: " + orjson.dumps(ret_json) + b"\n\n"
                        elif decoded_chunk.strip() == "[DONE]":
                            # Properly format [DONE] message
                            yield b"data: [DONE]\n\n"
                        elif decoded_chunk.strip():
                            # Only yield non-empty chunks to avoid HTTP chunked encoding issues
                            yield chunk
                else:
                    async for chunk in decode_response.content:
                        # Only yield non-empty chunks to avoid HTTP chunked encoding issues
                        if chunk and chunk.strip():
                            yield chunk

        return StreamingResponse(
            stream_results(),
            media_type="text/event-stream",
        )

    async def profile(self, prefill_server, decode_server, endpoint="start_profile"):
        assert endpoint[0] != "/", f"Endpoint should not start with '/': {endpoint}"
        async with aiohttp.ClientSession(
            timeout=aiohttp.ClientTimeout(
                total=3600
            )  # Add timeout for request reliability
        ) as session:
            tasks = [
                session.post(f"{prefill_server}/{endpoint}"),
                session.post(f"{decode_server}/{endpoint}"),
            ]
            # Wait for both responses to complete.
            await asyncio.gather(*tasks)

            return Response(
                content=(
                    "Start profiling.\n"
                    if endpoint == "start_profile"
                    else "Stop profiling. This will take some time.\n"
                ),
                status_code=200,
            )


app = FastAPI()
load_balancer: Optional[MiniLoadBalancer] = None


@app.get("/health")
async def health_check():
    """Simple health check that always returns 200."""
    return Response(status_code=200)


@app.get("/health_generate")
async def health_check_generate():
    """Health check for generate that verifies backend readiness."""
    if load_balancer is None:
        return Response(status_code=503)  # Service Unavailable

    prefill_servers, decode_servers = (
        load_balancer.prefill_servers,
        load_balancer.decode_servers,
    )
    async with aiohttp.ClientSession() as session:
        # Create the tasks
        tasks = []
        for server in chain(prefill_servers, decode_servers):
            tasks.append(session.get(f"{server}/health_generate"))
        for response in asyncio.as_completed(tasks):
            await response
    return Response(status_code=200)


@app.post("/flush_cache")
async def flush_cache():
    prefill_servers, decode_servers = (
        load_balancer.prefill_servers,
        load_balancer.decode_servers,
    )
    async with aiohttp.ClientSession() as session:
        # Create the tasks
        tasks = []
        for server in chain(prefill_servers, decode_servers):
            tasks.append(session.post(f"{server}/flush_cache"))
        for response in asyncio.as_completed(tasks):
            await response
    return Response(status_code=200)


@app.get("/get_server_info")
async def get_server_info():
    prefill_servers, decode_servers = (
        load_balancer.prefill_servers,
        load_balancer.decode_servers,
    )
    prefill_infos = []
    decode_infos = []
    async with aiohttp.ClientSession() as session:
        for server in prefill_servers:
            server_info = await session.get(f"{server}/get_server_info")
            prefill_infos.append(await server_info.json())
        for server in decode_servers:
            server_info = await session.get(f"{server}/get_server_info")
            decode_infos.append(await server_info.json())

    return {"prefill": prefill_infos, "decode": decode_infos}


@app.get("/get_model_info")
async def get_model_info():
    # Dummy model information
    model_info = {
        "model_path": "/path/to/dummy/model",
        "tokenizer_path": "/path/to/dummy/tokenizer",
        "is_generation": True,
        "preferred_sampling_params": {"temperature": 0.7, "max_new_tokens": 128},
    }
    return ORJSONResponse(content=model_info)


@app.api_route("/start_profile", methods=["GET", "POST"])
async def start_profile():
    """Start profiling."""
    prefill_server, _, decode_server = load_balancer.select_pair_round_robin()
    return await load_balancer.profile(prefill_server, decode_server, "start_profile")


@app.api_route("/stop_profile", methods=["GET", "POST"])
async def stop_profile():
    """Stop profiling."""
    prefill_server, _, decode_server = load_balancer.select_pair_round_robin()
    return await load_balancer.profile(prefill_server, decode_server, "stop_profile")


@app.post("/generate")
async def handle_generate_request(request_data: dict, raw_request: Request):
    # Log incoming request
    logger.debug(
        "LB received generate request: stream=%s, text_length=%s",
        request_data.get("stream", False),
        (
            len(request_data.get("text", ""))
            if isinstance(request_data.get("text"), str)
            else "batch"
        ),
    )

    prefill_server, bootstrap_port, decode_server = (
        load_balancer.select_pair_round_robin()
    )

    # Parse and transform prefill_server for bootstrap data
    parsed_url = urllib.parse.urlparse(prefill_server)
    hostname = parsed_url.hostname
    modified_request = request_data.copy()

    batch_size = _get_request_batch_size(modified_request)
    if batch_size is not None:
        modified_request.update(
            {
                "bootstrap_host": [hostname] * batch_size,
                "bootstrap_port": [bootstrap_port] * batch_size,
                "bootstrap_room": [
                    _generate_bootstrap_room()
                    for _ in range(batch_size)
                    # await load_balancer.get_bootstrap_room() for _ in range(batch_size)
                ],
            }
        )
    else:
        modified_request.update(
            {
                "bootstrap_host": hostname,
                "bootstrap_port": bootstrap_port,
                "bootstrap_room": _generate_bootstrap_room(),
                # "bootstrap_room": await load_balancer.get_bootstrap_room(),
            }
        )

    if request_data.get("stream", False):
        return await load_balancer.generate_stream(
            modified_request, prefill_server, decode_server, "generate"
        )
    else:
        return await load_balancer.generate(
            modified_request, prefill_server, decode_server, "generate"
        )


async def _forward_to_backend(request_data: dict, endpoint_name: str, raw_request: Request):
    prefill_server, bootstrap_port, decode_server = (
        load_balancer.select_pair_round_robin()
    )

    # Parse and transform prefill_server for bootstrap data
    parsed_url = urllib.parse.urlparse(prefill_server)
    hostname = parsed_url.hostname
    modified_request = request_data.copy()
    modified_request.update(
        {
            "bootstrap_host": hostname,
            "bootstrap_port": bootstrap_port,
            "bootstrap_room": _generate_bootstrap_room(),
        }
    )

    if request_data.get("stream", False):
        return await load_balancer.generate_stream(
            modified_request,
            prefill_server,
            decode_server,
            endpoint=endpoint_name,
        )
    else:
        return await load_balancer.generate(
            modified_request,
            prefill_server,
            decode_server,
            endpoint=endpoint_name,
            raw_request=raw_request,
        )


@app.post("/v1/chat/completions")
async def handle_chat_completion_request(request_data: dict, raw_request: Request):
    return await _forward_to_backend(request_data, "v1/chat/completions", raw_request)


@app.post("/v1/completions")
async def handle_completion_request(request_data: dict, raw_request: Request):
    return await _forward_to_backend(request_data, "v1/completions", raw_request)


def _generate_bootstrap_room():
    return random.randint(0, 2**63 - 1)


# We may utilize `GenerateReqInput`'s logic later
def _get_request_batch_size(request):
    if (text := request.get("text")) is not None:
        return None if isinstance(text, str) else len(text)
    if (input_ids := request.get("input_ids")) is not None:
        return None if isinstance(input_ids[0], int) else len(input_ids)
    return None


@app.get("/v1/models")
async def get_models():
    prefill_server = load_balancer.prefill_servers[0]  # Get the first prefill server
    async with aiohttp.ClientSession() as session:
        try:
            response = await session.get(f"{prefill_server}/v1/models")
            if response.status != 200:
                raise HTTPException(
                    status_code=response.status,
                    detail=f"Prefill server error: Status {response.status}",
                )
            return ORJSONResponse(content=await response.json())
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))


@app.post("/register")
async def register(obj: PDRegistryRequest):
    if obj.mode == "prefill":
        load_balancer.add_prefill_server(
            PrefillConfig(obj.registry_url, obj.bootstrap_port)
        )
        logger.info(
            "Registered prefill server: %s with bootstrap port: %s",
            obj.registry_url,
            obj.bootstrap_port,
        )
    elif obj.mode == "decode":
        load_balancer.add_decode_server(obj.registry_url)
        logger.info("Registered decode server: %s", obj.registry_url)
    else:
        raise HTTPException(
            status_code=400,
            detail="Invalid mode. Must be either PREFILL or DECODE.",
        )

    logger.info(
        "#Prefill servers: %s, #Decode servers: %s",
        len(load_balancer.prefill_configs),
        len(load_balancer.decode_servers),
    )

    return Response(status_code=200)


def run(prefill_configs, decode_addrs, host, port, enable_cache_report=False):
    global load_balancer
    load_balancer = MiniLoadBalancer(prefill_configs, decode_addrs, enable_cache_report)
    uvicorn.run(app, host=host, port=port)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Mini Load Balancer Server")
    parser.add_argument(
        "--prefill", type=str, default=[], nargs="+", help="URLs for prefill servers"
    )
    parser.add_argument(
        "--decode", type=str, default=[], nargs="+", help="URLs for decode servers"
    )
    parser.add_argument(
        "--prefill-bootstrap-ports",
        type=int,
        nargs="+",
        help="Bootstrap ports for prefill servers",
    )
    parser.add_argument(
        "--host", default="0.0.0.0", help="Host to bind the server (default: 0.0.0.0)"
    )
    parser.add_argument(
        "--port", type=int, default=8000, help="Port to bind the server (default: 8000)"
    )
    parser.add_argument(
        "--enable-cache-report",
        action="store_true",
        help="Enable cached tokens report in usage information",
    )
    args = parser.parse_args()

    bootstrap_ports = args.prefill_bootstrap_ports
    if bootstrap_ports is None:
        bootstrap_ports = [None] * len(args.prefill)
    elif len(bootstrap_ports) == 1:
        bootstrap_ports = bootstrap_ports * len(args.prefill)
    else:
        if len(bootstrap_ports) != len(args.prefill):
            raise ValueError(
                "Number of prefill URLs must match number of bootstrap ports"
            )

    prefill_configs = [
        PrefillConfig(url, port) for url, port in zip(args.prefill, bootstrap_ports)
    ]

    run(prefill_configs, args.decode, args.host, args.port, args.enable_cache_report)
