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

"""RL weight sync over NCCL (disaggregated trainer + inference).

This is the **trainer-side client** for TokenSpeed's vLLM-compatible weight
transfer API. Because the HTTP surface byte-matches vLLM, the exact same client
drives a vLLM server -- only the base URL changes.

Run a TokenSpeed server with the control plane enabled:

    tokenspeed serve <model> --host 0.0.0.0 --port 8000 \\
        --enable-weight-transfer --weight-transfer-config '{"backend":"nccl"}'

then point this script at it:

    python examples/rl/weight_sync_nccl.py --url http://127.0.0.1:8000

The control plane only carries metadata. The trainer broadcasts the actual
tensors over a NCCL group it shares with the inference workers (see
``broadcast_weights`` below for where that goes).
"""

import argparse

import requests


class WeightSyncClient:
    """Minimal client for the vLLM-compatible weight-transfer endpoints."""

    def __init__(self, url: str, timeout: float = 300.0):
        self.base = url.rstrip("/")
        self.timeout = timeout

    def _post(self, path: str, **kwargs) -> dict:
        r = requests.post(f"{self.base}{path}", timeout=self.timeout, **kwargs)
        r.raise_for_status()
        return r.json()

    def _get(self, path: str, **kwargs) -> dict:
        r = requests.get(f"{self.base}{path}", timeout=self.timeout, **kwargs)
        r.raise_for_status()
        return r.json()

    def get_world_size(self, include_dp: bool = True) -> int:
        return self._get(
            "/get_world_size", params={"include_dp": str(include_dp).lower()}
        )["world_size"]

    def init_engine(self, init_info: dict) -> dict:
        return self._post("/init_weight_transfer_engine", json={"init_info": init_info})

    def start_update(self, is_checkpoint_format: bool = True) -> dict:
        return self._post(
            "/start_weight_update", json={"is_checkpoint_format": is_checkpoint_format}
        )

    def update_weights(self, update_info: dict) -> dict:
        return self._post("/update_weights", json={"update_info": update_info})

    def finish_update(self) -> dict:
        return self._post("/finish_weight_update", json={})

    def pause(self, mode: str = "keep", clear_cache: bool = True) -> dict:
        return self._post(
            "/pause", params={"mode": mode, "clear_cache": str(clear_cache).lower()}
        )

    def resume(self) -> dict:
        return self._post("/resume", json={})

    def is_paused(self) -> bool:
        return self._get("/is_paused")["is_paused"]


def broadcast_weights(model, group) -> None:
    """Trainer side: broadcast each parameter over the shared NCCL group.

    Pseudocode -- wire this to your training framework. Must be called *between*
    ``start_update`` and ``finish_update``, in the same parameter order as the
    ``update_weights`` metadata, with the trainer as src rank 0::

        for _, tensor in model.named_parameters():
            torch.distributed.broadcast(tensor.data, src=0, group=group)
    """
    raise NotImplementedError("Wire broadcast_weights to your trainer's NCCL group")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", default="http://127.0.0.1:8000")
    parser.add_argument("--master-address", default="127.0.0.1")
    parser.add_argument("--master-port", type=int, default=29500)
    args = parser.parse_args()

    client = WeightSyncClient(args.url)

    # 1) Size the NCCL group: trainer ranks + inference workers.
    num_inference_workers = client.get_world_size(include_dp=True)
    trainer_ranks = 1
    world_size = trainer_ranks + num_inference_workers
    print(f"inference workers={num_inference_workers}, NCCL world_size={world_size}")

    # 2) Build the group once. rank_offset shifts inference ranks past the
    #    trainer ranks (trainer is rank 0). The trainer side joins this same
    #    group with rank 0 (not shown).
    client.init_engine(
        {
            "master_address": args.master_address,
            "master_port": args.master_port,
            "rank_offset": trainer_ranks,
            "world_size": world_size,
        }
    )

    # 3) Per RL step: pause -> start -> update(s) -> finish -> resume.
    #    Replace the metadata below with your model's real parameters.
    names = ["model.embed_tokens.weight"]
    dtype_names = ["bfloat16"]
    shapes = [[32000, 4096]]

    client.pause(mode="keep")
    try:
        client.start_update(is_checkpoint_format=True)
        client.update_weights(
            {"names": names, "dtype_names": dtype_names, "shapes": shapes}
        )
        # broadcast_weights(model, group)  # trainer broadcasts the tensors here
        client.finish_update()
    finally:
        client.resume()

    print("paused:", client.is_paused())
    print("weight sync cycle complete")


if __name__ == "__main__":
    main()
