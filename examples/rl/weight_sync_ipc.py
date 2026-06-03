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

"""RL weight sync over CUDA-IPC (colocated trainer + inference, shared GPUs).

Trainer-side client for the weight-transfer API (CUDA-IPC backend). The HTTP
surface follows the contract common RL trainers speak.

Run a TokenSpeed server with the IPC backend:

    TOKENSPEED_ALLOW_INSECURE_SERIALIZATION=1 \\
    tokenspeed serve <model> --host 0.0.0.0 --port 8000 \\
        --weight-transfer-config '{"backend":"ipc"}'

The trainer exports CUDA-IPC handles for its tensors and ships them as a
base64-encoded pickle under ``update_info.ipc_handles_pickled``. The inference
worker opens the handles on the shared GPU and copies the weights -- no NCCL
group, no init step (IPC ``init_info`` is a no-op).

NOTE: deserializing ``ipc_handles_pickled`` requires
``TOKENSPEED_ALLOW_INSECURE_SERIALIZATION=1`` on the server (it unpickles the
payload).
"""

import argparse
import base64
import pickle

from weight_sync_nccl import WeightSyncClient


def export_ipc_handles(model):
    """Trainer side: build per-parameter CUDA-IPC handles.

    Pseudocode -- wire this to your training framework::

        from torch.multiprocessing.reductions import reduce_tensor
        names, dtype_names, shapes, handles = [], [], [], []
        for name, p in model.named_parameters():
            t = p.detach().contiguous()
            uuid = str(torch.cuda.get_device_properties(t.device).uuid)
            _, ipc_args = reduce_tensor(t)
            names.append(name)
            dtype_names.append(str(t.dtype).split(".")[-1])
            shapes.append(list(t.shape))
            handles.append({uuid: ipc_args})
        return names, dtype_names, shapes, handles
    """
    raise NotImplementedError("Wire export_ipc_handles to your trainer")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", default="http://127.0.0.1:8000")
    args = parser.parse_args()

    client = WeightSyncClient(args.url)

    # IPC needs no process group; init is a no-op but still called for parity.
    client.init_engine({})

    # Replace with real handles from export_ipc_handles(model).
    names = ["model.embed_tokens.weight"]
    dtype_names = ["bfloat16"]
    shapes = [[32000, 4096]]
    ipc_handles = [{"<gpu-uuid>": ("...reduce_tensor args...",)}]
    ipc_handles_pickled = base64.b64encode(pickle.dumps(ipc_handles)).decode("utf-8")

    client.pause(mode="keep")
    try:
        client.start_update(is_checkpoint_format=False)
        client.update_weights(
            {
                "names": names,
                "dtype_names": dtype_names,
                "shapes": shapes,
                "ipc_handles_pickled": ipc_handles_pickled,
                "packed": False,
            }
        )
        client.finish_update()
    finally:
        client.resume()

    print("weight sync cycle complete")


if __name__ == "__main__":
    main()
