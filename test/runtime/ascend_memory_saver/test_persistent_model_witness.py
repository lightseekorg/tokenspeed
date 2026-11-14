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

"""Read-only production parameter/PTA/native ownership trace, installed in ts-host."""

import time
from functools import wraps

PARAMETERS = None


def instrument(module, record):
    plane = module.AscendMemorySaverAdapter
    original_validate = plane.validate_startup

    def snapshot(adapter, operation, when, tag):
        import torch

        state = adapter._state()
        if PARAMETERS is None:
            raise RuntimeError(
                "No real startup model parameters available for the ownership witness"
            )
        pools = [
            {
                "tag": key.tag,
                "policy": key.policy.value,
                "scope": key.scope,
                "id": list(owned.pool.id),
            }
            for key, owned in state.state.pools.items()
        ]
        segments = [
            {
                "address": segment["address"],
                "total_size": segment["total_size"],
                "segment_pool_id": list(segment["segment_pool_id"]),
            }
            for segment in torch.npu.memory_snapshot()
            if segment["device"] == state.device
        ]
        record(
            "persistent-ownership",
            rank=torch.distributed.get_rank(),
            operation=operation,
            when=when,
            tag=tag,
            monotonic_ns=time.monotonic_ns(),
            parameters=PARAMETERS,
            pools=pools,
            PTA_segments=segments,
            native=state.native.stats(),
        )

    @wraps(original_validate)
    def observe_validate(self, *args, **kwargs):
        global PARAMETERS
        result = original_validate(self, *args, **kwargs)
        parameters = []
        for role, model in kwargs["models"]:
            for name, value in model.named_parameters():
                if value.device.type == "npu" and value.numel():
                    storage = value.untyped_storage()
                    parameters.append(
                        {
                            "name": f"{role}.{name}",
                            "storage_ptr": storage.data_ptr(),
                            "storage_bytes": storage.nbytes(),
                        }
                    )
        PARAMETERS = parameters
        snapshot(self, "validate_startup", "after", None)
        return result

    plane.validate_startup = observe_validate
    for name in ("pause", "resume"):
        original = getattr(plane, name)

        def wrap(method, operation):
            @wraps(method)
            def observed(self, *args, **kwargs):
                tag = args[0] if args else kwargs.get("tag")
                snapshot(self, operation, "before", tag)
                result = method(self, *args, **kwargs)
                snapshot(self, operation, "after", tag)
                return result

            return observed

        setattr(plane, name, wrap(original, name))
