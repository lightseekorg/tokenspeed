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

"""Extend the canonical graph trace with weight sessions and actual pool ownership."""

import hashlib
import importlib.util
import sys
from functools import wraps
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from test_timeline import install

install()
spec = importlib.util.spec_from_file_location(
    "tokenspeed_test_canonical_ascend_trace", ROOT / "trace/sitecustomize.py"
)
trace_module = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = trace_module
spec.loader.exec_module(trace_module)


def trace_weight_session(name, method):
    @wraps(method)
    def observed(model, *args, **kwargs):
        trace_module.record(name)
        return method(model, *args, **kwargs)

    return observed


def trace_disk_update(method):
    def address_hash(model):
        addresses = [
            (name, value.data_ptr()) for name, value in model.named_parameters()
        ]
        return hashlib.sha256(repr(addresses).encode()).hexdigest()

    @wraps(method)
    def observed(runner, obj, *args, **kwargs):
        import os

        import msgspec

        rank = runner.mapping.rank
        bad_rank_path = os.environ.get("TS_WEIGHT_BAD_RANK1")
        if rank == 1 and bad_rank_path:
            # Preserve the original explicit malformed-rank fixture injection.
            obj = msgspec.structs.replace(obj, model_path=bad_rank_path)
        fields = {
            "rank": rank,
            "checkpoint": str(obj.model_path),
            "load_format": obj.load_format,
        }
        trace_module.record(
            "disk_update_enter", parameter_va=address_hash(runner.model), **fields
        )
        success, message = method(runner, obj, *args, **kwargs)
        trace_module.record(
            "disk_update_exit",
            parameter_va=address_hash(runner.model),
            success=success,
            message=message,
            **fields
        )
        return success, message

    return observed


class WeightTracingLoader(trace_module.TracingLoader):
    def exec_module(self, module):
        if module.__name__ == "tokenspeed.runtime.models.base.causal_lm":
            self.loader.exec_module(module)
            model = module.BaseCausalLM
            for name in ("begin_weight_update", "end_weight_update"):
                setattr(model, name, trace_weight_session(name, getattr(model, name)))
        elif module.__name__ == "tokenspeed.runtime.execution.model_runner":
            self.loader.exec_module(module)
            runner = module.ModelRunner
            runner.update_weights_from_disk = trace_disk_update(
                runner.update_weights_from_disk
            )
        else:
            super().exec_module(module)
            if (
                module.__name__
                == "tokenspeed.runtime.utils.ascend_memory_saver_adapter"
            ):
                from test_persistent_model_witness import instrument

                instrument(module, trace_module.record)


# Finder methods retain the real imported module's globals, including this
# loader assignment. Its original executable guard is unchanged.
trace_module.TracingLoader = WeightTracingLoader
trace_module.TracingFinder.targets.update(
    {
        "tokenspeed.runtime.models.base.causal_lm",
        "tokenspeed.runtime.execution.model_runner",
    }
)
