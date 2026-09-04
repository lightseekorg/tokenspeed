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

"""Drafter registry: resolve a speculative algorithm to its drafter class."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import torch

    from tokenspeed.runtime.execution.drafter.base import BaseDrafter

__all__ = ["get_drafter_impl"]


def get_drafter_impl(spec_algo: str, model: torch.nn.Module) -> type[BaseDrafter]:
    """Resolve the drafter class for ``spec_algo`` and a loaded draft model.

    Args:
        spec_algo: The speculative algorithm name from server args
            (``EAGLE3``, ``MTP``, ``DFLASH``, or ``DSPARK``).
        model: The loaded draft model; some algorithms route on its class.

    Returns:
        The ``BaseDrafter`` subclass to instantiate (not an instance).
    """
    # Imports are local: drafter modules pull in kernel ops and model code,
    # and this package init must stay importable from lightweight contexts.
    from tokenspeed.runtime.execution.drafter.dflash import DFlash
    from tokenspeed.runtime.execution.drafter.dflash2 import DFlash2
    from tokenspeed.runtime.execution.drafter.dspark import DSpark
    from tokenspeed.runtime.execution.drafter.eagle import Eagle
    from tokenspeed.runtime.models.inkling_nextn import (
        InklingForConditionalGenerationNextN,
    )

    DRAFTER_MAPPING = {
        "EAGLE3": Eagle,
        "MTP": Eagle,
        "DFLASH": DFlash,
        "DSPARK": DSpark,
    }

    if spec_algo == "DFLASH":
        from tokenspeed.runtime.models.dflash2 import DFlash2DraftModel

        if isinstance(model, DFlash2DraftModel):
            return DFlash2

    # "MTP" covers two algorithms:
    # (1) Eagle-like MTP (e.g. DeepSeek) stays on Eagle in eagle.py;
    # (2) Vanilla MTP (e.g. Inkling) with multi-layer weights stays on Mtp in mtp.py.
    if spec_algo == "DSPARK":
        from tokenspeed.runtime.execution.drafter.deepseek_v4_dspark import (
            DeepseekV4DSpark,
        )
        from tokenspeed.runtime.models.deepseek_v4_dspark import (
            DeepseekV4ForCausalLMDSpark,
        )

        if isinstance(model, DeepseekV4ForCausalLMDSpark):
            return DeepseekV4DSpark
    if spec_algo == "MTP" and isinstance(model, InklingForConditionalGenerationNextN):
        from tokenspeed.runtime.execution.drafter.mtp import Mtp

        return Mtp
    return DRAFTER_MAPPING[spec_algo]
