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

"""Which logprobs an engine returns beyond the sampled tokens': the facts the
engine host's ingress checks each request's logprob asks against
(``ts-control``'s ``LogprobServability``)."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class LogprobServability:
    """Which logprobs an engine can return, as its ingress checks a request.

    Attributes:
        enable_output_logprobs: ``--enable-output-logprobs``: the sampler
            gathers the sampled tokens' logprobs.
        supports_prompt_logprobs: The engine's startup verdict
            (``DeviceSpecs.supports_prompt_logprobs``): its model keeps every
            prompt row's activations.
        logprob_order: ``--logprob-order``; the top-k alternatives need
            ``torch``.
        pipeline_parallel: The engine runs pipeline stages.
        disaggregated: The engine is one role of a disaggregated deployment.
        vocab_size: The model's vocabulary size.
    """

    enable_output_logprobs: bool
    supports_prompt_logprobs: bool
    logprob_order: str
    pipeline_parallel: bool
    disaggregated: bool
    vocab_size: int

    @classmethod
    def of(
        cls, server_args, *, supports_prompt_logprobs: bool, vocab_size: int
    ) -> LogprobServability:
        """The servability of an engine started with ``server_args``."""
        return cls(
            enable_output_logprobs=server_args.enable_output_logprobs,
            supports_prompt_logprobs=supports_prompt_logprobs,
            logprob_order=server_args.logprob_order,
            pipeline_parallel=(
                server_args.mapping is not None and server_args.mapping.has_pp
            ),
            disaggregated=server_args.disaggregation_mode != "null",
            vocab_size=vocab_size,
        )
