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

"""Prepare pre-tokenized multimodal requests at scheduler ingress."""

from tokenspeed.runtime.multimodal.mrope import prepare_mrope_inputs


def prepare_tokenized_multimodal_request(
    request,
    *,
    hf_config,
    model_is_multimodal: bool,
    language_model_only: bool,
    disaggregation_mode: str,
) -> None:
    """Apply the same decode-role rules as native InputProcessor ingress."""
    if request.multimodal_inputs is None:
        return
    if request.input_ids_unpadded is None:
        raise ValueError("multimodal input requires input_ids_unpadded")
    if disaggregation_mode == "decode" and language_model_only and model_is_multimodal:
        request.input_ids = list(request.input_ids_unpadded)
        request.multimodal_inputs = None
        return
    prepare_mrope_inputs(
        hf_config, request.input_ids_unpadded, request.multimodal_inputs
    )
