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

"""Validate TokenSpeed model and draft weight policies before startup."""


def validate_memory_saver_scope(
    *,
    has_draft_weights: bool,
    weight_policy: str | None,
    draft_policy: str | None,
) -> None:
    """Require a restorable parameter policy before allocating managed storage."""
    if weight_policy not in ("cpu_backup", "discard"):
        raise ValueError("Ascend memory saver requires cpu_backup or discard weights")
    if (
        has_draft_weights
        and weight_policy == "discard"
        and draft_policy
        not in (
            "retain",
            "refresh",
        )
    ):
        raise ValueError(
            "Ascend weight discard with a draft checkpoint requires an explicit "
            "--model-update-draft-weights policy"
        )
