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

"""Compare TP and DCP model outputs through the production HTTP path.

Run once against TP with ``--write-reference``, then against each DCP degree
with ``--reference``. Both a cold request and an identical prefix-cache replay
are captured. Token IDs must be exact and the sampled tokens' log probabilities
are compared numerically. Final hidden states are also compared when the server
returns them; pass ``--require-hidden-states`` when validating a response path
that promises that optional field.
"""

from __future__ import annotations

import argparse
import json
import math
import urllib.request
from pathlib import Path
from typing import Any


def _request(
    url: str,
    input_ids: list[int],
    decode_tokens: int,
    model: str | None,
) -> dict[str, Any]:
    payload = {
        "input_ids": input_ids,
        "sampling_params": {
            "temperature": 0.0,
            "max_new_tokens": decode_tokens,
            "ignore_eos": True,
        },
        "return_logprob": True,
        "return_hidden_states": True,
    }
    if model is not None:
        # The model gateway routes raw /generate requests by served-model name.
        # Direct scheduler endpoints do not require this field.
        payload["model"] = model
    body = json.dumps(payload).encode()
    request = urllib.request.Request(
        f"{url.rstrip('/')}/generate",
        data=body,
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(request, timeout=1800) as response:
        result = json.load(response)
    if isinstance(result, list):
        if len(result) != 1:
            raise AssertionError(f"expected one result, got {len(result)}")
        return result[0]
    return result


def _field(result: dict[str, Any], name: str) -> Any:
    if name in result:
        return result[name]
    meta_info = result.get("meta_info", {})
    if name in meta_info:
        return meta_info[name]
    if name == "output_hidden_states":
        return meta_info.get("hidden_states")
    return None


def _flatten_numbers(value: Any) -> list[float]:
    if value is None:
        return []
    if isinstance(value, (int, float)):
        return [float(value)]
    result = []
    for item in value:
        result.extend(_flatten_numbers(item))
    return result


def _sampled_logprob_values(value: Any) -> list[float]:
    """Extract values from SGLang ``[logprob, token_id, ...]`` entries."""
    if value is None:
        return []
    values = []
    for entry in value:
        if not isinstance(entry, list) or not entry:
            raise AssertionError(f"invalid sampled-logprob entry: {entry!r}")
        values.append(float(entry[0]))
    return values


def _compare_numbers(
    label: str,
    actual: Any,
    expected: Any,
    *,
    atol: float,
    rtol: float,
) -> None:
    actual_values = _flatten_numbers(actual)
    expected_values = _flatten_numbers(expected)
    if len(actual_values) != len(expected_values):
        raise AssertionError(
            f"{label} length differs: {len(actual_values)} != {len(expected_values)}"
        )
    worst = (0.0, -1)
    mismatches = []
    for index, (got, want) in enumerate(zip(actual_values, expected_values)):
        error = abs(got - want)
        if error > worst[0]:
            worst = (error, index)
        if not math.isclose(got, want, abs_tol=atol, rel_tol=rtol):
            mismatches.append((index, got, want, error))
    print(
        f"{label}: {len(actual_values)} values, max_abs={worst[0]:.6g}, "
        f"mismatches={len(mismatches)}"
    )
    if mismatches:
        index, got, want, error = mismatches[0]
        raise AssertionError(
            f"{label}[{index}] differs: {got} != {want} "
            f"(abs={error}, max_abs={worst[0]}, mismatches={len(mismatches)}, "
            f"atol={atol}, rtol={rtol})"
        )


def _compare(
    actual: list[dict[str, Any]], expected: list[dict[str, Any]], args
) -> None:
    failures = []
    for pass_index, (got, want) in enumerate(zip(actual, expected)):
        label = "cold" if pass_index == 0 else "prefix_replay"
        got_ids = _field(got, "output_ids")
        want_ids = _field(want, "output_ids")
        if got_ids != want_ids:
            failures.append(f"{label} output token IDs differ")
        got_hidden = _field(got, "output_hidden_states")
        want_hidden = _field(want, "output_hidden_states")
        if not _flatten_numbers(got_hidden) or not _flatten_numbers(want_hidden):
            if args.require_hidden_states:
                failures.append(f"{label}: server did not return output_hidden_states")
            else:
                print(f"{label}.hidden_states: unavailable (optional)")
        else:
            try:
                _compare_numbers(
                    f"{label}.hidden_states",
                    got_hidden,
                    want_hidden,
                    atol=args.hidden_atol,
                    rtol=args.rtol,
                )
            except AssertionError as error:
                failures.append(str(error))
        try:
            _compare_numbers(
                f"{label}.sampled_logprobs",
                _sampled_logprob_values(_field(got, "output_token_logprobs")),
                _sampled_logprob_values(_field(want, "output_token_logprobs")),
                atol=args.logprob_atol,
                rtol=args.rtol,
            )
        except AssertionError as error:
            failures.append(str(error))
    if failures:
        raise AssertionError("\n".join(failures))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", default="http://127.0.0.1:30000")
    parser.add_argument("--model")
    parser.add_argument("--write-reference", type=Path)
    parser.add_argument("--reference", type=Path)
    parser.add_argument("--actual-output", type=Path)
    parser.add_argument("--prompt-tokens", type=int, default=4096)
    parser.add_argument("--decode-tokens", type=int, default=16)
    parser.add_argument("--vocab-size", type=int, default=160000)
    parser.add_argument("--logprob-atol", type=float, default=2e-2)
    parser.add_argument("--hidden-atol", type=float, default=3e-2)
    parser.add_argument("--rtol", type=float, default=3e-2)
    parser.add_argument("--require-hidden-states", action="store_true")
    args = parser.parse_args()
    if (args.write_reference is None) == (args.reference is None):
        parser.error("pass exactly one of --write-reference or --reference")
    input_ids = [
        1 + (index * 7919) % (args.vocab_size - 1)
        for index in range(args.prompt_tokens)
    ]
    # The second identical request must traverse the cached-prefix replay path.
    results = [
        _request(args.url, input_ids, args.decode_tokens, args.model) for _ in range(2)
    ]
    if args.actual_output is not None:
        args.actual_output.write_text(json.dumps(results))
    if args.write_reference is not None:
        args.write_reference.write_text(json.dumps(results))
        print(f"wrote TP reference to {args.write_reference}")
        return
    expected = json.loads(args.reference.read_text())
    _compare(results, expected, args)
    print("TP/DCP activation and sampled-logprob parity passed")


if __name__ == "__main__":
    main()
