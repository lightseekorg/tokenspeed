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
with ``--reference``. The script flushes the cache, captures one cold request,
and then extends the same page-aligned prompt by one token. The second response
must report the whole shared prompt as cached; repeating an identical request
does not prove that prefix lookup ran.

Generated IDs must be exact. Sampled-token log probabilities are compared only
for the common generated prefix, where both the token and its preceding history
are identical. The HTTP API does not expose full logits or a forced-token decode
operation, so this script deliberately does not claim teacher-forced decode or
full-logit parity. Those require a test hook at ``ModelExecutor._run_target_forward``
before ``ModelExecutor._run_sampling``.
"""

from __future__ import annotations

import argparse
import json
import math
import urllib.request
from pathlib import Path
from typing import Any


def _post_json(url: str, path: str, payload: dict[str, Any]) -> Any:
    body = json.dumps(payload).encode()
    request = urllib.request.Request(
        f"{url.rstrip('/')}/{path.lstrip('/')}",
        data=body,
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(request, timeout=1800) as response:
        return json.load(response)


def _flush_cache(url: str) -> None:
    _post_json(url, "/flush_cache", {})


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
    result = _post_json(url, "/generate", payload)
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


def _sampled_logprob_entries(value: Any) -> list[tuple[float, int]]:
    """Extract and validate SGLang ``[logprob, token_id, ...]`` entries."""
    if value is None:
        return []
    values: list[tuple[float, int]] = []
    for entry in value:
        if not isinstance(entry, list) or len(entry) < 2:
            raise AssertionError(f"invalid sampled-logprob entry: {entry!r}")
        token_id = int(entry[1])
        if float(entry[1]) != token_id:
            raise AssertionError(f"non-integral sampled token id: {entry!r}")
        values.append((float(entry[0]), token_id))
    return values


def _matching_history_logprobs(
    result: dict[str, Any], output_ids: list[int], limit: int
) -> list[float]:
    entries = _sampled_logprob_entries(_field(result, "output_token_logprobs"))
    if len(entries) != len(output_ids):
        raise AssertionError(
            "sampled-logprob/output-id length differs: "
            f"{len(entries)} != {len(output_ids)}"
        )
    for index, ((_, entry_token_id), output_token_id) in enumerate(
        zip(entries, output_ids)
    ):
        if entry_token_id != output_token_id:
            raise AssertionError(
                f"sampled-logprob token id at {index} is {entry_token_id}, "
                f"output id is {output_token_id}"
            )
    return [logprob for logprob, _ in entries[:limit]]


def _common_prefix_length(left: list[int], right: list[int]) -> int:
    result = 0
    for got, want in zip(left, right):
        if got != want:
            break
        result += 1
    return result


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
    if len(actual) != len(expected):
        raise AssertionError(
            f"request pass count differs: {len(actual)} != {len(expected)}"
        )
    failures = []
    for pass_index, (got, want) in enumerate(zip(actual, expected)):
        label = "cold" if pass_index == 0 else "prefix_replay"
        got_ids = list(_field(got, "output_ids") or [])
        want_ids = list(_field(want, "output_ids") or [])
        if got_ids != want_ids:
            first_divergence = _common_prefix_length(got_ids, want_ids)
            message = (
                f"{label} output token IDs first differ at {first_divergence}: "
                f"{got_ids[first_divergence : first_divergence + 1]} != "
                f"{want_ids[first_divergence : first_divergence + 1]}"
            )
            if getattr(args, "allow_output_id_divergence", False):
                print(f"{message} (allowed after measuring TP baseline noise)")
            else:
                failures.append(message)
            print(
                f"{label}.hidden_states: skipped after token divergence "
                f"at {first_divergence}"
            )
        else:
            got_hidden = _field(got, "output_hidden_states")
            want_hidden = _field(want, "output_hidden_states")
            if not _flatten_numbers(got_hidden) or not _flatten_numbers(want_hidden):
                if args.require_hidden_states:
                    failures.append(
                        f"{label}: server did not return output_hidden_states"
                    )
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
        common_prefix = _common_prefix_length(got_ids, want_ids)
        if common_prefix:
            try:
                _compare_numbers(
                    f"{label}.matching_history_sampled_logprobs",
                    _matching_history_logprobs(got, got_ids, common_prefix),
                    _matching_history_logprobs(want, want_ids, common_prefix),
                    atol=args.logprob_atol,
                    rtol=args.rtol,
                )
            except AssertionError as error:
                failures.append(str(error))
        else:
            print(f"{label}.sampled_logprobs: no comparable generated prefix")
    if failures:
        raise AssertionError("\n".join(failures))


def _cached_tokens(result: dict[str, Any]) -> int:
    value = _field(result, "cached_tokens")
    return 0 if value is None else int(value)


def _validate_cache_evidence(
    cold: dict[str, Any], replay: dict[str, Any], prompt_tokens: int
) -> None:
    cold_cached_tokens = _cached_tokens(cold)
    if cold_cached_tokens != 0:
        raise AssertionError(
            f"cold request reported {cold_cached_tokens} cached tokens after flush"
        )
    replay_cached_tokens = _cached_tokens(replay)
    if replay_cached_tokens < prompt_tokens:
        raise AssertionError(
            "extended request did not report the full shared prompt as a "
            f"prefix-cache hit ({replay_cached_tokens} < {prompt_tokens}); "
            "start the server with prefix caching enabled"
        )


def _fixed_context_prefill_requests(
    url: str,
    input_ids: list[int],
    continuation: list[int],
    model: str | None,
) -> list[dict[str, Any]]:
    """Probe next-token argmaxes from fixed contexts through extend/prefill."""
    return [
        _request(url, input_ids + continuation[:step], 1, model)
        for step in range(len(continuation))
    ]


def _compare_fixed_context_prefill(
    actual: list[dict[str, Any]],
    expected: list[dict[str, Any]],
    args,
) -> None:
    if len(actual) != len(expected):
        raise AssertionError(
            "fixed-context prefill step count differs: "
            f"{len(actual)} != {len(expected)}"
        )
    failures = []
    for step, (got, want) in enumerate(zip(actual, expected)):
        got_ids = list(_field(got, "output_ids") or [])
        want_ids = list(_field(want, "output_ids") or [])
        if got_ids != want_ids:
            failures.append(
                f"fixed_context_prefill[{step}] next token differs: "
                f"{got_ids} != {want_ids}"
            )
            continue
        try:
            _compare_numbers(
                f"fixed_context_prefill[{step}].sampled_logprob",
                _matching_history_logprobs(got, got_ids, len(got_ids)),
                _matching_history_logprobs(want, want_ids, len(want_ids)),
                atol=args.logprob_atol,
                rtol=args.rtol,
            )
        except AssertionError as error:
            failures.append(str(error))
    if failures:
        raise AssertionError("\n".join(failures))


def _choose_replay_suffix(
    prompt_tokens: int,
    vocab_size: int,
    cold_output_ids: list[int],
) -> int:
    token = 1 + (prompt_tokens * 104729) % (vocab_size - 1)
    if cold_output_ids and token == cold_output_ids[0]:
        token = 1 + token % (vocab_size - 1)
    return token


def _validate_reference_metadata(metadata: dict[str, Any], args) -> None:
    expected = {
        "schema_version": 2,
        "source_id": args.source_id,
        "checkpoint": args.checkpoint,
        "model": args.model,
        "prompt_tokens": args.prompt_tokens,
        "decode_tokens": args.decode_tokens,
        "vocab_size": args.vocab_size,
        "prefix_granularity": args.prefix_granularity,
        "fixed_context_prefill_probes": args.fixed_context_prefill_probes,
        "allow_output_id_divergence": args.allow_output_id_divergence,
    }
    mismatches = [
        f"{key}: {metadata.get(key)!r} != {value!r}"
        for key, value in expected.items()
        if metadata.get(key) != value
    ]
    if mismatches:
        raise AssertionError(
            "reference metadata does not match this run:\n" + "\n".join(mismatches)
        )


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
    parser.add_argument("--prefix-granularity", type=int, default=256)
    parser.add_argument("--logprob-atol", type=float, default=2e-2)
    parser.add_argument("--hidden-atol", type=float, default=3e-2)
    parser.add_argument("--rtol", type=float, default=0.0)
    parser.add_argument("--require-hidden-states", action="store_true")
    parser.add_argument("--fixed-context-prefill-probes", action="store_true")
    parser.add_argument(
        "--allow-output-id-divergence",
        action="store_true",
        help=(
            "Do not fail on generated-ID divergence. Use only after repeated TP "
            "control runs prove the same nondeterminism; common-history "
            "logprobs are still checked."
        ),
    )
    parser.add_argument("--run-label", required=True)
    parser.add_argument("--source-id", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--dcp-size", type=int, required=True)
    parser.add_argument(
        "--decode-backend",
        choices=("paged", "flashmla"),
        required=True,
    )
    args = parser.parse_args()
    if (args.write_reference is None) == (args.reference is None):
        parser.error("pass exactly one of --write-reference or --reference")
    if args.prefix_granularity <= 0:
        parser.error("--prefix-granularity must be positive")
    if args.prompt_tokens % args.prefix_granularity:
        parser.error(
            "--prompt-tokens must be a multiple of --prefix-granularity "
            "so the replay candidate is page-aligned"
        )
    expected_artifact = (
        None if args.reference is None else json.loads(args.reference.read_text())
    )
    if expected_artifact is not None:
        _validate_reference_metadata(expected_artifact.get("metadata", {}), args)
    input_ids = [
        1 + (index * 7919) % (args.vocab_size - 1)
        for index in range(args.prompt_tokens)
    ]

    _flush_cache(args.url)
    cold = _request(args.url, input_ids, args.decode_tokens, args.model)
    if expected_artifact is None:
        replay_suffix_token = _choose_replay_suffix(
            args.prompt_tokens,
            args.vocab_size,
            list(_field(cold, "output_ids") or []),
        )
    else:
        replay_suffix_token = int(expected_artifact["metadata"]["replay_suffix_token"])
    replay = _request(
        args.url,
        [*input_ids, replay_suffix_token],
        args.decode_tokens,
        args.model,
    )
    _validate_cache_evidence(cold, replay, len(input_ids))
    passes = [cold, replay]
    if expected_artifact is None:
        continuation = list(_field(cold, "output_ids") or [])
    else:
        continuation = list(_field(expected_artifact["passes"][0], "output_ids") or [])
    fixed_context_prefill = (
        _fixed_context_prefill_requests(
            args.url,
            input_ids,
            continuation,
            args.model,
        )
        if args.fixed_context_prefill_probes
        else []
    )
    artifact = {
        "metadata": {
            "schema_version": 2,
            "run_label": args.run_label,
            "source_id": args.source_id,
            "checkpoint": args.checkpoint,
            "model": args.model,
            "dcp_size": args.dcp_size,
            "decode_backend": args.decode_backend,
            "prompt_tokens": args.prompt_tokens,
            "decode_tokens": args.decode_tokens,
            "vocab_size": args.vocab_size,
            "prefix_granularity": args.prefix_granularity,
            "replay_suffix_token": replay_suffix_token,
            "fixed_context_prefill_probes": args.fixed_context_prefill_probes,
            "allow_output_id_divergence": args.allow_output_id_divergence,
        },
        "passes": passes,
        "fixed_context_prefill": fixed_context_prefill,
    }
    if args.actual_output is not None:
        args.actual_output.write_text(json.dumps(artifact))
    if args.write_reference is not None:
        args.write_reference.write_text(json.dumps(artifact))
        print(f"wrote TP reference to {args.write_reference}")
        return
    _compare(passes, expected_artifact["passes"], args)
    _compare_fixed_context_prefill(
        fixed_context_prefill,
        expected_artifact["fixed_context_prefill"],
        args,
    )
    print("TP/DCP cold-decode and verified-prefix token/logprob parity passed")


if __name__ == "__main__":
    main()
