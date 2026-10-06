#!/usr/bin/env python3
"""Expand a build_swe_smith_dataset.py dataset by replicating conversations
with per-replica cache-bust prefixes.

Only ~71 source trajectories are long enough for 50k-token first turns, which
is too few to saturate an attention-DP decode instance. Replaying a trajectory
verbatim would score a full prefix-cache hit and skip the prefill entirely, so
each replica gets a short unique "[rid:<hash>]" mark prepended to its FIRST
user message (the first point real traffic diverges). The mark is stable
within the conversation, so multi-turn KV reuse inside a session still works;
across replicas the prefix diverges at the first token, so the server treats
every replica as a fresh session. The mark is sha256(replica:conversation)[:12].

Replicas are interleaved (r0c0, r0c1, ..., r1c0, ...) so any --number window
covers distinct source trajectories before it sees repeats.
"""

import argparse
import hashlib
import json


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--replicas", type=int, default=32)
    args = ap.parse_args()

    with open(args.input) as f:
        data = json.load(f)
    conversations = data["conversations"]

    expanded = []
    for replica in range(args.replicas):
        for conv_idx, conv in enumerate(conversations):
            rid = hashlib.sha256(f"{replica}:{conv_idx}".encode()).hexdigest()[:12]
            copy = json.loads(json.dumps(conv))  # deep copy
            first_msg = copy[0]["messages"][0]
            first_msg["content"] = f'[rid:{rid}]\n\n{first_msg["content"]}'
            # prompt_tokens metadata is now ~10 tokens low per turn; harmless,
            # it is informational only.
            expanded.append(copy)

    data["conversations"] = expanded
    meta = data.get("metadata", {})
    meta["num_conversations"] = len(expanded)
    meta["cache_bust_replicas"] = args.replicas

    with open(args.output, "w") as f:
        json.dump(data, f, indent=2)
    print(
        f"{len(conversations)} conversations x {args.replicas} replicas "
        f"-> {len(expanded)} written to {args.output}"
    )


if __name__ == "__main__":
    main()
