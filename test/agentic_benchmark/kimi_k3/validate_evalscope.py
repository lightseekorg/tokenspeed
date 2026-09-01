#!/usr/bin/env python3
"""Fail unless every EvalScope request succeeded with exactly 500 tokens."""

import argparse
import json
import sqlite3
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("roots", nargs="+", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    rows = []
    for root in args.roots:
        dbs = sorted(root.rglob("benchmark_data.db"))
        if not dbs:
            raise RuntimeError(f"no benchmark_data.db under {root}")
        for db in dbs:
            with sqlite3.connect(str(db)) as conn:
                total, successful, min_tokens, max_tokens = conn.execute(
                    "SELECT COUNT(*), SUM(success), MIN(completion_tokens), "
                    "MAX(completion_tokens) FROM result"
                ).fetchone()
            row = {
                "database": str(db),
                "requests": total,
                "successful": successful,
                "min_completion_tokens": min_tokens,
                "max_completion_tokens": max_tokens,
            }
            rows.append(row)
            if total <= 0 or successful != total:
                raise RuntimeError(f"request failure in {db}: {row}")
            if min_tokens != 500 or max_tokens != 500:
                raise RuntimeError(f"wrong completion length in {db}: {row}")

    args.output.write_text(json.dumps(rows, indent=2) + "\n")
    print(json.dumps(rows, indent=2))


if __name__ == "__main__":
    main()
