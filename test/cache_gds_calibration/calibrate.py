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

"""Collect and fit explicit GDS costs; run as python -m test.cache_gds_calibration.calibrate."""

import argparse
from test.cache_gds_calibration.data import positive
from test.cache_gds_calibration.disk import collect_disk
from test.cache_gds_calibration.fitting import fit
from test.cache_gds_calibration.prefill import collect_prefill, import_prefill


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest="command", required=True)
    pre = sub.add_parser(
        "prefill", help="send isolated native token-ID requests; retain join IDs"
    )
    for name in ("url", "model-path", "text-file", "deployment", "output"):
        pre.add_argument(f"--{name}", required=True)
    pre.add_argument("--trust-remote-code", choices=("yes", "no"), required=True)
    pre.add_argument(
        "--prefix-lengths",
        type=int,
        nargs="+",
        required=True,
        help="cached prefix token counts (prefix_tokens in cases.json)",
    )
    pre.add_argument(
        "--suffix-lengths",
        type=int,
        nargs="+",
        required=True,
        help="additional input token counts (new_tokens in cases.json); excludes output",
    )
    for name in ("repeats", "warmup-repeats", "data-parallel-rank", "seed"):
        pre.add_argument(f"--{name}", type=int, required=True)
    pre.add_argument(
        "--split",
        choices=("train", "validation"),
        required=True,
        help="train: fit cost coefficients; validation: check held-out timing predictions",
    )
    pre.add_argument(
        "--timeout", type=positive, default=1800, help="HTTP timeout, seconds"
    )
    pre.set_defaults(run=collect_prefill)

    imp = sub.add_parser(
        "import-prefill", help="join server timing; check hits and reported preemption"
    )
    imp.add_argument("--requests", nargs="+", required=True)
    imp.add_argument("--server-logs", nargs="+", required=True)
    imp.add_argument(
        "--timing-source",
        choices=("plain-request-stats", "pd-prefill-trace"),
        required=True,
    )
    imp.add_argument("--max-queue-ms", type=positive, required=True)
    imp.add_argument(
        "--queue-source",
        choices=("request-stats", "server-accepted", "pd-admission"),
        required=True,
        help="start of the isolation check: frontend creation, server acceptance, or PD admission phase",
    )
    imp.add_argument("--max-prefix-shortfall", type=int, required=True)
    imp.add_argument("--output", required=True)
    imp.add_argument("--rejected", required=True)
    imp.set_defaults(run=import_prefill)

    disk = sub.add_parser("disk", help="sample the native direct-I/O executor")
    for name in ("bench", "directory", "deployment", "output"):
        disk.add_argument(f"--{name}", required=True)
    for name in (
        "file-bytes",
        "inflight",
        "queue-depth",
        "max-io-bytes",
        "repeats",
        "seed",
    ):
        disk.add_argument(f"--{name}", type=int, required=True)
    disk.add_argument("--block-bytes", type=int, nargs="+", required=True)
    for name in ("seconds", "warmup-seconds"):
        disk.add_argument(f"--{name}", type=positive, required=True)
    disk.add_argument("--backend", choices=("uring", "psync"), required=True)
    disk.add_argument(
        "--time-basis", choices=("job-latency", "amortized"), required=True
    )
    disk.add_argument(
        "--split",
        choices=("train", "validation"),
        required=True,
        help="train: fit cost coefficients; validation: check held-out timing predictions",
    )
    disk.add_argument(
        "--timeout", type=positive, default=1800, help="per-run timeout, seconds"
    )
    disk.set_defaults(run=collect_disk)

    f = sub.add_parser(
        "fit", help="constrained fit, held-out validation, config generation"
    )
    f.add_argument("--prefill", nargs="+", required=True)
    f.add_argument("--disk", nargs="+", required=True)
    f.add_argument("--weighting", choices=("absolute", "relative"), required=True)
    for name in ("disk-scale", "max-mape", "max-error", "max-relative-mad"):
        f.add_argument(f"--{name}", type=positive, required=True)
    f.add_argument("--output-dir", required=True)
    f.set_defaults(run=fit)
    return p


def main():
    args = parser().parse_args()
    args.run(args)


if __name__ == "__main__":
    main()
