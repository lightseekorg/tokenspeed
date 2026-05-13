"""Sweep num_clients for bench.py and produce a comparison table.

Usage:
    python sweep.py --dataset-path agentic_dataset.json --port 8000
    python sweep.py --dataset-path agentic_dataset.json --num-clients 1 2 4 8 16
"""

import argparse
import json
import os
import subprocess
import sys


def parse_args():
    parser = argparse.ArgumentParser(
        description="Sweep num_clients for agentic multi-turn benchmark."
    )
    parser.add_argument(
        "--dataset-path",
        type=str,
        required=True,
        help="Path to agentic dataset JSON",
    )
    parser.add_argument(
        "--num-requests-per-client",
        type=int,
        default=1,
        help="Conversations per client (default: 1)",
    )
    parser.add_argument(
        "--num-clients",
        type=int,
        nargs="+",
        default=[1, 2, 4, 8],
        help="List of num_clients values to sweep (default: 1 2 4 8)",
    )
    parser.add_argument(
        "--host",
        type=str,
        default="localhost",
        help="Server hostname (default: localhost)",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=8000,
        help="Server port (default: 8000)",
    )
    parser.add_argument(
        "--flush-cache",
        action="store_true",
        default=False,
        help="Flush server cache before each run",
    )
    parser.add_argument(
        "--log-file",
        type=str,
        default="sweep_results.jsonl",
        help="Output metrics file (default: sweep_results.jsonl)",
    )
    parser.add_argument(
        "--offset",
        type=int,
        default=0,
        help="Initial conversation offset (default: 0)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=1,
        help="Random seed (default: 1)",
    )
    return parser.parse_args()


def run_bench(args, nc, offset, log_file):
    """Run bench.py as a subprocess."""
    script = os.path.join(os.path.dirname(__file__), "bench.py")
    cmd = [
        sys.executable,
        script,
        "--dataset-path",
        args.dataset_path,
        "--num-requests-per-client",
        str(args.num_requests_per_client),
        "--num-clients",
        str(nc),
        "--host",
        args.host,
        "--port",
        str(args.port),
        "--log-file",
        log_file,
        "--tag",
        f"nc_{nc}",
        "--offset",
        str(offset),
        "--seed",
        str(args.seed),
    ]
    if args.flush_cache:
        cmd.append("--flush-cache")

    print(f"\n$ {' '.join(cmd)}")
    result = subprocess.run(cmd)
    if result.returncode != 0:
        print(f"Warning: bench.py exited with code {result.returncode}")
        return None
    return True


def load_results(log_file):
    """Load all results from the JSONL log file."""
    results = []
    with open(log_file) as f:
        for line in f:
            line = line.strip()
            if line:
                results.append(json.loads(line))
    return results


def print_comparison(results_by_nc):
    """Print a comparison table across all sweep runs."""
    print("\n" + "=" * 120)
    print("{s:{c}^{n}}".format(s=" Sweep Comparison ", n=120, c="="))
    print("=" * 120)

    header = (
        f"{'Clients':<10} {'Convs':<8} {'Reqs':<8} {'Duration':<10} "
        f"{'TTFT p50':<10} {'TTFT p90':<10} "
        f"{'TPOT p50':<10} {'TPOT p90':<10} "
        f"{'E2EL p50':<10} {'E2EL p90':<10} "
        f"{'Out tok/s':<10} {'Tot tok/s':<10} "
        f"{'Cache%':<8}"
    )
    print(header)
    print("-" * 120)

    for nc, r in results_by_nc:
        s = r["summary"]
        print(
            f"{nc:<10} {s['num_requests']:<8} {s['total_requests']:<8} "
            f"{s['duration_s']:<10.2f} "
            f"{s.get('p50_TTFT', 0):<10.2f} {s.get('p90_TTFT', 0):<10.2f} "
            f"{s.get('p50_TPOT', 0):<10.2f} {s.get('p90_TPOT', 0):<10.2f} "
            f"{s.get('p50_E2EL', 0):<10.2f} {s.get('p90_E2EL', 0):<10.2f} "
            f"{s['output_token_throughput']:<10.2f} "
            f"{s['total_token_throughput']:<10.2f} "
            f"{s['cache_hit_rate']:<8.4f}"
        )

    print("=" * 120)
    print("(TTFT/TPOT/E2EL in ms)")


def main():
    args = parse_args()

    # Check server is reachable
    health_url = f"http://{args.host}:{args.port}/readiness"
    ret = subprocess.run(
        ["curl", "-sf", "--max-time", "5", health_url],
        capture_output=True,
    )
    if ret.returncode != 0:
        raise RuntimeError(f"Cannot connect to server at {args.host}:{args.port}")
    print(f"Server at {args.host}:{args.port} is reachable.")

    current_offset = args.offset
    log_files = {}

    # Create folder for per-nc log files
    base, ext = os.path.splitext(args.log_file)
    log_dir = base
    os.makedirs(log_dir, exist_ok=True)

    for nc in args.num_clients:
        print(f"\n{'='*60}")
        print(f"  Sweep: num_clients = {nc}, offset = {current_offset}")
        print(f"{'='*60}")

        log_file = os.path.join(log_dir, f"{nc}{ext}")
        log_files[nc] = log_file

        run_bench(args, nc, current_offset, log_file)

        # Advance offset so next run uses different conversations
        num_requests = nc * args.num_requests_per_client
        current_offset += num_requests

    # Load results and print comparison
    results_by_nc = []
    for nc in args.num_clients:
        entries = load_results(log_files[nc])
        if entries:
            results_by_nc.append((nc, entries[-1]))

    if results_by_nc:
        print_comparison(results_by_nc)
        # Save aggregated sweep results
        sweep_output = {
            "sweep_config": {
                "dataset_path": args.dataset_path,
                "num_requests_per_client": args.num_requests_per_client,
                "num_clients": args.num_clients,
                "seed": args.seed,
            },
            "results": [{"num_clients": nc, **r["summary"]} for nc, r in results_by_nc],
        }
        with open(args.log_file, "w") as f:
            f.write(json.dumps(sweep_output) + "\n")
        print(f"\nSweep results saved to {args.log_file}")
    else:
        print("No results to compare.")


if __name__ == "__main__":
    main()
