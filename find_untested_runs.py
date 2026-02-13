#!/usr/bin/env python3
"""
Find untested WandB runs that match a given infix filter.

A run is considered "untested" if it:
1. Has state "finished" or "failed"
2. Does NOT have test metrics logged (e.g., "Macro F1", "Accuracy" in summary)

Usage:
    python find_untested_runs.py [--infix INFIX] [--list-all] [--verbose]

Examples:
    python find_untested_runs.py --infix r32_b4
    python find_untested_runs.py --infix diffusion --verbose
    python find_untested_runs.py --list-all
"""

import argparse
import sys
from typing import List, Optional

import wandb

from config import WANDB_ENTITY, WANDB_PROJECT


# Keys that indicate a run has been tested
TEST_METRIC_KEYS = {"Macro F1", "Accuracy", "Balanced Accuracy", "Weighted F1"}


def is_run_tested(run: wandb.apis.public.Run) -> bool:
    """Check if a run has test metrics in its summary."""
    summary = run.summary
    return any(key in summary for key in TEST_METRIC_KEYS)


def get_untested_runs(
    infix: Optional[str] = None, verbose: bool = False
) -> List[wandb.apis.public.Run]:
    """
    Query WandB API for finished runs that haven't been tested.

    Args:
        infix: Optional string to filter run names (must be contained in display_name)
        verbose: Print detailed info about each run

    Returns:
        List of untested WandB Run objects
    """
    api = wandb.Api()

    # Get all finished and failed runs
    runs = api.runs(
        f"{WANDB_ENTITY}/{WANDB_PROJECT}",
        filters={"$or": [{"state": "finished"}, {"state": "failed"}]},
    )

    untested_runs = []
    tested_count = 0
    filtered_count = 0

    for run in runs:
        display_name = run.display_name or run.name

        # Filter by infix if provided
        if infix and infix not in display_name:
            filtered_count += 1
            continue

        # Check if run has been tested
        if is_run_tested(run):
            tested_count += 1
            if verbose:
                print(f"[TESTED] {display_name}")
            continue

        # This run is untested
        untested_runs.append(run)
        if verbose:
            print(f"[UNTESTED] {display_name}")

    if verbose:
        print(f"\n--- Summary ---")
        print(f"Total finished/failed runs: {len(runs)}")
        if infix:
            print(f"Filtered out (no match for '{infix}'): {filtered_count}")
        print(f"Already tested: {tested_count}")
        print(f"Untested: {len(untested_runs)}")

    return untested_runs


def main():
    parser = argparse.ArgumentParser(
        description="Find untested WandB runs matching an infix filter"
    )
    parser.add_argument(
        "--infix",
        type=str,
        default=None,
        help="Filter runs by this infix string in the run name",
    )
    parser.add_argument(
        "--list-all",
        action="store_true",
        help="List all finished runs (tested and untested)",
    )
    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Print detailed information about each run",
    )
    parser.add_argument(
        "--output",
        "-o",
        type=str,
        default=None,
        help="Write run names to a file (one per line)",
    )
    args = parser.parse_args()

    if args.list_all:
        api = wandb.Api()
        runs = api.runs(
            f"{WANDB_ENTITY}/{WANDB_PROJECT}",
            filters={"$or": [{"state": "finished"}]},
        )
        for run in runs:
            display_name = run.display_name or run.name
            if args.infix and args.infix not in display_name:
                continue
            status = "TESTED" if is_run_tested(run) else "UNTESTED"
            print(f"[{status}] {display_name}")
        return

    untested_runs = get_untested_runs(infix=args.infix, verbose=args.verbose)

    if not untested_runs:
        print("No untested runs found.", file=sys.stderr)
        sys.exit(0)

    # Output run names with architecture (format: run_name|architecture)
    for run in untested_runs:
        name = run.display_name or run.name
        # Get architecture from run config, default to resnet34
        arch = run.config.get("architecture", "resnet34")
        if args.output:
            with open(args.output, "a") as f:
                f.write(f"{name}|{arch}\n")
        else:
            print(f"{name}|{arch}")

    if args.output:
        print(
            f"Wrote {len(untested_runs)} run entries to {args.output}", file=sys.stderr
        )


if __name__ == "__main__":
    main()
