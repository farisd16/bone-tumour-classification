import argparse
import subprocess
import sys
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run test.py sequentially for all checkpoints matching a pattern.",
    )
    parser.add_argument(
        "--pattern",
        default="2025-12-08",
        help="Substring to match checkpoint folder names (default: 2025-12-08)",
    )
    parser.add_argument(
        "--limit",
        type=int,
        help="Optional: only process the first N matched runs after sorting.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="List matching runs without executing test.py",
    )
    args = parser.parse_args()

    root = Path(__file__).resolve().parent
    checkpoints_dir = root / "checkpoints"

    if not checkpoints_dir.is_dir():
        print(f"Checkpoint folder not found: {checkpoints_dir}")
        sys.exit(1)

    run_names = sorted(
        d.name for d in checkpoints_dir.iterdir() if d.is_dir() and args.pattern in d.name
    )

    if args.limit is not None:
        run_names = run_names[: args.limit]

    if not run_names:
        print(f"No checkpoint folders matched pattern '{args.pattern}'.")
        sys.exit(1)

    print("Runs to test:")
    for name in run_names:
        print(f"  - {name}")

    if args.dry_run:
        return

    failed_runs = []

    for name in run_names:
        cmd = [sys.executable, "test.py", "--run-name", name]
        print(f"\n=== Running test.py for {name} ===")
        result = subprocess.run(cmd, cwd=root)
        if result.returncode != 0:
            print(f"test.py failed for {name} (exit code {result.returncode}); continuing.")
            failed_runs.append(name)

    if failed_runs:
        print("\nRuns with failed tests:")
        for name in failed_runs:
            print(f"  - {name}")
        sys.exit(1)
    else:
        print("\nAll tests completed successfully.")


if __name__ == "__main__":
    main()
