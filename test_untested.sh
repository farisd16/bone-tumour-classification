#!/bin/bash
#SBATCH --job-name=test-untested
#SBATCH --output=./test-untested-%j.out
#SBATCH --error=./test-untested-%j.err
#SBATCH --time=24:00:00
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G

#
# Test all untested WandB runs matching an infix filter sequentially.
#
# Usage:
#   sbatch test_untested.sh <infix>
#
# Examples:
#   sbatch test_untested.sh r32_b4
#   sbatch test_untested.sh diffusion
#   sbatch test_untested.sh step7_ce_noaug
#

set -e

INFIX="$1"

if [ -z "$INFIX" ]; then
    echo "Usage: sbatch $0 <infix>"
    exit 1
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Load python module and activate environment
ml python/anaconda3 2>/dev/null || true
eval "$(conda shell.bash hook)"
conda deactivate 2>/dev/null || true
conda activate bone-tumour-classification

# Find untested runs
echo "Finding untested runs matching '$INFIX'..."
mapfile -t RUNS < <(python "$SCRIPT_DIR/find_untested_runs.py" --infix "$INFIX")

NUM_RUNS=${#RUNS[@]}

if [ "$NUM_RUNS" -eq 0 ]; then
    echo "No untested runs found matching '$INFIX'"
    exit 0
fi

echo "Found $NUM_RUNS untested runs. Testing sequentially..."
echo ""

# Test each run sequentially
for i in "${!RUNS[@]}"; do
    ENTRY="${RUNS[$i]}"
    # Parse run name and architecture (format: run_name|architecture)
    RUN_NAME="${ENTRY%%|*}"
    ARCH="${ENTRY##*|}"
    
    echo "========================================"
    echo "[$((i+1))/$NUM_RUNS] Testing: $RUN_NAME (arch: $ARCH)"
    echo "========================================"
    
    python "$SCRIPT_DIR/test.py" --run-name "$RUN_NAME" --architecture "$ARCH"
    
    echo ""
done

echo "All runs tested!"
