#!/usr/bin/env bash
set -euo pipefail

# Usage:
#   ./run_fraction_busy_10x.sh <base_filename> [output_dir]
#
# Example:
#   ./run_fraction_busy_10x.sh results
#   -> creates: results_run01.csv ... results_run10.csv
#
#   ./run_fraction_busy_10x.sh results out/
#   -> creates: out/results_run01.csv ... out/results_run10.csv

if [[ $# -lt 1 ]]; then
  echo "Usage: $0 <base_filename> [output_dir]"
  exit 1
fi

BASE_NAME="$1"
OUT_DIR="${2:-.}"

# Check the command exists
if [[ ! -x "./fraction_busy.sh" ]]; then
  echo "Error: ./fraction_busy.sh not found or not executable"
  exit 1
fi

mkdir -p "$OUT_DIR"

for i in $(seq 1 10); do
  OUT_FILE="${OUT_DIR}/${BASE_NAME}_run$(printf "%02d" "$i").csv"
  echo "[INFO] Running $i/10 -> $OUT_FILE"
  ./fraction_busy.sh 500 0.04 > "$OUT_FILE"
done

echo "[DONE] Generated 10 CSV files in: $OUT_DIR"

