#!/bin/bash
set -u

EPI_INDEX=6
RT_INDEX=31
PARAM=0x0000

# Defaults requested:
SAMPLES=${1:-5000}
SLEEP_SEC=${2:-0.01}

read_pkg_hex() {
  local idx="$1"
  local param="$2"

  peci_cmds RdPkgConfig "$idx" "$param" 2>/dev/null \
    | tr -d '\r' \
    | grep -oE '0x[0-9a-fA-F]+' \
    | tail -n 1
}

prev_epi_hex=$(read_pkg_hex "$EPI_INDEX" "$PARAM")
prev_rt_hex=$(read_pkg_hex "$RT_INDEX" "$PARAM")

if [[ -z "$prev_epi_hex" || -z "$prev_rt_hex" ]]; then
  echo "[ERROR] Initial PECI read failed."
  echo "EPI raw: $prev_epi_hex"
  echo "RT  raw: $prev_rt_hex"
  echo "Try running these manually:"
  echo "  peci_cmds RdPkgConfig $EPI_INDEX $PARAM"
  echo "  peci_cmds RdPkgConfig $RT_INDEX  $PARAM"
  exit 1
fi

prev_epi=$((prev_epi_hex))
prev_rt=$((prev_rt_hex))

echo "sample,epi_delta,runtime_delta,busy_fraction"

for ((i=1; i<=SAMPLES; i++)); do
  cur_epi_hex=$(read_pkg_hex "$EPI_INDEX" "$PARAM")
  cur_rt_hex=$(read_pkg_hex "$RT_INDEX" "$PARAM")

  if [[ -z "$cur_epi_hex" || -z "$cur_rt_hex" ]]; then
    continue
  fi

  cur_epi=$((cur_epi_hex))
  cur_rt=$((cur_rt_hex))

  epi_delta=$(( (cur_epi - prev_epi) & 0xFFFFFFFF ))
  rt_delta=$(( (cur_rt  - prev_rt ) & 0xFFFFFFFF ))

  prev_epi=$cur_epi
  prev_rt=$cur_rt

  if [[ "$rt_delta" -eq 0 ]]; then
    echo "$i,$epi_delta,$rt_delta,"
  else
    busy_frac=$(awk -v e="$epi_delta" -v r="$rt_delta" 'BEGIN { printf "%.6f", (e / r) }')
    echo "$i,$epi_delta,$rt_delta,$busy_frac"
  fi

  sleep "$SLEEP_SEC"
done

