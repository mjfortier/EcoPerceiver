#!/bin/bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$REPO_ROOT"

DEFAULT_RUN_PATH="experiments/runs/final_v2_3e-06_ws_l128_f12_e32_c32_o0.3_wcswcswcswcsssss_CC/seed_0"

usage() {
  cat >&2 <<EOF
Usage: $0 YEAR [options]

Assemble a calendar-year ERA5 NetCDF file, then upload it to Google Drive with rclone.
The first step is always scripts/era5/build_year_nc.sh.

Options:
  --year YEAR                 Year to assemble and upload. Positional YEAR is also accepted.
  --run-path PATH             EcoPerceiver run directory.
  --input-dir PATH            Directory containing quarterly .nc files.
  --output-path PATH          Yearly NetCDF output path.
  --engine ENGINE             xarray NetCDF engine for assembly. Default: h5netcdf.
  --inputs Q1 Q2 Q3 Q4        Explicit quarter NetCDF paths for the assembler.
  --allow-missing             Pass through to build_year_nc.sh.
  --overwrite                 Replace an existing assembled output file.
  --rclone-remote REMOTE      rclone remote. Default: RCLONE_REMOTE or gdrive:
  --drive-dir PATH            Google Drive folder/path. Default: DRIVE_DIR or ep_era5.
  --remote-dir REMOTE_DIR     Full rclone destination dir, overriding remote/drive-dir.
  --rclone-arg ARG            Extra argument passed to rclone copyto. Repeatable.
  --dry-run                   Print assembler paths and upload target without writing.
  -h, --help                  Show this help.

Examples:
  $0 2017
  $0 --year 2017 --overwrite
  $0 2017 --output-path /scratch/l/luislara/EcoPerceiver/out/era5_predictions_2017.nc
EOF
}

die() {
  echo "$1" >&2
  usage
  exit 2
}

join_remote_dir() {
  local remote="$1"
  local drive_dir="$2"

  remote="${remote%/}"
  drive_dir="${drive_dir#/}"
  drive_dir="${drive_dir%/}"

  if [[ -z "$drive_dir" ]]; then
    echo "$remote"
  elif [[ "$remote" == *: ]]; then
    echo "${remote}${drive_dir}"
  else
    echo "${remote}/${drive_dir}"
  fi
}

join_remote_file() {
  local remote_dir="$1"
  local filename="$2"

  if [[ "$remote_dir" == *: ]]; then
    echo "${remote_dir}${filename}"
  else
    echo "${remote_dir%/}/${filename}"
  fi
}

YEAR="${YEAR:-}"
RUN_PATH="${RUN_PATH:-$DEFAULT_RUN_PATH}"
INPUT_DIR="${INPUT_DIR:-}"
OUTPUT_PATH="${OUTPUT_PATH:-}"
ENGINE="${XARRAY_ENGINE:-h5netcdf}"
ALLOW_MISSING=0
OVERWRITE=0
DRY_RUN=0
RCLONE_REMOTE="${RCLONE_REMOTE:-gdrive:}"
DRIVE_DIR="${DRIVE_DIR:-ep_era5}"
REMOTE_DIR="${REMOTE_DIR:-}"
INPUTS=()
RCLONE_ARGS=()

while [[ $# -gt 0 ]]; do
  case "$1" in
    --year)
      [[ $# -ge 2 ]] || die "--year requires a value."
      YEAR="$2"
      shift 2
      ;;
    --year=*)
      YEAR="${1#*=}"
      shift
      ;;
    --run-path)
      [[ $# -ge 2 ]] || die "--run-path requires a value."
      RUN_PATH="$2"
      shift 2
      ;;
    --run-path=*)
      RUN_PATH="${1#*=}"
      shift
      ;;
    --input-dir)
      [[ $# -ge 2 ]] || die "--input-dir requires a value."
      INPUT_DIR="$2"
      shift 2
      ;;
    --input-dir=*)
      INPUT_DIR="${1#*=}"
      shift
      ;;
    --output-path)
      [[ $# -ge 2 ]] || die "--output-path requires a value."
      OUTPUT_PATH="$2"
      shift 2
      ;;
    --output-path=*)
      OUTPUT_PATH="${1#*=}"
      shift
      ;;
    --engine)
      [[ $# -ge 2 ]] || die "--engine requires a value."
      ENGINE="$2"
      shift 2
      ;;
    --engine=*)
      ENGINE="${1#*=}"
      shift
      ;;
    --inputs)
      [[ $# -ge 5 ]] || die "--inputs requires four paths."
      INPUTS=("$2" "$3" "$4" "$5")
      shift 5
      ;;
    --allow-missing)
      ALLOW_MISSING=1
      shift
      ;;
    --overwrite)
      OVERWRITE=1
      shift
      ;;
    --rclone-remote)
      [[ $# -ge 2 ]] || die "--rclone-remote requires a value."
      RCLONE_REMOTE="$2"
      shift 2
      ;;
    --rclone-remote=*)
      RCLONE_REMOTE="${1#*=}"
      shift
      ;;
    --drive-dir)
      [[ $# -ge 2 ]] || die "--drive-dir requires a value."
      DRIVE_DIR="$2"
      shift 2
      ;;
    --drive-dir=*)
      DRIVE_DIR="${1#*=}"
      shift
      ;;
    --remote-dir)
      [[ $# -ge 2 ]] || die "--remote-dir requires a value."
      REMOTE_DIR="$2"
      shift 2
      ;;
    --remote-dir=*)
      REMOTE_DIR="${1#*=}"
      shift
      ;;
    --rclone-arg)
      [[ $# -ge 2 ]] || die "--rclone-arg requires a value."
      RCLONE_ARGS+=("$2")
      shift 2
      ;;
    --dry-run)
      DRY_RUN=1
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    --*)
      die "Unknown argument: $1"
      ;;
    *)
      if [[ -z "$YEAR" ]]; then
        YEAR="$1"
        shift
      else
        die "Unknown argument: $1"
      fi
      ;;
  esac
done

[[ -n "$YEAR" ]] || die "YEAR is required."
[[ "$YEAR" =~ ^[0-9]{4}$ ]] || die "YEAR must be a four-digit year, got: $YEAR"

if [[ -z "$INPUT_DIR" ]]; then
  INPUT_DIR="$RUN_PATH/eval"
fi

if [[ -z "$OUTPUT_PATH" ]]; then
  OUTPUT_PATH="$INPUT_DIR/era5_predictions_${YEAR}.nc"
fi

if [[ -z "$REMOTE_DIR" ]]; then
  REMOTE_DIR="$(join_remote_dir "$RCLONE_REMOTE" "$DRIVE_DIR")"
fi

REMOTE_PATH="$(join_remote_file "$REMOTE_DIR" "$(basename "$OUTPUT_PATH")")"

ASSEMBLE_ARGS=(
  --year "$YEAR"
  --run-path "$RUN_PATH"
  --input-dir "$INPUT_DIR"
  --output-path "$OUTPUT_PATH"
  --engine "$ENGINE"
)

if [[ "${#INPUTS[@]}" -gt 0 ]]; then
  ASSEMBLE_ARGS+=(--inputs "${INPUTS[@]}")
fi
if [[ "$ALLOW_MISSING" == "1" ]]; then
  ASSEMBLE_ARGS+=(--allow-missing)
fi
if [[ "$OVERWRITE" == "1" ]]; then
  ASSEMBLE_ARGS+=(--overwrite)
fi
if [[ "$DRY_RUN" == "1" ]]; then
  ASSEMBLE_ARGS+=(--dry-run)
fi

echo "[$(date)] Assembling yearly ERA5 NetCDF."
echo "Year: $YEAR"
echo "Output path: $OUTPUT_PATH"
"$SCRIPT_DIR/build_year_nc.sh" "${ASSEMBLE_ARGS[@]}"

if [[ "$DRY_RUN" == "1" ]]; then
  echo "[$(date)] Dry-run upload target: $REMOTE_PATH"
  printf 'Dry-run upload command: rclone copyto %q %q --progress' "$OUTPUT_PATH" "$REMOTE_PATH"
  if [[ "${#RCLONE_ARGS[@]}" -gt 0 ]]; then
    printf ' %q' "${RCLONE_ARGS[@]}"
  fi
  printf '\n'
  exit 0
fi

if [[ ! -f "$OUTPUT_PATH" ]]; then
  echo "Assembled output not found: $OUTPUT_PATH" >&2
  exit 1
fi

echo "[$(date)] Uploading yearly ERA5 NetCDF with rclone."
echo "Destination: $REMOTE_PATH"
rclone mkdir "$REMOTE_DIR"
rclone copyto "$OUTPUT_PATH" "$REMOTE_PATH" --progress "${RCLONE_ARGS[@]}"
rclone lsf "$REMOTE_PATH"
echo "[$(date)] Upload complete: $REMOTE_PATH"
