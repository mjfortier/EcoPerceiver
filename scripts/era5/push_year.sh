#!/bin/bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT=""
for candidate in "$PWD" "${SLURM_SUBMIT_DIR:-}" "${SLURM_SUBMIT_DIR:+$SLURM_SUBMIT_DIR/../..}" "$SCRIPT_DIR/../.."; do
  [[ -n "$candidate" ]] || continue
  candidate="$(cd "$candidate" 2>/dev/null && pwd)" || continue
  if [[ -f "$candidate/setup.py" && -d "$candidate/scripts/era5" ]]; then
    REPO_ROOT="$candidate"
    break
  fi
done
[[ -n "$REPO_ROOT" ]] || {
  echo "Could not locate EcoPerceiver repo root." >&2
  exit 2
}
cd "$REPO_ROOT"

DEFAULT_HF_REPO_ID="ludolara/era5"
DEFAULT_RCLONE_REMOTE="gdrive:"
DEFAULT_DRIVE_DIR="ep_era5"
PYTHON_BIN="${PYTHON_BIN:-}"

if [[ -f "$REPO_ROOT/.env" ]]; then
  set -a
  # shellcheck disable=SC1091
  source "$REPO_ROOT/.env"
  set +a
fi

usage() {
  cat >&2 <<EOF
Usage: $0 --to hf|drive NC_PATH [options]
       $0 hf|drive NC_PATH [options]

Upload an existing yearly ERA5 NetCDF file. This script only pushes the file it
receives; it does not build or modify the .nc file.

Options:
  --to TARGET                 Destination target: hf or drive.
  --target TARGET             Alias for --to.
  --nc PATH                   Existing .nc file to upload. Positional NC_PATH is also accepted.
  --file PATH                 Alias for --nc.
  --path PATH                 Destination path/filename. For hf, path in repo. For drive, remote file path.
  --dry-run                   Print the upload command/target without uploading.
  -h, --help                  Show this help.

Hugging Face options:
  --hf-repo-id REPO_ID        Hugging Face dataset repo. Default: HF_REPO_ID or ludolara/era5.
  --path-in-repo PATH         Destination path in the dataset. Default: basename of NC_PATH.
  --revision REVISION         Branch/revision to upload to. Default: HF_REVISION or main.
  --commit-message MESSAGE    Commit message for the upload.
  --private                   Create the dataset as private if it does not already exist.
  --skip-create-repo          Do not create the dataset if it is missing.

Google Drive/rclone options:
  --rclone-remote REMOTE      rclone remote. Default: RCLONE_REMOTE or gdrive:
  --drive-dir PATH            Google Drive folder/path. Default: DRIVE_DIR or ep_era5.
  --remote-dir REMOTE_DIR     Full rclone destination directory, overriding remote/drive-dir.
  --remote-path REMOTE_PATH   Full rclone destination file path.
  --rclone-arg ARG            Extra argument passed to rclone copyto. Repeatable.

Environment:
  HF_WRITE                    Hugging Face write token. Loaded from .env when present.
  HF_REPO_ID                  Optional default dataset repo id.
  HF_REVISION                 Optional default Hugging Face revision.
  RCLONE_REMOTE               Optional default rclone remote.
  DRIVE_DIR                   Optional default Drive folder/path.
  PYTHON_BIN                  Optional Python executable for the Hugging Face upload.

Examples:
  $0 --to drive /scratch/l/luislara/EcoPerceiver/era5_predictions_2017.nc
  $0 drive /scratch/l/luislara/EcoPerceiver/era5_predictions_2017.nc
  $0 --to hf /scratch/l/luislara/EcoPerceiver/era5_predictions_2017.nc --path-in-repo predictions/era5_predictions_2017.nc
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

remote_parent_dir() {
  local remote_path="$1"

  if [[ "$remote_path" == */* ]]; then
    echo "${remote_path%/*}"
  elif [[ "$remote_path" == *:* ]]; then
    echo "${remote_path%%:*}:"
  else
    echo "."
  fi
}

print_rclone_command() {
  printf 'rclone copyto %q %q --progress' "$NC_PATH" "$REMOTE_PATH"
  if [[ "${#RCLONE_ARGS[@]}" -gt 0 ]]; then
    printf ' %q' "${RCLONE_ARGS[@]}"
  fi
  printf '\n'
}

TARGET="${TARGET:-}"
NC_PATH="${NC_PATH:-}"
DESTINATION_PATH=""
DRY_RUN=0
HF_REPO_ID="${HF_REPO_ID:-$DEFAULT_HF_REPO_ID}"
PATH_IN_REPO="${PATH_IN_REPO:-}"
REVISION="${HF_REVISION:-main}"
COMMIT_MESSAGE="${HF_COMMIT_MESSAGE:-}"
CREATE_REPO=1
PRIVATE=0
RCLONE_REMOTE="${RCLONE_REMOTE:-$DEFAULT_RCLONE_REMOTE}"
DRIVE_DIR="${DRIVE_DIR:-$DEFAULT_DRIVE_DIR}"
REMOTE_DIR="${REMOTE_DIR:-}"
REMOTE_PATH="${REMOTE_PATH:-}"
RCLONE_ARGS=()

if [[ -z "$PYTHON_BIN" ]]; then
  if [[ -n "${ECOPERCEIVER_ENV:-}" && -x "$ECOPERCEIVER_ENV/bin/python" ]]; then
    PYTHON_BIN="$ECOPERCEIVER_ENV/bin/python"
  elif [[ -n "${SCRATCH:-}" && -x "$SCRATCH/env/ecoperceiver/bin/python" ]]; then
    PYTHON_BIN="$SCRATCH/env/ecoperceiver/bin/python"
  else
    PYTHON_BIN="python3"
  fi
fi

while [[ $# -gt 0 ]]; do
  case "$1" in
    --to|--target)
      [[ $# -ge 2 ]] || die "$1 requires a value."
      TARGET="$2"
      shift 2
      ;;
    --to=*|--target=*)
      TARGET="${1#*=}"
      shift
      ;;
    --nc|--file)
      [[ $# -ge 2 ]] || die "$1 requires a value."
      NC_PATH="$2"
      shift 2
      ;;
    --nc=*|--file=*)
      NC_PATH="${1#*=}"
      shift
      ;;
    --path)
      [[ $# -ge 2 ]] || die "--path requires a value."
      DESTINATION_PATH="$2"
      shift 2
      ;;
    --path=*)
      DESTINATION_PATH="${1#*=}"
      shift
      ;;
    --hf-repo-id)
      [[ $# -ge 2 ]] || die "--hf-repo-id requires a value."
      HF_REPO_ID="$2"
      shift 2
      ;;
    --hf-repo-id=*)
      HF_REPO_ID="${1#*=}"
      shift
      ;;
    --path-in-repo)
      [[ $# -ge 2 ]] || die "--path-in-repo requires a value."
      PATH_IN_REPO="$2"
      shift 2
      ;;
    --path-in-repo=*)
      PATH_IN_REPO="${1#*=}"
      shift
      ;;
    --revision)
      [[ $# -ge 2 ]] || die "--revision requires a value."
      REVISION="$2"
      shift 2
      ;;
    --revision=*)
      REVISION="${1#*=}"
      shift
      ;;
    --commit-message)
      [[ $# -ge 2 ]] || die "--commit-message requires a value."
      COMMIT_MESSAGE="$2"
      shift 2
      ;;
    --commit-message=*)
      COMMIT_MESSAGE="${1#*=}"
      shift
      ;;
    --private)
      PRIVATE=1
      shift
      ;;
    --skip-create-repo)
      CREATE_REPO=0
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
    --remote-path)
      [[ $# -ge 2 ]] || die "--remote-path requires a value."
      REMOTE_PATH="$2"
      shift 2
      ;;
    --remote-path=*)
      REMOTE_PATH="${1#*=}"
      shift
      ;;
    --rclone-arg)
      [[ $# -ge 2 ]] || die "--rclone-arg requires a value."
      RCLONE_ARGS+=("$2")
      shift 2
      ;;
    --rclone-arg=*)
      RCLONE_ARGS+=("${1#*=}")
      shift
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
      if [[ -z "$TARGET" && ( "$1" == "hf" || "$1" == "drive" ) ]]; then
        TARGET="$1"
      elif [[ -z "$NC_PATH" ]]; then
        NC_PATH="$1"
      else
        die "Unknown argument: $1"
      fi
      shift
      ;;
  esac
done

TARGET="${TARGET,,}"
[[ -n "$TARGET" ]] || die "--to hf|drive is required."
case "$TARGET" in
  hf|drive)
    ;;
  *)
    die "--to must be hf or drive, got: $TARGET"
    ;;
esac

if [[ -n "$DESTINATION_PATH" ]]; then
  if [[ "$TARGET" == "drive" ]]; then
    [[ -z "$REMOTE_PATH" ]] || die "Use either --path or --remote-path, not both."
    REMOTE_PATH="$DESTINATION_PATH"
  else
    [[ -z "$PATH_IN_REPO" ]] || die "Use either --path or --path-in-repo, not both."
    PATH_IN_REPO="$DESTINATION_PATH"
  fi
fi

[[ -n "$NC_PATH" ]] || die "An existing .nc file path is required."
NC_PATH="${NC_PATH/#\~/$HOME}"
[[ -f "$NC_PATH" ]] || die "NetCDF file not found: $NC_PATH"
[[ "$NC_PATH" == *.nc ]] || die "NetCDF file must end in .nc: $NC_PATH"

if [[ "$TARGET" == "hf" ]]; then
  [[ -n "$HF_REPO_ID" ]] || die "HF repo id is required."
  if [[ -z "$PATH_IN_REPO" ]]; then
    PATH_IN_REPO="$(basename "$NC_PATH")"
  fi
  PATH_IN_REPO="${PATH_IN_REPO#/}"
  if [[ -z "$COMMIT_MESSAGE" ]]; then
    COMMIT_MESSAGE="Upload $(basename "$NC_PATH")"
  fi

  echo "[$(date)] Hugging Face upload target:"
  echo "Source: $NC_PATH"
  echo "Destination: dataset/$HF_REPO_ID@$REVISION:$PATH_IN_REPO"

  if [[ "$DRY_RUN" == "1" ]]; then
    echo "Dry run only; no file uploaded."
    exit 0
  fi

  if [[ -z "${HF_WRITE:-}" ]]; then
    echo "HF_WRITE is required in the environment or .env." >&2
    exit 1
  fi

  "$PYTHON_BIN" - "$HF_REPO_ID" "$NC_PATH" "$PATH_IN_REPO" "$REVISION" "$COMMIT_MESSAGE" "$CREATE_REPO" "$PRIVATE" <<'PY'
import os
from pathlib import Path
import sys

repo_id, nc_path, path_in_repo, revision, commit_message, create_repo, private = sys.argv[1:8]
token = os.environ.get("HF_WRITE")

try:
    from huggingface_hub import HfApi
except ModuleNotFoundError as exc:
    missing = exc.name or "a dependency"
    raise SystemExit(
        "Missing Python dependency while importing huggingface_hub: "
        f"{missing}. Install with: python3 -m pip install --user huggingface_hub filelock"
    ) from exc

api = HfApi()
if create_repo == "1":
    api.create_repo(
        repo_id=repo_id,
        repo_type="dataset",
        token=token,
        private=(private == "1"),
        exist_ok=True,
    )

result = api.upload_file(
    path_or_fileobj=str(Path(nc_path)),
    path_in_repo=path_in_repo,
    repo_id=repo_id,
    repo_type="dataset",
    revision=revision,
    token=token,
    commit_message=commit_message,
)
print(f"Upload complete: {result}")
PY
else
  command -v rclone >/dev/null 2>&1 || {
    echo "rclone is required for Drive uploads." >&2
    exit 1
  }

  if [[ -z "$REMOTE_PATH" ]]; then
    if [[ -z "$REMOTE_DIR" ]]; then
      REMOTE_DIR="$(join_remote_dir "$RCLONE_REMOTE" "$DRIVE_DIR")"
    fi
    REMOTE_PATH="$(join_remote_file "$REMOTE_DIR" "$(basename "$NC_PATH")")"
  elif [[ -z "$REMOTE_DIR" ]]; then
    REMOTE_DIR="$(remote_parent_dir "$REMOTE_PATH")"
  fi

  echo "[$(date)] Drive/rclone upload target:"
  echo "Source: $NC_PATH"
  echo "Destination: $REMOTE_PATH"

  if [[ "$DRY_RUN" == "1" ]]; then
    echo "Dry run only; upload command would be:"
    print_rclone_command
    exit 0
  fi

  rclone mkdir "$REMOTE_DIR"
  rclone copyto "$NC_PATH" "$REMOTE_PATH" --progress "${RCLONE_ARGS[@]}"
  echo "[$(date)] Upload complete: $REMOTE_PATH"
fi
