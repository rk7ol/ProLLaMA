#!/usr/bin/env bash
set -euo pipefail

# If invoked with `sh upload_to_S3.sh`, re-exec under bash.
if [[ -z "${BASH_VERSION:-}" ]]; then
  exec bash "$0" "$@"
fi

usage() {
  cat <<'EOF'
Usage:
  ./upload_to_S3.sh [--profile PROFILE] [--region REGION] [--dryrun] [--base-uri S3_URI]
EOF
}

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"

S3_BASE_URI_DEFAULT="s3://plm-ml/code/prollama"
base_uri="${S3_BASE_URI:-${S3_BASE_URI_DEFAULT}}"

aws_args=()
dryrun=0

while [[ $# -gt 0 ]]; do
  case "$1" in
    --profile) aws_args+=(--profile "$2"); shift 2;;
    --region) aws_args+=(--region "$2"); shift 2;;
    --dryrun) dryrun=1; shift;;
    --base-uri) base_uri="$2"; shift 2;;
    -h|--help) usage; exit 0;;
    *) echo "ERROR: Unknown arg: $1" >&2; usage; exit 2;;
  esac
done

command -v aws >/dev/null 2>&1 || { echo "ERROR: Missing command in PATH: aws" >&2; exit 1; }
[[ -n "${base_uri}" ]] || { echo "ERROR: Missing S3 base uri." >&2; exit 1; }

echo "Uploading to: ${base_uri}"

aws_dryrun=()
if [[ "${dryrun}" -eq 1 ]]; then
  aws_dryrun+=(--dryrun)
fi

aws "${aws_args[@]}" s3 cp "${SCRIPT_DIR}/infer.py" "${base_uri%/}/infer.py" --only-show-errors "${aws_dryrun[@]}"
aws "${aws_args[@]}" s3 cp "${SCRIPT_DIR}/requirements.txt" "${base_uri%/}/requirements.txt" --only-show-errors "${aws_dryrun[@]}"
aws "${aws_args[@]}" s3 cp "${SCRIPT_DIR}/run_infer.sh" "${base_uri%/}/run_infer.sh" --only-show-errors "${aws_dryrun[@]}"

echo "Done."

