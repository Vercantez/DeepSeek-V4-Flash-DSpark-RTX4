#!/usr/bin/env bash
set -euo pipefail

: "${REGION:?Set the AWS region}"
: "${LAUNCH_TEMPLATE_ID:?Set the worker launch template ID}"
: "${MODEL_ARTIFACT_URI:?Set the regional S3 release URI}"

script_dir=$(cd "$(dirname "$0")" && pwd)
user_data_file=$(mktemp)
template_data_file=$(mktemp)
trap 'rm -f "$user_data_file" "$template_data_file"' EXIT

head -n 1 "$script_dir/worker-user-data-s3-nvme.sh" >"$user_data_file"
printf 'MODEL_ARTIFACT_URI=%q\n' "$MODEL_ARTIFACT_URI" >>"$user_data_file"
if [ -n "${RUNTIME_CACHE_OBJECT:-}" ]; then
  printf 'RUNTIME_CACHE_OBJECT=%q\n' "$RUNTIME_CACHE_OBJECT" >>"$user_data_file"
fi
tail -n +2 "$script_dir/worker-user-data-s3-nvme.sh" >>"$user_data_file"
user_data=$(base64 <"$user_data_file" | tr -d '\n')

aws ec2 describe-launch-template-versions \
  --region "$REGION" \
  --launch-template-id "$LAUNCH_TEMPLATE_ID" \
  --versions '$Default' \
  --query 'LaunchTemplateVersions[0].LaunchTemplateData' \
  --output json |
  jq --arg user_data "$user_data" \
    'del(.BlockDeviceMappings[]? | select(.DeviceName == "/dev/sdf")) | .UserData = $user_data' \
    >"$template_data_file"

version=$(aws ec2 create-launch-template-version \
  --region "$REGION" \
  --launch-template-id "$LAUNCH_TEMPLATE_ID" \
  --source-version '$Default' \
  --version-description 'stock-sm120-tp4-ep-stage-model-from-s3-to-local-nvme' \
  --launch-template-data "file://$template_data_file" \
  --query 'LaunchTemplateVersion.VersionNumber' \
  --output text)

aws ec2 modify-launch-template \
  --region "$REGION" \
  --launch-template-id "$LAUNCH_TEMPLATE_ID" \
  --default-version "$version" >/dev/null

echo "Promoted $LAUNCH_TEMPLATE_ID in $REGION to version $version."
