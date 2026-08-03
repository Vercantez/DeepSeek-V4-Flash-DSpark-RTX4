#!/usr/bin/env bash
set -euo pipefail

# Replicate a published immutable release to another regional artifact bucket.

: "${SOURCE_ARTIFACT_URI:?Set the published source S3 release URI}"
: "${TARGET_ARTIFACT_URI:?Set the empty target S3 release URI}"
: "${SOURCE_REGION:=us-east-2}"
: "${TARGET_REGION:?Set the destination AWS region}"

if ! aws s3 ls "$SOURCE_ARTIFACT_URI/artifact.json" >/dev/null 2>&1; then
  echo "Source release is not published: $SOURCE_ARTIFACT_URI" >&2
  exit 1
fi
if aws s3 ls "$TARGET_ARTIFACT_URI/artifact.json" >/dev/null 2>&1; then
  echo "Refusing to overwrite published target release: $TARGET_ARTIFACT_URI" >&2
  exit 1
fi

aws s3 sync "$SOURCE_ARTIFACT_URI/hf/" "$TARGET_ARTIFACT_URI/hf/" \
  --source-region "$SOURCE_REGION" --region "$TARGET_REGION" --only-show-errors
aws s3 cp "$SOURCE_ARTIFACT_URI/manifest.sha256" "$TARGET_ARTIFACT_URI/manifest.sha256" \
  --source-region "$SOURCE_REGION" --region "$TARGET_REGION" --only-show-errors
aws s3 cp "$SOURCE_ARTIFACT_URI/manifest.sha256.sha256" "$TARGET_ARTIFACT_URI/manifest.sha256.sha256" \
  --source-region "$SOURCE_REGION" --region "$TARGET_REGION" --only-show-errors
aws s3 cp "$SOURCE_ARTIFACT_URI/artifact.json" "$TARGET_ARTIFACT_URI/artifact.json" \
  --source-region "$SOURCE_REGION" --region "$TARGET_REGION" --only-show-errors

echo "Replicated $SOURCE_ARTIFACT_URI to $TARGET_ARTIFACT_URI"
