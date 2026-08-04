#!/usr/bin/env bash
set -euo pipefail

: "${SOURCE_ARTIFACT_URI:?Set the published source S3 release URI}"
: "${TARGET_ARTIFACT_URI:?Set the published target S3 release URI}"
: "${RUNTIME_CACHE_OBJECT:?Set the immutable runtime-cache object name}"
: "${SOURCE_REGION:=us-east-2}"
: "${TARGET_REGION:?Set the destination AWS region}"

if ! aws s3 ls "$TARGET_ARTIFACT_URI/artifact.json" >/dev/null 2>&1; then
  echo "Target model release is not published: $TARGET_ARTIFACT_URI" >&2
  exit 1
fi
if aws s3 ls "$TARGET_ARTIFACT_URI/$RUNTIME_CACHE_OBJECT" >/dev/null 2>&1; then
  echo "Refusing to overwrite runtime cache: $TARGET_ARTIFACT_URI/$RUNTIME_CACHE_OBJECT" >&2
  exit 1
fi

aws s3 cp "$SOURCE_ARTIFACT_URI/$RUNTIME_CACHE_OBJECT" "$TARGET_ARTIFACT_URI/$RUNTIME_CACHE_OBJECT" \
  --source-region "$SOURCE_REGION" --region "$TARGET_REGION" --only-show-errors
aws s3 cp "$SOURCE_ARTIFACT_URI/$RUNTIME_CACHE_OBJECT.sha256" "$TARGET_ARTIFACT_URI/$RUNTIME_CACHE_OBJECT.sha256" \
  --source-region "$SOURCE_REGION" --region "$TARGET_REGION" --only-show-errors

echo "Replicated $RUNTIME_CACHE_OBJECT to $TARGET_ARTIFACT_URI"
