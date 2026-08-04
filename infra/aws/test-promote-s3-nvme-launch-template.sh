#!/usr/bin/env bash
set -euo pipefail

script_dir=$(cd "$(dirname "$0")" && pwd)
tmp=$(mktemp -d)
trap 'rm -rf "$tmp"' EXIT

mkdir -p "$tmp/bin"
cat >"$tmp/bin/aws" <<'EOF'
#!/usr/bin/env bash
set -euo pipefail

service=$1
operation=$2
shift 2

if [ "$service" != ec2 ]; then
  echo "unexpected AWS service: $service" >&2
  exit 1
fi

case "$operation" in
  describe-launch-template-versions)
    cat <<'JSON'
{
  "ImageId": "ami-test",
  "BlockDeviceMappings": [
    {
      "DeviceName": "/dev/sda1",
      "Ebs": {
        "DeleteOnTermination": true,
        "VolumeSize": 900,
        "VolumeType": "gp3"
      }
    },
    {
      "DeviceName": "/dev/sdf",
      "Ebs": {
        "DeleteOnTermination": false,
        "SnapshotId": "snap-obsolete",
        "VolumeSize": 300,
        "VolumeType": "gp3"
      }
    }
  ],
  "UserData": "old-user-data"
}
JSON
    ;;
  create-launch-template-version)
    template_data_uri=""
    while [ "$#" -gt 0 ]; do
      if [ "$1" = --launch-template-data ]; then
        shift
        template_data_uri=$1
        break
      fi
      shift
    done
    test -n "$template_data_uri"
    cp "${template_data_uri#file://}" "$FAKE_TEMPLATE_DATA"
    printf '20\n'
    ;;
  modify-launch-template)
    printf '%s\n' "$*" >"$FAKE_MODIFY_LOG"
    ;;
  *)
    echo "unexpected EC2 operation: $operation" >&2
    exit 1
    ;;
esac
EOF
chmod +x "$tmp/bin/aws"

export FAKE_TEMPLATE_DATA="$tmp/template-data.json"
export FAKE_MODIFY_LOG="$tmp/modify.log"

PATH="$tmp/bin:$PATH" \
REGION=us-east-2 \
LAUNCH_TEMPLATE_ID=lt-test \
MODEL_ARTIFACT_URI=s3://example/release \
  "$script_dir/promote-s3-nvme-launch-template.sh" >"$tmp/promote.log"

jq -e '
  [.BlockDeviceMappings[] | select(.DeviceName == "/dev/sdf")]
  == [{"DeviceName": "/dev/sdf", "NoDevice": ""}]
' "$FAKE_TEMPLATE_DATA" >/dev/null
jq -e '
  any(.BlockDeviceMappings[];
    .DeviceName == "/dev/sda1" and .Ebs.DeleteOnTermination == true)
' "$FAKE_TEMPLATE_DATA" >/dev/null
jq -e '.UserData != "old-user-data" and (.UserData | length > 0)' \
  "$FAKE_TEMPLATE_DATA" >/dev/null
grep -q -- '--default-version 20' "$FAKE_MODIFY_LOG"
grep -q 'Promoted lt-test in us-east-2 to version 20.' "$tmp/promote.log"

printf 'launch template promotion tests passed\n'
