#!/usr/bin/env bash
set -euo pipefail

script_dir=$(cd "$(dirname "$0")" && pwd)
watcher="$script_dir/deepseek-spot-interruption-watcher.sh"
tmp=$(mktemp -d)
trap 'rm -rf "$tmp"' EXIT

bash -n "$watcher"
bash -n "$script_dir/install-spot-interruption-watcher.sh"
bash -n "$script_dir/install-gpu-node-service.sh"
bash -n "$script_dir/worker-user-data-s3-nvme.sh"

cat >"$tmp/curl" <<'EOF'
#!/usr/bin/env bash
set -euo pipefail
printf '%q ' "$@" >>"$FAKE_CURL_LOG"
printf '\n' >>"$FAKE_CURL_LOG"
url=${!#}
if [ "$url" = "$IMDS_BASE_URL/api/token" ]; then
  printf 'test-token'
  exit 0
fi
output=""
previous=""
for argument in "$@"; do
  if [ "$previous" = --output ]; then
    output=$argument
    break
  fi
  previous=$argument
done
case "$url" in
  */meta-data/spot/instance-action)
    printf '404'
    ;;
  */meta-data/events/recommendations/rebalance)
    printf '{"noticeTime":"test"}' >"$output"
    printf '200'
    ;;
  *)
    echo "unexpected URL: $url" >&2
    exit 1
    ;;
esac
EOF

cat >"$tmp/iptables" <<'EOF'
#!/usr/bin/env bash
set -euo pipefail
printf '%q ' "$@" >>"$FAKE_IPTABLES_LOG"
printf '\n' >>"$FAKE_IPTABLES_LOG"
# The chain does not exist in this isolated test before the watcher creates it.
if printf '%s\n' "$*" | grep -q -- '-C INPUT'; then
  exit 1
fi
EOF

chmod +x "$tmp/curl" "$tmp/iptables"
export FAKE_CURL_LOG="$tmp/curl.log"
export FAKE_IPTABLES_LOG="$tmp/iptables.log"
export IMDS_BASE_URL=http://imds.test/latest

RUN_DIR="$tmp/run" \
CURL_BIN="$tmp/curl" \
IPTABLES_BIN="$tmp/iptables" \
EXIT_AFTER_DRAIN=1 \
  "$watcher" watch >"$tmp/watcher.log"

grep -q '^rebalance-recommendation:' "$tmp/run/draining"
grep -q -- '--ctstate NEW' "$tmp/iptables.log"
grep -q -- '--dport 8000' "$tmp/iptables.log"
grep -q 'spot/instance-action' "$tmp/curl.log"
grep -q 'events/recommendations/rebalance' "$tmp/curl.log"
grep -q 'rejecting new TCP connections' "$tmp/watcher.log"

printf 'spot interruption watcher tests passed\n'
