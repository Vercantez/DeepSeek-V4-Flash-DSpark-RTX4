#!/usr/bin/env bash
set -euo pipefail

# Poll EC2 IMDSv2 for Spot interruption and rebalance notices. When either is
# present, reject only NEW connections to vLLM. Existing TCP streams remain
# established and can finish during the remaining notice window. The sticky
# router's health probe then fails and removes this worker from its backend set.

IMDS_BASE_URL=${IMDS_BASE_URL:-http://169.254.169.254/latest}
BACKEND_PORT=${BACKEND_PORT:-8000}
POLL_INTERVAL=${POLL_INTERVAL:-5}
RUN_DIR=${RUN_DIR:-/run/deepseek-rtx4}
DRAIN_MARKER=${DRAIN_MARKER:-$RUN_DIR/draining}
IPTABLES_BIN=${IPTABLES_BIN:-iptables}
CURL_BIN=${CURL_BIN:-curl}
SLEEP_BIN=${SLEEP_BIN:-sleep}
CHAIN=${CHAIN:-DEEPSEEK_RTX4_DRAIN}
EXIT_AFTER_DRAIN=${EXIT_AFTER_DRAIN:-0}
IMDS_TOKEN_TTL=${IMDS_TOKEN_TTL:-21600}

token=""

log() {
  printf '%s deepseek-spot-watcher: %s\n' "$(date --iso-8601=seconds)" "$*"
}

ensure_chain() {
  "$IPTABLES_BIN" -w -N "$CHAIN" 2>/dev/null || true
  if ! "$IPTABLES_BIN" -w -C INPUT -j "$CHAIN" 2>/dev/null; then
    "$IPTABLES_BIN" -w -I INPUT 1 -j "$CHAIN"
  fi
}

drain() {
  local reason=${1:-manual}
  local was_draining=0
  install -d -m 0755 "$RUN_DIR"
  [ -e "$DRAIN_MARKER" ] && was_draining=1

  # Reconcile the chain even when the marker already exists, so restarting the
  # watcher cannot accidentally leave an advertised worker accepting traffic.
  ensure_chain
  "$IPTABLES_BIN" -w -F "$CHAIN"
  "$IPTABLES_BIN" -w -A "$CHAIN" \
    -p tcp --dport "$BACKEND_PORT" \
    -m conntrack --ctstate NEW \
    -j REJECT --reject-with tcp-reset
  if [ "$was_draining" -eq 0 ]; then
    printf '%s\n' "$reason" >"$DRAIN_MARKER"
    log "draining; rejecting new TCP connections on port $BACKEND_PORT; reason=$reason"
  else
    log "drain rule reconciled; reason=$reason"
  fi
}

reset_drain() {
  if "$IPTABLES_BIN" -w -C INPUT -j "$CHAIN" 2>/dev/null; then
    "$IPTABLES_BIN" -w -D INPUT -j "$CHAIN"
  fi
  "$IPTABLES_BIN" -w -F "$CHAIN" 2>/dev/null || true
  "$IPTABLES_BIN" -w -X "$CHAIN" 2>/dev/null || true
  rm -f "$DRAIN_MARKER"
  log "drain state reset"
}

refresh_token() {
  token=$(
    "$CURL_BIN" --fail --silent --show-error --max-time 2 \
      --request PUT \
      --header "X-aws-ec2-metadata-token-ttl-seconds: $IMDS_TOKEN_TTL" \
      "$IMDS_BASE_URL/api/token"
  )
}

# Prints the response body and returns success only when the metadata endpoint
# reports a notice. A 404 means no notice; 401 asks the caller to refresh IMDSv2.
imds_notice() {
  local path=$1
  local body_file status
  body_file=$(mktemp)
  status=$(
    "$CURL_BIN" --silent --show-error --max-time 2 \
      --output "$body_file" --write-out '%{http_code}' \
      --header "X-aws-ec2-metadata-token: $token" \
      "$IMDS_BASE_URL/meta-data/$path" || true
  )
  case "$status" in
    200)
      cat "$body_file"
      rm -f "$body_file"
      return 0
      ;;
    401)
      token=""
      ;;
  esac
  rm -f "$body_file"
  return 1
}

watch() {
  local notice
  if [ -e "$DRAIN_MARKER" ]; then
    drain "marker-present"
    return
  fi

  log "watching IMDSv2 for Spot interruption and rebalance notices"
  while true; do
    if [ -z "$token" ] && ! refresh_token; then
      log "unable to obtain IMDSv2 token; retrying"
      "$SLEEP_BIN" "$POLL_INTERVAL"
      continue
    fi

    if notice=$(imds_notice 'spot/instance-action'); then
      drain "spot-instance-action:$notice"
      return
    fi
    if [ -z "$token" ]; then
      continue
    fi
    if notice=$(imds_notice 'events/recommendations/rebalance'); then
      drain "rebalance-recommendation:$notice"
      return
    fi

    "$SLEEP_BIN" "$POLL_INTERVAL"
  done
}

case "${1:-watch}" in
  --drain)
    drain "manual"
    ;;
  --reset)
    reset_drain
    ;;
  watch)
    watch
    ;;
  *)
    echo "usage: $0 [watch|--drain|--reset]" >&2
    exit 2
    ;;
esac

if [ -e "$DRAIN_MARKER" ] && [ "$EXIT_AFTER_DRAIN" != 1 ]; then
  while true; do
    "$SLEEP_BIN" 3600
  done
fi
