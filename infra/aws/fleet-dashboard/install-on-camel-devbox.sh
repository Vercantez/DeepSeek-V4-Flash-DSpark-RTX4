#!/usr/bin/env bash
set -euo pipefail

TARGET="camel-devbox"
REMOTE_DIR="/opt/deepseek-fleet-dashboard"
SOURCE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

scp -q \
  "$SOURCE_DIR/fleet_dashboard.py" \
  "$SOURCE_DIR/deepseek-fleet-dashboard.service" \
  "$TARGET":/tmp/

ssh "$TARGET" "set -e; \
  inbox=/tmp; \
  user=\$(id -un); home=\$(getent passwd \"\$user\" | cut -d: -f6); \
  sudo install -d -m 0755 $REMOTE_DIR; \
  sudo install -m 0644 \"\$inbox/fleet_dashboard.py\" $REMOTE_DIR/fleet_dashboard.py; \
  sed -e \"s|__INSTALL_USER__|\$user|\" -e \"s|__INSTALL_HOME__|\$home|\" \"\$inbox/deepseek-fleet-dashboard.service\" | \
    sudo tee /etc/systemd/system/deepseek-fleet-dashboard.service >/dev/null; \
  rm -f \"\$inbox/fleet_dashboard.py\" \"\$inbox/deepseek-fleet-dashboard.service\"; \
  sudo systemctl daemon-reload; sudo systemctl enable deepseek-fleet-dashboard.service; \
  sudo systemctl restart deepseek-fleet-dashboard.service; \
  sudo systemctl --no-pager --full status deepseek-fleet-dashboard.service"
