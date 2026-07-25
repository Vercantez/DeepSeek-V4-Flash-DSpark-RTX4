#!/usr/bin/env bash
set -euo pipefail

script_dir=$(cd "$(dirname "$0")" && pwd)

install -m 0755 \
  "$script_dir/deepseek-spot-interruption-watcher.sh" \
  /usr/local/bin/deepseek-spot-interruption-watcher

cat >/etc/systemd/system/deepseek-spot-interruption-watcher.service <<'EOF'
[Unit]
Description=Drain DeepSeek vLLM worker on EC2 Spot interruption
Documentation=https://docs.aws.amazon.com/AWSEC2/latest/UserGuide/spot-instance-termination-notices.html
After=network-online.target
Wants=network-online.target
Before=deepseek-rtx4.service

[Service]
Type=simple
Environment=BACKEND_PORT=8000
ExecStart=/usr/local/bin/deepseek-spot-interruption-watcher watch
Restart=on-failure
RestartSec=2
User=root

[Install]
WantedBy=multi-user.target
EOF

systemctl daemon-reload
systemctl enable --now deepseek-spot-interruption-watcher.service
