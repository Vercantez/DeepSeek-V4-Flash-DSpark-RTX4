#!/usr/bin/env bash
set -euo pipefail

REPO_DIR="${REPO_DIR:-/opt/deepseek/mia-dspark-rtx4}"
HF_CACHE="${HF_CACHE:-/opt/deepseek-cache/hf}"

if [ ! -d "$REPO_DIR" ]; then
  sudo mkdir -p "$(dirname "$REPO_DIR")"
  sudo git clone https://github.com/Vercantez/DeepSeek-V4-Flash-DSpark-RTX4 "$REPO_DIR"
fi

sudo chown -R ubuntu:ubuntu "$REPO_DIR"
sudo -u ubuntu git -C "$REPO_DIR" fetch --all --prune
sudo -u ubuntu git -C "$REPO_DIR" checkout main
sudo -u ubuntu git -C "$REPO_DIR" pull --ff-only

sudo mkdir -p "$HF_CACHE"
sudo chown -R ubuntu:ubuntu "$(dirname "$HF_CACHE")"

ENV_FILE="$REPO_DIR/.env.rtx4.stock-sm120"
sudo -u ubuntu cp "$REPO_DIR/.env.rtx4.stock-sm120.example" "$ENV_FILE"
sudo -u ubuntu sed -i "s#^HF_CACHE=.*#HF_CACHE=$HF_CACHE#" "$ENV_FILE"
sudo "$REPO_DIR/infra/aws/prepare-kv-offload-nvme.sh"
if [ -d "$HF_CACHE/hub/models--deepseek-ai--DeepSeek-V4-Flash-0731/snapshots" ]; then
  snapshot="$(find "$HF_CACHE/hub/models--deepseek-ai--DeepSeek-V4-Flash-0731/snapshots" -mindepth 1 -maxdepth 1 -type d | sort | tail -1)"
  if [ -n "$snapshot" ]; then
    container_snapshot="/cache/huggingface${snapshot#"$HF_CACHE"}"
    if grep -q '^MODEL_DIR=' "$ENV_FILE"; then
      sudo -u ubuntu sed -i "s#^MODEL_DIR=.*#MODEL_DIR=$container_snapshot#" "$ENV_FILE"
    else
      echo "MODEL_DIR=$container_snapshot" | sudo -u ubuntu tee -a "$ENV_FILE" >/dev/null
    fi
  fi
fi

sudo tee /etc/systemd/system/deepseek-rtx4.service >/dev/null <<EOF
[Unit]
Description=DeepSeek V4 Flash stock SM120 TP4+EP vLLM service
After=docker.service network-online.target deepseek-spot-interruption-watcher.service
Wants=network-online.target deepseek-spot-interruption-watcher.service
Requires=docker.service
RequiresMountsFor=$HF_CACHE /opt/dlami/nvme/kv-offload

[Service]
Type=oneshot
User=ubuntu
WorkingDirectory=$REPO_DIR
Environment=ENV_FILE=$ENV_FILE
ExecStartPre=+/usr/bin/find /dev/shm -maxdepth 1 -type f -name vllm_offload_*.mmap -delete
ExecStart=$REPO_DIR/start-deepseek-v4-flash-stock-sm120-rtx4.sh
ExecStop=-/usr/bin/docker rm -f deepseek-v4-flash-stock-sm120-rtx4
RemainAfterExit=yes
TimeoutStartSec=1800

[Install]
WantedBy=multi-user.target
EOF

sudo tee /usr/local/bin/deepseek-rtx4-health >/dev/null <<'EOF'
#!/usr/bin/env bash
set -euo pipefail
curl -fsS --max-time 5 http://127.0.0.1:8000/v1/models >/dev/null
EOF
sudo chmod +x /usr/local/bin/deepseek-rtx4-health

sudo "$REPO_DIR/infra/aws/install-spot-interruption-watcher.sh"
sudo systemctl daemon-reload
sudo systemctl enable deepseek-rtx4.service
