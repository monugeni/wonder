#!/usr/bin/env bash
#
# Deploy Wonder MCP Server on Ubuntu
#
# Usage:
#   git clone <repo> /btrfs/wonder
#   cd /btrfs/wonder && sudo bash deploy/setup.sh
#
set -euo pipefail

APP_DIR="/btrfs/wonder"
APP_USER="wonder"

echo "=== Wonder — Ubuntu Deployment ==="

# 1. System packages
echo "[1/6] Installing system packages..."
apt-get update -qq
apt-get install -y -qq python3 python3-venv python3-pip docker.io docker-compose-v2 > /dev/null

# 2. Create app user
echo "[2/6] Creating app user..."
id -u "$APP_USER" &>/dev/null || useradd -r -m -s /bin/bash "$APP_USER"
usermod -aG docker "$APP_USER"
chown -R "$APP_USER:$APP_USER" "$APP_DIR"

# 3. Python venv + dependencies
echo "[3/6] Creating Python venv and installing dependencies..."
sudo -u "$APP_USER" bash -c "
    cd $APP_DIR
    python3 -m venv venv
    source venv/bin/activate
    pip install --upgrade pip -q
    pip install -r requirements.txt -q
"

# 5. .env file
if [ ! -f "$APP_DIR/.env" ]; then
    echo "[4/6] Creating .env from template..."
    cp "$APP_DIR/.env.example" "$APP_DIR/.env"
    echo ""
    echo "  *** IMPORTANT: Edit $APP_DIR/.env and set your ANTHROPIC_API_KEY ***"
    echo ""
else
    echo "[4/6] .env already exists — keeping it"
fi

# 6. Start Qdrant
echo "[5/6] Starting Qdrant..."
systemctl enable docker
systemctl start docker
cd "$APP_DIR"
sudo -u "$APP_USER" docker compose up -d

# 7. Install and start systemd service
echo "[6/6] Installing systemd service..."
cp "$APP_DIR/deploy/wonder.service" /etc/systemd/system/
systemctl daemon-reload
systemctl enable wonder

echo ""
echo "=== Setup complete ==="
echo ""
echo "Next steps:"
echo "  1. Edit /btrfs/wonder/.env and set ANTHROPIC_API_KEY"
echo "  2. Start the server:"
echo "       sudo systemctl start wonder"
echo "  3. Check logs:"
echo "       journalctl -u wonder -f"
echo "  4. MCP SSE endpoint will be at:"
echo "       http://<your-server-ip>:8080/sse"
echo ""
echo "Claude Desktop config (on your Mac):"
echo '  {'
echo '    "mcpServers": {'
echo '      "wonder": {'
echo '        "url": "http://<your-server-ip>:8080/sse"'
echo '      }'
echo '    }'
echo '  }'
