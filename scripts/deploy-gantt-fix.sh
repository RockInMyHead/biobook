#!/bin/bash
# Выкладка правки диаграммы Ганта (app.py + projects.html)
set -euo pipefail
HOST="${DEPLOY_HOST:-user1@82.202.137.235}"
KEY="${DEPLOY_KEY:-$HOME/.ssh/id_ed25519_anticus_cloudru}"
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
scp -i "$KEY" "$ROOT/biobook/web/app.py" "$ROOT/biobook/web/templates/projects.html" "$HOST:/tmp/"
ssh -i "$KEY" "$HOST" 'sudo cp /tmp/app.py /opt/bio-notebook/web/app.py && \
  sudo cp /tmp/projects.html /opt/bio-notebook/web/templates/projects.html && \
  sudo chown www-data:www-data /opt/bio-notebook/web/app.py /opt/bio-notebook/web/templates/projects.html && \
  sudo systemctl restart biobook && systemctl is-active biobook'
echo "OK: gantt fix deployed"
