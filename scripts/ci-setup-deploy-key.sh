#!/usr/bin/env bash
# One-shot helper: generate GitHub Actions deploy key, install pubkey on LJ VM,
# push secrets to RockInMyHead/biobook. Run from a machine that can SSH as user1.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
KEY_DIR="${ROOT}/.deploy-keys"
KEY_PATH="${KEY_DIR}/github-actions-labjournal"
HOST="${DEPLOY_HOST:-176.109.103.209}"
USER_NAME="${DEPLOY_USER:-user1}"
ADMIN_KEY="${ADMIN_SSH_KEY:-$HOME/.ssh/id_ed25519_anticus_cloudru}"
REPO="${GITHUB_REPO:-RockInMyHead/biobook}"

mkdir -p "$KEY_DIR"
chmod 700 "$KEY_DIR"

if [[ ! -f "$KEY_PATH" ]]; then
  ssh-keygen -t ed25519 -N "" -C "github-actions@labjournal.tech" -f "$KEY_PATH"
fi

PUB="$(cat "${KEY_PATH}.pub")"
echo "Installing pubkey on ${USER_NAME}@${HOST}…"
ssh -i "$ADMIN_KEY" -o IdentitiesOnly=yes -o IPQoS=throughput -o ConnectTimeout=30 \
  "${USER_NAME}@${HOST}" "bash -s" <<REMOTE
set -euo pipefail
mkdir -p ~/.ssh
chmod 700 ~/.ssh
touch ~/.ssh/authorized_keys
chmod 600 ~/.ssh/authorized_keys
if ! grep -qF '$PUB' ~/.ssh/authorized_keys; then
  echo '$PUB' >> ~/.ssh/authorized_keys
  echo added
else
  echo already_present
fi
REMOTE

echo "Writing GitHub secrets on ${REPO}…"
gh secret set DEPLOY_SSH_KEY -R "$REPO" < "$KEY_PATH"
gh secret set DEPLOY_HOST -R "$REPO" -b "$HOST"
gh secret set DEPLOY_USER -R "$REPO" -b "$USER_NAME"
gh secret set DEPLOY_PORT -R "$REPO" -b "22"

echo "Done. Private key kept at ${KEY_PATH} (gitignored). Do not commit it."
