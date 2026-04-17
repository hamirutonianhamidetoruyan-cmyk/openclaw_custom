#!/usr/bin/env bash
set -e

cd "$(dirname "$0")"

if ! command -v docker >/dev/null 2>&1; then
  echo "Docker が見つかりません。Docker Desktop を起動してください。"
  exit 1
fi

if [ ! -f .env.openclaw ]; then
  echo ".env.openclaw がありません。"
  echo "先に ./setup.sh を実行してください。"
  exit 1
fi

mkdir -p workspace
touch workspace/.gitkeep

echo "openclaw_custom を起動します..."
docker compose up --build
