#!/usr/bin/env bash
set -e

cd "$(dirname "$0")"

mkdir -p workspace
touch workspace/.gitkeep

if [ ! -f .env.openclaw ]; then
  cp .env.example .env.openclaw
  echo ".env.openclaw を作成しました。"
  echo "次に nano .env.openclaw で DISCORD_TOKEN と API キーを入力してください。"
else
  echo ".env.openclaw は既に存在します。"
fi

echo
echo "セットアップ完了"
echo "次の手順:"
echo "  nano .env.openclaw"
echo "  ./start.sh"
