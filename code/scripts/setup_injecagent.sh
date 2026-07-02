#!/usr/bin/env bash
# Clone the InjecAgent benchmark (data + tools + prompts) used by the injection domain.
# Idempotent. The clone lives under data/ (git-ignored) — it's an external dataset.
set -euo pipefail
CODE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ROOT="$(cd "$CODE_DIR/.." && pwd)"
TARGET="$ROOT/data/InjecAgent"
if [ -d "$TARGET/data" ]; then
  echo "InjecAgent already present: $TARGET"
else
  echo "Cloning InjecAgent -> $TARGET"
  git clone --depth 1 https://github.com/uiuc-kang-lab/InjecAgent.git "$TARGET"
fi
