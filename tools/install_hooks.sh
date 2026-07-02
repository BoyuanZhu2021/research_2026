#!/usr/bin/env bash
# One-shot installer: point git at the in-repo .githooks/ directory.
# Run once per fresh clone.
set -e

REPO_ROOT="$(git rev-parse --show-toplevel 2>/dev/null || true)"
if [ -z "$REPO_ROOT" ]; then
  echo "[install_hooks] Not inside a git repo. Run 'git init' first." >&2
  exit 1
fi

cd "$REPO_ROOT"
chmod +x .githooks/pre-commit
git config core.hooksPath .githooks

echo "[install_hooks] ok: pre-commit hook enabled via .githooks/"
echo "[install_hooks] Try a dry run: python3 tools/lint_protocol.py"
