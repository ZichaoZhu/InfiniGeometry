#!/usr/bin/env bash
set -euo pipefail
EXPERIMENT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$EXPERIMENT/../.."
exec python -u "$EXPERIMENT/automate.py"
