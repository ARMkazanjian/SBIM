#!/usr/bin/env bash
# Freeze the exact resolved environment into requirements.lock (spec §13.1). Run inside the training venv.
set -euo pipefail
cd "$(dirname "$0")/.."
python -m pip freeze --exclude-editable | grep -v '^pkg_resources' > requirements.lock
echo "wrote requirements.lock ($(wc -l < requirements.lock) packages)"
