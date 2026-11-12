#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"
if [[ ! -x .venv/bin/python || ! -x integration/build/ramulator2-replay || ! -f ramulator2/libramulator.so ]]; then
    bash setup.sh
fi
mkdir -p results/pi05
.venv/bin/python integration/pi05_model.py --output results/pi05/input_manifest.json
exec .venv/bin/python integration/pi05_runner.py \
    --manifest results/pi05/input_manifest.json --output results/pi05 "$@"
