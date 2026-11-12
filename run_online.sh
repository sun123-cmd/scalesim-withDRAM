#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"
if [[ ! -x .venv/bin/python || ! -x integration/build/ramulator2-online || ! -f ramulator2/libramulator.so ]]; then
    bash setup.sh
fi
export PYTHONPATH="$PWD/SCALE-Sim:$PWD/ramulator2/python:$PWD/integration${PYTHONPATH:+:$PYTHONPATH}"
if [[ "$#" -eq 0 ]]; then
    set -- simulate
fi
exec .venv/bin/python integration/online.py "$@"
