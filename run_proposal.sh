#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"
if [[ ! -x .venv/bin/python ]]; then
    bash setup.sh
fi
export PYTHONPATH="$PWD/SCALE-Sim:$PWD/ramulator2/python${PYTHONPATH:+:$PYTHONPATH}"
exec .venv/bin/python integration/proposal.py "$@"
