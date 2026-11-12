#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"
if [[ ! -x .venv/bin/python || ! -x integration/build/ramulator2-replay || ! -f ramulator2/libramulator.so ]]; then
    bash setup.sh
fi
if [[ "$#" -eq 0 ]]; then
    set -- -c examples/scale.cfg -t examples/gemm.csv -l examples/layout.csv -i gemm -p results/smoke
fi
exec .venv/bin/python -m scalesim.ramulator2 "$@"
