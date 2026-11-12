#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."

python_bin="${PYTHON:-}"
if [[ -z "$python_bin" ]]; then
    for candidate in python3.11 python3.12 python3.10 python3; do
        if command -v "$candidate" >/dev/null 2>&1 && "$candidate" -c 'import sys; sys.exit(not ((3, 10) <= sys.version_info[:2] <= (3, 12)))'; then
            python_bin="$candidate"
            break
        fi
    done
fi
if [[ -z "$python_bin" ]]; then
    echo 'Python 3.10, 3.11, or 3.12 is required. Set PYTHON to its executable.' >&2
    exit 1
fi
"$python_bin" -c 'import sys; sys.exit(not ((3, 10) <= sys.version_info[:2] <= (3, 12)))'
command -v cmake >/dev/null 2>&1 || { echo 'CMake 3.14 or newer is required.' >&2; exit 1; }
if [[ -z "${CXX:-}" ]]; then
    if [[ -f /opt/rh/gcc-toolset-14/enable ]]; then
        source /opt/rh/gcc-toolset-14/enable
    else
        for compiler in g++-14 g++-13 g++-12 clang++-20 clang++-19 clang++-18 clang++-17 clang++-16 clang++-15 g++ clang++; do
            if command -v "$compiler" >/dev/null 2>&1; then
                export CXX="$compiler"
                break
            fi
        done
    fi
fi
if [[ ! -d .venv ]]; then
    "$python_bin" -m venv .venv
fi
.venv/bin/python -m pip install -r integration/requirements.txt -e ./SCALE-Sim -e ./ramulator2
cmake -S ramulator2 -B ramulator2/build \
    -DRAMULATOR_PYTHON_BINDINGS=OFF \
    "-DFETCHCONTENT_SOURCE_DIR_YAML-CPP=$PWD/ramulator2/ext/yaml-cpp" \
    "-DFETCHCONTENT_SOURCE_DIR_FMT=$PWD/ramulator2/ext/fmt"
cmake --build ramulator2/build -j "${BUILD_JOBS:-2}"
cmake -S integration -B integration/build
cmake --build integration/build -j "${BUILD_JOBS:-2}"
echo 'Setup complete. Run ./run.sh for the included two-layer GEMM example.'
