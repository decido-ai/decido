#!/bin/bash
# Build the Decido readout server against a CPU llama.cpp build (shared libs, Metal/CUDA not needed).
# Usage: ./build_vserve.sh /path/to/llama.cpp [build-dir]
#   llama.cpp build example: cmake -B build -DGGML_METAL=OFF -DBUILD_SHARED_LIBS=ON && cmake --build build -j
set -e
L=${1:?usage: $0 /path/to/llama.cpp [build-dir]}; BD=${2:-build}
cd "$(dirname "$0")/tools/vserve"
c++ -O2 -std=c++17 -o vserve vserve.cpp -I"$L/include" -I"$L/ggml/include" -L"$L/$BD/bin" \
    -lllama -lggml -lggml-base -Wl,-rpath,"$L/$BD/bin"
echo "built $(pwd)/vserve"
