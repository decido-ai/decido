#!/usr/bin/env bash
# Decido one-command setup (macOS / Linux, CPU only):
#   1. llama.cpp (pinned commit) built as CPU shared libraries
#   2. the vserve readout server
#   3. a Python venv with tokenizers + numpy
#   4. the model (Hugging Face) if it is not already in this folder
#   5. a smoke test (example.py)
#
# Usage: ./install.sh
# Options (environment variables):
#   LLAMA_CPP_DIR=/path/to/llama.cpp   reuse an existing checkout (default: third_party/llama.cpp)
#   LLAMA_BUILD_DIR=build-cpu          build directory inside it
#   DECIDO_HF_REPO=decido-ai/decido-26b-a4b  model repo to download from (default)
#   DECIDO_PYTHON=/path/to/python       use an existing Python that has tokenizers + numpy (skips the venv)
#   SKIP_EXAMPLE=1                     do not run the smoke test
set -euo pipefail
cd "$(dirname "$0")"
HERE=$(pwd)
LLAMA_DIR=${LLAMA_CPP_DIR:-third_party/llama.cpp}
LLAMA_REF=${LLAMA_CPP_REF:-b84ea5cf5e56}        # llama.cpp commit used for every reported number
BUILD=${LLAMA_BUILD_DIR:-build-cpu}
HF_REPO=${DECIDO_HF_REPO:-decido-ai/decido-26b-a4b}
GGUF=decido-26b-a4b-q8_0.gguf

need() { command -v "$1" >/dev/null 2>&1 || { echo "error: '$1' is required" >&2; exit 1; }; }
need c++; need python3

if [ "$(uname)" = Darwin ]; then MEM_GB=$(( $(sysctl -n hw.memsize) / 1073741824 )); else
  MEM_GB=$(( $(awk '/MemTotal/ {print $2}' /proc/meminfo) / 1048576 )); fi
if [ "$MEM_GB" -lt 32 ]; then echo "warning: ${MEM_GB} GB RAM detected; Decido Q8_0 needs ~30 GB per process" >&2; fi

echo "==> 1/5 llama.cpp ($LLAMA_REF)"
if [ ! -f "$LLAMA_DIR/include/llama.h" ]; then
  need git
  git clone https://github.com/ggml-org/llama.cpp "$LLAMA_DIR"
  git -C "$LLAMA_DIR" checkout -q "$LLAMA_REF"
fi
LLAMA_DIR=$(cd "$LLAMA_DIR" && pwd)
if ! ls "$LLAMA_DIR/$BUILD/bin/"libllama.* >/dev/null 2>&1; then
  command -v cmake >/dev/null 2>&1 || { echo "error: cmake is required to build llama.cpp. Install it with" >&2
    echo "  brew install cmake      (macOS, Homebrew)   or   python3 -m pip install cmake" >&2; exit 1; }
  FLAGS=(-DCMAKE_BUILD_TYPE=Release -DBUILD_SHARED_LIBS=ON -DLLAMA_CURL=OFF -DLLAMA_BUILD_TESTS=OFF -DGGML_METAL=OFF -DGGML_CUDA=OFF)
  if [ "$(uname)" = Darwin ]; then FLAGS+=(-DGGML_BLAS=ON -DGGML_BLAS_VENDOR=Apple -DGGML_CPU_KLEIDIAI=ON); fi
  cmake -S "$LLAMA_DIR" -B "$LLAMA_DIR/$BUILD" "${FLAGS[@]}"
  cmake --build "$LLAMA_DIR/$BUILD" -j --target llama
fi

echo "==> 2/5 vserve"
./build_vserve.sh "$LLAMA_DIR" "$BUILD"

echo "==> 3/5 Python"
if [ -n "${DECIDO_PYTHON:-}" ]; then PY=$DECIDO_PYTHON; else
  [ -x .venv-decido/bin/python ] || python3 -m venv .venv-decido
  .venv-decido/bin/pip install -q -r requirements.txt
  PY=.venv-decido/bin/python
fi

echo "==> 4/5 model"
if [ ! -f "$GGUF" ] || [ ! -f bundle/template.json ]; then
  "$PY" -m pip install -q huggingface_hub
  "$PY" -c "from huggingface_hub import snapshot_download as d; d('$HF_REPO', local_dir='.', allow_patterns=['$GGUF', 'bundle/*'])"
fi

echo "==> 5/5 smoke test (first load takes 1-2 min)"
if [ -z "${SKIP_EXAMPLE:-}" ]; then "$PY" example.py; fi
echo "done. Python: from decido.predictor_llamacpp import DecidoLlamaCpp (see example.py)"
echo "      Jev-compatible API: $PY -m decido.server --port 8080   (then TYPESAFE_BASE_URL=http://127.0.0.1:8080)"
