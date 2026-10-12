#!/usr/bin/env bash
# Build llama.cpp with the shinbunbun P100 baseline and these patches, for Tesla P100 (sm_60).
#   scripts/build.sh [DIR]          (default DIR: ./llama.cpp-p100)
# Needs git, cmake, a CUDA 12.x toolkit (CUDA 13 dropped sm_60) and a host compiler it accepts (gcc <= 14).
# Set CUDA_HOME if nvcc is not on PATH.
set -euo pipefail

HERE=$(cd "$(dirname "$0")/.." && pwd)
DIR=${1:-llama.cpp-p100}

LLAMA_COMMIT=c21284cdf5fd833b90ac1f824cdf8065e2700dc4      # upstream llama.cpp the baseline was taken on
SHIN_COMMIT=52469952952ee6446207acc574c72055a0685ac4       # shinbunbun/llama-cpp-p100-patches (v0.4.0 rebase)
# the shinbunbun patches that apply cleanly to LLAMA_COMMIT (22 and 31 conflict, the rest need manual rebases)
SHIN_PATCHES="01 02 04 05 07 08 10 11 13 14 16 17 18 21 23 28 29"

git clone https://github.com/ggml-org/llama.cpp "$DIR"
git -C "$DIR" checkout -q "$LLAMA_COMMIT"

git clone -q https://github.com/shinbunbun/llama-cpp-p100-patches "$DIR/.shinbunbun"
git -C "$DIR/.shinbunbun" checkout -q "$SHIN_COMMIT"
for n in $SHIN_PATCHES; do
    git -C "$DIR" apply "$(cd "$DIR" && pwd)/.shinbunbun/patches/$n-"*.patch
done
git -C "$DIR" add -A -- . ':!.shinbunbun'
git -C "$DIR" -c user.name=baseline -c user.email=baseline@localhost commit -qm "shinbunbun/llama-cpp-p100-patches baseline ($SHIN_PATCHES)"

git -C "$DIR" -c user.name=build -c user.email=build@localhost am -q "$HERE"/patches/*.patch

NVCC=${CUDA_HOME:+$CUDA_HOME/bin/}nvcc
EXTRA=()
if [ -n "${CUDA_HOME:-}" ]; then # a toolkit outside the system library path: point the build and the binaries at it
    L="-Wl,-rpath-link,$CUDA_HOME/lib:$CUDA_HOME/lib64 -Wl,-rpath,$CUDA_HOME/lib:$CUDA_HOME/lib64"
    EXTRA=(-DCUDAToolkit_ROOT="$CUDA_HOME" -DCMAKE_EXE_LINKER_FLAGS="$L" -DCMAKE_SHARED_LINKER_FLAGS="$L")
fi
cmake -S "$DIR" -B "$DIR/build" -DCMAKE_BUILD_TYPE=Release -DGGML_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=60 \
    -DCMAKE_CUDA_COMPILER="$(command -v "$NVCC")" -DLLAMA_CURL=OFF "${EXTRA[@]}"
cmake --build "$DIR/build" -j "$(nproc)" --target llama-server llama-bench llama-cli llama-perplexity llama-tokenize

echo "built: $DIR/build/bin (see README.md for the environment variables that switch the kernels on)"
