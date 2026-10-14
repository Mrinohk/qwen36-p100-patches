#!/usr/bin/env bash
# The configuration the numbers in README.md were measured with: 2x P100, Qwen3.6-35B-A3B UD-Q4_K_XL fully resident,
# MTP speculative decoding with the reduced draft vocabulary.
#   scripts/run-server.sh /path/to/llama.cpp-p100 /path/to/Qwen3.6-35B-A3B-UD-Q4_K_XL.gguf [extra llama-server args]
set -euo pipefail
HERE=$(cd "$(dirname "$0")/.." && pwd)
LLAMA=$1; MODEL=$2; shift 2

export GGML_CUDA_P60_MMVQ=1                                  # sm_60 fp16 matvec + fusions (required by the rest)
export GGML_CUDA_P60_MEGA=1                                  # persistent layer kernels, verify / prefill kernels
export LLAMA_MTP_DRAFT_VOCAB="$HERE/draftvocab/dv32k.bin"    # draft head scores 32k frequent tokens, not 248k
export GGML_CUDA_P60_INVAR=1                                 # same output with and without MTP (~1-3% slower)
export CUDA_DEVICE_ORDER=PCI_BUS_ID

exec "$LLAMA/build/bin/llama-server" --model "$MODEL" -ngl 99 -sm layer -ts 1.15,1 -fa on \
    -c 65536 -b 6144 -ub 1536 --parallel 1 --kv-unified \
    --spec-type draft-mtp --spec-draft-n-max 5 --spec-draft-p-min 0.55 \
    --temperature 1.0 --top-k 40 --top-p 1.0 --min-p 0.05 "$@"
