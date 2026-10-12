#!/usr/bin/env bash
# Build a draft vocabulary for LLAMA_MTP_DRAFT_VOCAB from your own text.
#   scripts/build-draft-vocab.sh LLAMA_DIR MODEL.gguf OUT.bin SIZE file1.txt [file2.txt ...]
# Each input file is one "source" and gets equal weight, whatever its size: the token frequencies of each file are
# normalized before they are summed, so a huge file does not drown out a small one. The SIZE most frequent tokens
# plus every special/control token are written as int32 token ids. Needs python3 with numpy and gguf (pip install gguf).
# Example (how draftvocab/dv32k.bin was made, with the sources listed in README.md):
#   scripts/build-draft-vocab.sh ./llama.cpp-p100 Qwen3.6-35B-A3B-UD-Q4_K_XL.gguf dv32k.bin 32768 prose.txt code.txt docs.txt man.txt py.txt
set -euo pipefail
LLAMA=$1; MODEL=$2; OUT=$3; SIZE=$4; shift 4
TMP=$(mktemp -d); trap 'rm -rf "$TMP"' EXIT
i=0
for f in "$@"; do
    CUDA_VISIBLE_DEVICES= "$LLAMA/build/bin/llama-tokenize" -m "$MODEL" -f "$f" --ids --log-disable 2>/dev/null > "$TMP/$i.ids"
    i=$((i + 1))
done
python3 - "$MODEL" "$OUT" "$SIZE" "$TMP" "$i" <<'PY'
import json, sys, numpy as np
from gguf import GGUFReader
model, out, size, tmp, n = sys.argv[1], sys.argv[2], int(sys.argv[3]), sys.argv[4], int(sys.argv[5])
f = GGUFReader(model).fields['tokenizer.ggml.token_type']
tt = np.array([f.parts[i][0] for i in f.data]); nv = len(tt)
special = np.nonzero(tt != 1)[0]                 # everything that is not a NORMAL token
cnt = np.zeros(nv)
for k in range(n):
    ids = np.array(json.loads(open(f'{tmp}/{k}.ids').read()), dtype=np.int64)
    cnt += np.bincount(ids, minlength=nv) / max(len(ids), 1)
order = np.argsort(-cnt, kind='stable')
v = np.unique(np.concatenate([order[:size], special])).astype(np.int32)
v.tofile(out)
print(f'{out}: {len(v)} tokens ({size} most frequent + {len(special)} special) of {nv}')
PY
