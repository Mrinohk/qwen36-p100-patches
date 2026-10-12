# Qwen3.6-35B-A3B on Tesla P100: llama.cpp patches

31 patches for [llama.cpp](https://github.com/ggml-org/llama.cpp) that make the Qwen3.5/3.6 MoE models (tested on
**Qwen3.6-35B-A3B, Unsloth UD-Q4_K_XL**) run about **2x faster on Tesla P100 (GP100, sm_60)** cards, for both
decoding and prompt processing.

## Built on shinbunbun's P100 patches

This work starts from **[shinbunbun/llama-cpp-p100-patches](https://github.com/shinbunbun/llama-cpp-p100-patches)**,
the P100 patch set that worked out why GP100 is different (no DP4A, full-rate fp16 HFMA2) and turned that into
measured, documented llama.cpp changes. Their patches are the baseline here, and every number below compares
against llama.cpp *with their patches applied* - not against plain llama.cpp. Go read their README; most of the
ideas in this repo grew out of it.

The baseline used is upstream llama.cpp `c21284cdf` plus the 17 shinbunbun patches (at their commit `5246995`,
the v0.4.0 rebase) that apply cleanly to that tree: `01 02 04 05 07 08 10 11 13 14 16 17 18 21 23 28 29`.
`scripts/build.sh` reproduces it exactly, then applies the patches from this repo on top.

## Ideas from open-jet's megakernel

The two ideas behind the biggest decode gains came from
**[L-Forster/open-jet's megakernel](https://github.com/L-Forster/open-jet/tree/master/megakernel)** (Qwen3.8-27B on an
RTX 3090): running whole layers as persistent kernels instead of hundreds of small launches, and restricting the
MTP draft head's output layer to a list of frequent tokens. No code from it is used here - it is AGPL and built on
sm_80 features (cp.async, tensor cores) a P100 does not have - but the approach is theirs.

## Results

Two Tesla P100 16 GB PCIe cards, layer split (`-sm layer -ts 1.15,1`; one card on PCIe 3.0 x4, one on x16),
Qwen3.6-35B-A3B UD-Q4_K_XL fully on the GPUs, flash attention on, CUDA 12.9, driver 575. Same build options and
same flags for both; every run started with both cards at or below 60 C.

| Test | llama.cpp + shinbunbun | + these patches | |
|---|---:|---:|---:|
| Decode, no speculation (`llama-bench` tg128) | 68.0 t/s | **125.7 t/s** | 1.85x |
| Decode with MTP speculation (`llama-server`, 8 prompts x 256 tokens, sampler temp 1.0 / top-k 40 / min-p 0.05) | 93.4 t/s | **183.1 t/s** | 1.96x |
| Prompt processing, 512 tokens (`-ub 1536`) | 468.5 t/s | **974.1 t/s** | 2.08x |
| Prompt processing, 4096 tokens (`-ub 1536`) | 911.4 t/s | **1590.4 t/s** | 1.75x |
| Prompt processing, 1536 new tokens at 24k context | 466.3 t/s | **708.4 t/s** | 1.52x |
| 28k-token prompt in `llama-server`: prompt processing | ~712 t/s | **~1095 t/s** | 1.53x |
| 28k-token prompt in `llama-server`: MTP decode afterwards | ~78 t/s | **~156 t/s** | ~2x |

MTP draft acceptance was about the same on both builds (0.76 vs 0.74), so the MTP gain is kernel speed, not luck
with the drafts. Full numbers and method: [results/stock-vs-patched-2026-10-12.md](results/stock-vs-patched-2026-10-12.md).

### Quality

The new kernels use fp16 activations where llama.cpp uses int8 (q8_1) activations, so the output is not bit-exact
with stock. Wikitext-2 KL divergence of the token distributions against the stock build's own ub=1 output
(20 chunks of 512):

| | KLD | same top token |
|---|---:|---:|
| stock vs itself at a different batch size (the noise floor) | 0.0039 | 97.5% |
| single-token decode path | 0.0043 | 97.6% |
| multi-token (MTP verify) path | 0.0042 | 97.8% |
| prompt processing path | 0.0045 | 97.4% |
| stock prompt processing (int8 MMQ), for comparison | 0.0053 | 97.1% |

Perplexity is unchanged within its error bars. For prompt processing the patched build is closer to the reference
than stock is (fp16 instead of int8 activations).

## What changed

All of it is opt-in through environment variables; without them the build behaves like the baseline.

| Variable | Effect |
|---|---|
| `GGML_CUDA_P60_MMVQ=1` | sm_60 fp16 matvec and the fusions below (patches 1-16). Required by everything else. |
| `GGML_CUDA_P60_MEGA=1` | Persistent layer kernels, the MTP verify kernels and the prompt-processing kernels (patches 18-31). |
| `LLAMA_MTP_DRAFT_VOCAB=draftvocab/dv32k.bin` | The MTP draft head scores 32k frequent tokens instead of all 248k (patch 17). |

Each piece also has its own `GGML_CUDA_P60_*=0` switch to turn it off for A/B testing (`_ATTN`, `_FA`, `_FA_PF`,
`_Q8T`, `_MOE_GEMM`, `_GDN_CHUNK`, `_FUSE_*`, ...); `GGML_CUDA_P60_MEGA_DEBUG=1` logs why a layer did not match the
persistent kernels. Every patch's commit message has the measurement and the exactness note for that change.

**Decode on sm_60 (patches 1-16).** fp16 (HFMA2) matrix-vector kernels instead of the emulated int8 dot products;
RMS norm fused with the residual adds and with the activation prep for the next matvec; gated delta net (GDN)
fusions (q/k norm + scale, gated output norm, alpha bias, conv-state shuffle folded into the conv, state gather
skipped); bit-exact staged-load Q8_0 / Q6_K matvecs; MoE MMQ tile sizes picked by tokens per expert; a cuBLAS
algorithm choice that is faster on GP100. Several of these also help the dense Qwen3.5/3.6 models.

**MTP draft vocabulary (patch 17).** The draft head's output layer over all 248k tokens was ~11% of MTP decode
time. With `LLAMA_MTP_DRAFT_VOCAB` it scores a 32k list of frequent tokens plus the tokens already in the
sequence; the main model still verifies every token, so this only changes which drafts get proposed.
The idea is open-jet's (see above); shinbunbun also has a draft-vocabulary patch for their models. This one is
written for the 35B-A3B MTP graph, and its token list was built independently (see
[The draft vocabulary file](#the-draft-vocabulary-file)).

**Persistent layer kernels (patches 18-25), after open-jet's megakernel.** Decode on this model was ~1300 small kernel launches per token, so the
GPU mostly waited on launches and on short kernels that never reached full bandwidth. Each half of a layer (the
GDN or attention half, and the MoE half) now runs as one or a few cooperative kernels with all blocks resident and
software grid barriers between phases. There are versions for 1 token (plain decode and MTP drafting) and for
2-6 tokens (the MTP verify pass): expert selection, the union of experts across tokens, expert gate/up/down,
shared expert, GDN recurrence with the rollback snapshots llama.cpp keeps, attention with the KV cache writes,
all inside. Plus a multi-token output-layer matvec and a flash-attention kernel for the verify pass. The graph
matchers check every node, weight name and tensor layout and fall back to the normal path on anything
unexpected.

**Prompt processing (patches 26-31).** An fp16 grouped GEMM for the MoE experts (q4_K / q5_K / q6_K dequantized to
fp16 in shared memory, HFMA2 math, fp32 accumulation) instead of int8 MMQ, which sm_60 can only emulate: ~2.8x
faster on these shapes. A gated delta net kernel for long batches (2x), and a flash-attention kernel for head
size 256 with grouped-query attention (+26%).

## Scope and limits

- **sm_60 only.** The kernels are tuned for GP100 (no DP4A, full-rate fp16, 56 SMs, 48 KB shared memory per block)
  and are gated to it. Other GPUs run the baseline code.
- **Model-specific.** The persistent kernels match the Qwen3.5/3.6 MoE layer layout (2048 hidden, 256 experts / 8
  used, GDN with 16/32 heads of 128, attention with 16 query / 2 KV heads of 256) and the tensor types of the
  UD-Q4_K_XL quant (Q8_0 attention and GDN weights, Q4_K/Q5_K/Q6_K experts). Layers that do not match run the
  normal path, so other quants still work, just without the speedup.
- **Tested on one machine**: 2x P100 PCIe 16 GB, CUDA 12.9.1, driver 575, both cards holding the whole model.
  One card with CPU-offloaded experts has not been tested with the persistent kernels.
- **CUDA 12.x is required**; CUDA 13 dropped sm_60. Build with `-DCMAKE_CUDA_ARCHITECTURES=60` (the default
  architecture list makes a P100 run Maxwell PTX with fp16 disabled).

## Build and run

```console
$ scripts/build.sh                     # clones llama.cpp, applies the baseline and these patches, builds
$ scripts/run-server.sh ./llama.cpp-p100 /path/to/Qwen3.6-35B-A3B-UD-Q4_K_XL.gguf --port 8080
```

`run-server.sh` is the configuration the numbers above were measured with (64k context, `-b 6144 -ub 1536`,
MTP with 5 draft tokens and p-min 0.55).

## The draft vocabulary file

`draftvocab/dv32k.bin` is a flat list of int32 token ids for the Qwen3.5/3.6 tokenizer (248,320 tokens): the 32,768
most frequent tokens of five text sources, plus all 276 special/control tokens (33,040 ids in total). The sources
were weighted equally, whatever their size:

- English prose: the wikitext-2 training set;
- source code: llama.cpp's C++/CUDA/Python/Markdown sources plus some personal shell and Python scripts;
- documentation: the Markdown/text/HTML files under `/usr/share/doc` of an Arch Linux system;
- man pages: sections 1, 5 and 8 of the same system;
- Python: the standard library modules.

At run time the draft head also scores every token that has appeared in the current conversation (up to 4096 of
them), so names and words from your prompt can be drafted even if they are not in the list.

**What it does and does not change.** The main model still checks every drafted token, so the text you get is
exactly what you would get without the list; only the speed changes. On our 8-prompt test (chat, code, agent and
reasoning prompts in English) it was worth +6.3% with the production sampler. When the next token is outside the
list, the draft head cannot propose it and the speculation stops there, so **workloads far from English prose and
code - other languages (Chinese, Japanese, ...), heavy math notation, unusual domain vocabulary - will benefit
less, and could in principle draft worse than without the list.** If that is your use case, build a list from your
own text with `scripts/build-draft-vocab.sh` (one or more text files, each weighted equally), or leave
`LLAMA_MTP_DRAFT_VOCAB` unset to use the full vocabulary. We tried 16k, 48k and 64k lists as well; 32k was the best
on our test (16k: -3.5%, 48k: -0.9%, 64k: -1.8% relative to 32k).

## How this was made

The code was written by Claude (Anthropic's Claude Code, model Opus 5.5), with the owner of the machine directing
the work and setting the rules: every change was measured on the cards, kept only if it was faster, and checked
with the KL-divergence gate above before it was committed. The profiling used NVIDIA's CUPTI PC sampling (driver 575;
driver 580 removed profiling support for Pascal). Each commit message records the measurement behind it.

## License

MIT, like llama.cpp and the shinbunbun patches.
