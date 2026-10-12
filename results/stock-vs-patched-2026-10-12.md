# llama.cpp + shinbunbun P100 patches vs + these patches (2026-10-12)

2x Tesla P100 16 GB PCIe (layer split -ts 1.15/1; card 0 on PCIe 3.0 x4, card 1 on x16), Qwen3.6-35B-A3B UD-Q4_K_XL, fully resident, flash attention.
Stock = upstream llama.cpp c21284cdf + the shinbunbun patch baseline (see README), built with the same toolchain and options.
Ours = the baseline + the 31 patches of this repo, run with GGML_CUDA_P60_MMVQ=1 GGML_CUDA_P60_MEGA=1 (and the draft vocabulary for MTP).
All runs started with both cards <= 60 C.

| test                                              | stock  | ours   | ratio |
|---------------------------------------------------|--------|--------|-------|
| decode, no speculation (llama-bench tg128)        | 68.0   | 125.7  | 1.85x |
| prefill 512 tokens (pp512, ub 1536)               | 468.5  | 974.1  | 2.08x |
| prefill 4096 tokens (pp4096, ub 1536)             | 911.4  | 1590.4 | 1.75x |
| prefill 1536 tokens at 24k depth                  | 466.3  | 708.4  | 1.52x |
| MTP decode, prod sampler, 8 prompts x 256 (server)| 93.4   | 183.1  | 1.96x |
| 28k-token prompt: prefill (server, 2 runs)        | 707/718| 1094/1097 | 1.53x |
| 28k-token prompt: MTP decode after it             | 75.4/81.7 | 147.8/164.4 | ~1.98x |

MTP acceptance was similar on both (0.757 vs 0.740 on the 8-prompt set; long-context runs 0.70-0.77).
Quality (wikitext KLD vs stock ub=1 base, 20 chunks): ours decode path 0.0043, multi-token 0.0042, prefill 0.0045;
stock vs itself at another batch size 0.0039, stock prefill (MMQ) 0.0053.
Prefill: llama-bench with -b 4096 -ub 1536 (depth test -b/-ub 1536). MTP: llama-server, 8 chat/code/agent/reasoning prompts x 256 tokens, n-max 5 / p-min 0.55. 28k prompt: wikitext article + summarization request, 256 tokens, two seeds.
