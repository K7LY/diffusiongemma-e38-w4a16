# Overlay design

The artifact combines two fixed inputs: the Goedel W4A16 base at revision `0fede6e3e65d258d444822cfb6a79092cdc5c0af` and the Goodoldjam E38 NVFP4 source at revision `3f007177bf67706bd335f8275a1bd2657cd4e686`. The source's expert quantization is not used in the result.

## Tensor boundary

A byte-level audit of the installed source snapshots found exactly 20 differences among 987 shared non-expert tensors. Each difference was a BF16 matrix of matching shape. The overlay contains these decoder tensors for layers 7 through 16 inclusive:

| Suffix after `model.decoder.layers.{layer}.` | Shape |
| --- | --- |
| `mlp.down_proj.weight` | `[2816, 2112]` |
| `self_attn.o_proj.weight`, except layer 11 | `[2816, 4096]` |
| `self_attn.o_proj.weight`, layer 11 | `[2816, 8192]` |

Their raw payload totals **372,703,232 bytes**. The deterministic safetensors file, including its header, is **372,705,672 bytes** with SHA256 `7a913d9bdb580fe06c42efc0833d3eab9ec7d2d255144e9d7b102458e124fec7`. The builder requires that exact container hash as well as each raw tensor hash.

The builder copies source tensor bytes into one safetensors file and records tensor hashes, shapes, dtypes and source provenance in a manifest. It neither trains weights nor requantizes experts. The two-file output is portable alongside the required base snapshot; it does not contain a complete model.

## Loading

The runner loads the original base through its pinned fast loader, verifies the overlay, then copies the BF16 tensors into existing parameters under `torch.no_grad()`. It validates every destination before the first write. Decoder parameters share storage with corresponding encoder parameters; copying into those parameters preserves the aliases. Replacing parameter objects would risk leaving the encoder unchanged.

The W4A16 expert buffers and the base's fast execution path remain in place. No NVFP4 expert kernel is involved. The overlay therefore represents the E38 dense modifications applied to Goedel's expert quantization, rather than a byte-identical conversion of the complete E38 NVFP4 model.

## Trust and responsibilities

Build and verification treat checkpoints as data. Inference crosses a different boundary: it imports Python from the downloaded Goedel base. `--trust-base-code` makes this execution explicit. Source pins and hashes identify files; they do not establish that code is safe to execute. Review the downloaded implementation in the environment where it will run.

Model files, credentials, caches and generated outputs remain local. The repository contains the recipe and sanitized measurements. Service lifecycle, API authentication, tool execution and application permissions belong to the integrating application and are outside this package.

The verified import resolver remains installed after successful loading because the upstream expert implementation imports some kernels during generation. It resolves only the pinned source modules and suppresses bytecode writes for them. A failed load removes its newly installed resolver; switching checkpoint roots requires a fresh process.

## Generation and measurement

DiffusionGemma commits tokens from a denoised canvas. Output accounting must exclude uncommitted canvas padding and stop at the actual generation terminator. The original integration used the generation configuration's padding and EOS IDs, rather than assuming the tokenizer's padding ID matched them.

End-to-end output rate includes prompt processing and generation. It is distinct from a decode-only rate. The original API buffered delivery, so its delivered TTFT equaled the entire request duration; it did not expose the time of the first internally committed token. See [benchmark methodology](benchmarks.md).
