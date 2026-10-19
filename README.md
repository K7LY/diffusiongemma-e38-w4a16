# DiffusionGemma E38 W4A16

Build a small E38 weight overlay for the fast Goedel W4A16 DiffusionGemma checkpoint, then run it locally on a single NVIDIA DGX Spark.

The recipe retains the base checkpoint's quantized experts and fast execution path. It copies 20 BF16 projection tensors from the pinned E38 checkpoint into their existing tied parameters. The locally built overlay adds about **373 MB**; this repository distributes code, documentation and measurements, without model weights or third-party kernels.

On one GB10 / 128 GB Spark, the original integration produced Japanese prose at **63–84 output tokens/s**, including input processing, with **18.51 GB idle / 21.91 GB peak Torch CUDA allocation**. A separate warm, batch-1 English reasoning measurement at the native 48-step limit reached **209.14 tokens/s median / 229.60 peak**, excluding initial prefill. Its 1,024-token output budget cut off the reasoning before completion. These are small workload samples, not a general benchmark. See [measurements and limitations](docs/benchmarks.md).

## Sources

| Component | Pinned revision |
| --- | --- |
| [Goedel W4A16 base](https://huggingface.co/GoedelMachines/diffusiongemma-26B-A4B-w4a16) | `0fede6e3e65d258d444822cfb6a79092cdc5c0af` |
| [Goodoldjam E38 source](https://huggingface.co/Goodoldjam/DiffusionGemma-26B-E38-Abliterated-NVFP4) | `3f007177bf67706bd335f8275a1bd2657cd4e686` |
| [Original DiffusionGemma](https://huggingface.co/google/diffusiongemma-26B-A4B-it) | Google DeepMind upstream architecture and model |

E38 supplies the ablation; Goedel supplies the quantization and fast kernels. This project supplies the overlay recipe, validation and local runner. It does not train a new base model. The resulting expert weights differ from the E38 publisher's NVFP4 checkpoint, so that checkpoint's reported refusal scores do not apply to this combination.

## Setup

Use Python 3.12 or newer on Linux with a CUDA-capable NVIDIA GPU. The measured machine was an ARM64 DGX Spark with a GB10 GPU and 128 GB unified memory. Other hardware has not been validated here. Allow at least **40 GB free disk space** for both source snapshots and the overlay, plus Python packages, download cache and compiler cache. Hugging Face's cache can increase total usage if files are copied.

Create a Python environment and install the recipe:

```sh
python3 -m venv .venv
. .venv/bin/activate
python -m pip install -e '.[download]'
```

The build and verification code uses the Python standard library. Downloading adds Hugging Face dependencies. Inference additionally requires a working CUDA PyTorch installation and the optional dependencies:

```sh
python -m pip install -e '.[inference]'
```

Install the appropriate CUDA PyTorch build for your platform before the inference extra. The original integration used PyTorch `2.11.0+cu130`, Transformers `5.12.1`, Triton `3.6.0`, Accelerate `1.15.0`, Hugging Face Hub `1.33.0` and Safetensors `0.8.0`. Installing the extra alone does not establish a tested GPU environment. The first inference may compile kernels and take longer than subsequent calls.

## Download, build and verify

Review the [upstream licensing notices](docs/licensing.md) before obtaining the checkpoints. Authenticate with your normal Hugging Face CLI if access requires it; do not put tokens in commands committed to this repository.

```sh
e38-w4a16 download --base-dir models/base --source-dir models/source

e38-w4a16 build \
  --base-dir models/base \
  --source-dir models/source \
  --output models/e38-overlay

e38-w4a16 verify --overlay-dir models/e38-overlay
```

Downloads use the revisions listed above. The builder checks the checkpoint inputs and writes `e38-overlay.safetensors` with a provenance `manifest.json`. Keep the original base snapshot available: the overlay is not a standalone checkpoint. Build and verification do not execute model repository code or use the GPU.

## Generate

Inference explicitly executes the pinned base checkpoint's Python loader and kernels. Review that downloaded code before passing `--trust-base-code`.

```sh
e38-w4a16 generate \
  --base-dir models/base \
  --overlay-dir models/e38-overlay \
  --trust-base-code \
  --prompt '日本語で、拡散言語モデルの仕組みを簡潔に説明してください。' \
  --max-new-tokens 1024
```

Add `--thinking` to enable the model's native reasoning channel. For a recorded local benchmark:

```sh
e38-w4a16 benchmark \
  --base-dir models/base \
  --overlay-dir models/e38-overlay \
  --trust-base-code \
  --output results.json
```

These commands run text inference as an explicitly invoked process. This repository does not install a resident agent, system service or API server.

For single-stream decode timing with initial prompt processing excluded:

```sh
e38-w4a16 benchmark-decode \
  --base-dir models/base \
  --overlay-dir models/e38-overlay \
  --trust-base-code \
  --output results-decode.json
```

This bounded sweep uses three English workloads, a 1,024-token output budget, and 48/16/8 denoising-step limits. It repeats the fastest standard and speed-oriented workload/settings pairs with different seeds. The measured 8-step code samples contained malformed code and extensive repetition despite higher raw throughput. See [the timing boundary and results](docs/benchmarks.md#decode-only-method).

## Validation scope

The original private integration passed Japanese conversation, retrieval from an 8,527-token input, native function calling, tool-result continuation and three reasoning requests. A separate application integration also completed conversation and a real terminal tool invocation. Those application components are not included here; the historical results do not substitute for testing this standalone CLI.

This repository's standalone loader and generation wrapper were separately exercised on the same GB10: a short Japanese reply stopped normally, followed by 13 decode measurements. The runner's CPU tests also cover deferred kernel imports and the decode timer boundary. The decode sweep measures throughput and does not establish task accuracy.

The model can repeat words or emit redundant tool delimiters. No stray `own` artifact appeared in the nine recorded direct requests, but that is not a guarantee. Refusal behavior has not been independently benchmarked, and the 262,144-token configured context has not been tested to its limit. Image and video inputs are outside this runner's validation scope.

See [design](docs/design.md), [benchmarks](docs/benchmarks.md), [licensing](docs/licensing.md) and [third-party credits](THIRD_PARTY_NOTICES.md).

## License

Project code is Apache-2.0; see [LICENSE](LICENSE). That license does not grant rights to external model weights or third-party code. The upstream checkpoint notices are not fully consistent, as documented in [licensing](docs/licensing.md). No weight redistribution is included in this repository.
