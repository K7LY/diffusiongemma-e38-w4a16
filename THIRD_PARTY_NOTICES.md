# Third-party sources and credits

This project integrates externally obtained model data and execution code. Those artifacts are not vendored in this repository.

| Author / project | Contribution used locally | Source |
| --- | --- | --- |
| Google DeepMind / DiffusionGemma | Original architecture and upstream model | [Model card](https://huggingface.co/google/diffusiongemma-26B-A4B-it) |
| Goedel Machines | W4A16 base checkpoint, loader and kernels | [Pinned snapshot](https://huggingface.co/GoedelMachines/diffusiongemma-26B-A4B-w4a16/tree/0fede6e3e65d258d444822cfb6a79092cdc5c0af) |
| Goodoldjam | E38 projection-weight modifications | [Pinned snapshot](https://huggingface.co/Goodoldjam/DiffusionGemma-26B-E38-Abliterated-NVFP4/tree/3f007177bf67706bd335f8275a1bd2657cd4e686) |

The quantization and kernel work belong to Goedel Machines; the E38 ablation belongs to Goodoldjam. This project's contribution is the reproducible combination and its validation, rather than those underlying methods.

Optional inference and download dependencies include PyTorch, Transformers, Triton, Accelerate, NumPy, Safetensors and Hugging Face Hub. They are installed separately and retain their own licenses and notices.

See [licensing](docs/licensing.md) for the distinction between this project's Apache-2.0 code license and the upstream weight notices, including the unresolved difference in Goedel's weight notice.
