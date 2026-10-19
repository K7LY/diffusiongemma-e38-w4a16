# Licensing and distribution scope

This repository distributes project code under Apache-2.0, plus documentation and sanitized measurements. It does not distribute a base checkpoint, the E38 checkpoint, an overlay binary, or Goedel's loader and kernels. Users obtain the source snapshots separately and build the overlay locally.

## Upstream notices

The following notices were checked on October 19, 2026:

| Source | Published notice |
| --- | --- |
| [Google DiffusionGemma model card](https://huggingface.co/google/diffusiongemma-26B-A4B-it) and [official license page](https://ai.google.dev/gemma/docs/gemma_4_license) | Apache-2.0 |
| [Goodoldjam E38 NVFP4 model card](https://huggingface.co/Goodoldjam/DiffusionGemma-26B-E38-Abliterated-NVFP4) | Apache-2.0 metadata |
| [Goedel W4A16 model card, License section](https://huggingface.co/GoedelMachines/diffusiongemma-26B-A4B-w4a16#license) | Kernels identified as Apache-2.0; weights identified as subject to Gemma Terms and a separate `LICENSE-NOTICE.md` |

The Goedel weight notice differs from the original Google model's published license. This project records that inconsistency without resolving or overriding it. Its Apache-2.0 license applies to this project's code; it does not relicense downloaded weights or external code.

## Local builds and further distribution

Review the notices shipped with the exact pinned snapshots before using them. The builder's provenance manifest identifies its inputs, and the downloaded source notices should remain with those inputs.

A locally generated overlay contains source weight bytes. Its small size and its description as a difference or overlay do not make it ordinary project code. Redistributing it, or a merged checkpoint, requires a separate assessment of the relevant upstream rights and notices. Neither binary is part of this repository's distribution.

Retain [NOTICE](../NOTICE) and [third-party credits](../THIRD_PARTY_NOTICES.md) when distributing this project's source. If third-party implementation files are later vendored, include their actual license and required notices and identify modifications. The present recipe downloads its dependencies rather than vendoring them.
