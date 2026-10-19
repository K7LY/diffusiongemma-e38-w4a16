# DGX Spark measurements

These results were recorded on October 19, 2026, on one NVIDIA DGX Spark with a GB10 GPU and 128 GB unified memory. The application measurements and the standalone decode measurements use different timing boundaries, described below.

## Environment

| Component | Recorded version |
| --- | --- |
| Platform | Linux ARM64, GB10 / SM121 |
| PyTorch | `2.11.0+cu130` |
| Transformers | `5.12.1` |
| Triton | `3.6.0` |
| Accelerate | `1.15.0` |
| Hugging Face Hub | `1.33.0` |
| Safetensors | `0.8.0` |
| Base loader mode | Goedel W4A16 `turbo` |
| Workload | One GPU model, sequential requests |

The input snapshots are pinned in the [README](../README.md). The first measurement suite used the original application API, with buffered output delivery. The later decode sweep used this repository's loader and benchmark module directly. No model weights or private application records are included in this repository.

## Observed results

| Workload or metric | Result |
| --- | --- |
| First start through readiness and warmup | 14.32 s |
| Short Japanese reply | 9 output tokens, 0.382 s |
| Japanese explanation | 295 output tokens, 4.668 s, **63.20 tokens/s** |
| Repeated Japanese explanation | 283 output tokens, 3.378 s, **83.78 tokens/s** |
| Retrieval from 8,527 input tokens | Correct marker, 4.064 s |
| Native function call | Correct function and empty arguments, 0.430 s |
| Tool-result continuation with thinking | Correct marker, 1.387 s |
| Reasoning arithmetic, three requests | Correct final answer in all three; separate reasoning channel |
| Reasoning arithmetic output rate | 128.67–148.75 tokens/s |
| Idle Torch CUDA allocation | **18.51 GB / 17.24 GiB** |
| Peak Torch CUDA allocation during suite | **21.91 GB / 20.40 GiB** |
| Process peak RSS | 9.10 GB / 8.48 GiB |

All nine direct requests completed. A separate application-level check passed Japanese conversation, reasoning with a final answer, and an actual terminal tool invocation. That application and its tool runner are outside this repository.

## Interpretation

Output rate is committed output tokens divided by request wall time, including input processing and generation. It is not a decode-only measurement. Buffered API delivery made delivered TTFT equal to request duration; internal first-token latency was not measured. Do not compare that TTFT with a streaming server's first-token timing.

The prose and arithmetic results use different prompts and output structures. Their rates should not be averaged into a single advertised model speed. These are individual observations without a latency distribution or controlled baseline sweep. Model behavior, prompt length, stopping, kernel caches and denoising trajectory can materially change the result.

CUDA allocation is PyTorch's recorded allocation, not total device residency or system memory use. GB uses decimal bytes and GiB uses binary bytes. On the Spark's unified-memory architecture, CUDA allocation and process RSS must not be added as independent physical-memory totals.

## Quality and coverage limits

The inspected outputs contained no stray `own` artifact. Two Japanese outputs nevertheless repeated text: “一時的に的にする” and “次回のためにに備えて”. A native function-call response included a redundant `<tool_call|>` in its content field, although the function name, arguments and tool continuation were correct. Outputs were not edited to hide these defects.

The configured context length was 262,144 tokens, but the longest tested input was 8,527 tokens. Maximum-context memory, concurrency, image/video inputs, broad language quality and refusal rates were not evaluated. The E38 publisher's evaluation used a different complete checkpoint and cannot establish this artifact's refusal rate.

## Reproduce a local measurement

After following the setup instructions:

```sh
e38-w4a16 benchmark \
  --base-dir models/base \
  --overlay-dir models/e38-overlay \
  --trust-base-code \
  --output results.json
```

Record the command, dependency versions, hardware, output, token count, wall time and cache state with any result you publish. This standalone command and the historical application API have different surrounding overhead; compare measurements with their boundaries stated.

The sanitized per-request outputs, token counts and allocator measurements are in [the machine-readable record](../benchmarks/gb10-20261019.json). Generated text is preserved verbatim; request identifiers and application records are omitted.

## Standalone builder validation

The portable builder was separately executed on the same Spark using the existing pinned input files. It rehashed all 50 base and 14 E38 source files and compared all 987 dense tensors on CPU. The resulting 372,705,672-byte overlay exactly matched the adopted artifact SHA256. No Torch or Transformers module was imported during the build; the production model operation and GPU process list were unchanged. See [the CPU validation record](../benchmarks/portable-cpu-20261019.json).

The standalone loader and generation wrapper were then executed on the GPU. A Japanese smoke request returned `動作確認できました` in 8 output tokens and stopped normally. The same loaded model completed the decode sweep below. This validates those execution paths on GB10; it does not extend the standalone runner's coverage to the application's API or tool runner.

## Standalone CLI validation

A wheel built from commit `4cab80d85f2f5961a4f7d959fdd9eb770c33c305` was installed into a fresh package target on the same GB10. The test reused the pinned CUDA dependencies and existing source snapshots. It ran offline, with the production application's modules absent from the import path. It did not repeat a network download or a fresh dependency installation.

The CLI rebuilt the exact overlay in 22.91 seconds and verified it successfully. Inference without explicit source-code trust was rejected before GPU allocation. A separate `generate` process returned the requested Japanese sentence and stopped normally. The `benchmark` command then passed all nine requests: Japanese short output, Japanese prose, retrieval from 8,527 input tokens, a native function call, continuation with a fixture tool result, three reasoning answers with correct final values, and a second Japanese prose request. The GPU was released when each CLI process exited.

All functional requests finished normally or with the expected tool-call stop. The first prose sample omitted a Japanese particle and used an awkward verb form; passing the functional checks does not imply flawless language quality or broad task accuracy. The original application service and selected model were restored, with conversation history preserved. Full responses and checks are in [the standalone record](../benchmarks/standalone-gb10-20261019.json).

## Decode-only method

`benchmark-decode` measures one generation stream at batch size 1 with the Goedel `turbo` path. The initial prompt encoder is synchronized once and the timer starts immediately afterward. Timing ends after `generate()` returns and the GPU completes its queued work. Later encoder passes that commit generated canvases remain inside the measured decode interval.

Tokenization, initial prompt setup/prefill, streamer delivery, final CPU token transfer and detokenization are outside this interval. No streamer or per-forward synchronization is used. The numerator counts actual output through the first generation EOS token, inclusive, and excludes subsequent canvas padding. The 1,024-token budget is a multiple of the fixed 256-token canvas.

The sweep compares English code, mathematical reasoning and technical prose at 48, 16 and 8 maximum denoising steps. Other native sampler settings are unchanged and recorded in the result. One 256-token warmup is run for each setting. The fastest standard and overall workload/settings pairs are repeated with two more seeds; peak, median and aggregate output/decode-time rates are reported separately. This is a bounded search, not proof of a hardware-wide maximum or equivalent quality at reduced step limits.

## Decode-only results

| Repeated workload | Maximum denoising steps | Samples | Median tokens/s | Peak tokens/s | Aggregate tokens/s |
| --- | --- | --- | --- | --- | --- |
| Mathematical reasoning, thinking enabled | 48 (native default) | 3 | **209.14** | **229.60** | 212.77 |
| Python code | 8 | 3 | 245.69 | 246.26 | 240.08 |

The native-default peak generated 1,024 tokens in 4.460 seconds of decode time. Across the three initial native-default workloads, rates ranged from 124.10 to 201.50 tokens/s before repeating the fastest pair. The result depends strongly on the prompt and denoising trajectory.

The 8-step code output visibly deteriorated: inspected samples contained malformed Python and hundreds of repeated newline characters. Its higher raw token rate must not be presented as useful code-generation performance. No accuracy benchmark was performed at any step setting.

All 13 measured samples reached the 1,024-token budget without EOS. In particular, the native-default mathematical output was an incomplete reasoning trace, not a completed proof. These results measure output throughput within that budget, not successful task completion. Full prompts, generated text, per-sample timings, sampler settings and output-inspection notes are preserved in [the decode record](../benchmarks/gb10-decode-20261019.json).
