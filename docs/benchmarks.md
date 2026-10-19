# DGX Spark measurements

These results were recorded on October 19, 2026, on one NVIDIA DGX Spark with a GB10 GPU and 128 GB unified memory. They describe the original integration of this overlay recipe. They are historical measurements, not GPU validation of every later standalone CLI revision.

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

The input snapshots are pinned in the [README](../README.md). The measurement suite used the original application API, with buffered output delivery. No model weights or private application records are included in this repository.

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

The portable builder was separately executed on the same Spark using the existing pinned input files. It rehashed all 50 base and 14 E38 source files and compared all 987 dense tensors on CPU. The resulting 372,705,672-byte overlay exactly matched the adopted artifact SHA256. No Torch or Transformers module was imported; the production model operation and GPU process list were unchanged. See [the CPU validation record](../benchmarks/portable-cpu-20261019.json). Standalone CLI GPU generation remains unverified.
