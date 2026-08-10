# RTX4 DCP serving sweep

Use `rtx4_dcp_serving_sweep.py` to compare an already-running server. The
benchmark never starts, stops, or changes a server.

## Measurement contract

- Run on an isolated worker with no production traffic.
- Run the client on the GPU host if GPU telemetry is required; otherwise the
  harness still records endpoint metrics but cannot invoke that host's
  `nvidia-smi`.
- Keep the model revision, weights, KV dtype, memory utilization, block size,
  prompt, output length, and concurrency list fixed across profiles.
- Run one ignored warmup and three measured trials at every concurrency. Report
  trial medians, not the best result.
- Require final `usage.completion_tokens`. DSpark may return several tokens in
  one SSE event, so event counts are not token counts.
- `TTFT` is request start to the first nonempty content or reasoning delta.
- Prompt TPS-to-first is `prompt_tokens / TTFT`; it includes queueing and the
  first-token decode. Use `--output-tokens 1` in a separate prefill run so no
  ongoing decode tail contaminates later requests' TTFT.
- Per-stream decode TPS is `(completion_tokens - 1) / (last_content -
  first_content)`. Its reciprocal is reported as effective mean ITL.
- Aggregate decode TPS is the sum of those post-first-token output tokens over
  the global first-to-last-content window. End-to-end output TPS, including
  prefill, is reported separately.
- SSE event-gap percentiles are diagnostic only; with speculative decoding they
  are not per-token ITL percentiles.
- Client usage totals are the primary throughput source. The vLLM generation
  counter delta is retained as a cross-check, along with cache/request gauges,
  cache configuration labels, and per-GPU utilization, memory, power, clocks,
  and temperature.
- A deterministic concurrent correctness wave runs before performance trials.
  Any mismatch or request error invalidates the profile.

## Isolation matrix

First quantify DCP itself with otherwise identical settings:

| Profile | DCP | DSpark | Graphs | Backend |
|---|---:|---:|---|---|
| `dcp1-eager-control` | 1 | 0 | eager | B12X |
| `dcp2-eager-control` | 2 | 0 | eager | B12X |
| `dcp4-eager-control` | 4 | 0 | eager | B12X |

Only after all three pass correctness, enable one optimization at a time:

1. Piecewise decode CUDA graphs.
2. Async scheduling.
3. Prefix caching and backend autotuning.
4. Interleave and max-batched-token sweeps.

Keep the established DCP1 production-shaped result as a ceiling/reference:
lucifer-cutlass, DSpark `k=5`, `FULL_AND_PIECEWISE`, async scheduling, and
autotuning. Do not attribute its DSpark gain to DCP.

## Commands

Run the same command after each server profile is ready, changing only the
profile metadata and output filename:

```bash
python benchmarks/rtx4_dcp_serving_sweep.py \
  --dcp 1 \
  --profile-label dcp1-eager-control \
  --backend b12x \
  --dspark-tokens 0 \
  --cudagraph-mode eager \
  --async-scheduling off \
  --concurrency 1,2,4,8,16,32,48,64 \
  --trials 3 \
  --warmups 1 \
  --output-tokens 512 \
  --correctness-waves 2 \
  --output benchmarks/results/raw/dcp1-eager-control.json
```

Repeat for DCP2 and DCP4. For a production-shaped control, pass its actual
metadata, for example `--backend lucifer-cutlass --dspark-tokens 5
--cudagraph-mode FULL_AND_PIECEWISE --async-scheduling on`.

For long-context behavior, use a separate run with `--prompt-file` and do not
mix its results into this short-prefill decode sweep. The harness records the
file's SHA-256 and the actual prompt-token count returned by the server.

For a dedicated prefill/TTFT probe, set `--output-tokens 1`. Decode throughput
fields will be empty by design; use `aggregate_prompt_tps_to_first` and the TTFT
distribution. Decode runs should retain at least 512 output tokens and use only
`aggregate_output_tps_after_first` for the headline tokens/sec.

## Stock vLLM 0.25.1 TP4+EP control

Hermia's stock-SM120 recipe is a useful independent control: TP4, expert
parallel enabled, FP8 KV, Marlin MXFP4 MoE, no speculative decoding, and a 256K
model-length cap. Its documented single-stream throughput uses final
`usage.completion_tokens`, which matches this harness. Its published prefill
probe divides prompt tokens by a non-streaming request's complete wall time,
including up to 40 generated tokens; the dedicated one-token streaming mode
above isolates TTFT more cleanly.

Use this progression for the stock control:

| Stage | Concurrency | Output | Trials | Purpose |
|---|---|---:|---:|---|
| Fast canary | `1,16,64` | 128 | 1 | Liveness, EP rank coherence, gross scaling |
| Decode sweep | `1,2,4,8,16,32,48,64,96,128,192,256` | 512 | 3 | Decode-only curve and saturation |
| 8K prefill | `1,2,4,8,16` | 1 | 3 | TTFT and concurrent prefill |
| 64K prefill | `1,2,4,8` | 1 | 3 | Long-prefill scaling |
| 128K prefill | `1,2,4` | 1 | 3 | Long-context TTFT |
| 238K prefill | `1` | 1 | 3 | Near Hermia's validated stock ceiling |

Do not send a stock-path canary beyond 256K: Hermia reports a destructive GPU
launch failure above that single-request prefill length. Build the prefill files
below the target so the chat template and nonce still fit under the ceiling.
The `192` and `256` decode levels assume `--max-num-seqs` is at least 256; cap
the matrix at the server's configured value and record that value otherwise.

The decode command should identify the profile explicitly:

```bash
python benchmarks/rtx4_dcp_serving_sweep.py \
  --base-url http://127.0.0.1:8000 \
  --model deepseek-v4-flash \
  --dcp 1 \
  --profile-label stock-vllm-0.25.1-tp4-ep \
  --backend stock-flashinfer-marlin \
  --tensor-parallel-size 4 \
  --expert-parallel on \
  --dspark-tokens 0 \
  --kv-dtype fp8 \
  --gpu-memory-utilization 0.90 \
  --concurrency 1,2,4,8,16,32,48,64,96,128,192,256 \
  --output-tokens 512 \
  --warmups 1 \
  --trials 3 \
  --correctness-waves 2 \
  --output benchmarks/results/raw/stock-vllm-0.25.1-tp4-ep-decode.json
```
