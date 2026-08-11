# DeepSeek V4 Flash on 4× RTX PRO 6000 (AWS g7e)

Serve [`deepseek-ai/DeepSeek-V4-Flash-DSpark`](https://huggingface.co/deepseek-ai/DeepSeek-V4-Flash-DSpark) on a single **AWS `g7e.24xlarge`** (4× RTX PRO 6000 Blackwell, 96GB each).

This is the stack behind [camelAI](https://camelai.com)'s free-tier model. It is a port of the community DGX Spark recipes to x86 RTX, plus a small AWS spot deployment (sticky router, S3→NVMe weight staging, ASG workers).

> **Supported RTX path = stock vLLM TP4+EP with FP8 KV.**
> The DCP, DSpark speculative-decoding, DGX Spark, and custom KV-offload
> material is retained as lineage and research, not the worker-template
> default.

## What you get

| | Production profile |
|---|---|
| Instance | `g7e.24xlarge`, TP=4 + expert parallel |
| Model | `DeepSeek-V4-Flash-DSpark` |
| Runtime | vLLM 0.25.1 + FlashInfer Python 0.6.14 |
| Speculative decoding | Off |
| KV cache | FP8, block size 256 |
| Context | 262,144-token per-request cap |
| MoE | Marlin MXFP4 |
| KV offload | Off |

Live worker boot (example):

```text
Available KV cache memory: 43.02 GiB
GPU KV cache size: 2,917,131 tokens
Maximum concurrency for 262,144 tokens per request: 11.13x
```

Rough throughput from our concurrency sweep (same image family):

| concurrent streams | aggregate tok/s | per-stream tok/s |
| ---: | ---: | ---: |
| 1 | ~111 | ~111 |
| 16 | ~913 | ~57 |
| 64 | ~2,072 | ~32 |
| 256 | ~3,804 | ~15 |
| 512 | ~4,734 | ~9 |

Details: [`benchmarks/results/`](benchmarks/results/).

## Quick start (single box)

Needs a g7e.24xlarge (or equivalent 4× RTX PRO 6000), Docker, and enough fast disk for the weights.

```bash
# 1. Build the pinned stock SM120 runtime
./build-stock-sm120-vllm-runtime.sh

# 2. Configure
cp .env.rtx4.stock-sm120.example .env.rtx4.stock-sm120
# edit HF_CACHE / paths / port as needed

# 3. Pull weights onto fast local disk
./prepare-dspark-model-cache-rtx4.sh

# 4. Run
./start-deepseek-v4-flash-stock-sm120-rtx4.sh
./status-deepseek-v4-flash-dspark-rtx4.sh
./smoke-deepseek-v4-flash-dspark.sh
```

The build pins the exact vLLM 0.25.1 source commit and compiles the upstream
SM120 Marlin MoE `c_tmp`/shared-memory correction into the CUDA extension.
The patch is intentionally source-built: layering Python files onto the
official runtime would leave its vulnerable precompiled `_C` extension in
place.

OpenAI-compatible API defaults to `http://<host>:8000/v1`.

Important env knobs (see `.env.rtx4.stock-sm120.example`):

```bash
KV_CACHE_DTYPE=fp8
DCP_SIZE=1
DSPARK_NUM_TOKENS=0
MAX_MODEL_LEN=262144
GPU_MEMORY_UTILIZATION=0.90
```

## AWS spot deployment

For multi-node free-tier style serving, see [`infra/aws/`](infra/aws/):

- **Workers** — spot ASG, one g7e.24xlarge each, systemd → Docker  
- **Weights** — immutable regional S3 release staged to local NVMe on boot (not baked into the AMI)  
- **Router** — small on-demand node, rendezvous-hash sticky routing so a session keeps hitting the worker that holds its KV  
- **Priority** — trusted callers send `X-Chiridion-VLLM-Priority`; the router injects vLLM `priority` only toward self-hosted backends  

```text
Client → Cloudflare AI Gateway → sticky router → spot GPU workers
                                      ↘ fallback (e.g. Azure hosted DeepSeek)
```

Ops runbook: [`infra/aws/OPERATIONS.md`](infra/aws/OPERATIONS.md)  
Router: [`infra/router/sticky_openai_router.py`](infra/router/sticky_openai_router.py)

## Layout

```text
.env.rtx4.example          # production-shaped single-box config
start-*-rtx4.sh            # RTX4 lifecycle scripts
recipe/rtx4/               # x86 runtime Dockerfile
recipe/overlay/            # vLLM/DSpark patches (incl. concurrency)
patches/                   # Keys concurrency patch, etc.
infra/aws/                 # spot ASG + S3/NVMe worker + router install
infra/router/              # sticky OpenAI-compatible router
benchmarks/                # concurrent bench + recorded results
docs/                      # setup notes, patch notes
```

## Lineage

This repo is **not** a GitHub fork button clone; it combines and ports several public efforts:

- [MiaAI-Lab DGX Spark recipe](https://github.com/MiaAI-Lab/DeepSeek-v4-Flash-DSpark-2x-DGX-Spark) — original packaging lineage  
- [Keys / drowzeys DSpark concurrency patch](https://github.com/drowzeys/Keys-Concurrency-Patch-for-DSpark-DeepSeek-V4-Flash) — required for `max_num_seqs > 1`  
- [Rafael Caricio DSpark vLLM work](https://github.com/rafaelcaricio/vllm/pull/1)  
- [Fraser Price DSpark runtime/model work](https://huggingface.co/fraserprice/DeepSeek-V4-Flash-DSpark)  
- DeepSeek V4 Flash + DSpark, vLLM, FlashInfer, NVIDIA Blackwell stack  

Full attribution: [`CREDITS.md`](CREDITS.md).

The runtime is a **custom vLLM image**, not stock upstream. It boots, smokes, and serves production traffic for us; still treat image tags and env files in this repo as the contract.

## DGX Spark path (optional / upstream)

Two-node DGX Spark scripts and the experimental `nvfp4_ds_mla` KV profile remain in-tree for people on that hardware (`./start-deepseek-v4-flash-dspark.sh`, `.env.dspark.example`, `recipe/nvfp4/`). That path is a different machine and a different default KV dtype than AWS production.

If you only care about g7e / RTX PRO 6000 ×4, ignore the Spark scripts.

## License

- Repo scripts/docs: MIT ([`LICENSE`](LICENSE))  
- vLLM overlay / Keys patch: Apache-2.0 lineage  
- Model weights and base images: their upstream terms  
