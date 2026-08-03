# DeepSeek V4 Flash on 4× RTX PRO 6000 (AWS g7e)

Serve [`deepseek-ai/DeepSeek-V4-Flash-DSpark`](https://huggingface.co/deepseek-ai/DeepSeek-V4-Flash-DSpark) with DSpark speculative decoding on a single **AWS `g7e.24xlarge`** (4× RTX PRO 6000 Blackwell, 96GB each).

This is the stack behind [camelAI](https://camelai.com)'s free-tier model. It is a port of the community DGX Spark recipes to x86 RTX, plus a small AWS spot deployment (sticky router, S3→NVMe weight staging, ASG workers).

> **Production path = AWS RTX4 + `fp8_ds_mla` KV.**  
> The DGX Spark / experimental `nvfp4_ds_mla` material further down is lineage and research, not what we run in prod.

## What you get

| | Production profile |
|---|---|
| Instance | `g7e.24xlarge`, TP=4 |
| Model | `DeepSeek-V4-Flash-DSpark` |
| Speculative decoding | DSpark, 5 draft tokens |
| KV cache | `fp8_ds_mla` |
| Context | `max_model_len=262144` (app working context ~220K) |
| Concurrency | `max_num_seqs=64`, `max_num_batched_tokens=8192` |
| Scheduling | `priority` |
| KV offload | 256GB host RAM + local NVMe spill |

Live worker boot (example):

```text
Available KV cache memory: 43.84 GiB
GPU KV cache size: 2,712,968 tokens
Maximum concurrency for 262,144 tokens per request: 10.35x
```

Rough throughput from our concurrency sweep (same image family):

| concurrent streams | aggregate tok/s | per-stream tok/s |
| ---: | ---: | ---: |
| 1 | ~300 | ~300 |
| 48 | ~3,300 | ~70 |
| 64 | ~3,300 | ~50 |

Details: [`benchmarks/results/`](benchmarks/results/).

## Quick start (single box)

Needs a g7e.24xlarge (or equivalent 4× RTX PRO 6000), Docker, and enough fast disk for the weights.

```bash
# 1. Build the x86 runtime (DGX Spark images are ARM64 and will not run here)
docker build -f recipe/rtx4/Dockerfile.nvfp4-port \
  -t vllm-dspark-runtime:rtx4-nvfp4-port-v3 .

# 2. Configure
cp .env.rtx4.example .env.rtx4
# edit HF_CACHE / paths / port as needed

# 3. Pull weights onto fast local disk
./prepare-dspark-model-cache-rtx4.sh

# 4. Run
./start-deepseek-v4-flash-dspark-rtx4.sh
./status-deepseek-v4-flash-dspark-rtx4.sh
./smoke-deepseek-v4-flash-dspark.sh
```

OpenAI-compatible API defaults to `http://<host>:8000/v1`.

Important env knobs (see `.env.rtx4.example`):

```bash
KV_CACHE_DTYPE=fp8_ds_mla
MAX_MODEL_LEN=262144
MAX_NUM_SEQS=64
MTP_NUM_TOKENS=5
KV_OFFLOAD_GB=256
KV_OFFLOAD_DISK_DIR=/opt/dlami/nvme/kv-offload
SCHEDULING_POLICY=priority
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
