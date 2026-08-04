# DeepSeek RTX4 Operations

This document describes the deployed worker and router contract. Do not put
account IDs, private addresses, API keys, or S3 release names in this file.

## Topology

- `g7e.24xlarge` workers serve DeepSeek V4 Flash DSpark with four RTX PRO 6000
  Blackwell GPUs, TP=4, FP8 MLA KV cache, and DSpark speculative decoding.
- Worker Auto Scaling Groups are maintained in Ohio (`us-east-2`), Oregon
  (`us-west-2`), and Virginia (`us-east-1`). Workers use Spot capacity by
  default. Any temporary on-demand base capacity must be explicitly removed
  after validation.
- One on-demand sticky router discovers the configured regional ASGs through
  AWS APIs. Its `AWS_ASG_TARGETS` value uses
  `region:auto-scaling-group,region:auto-scaling-group` entries.
- Network paths from the router to every worker are private. Cloudflare AI
  Gateway selects the self-hosted route before external provider fallbacks.

## Worker startup and releases

The active launch-template user data is `worker-user-data-s3-nvme.sh`.

1. It copies an immutable regional S3 release to
   `/opt/dlami/nvme/deepseek-model`.
2. It verifies the release manifest before starting the service.
3. It sets `HF_CACHE` and `MODEL_DIR` to the staged NVMe model path.
4. It starts `deepseek-rtx4.service`.

The S3 release includes weights plus an explicitly versioned runtime-cache
archive containing reusable vLLM, TileLang, TorchInductor, Triton, and
FlashInfer artifacts. `RUNTIME_CACHE_OBJECT` selects the archive in launch
template user data. Local NVMe is intentionally disposable after an eviction;
the regional S3 release is the durable source. The runtime AMI must already
include Docker, the runtime image, this repository, and the systemd service.

Use `promote-s3-nvme-launch-template.sh` for launch-template updates. It
embeds the S3/NVMe user data and removes the legacy model-cache EBS mapping.
Do not reintroduce snapshot-backed model volumes or Fast Snapshot Restore
without a separate measured migration proposal.

## Routing, cache affinity, and priority

The router only sends a worker traffic after `GET /v1/models` succeeds. An ASG
instance in `InService` is not necessarily ready while weights load.

Every worker runs `deepseek-spot-interruption-watcher.service`. It polls the
IMDSv2 Spot interruption and rebalance-recommendation endpoints every five
seconds. On a notice it installs an iptables rule that rejects only `NEW` TCP
connections to port 8000. Established streams remain connected for the rest of
the EC2 notice window, while the router's next health probe removes the worker
from its healthy set. The marker and reason are recorded in
`/run/deepseek-rtx4/draining` and in the systemd journal.

Rendezvous hashing keeps a stable `X-Session-ID`, `X-Conversation-ID`,
`X-Sticky-Key`, or `X-User-ID` on one healthy worker. A missing backend causes
only its assigned sessions to remap.

The trusted caller may set `X-Chiridion-VLLM-Priority`. The router clamps it to
`0..1000` and converts it to vLLM's JSON `priority` only for self-hosted
requests. Lower values run first; `0` is the paid-service convention and `100`
is the free-service convention. The external fallback request body remains
OpenAI-compatible.

## Runtime baseline

The service default is `MAX_NUM_SEQS=64`, `MAX_NUM_BATCHED_TOKENS=8192`,
`MAX_MODEL_LEN=262144`, and priority scheduling. Keep batch-size changes behind
a reproducible benchmark. A prior `12288` token prefetch canary regressed from
the 8192 baseline and was not adopted.

New S3/NVMe worker templates enable tiered KV offload by default: a 256 GiB
host-memory primary tier plus local-NVMe secondary tier. The isolated
`g7e.24xlarge` canary initialized DSpark, created both tiers, and passed an
OpenAI-compatible smoke request. This increases retained cache capacity for
long-context sessions, but a miss that reaches a lower tier has higher latency;
keep it out of short-latency benchmarks unless the workload actually spills.
Existing workers retain their current configuration until replacement.

## Recovery and verification

1. The interruption watcher rejects new worker connections; existing streams
   continue until they finish or EC2 reaches its termination deadline.
2. The router health loop removes the draining worker and the ASG requests a
   replacement using its configured Spot allocation policy.
3. User data stages and verifies the S3 artifact, installs the watcher, then
   starts vLLM.
4. The router discovery loop probes `/v1/models` and adds the replacement only
   when it is healthy.
5. Confirm `GET /healthz` and `GET /router/backends` on the router before
   moving traffic or declaring capacity recovered.

For a controlled validation, use a temporary on-demand instance only long
enough to verify staging, model loading, and a request through the router. Then
restore the ASG's Spot-only on-demand base capacity and desired count.
