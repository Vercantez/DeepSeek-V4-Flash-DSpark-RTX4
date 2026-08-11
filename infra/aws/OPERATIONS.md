# DeepSeek RTX4 Operations

This document describes the deployed worker and router contract. Do not put
account IDs, private addresses, API keys, or S3 release names in this file.

## Topology

- `g7e.24xlarge` workers serve DeepSeek V4 Flash with four RTX PRO 6000
  Blackwell GPUs as one TP=4 replica with expert parallelism, FP8 KV cache,
  and Marlin MXFP4 experts. DCP and speculative decoding are disabled.
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

The S3 release contains the weights. An optional `RUNTIME_CACHE_OBJECT` must
come from the same pinned stock runtime; caches from the former custom
DCP/DSpark runtime are incompatible. Local NVMe is intentionally disposable
after an eviction. The runtime AMI should include Docker, the pinned stock
image, this repository, and the systemd service; user data can build the image
as a fallback.

The launcher bind-mounts the versioned stock cache directory under `$HF_CACHE`
at `/root/.cache` so
DeepGEMM, FlashInfer, and related compiled/autotune artifacts survive a
container replacement. A promoted `RUNTIME_CACHE_OBJECT` is extracted into
that directory before the service starts. Keep it separate from the former
custom runtime's unversioned `vllm-cache` directory.

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

The supported baseline is vLLM 0.25.1, FlashInfer Python 0.6.14 with cubins
0.6.13, TP=4, expert parallelism, FP8 KV, block size 256, max model length
262,144, and GPU memory utilization 0.90. The isolated reference host allocated
2,917,131 KV tokens (11.13 full 256K contexts) and passed exact correctness at
1, 16, and 64 concurrent streams. The stock sparse-prefill backend can fail
destructively above 256K for one request, so do not raise the per-request cap
based only on aggregate KV capacity.

The stock profile adds vLLM's native tiered KV offloader to the 2,917,131-token
GPU pool. It reserves 256 GiB of host memory across the four TP workers and
cascades content-addressed blocks to `/opt/dlami/nvme/kv-offload`. That path is
a 4 TiB ext4 loop filesystem backed by a sparse image on local instance-store
NVMe, then bind-mounted into the container at the same absolute path. The hard
4 TiB boundary prevents vLLM's otherwise unbounded filesystem cache from
consuming the entire model filesystem. Before Docker can start, the launcher
requires the directory to resolve to that dedicated mount and requires at least
1 TiB free; a missing mount therefore fails closed instead of writing into the
container or root EBS disk.
`PYTHONHASHSEED=0` keeps block names stable across process restarts. Stale
process-local host-memory mmap files are removed before startup, while the
filesystem cache remains reusable.

KV offload is a lower-tier cache, not permission to raise the 262,144-token
single-request cap. The sparse indexer still needs GPU scratch memory, and a
lower-tier cache hit trades recomputation for PCIe/NVMe latency. Compare
decode-only throughput separately from TTFT when evaluating it.

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
