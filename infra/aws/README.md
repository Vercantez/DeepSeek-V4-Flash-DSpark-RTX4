# AWS RTX4 Spot Service

This directory contains the deployment shape for serving DeepSeek V4 Flash on
`g7e.24xlarge` RTX PRO 6000 Blackwell hosts.

## Shape

- GPU workers run in an Auto Scaling Group with Spot capacity.
- An IMDSv2 watcher drains workers on interruption or rebalance notices by
  rejecting new connections while preserving established streams.
- Each GPU worker starts `deepseek-rtx4.service`, which runs the validated
  stock-SM120 profile with vLLM 0.25.1 and FlashInfer Python 0.6.14:
  - one TP=4 replica with expert parallelism enabled
  - FP8 KV cache with 256-token blocks
  - Marlin MXFP4 MoE kernels
  - `MAX_MODEL_LEN=262144`
  - DCP disabled and speculative decoding disabled
- A small on-demand router node runs `deepseek-sticky-router.service`.
- The router discovers healthy ASG workers through AWS APIs and forwards
  vLLM's native OpenAI and Anthropic traffic to
  `http://<worker-private-ip>:8000`.

The deployed vLLM server exposes these agent-facing endpoints without a
protocol translation layer:

- `POST /v1/chat/completions`
- `POST /v1/responses`
- `POST /v1/messages`
- `POST /v1/messages/count_tokens`

For multi-region workers, set `AWS_ASG_TARGETS` on the router. It accepts a
comma-separated list of `region:auto-scaling-group` values, for example:

```text
AWS_ASG_TARGETS=us-east-2:deepseek-rtx4-spot-asg,us-west-2:deepseek-rtx4-spot-asg-oregon
```

The regional VPCs must have private routing between the router and workers.
Rendezvous hashing is calculated across the combined healthy backend list, so
a stable sticky key remains on the same regional worker until that worker is
unhealthy or replaced.

## Sticky Routing

The router picks a backend with rendezvous hashing. Send one of these headers
to keep a user/session/conversation on the same GPU host:

- `X-Sticky-Key`
- `X-Session-ID`
- `X-Conversation-ID`
- `X-User-ID`

If no sticky header exists, the router uses the protocol's stable identifiers
(`user`, Anthropic `metadata.user_id`, or Responses `prompt_cache_key`) before
hashing the first user message and finally the `Authorization` header. It skips
shared system/developer prompts so coding-agent traffic does not collapse onto
one GPU. For stored Responses requests, the router gives the first request a
vLLM `request_id`; a later `previous_response_id` hashes to the same worker
without keeping router-local affinity state. Explicit sticky headers remain the
most reliable option for custom clients.

Health endpoints:

- `GET /healthz`
- `GET /router/backends`
- `GET /router/capabilities`

### vLLM request priority

AI Gateway must send vLLM priority as the `X-Chiridion-VLLM-Priority` request
header, not as a JSON body field. The sticky router consumes the header,
clamps it to `0..1000`, and adds vLLM's `priority` field to Chat Completions and
Responses requests only after selecting the self-hosted route. Anthropic
Messages does not accept that vLLM extension. This keeps Azure and OpenRouter
fallbacks free of a vLLM-only request parameter. Lower values run first; use
`0` for paid traffic and `100` for free traffic.

## Model Contract

The router maps the shared default/output guardrails to each native schema:
`max_tokens` for Chat Completions and Anthropic Messages, and
`max_output_tokens` for Responses. It does not rewrite a request based on
`/tokenize`: vLLM's tokenizer
endpoint and its final chat-generation validation can account for templates
differently. Instead, the deployed model contract is explicit:

- hard context limit: `262144` tokens
- application working context: `262144` tokens
- maximum output: `262144` tokens for an empty prompt

The application may compact before the model's native limit. By default, the
router preserves the caller's `max_tokens`; vLLM enforces the native context
limit. `MAX_REQUEST_OUTPUT_TOKENS` is an optional operational guardrail, not a
default policy. This keeps request routing deterministic and makes vLLM the
authority for hard context validation.

`TTFT_TIMEOUT` is enforced by the sticky router while waiting for vLLM response
headers and, for streaming requests, through the first SSE body event. On
expiry the router closes the vLLM socket and returns `504`, allowing AI Gateway
to fall back without leaving the timed-out generation queued on the GPU
worker. Its production default is 60 seconds. Keep the AI Gateway model-node
timeout above this router-owned deadline.

The router also rejects work before vLLM admission when the serialized request
is larger than `MAX_ADMITTED_REQUEST_BYTES` (default `2097152`) or when
`MAX_INFLIGHT_REQUESTS` (default `8`) requests are already active. These are
fast `503` responses intended to activate the AI Gateway fallback without
polluting vLLM's queue or KV spill tier. They are routing controls, not context
limits: the RTX model retains its configured 256K context window, and setting the byte
limit to `0` disables only the size-based admission check.

## Worker Startup

The active startup path uses a versioned, regional S3 artifact rather than a
model EBS snapshot. The artifact contains the dereferenced model snapshot and
may contain an explicitly versioned runtime-cache archive. Worker user data
downloads it to `/opt/dlami/nvme/deepseek-model`, validates its SHA-256
checksums, then points `HF_CACHE` at that local NVMe path before starting vLLM.
Only restore a runtime cache produced by the same pinned stock image; caches
from the former custom DCP/DSpark runtime are incompatible.
The stock launcher persists the container's `/root/.cache` tree in a
versioned stock-only directory under `$HF_CACHE`, which is archived by the
publish script. It is deliberately separate from the custom runtime's old
unversioned cache directory.

The local NVMe cache is intentionally ephemeral: it is discarded on Spot
termination and rebuilt from S3. Bake the Docker image, repository, and
systemd unit into the runtime AMI; keep the model artifact immutable and
regional. This removes EBS snapshot hydration and Fast Snapshot Restore from
the recovery path.

`deepseek-spot-interruption-watcher.service` polls IMDSv2 every five seconds.
When `spot/instance-action` or `events/recommendations/rebalance` appears, it
rejects only new TCP connections to vLLM port 8000. Existing connections keep
running during the remaining notice window, and the router removes the worker
when its next health probe fails.

`worker-user-data-s3-nvme.sh` requires `MODEL_ARTIFACT_URI` to be set to an
immutable release prefix, such as:

```text
s3://deepseek-rtx4-artifacts-<account>-us-east-2/deepseek-v4-flash-dspark/<release>
```

Use `promote-s3-nvme-launch-template.sh` to create a new launch-template
version. It embeds `worker-user-data-s3-nvme.sh`, removes the obsolete cache
volume mapping, and promotes the new version:

```bash
REGION=us-east-2 \
LAUNCH_TEMPLATE_ID=lt-... \
MODEL_ARTIFACT_URI=s3://deepseek-rtx4-artifacts-<account>-us-east-2/deepseek-v4-flash-dspark/<release> \
./promote-s3-nvme-launch-template.sh
```

After the first worker for a new model/runtime combination reports ready,
publish its cache once and promote that immutable object in every region:

```bash
ARTIFACT_URI=s3://deepseek-rtx4-artifacts-<account>-us-east-2/deepseek-v4-flash-dspark/<release> \
RUNTIME_CACHE_OBJECT=runtime-cache/0731-sm120-v1.tar.zst \
./publish-s3-runtime-cache.sh
```

See `OPERATIONS.md` for the service topology, readiness contract, and recovery
checks.

## Publishing Flash-0731 weights

Stage the official commit once, then replicate the immutable artifact to each
worker region. The worker bootstrap deliberately downloads only from S3.

```bash
ARTIFACT_URI=s3://deepseek-rtx4-artifacts-<account>-us-east-2/deepseek-v4-flash-dspark/2026-07-31-flash-0731 \
./stage-s3-model-artifact.sh
```

`artifact.json` is uploaded last, so an interrupted upload is never treated as
a valid worker release. Flash-0731 includes a DSpark drafter, but the stock RTX
launcher does not enable it.

After staging the source release, copy it to each worker region:

```bash
SOURCE_ARTIFACT_URI=s3://deepseek-rtx4-artifacts-<account>-us-east-2/deepseek-v4-flash-dspark/2026-07-31-flash-0731 \
TARGET_ARTIFACT_URI=s3://deepseek-rtx4-artifacts-<account>-us-west-2/deepseek-v4-flash-dspark/2026-07-31-flash-0731 \
TARGET_REGION=us-west-2 \
./replicate-s3-model-artifact.sh
```

Runtime caches are published after the model release already exists. Replicate
them separately without rewriting the model artifact:

```bash
SOURCE_ARTIFACT_URI=s3://deepseek-rtx4-artifacts-<account>-us-east-2/deepseek-v4-flash-dspark/2026-07-31-flash-0731 \
TARGET_ARTIFACT_URI=s3://deepseek-rtx4-artifacts-<account>-us-west-2/deepseek-v4-flash-dspark/2026-07-31-flash-0731 \
TARGET_REGION=us-west-2 \
RUNTIME_CACHE_OBJECT=runtime-cache/0731-sm120-v1.tar.zst \
./replicate-s3-runtime-cache.sh
```
