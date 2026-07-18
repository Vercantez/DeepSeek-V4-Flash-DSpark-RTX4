# DeepSeek RTX fleet dashboard

Status and capacity-control service for the three GPU Auto Scaling Groups and
the Ohio sticky router. It uses the AWS identity already configured on
`camel-devbox`; it stores no AWS or Cloudflare credentials.

The service binds only to the devbox Tailscale address:

`http://100.99.146.108:8790/`

It also exposes machine-readable state at `/api/status`.

## Capacity controls

Each fleet card can update its `min`, `desired`, and `max` capacity. The form
only permits the known Oregon, Ohio, and Virginia ASGs and validates
`0 <= min <= desired <= max <= 2`. A browser confirmation and a per-process
CSRF token protect updates; the dashboard remains reachable only over
Tailscale.

## Install

From this directory, after the `camel-devbox` SSH host entry is available:

```bash
./install-on-camel-devbox.sh
```

The installer uses the existing SSH config (including its jumpbox); it does not
depend on Tailscale Files or direct peer connectivity.

The devbox identity needs read-only permission for:

- `autoscaling:DescribeAutoScalingGroups`
- `autoscaling:UpdateAutoScalingGroup`
- `ssm:SendCommand`
- `ssm:GetCommandInvocation`

The router command executes only `curl -fsS --max-time 5
http://127.0.0.1:8080/router/backends` on the known Ohio router instance.
