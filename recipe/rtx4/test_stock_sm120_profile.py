#!/usr/bin/env python3
"""Focused dry-run checks for the isolated stock SM120 launcher."""

from __future__ import annotations

import os
from pathlib import Path
import subprocess
import tempfile
import unittest


REPO_ROOT = Path(__file__).resolve().parents[2]
LAUNCHER = REPO_ROOT / "start-deepseek-v4-flash-stock-sm120-rtx4.sh"
DOCKERFILE = REPO_ROOT / "recipe/rtx4/Dockerfile.stock-sm120"


class StockSm120ProfileTest(unittest.TestCase):
    def run_launcher(
        self, **overrides: str
    ) -> tuple[subprocess.CompletedProcess[str], list[str]]:
        with tempfile.TemporaryDirectory() as temp_dir:
            temp = Path(temp_dir)
            fake_bin = temp / "bin"
            fake_bin.mkdir()
            capture = temp / "docker-args"
            fake_docker = fake_bin / "docker"
            fake_docker.write_text(
                """#!/usr/bin/env bash
set -euo pipefail
if [ "${1:-}" = container ] && [ "${2:-}" = inspect ]; then
  exit 1
fi
if [ "${1:-}" = run ]; then
  printf '%s\\0' "$@" > "$DOCKER_CAPTURE"
  echo fake-container-id
  exit 0
fi
if [ "${1:-}" = pull ]; then
  exit 0
fi
echo "unexpected docker command: $*" >&2
exit 9
"""
            )
            fake_docker.chmod(0o755)

            fake_findmnt = fake_bin / "findmnt"
            fake_findmnt.write_text(
                """#!/usr/bin/env bash
set -euo pipefail
printf '%s\n' "$FAKE_FINDMNT_TARGET"
"""
            )
            fake_findmnt.chmod(0o755)

            fake_df = fake_bin / "df"
            fake_df.write_text(
                """#!/usr/bin/env bash
set -euo pipefail
printf '%s\n' 'Filesystem 1024-blocks Used Available Capacity Mounted on'
printf '%s\n' '/dev/fake 4294967296 0 4294967296 0% /fake-nvme'
"""
            )
            fake_df.chmod(0o755)

            nvme_mount = temp / "nvme"

            env = {
                "PATH": f"{fake_bin}:{os.environ['PATH']}",
                "HOME": str(temp),
                "ENV_FILE": "/dev/null",
                "HF_CACHE": str(temp / "hf"),
                "DOCKER_CAPTURE": str(capture),
                "KV_OFFLOAD_DISK_DIR": str(nvme_mount / "kv-offload"),
                "KV_OFFLOAD_REQUIRED_MOUNT": str(nvme_mount),
                "KV_OFFLOAD_MIN_FREE_GB": "1",
                "FAKE_FINDMNT_TARGET": str(nvme_mount),
                **overrides,
            }
            result = subprocess.run(
                [str(LAUNCHER)],
                cwd=REPO_ROOT,
                env=env,
                text=True,
                capture_output=True,
                check=False,
            )
            args = []
            if capture.exists():
                args = capture.read_bytes().rstrip(b"\0").decode().split("\0")
            return result, args

    def test_default_profile_matches_hermia_stock_recipe(self) -> None:
        result, args = self.run_launcher()
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(args[0], "run")

        def value_for(flag: str) -> str:
            index = args.index(flag)
            return args[index + 1]

        self.assertIn("vllm-dspark-runtime:stock-sm120-vllm-0.25.1", args)
        self.assertEqual(value_for("--tensor-parallel-size"), "4")
        self.assertIn("--enable-expert-parallel", args)
        self.assertEqual(value_for("--kv-cache-dtype"), "fp8")
        self.assertEqual(value_for("--block-size"), "256")
        self.assertEqual(value_for("--max-model-len"), "524288")
        self.assertEqual(value_for("--gpu-memory-utilization"), "0.90")
        self.assertEqual(value_for("--kernel-config"), '{"moe_backend":"marlin"}')
        self.assertEqual(value_for("--kv-offloading-size"), "256")
        self.assertEqual(value_for("--kv-offloading-backend"), "native")
        kv_transfer_config = value_for("--kv-transfer-config")
        self.assertIn('"spec_name":"TieringOffloadingSpec"', kv_transfer_config)
        self.assertIn('"type":"fs"', kv_transfer_config)

        docker_volumes = {
            args[index + 1]
            for index, arg in enumerate(args[:-1])
            if arg == "-v"
        }
        self.assertTrue(
            any(
                volume.endswith(
                    "/hf/vllm-cache-stock-sm120-v0.25.1-fi0.6.14:/root/.cache"
                )
                for volume in docker_volumes
            ),
            docker_volumes,
        )
        self.assertTrue(
            any(
                (source := volume.split(":", 1)[0]).endswith("/kv-offload")
                and volume == f"{source}:{source}"
                for volume in docker_volumes
            ),
            docker_volumes,
        )

        docker_env = {
            args[index + 1]
            for index, arg in enumerate(args[:-1])
            if arg == "-e"
        }
        self.assertIn("FLASHINFER_DISABLE_VERSION_CHECK=1", docker_env)
        self.assertIn("NCCL_P2P_DISABLE=1", docker_env)
        self.assertIn("PYTHONHASHSEED=0", docker_env)
        self.assertIn("CUDA_HOME=/usr/local/cuda", docker_env)

        for custom_flag in (
            "--attention-backend",
            "--decode-context-parallel-size",
            "--disable-custom-all-reduce",
            "--linear-backend",
            "--moe-backend",
            "--speculative-config",
        ):
            self.assertNotIn(custom_flag, args)

    def test_rejects_custom_runtime_features(self) -> None:
        for override, expected in (
            ({"DCP_SIZE": "2"}, "does not support DCP"),
            ({"DSPARK_NUM_TOKENS": "5"}, "does not support DSpark"),
            ({"KV_CACHE_DTYPE": "fp8_ds_mla"}, "requires KV_CACHE_DTYPE=fp8"),
            ({"MAX_MODEL_LEN": "524289"}, "at or below 524288"),
            ({"KV_OFFLOAD_GB": "0"}, "positive integer"),
            (
                {"FAKE_FINDMNT_TARGET": "/"},
                "Refusing KV offload",
            ),
        ):
            with self.subTest(override=override):
                result, args = self.run_launcher(**override)
                self.assertEqual(result.returncode, 2)
                self.assertIn(expected, result.stderr)
                self.assertEqual(args, [])

    def test_runtime_is_stock_and_pinned(self) -> None:
        dockerfile = DOCKERFILE.read_text()
        self.assertIn("vllm/vllm-openai:v0.25.1", dockerfile)
        self.assertIn('flashinfer-python==0.6.14', dockerfile)
        self.assertIn('--no-deps', dockerfile)
        self.assertIn('version("flashinfer-cubin") == "0.6.13"', dockerfile)
        self.assertNotIn("COPY ", dockerfile)


if __name__ == "__main__":
    unittest.main()
