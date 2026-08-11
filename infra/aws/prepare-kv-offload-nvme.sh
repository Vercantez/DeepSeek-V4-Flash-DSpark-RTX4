#!/usr/bin/env bash
set -euo pipefail

# Give vLLM's unbounded filesystem KV tier a hard capacity boundary without
# putting it on Docker's root disk. The sparse ext4 image grows only as blocks
# are written, but can never consume more than KV_OFFLOAD_MAX_DISK_GB from the
# local instance-store NVMe filesystem.

: "${KV_OFFLOAD_PARENT_MOUNT:=/opt/dlami/nvme}"
: "${KV_OFFLOAD_DISK_DIR:=/opt/dlami/nvme/kv-offload}"
: "${KV_OFFLOAD_IMAGE:=/opt/dlami/nvme/.kv-offload.ext4}"
: "${KV_OFFLOAD_MAX_DISK_GB:=4096}"
: "${KV_OFFLOAD_OWNER:=ubuntu}"

if [ "$(id -u)" -ne 0 ]; then
  echo "prepare-kv-offload-nvme.sh must run as root" >&2
  exit 2
fi
if ! [[ "$KV_OFFLOAD_MAX_DISK_GB" =~ ^[1-9][0-9]*$ ]]; then
  echo "KV_OFFLOAD_MAX_DISK_GB must be a positive integer" >&2
  exit 2
fi
if [ "$(findmnt -n -o TARGET -T "$KV_OFFLOAD_PARENT_MOUNT")" != "$KV_OFFLOAD_PARENT_MOUNT" ]; then
  echo "$KV_OFFLOAD_PARENT_MOUNT is not a dedicated mount" >&2
  exit 2
fi
case "$KV_OFFLOAD_IMAGE" in
  "$KV_OFFLOAD_PARENT_MOUNT"/*) ;;
  *) echo "KV offload image must be below $KV_OFFLOAD_PARENT_MOUNT" >&2; exit 2 ;;
esac
case "$KV_OFFLOAD_DISK_DIR" in
  "$KV_OFFLOAD_PARENT_MOUNT"/*) ;;
  *) echo "KV offload directory must be below $KV_OFFLOAD_PARENT_MOUNT" >&2; exit 2 ;;
esac

install -d "$KV_OFFLOAD_DISK_DIR"
if ! mountpoint -q "$KV_OFFLOAD_DISK_DIR"; then
  if [ ! -e "$KV_OFFLOAD_IMAGE" ]; then
    truncate -s "${KV_OFFLOAD_MAX_DISK_GB}G" "$KV_OFFLOAD_IMAGE"
    # KV blocks are multi-megabyte files. largefile4 avoids reserving tens of
    # gigabytes of inode tables for tiny files that this cache never creates.
    mkfs.ext4 -q -F -m 0 -T largefile4 \
      -E lazy_itable_init=1,lazy_journal_init=1 \
      -L dsv4-kv-offload "$KV_OFFLOAD_IMAGE"
  fi
  if [ "$(blkid -p -s TYPE -o value "$KV_OFFLOAD_IMAGE")" != ext4 ]; then
    echo "$KV_OFFLOAD_IMAGE is not an ext4 filesystem image" >&2
    exit 2
  fi
  mount -o loop,noatime,nodiratime "$KV_OFFLOAD_IMAGE" "$KV_OFFLOAD_DISK_DIR"
fi

offload_source="$(findmnt -n -o SOURCE -T "$KV_OFFLOAD_DISK_DIR")"
case "$offload_source" in
  /dev/loop*) ;;
  *) echo "$KV_OFFLOAD_DISK_DIR is not mounted from a loop device" >&2; exit 2 ;;
esac
backing_file="$(losetup -n -O BACK-FILE "$offload_source")"
if [ "$backing_file" != "$KV_OFFLOAD_IMAGE" ]; then
  echo "$offload_source is backed by $backing_file, expected $KV_OFFLOAD_IMAGE" >&2
  exit 2
fi

fstab_entry="$KV_OFFLOAD_IMAGE $KV_OFFLOAD_DISK_DIR ext4 loop,noatime,nodiratime,nofail,x-systemd.requires-mounts-for=$KV_OFFLOAD_PARENT_MOUNT 0 0"
if ! grep -Fq "$KV_OFFLOAD_IMAGE $KV_OFFLOAD_DISK_DIR " /etc/fstab; then
  printf '%s\n' "$fstab_entry" >>/etc/fstab
fi

chown "$KV_OFFLOAD_OWNER:$KV_OFFLOAD_OWNER" "$KV_OFFLOAD_DISK_DIR"
echo "KV offload filesystem ready: $offload_source -> $KV_OFFLOAD_DISK_DIR (${KV_OFFLOAD_MAX_DISK_GB} GiB cap, backing $KV_OFFLOAD_IMAGE)"
