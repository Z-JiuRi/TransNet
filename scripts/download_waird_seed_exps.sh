#!/usr/bin/env bash
set -euo pipefail

# Download missing WAIRD seed experiment files from the H100 server.
#
# Defaults are intentionally non-destructive:
# - preserve the seed796/seed797 directory structure
# - create missing directories
# - download files that do not exist locally
# - never delete or overwrite local files
#
# Authentication:
# - uses existing SSH config/key by default
# - if password auth is needed, install sshpass and run:
#     SSHPASS='your-password' scripts/download_waird_seed_exps.sh

REMOTE_HOST="${REMOTE_HOST:-H100}"
REMOTE_USER="${REMOTE_USER:-hujiacong}"
REMOTE_ROOT="${REMOTE_ROOT:-/storage/hujiacong/zxd/Huawei/TransNet/exps/WAIRD}"
LOCAL_ROOT="${LOCAL_ROOT:-exps/WAIRD}"
SEEDS="${SEEDS:-seed796 seed797}"
SSH_OPTS="${SSH_OPTS:--o ServerAliveInterval=60 -o ServerAliveCountMax=10}"
DRY_RUN=0

usage() {
    cat <<'EOF'
Usage:
  scripts/download_waird_seed_exps.sh [--dry-run]

Environment overrides:
  REMOTE_HOST   SSH host or alias. Default: H100
  REMOTE_USER   SSH user. Default: hujiacong
  REMOTE_ROOT   Remote WAIRD exp root.
                Default: /storage/hujiacong/zxd/Huawei/TransNet/exps/WAIRD
  LOCAL_ROOT    Local WAIRD exp root. Default: exps/WAIRD
  SEEDS         Space-separated seed dirs. Default: "seed796 seed797"
  SSH_OPTS      Extra ssh options.
  SSHPASS       Optional password for sshpass -e, if key auth is unavailable.

Examples:
  scripts/download_waird_seed_exps.sh --dry-run
  SSHPASS='***' scripts/download_waird_seed_exps.sh
  REMOTE_HOST=10.1.3.1 REMOTE_USER=hujiacong scripts/download_waird_seed_exps.sh
EOF
}

while [ "$#" -gt 0 ]; do
    case "$1" in
        --dry-run|-n)
            DRY_RUN=1
            shift
            ;;
        --help|-h)
            usage
            exit 0
            ;;
        *)
            echo "Unknown argument: $1" >&2
            usage >&2
            exit 2
            ;;
    esac
done

if ! command -v rsync >/dev/null 2>&1; then
    echo "rsync is required but was not found." >&2
    exit 1
fi

RSYNC_PREFIX=()
if [ -n "${SSHPASS:-}" ]; then
    if ! command -v sshpass >/dev/null 2>&1; then
        echo "SSHPASS is set, but sshpass is not installed." >&2
        exit 1
    fi
    RSYNC_PREFIX=(sshpass -e)
fi

RSYNC_OPTS=(
    -a
    --ignore-existing
    --partial
    --human-readable
    --itemize-changes
    --info=progress2,stats2
)

if [ "$DRY_RUN" -eq 1 ]; then
    RSYNC_OPTS+=(--dry-run)
fi

mkdir -p "$LOCAL_ROOT"

REMOTE_SPEC="${REMOTE_USER}@${REMOTE_HOST}"
SSH_CMD="ssh ${SSH_OPTS}"

echo "Remote: ${REMOTE_SPEC}:${REMOTE_ROOT}/{${SEEDS}}/"
echo "Local : ${LOCAL_ROOT}/"
if [ "$DRY_RUN" -eq 1 ]; then
    echo "Mode  : dry run; no files will be downloaded"
else
    echo "Mode  : download missing files only; existing local files are kept"
fi
echo

for seed in $SEEDS; do
    remote_dir="${REMOTE_ROOT}/${seed}/"
    local_dir="${LOCAL_ROOT}/${seed}/"

    echo "==> Syncing ${seed}"
    mkdir -p "$local_dir"

    "${RSYNC_PREFIX[@]}" rsync "${RSYNC_OPTS[@]}" \
        -e "$SSH_CMD" \
        "${REMOTE_SPEC}:${remote_dir}" \
        "$local_dir"
    echo
done

echo "Done."
