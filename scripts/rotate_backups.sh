#!/usr/bin/env bash
set -euo pipefail

BACKUP_ROOT="/opt/goldtrigger_backups"
SOURCE_DIR="/opt/GoldTrigger_bot"
KEEP_DAYS="${KEEP_DAYS:-7}"

mkdir -p "$BACKUP_ROOT"
timestamp="$(date +%Y%m%d_%H%M%S)"
archive="$BACKUP_ROOT/GoldTrigger_bot_backup_${timestamp}.tar.gz"

tar -czf "$archive" -C "$(dirname "$SOURCE_DIR")" "$(basename "$SOURCE_DIR")"

find "$BACKUP_ROOT" -type f -name "GoldTrigger_bot_backup_*.tar.gz" -mtime +"$KEEP_DAYS" -delete || true
find "$SOURCE_DIR/logs" -type f -name "*.log" -mtime +"$KEEP_DAYS" -delete || true
