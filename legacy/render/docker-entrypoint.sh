#!/bin/sh
# Container entrypoint.
#
# Seeds the documents directory before handing off to the CMD (gunicorn). The
# bundled subject PDFs are baked at /opt/seed-documents; we copy them into
# /app/documents WITHOUT clobbering, so:
#   • a fresh, empty volume mounted at /app/documents gets the bundled subjects, and
#   • any subjects a user uploaded into that volume are preserved across restarts.
# With no volume mounted, /app/documents is the image's own (empty) dir and is
# seeded the same way. Safe and idempotent on every start.
set -e

SEED_DIR="/opt/seed-documents"
DATA_DIR="/app/documents"

if [ -d "$SEED_DIR" ]; then
  mkdir -p "$DATA_DIR"
  # -r recursive, -n no-clobber: never overwrite an existing (possibly newer) file.
  cp -rpn "$SEED_DIR"/. "$DATA_DIR"/ 2>/dev/null || true
fi

# Replace the shell with the CMD so signals (SIGTERM from the container runtime)
# reach gunicorn directly for a graceful shutdown.
exec "$@"
