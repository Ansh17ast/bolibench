#!/bin/sh
set -e

# ==============================================================================
# BoliBench — Production Entrypoint for Railway Deployment
# ==============================================================================
# 1. Resolves persistent volume mount directory ($BOLIBENCH_DATA_DIR, default /data)
# 2. If running as root (UID 0), automatically fixes volume permissions for appuser (UID 10001)
# 3. Verifies that the persistent volume is writable by appuser (UID 10001)
# 4. If writable check fails, halts startup immediately with a clear error
# 5. Drops root privileges and executes the CMD as appuser using gosu
# ==============================================================================

DATA_DIR="${BOLIBENCH_DATA_DIR:-/data}"
APP_USER="appuser"
APP_UID=10001

if [ "$(id -u)" = "0" ]; then
    # Running as root: ensure data mount exists and fix ownership if needed
    if [ -d "$DATA_DIR" ]; then
        CURRENT_OWNER=$(stat -c '%u' "$DATA_DIR" 2>/dev/null || stat -f '%u' "$DATA_DIR" 2>/dev/null || echo "unknown")
        if [ "$CURRENT_OWNER" != "$APP_UID" ]; then
            echo "[BoliBench Entrypoint] Persistent volume at $DATA_DIR is owned by UID $CURRENT_OWNER (expected $APP_UID)."
            echo "[BoliBench Entrypoint] Adjusting ownership of $DATA_DIR to $APP_USER ($APP_UID)..."
            chown -R "$APP_UID:$APP_UID" "$DATA_DIR" || {
                echo "[BoliBench Entrypoint] CRITICAL ERROR: Failed to chown $DATA_DIR to $APP_UID." >&2
                exit 1
            }
            chmod 775 "$DATA_DIR"
        fi
    fi

    # Verify write access as appuser
    if [ -d "$DATA_DIR" ]; then
        if ! su -s /bin/sh "$APP_USER" -c "touch '$DATA_DIR/.perm_check' && rm -f '$DATA_DIR/.perm_check'" 2>/dev/null; then
            echo "[BoliBench Entrypoint] CRITICAL: Persistent volume at $DATA_DIR is not writable by $APP_USER (UID $APP_UID)." >&2
            echo "[BoliBench Entrypoint] Refusing to start to prevent data loss or silent fallback to container filesystem." >&2
            exit 1
        fi
        echo "[BoliBench Entrypoint] Verified: $DATA_DIR is writable by $APP_USER (UID $APP_UID)."
    fi

    # Drop root privileges and exec command as appuser
    echo "[BoliBench Entrypoint] Dropping privileges to $APP_USER ($APP_UID) and launching application..."
    exec gosu "$APP_USER" "$@"
else
    # Already running as non-root (e.g. if RAILWAY_RUN_UID or container runtime overrides user)
    echo "[BoliBench Entrypoint] Running as UID $(id -u)..."
    if [ -d "$DATA_DIR" ]; then
        if ! touch "$DATA_DIR/.perm_check" 2>/dev/null; then
            echo "[BoliBench Entrypoint] CRITICAL: Persistent volume at $DATA_DIR is NOT writable by current user (UID $(id -u))." >&2
            echo "[BoliBench Entrypoint] Because Railway mounts persistent volumes as root:root by default, remove any RAILWAY_RUN_UID override" >&2
            echo "[BoliBench Entrypoint] so the entrypoint can initialize permissions, or ensure the volume is writable by UID $APP_UID." >&2
            exit 1
        fi
        rm -f "$DATA_DIR/.perm_check"
        echo "[BoliBench Entrypoint] Verified: $DATA_DIR is writable by current user."
    fi
    exec "$@"
fi
