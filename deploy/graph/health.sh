#!/usr/bin/env bash
# Health check of the local production graph: the settings of install.sh,
# then scripts/kg/graph_health.py with any extra arguments (--wait, --empty-ok).
#
# --from-timer: the periodic run; while a backup or a load holds the lock the
# database may be down on purpose, so the run reports maintenance and leaves
# the last result in place.
set -euo pipefail
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
SETTINGS="${GRAPH_SETTINGS:-$HOME/.config/graphrag/graph-db.env}"
set -a
# shellcheck disable=SC1090
source "$SETTINGS"
set +a
if [[ "${1:-}" == "--from-timer" ]]; then
  shift
  exec 9>"$GRAPH_BACKUPS/.lock"
  flock -n 9 || { echo "manutenzione in corso (backup o caricamento): controllo rimandato"; exit 0; }
fi
PYTHONNOUSERSITE=1 exec "$GRAPH_PYTHON" "$REPO/scripts/kg/graph_health.py" \
  --backups "$GRAPH_BACKUPS" --data-dir "$GRAPH_HOME/data" "$@"
